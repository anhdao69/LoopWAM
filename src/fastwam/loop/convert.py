"""Structured width slicing and cycle folding for LoopWAM and its controls.

Usage: python -m fastwam.loop.convert --teacher PATH --output PATH --arch loopwam
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.wan_video_dit import CrossAttention, DiTBlock, WanVideoDiT
from fastwam.loop.slots import SlotDiTBlock

logger = logging.getLogger(__name__)
UNTIED12 = [0, 1, 3, 5, 7, 10, 13, 16, 19, 22, 25, 29]
V30A12 = [0, 1, 2, 21, 22, 23, 24, 25, 26, 27, 28, 29]
ARCHITECTURES = ("loopwam", "untied30", "untied12", "untied_v30a12")


def _canonical_arch(arch):
    return "untied_v30a12" if arch == "v30a12" else arch


def expert_configs(arch="loopwam", tiny_config=None):
    """Canonical constructor metadata, with explicit cross attention head count."""
    arch = _canonical_arch(arch)
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {ARCHITECTURES}")
    video = dict(hidden_dim=2048, in_dim=48, ffn_dim=8192, out_dim=48,
                 text_dim=4096, freq_dim=256, eps=1e-6, patch_size=(1, 2, 2),
                 num_heads=16, attn_head_dim=128, has_image_input=False,
                 seperated_timestep=True, video_attention_mask_mode="first_frame_causal")
    action = dict(hidden_dim=768, action_dim=7, ffn_dim=3072, text_dim=4096,
                  freq_dim=256, eps=1e-6, num_heads=16, cross_num_heads=6,
                  attn_head_dim=128)
    if tiny_config:
        video.update(tiny_config.get("video", tiny_config.get("video_config", {})))
        action.update(tiny_config.get("action", tiny_config.get("action_config", {})))
    video["num_layers"] = 30 if arch in ("untied30", "untied_v30a12") else 12
    action["num_layers"] = 30 if arch == "untied30" else 12
    if video["num_heads"] != action["num_heads"] or video["attn_head_dim"] != action["attn_head_dim"]:
        raise ValueError("Video and action self attention must share head dimensions")
    if video["hidden_dim"] != video["num_heads"] * video["attn_head_dim"]:
        raise ValueError("Structured video slicing requires hidden_dim = heads * head_dim")
    return video, action


def _new_block(config, device="cpu"):
    with torch.device(device):
        block = DiTBlock(config["hidden_dim"], config["attn_head_dim"], config["num_heads"],
                         config["ffn_dim"], config["eps"])
        if config.get("cross_num_heads", config["num_heads"]) != config["num_heads"]:
            block.cross_attn = CrossAttention(config["hidden_dim"], config["attn_head_dim"],
                                              config["cross_num_heads"], config["eps"])
    return block


def build_experts(arch="loopwam", lora_rank=32, device="cpu", dtype=torch.float32, tiny_config=None):
    """Instantiate experts, retaining standard FastWAM preparation and heads.

    ``tiny_config`` can also hold the complete ``video``/``action`` dictionaries
    from checkpoint metadata, making reconstruction independent of defaults.
    """
    arch = _canonical_arch(arch)
    video_config, action_config = expert_configs(arch, tiny_config)
    action_ctor = dict(action_config)
    cross_heads = action_ctor.pop("cross_num_heads")
    with torch.device(device):
        video = WanVideoDiT(**video_config)
        action = ActionDiT(**action_ctor)
        for block in action.blocks:
            if cross_heads != block.cross_attn.num_heads:
                block.cross_attn = CrossAttention(action.hidden_dim, action.attn_head_dim,
                                                  cross_heads, action_config["eps"])
        if arch == "loopwam":
            for expert in (video, action):
                for index in range(3, 9):
                    expert.blocks[index] = SlotDiTBlock(expert.blocks[index], rank=lora_rank)
    return video.to(dtype=dtype), action.to(dtype=dtype)


def spaced_indices(source: int, target: int) -> torch.Tensor:
    if not 0 < target <= source:
        raise ValueError(f"Cannot select {target} entries from {source}")
    return torch.linspace(0, source - 1, target, dtype=torch.float64).round().long()


def head_channels(source_heads: int, target_heads: int, head_dim: int) -> torch.Tensor:
    heads = spaced_indices(source_heads, target_heads)
    return (heads[:, None] * head_dim + torch.arange(head_dim)[None, :]).flatten()


def _interpolate_axis(value: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    if value.shape[axis] == size:
        return value
    moved = value.movedim(axis, -1)
    output = F.interpolate(moved.reshape(-1, 1, moved.shape[-1]), size=size,
                           mode="linear", align_corners=True)
    return output.reshape(*moved.shape[:-1], size).movedim(-1, axis).contiguous()


def slice_expert_state(source, kind, config):
    """Slice all source layers; depth selection is a separate operation.

    Attention axes select whole heads first. Only action hidden axes interpolate.
    Last hidden axes use FastWAM's sqrt(source_width / target_width) scaling;
    selected attention/FFN axes are never interpolated or rescaled.
    """
    if kind not in ("video", "action"):
        raise ValueError(f"Invalid expert kind: {kind}")
    source_hidden = source["blocks.0.modulation"].shape[-1]
    target_hidden = config["hidden_dim"]
    head_dim = config["attn_head_dim"]
    source_heads = source["blocks.0.self_attn.q.bias"].numel() // head_dim
    channels = head_channels(source_heads, config["num_heads"], head_dim)
    cross_source_heads = source["blocks.0.cross_attn.q.bias"].numel() // head_dim
    cross = head_channels(cross_source_heads, config.get("cross_num_heads", config["num_heads"]), head_dim)
    ffn = spaced_indices(source["blocks.0.ffn.0.bias"].numel(), config["ffn_dim"])
    if kind == "video" and source_hidden != source_heads * head_dim:
        raise ValueError("Video teacher hidden width must match its attention width")

    def select(value, axis, indices):
        return value.index_select(axis, indices.to(value.device))

    def hidden(value, axis, scale=False):
        if kind == "video":
            return select(value, axis, channels)
        output = _interpolate_axis(value, axis, target_hidden)
        if scale and source_hidden != target_hidden:
            output = output * (source_hidden / target_hidden) ** 0.5
        return output

    result = {}
    for key, original in source.items():
        value = original.detach().float()
        name = key.split(".", 2)[2] if key.startswith("blocks.") else key
        if name == "modulation" or key == "head.modulation":
            value = hidden(value, -1, scale=kind == "action")
        elif key.startswith("blocks.") and name.startswith(("self_attn.", "cross_attn.")):
            attention, part, parameter = name.split(".")
            indices = channels if attention == "self_attn" else cross
            if part in ("norm_q", "norm_k") or (part in ("q", "k", "v") and parameter == "bias"):
                value = select(value, 0, indices)
            elif part in ("q", "k", "v"):
                value = hidden(select(value, 0, indices), 1, scale=True)
            elif part == "o" and parameter == "weight":
                value = hidden(select(value, 1, indices), 0)
            elif part == "o" and parameter == "bias":
                value = hidden(value, 0)
            else:
                raise ValueError(f"Unrecognized attention tensor {key}")
        elif key.startswith("blocks.") and name.startswith("norm3."):
            value = hidden(value, 0)
        elif key.startswith("blocks.") and name.startswith("ffn.0."):
            value = select(value, 0, ffn)
            if name.endswith("weight"):
                value = hidden(value, 1, scale=True)
        elif key.startswith("blocks.") and name.startswith("ffn.2."):
            if name.endswith("weight"):
                value = select(value, 1, ffn)
            value = hidden(value, 0)
        elif key.startswith("patch_embedding.") or key.startswith("action_encoder."):
            value = hidden(value, 0)
        elif key.startswith(("text_embedding.", "time_embedding.")):
            value = hidden(value, 0)
            if ".2.weight" in key:
                value = hidden(value, 1, scale=True)
        elif key == "time_projection.1.weight":
            value = value.reshape(6, source_hidden, source_hidden)
            value = hidden(hidden(value, 1), 2, scale=True).reshape(6 * target_hidden, target_hidden)
        elif key == "time_projection.1.bias":
            # Do not interpolate across the six independent modulation groups.
            value = hidden(value.reshape(6, source_hidden), 1).reshape(6 * target_hidden)
        elif key in ("head.head.weight", "head.weight"):
            value = hidden(value, 1, scale=True)
        elif key in ("head.head.bias", "head.bias"):
            pass
        else:
            raise ValueError(f"No structured conversion rule for tensor {key!r}")
        result[key] = value.contiguous()
    return result


@torch.no_grad()
def fold_blocks(blocks, rank=32, device="cpu"):
    """Return P1..P3, six shared core blocks, Q1..Q3 from thirty source blocks."""
    if len(blocks) != 30:
        raise ValueError(f"Depth folding requires 30 source layers, got {len(blocks)}")
    result = [copy.deepcopy(block).to(device) for block in blocks[:3]]
    for index in range(6):
        group = [copy.deepcopy(blocks[3 + index + 6 * slot]).to(device) for slot in range(4)]
        result.append(SlotDiTBlock.from_blocks(group, rank, seed=index * 1000))
    result.extend(copy.deepcopy(block).to(device) for block in blocks[27:])
    return nn.ModuleList(result)


def _teacher_parts(payload):
    if "mot" not in payload or "proprio_encoder" not in payload:
        raise ValueError("Expected released FastWAM checkpoint with mot and proprio_encoder")
    source = payload["mot"]
    parts = {}
    allowed = ("mixtures.video.", "mixtures.action.")
    unknown = [key for key in source if not key.startswith(allowed)]
    if unknown:
        raise ValueError(f"Unexpected teacher MoT keys: {unknown[:5]}")
    for name in ("video", "action"):
        prefix = f"mixtures.{name}."
        parts[name] = {key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)}
        layer_ids = {int(key.split(".")[1]) for key in parts[name] if key.startswith("blocks.")}
        if layer_ids != set(range(30)):
            raise ValueError(f"Expected exactly thirty {name} teacher layers")
    return parts


@torch.no_grad()
def convert_checkpoint(teacher_path, output_path, arch="loopwam", lora_rank=32,
                       device="cuda:0", tiny_config=None, output_dtype=torch.bfloat16):
    """Convert a release checkpoint; use the GPU only for one folding group at a time.

    The teacher stays memory mapped on CPU. The output is atomic and contains
    canonical state dictionaries with constructor/head-selection metadata.
    """
    arch = _canonical_arch(arch)
    video_config, action_config = expert_configs(arch, tiny_config)
    payload = torch.load(teacher_path, map_location="cpu", mmap=True, weights_only=False)
    source = _teacher_parts(payload)
    # Meta construction validates every resulting key/shape without initializing
    # billions of disposable random parameters or allocating the full model on GPU.
    templates = build_experts(arch, lora_rank, device="meta", tiny_config=tiny_config)
    output = {"format": "loopwam_v1"}
    energies = []
    for kind, config, template in zip(("video", "action"), (video_config, action_config), templates):
        logger.info("Slicing %s expert", kind)
        sliced = slice_expert_state(source[kind], kind, config)
        final = {key: value for key, value in sliced.items() if not key.startswith("blocks.")}
        if arch == "loopwam":
            direct = list(zip(range(3), range(3))) + list(zip(range(9, 12), range(27, 30)))
            for index in range(6):
                logger.info("Folding %s core %d/6", kind, index + 1)
                group = []
                for slot in range(4):
                    prefix = f"blocks.{3 + index + 6 * slot}."
                    block_state = {key[len(prefix):]: value for key, value in sliced.items() if key.startswith(prefix)}
                    block = _new_block(config, device="meta")
                    block.load_state_dict(block_state, strict=True, assign=True)
                    group.append(block.to(device=device, dtype=torch.float32))
                core = SlotDiTBlock.from_blocks(group, lora_rank, seed=index * 1000).cpu()
                for row in core.energy_report:
                    row = dict(row, expert=kind, core=index)
                    energies.append(row)
                    logger.info("SVD %s core=%d %s slot=%d captured=%.6f", kind, index,
                                row["linear"], row["slot"], row["captured_energy"])
                final.update({f"blocks.{3 + index}.{key}": value for key, value in core.state_dict().items()})
                del core, group, block, block_state
        else:
            mapping = (UNTIED12 if arch == "untied12" else V30A12 if arch == "untied_v30a12" and kind == "action" else list(range(30)))
            direct = list(enumerate(mapping))
        for target_layer, source_layer in direct:
            prefix = f"blocks.{source_layer}."
            final.update({f"blocks.{target_layer}.{key[len(prefix):]}": value
                          for key, value in sliced.items() if key.startswith(prefix)})
        template.load_state_dict(final, strict=True, assign=True)
        output[kind] = {key: value.detach().to(device="cpu", dtype=output_dtype).clone(memory_format=torch.contiguous_format)
                        for key, value in template.state_dict().items()}
        del sliced, final
    proprio = payload["proprio_encoder"]
    if set(proprio) != {"weight", "bias"} or proprio["weight"].shape[0] != video_config["text_dim"]:
        raise ValueError("Proprio encoder must project into the unchanged text context width")
    # Release checkpoints can hold tiny tensors as views into a multi-GB flat
    # optimizer/model storage. contiguous() alone does not break such aliases.
    output["proprio"] = {key: value.detach().to(device="cpu", dtype=output_dtype).clone(memory_format=torch.contiguous_format)
                          for key, value in proprio.items()}
    head_dim = video_config["attn_head_dim"]
    teacher_heads = source["video"]["blocks.0.self_attn.q.bias"].numel() // head_dim
    action_cross_heads = source["action"]["blocks.0.cross_attn.q.bias"].numel() // head_dim
    output["meta"] = dict(
        arch=arch, lora_rank=lora_rank, video_config=video_config, action_config=action_config,
        head_indices={"self": spaced_indices(teacher_heads, video_config["num_heads"]).tolist(),
                      "video_cross": spaced_indices(teacher_heads, video_config["num_heads"]).tolist(),
                      "action_cross": spaced_indices(action_cross_heads, action_config["cross_num_heads"]).tolist()},
        layer_indices={"video": UNTIED12 if arch == "untied12" else list(range(30)),
                       "action": UNTIED12 if arch == "untied12" else V30A12 if arch == "untied_v30a12" else list(range(30))},
        proprio_dim=proprio["weight"].shape[1], teacher_path=str(Path(teacher_path).resolve()),
        teacher_step=payload.get("step"), dtype=str(output_dtype),
        video_shift=5.0, action_shift=5.0, sigma_shift=5.0, svd_energy=energies,
        svd_method="exact_full_rank_else_randomized_oversample8_niter2",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(destination)
    logger.info("Saved %s (%s); low-energy slots=%d/%d", destination, arch,
                sum(row["captured_energy"] < 0.3 for row in energies), len(energies))
    return output["meta"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", choices=ARCHITECTURES, default="loopwam")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    convert_checkpoint(args.teacher, args.output, args.arch, args.lora_rank,
                       args.device, output_dtype=getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
