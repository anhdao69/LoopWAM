#!/usr/bin/env python3
"""Measure D1 teacher layer similarity and D2 conversion fidelity before training.

Example (requires a free GPU; this script never launches training)::

    python scripts/loopwam/diagnose_conversion.py \
      --teacher checkpoints/fastwam_release/libero_uncond_2cam224.pt \
      --checkpoints checkpoints/loopwam_v1 --manifest PATH/split_manifest.json \
      --output outputs/loopwam_v1/initialization --device cuda:0

D1 uses 1,000 deterministically selected, unpadded training clips by default.
D2 always uses one unpadded midpoint clip from each of the 20 held-out demos.
The rank-zero diagnostic disables all adapters in the converted rank-32 model
for inference, retaining shared weights, slot norms, biases and modulation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

LOG = logging.getLogger("loopwam.conversion_diagnostics")
TAU_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def select_training_indices(manifest, count, seed):
    """Select without replacement from unpadded starts, retaining original IDs."""
    eligible = [window for episode in manifest["episodes"] if episode["split"] == "train"
                for window in range(episode["window_start"],
                                    episode["window_start"] + episode["unpadded_windows"])]
    if count < 2 or count > len(eligible):
        raise ValueError(f"D1 needs 2..{len(eligible)} unpadded clips, requested {count}")
    windows = sorted(int(x) for x in np.random.default_rng(seed).choice(eligible, count, replace=False))
    positions = {window: index for index, window in enumerate(manifest["train_window_ids"])}
    return [positions[window] for window in windows]


def layer_similarity(features):
    """Compute clip-mean angles and centered linear CKA for [layer, clip, width].

    Raw token-mean features are used. Angular distance averages the angle for
    each clip, in radians. CKA centers examples in feature space, then compares
    their linear Gram matrices using normalized Frobenius inner products.
    Degenerate constant/zero features fail explicitly rather than producing NaN.
    """
    if features.ndim != 3 or features.shape[1] < 2:
        raise ValueError("Features must have shape [layers, at least two clips, width]")
    x = features.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(x).all():
        raise ValueError("Layer features contain nonfinite values")
    lengths = x.norm(dim=-1, keepdim=True)
    if (lengths == 0).any():
        raise ValueError("Angular similarity is undefined for zero feature vectors")
    normalized = x / lengths
    cosine = torch.einsum("lnd,mnd->nlm", normalized, normalized).clamp(-1, 1)
    angles = cosine.acos().mean(dim=0)
    angles.fill_diagonal_(0)
    layers, clips, _ = x.shape
    # Only 30 x N x N CPU storage; no D x D covariance matrices or GPU history.
    grams = torch.empty(layers, clips * clips, dtype=torch.float64)
    for layer in range(layers):
        centered = x[layer] - x[layer].mean(dim=0, keepdim=True)
        gram = centered @ centered.mT
        norm = gram.norm()
        if norm == 0:
            raise ValueError(f"Centered CKA is undefined for constant layer {layer}")
        grams[layer] = gram.flatten() / norm
    cka = (grams @ grams.mT).clamp(0, 1)
    cka.fill_diagonal_(1)
    return {"angular_distance_radians": angles.tolist(), "linear_cka": cka.tolist()}


def masked_error(prediction, target, is_pad):
    """Return sufficient statistics for an element-weighted action velocity MSE."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Action velocities must share shape [batch, horizon, dimensions]")
    if is_pad.shape != prediction.shape[:2]:
        raise ValueError("Action padding mask must match [batch, horizon]")
    valid = (~is_pad.to(device=prediction.device, dtype=torch.bool)).unsqueeze(-1).expand_as(prediction)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("Action velocity comparison has no valid elements")
    errors = (prediction.float() - target.float()).square().masked_select(valid)
    if not torch.isfinite(errors).all():
        raise ValueError("Action velocity comparison contains nonfinite values")
    squared_error = float(errors.double().sum())
    return {"squared_error": squared_error, "valid_elements": count, "mse": squared_error / count}


@contextmanager
def without_lora(model):
    """Temporarily skip all SlotLinear low-rank terms; never change parameters."""
    from fastwam.loop.slots import SlotLinear
    modules = [(module, module.rank) for module in model.modules() if isinstance(module, SlotLinear)]
    if not modules:
        raise ValueError("Rank-zero diagnostic requires a model containing slot adapters")
    try:
        for module, _ in modules:
            module.rank = 0
        yield
    finally:
        for module, rank in modules:
            module.rank = rank


@contextmanager
def capture_layers(mot):
    """Capture actual MoT updates; DiTBlock.forward is bypassed by joint MoT."""
    if mot.training:
        raise ValueError("Teacher layer capture requires evaluation mode")
    original = mot._forward_joint_layer
    had_override = "_forward_joint_layer" in mot.__dict__
    features = {"video": [], "action": []}

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        for kind, tokens in zip(("video", "action"), result):
            if tokens.ndim != 3 or tokens.shape[0] != 1:
                raise ValueError("D1 captures one clip per forward")
            features[kind].append(tokens.detach().float().mean(dim=1)[0].cpu())
        return result

    mot._forward_joint_layer = wrapped
    try:
        yield features
    finally:
        if had_override:
            mot._forward_joint_layer = original
        else:
            del mot._forward_joint_layer


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _check_shifts(model):
    from fastwam.loop.losses import assert_shifts
    assert_shifts(model.train_video_scheduler.shift, model.train_action_scheduler.shift)


@torch.no_grad()
def _inputs(teacher, sample):
    device, dtype = teacher.device, teacher.torch_dtype
    if "input_latents" in sample:
        clean = sample["input_latents"].to(device=device, dtype=dtype)
    else:
        # Eager frozen VAE avoids the base policy's inference cudagraph buffers.
        clean = teacher.vae.model.encode(sample["video"].to(device=device, dtype=dtype), teacher.vae.scale)
    proprio = sample["proprio"].to(device=device, dtype=dtype)
    if proprio.ndim == 3:
        proprio = proprio[:, 0]
    if bool(torch.as_tensor(sample.get("is_augmented", False)).any()):
        raise ValueError("Conversion diagnostics require unaugmented clips")
    return {"clean": clean, "action": sample["action"].to(device=device, dtype=dtype),
            "context": sample["context"].to(device=device, dtype=dtype),
            "context_mask": sample["context_mask"].to(device=device, dtype=torch.bool),
            "proprio": proprio}


def _noise(inputs, seed, window_id):
    generator = torch.Generator(device=inputs["clean"].device).manual_seed(seed + int(window_id))
    return tuple(torch.randn(inputs[key].shape, generator=generator,
                             device=inputs[key].device, dtype=inputs[key].dtype)
                 for key in ("clean", "action"))


def _noisy_inputs(teacher, inputs, noise, tau):
    timestep = torch.tensor([tau * 1000], device=teacher.device, dtype=torch.float32)
    video = teacher.train_video_scheduler.add_noise(inputs["clean"], noise[0], timestep)
    video[:, :, :1] = inputs["clean"][:, :, :1]
    action = teacher.train_action_scheduler.add_noise(inputs["action"], noise[1], timestep)
    return video, action, timestep


def _predict(model, inputs, noisy):
    # Each model receives the same normalized raw proprio and encodes it itself.
    context, mask = model._append_proprio_to_context(inputs["context"], inputs["context_mask"], inputs["proprio"])
    video, action, timestep = noisy
    return model._predict_joint_noise(video, action, timestep, timestep, context, mask, True)[1]


def _validate_checkpoint(path, expected_arch):
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if payload.get("format") != "loopwam_v1" or payload["meta"]["arch"] != expected_arch:
        raise ValueError(f"{path} must contain canonical architecture {expected_arch}")
    if payload.get("step") not in (None, 0):
        raise ValueError(f"{path} is a trained checkpoint; D2 requires converted initialization")
    if float(payload["meta"].get("sigma_shift", 5)) != 5:
        raise ValueError(f"{path} has incompatible sigma shift")
    if expected_arch == "loopwam" and payload["meta"]["lora_rank"] != 32:
        raise ValueError("D2 rank-32 checkpoint must have meta.lora_rank == 32")
    return payload


def _load_student(path, expected_arch, teacher):
    """Strict loading, with the already loaded teacher VAE shared read-only."""
    from fastwam.loop.convert import build_experts
    from fastwam.loop.model import LoopWAM
    from fastwam.models.wan22.wan_video_dit import precompute_freqs_cis, precompute_freqs_cis_3d
    payload = _validate_checkpoint(path, expected_arch)
    meta = payload["meta"]
    if meta["video_config"]["text_dim"] != teacher.text_dim or meta["proprio_dim"] != teacher.proprio_dim:
        raise ValueError("Student and teacher disagree on text/proprio input dimensions")
    video, action = build_experts(expected_arch, meta["lora_rank"], device="meta",
                                 tiny_config={"video": meta["video_config"], "action": meta["action_config"]})
    video.freqs = precompute_freqs_cis_3d(video.attn_head_dim)
    action.freqs = precompute_freqs_cis(action.attn_head_dim, 1024)
    for kind, expert in (("video", video), ("action", action)):
        expert.load_state_dict(payload[kind], strict=True, assign=True)
        expert.to(device=teacher.device, dtype=teacher.torch_dtype)
    model = LoopWAM(video, action, teacher.vae, meta, device=teacher.device,
                    training=False, loss_recipe="L2")
    model.proprio_encoder.load_state_dict(payload["proprio"], strict=True)
    model.requires_grad_(False).eval()
    _check_shifts(model)
    return model


@torch.no_grad()
def run_d1(teacher, dataset, count, seed, tau, output, save_features=False):
    indices = select_training_indices(dataset.manifest, count, seed)
    depths = {kind: len(teacher.mot.mixtures[kind].blocks) for kind in ("video", "action")}
    if set(depths.values()) != {30}:
        raise ValueError(f"D1 requires thirty teacher layers in both streams, got {depths}")
    features = {kind: torch.empty(30, count, teacher.mot.mixtures[kind].hidden_dim)
                for kind in ("video", "action")}
    windows, total_seconds = [], time.perf_counter()
    with torch.autocast(torch.device(teacher.device).type, dtype=torch.bfloat16):
        for position, index in enumerate(indices):
            sample = default_collate([dataset[index]])
            window = int(sample["window_id"][0])
            inputs = _inputs(teacher, sample)
            noisy = _noisy_inputs(teacher, inputs, _noise(inputs, seed, window), tau)
            with capture_layers(teacher.mot) as captured:
                _predict(teacher, inputs, noisy)
            for kind in features:
                if len(captured[kind]) != 30:
                    raise RuntimeError(f"D1 captured {len(captured[kind])} {kind} layers instead of thirty")
                features[kind][:, position] = torch.stack(captured[kind])
            windows.append(window)
            if (position + 1) % 25 == 0 or position + 1 == count:
                LOG.info("D1 captured %d/%d training clips", position + 1, count)
    _synchronize(teacher.device)
    inference_seconds = time.perf_counter() - total_seconds
    if save_features:
        path = Path(output) / "d1_features.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"window_ids": windows, "features": features}, temporary)
        temporary.replace(path)
    record = {"clips": len(windows), "window_ids": windows, "seed": seed, "tau": tau,
              "split": "train", "sampling": "seeded uniform unpadded starts without replacement, sorted by window ID",
              "pooling": "arithmetic mean of all tokens after each complete joint MoT block; no feature normalization before CKA",
              "video_tokens": "all first-frame and future-frame tokens",
              "action_tokens": "all 32 action tokens (unpadded clips only)",
              "layer_indices": list(range(30)), "layer_numbering": "zero based teacher block indices",
              "angular_definition": "mean across clips of acos of the cosine between pooled layer vectors, radians",
              "cka_definition": "centered linear Gram matrices, normalized Frobenius inner product; biased linear CKA",
              "capture_seconds": inference_seconds,
              "features_shape": {kind: list(value.shape) for kind, value in features.items()},
              "similarity": {kind: layer_similarity(value) for kind, value in features.items()}}
    record["total_seconds"] = time.perf_counter() - total_seconds
    _write_json(Path(output) / "d1.json", record)
    return record


@torch.no_grad()
def run_d2(teacher, dataset, checkpoints, seed, output):
    from fastwam.loop.diagnostics import panel_indices
    start = time.perf_counter()
    dense = _load_student(Path(checkpoints) / "untied30.pt", "untied30", teacher)
    loop = _load_student(Path(checkpoints) / "loopwam_r32.pt", "loopwam", teacher)
    load_seconds = time.perf_counter() - start
    records, windows = [], []
    timing = {name: 0.0 for name in ("teacher", "untied30", "loopwam_r32", "loopwam_r0")}
    with torch.autocast(torch.device(teacher.device).type, dtype=torch.bfloat16):
        for position, index in enumerate(panel_indices(dataset)):
            sample = default_collate([dataset[index]])
            window = int(sample["window_id"][0])
            inputs = _inputs(teacher, sample)
            noise = _noise(inputs, seed, window)
            is_pad = sample["action_is_pad"].to(teacher.device, dtype=torch.bool)
            for tau in TAU_GRID:
                noisy = _noisy_inputs(teacher, inputs, noise, tau)
                _synchronize(teacher.device)
                tick = time.perf_counter()
                target = _predict(teacher, inputs, noisy)
                _synchronize(teacher.device)
                timing["teacher"] += time.perf_counter() - tick
                row = {"window_id": window, "tau": tau, "models": {}}
                for name, model in (("untied30", dense), ("loopwam_r32", loop), ("loopwam_r0", loop)):
                    _synchronize(teacher.device)
                    tick = time.perf_counter()
                    if name == "loopwam_r0":
                        with without_lora(model):
                            prediction = _predict(model, inputs, noisy)
                    else:
                        prediction = _predict(model, inputs, noisy)
                    _synchronize(teacher.device)
                    timing[name] += time.perf_counter() - tick
                    row["models"][name] = masked_error(prediction, target, is_pad)
                records.append(row)
            windows.append(window)
            LOG.info("D2 measured %d/20 held-out clips at five timesteps", position + 1)
    aggregate = {}
    for name in ("untied30", "loopwam_r32", "loopwam_r0"):
        aggregate[name] = {}
        for label, rows in [("all", records)] + [(str(tau), [r for r in records if r["tau"] == tau]) for tau in TAU_GRID]:
            squared = sum(row["models"][name]["squared_error"] for row in rows)
            valid = sum(row["models"][name]["valid_elements"] for row in rows)
            aggregate[name][label] = {"squared_error": squared, "valid_elements": valid, "mse": squared / valid}
    record = {"clips": len(windows), "window_ids": windows, "split": "validation", "seed": seed,
              "panel": "one unpadded midpoint clip per held-out demonstration", "tau_grid": list(TAU_GRID),
              "timestep_definition": "tau is sigma=timestep/1000, not an unshifted scheduler quantile",
              "noise": "one bf16 video/action noise draw per window using seed+window_id, reused for all timesteps and models",
              "inputs": "same noisy latents, noisy actions, text, normalized raw proprio and timesteps; each model applies its own proprio encoder",
              "metric": "unweighted action velocity squared error against teacher, divided by valid action elements; padding excluded",
              "configuration": [4, 4], "rank_zero": "all SlotLinear adapter terms temporarily disabled in both rank-32 experts; no parameter mutation or training",
              "full_rank": "not materialized at production size; exact folding is covered by CPU tiny-model numerical tests",
              "aggregate": aggregate, "per_clip_and_tau": records, "forward_seconds": timing,
              "student_load_seconds": load_seconds, "total_seconds": time.perf_counter() - start}
    _write_json(Path(output) / "d2.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="checkpoints/fastwam_release/libero_uncond_2cam224.pt")
    parser.add_argument("--checkpoints", default="checkpoints/loopwam_v1", help="Directory containing untied30.pt and loopwam_r32.pt")
    parser.add_argument("--manifest", required=True, help="Existing immutable training split_manifest.json")
    parser.add_argument("--output", required=True, help="Directory for measured d1.json, d2.json and provenance.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-clips", type=int, default=1000, help="D1 clip count; D2 always uses twenty held-out demos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d1-tau", type=float, default=0.5)
    parser.add_argument("--only", choices=("both", "d1", "d2"), default="both")
    parser.add_argument("--stats", help="Teacher normalization JSON; defaults to the JSON beside --teacher")
    parser.add_argument("--text-cache", default="data/text_embeds_cache/libero")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--save-features", action="store_true", help="Also retain D1 pooled CPU features for later analysis")
    args = parser.parse_args(argv)
    if not 0 < args.d1_tau < 1 or args.cpu_threads < 1 or args.seed < 0:
        parser.error("Require 0 < --d1-tau < 1, positive --cpu-threads and nonnegative --seed")
    torch.set_num_threads(args.cpu_threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from fastwam.loop.data import build_dataset, file_sha256, manifest_digest, ManifestDataset
    from fastwam.loop.model import load_teacher
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    teacher_path = Path(args.teacher)
    stats = Path(args.stats) if args.stats else teacher_path.with_name(teacher_path.stem + "_dataset_stats.json")
    if not stats.is_file():
        raise FileNotFoundError(f"Teacher normalization JSON is missing: {stats}")
    manifest = json.loads(Path(args.manifest).read_text())
    checkpoint_paths = {"loopwam_r32": Path(args.checkpoints) / "loopwam_r32.pt"}
    if args.only != "d1":
        checkpoint_paths["untied30"] = Path(args.checkpoints) / "untied30.pt"
    metas = {name: _validate_checkpoint(path, "loopwam" if name == "loopwam_r32" else name)["meta"]
             for name, path in checkpoint_paths.items()}
    LOG.info("Hashing teacher and initialization checkpoints for provenance")
    provenance = {"teacher": {"path": str(teacher_path.resolve()), "sha256": file_sha256(teacher_path)},
                  "checkpoints": {name: {"path": str(path.resolve()), "sha256": file_sha256(path), "arch": metas[name]["arch"]}
                                  for name, path in checkpoint_paths.items()},
                  "stats": {"path": str(stats.resolve()), "sha256": file_sha256(stats)},
                  "manifest": {"path": str(Path(args.manifest).resolve()), "sha256": manifest_digest(manifest)},
                  "seed": args.seed, "device": args.device, "precision": "bf16 inference with fp64 CPU similarity reductions",
                  "sigma_shift": 5.0, "requested_diagnostics": args.only,
                  "svd_energy": metas["loopwam_r32"]["svd_energy"]}
    _write_json(output / "provenance.json", provenance)
    dataset = build_dataset(manifest, "train", stats=stats, text_cache=args.text_cache)
    validation = ManifestDataset(dataset.dataset, manifest, "validation")
    teacher = load_teacher(teacher_path, args.device)
    _check_shifts(teacher)
    with torch.inference_mode():
        if args.only != "d2":
            run_d1(teacher, dataset, args.max_clips, args.seed, args.d1_tau, output, args.save_features)
        if args.only != "d1":
            run_d2(teacher, validation, args.checkpoints, args.seed, output)
    LOG.info("Completed requested measured diagnostics in %s", output)


if __name__ == "__main__":
    main()
