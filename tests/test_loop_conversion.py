"""Numerical conversion tests use actual Wan blocks, including asymmetric attention."""
import copy
import importlib.util

import pytest
import torch
from torch import nn

from fastwam.models.wan22.wan_video_dit import (
    CrossAttention, DiTBlock, precompute_freqs_cis,
)


def _slots():
    assert importlib.util.find_spec("fastwam.loop.slots") is not None, "slot implementation is missing"
    from fastwam.loop import slots
    return slots


def _blocks():
    torch.manual_seed(15)
    blocks = []
    for _ in range(4):
        block = DiTBlock(12, 4, 4, 20)
        block.cross_attn = CrossAttention(12, 4, 2)
        with torch.no_grad():
            for name, p in block.named_parameters():
                if "norm" in name:
                    p.add_(torch.randn_like(p) * 0.15)
        blocks.append(block)
    return blocks


@pytest.mark.parametrize("per_token", [False, True])
def test_full_rank_slots_reproduce_real_blocks_with_masks(per_token):
    # Missing FFN adapters, wrong slot norms, or double residuals break equality.
    source = _blocks()
    folded = _slots().SlotDiTBlock.from_blocks(source, rank=32)
    x, ctx = torch.randn(2, 3, 12), torch.randn(2, 5, 12)
    t = torch.randn(2, 3, 6, 12) if per_token else torch.randn(2, 6, 12)
    freqs = precompute_freqs_cis(4, 3).view(3, 1, 2)
    context_mask = torch.ones(2, 3, 5, dtype=torch.bool)
    context_mask[:, :, -1] = False
    mask = torch.ones(3, 3, dtype=torch.bool).tril()
    for slot, original in enumerate(source):
        want = original(x, ctx, t, freqs, context_mask, mask)
        got = folded(x, ctx, t, freqs, context_mask, mask, slot=slot)
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)
    assert len(folded.energy_report) == 40
    assert min(row["captured_energy"] for row in folded.energy_report) > 0.99999


def test_slots_keep_explicit_identity_during_checkpoint_backward():
    from torch.utils.checkpoint import checkpoint
    folded = _slots().SlotDiTBlock.from_blocks(_blocks(), rank=4)
    reference = copy.deepcopy(folded)
    x = torch.randn(1, 3, 12, requires_grad=True)
    ctx, t = torch.randn(1, 2, 12), torch.randn(1, 6, 12)
    freqs = precompute_freqs_cis(4, 3).view(3, 1, 2)
    def run(model, value, checked):
        for slot in (3, 0, 2):
            if checked:
                value = checkpoint(model, value, ctx, t, freqs, slot=slot, use_reentrant=False)
            else:
                value = model(value, ctx, t, freqs, slot=slot)
        return value
    run(folded, x, True).square().sum().backward()
    run(reference, x.detach().clone().requires_grad_(True), False).square().sum().backward()
    for (name, p), (_, expected) in zip(folded.named_parameters(), reference.named_parameters()):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, expected.grad, rtol=1e-5, atol=1e-5)


def test_rank_zero_retains_exact_bias_and_shares_base_weights():
    modules = [nn.Linear(3, 2) for _ in range(4)]
    folded = _slots().SlotLinear.from_linears(modules, rank=0)
    expected_weight = sum(m.weight for m in modules) / 4
    x = torch.zeros(2, 3)
    pointer = folded.weight.data_ptr()
    for slot in range(4):
        torch.testing.assert_close(folded(x, slot), modules[slot](x))
        assert folded.weight.data_ptr() == pointer
    torch.testing.assert_close(folded.weight, expected_weight)
    for slot in (-1, 4):
        with pytest.raises(ValueError, match="slot"):
            folded(x, slot)


def test_truncated_svd_is_reproducible_and_reports_actual_energy():
    torch.manual_seed(27)
    matrix = torch.randn(120, 100)
    state = torch.random.get_rng_state().clone()
    b, a, energy = _slots().factor_residual(matrix, rank=7, seed=31)
    assert torch.equal(state, torch.random.get_rng_state())
    b2, a2, energy2 = _slots().factor_residual(matrix, rank=7, seed=31)
    torch.testing.assert_close(b @ a, b2 @ a2)
    assert energy == energy2
    assert energy == pytest.approx(float((b @ a).square().sum() / matrix.square().sum()), abs=1e-6)
    assert 0 < energy < 1


def _convert():
    assert importlib.util.find_spec("fastwam.loop.convert") is not None, "conversion implementation is missing"
    from fastwam.loop import convert
    return convert


def tiny_configs():
    return {
        "video": dict(hidden_dim=18, in_dim=2, ffn_dim=28, out_dim=2,
                      text_dim=10, freq_dim=8, num_heads=3, attn_head_dim=6,
                      patch_size=(1, 2, 2)),
        "action": dict(hidden_dim=12, action_dim=7, ffn_dim=20, text_dim=10,
                       freq_dim=8, num_heads=3, attn_head_dim=6, cross_num_heads=2),
    }


def _source_experts():
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    video = WanVideoDiT(hidden_dim=24, in_dim=2, ffn_dim=36, out_dim=2,
                        text_dim=10, freq_dim=8, eps=1e-6, patch_size=(1, 2, 2),
                        num_heads=4, attn_head_dim=6, num_layers=30, has_image_input=False)
    action = ActionDiT(hidden_dim=16, action_dim=7, ffn_dim=28, text_dim=10,
                       freq_dim=8, eps=1e-6, num_heads=4, attn_head_dim=6, num_layers=30)
    return video, action


def test_width_slicing_preserves_whole_heads_and_segmented_modulation():
    convert = _convert()
    sv, sa = _source_experts()
    vcfg, acfg = convert.expert_configs("untied30", tiny_configs())
    tv, ta = convert.build_experts("untied30", tiny_config=tiny_configs())
    with torch.no_grad():
        # Constant row groups expose interpolation crossing modulation boundaries.
        for group in range(6):
            sa.time_projection[1].bias[group * 16:(group + 1) * 16] = group
    video = convert.slice_expert_state(sv.state_dict(), "video", vcfg)
    action = convert.slice_expert_state(sa.state_dict(), "action", acfg)
    tv.load_state_dict(video, strict=True)
    ta.load_state_dict(action, strict=True)
    # round(linspace(0,3,3)) = [0,2,3]; H2 = [0,3].
    h3 = torch.tensor([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    h2 = torch.tensor([0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23])
    original = sv.blocks[0].self_attn.q.weight
    torch.testing.assert_close(tv.blocks[0].self_attn.q.weight, original[h3][:, h3])
    torch.testing.assert_close(ta.blocks[0].cross_attn.q.bias, sa.blocks[0].cross_attn.q.bias[h2])
    # Independently evaluate linear interpolation at its exact endpoint samples.
    got = ta.blocks[0].self_attn.q.weight
    want = sa.blocks[0].self_attn.q.weight[h3]
    torch.testing.assert_close(got[:, 0], want[:, 0] * (16 / 12) ** 0.5)
    torch.testing.assert_close(got[:, -1], want[:, -1] * (16 / 12) ** 0.5)
    torch.testing.assert_close(ta.time_projection[1].bias.reshape(6, 12),
                               torch.arange(6).float()[:, None].expand(6, 12))


def test_full_rank_fold_reproduces_thirty_layer_composition():
    convert = _convert()
    torch.manual_seed(21)
    source = _source_experts()[1]
    folded = convert.fold_blocks(source.blocks, rank=32)
    x, ctx = torch.randn(1, 3, 16), torch.randn(1, 2, 16)
    t = torch.randn(1, 6, 16) * 0.1
    freqs = precompute_freqs_cis(6, 3).view(3, 1, 3)
    expected = x
    for block in source.blocks:
        expected = block(expected, ctx, t, freqs)
    got = x
    for block in folded[:3]:
        got = block(got, ctx, t, freqs)
    for slot in range(4):
        for block in folded[3:9]:
            got = block(got, ctx, t, freqs, slot=slot)
    for block in folded[9:]:
        got = block(got, ctx, t, freqs)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("arch,video_map,action_map", [
    ("untied12", [0, 1, 3, 5, 7, 10, 13, 16, 19, 22, 25, 29],
                 [0, 1, 3, 5, 7, 10, 13, 16, 19, 22, 25, 29]),
    ("v30a12", list(range(30)), [0, 1, 2, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
    ("loopwam", None, None),
])
def test_checkpoint_conversion_roundtrip_and_control_mapping(tmp_path, arch, video_map, action_map):
    convert = _convert()
    sv, sa = _source_experts()
    for expert in (sv, sa):
        with torch.no_grad():
            for layer, block in enumerate(expert.blocks):
                block.modulation.fill_(layer)
    proprio = nn.Linear(8, 10).state_dict()
    mot = {f"mixtures.video.{k}": v for k, v in sv.state_dict().items()}
    mot.update({f"mixtures.action.{k}": v for k, v in sa.state_dict().items()})
    teacher, output = tmp_path / "teacher.pt", tmp_path / "converted.pt"
    torch.save({"mot": mot, "proprio_encoder": proprio}, teacher)
    convert.convert_checkpoint(teacher, output, arch, lora_rank=4,
                               device="cpu", tiny_config=tiny_configs(), output_dtype=torch.float32)
    payload = torch.load(output, weights_only=True)
    assert payload["format"] == "loopwam_v1"
    assert payload["meta"]["head_indices"] == {"self": [0, 2, 3], "video_cross": [0, 2, 3], "action_cross": [0, 3]}
    torch.testing.assert_close(payload["proprio"]["weight"], proprio["weight"])
    video, action = convert.build_experts(arch, lora_rank=4, tiny_config=tiny_configs())
    video.load_state_dict(payload["video"], strict=True)
    action.load_state_dict(payload["action"], strict=True)
    if video_map is not None:
        for expert, mapping in ((video, video_map), (action, action_map)):
            for block, layer in zip(expert.blocks, mapping):
                # Modulation interpolation follows original last-axis scaling.
                scale = 1.0 if expert is video else (16 / 12) ** 0.5
                torch.testing.assert_close(block.modulation, torch.full_like(block.modulation, layer * scale))
    else:
        assert len(payload["meta"]["svd_energy"]) == 480
        assert len(video.blocks) == len(action.blocks) == 12
