"""Wan blocks with shared linear weights and explicit, checkpoint-safe loop slots."""
from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fastwam.models.wan22.wan_video_dit import (
    DiTBlock, RMSNorm, flash_attention, modulate, rope_apply,
)


def _check_slot(slot: int, count: int) -> None:
    if not isinstance(slot, int) or not 0 <= slot < count:
        raise ValueError(f"slot must be an integer in [0, {count}), got {slot!r}")


@torch.no_grad()
def factor_residual(residual: torch.Tensor, rank: int, seed: int = 0):
    """Return B, A and captured squared Frobenius energy; preserve RNG state.

    Full residual rank uses exact SVD. Production low ranks use a deterministic
    randomized range finder with oversampling and two power iterations, avoiding
    the cubic cost of a dense full SVD on every 8192 x 2048 FFN matrix.
    """
    if rank < 0 or residual.ndim != 2:
        raise ValueError("rank must be nonnegative and residual must be a matrix")
    matrix = residual.float()
    r = min(rank, *matrix.shape)
    total = float(matrix.square().sum())
    if r == 0 or total == 0:
        return (matrix.new_zeros(matrix.shape[0], r),
                matrix.new_zeros(r, matrix.shape[1]), 1.0 if total == 0 else 0.0)
    if r == min(matrix.shape):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    else:
        devices = [matrix.device.index] if matrix.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if matrix.is_cuda:
                with torch.cuda.device(matrix.device):
                    torch.cuda.manual_seed(seed)
            u, s, v = torch.svd_lowrank(matrix, q=min(r + 8, min(matrix.shape)), niter=2)
        vh = v.mT
    b, a = u[:, :r] * s[:r], vh[:r]
    return b.contiguous(), a.contiguous(), min(1.0, float(s[:r].square().sum()) / total)


class SlotLinear(nn.Module):
    """W x + b + B[slot] A[slot] x + delta_bias[slot], without LoRA scaling."""

    def __init__(self, base: nn.Linear, rank: int = 32, num_slots: int = 4):
        super().__init__()
        if rank < 0 or num_slots < 1:
            raise ValueError("rank must be nonnegative and num_slots positive")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.num_slots = min(rank, base.in_features, base.out_features), num_slots
        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.bias_delta = (nn.Parameter(base.weight.new_zeros(num_slots, self.out_features))
                           if self.bias is not None else None)
        self.lora_a = nn.Parameter(base.weight.new_zeros(num_slots, self.rank, self.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(num_slots, self.out_features, self.rank))
        self.energy_report = []

    def forward(self, x: torch.Tensor, slot: int) -> torch.Tensor:
        _check_slot(slot, self.num_slots)
        bias = None if self.bias is None else self.bias + self.bias_delta[slot]
        output = F.linear(x, self.weight, bias)
        if self.rank:
            output = output + F.linear(F.linear(x, self.lora_a[slot]), self.lora_b[slot])
        return output

    @classmethod
    @torch.no_grad()
    def from_linears(cls, linears: Sequence[nn.Linear], rank: int = 32, seed: int = 0):
        if not linears:
            raise ValueError("At least one source linear is required")
        result = cls(linears[0], rank, len(linears))
        result.weight.copy_(sum(layer.weight.float() for layer in linears) / len(linears))
        if result.bias is not None:
            result.bias.copy_(sum(layer.bias.float() for layer in linears) / len(linears))
        for slot, layer in enumerate(linears):
            residual = layer.weight.float() - result.weight.float()
            b, a, energy = factor_residual(residual, rank, seed + slot)
            result.lora_a[slot].copy_(a)
            result.lora_b[slot].copy_(b)
            if result.bias is not None:
                result.bias_delta[slot].copy_(layer.bias - result.bias)
            result.energy_report.append({"slot": slot, "rank": result.rank,
                                         "captured_energy": energy,
                                         "residual_energy": float(residual.square().sum())})
        return result


class SlotRMSNorm(nn.Module):
    def __init__(self, base: RMSNorm, num_slots: int):
        super().__init__()
        self.eps = base.eps
        self.weight = nn.Parameter(base.weight.detach().repeat(num_slots, 1))

    def forward(self, x, slot):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight[slot]


class SlotLayerNorm(nn.Module):
    def __init__(self, base: nn.LayerNorm, num_slots: int):
        super().__init__()
        self.eps, self.normalized_shape = base.eps, base.normalized_shape
        self.weight = nn.Parameter(base.weight.detach().repeat(num_slots, 1))
        self.bias = nn.Parameter(base.bias.detach().repeat(num_slots, 1))

    def forward(self, x, slot):
        return F.layer_norm(x, self.normalized_shape, self.weight[slot], self.bias[slot], self.eps)


class SlotDiTBlock(nn.Module):
    """A real Wan block, with a single shared base and four layer identity slots."""

    def __init__(self, base: DiTBlock, rank: int = 32, num_slots: int = 4):
        super().__init__()
        block = copy.deepcopy(base)
        self.hidden_dim, self.attn_head_dim = block.hidden_dim, block.attn_head_dim
        self.num_heads, self.ffn_dim = block.num_heads, block.ffn_dim
        self.num_slots, self.rank = num_slots, rank
        self.self_attn, self.cross_attn = block.self_attn, block.cross_attn
        self.norm1, self.norm2 = block.norm1, block.norm2
        self.norm3 = SlotLayerNorm(block.norm3, num_slots)
        self.ffn, self.gate = block.ffn, block.gate
        self.modulation = block.modulation
        self.modulation_delta = nn.Parameter(block.modulation.new_zeros(num_slots, *block.modulation.shape))
        for attention in (self.self_attn, self.cross_attn):
            for name in ("q", "k", "v", "o"):
                setattr(attention, name, SlotLinear(getattr(attention, name), rank, num_slots))
            for name in ("norm_q", "norm_k"):
                setattr(attention, name, SlotRMSNorm(getattr(attention, name), num_slots))
        for index in (0, 2):
            self.ffn[index] = SlotLinear(self.ffn[index], rank, num_slots)
        self.energy_report = []

    @classmethod
    @torch.no_grad()
    def from_blocks(cls, blocks: Sequence[DiTBlock], rank: int = 32, seed: int = 0):
        if len(blocks) != 4:
            raise ValueError("LoopWAM core blocks require exactly four source layers")
        result = cls(blocks[0], rank, len(blocks))
        result.modulation.copy_(sum(block.modulation.float() for block in blocks) / len(blocks))
        for slot, block in enumerate(blocks):
            result.modulation_delta[slot].copy_(block.modulation - result.modulation)
        for index, (name, module) in enumerate(list(result.named_modules())):
            if isinstance(module, SlotLinear):
                layers = [block.get_submodule(name) for block in blocks]
                folded = SlotLinear.from_linears(layers, rank, seed + index * 4)
                parent, _, leaf = name.rpartition(".")
                setattr(result.get_submodule(parent), leaf, folded)
                result.energy_report.extend(dict(row, linear=name) for row in folded.energy_report)
            elif isinstance(module, (SlotRMSNorm, SlotLayerNorm)):
                for slot, block in enumerate(blocks):
                    original = block.get_submodule(name)
                    module.weight[slot].copy_(original.weight)
                    if isinstance(module, SlotLayerNorm):
                        module.bias[slot].copy_(original.bias)
        return result

    def attention_io(self, x, t_mod, freqs, slot):
        _check_slot(slot, self.num_slots)
        has_seq = t_mod.ndim == 4
        table = (self.modulation + self.modulation_delta[slot]).to(t_mod)
        pieces = (table + t_mod).chunk(6, dim=2 if has_seq else 1)
        if has_seq:
            pieces = tuple(piece.squeeze(2) for piece in pieces)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = pieces
        inputs = modulate(self.norm1(x), shift_msa, scale_msa)
        attn = self.self_attn
        q = rope_apply(attn.norm_q(attn.q(inputs, slot), slot), freqs, attn.num_heads)
        k = rope_apply(attn.norm_k(attn.k(inputs, slot), slot), freqs, attn.num_heads)
        return q, k, attn.v(inputs, slot), gate_msa, shift_mlp, scale_mlp, gate_mlp

    def post_attention(self, x, attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                       context, context_mask, slot):
        _check_slot(slot, self.num_slots)
        x = self.gate(x, gate_msa, self.self_attn.o(attn_out, slot))
        inputs, cross = self.norm3(x, slot), self.cross_attn
        q = cross.norm_q(cross.q(inputs, slot), slot)
        k = cross.norm_k(cross.k(context, slot), slot)
        v = cross.v(context, slot)
        if context_mask is not None and context_mask.ndim == 3:
            context_mask = context_mask.unsqueeze(1)
        cross_out = flash_attention(q, k, v, cross.num_heads, context_mask)
        x = x + cross.o(cross_out, slot)
        inputs = modulate(self.norm2(x), shift_mlp, scale_mlp)
        output = self.ffn[2](self.ffn[1](self.ffn[0](inputs, slot)), slot)
        return self.gate(x, gate_mlp, output)

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask=None, slot=0):
        q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attention_io(x, t_mod, freqs, slot)
        output = flash_attention(q, k, v, self.num_heads, self_attn_mask)
        return self.post_attention(x, output, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                                   context, context_mask, slot)
