"""MoE helper ops for fused gate/up expert projections."""

from __future__ import annotations

from ._compat import HAS_TORCH, torch
from .kernels import moe_expert_gemm_fp4

if not HAS_TORCH or torch is None:  # pragma: no cover - torch-only module
    raise ImportError("metal_marlin.moe_ops requires PyTorch to be installed")

import torch.nn.functional as F


def _infer_group_size(k_dim: int, scales: torch.Tensor, *, name: str) -> int:
    if scales.ndim < 2:
        raise ValueError(f"{name} scales must have at least 2 dims, got {scales.shape}")
    groups = int(scales.shape[1])
    if groups <= 0 or k_dim % groups != 0:
        raise ValueError(
            f"{name} scales shape {scales.shape} incompatible with k_dim={k_dim}"
        )
    return k_dim // groups


def _flatten_hidden(hidden: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if hidden.dim() == 2:
        return hidden, hidden.shape
    if hidden.dim() == 3:
        batch, seq, hidden_dim = hidden.shape
        return hidden.view(-1, hidden_dim), (batch, seq, hidden_dim)
    raise ValueError(f"hidden must be 2D or 3D, got shape {hidden.shape}")


def fused_moe_forward(
    hidden: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_packed: torch.Tensor,
    down_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Fused MoE forward for gate+up projection followed by down projection.

    Args:
        hidden: [tokens, hidden] or [batch, seq, hidden] activations.
        gate_up_packed: [num_experts, hidden/8, 2*intermediate] packed FP4.
        gate_up_scales: [num_experts, hidden/group, 2*intermediate] scales.
        down_packed: [num_experts, intermediate/8, hidden] packed FP4.
        down_scales: [num_experts, intermediate/group, hidden] scales.
        expert_ids: [tokens, top_k] expert indices.
        probs: [tokens, top_k] routing probabilities.

    Returns:
        Output tensor with same leading shape as hidden and last dim = hidden.
    """
    hidden_flat, orig_shape = _flatten_hidden(hidden)

    if expert_ids.shape[0] != hidden_flat.shape[0]:
        raise ValueError(
            "expert_ids/probs first dim must match hidden tokens: "
            f"{expert_ids.shape[0]} vs {hidden_flat.shape[0]}"
        )

    k_gate_up = hidden_flat.shape[-1]
    if gate_up_packed.shape[1] * 8 != k_gate_up:
        raise ValueError(
            f"gate_up_packed K mismatch: {gate_up_packed.shape[1] * 8} vs {k_gate_up}"
        )
    gate_up_group = _infer_group_size(k_gate_up, gate_up_scales, name="gate_up")

    gate_up = moe_expert_gemm_fp4(
        hidden_flat,
        gate_up_packed,
        gate_up_scales,
        expert_ids,
        probs,
        group_size=gate_up_group,
    )

    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"gate_up output dim must be even, got {gate_up.shape[-1]}"
        )

    intermediate = gate_up.shape[-1] // 2
    gate, up = gate_up.split(intermediate, dim=-1)
    hidden_mid = F.silu(gate) * up

    k_down = hidden_mid.shape[-1]
    if down_packed.shape[1] * 8 != k_down:
        raise ValueError(
            f"down_packed K mismatch: {down_packed.shape[1] * 8} vs {k_down}"
        )
    down_group = _infer_group_size(k_down, down_scales, name="down")

    output = moe_expert_gemm_fp4(
        hidden_mid,
        down_packed,
        down_scales,
        expert_ids,
        probs,
        group_size=down_group,
    )

    if len(orig_shape) == 3:
        return output.view(orig_shape[0], orig_shape[1], -1)
    return output


__all__ = ["fused_moe_forward"]
