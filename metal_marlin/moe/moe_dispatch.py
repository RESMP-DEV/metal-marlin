"""MoE dispatcher utilities for routed and shared experts.

This module provides a simple, correct Top-K dispatcher for inference that:
- Routes tokens to top-k experts based on gate logits
- Supports an always-on shared expert
- Works with quantized expert FFNs (FP8/INT4 via QuantizedLinear)
- Fused dispatch + shared expert kernel for reduced memory traffic
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn as nn

from ..moe_dispatch import (
    gather_for_experts,
    group_tokens_by_expert_full,
    scatter_expert_outputs,
)

if TYPE_CHECKING:
    pass


class ExpertForward(Protocol):
    """Protocol for expert forward pass callable."""

    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        """Forward pass for an expert on a batch of activations."""
        ...


class MoEDispatcher(nn.Module):
    """Top-K MoE dispatcher with optional shared expert.

    Args:
        num_experts: Total number of routed experts.
        num_experts_per_tok: Top-k experts per token.
        experts: Sequence of expert modules/callables.
        shared_expert: Optional shared expert module run for all tokens.
        shared_expert_weight: Weight applied to shared expert output.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        experts: Sequence[ExpertForward],
        shared_expert: ExpertForward | None = None,
        shared_expert_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(experts)  # type: ignore[arg-type]
        self.shared_expert = shared_expert
        self.shared_expert_weight = shared_expert_weight

    def forward(self, hidden: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        """Dispatch tokens to experts and combine outputs.

        Args:
            hidden: [batch, hidden] or [batch, seq, hidden] activations.
            gate_logits: [tokens, num_experts] router logits (pre-softmax).

        Returns:
            Combined expert output with same shape as hidden.
        """
        if hidden.dim() == 3:
            batch, seq, hidden_dim = hidden.shape
            hidden_flat = hidden.view(-1, hidden_dim)
        else:
            hidden_flat = hidden
            batch = None
            seq = None

        if gate_logits.dim() != 2:
            raise ValueError("gate_logits must be [tokens, num_experts]")
        if gate_logits.shape[0] != hidden_flat.shape[0]:
            raise ValueError("gate_logits batch must match hidden tokens")

        # Route tokens to top-k experts based on gate logits
        routing_probs = gate_logits.softmax(dim=-1)
        topk_weights, topk_indices = torch.topk(
            routing_probs, k=self.num_experts_per_tok, dim=-1
        )

        # Group tokens by expert for batched execution
        dispatch_info = group_tokens_by_expert_full(topk_indices, self.num_experts)
        expert_inputs = gather_for_experts(hidden_flat, dispatch_info)

        out_dim = getattr(self.experts[0], "out_features", hidden_flat.shape[-1])
        expert_outputs = hidden_flat.new_empty((expert_inputs.shape[0], out_dim))

        # Run each expert on its grouped tokens
        for expert_idx in range(self.num_experts):
            start = int(dispatch_info.expert_offsets[expert_idx].item())
            end = int(dispatch_info.expert_offsets[expert_idx + 1].item())
            if start == end:
                continue
            expert_outputs[start:end] = self.experts[expert_idx](expert_inputs[start:end])

        combined = scatter_expert_outputs(expert_outputs, topk_weights, dispatch_info)

        # Shared expert (always active)
        if self.shared_expert is not None:
            combined = combined + self.shared_expert_weight * self.shared_expert(hidden_flat)

        if batch is not None and seq is not None:
            return combined.view(batch, seq, -1)
        return combined


class FusedMoEDispatcher(nn.Module):
    """Fused MoE dispatcher with shared expert computation in a single kernel.

    This dispatcher fuses the entire MoE computation:
        output = shared_expert(x) + sum_k(prob[k] * routed_expert_k(x))

    Instead of separate kernels:
        moe_out = dispatch_moe(x)        -> write to global
        shared_out = shared_expert(x)    -> write to global
        result = moe_out + shared_out    -> read both, write result

    The fused kernel:
        1. Computes shared expert contribution
        2. Accumulates weighted routed expert contributions
        3. Writes final result once

    Memory savings per token (hidden_dim=7168, FP16):
        - Eliminates 2 intermediate writes + 2 reads = 57KB per layer

    Args:
        num_experts: Total number of routed experts.
        num_experts_per_tok: Top-k experts per token.
        expert_gate_up_weights: [num_experts, hidden, 2*intermediate] FP4 packed.
        expert_gate_up_scales: [num_experts, hidden/group, 2*intermediate] scales.
        expert_down_weights: [num_experts, intermediate, hidden] FP4 packed.
        expert_down_scales: [num_experts, intermediate/group, hidden] scales.
        shared_gate_up_weights: [hidden, 2*intermediate] FP4 packed.
        shared_gate_up_scales: [hidden/group, 2*intermediate] scales.
        shared_down_weights: [intermediate, hidden] FP4 packed.
        shared_down_scales: [intermediate/group, hidden] scales.
        group_size: Quantization group size (default 128).
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        expert_gate_up_weights: torch.Tensor,
        expert_gate_up_scales: torch.Tensor,
        expert_down_weights: torch.Tensor,
        expert_down_scales: torch.Tensor,
        shared_gate_up_weights: torch.Tensor,
        shared_gate_up_scales: torch.Tensor,
        shared_down_weights: torch.Tensor,
        shared_down_scales: torch.Tensor,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.group_size = group_size

        # Register weights as buffers (non-trainable for inference)
        self.register_buffer("expert_gate_up_weights", expert_gate_up_weights)
        self.register_buffer("expert_gate_up_scales", expert_gate_up_scales)
        self.register_buffer("expert_down_weights", expert_down_weights)
        self.register_buffer("expert_down_scales", expert_down_scales)
        self.register_buffer("shared_gate_up_weights", shared_gate_up_weights)
        self.register_buffer("shared_gate_up_scales", shared_gate_up_scales)
        self.register_buffer("shared_down_weights", shared_down_weights)
        self.register_buffer("shared_down_scales", shared_down_scales)

        # Lazy import to avoid circular dependency
        self._has_metal = False
        try:
            from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS

            self._has_metal = HAS_METAL and HAS_MPS
        except ImportError:
            pass

    def forward(
        self,
        hidden: torch.Tensor,
        gate_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Fused MoE forward with shared expert.

        Args:
            hidden: [batch, hidden] or [batch, seq, hidden] activations.
            gate_logits: [tokens, num_experts] router logits (pre-softmax).

        Returns:
            Combined expert output with same shape as hidden.
        """
        if hidden.dim() == 3:
            batch, seq, hidden_dim = hidden.shape
            hidden_flat = hidden.view(-1, hidden_dim)
        else:
            hidden_flat = hidden
            batch = None
            seq = None

        if gate_logits.dim() != 2:
            raise ValueError("gate_logits must be [tokens, num_experts]")
        if gate_logits.shape[0] != hidden_flat.shape[0]:
            raise ValueError("gate_logits batch must match hidden tokens")

        # Route tokens to top-k experts
        routing_probs = gate_logits.softmax(dim=-1)
        topk_weights, topk_indices = torch.topk(
            routing_probs, k=self.num_experts_per_tok, dim=-1
        )

        # Ensure indices and probs are on MPS
        if hidden_flat.device.type == "mps":
            topk_indices = topk_indices.to(hidden_flat.device)
            topk_weights = topk_weights.to(hidden_flat.device)

        # Use fused kernel if Metal is available
        if self._has_metal and hidden_flat.device.type == "mps":
            try:
                from ..kernels import (
                    moe_fused_dispatch_shared_fp4,
                )

                output = moe_fused_dispatch_shared_fp4(
                    hidden_states=hidden_flat,
                    shared_gate_up_packed=self.shared_gate_up_weights,
                    shared_gate_up_scales=self.shared_gate_up_scales,
                    shared_down_packed=self.shared_down_weights,
                    shared_down_scales=self.shared_down_scales,
                    routed_gate_up_packed=self.expert_gate_up_weights,
                    routed_gate_up_scales=self.expert_gate_up_scales,
                    routed_down_packed=self.expert_down_weights,
                    routed_down_scales=self.expert_down_scales,
                    expert_ids=topk_indices,
                    expert_probs=topk_weights,
                    group_size=self.group_size,
                )

                if batch is not None and seq is not None:
                    return output.view(batch, seq, -1)
                return output

            except Exception as e:
                # Fall back to sequential computation
                import warnings

                warnings.warn(f"Fused kernel failed, falling back: {e}")

        # Fallback: compute sequentially using standard dispatcher
        # This path doesn't require Metal and works on any device
        fallback_dispatcher = MoEDispatcher(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            experts=[],  # Would need actual expert modules for fallback
        )

        # Build dummy experts that use our quantized weights
        # This is a simplified fallback - full implementation would dequantize
        raise NotImplementedError(
            "Non-fused fallback not implemented. "
            "Use MoEDispatcher with standard expert modules for non-Metal paths."
        )


class FusedSharedExpertAdd(nn.Module):
    """Lightweight module to add shared expert to existing MoE output.

    This is useful when MoE output is already computed (e.g., by another
    system) and you just need to add the shared expert contribution.

    Kernel: moe_add_shared_expert_fp4

    Args:
        gate_up_weights: [hidden/8, 2*intermediate] FP4 packed.
        gate_up_scales: [hidden/group, 2*intermediate] scales.
        down_weights: [intermediate/8, hidden] FP4 packed.
        down_scales: [intermediate/group, hidden] scales.
        group_size: Quantization group size.
    """

    def __init__(
        self,
        gate_up_weights: torch.Tensor,
        gate_up_scales: torch.Tensor,
        down_weights: torch.Tensor,
        down_scales: torch.Tensor,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.group_size = group_size

        self.register_buffer("gate_up_weights", gate_up_weights)
        self.register_buffer("gate_up_scales", gate_up_scales)
        self.register_buffer("down_weights", down_weights)
        self.register_buffer("down_scales", down_scales)

        self._has_metal = False
        try:
            from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS

            self._has_metal = HAS_METAL and HAS_MPS
        except ImportError:
            pass

    def forward(
        self,
        hidden: torch.Tensor,
        moe_output: torch.Tensor,
    ) -> torch.Tensor:
        """Add shared expert to MoE output.

        Args:
            hidden: [tokens, hidden] input activations.
            moe_output: [tokens, hidden] MoE output to add to.

        Returns:
            Output [tokens, hidden] with shared expert added.
        """
        if hidden.shape != moe_output.shape:
            raise ValueError(f"Shape mismatch: hidden {hidden.shape} vs moe_output {moe_output.shape}")

        if self._has_metal and hidden.device.type == "mps":
            try:
                from ..kernels import moe_add_shared_expert_fp4

                return moe_add_shared_expert_fp4(
                    hidden_states=hidden,
                    moe_output=moe_output,
                    shared_gate_up_packed=self.gate_up_weights,
                    shared_gate_up_scales=self.gate_up_scales,
                    shared_down_packed=self.down_weights,
                    shared_down_scales=self.down_scales,
                    group_size=self.group_size,
                )
            except Exception as e:
                import warnings

                warnings.warn(f"Fused add kernel failed: {e}")

        # Fallback: PyTorch implementation
        # Would need to dequantize weights and compute FFN
        raise NotImplementedError("PyTorch fallback not implemented")
