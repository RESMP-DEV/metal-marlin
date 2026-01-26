"""MoE dispatcher utilities for routed and shared experts.

This module provides a simple, correct Top-K dispatcher for inference that:
- Routes tokens to top-k experts based on gate logits
- Supports an always-on shared expert
- Works with quantized expert FFNs (FP8/INT4 via QuantizedLinear)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn as nn

from ..moe_dispatch import (
    gather_for_experts,
    group_tokens_by_expert_full,
    scatter_expert_outputs,
)


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
