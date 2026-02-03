"""Sorted expert dispatch for improved memory access patterns.

This module provides token sorting strategies that improve cache utilization
during MoE inference by ensuring consecutive tokens route to the same expert.

Key Benefits:
    - Consecutive tokens go to same expert after sorting
    - Reduces random memory access overhead
    - Better cache utilization during expert FFN computation
    - Works alongside or as alternative to Metal kernels

The sorting approach is complementary to batched dispatch:
    - BatchedExpertDispatch: Groups tokens by expert for batched GEMM
    - SortedExpertDispatch: Explicit sorting API with boundary tracking

Both avoid MPS advanced indexing slowness by using argsort + contiguous gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class ExpertBoundary:
    """Boundary information for a single expert's token batch."""

    expert_id: int
    start: int
    end: int

    @property
    def count(self) -> int:
        """Number of tokens assigned to this expert."""
        return self.end - self.start

    def __bool__(self) -> bool:
        """True if expert has any assigned tokens."""
        return self.end > self.start


@dataclass
class SortedDispatchState:
    """State container for sorted dispatch, enabling efficient unsort.

    Attributes:
        sorted_indices: [num_tokens] indices that sort tokens by expert.
        sorted_states: [num_tokens, hidden_dim] tokens in expert-sorted order.
        sorted_experts: [num_tokens, top_k] expert indices in sorted order.
        boundaries: List of ExpertBoundary for each expert with tokens.
        original_shape: Shape of the original hidden_states tensor.
        inverse_indices: [num_tokens] indices to unsort back to original order.
    """

    sorted_indices: torch.Tensor
    sorted_states: torch.Tensor
    sorted_experts: torch.Tensor
    boundaries: list[ExpertBoundary]
    original_shape: tuple[int, ...]
    inverse_indices: torch.Tensor


class SortedExpertDispatch(nn.Module):
    """Sort tokens by expert index before dispatch.

    This improves memory access patterns:
    - Consecutive tokens go to same expert
    - Reduces random access overhead
    - Better cache utilization

    Unlike BatchedExpertDispatch which handles the full dispatch-compute-combine
    pipeline internally, SortedExpertDispatch provides explicit control over
    each phase, making it suitable for:
    - Custom expert implementations
    - Debugging and profiling
    - Integration with external compute frameworks

    Args:
        num_experts: Total number of experts in the MoE layer.
        top_k: Number of experts each token routes to (default 1).
            When top_k > 1, tokens are sorted by their primary expert.
        dtype: Data type for computations (default float16).

    Example:
        >>> dispatch = SortedExpertDispatch(num_experts=64, top_k=2)
        >>> # Route tokens
        >>> probs = torch.softmax(router_logits, dim=-1)
        >>> weights, indices = torch.topk(probs, k=2, dim=-1)
        >>> # Sort and get boundaries
        >>> state = dispatch.dispatch(hidden_states, indices)
        >>> # Process each expert's batch
        >>> outputs = []
        >>> for boundary in state.boundaries:
        ...     batch = state.sorted_states[boundary.start:boundary.end]
        ...     out = experts[boundary.expert_id](batch)
        ...     outputs.append((boundary, out))
        >>> # Unsort to original order
        >>> final = dispatch.unsort_outputs(outputs, state)
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> SortedDispatchState:
        """Sort tokens by expert index for cache-friendly processing.

        Sorts tokens so that all tokens assigned to the same expert are
        contiguous in memory. This enables efficient batched processing
        without random memory access.

        Args:
            hidden_states: [batch, hidden_dim] input activations.
            expert_indices: [batch, top_k] expert assignments per token.
                Uses first column (primary expert) for sorting.

        Returns:
            SortedDispatchState containing sorted tensors and boundaries.
        """
        device = hidden_states.device
        batch_size = hidden_states.shape[0]

        # Sort by primary expert (first column of expert_indices)
        primary_experts = expert_indices[:, 0]
        sorted_indices = torch.argsort(primary_experts, stable=True)

        # Gather in sorted order
        sorted_states = hidden_states[sorted_indices]
        sorted_experts = expert_indices[sorted_indices]

        # Compute inverse permutation for unsorting
        inverse_indices = torch.argsort(sorted_indices)

        # Find boundaries between experts
        boundaries = self._compute_boundaries(sorted_experts, device)

        return SortedDispatchState(
            sorted_indices=sorted_indices,
            sorted_states=sorted_states,
            sorted_experts=sorted_experts,
            boundaries=boundaries,
            original_shape=hidden_states.shape,
            inverse_indices=inverse_indices,
        )

    def _compute_boundaries(
        self,
        sorted_experts: torch.Tensor,
        device: torch.device,
    ) -> list[ExpertBoundary]:
        """Compute expert boundaries in sorted token sequence.

        Args:
            sorted_experts: [batch, top_k] expert indices in sorted order.
            device: Target device.

        Returns:
            List of ExpertBoundary for each expert with assigned tokens.
        """
        primary_experts = sorted_experts[:, 0]
        num_tokens = primary_experts.shape[0]

        # Count tokens per expert
        counts = torch.bincount(primary_experts, minlength=self.num_experts)

        # Compute cumulative offsets
        offsets = torch.zeros(self.num_experts + 1, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(counts, dim=0)

        # Build boundary list for non-empty experts
        boundaries: list[ExpertBoundary] = []
        for expert_id in range(self.num_experts):
            start = offsets[expert_id].item()
            end = offsets[expert_id + 1].item()
            if end > start:
                boundaries.append(ExpertBoundary(expert_id, int(start), int(end)))

        return boundaries

    def unsort_outputs(
        self,
        outputs: list[tuple[ExpertBoundary, torch.Tensor]],
        state: SortedDispatchState,
    ) -> torch.Tensor:
        """Unsort expert outputs back to original token order.

        Takes outputs from each expert and combines them, then reorders
        to match the original token positions.

        Args:
            outputs: List of (boundary, output_tensor) tuples where:
                - boundary: ExpertBoundary indicating the expert and range
                - output_tensor: [boundary.count, hidden_dim] expert outputs
            state: SortedDispatchState from dispatch() call.

        Returns:
            [batch, hidden_dim] outputs in original token order.
        """
        batch_size = state.original_shape[0]
        device = state.sorted_states.device

        # Infer hidden_dim from first output
        if not outputs:
            return torch.zeros(state.original_shape, device=device, dtype=self.dtype)

        hidden_dim = outputs[0][1].shape[-1]

        # Allocate output buffer in sorted order
        sorted_output = torch.zeros(
            batch_size, hidden_dim, device=device, dtype=self.dtype
        )

        # Place each expert's output at its boundary
        for boundary, expert_out in outputs:
            sorted_output[boundary.start : boundary.end] = expert_out.to(self.dtype)

        # Unsort to original order
        return sorted_output[state.inverse_indices]

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        experts: Sequence[nn.Module] | nn.ModuleList,
    ) -> torch.Tensor:
        """Full sorted dispatch pipeline with weighted combination.

        Convenience method that performs sort -> compute -> unsort in one call.

        Args:
            hidden_states: [batch, hidden_dim] input activations.
            expert_indices: [batch, top_k] expert assignments.
            routing_weights: [batch, top_k] normalized routing weights.
            experts: List or ModuleList of expert modules.

        Returns:
            [batch, hidden_dim] weighted combination of expert outputs.
        """
        state = self.dispatch(hidden_states, expert_indices)

        # Process each expert's tokens as a batch
        outputs: list[tuple[ExpertBoundary, torch.Tensor]] = []
        for boundary in state.boundaries:
            batch = state.sorted_states[boundary.start : boundary.end]
            expert_out = experts[boundary.expert_id](batch)
            outputs.append((boundary, expert_out))

        # Get unweighted outputs in original order
        combined = self.unsort_outputs(outputs, state)

        # For top_k > 1, we need proper weighted combination
        # Currently this only handles primary expert; for full top_k support
        # we'd need to compute all k expert outputs per token
        if self.top_k > 1:
            # Apply primary expert weight
            primary_weights = routing_weights[:, 0:1]
            combined = combined * primary_weights

        return combined


class MultiPassSortedDispatch(nn.Module):
    """Multi-pass sorted dispatch for top_k > 1.

    When tokens route to multiple experts (top_k > 1), this class performs
    k passes, each sorting by the k-th expert assignment. This ensures
    cache-friendly access for all expert computations, not just the primary.

    This is more expensive than single-pass (k times the sorting overhead)
    but provides optimal memory access patterns for all k expert passes.

    Args:
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        dtype: Data type for computations.

    Example:
        >>> dispatch = MultiPassSortedDispatch(num_experts=64, top_k=2)
        >>> output = dispatch(hidden_states, expert_indices, routing_weights, experts)
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        experts: Sequence[nn.Module] | nn.ModuleList,
    ) -> torch.Tensor:
        """Execute k passes with sorted dispatch for each expert slot.

        Args:
            hidden_states: [batch, hidden_dim] input activations.
            expert_indices: [batch, top_k] expert assignments.
            routing_weights: [batch, top_k] normalized routing weights.
            experts: Expert modules.

        Returns:
            [batch, hidden_dim] weighted sum of all k expert outputs.
        """
        device = hidden_states.device
        batch_size, hidden_dim = hidden_states.shape

        # Accumulate weighted outputs
        combined = torch.zeros(batch_size, hidden_dim, device=device, dtype=self.dtype)

        for k in range(self.top_k):
            # Sort by k-th expert assignment
            k_experts = expert_indices[:, k]
            sorted_indices = torch.argsort(k_experts, stable=True)
            inverse_indices = torch.argsort(sorted_indices)

            sorted_states = hidden_states[sorted_indices]
            sorted_experts = k_experts[sorted_indices]

            # Compute boundaries for this pass
            counts = torch.bincount(sorted_experts, minlength=self.num_experts)
            offsets = torch.zeros(
                self.num_experts + 1, device=device, dtype=torch.long
            )
            offsets[1:] = torch.cumsum(counts, dim=0)

            # Process experts
            sorted_output = torch.zeros_like(sorted_states)
            for expert_id in range(self.num_experts):
                start = offsets[expert_id].item()
                end = offsets[expert_id + 1].item()
                if end > start:
                    expert_out = experts[expert_id](sorted_states[start:end])
                    sorted_output[start:end] = expert_out

            # Unsort and weight
            output_k = sorted_output[inverse_indices]
            weight_k = routing_weights[:, k : k + 1]
            combined = combined + output_k * weight_k

        return combined


def sorted_expert_forward(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_fn: Callable[[torch.Tensor, int], torch.Tensor],
    num_experts: int,
) -> torch.Tensor:
    """Functional interface for sorted expert dispatch.

    Standalone function that performs sorted dispatch without maintaining
    module state. Useful for integration with existing code.

    Args:
        hidden_states: [batch, hidden_dim] input activations.
        expert_indices: [batch, top_k] expert assignments.
        routing_weights: [batch, top_k] routing weights.
        expert_fn: Callable (inputs, expert_id) -> outputs.
        num_experts: Total number of experts.

    Returns:
        [batch, hidden_dim] weighted expert outputs.
    """
    device = hidden_states.device
    batch_size, hidden_dim = hidden_states.shape
    top_k = expert_indices.shape[1]

    combined = torch.zeros(batch_size, hidden_dim, device=device, dtype=hidden_states.dtype)

    for k in range(top_k):
        # Sort by k-th expert
        k_experts = expert_indices[:, k]
        sorted_indices = torch.argsort(k_experts, stable=True)
        inverse_indices = torch.argsort(sorted_indices)

        sorted_states = hidden_states[sorted_indices]
        sorted_experts = k_experts[sorted_indices]

        # Compute boundaries
        counts = torch.bincount(sorted_experts, minlength=num_experts)
        offsets = torch.zeros(num_experts + 1, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(counts, dim=0)

        # Process
        sorted_output = torch.zeros_like(sorted_states)
        for expert_id in range(num_experts):
            start = offsets[expert_id].item()
            end = offsets[expert_id + 1].item()
            if end > start:
                sorted_output[start:end] = expert_fn(sorted_states[start:end], expert_id)

        # Unsort and accumulate
        output_k = sorted_output[inverse_indices]
        combined = combined + output_k * routing_weights[:, k : k + 1]

    return combined


__all__ = [
    "ExpertBoundary",
    "MultiPassSortedDispatch",
    "SortedDispatchState",
    "SortedExpertDispatch",
    "sorted_expert_forward",
]
