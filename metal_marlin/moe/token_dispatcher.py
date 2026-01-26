"""Efficient token batching dispatcher for MoE inference.

This module provides optimized token-to-expert dispatch that achieves 2-4x speedup
over naive per-token execution through:

1. Token grouping by expert (batch tokens going to same expert together)
2. Batched expert forward passes (one GEMM per expert instead of per-token)
3. Efficient output combination with routing weight application

Architecture:
    Naive MoE: For each token, load top-k experts, compute, combine.
               Memory: O(num_tokens * top_k * expert_size) loads

    Optimized: Group tokens by expert, batch compute per expert.
               Memory: O(num_experts_active * expert_size) loads

    The key insight is that when many tokens route to the same expert,
    we can share the expert weight loads across all those tokens.

Example:
    >>> dispatcher = TokenDispatcher(
    ...     num_experts=64,
    ...     hidden_dim=4096,
    ...     intermediate_dim=14336,
    ... )
    >>>
    >>> # Router produces top-k assignments
    >>> expert_ids, expert_probs = router(hidden_states)
    >>>
    >>> # Batched dispatch
    >>> output = dispatcher.dispatch(
    ...     hidden_states,
    ...     expert_ids,
    ...     expert_probs,
    ...     expert_weights,
    ... )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from .._compat import HAS_MPS

# Default device for MoE dispatch operations
_DEFAULT_DEVICE: str = "mps" if HAS_MPS else "cpu"


class ExpertForward(Protocol):
    """Protocol for expert forward pass callable."""

    def __call__(
        self,
        activations: torch.Tensor,
        expert_id: int,
    ) -> torch.Tensor:
        """Forward pass for a single expert.

        Args:
            activations: [batch, hidden_dim] input activations
            expert_id: Expert index

        Returns:
            [batch, out_dim] expert outputs
        """
        ...


@dataclass(frozen=True)
class DispatchInfo:
    """Dispatch information for grouped MoE execution.

    Contains all indexing tensors needed for efficient batched dispatch:
    - Token reordering for per-expert batching
    - Probability lookup for output weighting
    - Inverse mapping for output reassembly

    Attributes:
        sorted_token_indices: [total_assignments] token indices in expert-sorted order
        sorted_expert_slots: [total_assignments] which top-k slot (0 to k-1) each comes from
        expert_offsets: [num_experts + 1] cumulative token counts per expert
        inverse_indices: [total_assignments] mapping from sorted back to original order
        num_tokens: Original batch size
        top_k: Number of experts per token
        num_experts: Total number of experts
    """

    sorted_token_indices: torch.Tensor
    sorted_expert_slots: torch.Tensor
    expert_offsets: torch.Tensor
    inverse_indices: torch.Tensor
    num_tokens: int
    top_k: int
    num_experts: int

    @property
    def total_assignments(self) -> int:
        """Total token-expert assignments (num_tokens * top_k)."""
        return self.num_tokens * self.top_k

    def expert_batch_size(self, expert_id: int) -> int:
        """Number of tokens assigned to a specific expert."""
        start = int(self.expert_offsets[expert_id].item())
        end = int(self.expert_offsets[expert_id + 1].item())
        return end - start


@dataclass
class DispatchStats:
    """Statistics from a dispatch operation for profiling."""

    num_tokens: int
    top_k: int
    num_experts: int
    active_experts: int  # How many experts received at least one token
    max_expert_load: int  # Maximum tokens to any single expert
    min_expert_load: int  # Minimum tokens to any single expert (excluding zeros)
    load_imbalance: float  # std/mean of non-zero loads


def group_tokens_by_expert(
    expert_ids_or_tokens: torch.Tensor,
    num_experts_or_routing_weights: int | tuple[torch.Tensor, ...] | dict[str, torch.Tensor],
) -> DispatchInfo | tuple[torch.Tensor, DispatchInfo]:
    """Group tokens by their assigned expert for batched execution.

    Given expert assignments [batch, top_k], produces indexing tensors that
    reorder tokens so all tokens assigned to the same expert are contiguous.
    This enables batched GEMM per expert instead of per-token execution.

    The algorithm:
    1. Flatten expert_ids to [batch * top_k]
    2. Create stable sort keys: expert_id * total + position
    3. Argsort to get sorted indices
    4. Compute expert offsets via cumulative counts
    5. Compute inverse mapping for scatter back

    Args:
        expert_ids_or_tokens: Either [batch, top_k] int32 expert assignments,
            or [batch, hidden_dim] activations when using routing_weights.
        num_experts_or_routing_weights: Either total number of experts, or a
            routing payload (tuple or dict) containing expert_ids (+ optional
            num_experts). When routing payload is provided, this returns the
            grouped activations and dispatch info.

    Returns:
        DispatchInfo when called with (expert_ids, num_experts).
        If called with (tokens, routing_weights), returns (grouped_tokens, DispatchInfo).

    Example:
        >>> expert_ids = torch.tensor([[2, 0], [1, 2], [0, 1]], device="mps")  # 3 tokens, top_k=2
        >>> info = group_tokens_by_expert(expert_ids, num_experts=3)
        >>> # Tokens are now grouped:
        >>> # Expert 0: token 0 (slot 1), token 2 (slot 0)
        >>> # Expert 1: token 1 (slot 0), token 2 (slot 1)
        >>> # Expert 2: token 0 (slot 0), token 1 (slot 1)
    """
    if isinstance(num_experts_or_routing_weights, int):
        expert_ids = expert_ids_or_tokens
        num_experts = num_experts_or_routing_weights
        return _group_tokens_by_expert_ids(expert_ids, num_experts)

    tokens = expert_ids_or_tokens
    expert_ids, num_experts = _extract_expert_ids(num_experts_or_routing_weights)
    dispatch_info = _group_tokens_by_expert_ids(expert_ids, num_experts)
    grouped_tokens = tokens[dispatch_info.sorted_token_indices]
    return grouped_tokens, dispatch_info


def _group_tokens_by_expert_ids(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> DispatchInfo:
    """Implementation for expert-id grouping used by all call paths."""
    device = expert_ids.device
    batch_size, top_k = expert_ids.shape
    total_assignments = batch_size * top_k

    # Flatten to [batch * top_k]
    expert_ids_flat = expert_ids.reshape(-1).to(torch.int64)

    # Create stable sort key: expert_id * total + original_position
    # This ensures stable sorting within each expert group
    positions = torch.arange(total_assignments, dtype=torch.int64, device=device)
    sort_keys = expert_ids_flat * total_assignments + positions

    # Argsort for expert-grouped order
    sorted_indices = torch.argsort(sort_keys)

    # Compute which token and slot each sorted position came from
    sorted_token_indices = sorted_indices // top_k
    sorted_expert_slots = sorted_indices % top_k

    # Compute expert offsets using bincount (more efficient than one-hot)
    expert_counts = torch.bincount(expert_ids_flat, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # Inverse mapping: argsort of sorted_indices
    inverse_indices = torch.argsort(sorted_indices)

    return DispatchInfo(
        sorted_token_indices=sorted_token_indices,
        sorted_expert_slots=sorted_expert_slots,
        expert_offsets=expert_offsets,
        inverse_indices=inverse_indices,
        num_tokens=batch_size,
        top_k=top_k,
        num_experts=num_experts,
    )


def _extract_expert_ids(
    routing_weights: tuple[torch.Tensor, ...] | dict[str, torch.Tensor] | torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Extract expert_ids and infer num_experts from routing payload."""
    if isinstance(routing_weights, dict):
        expert_ids = routing_weights.get("expert_ids") or routing_weights.get("indices")
        if expert_ids is None:
            raise ValueError("routing_weights dict missing 'expert_ids' or 'indices'")
        num_experts = routing_weights.get("num_experts")
    elif isinstance(routing_weights, (tuple, list)):
        if not routing_weights:
            raise ValueError("routing_weights tuple is empty")
        expert_ids = routing_weights[0]
        num_experts = routing_weights[2] if len(routing_weights) > 2 else None
    else:
        expert_ids = routing_weights
        num_experts = None

    if num_experts is None:
        num_experts = int(expert_ids.max().item()) + 1

    return expert_ids, int(num_experts)


def gather_tokens_for_expert(
    activations: torch.Tensor,
    dispatch_info: DispatchInfo,
    expert_id: int,
) -> torch.Tensor:
    """Gather activations for a specific expert's assigned tokens.

    Args:
        activations: [batch, hidden_dim] input activations
        dispatch_info: Dispatch info from group_tokens_by_expert
        expert_id: Expert index to gather tokens for

    Returns:
        [num_assigned, hidden_dim] activations for this expert's tokens
    """
    start = int(dispatch_info.expert_offsets[expert_id].item())
    end = int(dispatch_info.expert_offsets[expert_id + 1].item())

    if start == end:
        # No tokens assigned to this expert
        return torch.zeros(
            (0, activations.shape[1]),
            dtype=activations.dtype,
            device=activations.device,
        )

    # Get token indices for this expert
    token_indices = dispatch_info.sorted_token_indices[start:end]

    return activations[token_indices]


def dispatch_to_experts(
    activations_or_grouped: torch.Tensor | tuple[torch.Tensor, DispatchInfo],
    dispatch_info_or_experts: DispatchInfo | Sequence[ExpertForward],
    expert_forward: ExpertForward | None = None,
) -> torch.Tensor:
    """Execute batched forward pass through all active experts.

    For each expert that received at least one token:
    1. Gather that expert's assigned tokens
    2. Run the expert's forward pass on the batch
    3. Store outputs in sorted order

    Args:
        activations_or_grouped: Either [batch, hidden_dim] activations, or
            (grouped_tokens, dispatch_info) from group_tokens_by_expert.
        dispatch_info_or_experts: Either DispatchInfo, or sequence of expert
            callables indexed by expert_id.
        expert_forward: Callable taking (activations, expert_id) -> outputs when
            using the DispatchInfo path.

    Returns:
        [total_assignments, out_dim] expert outputs in sorted order
    """
    if isinstance(dispatch_info_or_experts, DispatchInfo):
        if expert_forward is None:
            raise ValueError("expert_forward is required when passing DispatchInfo")
        dispatch_info = dispatch_info_or_experts
        activations = activations_or_grouped
        num_experts = dispatch_info.num_experts

        # Collect outputs from each expert in order (already sorted by expert)
        output_chunks: list[torch.Tensor] = []

        for expert_id in range(num_experts):
            start = int(dispatch_info.expert_offsets[expert_id].item())
            end = int(dispatch_info.expert_offsets[expert_id + 1].item())

            if start == end:
                continue  # No tokens for this expert

            # Gather tokens for this expert
            expert_tokens = gather_tokens_for_expert(
                activations,
                dispatch_info,
                expert_id,
            )

            # Forward pass
            expert_output = expert_forward(expert_tokens, expert_id)
            output_chunks.append(expert_output)

        if not output_chunks:
            raise ValueError("No tokens assigned to any expert")

        # Concatenate all outputs (already in sorted order)
        return torch.cat(output_chunks, dim=0)

    if expert_forward is not None:
        raise ValueError("expert_forward must be None when passing experts list")

    if not isinstance(activations_or_grouped, tuple) or len(activations_or_grouped) != 2:
        raise ValueError("grouped_tokens must be (tokens, dispatch_info)")

    grouped_tokens, dispatch_info = activations_or_grouped
    experts = dispatch_info_or_experts
    num_experts = dispatch_info.num_experts

    if len(experts) < num_experts:
        raise ValueError("experts list shorter than num_experts")

    output_chunks = []
    for expert_id in range(num_experts):
        start = int(dispatch_info.expert_offsets[expert_id].item())
        end = int(dispatch_info.expert_offsets[expert_id + 1].item())
        if start == end:
            continue

        expert_tokens = grouped_tokens[start:end]
        expert_output = _call_expert(experts[expert_id], expert_tokens, expert_id)
        output_chunks.append(expert_output)

    if not output_chunks:
        raise ValueError("No tokens assigned to any expert")

    return torch.cat(output_chunks, dim=0)


def combine_expert_outputs(
    expert_outputs: torch.Tensor,
    expert_probs_or_routing_weights: torch.Tensor
    | tuple[torch.Tensor, DispatchInfo]
    | dict[str, torch.Tensor],
    dispatch_info: DispatchInfo | None = None,
) -> torch.Tensor:
    """Combine weighted expert outputs back to original token order.

    For each token:
    1. Look up its top-k expert outputs (in sorted order)
    2. Weight each by the routing probability
    3. Sum weighted contributions

    Args:
        expert_outputs: [total_assignments, out_dim] outputs in sorted order.
        expert_probs_or_routing_weights: Either [batch, top_k] routing probs,
            or a tuple/dict containing (expert_probs, dispatch_info).
        dispatch_info: Dispatch info when expert_probs is passed separately.

    Returns:
        [batch, out_dim] combined outputs in original token order
    """
    if dispatch_info is None:
        expert_probs, dispatch_info = _extract_probs_and_info(expert_probs_or_routing_weights)
    else:
        expert_probs = expert_probs_or_routing_weights

    batch_size = dispatch_info.num_tokens
    top_k = dispatch_info.top_k
    out_dim = expert_outputs.shape[1]

    # Get probabilities for each sorted assignment
    # sorted_token_indices[i] = which token, sorted_expert_slots[i] = which slot
    probs_for_sorted = expert_probs[
        dispatch_info.sorted_token_indices,
        dispatch_info.sorted_expert_slots,
    ]

    # Weight outputs by routing probabilities
    weighted_outputs = expert_outputs * probs_for_sorted.unsqueeze(1)

    # Reorder from sorted to original flat order using inverse_indices
    weighted_original = weighted_outputs[dispatch_info.inverse_indices]

    # Reshape to [batch, top_k, out_dim] and sum over top_k
    weighted_reshaped = weighted_original.reshape(batch_size, top_k, out_dim)
    combined = weighted_reshaped.sum(dim=1)

    return combined


def _extract_probs_and_info(
    routing_weights: tuple[torch.Tensor, DispatchInfo] | dict[str, torch.Tensor],
) -> tuple[torch.Tensor, DispatchInfo]:
    """Extract expert_probs and dispatch_info from routing payload."""
    if isinstance(routing_weights, dict):
        expert_probs = routing_weights.get("expert_probs") or routing_weights.get("probs")
        dispatch_info = routing_weights.get("dispatch_info")
    elif isinstance(routing_weights, (tuple, list)):
        if len(routing_weights) < 2:
            raise ValueError("routing_weights tuple must include probs and dispatch_info")
        expert_probs = routing_weights[0]
        dispatch_info = routing_weights[1]
    else:
        raise ValueError("routing_weights must include probs and dispatch_info")

    if expert_probs is None or dispatch_info is None:
        raise ValueError("routing_weights missing expert_probs or dispatch_info")

    if not isinstance(dispatch_info, DispatchInfo):
        raise ValueError("dispatch_info must be a DispatchInfo instance")

    return expert_probs, dispatch_info


def _call_expert(
    expert_fn: ExpertForward, activations: torch.Tensor, expert_id: int
) -> torch.Tensor:
    """Call an expert with (activations, expert_id) or (activations)."""
    try:
        return expert_fn(activations, expert_id)
    except TypeError:
        return expert_fn(activations)


class TokenDispatcher:
    """High-level token dispatcher for MoE inference.

    Provides a simple interface for efficient MoE forward passes with automatic
    token grouping and output combination.

    Args:
        num_experts: Total number of experts
        hidden_dim: Input hidden dimension
        intermediate_dim: Expert intermediate dimension (for FFN)
        top_k: Number of experts per token (default 2)
        enable_stats: Whether to track dispatch statistics
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int = 2,
        enable_stats: bool = False,
    ):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.top_k = top_k
        self.enable_stats = enable_stats

        self._last_dispatch_info: DispatchInfo | None = None
        self._last_stats: DispatchStats | None = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        expert_forward: ExpertForward,
    ) -> torch.Tensor:
        """Execute MoE forward pass with efficient token batching.

        Args:
            hidden_states: [batch, hidden_dim] input activations
            expert_ids: [batch, top_k] expert assignments from router
            expert_probs: [batch, top_k] routing probabilities (should sum to 1)
            expert_forward: Function (activations, expert_id) -> outputs

        Returns:
            [batch, out_dim] combined expert outputs
        """
        # Step 1: Group tokens by expert
        dispatch_info = group_tokens_by_expert(expert_ids, self.num_experts)
        self._last_dispatch_info = dispatch_info

        # Step 2: Dispatch to experts
        expert_outputs = dispatch_to_experts(
            hidden_states,
            dispatch_info,
            expert_forward,
        )

        # Step 3: Combine outputs
        combined = combine_expert_outputs(expert_outputs, expert_probs, dispatch_info)

        # Optionally collect stats
        if self.enable_stats:
            self._collect_stats(dispatch_info)

        return combined

    def dispatch_with_shared_expert(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        expert_forward: ExpertForward,
        shared_expert_forward: ExpertForward,
        shared_expert_weight: float = 1.0,
    ) -> torch.Tensor:
        """Execute MoE forward pass with a shared expert.

        Some architectures (DeepSeek-V2, Qwen-MoE) have a "shared" expert that
        processes all tokens regardless of routing. This method handles both
        routed and shared experts efficiently.

        Args:
            hidden_states: [batch, hidden_dim] input activations
            expert_ids: [batch, top_k] expert assignments for routed experts
            expert_probs: [batch, top_k] routing probabilities
            expert_forward: Forward function for routed experts
            shared_expert_forward: Forward function for shared expert
            shared_expert_weight: Weight for shared expert output (default 1.0)

        Returns:
            [batch, out_dim] combined routed + shared expert outputs
        """
        # Routed experts
        routed_output = self.dispatch(
            hidden_states,
            expert_ids,
            expert_probs,
            expert_forward,
        )

        # Shared expert (runs on all tokens)
        shared_output = shared_expert_forward(hidden_states, -1)  # -1 signals shared

        # Combine
        return routed_output + shared_expert_weight * shared_output

    def _collect_stats(self, dispatch_info: DispatchInfo) -> None:
        """Collect dispatch statistics."""
        offsets = dispatch_info.expert_offsets.cpu().numpy()
        loads = np.diff(offsets)

        active_experts = int(np.sum(loads > 0))
        nonzero_loads = loads[loads > 0]

        self._last_stats = DispatchStats(
            num_tokens=dispatch_info.num_tokens,
            top_k=dispatch_info.top_k,
            num_experts=dispatch_info.num_experts,
            active_experts=active_experts,
            max_expert_load=int(np.max(loads)) if len(loads) > 0 else 0,
            min_expert_load=int(np.min(nonzero_loads)) if len(nonzero_loads) > 0 else 0,
            load_imbalance=float(np.std(nonzero_loads) / np.mean(nonzero_loads))
            if len(nonzero_loads) > 1
            else 0.0,
        )

    @property
    def last_dispatch_info(self) -> DispatchInfo | None:
        """Get dispatch info from the last dispatch call."""
        return self._last_dispatch_info

    @property
    def last_stats(self) -> DispatchStats | None:
        """Get statistics from the last dispatch call."""
        return self._last_stats


def compute_expert_load(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Compute load (token count) per expert.

    Args:
        expert_ids: [batch, top_k] expert assignments
        num_experts: Total number of experts

    Returns:
        [num_experts] int64 token counts per expert
    """
    flat_ids = expert_ids.reshape(-1).to(torch.int64)
    return torch.bincount(flat_ids, minlength=num_experts)


def compute_load_balancing_loss(
    router_probs: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute auxiliary load balancing loss for MoE training.

    Uses the Switch Transformer formulation:
        L = num_experts * sum_e(f_e * P_e)

    where f_e is fraction of tokens routed to expert e, and P_e is
    average probability assigned to expert e across all tokens.

    Args:
        router_probs: [batch, num_experts] pre-topk router probabilities
        expert_ids: [batch, top_k] selected expert indices
        num_experts: Total number of experts

    Returns:
        Scalar load balancing loss
    """
    # f_e: fraction of tokens routed to each expert
    expert_counts = compute_expert_load(expert_ids, num_experts).to(torch.float32)
    total_assignments = float(expert_ids.numel())
    f = expert_counts / total_assignments

    # P_e: average probability per expert
    P = router_probs.mean(dim=0)

    # Loss = num_experts * dot(f, P)
    return num_experts * (f * P).sum()
