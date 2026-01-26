"""
Dynamic token-to-expert grouping for batched MoE execution.

Problem: Naive MoE executes each token independently with its assigned experts.
This wastes GPU parallelism since different tokens assigned to the same expert
could share weight loads.

Solution: Group tokens by their assigned expert, batch the GEMM per-expert.
This module provides the CPU-side preparation for moe_expert_gemm kernels.

Workflow:
    1. Router produces expert_ids [batch, top_k] and expert_probs [batch, top_k]
    2. group_tokens_by_expert() reorders tokens to group by expert
    3. moe_expert_gemm() executes batched GEMM per expert
    4. scatter_expert_outputs() restores original token order

Example:
    >>> expert_ids = mx.array([[2, 0], [1, 2], [0, 1]])  # [3 tokens, top_k=2]
    >>> sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts=3)
    >>> # sorted_idx groups token-expert pairs by expert:
    >>> # expert 0: token 0 (2nd choice), token 2 (1st choice)
    >>> # expert 1: token 1 (1st choice), token 2 (2nd choice)
    >>> # expert 2: token 0 (1st choice), token 1 (2nd choice)
    >>> # offsets = [0, 2, 4, 6] (2 assignments each for 3 experts)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class MoEDispatchInfo:
    """Dispatch information for grouped MoE execution.

    This dataclass holds all the indexing tensors needed to:
    1. Reorder tokens by expert for batched GEMM
    2. Apply correct expert probabilities to outputs
    3. Restore original token order after expert computation

    Attributes:
        sorted_token_indices: [total_assignments] indices into original batch.
            Sorted so that token-expert pairs going to the same expert are
            contiguous. Use to gather activations before expert GEMM.
        sorted_expert_indices: [total_assignments] which expert slot (0 to top_k-1)
            each assignment came from. Use to look up expert_probs.
        expert_offsets: [num_experts + 1] start index for each expert's assignments
            in the sorted arrays. expert i's assignments are at indices
            [expert_offsets[i], expert_offsets[i+1]).
        inverse_indices: [total_assignments] indices to scatter expert outputs
            back to original order. After computing expert outputs in sorted
            order, use inverse_indices to restore original token order.
        num_tokens: Original batch size.
        top_k: Number of experts per token.
        num_experts: Total number of experts.
    """

    sorted_token_indices: mx.array  # [total_assignments] int32
    sorted_expert_indices: mx.array  # [total_assignments] int32
    expert_offsets: mx.array  # [num_experts + 1] int32
    inverse_indices: mx.array  # [total_assignments] int32
    num_tokens: int
    top_k: int
    num_experts: int

    @property
    def total_assignments(self) -> int:
        """Total number of token-expert assignments (num_tokens * top_k)."""
        return self.num_tokens * self.top_k


def group_tokens_by_expert(
    expert_ids: mx.array,
    num_experts: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Group tokens by their assigned expert for batched GEMM execution.

    Given expert assignments for each token, produces indexing tensors that
    reorder tokens so that all tokens assigned to the same expert are
    contiguous. This enables efficient batched GEMM per expert instead of
    per-token execution.

    Args:
        expert_ids: [batch, top_k] int32 array where expert_ids[i, j] is the
            j-th expert assigned to token i. Values must be in [0, num_experts).
        num_experts: Total number of experts in the MoE layer.

    Returns:
        Tuple of three arrays:
        - sorted_indices: [batch * top_k] int32 indices to reorder flattened
            token-expert pairs by expert. sorted_indices[i] gives the index
            into the flattened expert_ids for the i-th assignment in
            expert-sorted order.
        - expert_offsets: [num_experts + 1] int32 cumulative counts. Expert e's
            assignments are at sorted_indices[expert_offsets[e]:expert_offsets[e+1]].
        - inverse_indices: [batch * top_k] int32 indices to restore original
            order. For each position i in sorted order, inverse_indices[i]
            gives its original position before sorting.

    Example:
        >>> # 4 tokens, top_k=2, 3 experts
        >>> expert_ids = mx.array([[0, 2], [1, 0], [2, 1], [0, 1]])
        >>> sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, 3)
        >>>
        >>> # Token-expert assignments flattened: [(0,0), (0,2), (1,1), (1,0), (2,2), (2,1), (3,0), (3,1)]
        >>> # After sorting by expert:
        >>> #   Expert 0: positions 0, 3, 6 (tokens 0, 1, 3)
        >>> #   Expert 1: positions 2, 5, 7 (tokens 1, 2, 3)
        >>> #   Expert 2: positions 1, 4    (tokens 0, 2)
        >>> # offsets = [0, 3, 6, 8]
    """
    batch_size, top_k = expert_ids.shape
    total_assignments = batch_size * top_k

    # Flatten expert_ids to [batch * top_k]
    expert_ids_flat = expert_ids.reshape(-1).astype(mx.int32)

    # Create stable sort key: expert_id * batch_size + original_position
    # This ensures tokens for the same expert are grouped, with original
    # order preserved within each expert group
    original_positions = mx.arange(total_assignments, dtype=mx.int32)

    # Compute sort keys - stable sort by expert, preserving token order within expert
    # Using expert_id * total_assignments + position ensures stability
    sort_keys = expert_ids_flat * total_assignments + original_positions

    # Argsort to get indices that would sort by expert
    sorted_indices = mx.argsort(sort_keys)

    # Compute expert_offsets using cumsum on one-hot counts
    # Count how many assignments go to each expert
    expert_counts = mx.zeros((num_experts,), dtype=mx.int32)

    # Use scatter_add pattern: for each expert e, count occurrences
    # Since mx doesn't have scatter_add, we use a different approach:
    # Sort the expert_ids and count changes
    expert_ids_flat[sorted_indices]

    # expert_offsets[e] = number of assignments with expert_id < e
    # We can compute this by finding where each expert starts in sorted order
    #
    # Method: for each expert e, find first position where sorted_experts >= e
    # This is equivalent to cumsum of counts

    # Count occurrences of each expert
    # Use one-hot encoding and sum
    mx.zeros((total_assignments, num_experts), dtype=mx.int32)

    # Scatter 1s at expert positions
    # Since MLX doesn't have scatter, use alternative:
    # Create range [0, num_experts) and compare with each expert_id
    expert_range = mx.arange(num_experts, dtype=mx.int32)
    # [total_assignments, num_experts] comparison matrix
    matches = expert_ids_flat[:, None] == expert_range[None, :]
    expert_counts = mx.sum(matches.astype(mx.int32), axis=0)

    # Cumsum to get offsets (prepend 0)
    expert_offsets = mx.concatenate(
        [mx.array([0], dtype=mx.int32), mx.cumsum(expert_counts)]
    )

    # Compute inverse indices: for each sorted position, where did it come from?
    # inverse_indices[sorted_indices[i]] = i, or equivalently:
    # inverse_indices = argsort(sorted_indices)
    inverse_indices = mx.argsort(sorted_indices)

    return sorted_indices, expert_offsets, inverse_indices


def group_tokens_by_expert_full(
    expert_ids: mx.array,
    num_experts: int,
) -> MoEDispatchInfo:
    """Group tokens by expert with full dispatch information.

    Extended version of group_tokens_by_expert that returns a MoEDispatchInfo
    dataclass with additional information needed for the full MoE forward pass.

    Args:
        expert_ids: [batch, top_k] int32 array of expert assignments.
        num_experts: Total number of experts.

    Returns:
        MoEDispatchInfo with all indexing tensors for dispatch and scatter.
    """
    batch_size, top_k = expert_ids.shape
    batch_size * top_k

    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert(
        expert_ids, num_experts
    )

    # Compute which original token each sorted assignment came from
    # sorted_indices gives flat positions; token_idx = flat_pos // top_k
    sorted_token_indices = sorted_indices // top_k

    # Compute which expert slot (0 to top_k-1) each sorted assignment came from
    # expert_slot = flat_pos % top_k
    sorted_expert_indices = sorted_indices % top_k

    return MoEDispatchInfo(
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        expert_offsets=expert_offsets,
        inverse_indices=inverse_indices,
        num_tokens=batch_size,
        top_k=top_k,
        num_experts=num_experts,
    )


def gather_for_experts(
    activations: mx.array,
    dispatch_info: MoEDispatchInfo,
) -> mx.array:
    """Gather activations in expert-sorted order for batched GEMM.

    Reorders activations so that all tokens going to the same expert are
    contiguous. This is the input preparation step before moe_expert_gemm.

    Args:
        activations: [batch, hidden_dim] input activations.
        dispatch_info: Dispatch info from group_tokens_by_expert_full.

    Returns:
        [total_assignments, hidden_dim] activations in expert-sorted order.
        Tokens for expert e are at rows [offsets[e]:offsets[e+1]].
    """
    # Gather using sorted_token_indices
    # Each assignment gets a copy of its token's activation
    return activations[dispatch_info.sorted_token_indices]


def scatter_expert_outputs(
    expert_outputs: mx.array,
    expert_probs: mx.array,
    dispatch_info: MoEDispatchInfo,
) -> mx.array:
    """Scatter and combine expert outputs back to original token order.

    After running batched expert GEMM, this function:
    1. Weights each expert output by its routing probability
    2. Scatters outputs back to original token positions
    3. Sums contributions from multiple experts per token

    Args:
        expert_outputs: [total_assignments, out_dim] outputs from experts in
            sorted order (as produced by moe_expert_gemm).
        expert_probs: [batch, top_k] routing probabilities from router.
        dispatch_info: Dispatch info from group_tokens_by_expert_full.

    Returns:
        [batch, out_dim] combined outputs with original token order.
    """
    batch_size = dispatch_info.num_tokens
    top_k = dispatch_info.top_k
    out_dim = expert_outputs.shape[1]

    # Get probabilities for each sorted assignment
    # sorted_token_indices[i] = which token sorted position i corresponds to
    # sorted_expert_indices[i] = which expert slot (0 to top_k-1) for sorted position i
    probs_for_sorted = expert_probs[
        dispatch_info.sorted_token_indices, dispatch_info.sorted_expert_indices
    ]

    # Weight outputs by their routing probabilities
    weighted_outputs = expert_outputs * probs_for_sorted[:, None]

    # Reorder from sorted order to original flat order [batch * top_k]
    # inverse_indices[i] = position in sorted array that maps to original position i
    # So to go from sorted to original, we gather using inverse_indices:
    # original[i] = sorted[inverse_indices[i]]
    weighted_original = weighted_outputs[dispatch_info.inverse_indices]

    # Now reshape to [batch, top_k, out_dim] and sum over top_k dimension
    # The original flat order is [token0_expert0, token0_expert1, token1_expert0, ...]
    weighted_reshaped = weighted_original.reshape(batch_size, top_k, out_dim)
    output = mx.sum(weighted_reshaped, axis=1)

    return output


def compute_expert_load(
    expert_ids: mx.array,
    num_experts: int,
) -> mx.array:
    """Compute load (number of assigned tokens) per expert.

    Useful for load balancing analysis and auxiliary losses.

    Args:
        expert_ids: [batch, top_k] expert assignments.
        num_experts: Total number of experts.

    Returns:
        [num_experts] int32 array of token counts per expert.
    """
    expert_ids_flat = expert_ids.reshape(-1).astype(mx.int32)

    # Count occurrences using one-hot sum
    expert_range = mx.arange(num_experts, dtype=mx.int32)
    matches = expert_ids_flat[:, None] == expert_range[None, :]
    expert_counts = mx.sum(matches.astype(mx.int32), axis=0)

    return expert_counts


def compute_load_balancing_loss(
    expert_probs_pre_topk: mx.array,
    expert_ids: mx.array,
    num_experts: int,
) -> mx.array:
    """Compute auxiliary load balancing loss for MoE training.

    Uses the formulation from Switch Transformer:
    L_balance = num_experts * sum_e(f_e * P_e)

    Where:
    - f_e = fraction of tokens routed to expert e
    - P_e = average routing probability to expert e (before top-k selection)

    Args:
        expert_probs_pre_topk: [batch, num_experts] router probabilities
            before top-k selection (i.e., softmax output).
        expert_ids: [batch, top_k] selected expert indices.
        num_experts: Total number of experts.

    Returns:
        Scalar load balancing loss.
    """
    expert_ids.shape[0]

    # f_e: fraction of tokens routed to each expert
    expert_counts = compute_expert_load(expert_ids, num_experts).astype(mx.float32)
    # Normalize by total assignments
    total_assignments = expert_ids.size
    f = expert_counts / total_assignments

    # P_e: average probability assigned to each expert across all tokens
    P = mx.mean(expert_probs_pre_topk, axis=0)  # [num_experts]

    # Loss = num_experts * dot(f, P)
    loss = num_experts * mx.sum(f * P)

    return loss
