"""Metal kernel dispatch for MoE token grouping and scattering.

This module provides Metal-accelerated versions of the MoE dispatch operations
from moe_dispatch.py. It uses Metal compute kernels for:
    - Computing expert counts (histogram)
    - Computing expert offsets (prefix sum)
    - Computing sorted indices (scatter)
    - Gathering activations for experts
    - Scattering and combining expert outputs
    - Asynchronous parallel expert execution
    - Sparse-expert path for tokens with single-expert activation

Usage:
    from metal_marlin.moe_dispatch_metal import group_tokens_by_expert_metal, AsyncExpertDispatcher

    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert_metal(
        expert_ids, num_experts
    )

    # For parallel expert execution
    async_dispatcher = AsyncExpertDispatcher(get_default_library())
    async_dispatcher.encode_expert_dispatch(...)
    async_dispatcher.encode_expert_dispatch(...)
    async_dispatcher.commit_and_wait()

    # For sparse-expert path (tokens with only one active expert)
    sparse_mask = detect_sparse_expert_tokens(expert_probs, threshold=0.99)
    dispatch_info = create_sparse_expert_dispatch(expert_ids, expert_probs, sparse_mask)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from ._compat import Metal
from .metal_dispatch import (
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_metal,
    require_mps,
)
from .moe_dispatch import MoEDispatchInfo, group_tokens_by_expert

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Asynchronous Command Buffer Management for Parallel Expert Execution
# ---------------------------------------------------------------------------


class AsyncExpertDispatcher:
    """Manager for asynchronous command buffer submission for parallel expert execution.

    This class enables dispatching multiple expert kernels in parallel by:
    1. Encoding multiple kernel dispatches into separate command buffers
    2. Committing all command buffers simultaneously
    3. Waiting for all expert work to complete asynchronously

    This is particularly useful for MoE layers where different experts process
    independent token batches. By using separate command buffers, the GPU can
    schedule expert kernels more efficiently without CPU synchronization overhead.

    Usage:
        lib = get_default_library()
        async_dispatcher = AsyncExpertDispatcher(lib)

        # Encode dispatch for expert 0
        async_dispatcher.begin_expert_command()
        async_dispatcher.dispatch_expert(
            kernel="moe_expert_forward",
            grid=(num_tgs, 1, 1),
            threadgroup=(128, 1, 1),
            buffers=[activation_buf, weight_buf, output_buf, ...],
        )

        # Encode dispatch for expert 1
        async_dispatcher.begin_expert_command()
        async_dispatcher.dispatch_expert(
            kernel="moe_expert_forward",
            grid=(num_tgs, 1, 1),
            threadgroup=(128, 1, 1),
            buffers=[activation_buf, weight_buf, output_buf, ...],
        )

        # Submit all commands and wait for completion
        async_dispatcher.commit_all()
        async_dispatcher.wait_all()
    """

    __slots__ = (
        "_lib",
        "_command_buffers",
        "_encoders",
        "_pipelines",
        "_committed",
    )

    def __init__(self, lib: Any) -> None:
        """Initialize async expert dispatcher.

        Args:
            lib: MetalKernelLibrary instance for accessing Metal device and pipelines.
        """
        self._lib = lib
        self._command_buffers: list[Any] = []
        self._encoders: list[Any] = []
        self._pipelines: dict[str, Any] = {}
        self._committed = False

    def _get_pipeline(self, kernel_name: str) -> Any:
        """Get or create a compute pipeline state for a kernel.

        Args:
            kernel_name: Name of the Metal kernel function.

        Returns:
            MTLComputePipelineState for the kernel.
        """
        if kernel_name not in self._pipelines:
            self._pipelines[kernel_name] = self._lib.get_pipeline(kernel_name)
        return self._pipelines[kernel_name]

    def begin_expert_command(self) -> None:
        """Begin encoding a new command buffer for an expert.

        Creates a new command buffer and compute encoder. Call dispatch_expert()
        to add kernel dispatches, then repeat for additional experts.

        Must call commit_all() followed by wait_all() to execute.
        """
        if self._committed:
            raise RuntimeError(
                "Commands already committed. Create a new AsyncExpertDispatcher or "
                "use reset() to start over."
            )

        command_buffer = self._lib.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        self._command_buffers.append(command_buffer)
        self._encoders.append(encoder)

    def dispatch_expert(
        self,
        kernel_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: Sequence[Any],
    ) -> None:
        """Dispatch a kernel for the current expert command buffer.

        Args:
            kernel_name: Name of the Metal kernel function.
            grid: Threadgroup grid dimensions (x, y, z).
            threadgroup: Threads per threadgroup (x, y, z).
            buffers: Metal buffers to bind as kernel arguments.

        Raises:
            RuntimeError: If no command buffer is currently being encoded.
        """
        if not self._encoders:
            raise RuntimeError(
                "No active command buffer. Call begin_expert_command() first."
            )

        encoder = self._encoders[-1]
        pipeline = self._get_pipeline(kernel_name)

        encoder.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid),
            Metal.MTLSizeMake(*threadgroup),
        )

    def commit_all(self) -> None:
        """Commit all pending command buffers for asynchronous execution.

        After calling this method, the GPU begins executing all encoded kernels
        concurrently. Call wait_all() to block until all work completes.

        Once committed, no more dispatches can be added. Reset with reset() or
        create a new AsyncExpertDispatcher instance.
        """
        if self._committed:
            raise RuntimeError("Commands already committed.")

        for encoder, command_buffer in zip(self._encoders, self._command_buffers):
            encoder.endEncoding()
            command_buffer.commit()

        self._committed = True

    def wait_all(self) -> None:
        """Block until all committed command buffers complete execution.

        Must be called after commit_all().
        """
        if not self._committed:
            raise RuntimeError(
                "Cannot wait: commands not committed. Call commit_all() first."
            )

        for command_buffer in self._command_buffers:
            command_buffer.waitUntilCompleted()

    def reset(self) -> None:
        """Reset the dispatcher for a new batch of expert dispatches.

        Clears all internal state, allowing the same dispatcher to be reused.
        """
        self._command_buffers.clear()
        self._encoders.clear()
        self._pipelines.clear()
        self._committed = False

    def __enter__(self) -> AsyncExpertDispatcher:
        """Context manager entry: reset and return self for fluent usage."""
        self.reset()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: automatically commit and wait if no exception."""
        if exc_type is None and self._encoders:
            self.commit_all()
            self.wait_all()
        self.reset()


def dispatch_experts_parallel(
    lib: Any,
    dispatches: Sequence[tuple[str, tuple[int, int, int], tuple[int, int, int], Sequence[Any]]],
) -> None:
    """Dispatch multiple expert kernels in parallel using a single batched submission.

    This is a convenience function that creates an AsyncExpertDispatcher,
    encodes all dispatches, commits them simultaneously, and waits for completion.

    Args:
        lib: MetalKernelLibrary instance.
        dispatches: Sequence of (kernel_name, grid, threadgroup, buffers) tuples.
            Each tuple defines one expert kernel dispatch.

    Example:
        dispatches = [
            ("expert_0_forward", (16, 1, 1), (128, 1, 1), [buf0, w0, out0]),
            ("expert_1_forward", (16, 1, 1), (128, 1, 1), [buf1, w1, out1]),
            ("expert_2_forward", (8, 1, 1), (128, 1, 1), [buf2, w2, out2]),
        ]
        dispatch_experts_parallel(lib, dispatches)
    """
    dispatcher = AsyncExpertDispatcher(lib)

    for kernel_name, grid, threadgroup, buffers in dispatches:
        dispatcher.begin_expert_command()
        dispatcher.dispatch_expert(kernel_name, grid, threadgroup, buffers)

    dispatcher.commit_all()
    dispatcher.wait_all()


# ---------------------------------------------------------------------------
# Token Grouping (Metal-accelerated)
# ---------------------------------------------------------------------------


def group_tokens_by_expert_sparse(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group tokens by expert for sparse case (top_k=1).

    Optimized path for tokens assigned to exactly one expert each. This is
    significantly faster than the general case because:
    - No need for complex index sorting
    - Direct mapping from token to expert
    - Simpler expert count computation

    Args:
        expert_ids: [batch, 1] or [batch] int tensor where expert_ids[i] is the
            expert assigned to token i. Values must be in [0, num_experts).
        num_experts: Total number of experts in the MoE layer.

    Returns:
        Tuple of three tensors:
        - sorted_indices: [batch] int64 indices (identity for sparse case)
        - expert_offsets: [num_experts + 1] int64 cumulative counts
        - inverse_indices: [batch] int64 indices (identity for sparse case)
    """
    require_mps()

    # Ensure expert_ids is [batch] shape and contiguous
    if expert_ids.dim() == 2:
        expert_ids = expert_ids.squeeze(1)
    expert_ids = expert_ids.contiguous()

    device = expert_ids.device
    batch_size = expert_ids.shape[0]

    # For sparse case, sorted_indices is just 0..batch_size-1
    # (no reordering needed)
    sorted_indices = torch.arange(batch_size, dtype=torch.int64, device=device)

    # Compute expert counts using bincount (much faster than atomic kernels)
    expert_counts = torch.bincount(expert_ids, minlength=num_experts)

    # Cumsum to get offsets
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # For sparse case, inverse_indices is also identity
    inverse_indices = sorted_indices.clone()

    return sorted_indices, expert_offsets, inverse_indices


def group_tokens_by_expert_metal(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group tokens by expert using Metal kernels.

    Same API as moe_dispatch.group_tokens_by_expert(). Uses Metal compute
    kernels for counting, prefix sum, and index scattering.

    Args:
        expert_ids: [batch, top_k] int tensor where expert_ids[i, j] is the
            j-th expert assigned to token i. Values must be in [0, num_experts).
        num_experts: Total number of experts in the MoE layer.

    Returns:
        Tuple of three tensors:
        - sorted_indices: [batch * top_k] int64 indices that group by expert
        - expert_offsets: [num_experts + 1] int64 cumulative counts
        - inverse_indices: [batch * top_k] int64 indices to restore original order
    """
    require_metal()
    require_mps()

    lib = get_default_library()
    device = lib.device

    batch_size, top_k = expert_ids.shape
    total_assignments = batch_size * top_k

    # Ensure expert_ids is on MPS and contiguous
    expert_ids_mps = expert_ids.to(device="mps", dtype=torch.int32).contiguous()

    # Allocate outputs
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device="mps")
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device="mps")
    sorted_indices = torch.empty(total_assignments, dtype=torch.int32, device="mps")

    # Step 1: Compute expert counts using moe_compute_grouping kernel
    # This kernel atomically increments counters for each expert
    expert_ids_buf = mps_tensor_to_metal_buffer(expert_ids_mps, device)
    sorted_indices_buf = mps_tensor_to_metal_buffer(sorted_indices, device, copy_back=True)
    expert_counts_buf = mps_tensor_to_metal_buffer(expert_counts, device, copy_back=True)
    expert_offsets_buf = mps_tensor_to_metal_buffer(expert_offsets, device, copy_back=True)

    # Create constant buffers
    import numpy as np

    batch_size_buf = device.newBufferWithBytes_length_options_(
        np.array([batch_size], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    top_k_buf = device.newBufferWithBytes_length_options_(
        np.array([top_k], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    num_experts_buf = device.newBufferWithBytes_length_options_(
        np.array([num_experts], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    # Dispatch moe_compute_grouping kernel
    # Grid: 1D with enough threads to cover all assignments
    threads_per_tg = 128
    num_threadgroups = (total_assignments + threads_per_tg - 1) // threads_per_tg

    dispatch_kernel(
        lib,
        function_name="moe_compute_grouping",
        grid=(num_threadgroups, 1, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            expert_ids_buf,
            sorted_indices_buf,
            expert_counts_buf,
            expert_offsets_buf,
            batch_size_buf,
            top_k_buf,
            num_experts_buf,
        ],
        wait=True,
    )

    # Step 2: Compute expert offsets (prefix sum) using moe_compute_offsets kernel
    dispatch_kernel(
        lib,
        function_name="moe_compute_offsets",
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        buffers=[
            expert_counts_buf,
            expert_offsets_buf,
            num_experts_buf,
        ],
        wait=True,
    )

    # Step 3: Reset expert_counts to use as write offsets
    # Copy expert_offsets[0:num_experts] to expert_counts for atomic scatter
    expert_offsets_cpu = expert_offsets.cpu()
    write_offsets = expert_offsets_cpu[:-1].clone().to(device="mps", dtype=torch.int32)
    write_offsets_buf = mps_tensor_to_metal_buffer(write_offsets, device)

    # Step 4: Compute sorted indices using moe_scatter_indices kernel
    dispatch_kernel(
        lib,
        function_name="moe_scatter_indices",
        grid=((total_assignments + threads_per_tg - 1) // threads_per_tg, 1, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            expert_ids_buf,
            sorted_indices_buf,
            write_offsets_buf,
            batch_size_buf,
            top_k_buf,
            num_experts_buf,
        ],
        wait=True,
    )

    # Convert to int64 for consistency with moe_dispatch.py
    sorted_indices_int64 = sorted_indices.to(torch.int64)
    expert_offsets_int64 = expert_offsets.to(torch.int64)

    # Compute inverse indices: inverse[sorted_indices[i]] = i
    inverse_indices = torch.empty(total_assignments, dtype=torch.int64, device="mps")
    inverse_indices.scatter_(
        0, sorted_indices_int64, torch.arange(total_assignments, dtype=torch.int64, device="mps")
    )

    return sorted_indices_int64, expert_offsets_int64, inverse_indices


def group_tokens_by_expert_full_sparse(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> MoEDispatchInfo:
    """Full dispatch info for sparse case (top_k=1) using optimized path.

    Optimized version of group_tokens_by_expert_full_metal for the sparse
    case where each token is assigned to exactly one expert.

    Args:
        expert_ids: [batch, 1] or [batch] int tensor of expert assignments.
        num_experts: Total number of experts.

    Returns:
        MoEDispatchInfo with all indexing tensors for dispatch and scatter.
    """
    batch_size = expert_ids.shape[0] if expert_ids.dim() == 1 else expert_ids.shape[0]

    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert_sparse(
        expert_ids, num_experts
    )

    # For sparse case, sorted_token_indices = sorted_indices (identity)
    # and sorted_expert_indices is the expert_id for each token
    if expert_ids.dim() == 2:
        expert_ids_flat = expert_ids.squeeze(1)
    else:
        expert_ids_flat = expert_ids

    sorted_token_indices = sorted_indices
    sorted_expert_indices = expert_ids_flat.to(torch.int64)

    return MoEDispatchInfo(
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        expert_offsets=expert_offsets,
        inverse_indices=inverse_indices,
        num_tokens=batch_size,
        top_k=1,
        num_experts=num_experts,
    )


def group_tokens_by_expert_full_metal(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> MoEDispatchInfo:
    """Full dispatch info using Metal.

    Args:
        expert_ids: [batch, top_k] int tensor of expert assignments.
        num_experts: Total number of experts.

    Returns:
        MoEDispatchInfo with all indexing tensors for dispatch and scatter.
    """
    batch_size, top_k = expert_ids.shape
    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert_metal(
        expert_ids, num_experts
    )

    # Compute which original token each sorted assignment came from
    sorted_token_indices = sorted_indices // top_k

    # Compute which expert slot (0 to top_k-1) each sorted assignment came from
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


# ---------------------------------------------------------------------------
# Gather and Scatter (Metal-accelerated)
# ---------------------------------------------------------------------------


def gather_for_experts_metal(
    activations: torch.Tensor,
    dispatch_info: MoEDispatchInfo,
) -> torch.Tensor:
    """Gather activations using Metal kernel.

    Currently uses PyTorch gather as the Metal kernel for gather is not
    yet implemented. This is a placeholder for future Metal acceleration.

    Args:
        activations: [batch, hidden_dim] input activations.
        dispatch_info: Dispatch info from group_tokens_by_expert_full_metal.

    Returns:
        [total_assignments, hidden_dim] activations in expert-sorted order.
    """
    require_mps()

    # For now, use PyTorch gather (Metal kernel can be added later)
    # Gather using sorted_token_indices
    return activations[dispatch_info.sorted_token_indices]


def scatter_expert_outputs_metal(
    expert_outputs: torch.Tensor,
    expert_probs: torch.Tensor,
    dispatch_info: MoEDispatchInfo,
) -> torch.Tensor:
    """Scatter and combine outputs using Metal kernel.

    Currently uses PyTorch scatter as the Metal kernel for scatter-add is not
    yet fully implemented. This is a placeholder for future Metal acceleration.

    Args:
        expert_outputs: [total_assignments, out_dim] outputs from experts in
            sorted order (as produced by moe_expert_gemm).
        expert_probs: [batch, top_k] routing probabilities from router.
        dispatch_info: Dispatch info from group_tokens_by_expert_full_metal.

    Returns:
        [batch, out_dim] combined outputs with original token order.
    """
    require_mps()

    batch_size = dispatch_info.num_tokens
    top_k = dispatch_info.top_k
    out_dim = expert_outputs.shape[1]

    # Get probabilities for each sorted assignment
    probs_for_sorted = expert_probs[
        dispatch_info.sorted_token_indices, dispatch_info.sorted_expert_indices
    ]

    # Weight outputs by their routing probabilities
    weighted_outputs = expert_outputs * probs_for_sorted.unsqueeze(1)

    # Reorder from sorted order to original flat order [batch * top_k]
    weighted_original = weighted_outputs[dispatch_info.inverse_indices]

    # Reshape to [batch, top_k, out_dim] and sum over top_k dimension
    weighted_reshaped = weighted_original.reshape(batch_size, top_k, out_dim)
    output = weighted_reshaped.sum(dim=1)

    return output


def scatter_expert_outputs_sparse(
    expert_outputs: torch.Tensor,
    expert_probs: torch.Tensor,
    dispatch_info: MoEDispatchInfo,
) -> torch.Tensor:
    """Scatter and combine outputs for sparse case (top_k=1).

    Optimized path for tokens assigned to exactly one expert each. Avoids
    the expensive sum over top_k dimension and uses direct mapping.

    Args:
        expert_outputs: [batch, out_dim] outputs from experts in sorted order.
            For sparse case, this is [batch, out_dim] since each token has one
            expert assignment.
        expert_probs: [batch] or [batch, 1] routing probabilities.
        dispatch_info: Dispatch info from group_tokens_by_expert_full_sparse.

    Returns:
        [batch, out_dim] combined outputs with original token order.
    """
    require_mps()

    batch_size = dispatch_info.num_tokens
    out_dim = expert_outputs.shape[1]

    # For sparse case, expert_probs shape may be [batch, 1] or [batch]
    if expert_probs.dim() == 2:
        expert_probs = expert_probs.squeeze(1)

    # For sparse case where sorted_indices is identity:
    # - expert_outputs is already in sorted order
    # - We just need to weight by probabilities and possibly reorder
    # Since sorted_indices is identity, output is expert_outputs * probs
    output = expert_outputs * expert_probs.unsqueeze(1)

    return output


# ---------------------------------------------------------------------------
# Sparse-Expert Path (tokens with only one active expert)
# ---------------------------------------------------------------------------


def detect_sparse_expert_tokens(
    expert_probs: torch.Tensor,
    threshold: float = 0.99,
) -> torch.Tensor:
    """Detect tokens that have only one active expert (sparse activation).

    This identifies tokens where one expert has probability >= threshold,
    indicating that the token is effectively routed to a single expert.
    This is common in well-trained MoE models where routing converges.

    Args:
        expert_probs: [batch, top_k] or [batch, num_experts] routing probabilities.
            Should sum to 1.0 along the last dimension.
        threshold: Minimum probability for a single expert to be considered
            the "only" active expert. Default 0.99 (99% confidence).

    Returns:
        [batch] bool tensor where True indicates the token has sparse
        (single-expert) activation.

    Example:
        >>> expert_probs = torch.tensor([
        ...     [0.99, 0.005, 0.005],  # Sparse - one dominant expert
        ...     [0.5, 0.3, 0.2],       # Not sparse - distributed
        ...     [0.001, 0.998, 0.001], # Sparse - one dominant expert
        ... ])
        >>> detect_sparse_expert_tokens(expert_probs, threshold=0.99)
        tensor([True, False, True])
    """
    require_mps()

    # Get max probability per token
    max_probs = expert_probs.max(dim=-1).values

    # Token is sparse if max probability >= threshold
    return max_probs >= threshold


def create_sparse_expert_dispatch(
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    sparse_mask: torch.Tensor,
) -> tuple[MoEDispatchInfo, MoEDispatchInfo, torch.Tensor, torch.Tensor]:
    """Create dispatch info for sparse-expert path and dense path.

    Splits tokens into two groups:
    1. Sparse tokens: Those with only one active expert (direct routing)
    2. Dense tokens: Those with multiple active experts (standard MoE routing)

    This enables optimized execution where sparse tokens bypass the gather/scatter
    overhead and go directly to their single assigned expert.

    Args:
        expert_ids: [batch, top_k] expert assignments from top-k selection.
        expert_probs: [batch, top_k] routing probabilities for selected experts.
        sparse_mask: [batch] bool tensor from detect_sparse_expert_tokens().

    Returns:
        Tuple of:
        - sparse_dispatch: MoEDispatchInfo for sparse tokens (single expert each)
        - dense_dispatch: MoEDispatchInfo for dense tokens (multiple experts)
        - sparse_indices: [num_sparse] indices of sparse tokens in original batch
        - dense_indices: [num_dense] indices of dense tokens in original batch

    Example:
        >>> expert_ids = torch.tensor([[0, 1], [2, 0], [1, 2]])
        >>> expert_probs = torch.tensor([[0.99, 0.01], [0.5, 0.5], [0.995, 0.005]])
        >>> sparse_mask = detect_sparse_expert_tokens(expert_probs, 0.99)
        >>> sparse_dispatch, dense_dispatch, sparse_idx, dense_idx = \
        ...     create_sparse_expert_dispatch(expert_ids, expert_probs, sparse_mask)
    """
    require_mps()

    batch_size, top_k = expert_ids.shape
    num_experts = int(expert_ids.max().item()) + 1

    # Get indices of sparse and dense tokens
    sparse_indices = torch.where(sparse_mask)[0]
    dense_indices = torch.where(~sparse_mask)[0]

    num_sparse = sparse_indices.shape[0]
    num_dense = dense_indices.shape[0]

    # For sparse tokens: each goes to exactly one expert (the one with highest prob)
    if num_sparse > 0:
        # Get the expert with max probability for each sparse token
        sparse_expert_probs = expert_probs[sparse_indices]  # [num_sparse, top_k]
        sparse_expert_ids = expert_ids[sparse_indices]  # [num_sparse, top_k]

        # Find which expert slot has the max probability
        max_prob_indices = sparse_expert_probs.argmax(dim=-1)  # [num_sparse]

        # Get the actual expert ID for each sparse token (single expert)
        sparse_single_expert_ids = sparse_expert_ids[
            torch.arange(num_sparse, device=expert_ids.device),
            max_prob_indices,
        ]  # [num_sparse]

        # Create dispatch info for sparse tokens (top_k=1)
        sparse_dispatch = MoEDispatchInfo(
            sorted_token_indices=torch.arange(num_sparse, dtype=torch.int64, device=expert_ids.device),
            sorted_expert_indices=sparse_single_expert_ids,
            expert_offsets=_compute_expert_offsets_sparse(sparse_single_expert_ids, num_experts),
            inverse_indices=torch.arange(num_sparse, dtype=torch.int64, device=expert_ids.device),
            num_tokens=num_sparse,
            top_k=1,
            num_experts=num_experts,
        )
    else:
        # Empty sparse dispatch
        sparse_dispatch = MoEDispatchInfo(
            sorted_token_indices=torch.empty(0, dtype=torch.int64, device=expert_ids.device),
            sorted_expert_indices=torch.empty(0, dtype=torch.int64, device=expert_ids.device),
            expert_offsets=torch.zeros(num_experts + 1, dtype=torch.int64, device=expert_ids.device),
            inverse_indices=torch.empty(0, dtype=torch.int64, device=expert_ids.device),
            num_tokens=0,
            top_k=1,
            num_experts=num_experts,
        )

    # For dense tokens: use standard MoE routing with all top_k experts
    if num_dense > 0:
        dense_expert_ids = expert_ids[dense_indices]  # [num_dense, top_k]

        # Remap token indices: dense_indices[i] -> i in dense batch
        # Create dispatch info using standard grouping
        sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert(
            dense_expert_ids, num_experts
        )

        sorted_token_indices = sorted_indices // top_k
        sorted_expert_indices = sorted_indices % top_k

        dense_dispatch = MoEDispatchInfo(
            sorted_token_indices=sorted_token_indices,
            sorted_expert_indices=sorted_expert_indices,
            expert_offsets=expert_offsets,
            inverse_indices=inverse_indices,
            num_tokens=num_dense,
            top_k=top_k,
            num_experts=num_experts,
        )
    else:
        # Empty dense dispatch
        dense_dispatch = MoEDispatchInfo(
            sorted_token_indices=torch.empty(0, dtype=torch.int64, device=expert_ids.device),
            sorted_expert_indices=torch.empty(0, dtype=torch.int64, device=expert_ids.device),
            expert_offsets=torch.zeros(num_experts + 1, dtype=torch.int64, device=expert_ids.device),
            inverse_indices=torch.empty(0, dtype=torch.int64, device=expert_ids.device),
            num_tokens=0,
            top_k=top_k,
            num_experts=num_experts,
        )

    return sparse_dispatch, dense_dispatch, sparse_indices, dense_indices


def _compute_expert_offsets_sparse(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute expert offsets for sparse case (single expert per token).

    Args:
        expert_ids: [num_tokens] int tensor of expert assignments.
        num_experts: Total number of experts.

    Returns:
        [num_experts + 1] int64 cumulative counts (offsets).
    """
    device = expert_ids.device

    # Count occurrences of each expert
    expert_counts = torch.bincount(expert_ids, minlength=num_experts)

    # Cumsum to get offsets
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    return expert_offsets


def gather_for_sparse_experts(
    activations: torch.Tensor,
    sparse_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather activations for sparse-expert tokens.

    Optimized gather that only selects the tokens identified as sparse.

    Args:
        activations: [batch, hidden_dim] input activations.
        sparse_indices: [num_sparse] indices of sparse tokens.

    Returns:
        [num_sparse, hidden_dim] activations for sparse tokens.
    """
    require_mps()
    return activations[sparse_indices]


def scatter_sparse_expert_outputs(
    sparse_outputs: torch.Tensor,
    sparse_probs: torch.Tensor,
    output: torch.Tensor,
    sparse_indices: torch.Tensor,
) -> torch.Tensor:
    """Scatter sparse-expert outputs back to the full output tensor.

    Args:
        sparse_outputs: [num_sparse, out_dim] outputs from sparse experts.
        sparse_probs: [num_sparse] probabilities for sparse tokens.
        output: [batch, out_dim] output tensor to scatter into (modified in-place).
        sparse_indices: [num_sparse] indices where to place sparse outputs.

    Returns:
        [batch, out_dim] updated output tensor.
    """
    require_mps()

    # Weight by probability and scatter
    weighted = sparse_outputs * sparse_probs.unsqueeze(1)
    output[sparse_indices] = weighted

    return output


# ---------------------------------------------------------------------------
# High-level MoE Dispatch (Metal-accelerated)
# ---------------------------------------------------------------------------


def moe_dispatch_metal(
    activations: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    group_size: int = 128,
) -> torch.Tensor:
    """Full MoE dispatch using Metal kernels.

    This is a high-level function that combines:
    1. Token grouping by expert (Metal)
    2. Gather activations (PyTorch)
    3. Expert GEMM (Metal - via moe_dispatch_grouped or moe_expert_gemm)
    4. Scatter and combine outputs (PyTorch)

    Args:
        activations: [batch, hidden_dim] input activations.
        expert_ids: [batch, top_k] expert assignments.
        expert_probs: [batch, top_k] routing probabilities.
        expert_weights: [num_experts, K/8, N] packed FP4 expert weights.
        expert_scales: [num_experts, K/group_size, N] expert scales.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        group_size: Quantization group size.

    Returns:
        [batch, out_dim] combined outputs.
    """
    require_metal()
    require_mps()

    # Step 1: Group tokens by expert
    dispatch_info = group_tokens_by_expert_full_metal(expert_ids, num_experts)

    # Step 2: Gather activations (unused for now, will be used when GEMM is implemented)
    _ = gather_for_experts_metal(activations, dispatch_info)

    # Step 3: Expert GEMM (this would call the Metal GEMM kernel)
    # For now, this is a placeholder - the actual implementation would
    # dispatch to moe_dispatch_grouped or moe_expert_gemm_fp4_grouped
    # TODO: Implement Metal GEMM dispatch
    raise NotImplementedError(
        "Expert GEMM dispatch not yet implemented. "
        "Use metal_dispatch.dispatch_moe_optimized() for full MoE forward pass."
    )

    # Step 4: Scatter outputs
    # output = scatter_expert_outputs_metal(expert_outputs, expert_probs, dispatch_info)
    # return output
