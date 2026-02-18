"""GPU-based top-k expert grouping for MoE dispatch.

This module provides optimized GPU-accelerated token sorting to avoid CPU-based
token sorting overhead. It uses multi-threadgroup Metal kernels for scalable
performance on large batches.

Key features:
- Multi-threadgroup GPU counting sort (scales to large batches)
- Zero CPU-GPU synchronization during sorting
- Compatible with existing MoEDispatchInfo format
- Fallback to CPU sort when Metal is unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass
class TopKGroupingResult:
    """Result of GPU-based top-k expert grouping.
    
    Attributes:
        sorted_indices: [batch * top_k] indices into original batch, grouped by expert.
        expert_offsets: [num_experts + 1] cumulative counts for each expert.
        inverse_indices: [batch * top_k] indices to restore original order.
        sorted_token_indices: [batch * top_k] which original token each assignment came from.
        sorted_expert_indices: [batch * top_k] which expert slot (0 to top_k-1) each came from.
        device: Device where tensors are stored.
    """

    sorted_indices: torch.Tensor
    expert_offsets: torch.Tensor
    inverse_indices: torch.Tensor
    sorted_token_indices: torch.Tensor
    sorted_expert_indices: torch.Tensor
    device: torch.device

    @property
    def num_tokens(self) -> int:
        """Number of tokens in the batch."""
        return self.sorted_token_indices.shape[0] // self.sorted_expert_indices.shape[0]


def topk_expert_grouping_gpu(
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    num_experts: int,
    use_optimized: bool = True,
) -> TopKGroupingResult:
    """GPU-accelerated top-k expert grouping.
    
    Sorts token-expert assignments by expert ID on the GPU, avoiding CPU-based
    sorting overhead. Uses multi-threadgroup counting sort for scalability.
    
    Args:
        expert_ids: [batch, top_k] int tensor of expert assignments.
        expert_probs: [batch, top_k] float tensor of expert probabilities.
        num_experts: Total number of experts.
        use_optimized: If True, use the optimized multi-threadgroup kernel
            when available. Falls back to single-threadgroup for small batches.
    
    Returns:
        TopKGroupingResult with sorted indices and offsets.
    
    Raises:
        RuntimeError: If Metal is not available.
    
    Example:
        >>> expert_ids = torch.tensor([[0, 3], [1, 2], [0, 1]], device="mps")
        >>> expert_probs = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], device="mps")
        >>> result = topk_expert_grouping_gpu(expert_ids, expert_probs, num_experts=4)
        >>> # result.sorted_indices groups assignments by expert
        >>> # expert 0: indices [0, 4] (token 0 slot 0, token 2 slot 0)
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("GPU-based grouping requires MPS (Metal) backend")
    
    if expert_ids.device.type != "mps":
        raise RuntimeError(f"Inputs must be on MPS device, got {expert_ids.device}")
    
    batch_size, top_k = expert_ids.shape
    total_assignments = batch_size * top_k
    device = expert_ids.device
    
    # For small batches, use single-threadgroup kernel
    # For large batches, use optimized multi-threadgroup kernel
    if use_optimized and total_assignments > 2048:
        return _topk_grouping_optimized(expert_ids, expert_probs, num_experts)
    else:
        return _topk_grouping_simple(expert_ids, expert_probs, num_experts)


def _topk_grouping_simple(
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    num_experts: int,
) -> TopKGroupingResult:
    """Simple single-threadgroup GPU sorting (for small batches)."""
    from metal_marlin.moe_dispatch import group_tokens_by_expert_gpu
    
    batch_size, top_k = expert_ids.shape
    
    # Use existing kernel
    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert_gpu(
        expert_ids, expert_probs, num_experts
    )
    
    # Compute derived indices
    sorted_token_indices = sorted_indices // top_k
    sorted_expert_indices = sorted_indices % top_k
    
    return TopKGroupingResult(
        sorted_indices=sorted_indices,
        expert_offsets=expert_offsets,
        inverse_indices=inverse_indices,
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        device=expert_ids.device,
    )


def _topk_grouping_optimized(
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    num_experts: int,
) -> TopKGroupingResult:
    """Optimized multi-threadgroup GPU sorting (for large batches)."""
    import numpy as np
    
    from metal_marlin.metal_dispatch import (
        dispatch_kernel,
        get_default_library,
        mps_tensor_to_metal_buffer,
        require_metal,
    )
    
    require_metal()
    
    batch_size, top_k = expert_ids.shape
    device = expert_ids.device
    total_assignments = batch_size * top_k
    
    # Ensure inputs are contiguous
    expert_ids_i32 = expert_ids.contiguous().to(torch.int32)
    expert_probs_f16 = expert_probs.contiguous().to(torch.float16)
    
    # Allocate output buffers
    sorted_indices = torch.empty(total_assignments, dtype=torch.int32, device=device)
    sorted_expert_ids = torch.empty(total_assignments, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    
    # Get Metal library
    lib = get_default_library()
    metal_device = lib.device
    
    # Convert tensors to Metal buffers
    expert_ids_buf = mps_tensor_to_metal_buffer(expert_ids_i32, metal_device)
    expert_probs_buf = mps_tensor_to_metal_buffer(expert_probs_f16, metal_device)
    sorted_indices_buf = mps_tensor_to_metal_buffer(sorted_indices, metal_device, copy_back=True)
    sorted_expert_ids_buf = mps_tensor_to_metal_buffer(sorted_expert_ids, metal_device, copy_back=True)
    expert_offsets_buf = mps_tensor_to_metal_buffer(expert_offsets, metal_device, copy_back=True)
    
    # Create parameter buffer
    params = np.array([batch_size, top_k, num_experts], dtype=np.uint32)
    params_buf = metal_device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, 0
    )
    
    # Calculate grid size: use multiple threadgroups for large batches
    # Each threadgroup handles a portion of the assignments
    threadgroup_size = 256
    num_threadgroups = min(
        (total_assignments + threadgroup_size - 1) // threadgroup_size,
        32  # Cap at 32 threadgroups for efficiency
    )
    
    # Dispatch optimized kernel
    dispatch_kernel(
        lib,
        function_name="moe_topk_grouping_optimized",
        grid=(num_threadgroups, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        buffers=[
            expert_ids_buf,
            expert_probs_buf,
            sorted_indices_buf,
            sorted_expert_ids_buf,
            expert_offsets_buf,
            params_buf,
        ],
    )
    
    # Compute derived tensors
    sorted_indices_long = sorted_indices.long()
    sorted_token_indices = sorted_indices_long // top_k
    sorted_expert_indices = sorted_indices_long % top_k
    expert_offsets_long = expert_offsets.long()
    
    # Compute inverse indices
    inverse_indices = torch.argsort(sorted_indices_long)
    
    return TopKGroupingResult(
        sorted_indices=sorted_indices_long,
        expert_offsets=expert_offsets_long,
        inverse_indices=inverse_indices,
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        device=device,
    )


def topk_expert_grouping_cpu(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> TopKGroupingResult:
    """CPU-based top-k expert grouping (fallback).
    
    Uses the existing group_tokens_by_expert implementation for compatibility.
    
    Args:
        expert_ids: [batch, top_k] int tensor of expert assignments.
        num_experts: Total number of experts.
    
    Returns:
        TopKGroupingResult with sorted indices.
    """
    from metal_marlin.moe_dispatch import group_tokens_by_expert
    
    batch_size, top_k = expert_ids.shape
    device = expert_ids.device
    
    # Use existing CPU implementation
    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert(
        expert_ids, num_experts
    )
    
    # Compute derived indices
    sorted_token_indices = sorted_indices // top_k
    sorted_expert_indices = sorted_indices % top_k
    
    return TopKGroupingResult(
        sorted_indices=sorted_indices,
        expert_offsets=expert_offsets,
        inverse_indices=inverse_indices,
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        device=device,
    )


def topk_expert_grouping(
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor | None = None,
    num_experts: int | None = None,
    use_gpu: bool = True,
    use_optimized: bool = True,
) -> TopKGroupingResult:
    """Unified top-k expert grouping (auto-selects best implementation).
    
    Automatically selects between GPU and CPU implementations based on:
    - Device availability (MPS vs CPU)
    - Batch size (optimized kernel for large batches)
    - Configuration flags
    
    Args:
        expert_ids: [batch, top_k] int tensor of expert assignments.
        expert_probs: [batch, top_k] float tensor of expert probabilities.
            Required for GPU path, optional for CPU path.
        num_experts: Total number of experts. Inferred from expert_ids if None.
        use_gpu: If True, try GPU implementation first.
        use_optimized: If True, use multi-threadgroup optimized kernel.
    
    Returns:
        TopKGroupingResult with sorted indices.
    
    Example:
        >>> # Auto-selects best implementation
        >>> result = topk_expert_grouping(expert_ids, expert_probs, num_experts=64)
        >>> 
        >>> # Force CPU fallback
        >>> result = topk_expert_grouping(expert_ids, num_experts=64, use_gpu=False)
    """
    if num_experts is None:
        num_experts = int(expert_ids.max().item()) + 1
    
    # Try GPU path first if requested and available
    if use_gpu and torch.backends.mps.is_available():
        if expert_ids.device.type == "mps":
            try:
                return topk_expert_grouping_gpu(
                    expert_ids, expert_probs or torch.ones_like(expert_ids, dtype=torch.float32),
                    num_experts, use_optimized
                )
            except Exception:
                # Fall through to CPU
                pass
        elif expert_probs is not None:
            # Move to MPS
            try:
                return topk_expert_grouping_gpu(
                    expert_ids.to("mps"),
                    expert_probs.to("mps"),
                    num_experts, use_optimized
                )
            except Exception:
                # Fall through to CPU
                pass
    
    # CPU fallback
    return topk_expert_grouping_cpu(expert_ids, num_experts)


def create_dispatch_info_from_grouping(
    grouping_result: TopKGroupingResult,
    num_tokens: int,
    top_k: int,
    num_experts: int,
) -> "MoEDispatchInfo":
    """Convert TopKGroupingResult to MoEDispatchInfo.
    
    This allows the new GPU-based grouping to work with existing MoE dispatch code.
    
    Args:
        grouping_result: Result from topk_expert_grouping().
        num_tokens: Number of tokens in the batch.
        top_k: Number of experts per token.
        num_experts: Total number of experts.
    
    Returns:
        MoEDispatchInfo compatible with existing dispatch functions.
    """
    from metal_marlin.moe_dispatch import MoEDispatchInfo
    
    return MoEDispatchInfo(
        sorted_token_indices=grouping_result.sorted_token_indices,
        sorted_expert_indices=grouping_result.sorted_expert_indices,
        expert_offsets=grouping_result.expert_offsets,
        inverse_indices=grouping_result.inverse_indices,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
    )


__all__ = [
    "TopKGroupingResult",
    "topk_expert_grouping",
    "topk_expert_grouping_gpu",
    "topk_expert_grouping_cpu",
    "create_dispatch_info_from_grouping",
]
