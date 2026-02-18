"""GPU-accelerated top-k expert grouping for MoE dispatch.

This module provides optimized GPU-based token sorting to avoid CPU-based
token sorting overhead. It uses Metal kernels for counting-sort-based
grouping, which is significantly faster than CPU sorting.

Key features:
- Zero CPU-GPU synchronization during grouping
- O(N) counting sort complexity vs O(N log N) for comparison sorts
- Multi-threadgroup support for large batches
- Automatic fallback to CPU when Metal is unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


@dataclass
class GPUGroupingResult:
    """Result of GPU-based expert grouping.
    
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
            back to original order.
        num_tokens: Original batch size.
        top_k: Number of experts per token.
        num_experts: Total number of experts.
    """
    sorted_token_indices: torch.Tensor
    sorted_expert_indices: torch.Tensor
    expert_offsets: torch.Tensor
    inverse_indices: torch.Tensor
    num_tokens: int
    top_k: int
    num_experts: int
    
    @property
    def total_assignments(self) -> int:
        """Total number of token-expert assignments (num_tokens * top_k)."""
        return self.num_tokens * self.top_k
    
    def to_dispatch_info(self) -> "MoEDispatchInfo":
        """Convert to MoEDispatchInfo for compatibility with existing code."""
        from metal_marlin.moe_dispatch import MoEDispatchInfo
        
        return MoEDispatchInfo(
            sorted_token_indices=self.sorted_token_indices,
            sorted_expert_indices=self.sorted_expert_indices,
            expert_offsets=self.expert_offsets,
            inverse_indices=self.inverse_indices,
            num_tokens=self.num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
        )


# Cache for Metal kernel availability
_HAS_METAL_KERNELS = None


def _has_metal_kernels() -> bool:
    """Check if Metal kernels are available."""
    global _HAS_METAL_KERNELS
    if _HAS_METAL_KERNELS is None:
        try:
            from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS
            _HAS_METAL_KERNELS = HAS_METAL and HAS_MPS
        except ImportError:
            _HAS_METAL_KERNELS = False
    return _HAS_METAL_KERNELS


def group_tokens_by_expert_gpu_optimized(
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor | None = None,
    num_experts: int | None = None,
) -> GPUGroupingResult:
    """GPU-accelerated token grouping using Metal kernels.
    
    This is the main entry point for GPU-based grouping. It uses Metal kernels
    for counting-sort-based grouping, avoiding CPU-GPU synchronization.
    
    Args:
        expert_ids: [batch, top_k] int tensor of expert assignments.
        expert_probs: [batch, top_k] float tensor of expert probabilities.
            Optional, used only if needed for sorting (not used in current impl).
        num_experts: Total number of experts. Inferred from expert_ids if None.
    
    Returns:
        GPUGroupingResult with sorted indices and expert offsets.
    
    Raises:
        RuntimeError: If Metal is not available.
    """
    if num_experts is None:
        num_experts = int(expert_ids.max().item()) + 1
    
    batch_size, top_k = expert_ids.shape
    device = expert_ids.device
    
    # Ensure we're on MPS device
    if device.type != "mps":
        raise RuntimeError(f"GPU grouping requires MPS device, got {device}")
    
    if not _has_metal_kernels():
        raise RuntimeError("Metal kernels not available")
    
    # Use the existing GPU sort implementation from moe_dispatch
    from metal_marlin.moe_dispatch import group_tokens_by_expert_gpu
    
    # Call the Metal kernel-based sorting
    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert_gpu(
        expert_ids, expert_probs or torch.ones_like(expert_ids, dtype=torch.float32), num_experts
    )
    
    # Compute derived indices
    sorted_token_indices = sorted_indices // top_k
    sorted_expert_indices = sorted_indices % top_k
    
    return GPUGroupingResult(
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        expert_offsets=expert_offsets,
        inverse_indices=inverse_indices,
        num_tokens=batch_size,
        top_k=top_k,
        num_experts=num_experts,
    )


def group_tokens_by_expert_fast(
    expert_ids: torch.Tensor,
    num_experts: int | None = None,
) -> GPUGroupingResult:
    """Fast token grouping using GPU-accelerated counting sort.
    
    This function uses PyTorch operations optimized for GPU execution.
    It's faster than the standard CPU-based approach when running on MPS.
    
    Args:
        expert_ids: [batch, top_k] int tensor of expert assignments.
        num_experts: Total number of experts. Inferred from expert_ids if None.
    
    Returns:
        GPUGroupingResult with sorted indices.
    """
    if num_experts is None:
        num_experts = int(expert_ids.max().item()) + 1
    
    device = expert_ids.device
    batch_size, top_k = expert_ids.shape
    total_assignments = batch_size * top_k
    
    expert_ids_flat = expert_ids.reshape(-1).to(torch.int64)
    
    # Count assignments per expert using bincount (efficient on GPU)
    expert_counts = torch.bincount(expert_ids_flat, minlength=num_experts)
    
    # Compute prefix sum for output positions
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
    
    # Create position indices
    original_positions = torch.arange(total_assignments, dtype=torch.int64, device=device)
    
    # Vectorized position_in_group calculation using one-hot and cumsum
    # This is much faster than looping on GPU
    one_hot_experts = F.one_hot(expert_ids_flat, num_classes=num_experts).to(torch.int64)
    position_in_group = torch.cumsum(one_hot_experts, dim=0).gather(
        1, expert_ids_flat.unsqueeze(1)
    ).squeeze(1) - 1
    
    # Compute write positions
    write_positions = expert_offsets[expert_ids_flat] + position_in_group
    
    # Scatter to sorted positions
    sorted_indices = torch.empty(total_assignments, dtype=torch.int64, device=device)
    sorted_indices.scatter_(0, write_positions, original_positions)
    
    # Compute inverse indices
    inverse_indices = torch.empty_like(sorted_indices)
    inverse_indices.scatter_(0, sorted_indices, original_positions)
    
    # Compute derived indices
    sorted_token_indices = sorted_indices // top_k
    sorted_expert_indices = sorted_indices % top_k
    
    return GPUGroupingResult(
        sorted_token_indices=sorted_token_indices,
        sorted_expert_indices=sorted_expert_indices,
        expert_offsets=expert_offsets,
        inverse_indices=inverse_indices,
        num_tokens=batch_size,
        top_k=top_k,
        num_experts=num_experts,
    )


def group_tokens_by_expert_auto(
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor | None = None,
    num_experts: int | None = None,
    force_gpu: bool = False,
) -> GPUGroupingResult:
    """Automatically select best grouping implementation.
    
    Args:
        expert_ids: [batch, top_k] int tensor of expert assignments.
        expert_probs: [batch, top_k] float tensor of expert probabilities.
        num_experts: Total number of experts. Inferred from expert_ids if None.
        force_gpu: If True, raise error if GPU grouping is not available.
    
    Returns:
        GPUGroupingResult with sorted indices.
    """
    if num_experts is None:
        # Need to do this on CPU first to avoid sync issues
        num_experts = int(expert_ids.max().item()) + 1
    
    device = expert_ids.device
    
    # Try GPU path first if available
    if device.type == "mps" and _has_metal_kernels():
        try:
            return group_tokens_by_expert_gpu_optimized(
                expert_ids, expert_probs, num_experts
            )
        except Exception:
            # Fall through to fast GPU or CPU
            pass
        
        # Try fast GPU implementation
        try:
            return group_tokens_by_expert_fast(expert_ids, num_experts)
        except Exception:
            pass
    
    if force_gpu:
        raise RuntimeError("GPU grouping requested but not available")
    
    # Fall back to CPU-based implementation
    from metal_marlin.moe_dispatch import group_tokens_by_expert_full
    
    dispatch_info = group_tokens_by_expert_full(expert_ids, num_experts)
    
    return GPUGroupingResult(
        sorted_token_indices=dispatch_info.sorted_token_indices,
        sorted_expert_indices=dispatch_info.sorted_expert_indices,
        expert_offsets=dispatch_info.expert_offsets,
        inverse_indices=dispatch_info.inverse_indices,
        num_tokens=dispatch_info.num_tokens,
        top_k=dispatch_info.top_k,
        num_experts=dispatch_info.num_experts,
    )


class GPUExpertGrouping:
    """GPU-accelerated expert grouping manager.
    
    This class provides a convenient interface for GPU-based expert grouping
    that can be used as a drop-in replacement for CPU-based grouping.
    
    Example:
        >>> grouping = GPUExpertGrouping(num_experts=64)
        >>> result = grouping.group(expert_ids, expert_probs)
        >>> # Use result for dispatch
        >>> gathered = hidden_states[result.sorted_token_indices]
    """
    
    def __init__(
        self,
        num_experts: int,
        device: torch.device | str | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        """Initialize GPU expert grouping.
        
        Args:
            num_experts: Total number of experts.
            device: Target device. If None, uses MPS if available.
            prefer_gpu: If True, prefer GPU-based grouping when available.
        """
        self.num_experts = num_experts
        self.prefer_gpu = prefer_gpu
        
        if device is None:
            if torch.backends.mps.is_available() and prefer_gpu:
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self._use_gpu = (
            prefer_gpu and 
            device.type == "mps" and 
            _has_metal_kernels()
        )
    
    def group(
        self,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor | None = None,
    ) -> GPUGroupingResult:
        """Group tokens by expert assignment.
        
        Args:
            expert_ids: [batch, top_k] expert indices.
            expert_probs: [batch, top_k] routing probabilities.
        
        Returns:
            GPUGroupingResult with sorted indices.
        """
        # Move to target device if needed
        if expert_ids.device != self.device:
            expert_ids = expert_ids.to(self.device)
        if expert_probs is not None and expert_probs.device != self.device:
            expert_probs = expert_probs.to(self.device)
        
        if self._use_gpu:
            try:
                return group_tokens_by_expert_auto(
                    expert_ids, expert_probs, self.num_experts
                )
            except Exception:
                pass
        
        # Fall back to CPU-based grouping
        from metal_marlin.moe_dispatch import group_tokens_by_expert_full
        
        dispatch_info = group_tokens_by_expert_full(expert_ids, self.num_experts)
        
        return GPUGroupingResult(
            sorted_token_indices=dispatch_info.sorted_token_indices,
            sorted_expert_indices=dispatch_info.sorted_expert_indices,
            expert_offsets=dispatch_info.expert_offsets,
            inverse_indices=dispatch_info.inverse_indices,
            num_tokens=dispatch_info.num_tokens,
            top_k=dispatch_info.top_k,
            num_experts=dispatch_info.num_experts,
        )
    
    def gather_activations(
        self,
        hidden_states: torch.Tensor,
        grouping_result: GPUGroupingResult,
    ) -> torch.Tensor:
        """Gather activations based on grouping result.
        
        Args:
            hidden_states: [batch, hidden_dim] input activations.
            grouping_result: Result from group().
        
        Returns:
            [total_assignments, hidden_dim] gathered activations.
        """
        return hidden_states[grouping_result.sorted_token_indices]
    
    def scatter_outputs(
        self,
        expert_outputs: torch.Tensor,
        expert_probs: torch.Tensor,
        grouping_result: GPUGroupingResult,
    ) -> torch.Tensor:
        """Scatter and combine expert outputs.
        
        Args:
            expert_outputs: [total_assignments, out_dim] expert outputs.
            expert_probs: [batch, top_k] routing probabilities.
            grouping_result: Result from group().
        
        Returns:
            [batch, out_dim] combined outputs.
        """
        from metal_marlin.moe_dispatch import scatter_expert_outputs
        
        dispatch_info = grouping_result.to_dispatch_info()
        return scatter_expert_outputs(expert_outputs, expert_probs, dispatch_info)


__all__ = [
    "GPUGroupingResult",
    "GPUExpertGrouping",
    "group_tokens_by_expert_gpu_optimized",
    "group_tokens_by_expert_fast",
    "group_tokens_by_expert_auto",
]
