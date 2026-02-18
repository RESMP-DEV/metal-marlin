"""Fused MoE Router with sorted expert indices output.

This module provides a high-level interface to the fused MoE router kernel that:
1. Computes router GEMV: hidden @ router_weights -> logits
2. Applies softmax normalization
3. Selects top-k experts per token
4. Outputs sorted expert indices for efficient batched execution

The key benefit is eliminating CPU-side sorting by producing grouped token-expert
pairs directly on the GPU using atomic operations.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..metal_dispatch import MetalKernelLibrary


@dataclass
class FusedRouterOutput:
    """Output from fused router kernel with sorted expert indices.
    
    Attributes:
        expert_ids: [batch, top_k] selected expert indices (int64)
        expert_probs: [batch, top_k] normalized routing probabilities (float32)
        sorted_indices: [batch * top_k] indices sorted by expert ID (int64)
        expert_offsets: [num_experts + 1] start/end indices for each expert (int64)
        
    Memory layout for sorted output:
        expert 0: sorted_indices[expert_offsets[0]:expert_offsets[1]]
        expert 1: sorted_indices[expert_offsets[1]:expert_offsets[2]]
        ...
    """
    expert_ids: torch.Tensor
    expert_probs: torch.Tensor
    sorted_indices: torch.Tensor
    expert_offsets: torch.Tensor
    
    @property
    def batch_size(self) -> int:
        return self.expert_ids.shape[0]
    
    @property
    def top_k(self) -> int:
        return self.expert_ids.shape[1]
    
    @property
    def num_experts(self) -> int:
        return self.expert_offsets.shape[0] - 1


class FusedMoERouter:
    """Fused MoE router with GPU-side sorting.
    
    This router computes expert selection and produces sorted indices in a single
    GPU kernel, eliminating CPU-GPU synchronization for token grouping.
    
    Example:
        >>> router = FusedMoERouter(lib, num_experts=128, top_k=8, hidden_dim=7168)
        >>> output = router.forward(hidden_states, router_weights)
        >>> 
        >>> # Use sorted indices for batched expert execution
        >>> for e in range(num_experts):
        ...     start = output.expert_offsets[e].item()
        ...     end = output.expert_offsets[e + 1].item()
        ...     tokens_for_expert = output.sorted_indices[start:end] // top_k
    """
    
    def __init__(
        self,
        lib: MetalKernelLibrary,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
    ):
        """Initialize fused router.
        
        Args:
            lib: MetalKernelLibrary with compiled moe_fused_router kernels
            num_experts: Total number of experts (max 256)
            top_k: Number of experts to select per token (max 16)
            hidden_dim: Hidden dimension of input features
        """
        if num_experts > 256:
            raise ValueError(f"num_experts must be <= 256, got {num_experts}")
        if top_k > 16:
            raise ValueError(f"top_k must be <= 16, got {top_k}")
            
        self._lib = lib
        self._device = lib.device
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        
        # Cache pipelines
        self._pipeline = lib.get_pipeline("moe_fused_router_sorted_coalesced")
        
    def forward(
        self,
        hidden: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> FusedRouterOutput:
        """Execute fused router forward pass.
        
        Args:
            hidden: [batch, hidden_dim] input hidden states (fp16/fp32)
            router_weights: [hidden_dim, num_experts] or [num_experts, hidden_dim]
                If [hidden_dim, num_experts], will be transposed internally.
                
        Returns:
            FusedRouterOutput with sorted expert indices
        """
        from ..metal_dispatch import (
            dispatch_kernel,
            mps_tensor_to_metal_buffer,
        )
        
        batch_size = hidden.shape[0]
        total_assignments = batch_size * self.top_k
        
        # Prepare inputs
        hidden_fp16 = hidden.half().contiguous()
        hidden_buf = mps_tensor_to_metal_buffer(hidden_fp16, self._device)
        
        # Transpose weights to [num_experts, hidden_dim] for coalesced access
        if router_weights.shape[0] == self.hidden_dim:
            weights_t = router_weights.t()
        else:
            weights_t = router_weights
        weights_fp16 = weights_t.half().contiguous()
        weights_buf = mps_tensor_to_metal_buffer(weights_fp16, self._device)
        
        # Prepare outputs
        expert_offsets = torch.zeros(
            self.num_experts + 1, dtype=torch.uint32, device="mps"
        )
        sorted_indices = torch.zeros(
            total_assignments, dtype=torch.uint32, device="mps"
        )
        expert_ids = torch.zeros(
            batch_size, self.top_k, dtype=torch.uint32, device="mps"
        )
        expert_probs = torch.zeros(
            batch_size, self.top_k, dtype=torch.float16, device="mps"
        )
        
        offsets_buf = mps_tensor_to_metal_buffer(expert_offsets, self._device, copy_back=True)
        sorted_buf = mps_tensor_to_metal_buffer(sorted_indices, self._device, copy_back=True)
        ids_buf = mps_tensor_to_metal_buffer(expert_ids, self._device, copy_back=True)
        probs_buf = mps_tensor_to_metal_buffer(expert_probs, self._device, copy_back=True)
        
        # Pack parameters: batch_size, hidden_dim, num_experts, top_k
        RouterParams = struct.Struct('IIII')
        params_data = RouterParams.pack(batch_size, self.hidden_dim, self.num_experts, self.top_k)
        params_buf = self._device.newBufferWithBytes_length_options_(
            params_data, len(params_data), 0
        )
        
        # Dispatch kernel
        dispatch_kernel(
            self._lib,
            function_name="moe_fused_router_sorted_coalesced",
            grid=(batch_size, 1, 1),
            threadgroup=(256, 1, 1),
            buffers=[
                hidden_buf,      # buffer(0)
                weights_buf,     # buffer(1) - transposed
                offsets_buf,     # buffer(2) - output
                sorted_buf,      # buffer(3) - output
                ids_buf,         # buffer(4) - output
                probs_buf,       # buffer(5) - output
                params_buf,      # buffer(6)
            ],
            wait=True,
        )
        
        # Synchronize to ensure Metal kernel has completed and buffers are copied back
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        
        return FusedRouterOutput(
            expert_ids=expert_ids.to(torch.int64),
            expert_probs=expert_probs.to(torch.float32),
            sorted_indices=sorted_indices.to(torch.int64),
            expert_offsets=expert_offsets.to(torch.int64),
        )


def dispatch_fused_router_sorted(
    lib: MetalKernelLibrary,
    hidden: torch.Tensor,
    router_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional interface to fused router with sorted output.
    
    This is a convenience function that creates a FusedMoERouter and executes
    the forward pass. For repeated calls with the same parameters, consider
    creating a FusedMoERouter instance and reusing it.
    
    Args:
        lib: MetalKernelLibrary with compiled kernels
        hidden: [batch, hidden_dim] input hidden states
        router_weights: [hidden_dim, num_experts] router weight matrix
        num_experts: Total number of experts
        top_k: Number of experts to select per token
        
    Returns:
        Tuple of (expert_ids, expert_probs, sorted_indices, expert_offsets)
    """
    hidden_dim = hidden.shape[1]
    router = FusedMoERouter(lib, num_experts, top_k, hidden_dim)
    output = router.forward(hidden, router_weights)
    return (
        output.expert_ids,
        output.expert_probs,
        output.sorted_indices,
        output.expert_offsets,
    )
