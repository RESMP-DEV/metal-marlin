"""Metal kernel dispatch for MoE token grouping and scattering.

This module provides Metal-accelerated versions of the MoE dispatch operations
from moe_dispatch.py. It uses Metal compute kernels for:
    - Computing expert counts (histogram)
    - Computing expert offsets (prefix sum)
    - Computing sorted indices (scatter)
    - Gathering activations for experts
    - Scattering and combining expert outputs

Usage:
    from metal_marlin.moe_dispatch_metal import group_tokens_by_expert_metal

    sorted_indices, expert_offsets, inverse_indices = group_tokens_by_expert_metal(
        expert_ids, num_experts
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._compat import Metal
from .metal_dispatch import (
    dispatch_kernel,
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_metal,
    require_mps,
)
from .moe_dispatch import MoEDispatchInfo

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Token Grouping (Metal-accelerated)
# ---------------------------------------------------------------------------


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
