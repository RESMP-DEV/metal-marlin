"""Metal dispatch for Trellis MoE kernels.

Provides Python wrappers for gemm_trellis_moe.metal kernels:
- dispatch_moe_trellis_swiglu: Fused MoE GEMM with SwiGLU activation

This replaces the slow sequential expert iteration in TrellisMoEMLP with a
single batched kernel that processes all experts in parallel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..metal_dispatch import (
    HAS_METAL,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)

if HAS_METAL:
    import Metal


def dispatch_moe_trellis_swiglu(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    gate_weights: torch.Tensor,
    gate_scales: torch.Tensor,
    up_weights: torch.Tensor,
    up_scales: torch.Tensor,
    down_weights: torch.Tensor,
    down_scales: torch.Tensor,
    gate_su: torch.Tensor,
    gate_sv: torch.Tensor,
    up_su: torch.Tensor,
    up_sv: torch.Tensor,
    down_su: torch.Tensor,
    down_sv: torch.Tensor,
    grid: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    bits: int,
) -> torch.Tensor:
    """Fused MoE GEMM with Trellis quantization and SwiGLU activation.

    Computes for each token:
        output = sum_{i=0}^{top_k-1} prob[i] *
                 down_proj(silu(gate_proj(x_i)) * up_proj(x_i))

    Where x_i is routed to expert_id[i].

    This is a single-kernel replacement for the slow sequential MoE implementation
    that iterates through 512 (top_k * num_experts) expert calls.

    Args:
        lib: MetalKernelLibrary with gemm_trellis_moe compiled
        activations: Input activations [batch, hidden] half, MPS tensor
        gate_weights: Packed Trellis gate weights [num_experts, ...] uint8
        gate_scales: Gate scales [num_experts, n_groups, intermediate] half
        up_weights: Packed Trellis up weights [num_experts, ...] uint8
        up_scales: Up scales [num_experts, n_groups, intermediate] half
        down_weights: Packed Trellis down weights [num_experts, ...] uint8
        down_scales: Down scales [num_experts, n_groups, hidden] half
        gate_su: Gate row signs [num_experts, hidden] half
        gate_sv: Gate column signs [num_experts, intermediate] half
        up_su: Up row signs [num_experts, hidden] half
        up_sv: Up column signs [num_experts, intermediate] half
        down_su: Down row signs [num_experts, intermediate] half
        down_sv: Down column signs [num_experts, hidden] half
        grid: Codebook grid [n_levels] half
        expert_ids: Expert assignments [batch, top_k] uint32
        expert_probs: Expert probabilities [batch, top_k] half
        hidden_dim: Model hidden dimension
        intermediate_dim: FFN intermediate dimension
        num_experts: Total number of experts
        top_k: Number of experts per token
        bits: Quantization bit width (2, 3, or 4)

    Returns:
        Output tensor [batch, hidden] half, MPS tensor
    """
    require_mps()

    device = lib.device
    batch_size = activations.shape[0]
    n_levels = grid.shape[0]

    # Ensure proper types
    activations = activations.contiguous()
    gate_weights = gate_weights.contiguous()
    gate_scales = gate_scales.float().contiguous()
    up_weights = up_weights.contiguous()
    up_scales = up_scales.float().contiguous()
    down_weights = down_weights.contiguous()
    down_scales = down_scales.float().contiguous()
    gate_su = gate_su.float().contiguous()
    gate_sv = gate_sv.float().contiguous()
    up_su = up_su.float().contiguous()
    up_sv = up_sv.float().contiguous()
    down_su = down_su.float().contiguous()
    down_sv = down_sv.float().contiguous()
    grid = grid.float().contiguous()
    expert_ids = expert_ids.int().contiguous()
    expert_probs = expert_probs.contiguous()

    # Allocate output
    output = torch.zeros(batch_size, hidden_dim, dtype=torch.float16, device="mps")

    # Create Metal buffers
    activations_buf = mps_tensor_to_metal_buffer(activations, device)
    gate_weights_buf = mps_tensor_to_metal_buffer(gate_weights, device)
    gate_scales_buf = mps_tensor_to_metal_buffer(gate_scales, device)
    up_weights_buf = mps_tensor_to_metal_buffer(up_weights, device)
    up_scales_buf = mps_tensor_to_metal_buffer(up_scales, device)
    down_weights_buf = mps_tensor_to_metal_buffer(down_weights, device)
    down_scales_buf = mps_tensor_to_metal_buffer(down_scales, device)
    gate_su_buf = mps_tensor_to_metal_buffer(gate_su, device)
    gate_sv_buf = mps_tensor_to_metal_buffer(gate_sv, device)
    up_su_buf = mps_tensor_to_metal_buffer(up_su, device)
    up_sv_buf = mps_tensor_to_metal_buffer(up_sv, device)
    down_su_buf = mps_tensor_to_metal_buffer(down_su, device)
    down_sv_buf = mps_tensor_to_metal_buffer(down_sv, device)
    grid_buf = mps_tensor_to_metal_buffer(grid, device)
    expert_ids_buf = mps_tensor_to_metal_buffer(expert_ids, device)
    expert_probs_buf = mps_tensor_to_metal_buffer(expert_probs, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Parameters struct
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    params_data = np.array(
        [
            batch_size,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            bits,
            32,  # group_size (fixed for Trellis)
            n_levels,
        ],
        dtype=np.uint32,
    )
    params_buf = device.newBufferWithBytes_length_options_(
        params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Dispatch configuration
    TILE_M = 64
    TILE_N = 64
    threads_per_tg = 128  # 4 simdgroups * 32 threads

    grid_x = (hidden_dim + TILE_N - 1) // TILE_N
    grid_y = (batch_size + TILE_M - 1) // TILE_M

    dispatch_kernel(
        lib,
        function_name="moe_trellis_swiglu",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            activations_buf,
            gate_weights_buf,
            gate_scales_buf,
            up_weights_buf,
            up_scales_buf,
            down_weights_buf,
            down_scales_buf,
            gate_su_buf,
            gate_sv_buf,
            up_su_buf,
            up_sv_buf,
            down_su_buf,
            down_sv_buf,
            grid_buf,
            expert_ids_buf,
            expert_probs_buf,
            output_buf,
            params_buf,
        ],
        wait=True,
    )

    return output
