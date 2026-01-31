"""Metal dispatch for Trellis MoE kernels.

Provides Python wrappers for gemm_trellis_moe.metal kernels:
- dispatch_moe_trellis_swiglu: Fused MoE GEMM with SwiGLU activation

CRITICAL: The kernel uses per-token expert routing. Each token uses its OWN
assigned expert from expert_ids[token * top_k + slot], NOT a shared expert
for all tokens in a tile. The grid is 3D: (n_blocks, tokens, slots).

This replaces the slow sequential expert iteration in TrellisMoEMLP with a
single batched kernel that processes all experts in parallel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass
class CachedWeightBuffers:
    """Pre-allocated Metal buffers for static MoE weights.

    These buffers persist across dispatch calls, avoiding expensive
    buffer creation during the decode phase.
    """

    gate_weights: Any
    gate_scales: Any
    up_weights: Any
    up_scales: Any
    down_weights: Any
    down_scales: Any
    gate_su: Any
    gate_sv: Any
    up_su: Any
    up_sv: Any
    down_su: Any
    down_sv: Any
    grid: Any


def create_cached_weight_buffers(
    device: Any,
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
) -> CachedWeightBuffers:
    """Create cached Metal buffers for static MoE weights.

    Call this once during model initialization, then pass the returned
    CachedWeightBuffers to dispatch_moe_trellis_swiglu for each forward pass.

    Args:
        device: MTLDevice to create buffers on.
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

    Returns:
        CachedWeightBuffers containing Metal buffers for all static weights.
    """
    require_mps()

    # CRITICAL: Metal MoE kernel expects half (float16) for scales/su/sv/grid.
    # TrellisLinear stores these as float32, so convert here to avoid
    # dtype mismatch that causes garbage values (NaN).
    def to_half(t: torch.Tensor) -> torch.Tensor:
        return t.half().contiguous()

    return CachedWeightBuffers(
        gate_weights=mps_tensor_to_metal_buffer(gate_weights.contiguous(), device),
        gate_scales=mps_tensor_to_metal_buffer(to_half(gate_scales), device),
        up_weights=mps_tensor_to_metal_buffer(up_weights.contiguous(), device),
        up_scales=mps_tensor_to_metal_buffer(to_half(up_scales), device),
        down_weights=mps_tensor_to_metal_buffer(down_weights.contiguous(), device),
        down_scales=mps_tensor_to_metal_buffer(to_half(down_scales), device),
        gate_su=mps_tensor_to_metal_buffer(to_half(gate_su), device),
        gate_sv=mps_tensor_to_metal_buffer(to_half(gate_sv), device),
        up_su=mps_tensor_to_metal_buffer(to_half(up_su), device),
        up_sv=mps_tensor_to_metal_buffer(to_half(up_sv), device),
        down_su=mps_tensor_to_metal_buffer(to_half(down_su), device),
        down_sv=mps_tensor_to_metal_buffer(to_half(down_sv), device),
        grid=mps_tensor_to_metal_buffer(to_half(grid), device),
    )


# Debug counters for buffer tracking (module-level mutable state)
_dispatch_call_count = 0
_cached_buffer_hit_count = 0
_cached_buffer_miss_count = 0
_debug_moe_dispatch = False  # Set True or use env var METAL_DEBUG=1


def get_buffer_stats() -> dict[str, int]:
    """Get buffer caching statistics for debugging.

    Returns:
        Dictionary with dispatch_calls, cache_hits, cache_misses.
    """
    return {
        "dispatch_calls": _dispatch_call_count,
        "cache_hits": _cached_buffer_hit_count,
        "cache_misses": _cached_buffer_miss_count,
    }


def reset_buffer_stats() -> None:
    """Reset buffer caching statistics."""
    global _dispatch_call_count, _cached_buffer_hit_count, _cached_buffer_miss_count
    _dispatch_call_count = 0
    _cached_buffer_hit_count = 0
    _cached_buffer_miss_count = 0


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
    *,
    cached_buffers: CachedWeightBuffers | None = None,
    use_fp32_acc: bool = False,
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
        cached_buffers: Optional pre-allocated Metal buffers for static weights.
            If provided, weight tensor arguments are ignored and cached buffers
            are used instead. Create with create_cached_weight_buffers().
        use_fp32_acc: If True, use FP32 accumulators for numerical stability.
            Recommended for large K dimensions (e.g., K=3584, intermediate=18944)
            where FP16 accumulation may overflow or lose precision.

    Returns:
        Output tensor [batch, hidden] half, MPS tensor
    """
    require_mps()

    # Track buffer usage for debugging
    global _dispatch_call_count, _cached_buffer_hit_count, _cached_buffer_miss_count
    _dispatch_call_count += 1

    debug_enabled = _debug_moe_dispatch or os.environ.get("METAL_DEBUG") == "1"

    device = lib.device
    batch_size = activations.shape[0]
    n_levels = grid.shape[0]

    # Dynamic inputs - always create new buffers
    activations = activations.contiguous()
    expert_ids = expert_ids.int().contiguous()
    expert_probs = expert_probs.contiguous()

    activations_buf = mps_tensor_to_metal_buffer(activations, device)
    expert_ids_buf = mps_tensor_to_metal_buffer(expert_ids, device)
    expert_probs_buf = mps_tensor_to_metal_buffer(expert_probs, device)

    # Static weights - use cached buffers if provided
    if cached_buffers is not None:
        _cached_buffer_hit_count += 1
        if debug_enabled and _dispatch_call_count <= 5:
            print(
                f"[MoE DISPATCH #{_dispatch_call_count}] CACHED buffers reused "
                f"(hit #{_cached_buffer_hit_count}, batch={batch_size})"
            )
        gate_weights_buf = cached_buffers.gate_weights
        gate_scales_buf = cached_buffers.gate_scales
        up_weights_buf = cached_buffers.up_weights
        up_scales_buf = cached_buffers.up_scales
        down_weights_buf = cached_buffers.down_weights
        down_scales_buf = cached_buffers.down_scales
        gate_su_buf = cached_buffers.gate_su
        gate_sv_buf = cached_buffers.gate_sv
        up_su_buf = cached_buffers.up_su
        up_sv_buf = cached_buffers.up_sv
        down_su_buf = cached_buffers.down_su
        down_sv_buf = cached_buffers.down_sv
        grid_buf = cached_buffers.grid
    else:
        # No cache - create buffers on the fly (slow path)
        _cached_buffer_miss_count += 1
        if debug_enabled:
            print(
                f"[MoE DISPATCH #{_dispatch_call_count}] WARNING: Creating 13 new "
                f"Metal buffers (miss #{_cached_buffer_miss_count}, batch={batch_size})"
            )
            import traceback

            traceback.print_stack(limit=10)

        gate_weights = gate_weights.contiguous()
        gate_scales = gate_scales.contiguous()
        up_weights = up_weights.contiguous()
        up_scales = up_scales.contiguous()
        down_weights = down_weights.contiguous()
        down_scales = down_scales.contiguous()
        gate_su = gate_su.contiguous()
        gate_sv = gate_sv.contiguous()
        up_su = up_su.contiguous()
        up_sv = up_sv.contiguous()
        down_su = down_su.contiguous()
        down_sv = down_sv.contiguous()
        grid = grid.contiguous()

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

    # Allocate output
    output = torch.zeros(batch_size, hidden_dim, dtype=torch.float16, device="mps")
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Parameters struct
    params_data = np.array(
        [
            batch_size,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            bits,
            128,  # group_size (128 for Trellis scales - must match TrellisLinear.GROUP_SIZE)
            n_levels,
        ],
        dtype=np.uint32,
    )
    params_buf = device.newBufferWithBytes_length_options_(
        params_data.tobytes(), params_data.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Dispatch configuration - must match Metal shader constants
    # Grid layout: (ceil(hidden_dim / MOE_TILE_N), M, top_k)
    #   - tgid.x: output column block (into hidden_dim)
    #   - tgid.y: token index
    #   - tgid.z: expert slot (0 to top_k-1)
    #
    # Each threadgroup handles one (token, slot, n_block) combination to ensure
    # CORRECT per-token expert routing. This fixes the bug where all tokens in
    # a tile incorrectly used the first token's expert.
    TILE_N = 64  # Output column tile (matches MOE_TILE_N in kernel)
    threads_per_tg = 128  # 4 simdgroups * 32 threads

    grid_x = (hidden_dim + TILE_N - 1) // TILE_N  # Output column blocks
    grid_y = batch_size  # One threadgroup per token
    grid_z = top_k  # One threadgroup per expert slot

    # Select kernel based on FP32 accumulation flag
    kernel_name = "moe_trellis_swiglu_fp32acc" if use_fp32_acc else "moe_trellis_swiglu"

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, grid_z),
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
