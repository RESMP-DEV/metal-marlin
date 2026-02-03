"""Fused RoPE + Projection dispatch for Metal.

Provides Python wrappers for the rope_proj_fused.metal kernels:
- dispatch_rope_fused_proj_fp16: FP16 weights with fused RoPE
- dispatch_rope_fused_qk_proj_trellis: Trellis-quantized Q/K with fused RoPE

These kernels eliminate the memory round-trip between projection and RoPE:
    Standard: Q = x @ Wq; Q = apply_rope(Q)  # 2 kernel launches, intermediate Q in DRAM
    Fused:    Q = apply_rope(x @ Wq)         # 1 kernel launch, no intermediate

Benefits:
    - ~50% memory bandwidth reduction for projection+rope
    - Lower latency (fewer kernel launches)
    - Better cache utilization
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..metal_dispatch import (
    HAS_METAL,
    HAS_MPS,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)

if TYPE_CHECKING:
    from .linear import TrellisLinear

_logger = logging.getLogger(__name__)
_METAL_DEBUG = os.environ.get("METAL_DEBUG", "").lower() in ("1", "true", "yes")

if HAS_METAL:
    import Metal


# Tile dimensions (must match metal file)
TILE_M = 32
TILE_N = 32


def dispatch_rope_fused_proj_fp16(
    lib: MetalKernelLibrary,
    input: torch.Tensor,
    weight: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    head_dim: int,
    position_offset: int = 0,
) -> torch.Tensor:
    """Fused FP16 projection + RoPE kernel.

    Computes: output = apply_rope(input @ weight.T)

    Args:
        lib: MetalKernelLibrary with rope_proj_fused compiled
        input: Input tensor [M, K] float16, MPS tensor
        weight: Weight matrix [N, K] float16 (transposed), MPS tensor
        cos_cache: RoPE cos values [max_seq, head_dim/2] float16, MPS tensor
        sin_cache: RoPE sin values [max_seq, head_dim/2] float16, MPS tensor
        head_dim: Head dimension for RoPE application
        position_offset: Position offset for KV cache continuation

    Returns:
        Output tensor [M, N] float16 with RoPE applied, MPS tensor
    """
    require_mps()

    device = lib.device
    M, K = input.shape
    N = weight.shape[0]

    # Ensure contiguous and correct types
    input = input.contiguous()
    weight = weight.contiguous()
    cos_cache = cos_cache.contiguous()
    sin_cache = sin_cache.contiguous()

    # Allocate output
    output = torch.zeros(M, N, dtype=torch.float16, device="mps")

    # Create Metal buffers
    input_buf = mps_tensor_to_metal_buffer(input, device)
    weight_buf = mps_tensor_to_metal_buffer(weight, device)
    cos_buf = mps_tensor_to_metal_buffer(cos_cache, device)
    sin_buf = mps_tensor_to_metal_buffer(sin_cache, device)
    output_buf = mps_tensor_to_metal_buffer(output, device, copy_back=True)

    # Create parameter buffers
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    max_seq_len = cos_cache.shape[0]

    M_buf = make_uint_buffer(M)
    K_buf = make_uint_buffer(K)
    N_buf = make_uint_buffer(N)
    head_dim_buf = make_uint_buffer(head_dim)
    pos_offset_buf = make_uint_buffer(position_offset)
    max_seq_buf = make_uint_buffer(max_seq_len)

    # Choose kernel based on M
    if M <= 1:
        kernel_name = "rope_fused_proj_fp16_decode"
    else:
        kernel_name = "rope_fused_proj_fp16"

    # Grid dimensions
    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M
    threads_per_tg = 128

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            input_buf,
            weight_buf,
            cos_buf,
            sin_buf,
            output_buf,
            M_buf,
            K_buf,
            N_buf,
            head_dim_buf,
            pos_offset_buf,
            max_seq_buf,
        ],
        wait=True,
    )

    return output


def dispatch_rope_fused_qk_proj_trellis(
    lib: MetalKernelLibrary,
    input: torch.Tensor,
    q_proj: TrellisLinear,
    k_proj: TrellisLinear,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    head_dim: int,
    rope_dim: int | None = None,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused trellis Q/K projection + RoPE kernel.

    Computes:
        Q = apply_rope(input @ Wq)
        K = apply_rope(input @ Wk)

    For MLA models where RoPE is decoupled (rope_dim < head_dim), only the
    rope_dim portion receives RoPE encoding.

    Args:
        lib: MetalKernelLibrary with rope_proj_fused compiled
        input: Input tensor [M, K] float16, MPS tensor
        q_proj: TrellisLinear for Q projection
        k_proj: TrellisLinear for K projection
        cos_cache: RoPE cos values [max_seq, rope_dim/2] float16, MPS tensor
        sin_cache: RoPE sin values [max_seq, rope_dim/2] float16, MPS tensor
        head_dim: Head dimension for output layout
        rope_dim: RoPE application dimension (None = head_dim)
        position_offset: Position offset for KV cache continuation

    Returns:
        Tuple of (Q, K) output tensors, each [M, N_*] float16 with RoPE applied
    """
    require_mps()

    if rope_dim is None:
        rope_dim = head_dim

    device = lib.device
    M, K = input.shape
    Nq = q_proj.out_features
    Nk = k_proj.out_features

    # Verify compatible dimensions
    if k_proj.in_features != K or q_proj.in_features != K:
        raise ValueError(f"Input dimension mismatch: input K={K}, "
                        f"q_proj.in={q_proj.in_features}, k_proj.in={k_proj.in_features}")

    if q_proj.bits != k_proj.bits:
        raise ValueError(f"Bit width mismatch: Q={q_proj.bits}, K={k_proj.bits}")

    # Get quantization parameters
    bits = q_proj.bits
    n_levels = q_proj.grid.shape[0]
    n_groups = q_proj.scales.shape[0]
    group_size = (K + n_groups - 1) // n_groups

    # Ensure contiguous
    input = input.contiguous()
    cos_cache = cos_cache.contiguous()
    sin_cache = sin_cache.contiguous()

    # Allocate outputs
    out_q = torch.zeros(M, Nq, dtype=torch.float16, device="mps")
    out_k = torch.zeros(M, Nk, dtype=torch.float16, device="mps")

    # Create Metal buffers
    input_buf = mps_tensor_to_metal_buffer(input, device)

    # Q projection buffers
    Wq_buf = mps_tensor_to_metal_buffer(q_proj.packed_indices.contiguous(), device)
    scales_q_buf = mps_tensor_to_metal_buffer(q_proj.scales.float().contiguous(), device)
    su_q_buf = mps_tensor_to_metal_buffer(q_proj.su.float().contiguous(), device)
    sv_q_buf = mps_tensor_to_metal_buffer(q_proj.sv.float().contiguous(), device)

    # K projection buffers
    Wk_buf = mps_tensor_to_metal_buffer(k_proj.packed_indices.contiguous(), device)
    scales_k_buf = mps_tensor_to_metal_buffer(k_proj.scales.float().contiguous(), device)
    su_k_buf = mps_tensor_to_metal_buffer(k_proj.su.float().contiguous(), device)
    sv_k_buf = mps_tensor_to_metal_buffer(k_proj.sv.float().contiguous(), device)

    # Shared resources
    grid_buf = mps_tensor_to_metal_buffer(q_proj.grid.float().contiguous(), device)
    cos_buf = mps_tensor_to_metal_buffer(cos_cache, device)
    sin_buf = mps_tensor_to_metal_buffer(sin_cache, device)

    # Output buffers
    out_q_buf = mps_tensor_to_metal_buffer(out_q, device, copy_back=True)
    out_k_buf = mps_tensor_to_metal_buffer(out_k, device, copy_back=True)

    # Create parameter buffers
    def make_uint_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    max_seq_len = cos_cache.shape[0]
    half_rope_dim = rope_dim // 2

    M_buf = make_uint_buffer(M)
    K_buf = make_uint_buffer(K)
    Nq_buf = make_uint_buffer(Nq)
    Nk_buf = make_uint_buffer(Nk)
    bits_buf = make_uint_buffer(bits)
    n_levels_buf = make_uint_buffer(n_levels)
    group_size_buf = make_uint_buffer(group_size)
    head_dim_buf = make_uint_buffer(head_dim)
    rope_dim_buf = make_uint_buffer(rope_dim)
    pos_offset_buf = make_uint_buffer(position_offset)
    max_seq_buf = make_uint_buffer(max_seq_len)

    # Grid dimensions
    max_N = max(Nq, Nk)
    grid_x = (max_N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M
    threads_per_tg = 128

    dispatch_kernel(
        lib,
        function_name="rope_fused_qk_proj_trellis",
        grid=(grid_x, grid_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
        buffers=[
            input_buf,
            Wq_buf, scales_q_buf, su_q_buf, sv_q_buf,
            Wk_buf, scales_k_buf, su_k_buf, sv_k_buf,
            grid_buf,
            cos_buf, sin_buf,
            out_q_buf, out_k_buf,
            M_buf, K_buf, Nq_buf, Nk_buf,
            bits_buf, n_levels_buf, group_size_buf,
            head_dim_buf, rope_dim_buf, pos_offset_buf, max_seq_buf,
        ],
        wait=True,
    )

    return out_q, out_k


def apply_rope_fused_linear(
    input: torch.Tensor,
    linear: Any,  # TrellisLinear
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    head_dim: int,
    rope_dim: int | None = None,
    position_offset: int = 0,
    lib: MetalKernelLibrary | None = None,
) -> torch.Tensor:
    """High-level API for fused linear projection + RoPE.

    This is a drop-in replacement for:
        output = linear(input)
        output = apply_rope(output, cos_cache, sin_cache, position_offset)

    Args:
        input: Input tensor [..., K] float16, MPS tensor
        linear: TrellisLinear layer
        cos_cache: RoPE cos cache [max_seq, rope_dim/2]
        sin_cache: RoPE sin cache [max_seq, rope_dim/2]
        head_dim: Head dimension for RoPE application
        rope_dim: RoPE dimension (None = head_dim)
        position_offset: Position offset for KV cache
        lib: Optional MetalKernelLibrary (created if not provided)

    Returns:
        Output tensor with RoPE applied [..., N]
    """
    if not HAS_METAL or not HAS_MPS:
        raise RuntimeError("Metal not available")

    if lib is None:
        lib = MetalKernelLibrary.from_source_dir()

    # Flatten batch dimensions
    batch_shape = input.shape[:-1]
    input_flat = input.reshape(-1, input.shape[-1])
    M = input_flat.shape[0]

    if rope_dim is None:
        rope_dim = head_dim

    # Check if weights are dequantized (FP16) or trellis-quantized
    if hasattr(linear, 'weight') and linear.weight is not None:
        # FP16 path - weights are materialized
        # weight shape is [out_features, in_features], need to transpose
        output = dispatch_rope_fused_proj_fp16(
            lib,
            input_flat,
            linear.weight,  # Already transposed if using nn.Linear convention
            cos_cache,
            sin_cache,
            head_dim,
            position_offset,
        )
    else:
        # Trellis-quantized path - need Q and K together for fused kernel
        # For single projection, we use a simplified approach
        # TODO: Implement single-projection trellis + RoPE kernel
        raise NotImplementedError(
            "Single trellis projection + RoPE not yet implemented. "
            "Use dispatch_rope_fused_qk_proj_trellis for Q/K together."
        )

    # Reshape back
    output = output.view(*batch_shape, linear.out_features)
    return output
