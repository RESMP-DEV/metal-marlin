"""
Metal kernel wrappers for Marlin GEMM and dequantization.

Provides Metal kernel interfaces to the fused dequant-GEMM kernels
(FP4 E2M1 and INT4 U4) and standalone dequant kernels.

This module uses PyObjC for direct Metal shader dispatch without MLX dependency.
All public APIs work with PyTorch MPS tensors.

Usage:
    from kernels import marlin_gemm_fp4, marlin_gemm_int4, dequant_fp4

    # FP4 fused dequant-GEMM (PyTorch MPS tensors)
    C = marlin_gemm_fp4(A, B_packed, scales, group_size=32)

    # INT4 fused dequant-GEMM with zero points
    C = marlin_gemm_int4(A, B_packed, scales, zeros, group_size=32)

    # Standalone FP4 dequantization
    W_fp16 = dequant_fp4(B_packed, scales, K, N, group_size=32)

Note:
    Metal kernel dispatch requires PyObjC and PyTorch MPS. When these are not
    available, all kernel functions will raise ImportError with installation
    instructions.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import numpy as np

from .metal_dispatch import (
    HAS_METAL,
    HAS_MPS,
    HAS_TORCH,
    MetalKernelLibrary,
    dispatch_kernel,
    get_default_library,
    mps_tensor_to_metal_buffer,
    require_mps,
)
from .utils.padding import pad_torch_2d, round_up

if HAS_TORCH:
    import torch

if TYPE_CHECKING:
    import torch


def _metal_required(*args: Any, **kwargs: Any) -> Any:
    """Stub function that raises ImportError when Metal/MPS is unavailable."""
    raise ImportError(
        "Metal kernel dispatch requires PyObjC and PyTorch MPS. "
        "Install with: pip install pyobjc-framework-Metal torch"
    )


# When Metal/MPS is unavailable, export stubs for all public kernel functions
if not HAS_METAL or not HAS_MPS:
    # GEMM kernels
    marlin_gemm_fp4 = _metal_required
    marlin_gemm_fused_fp4 = _metal_required
    marlin_gemm_fp4_tuned = _metal_required
    marlin_gemm_int4 = _metal_required
    marlin_gemm_fused_u4 = _metal_required

    # Dequantization kernels
    dequant_fp4 = _metal_required
    dequant_u4_standalone = _metal_required
    dequant_int2 = _metal_required
    dequant_int3 = _metal_required

    # Weight packing functions
    pack_fp4_weights = _metal_required
    pack_int2_weights = _metal_required
    pack_int3_weights = _metal_required

    # Flash attention
    flash_attention_kv_fp4 = _metal_required

    # MoE kernels
    moe_expert_gemm_fp4 = _metal_required
    moe_router_topk = _metal_required

    # Hadamard transform
    hadamard_transform = _metal_required

# Constants and Metal shader strings don't require Metal - defined unconditionally
# ---------------------------------------------------------------------------
# Tile / threadgroup constants (must match the kernel logic)
# ---------------------------------------------------------------------------

TILE_M = 64
TILE_N = 64
TILE_K = 32
SIMDGROUPS_PER_TG = 4
THREADS_PER_TG = SIMDGROUPS_PER_TG * 32  # 128
SG_M_TILES = 2
SG_N_TILES = 4
K_TILES = TILE_K // 8  # 4
FP4_PER_UINT = 8
NUM_BUFFERS = 2

# ---------------------------------------------------------------------------
# Kernel source: all helper functions, dequant primitives, tile logic
# ---------------------------------------------------------------------------

_GEMM_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// --- FP4 E2M1 branchless dequant (pure ALU, no LUT) ---
inline half dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.25h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

// NOTE: Uses float scale to work around Metal compiler bug where
// half parameters in inline functions have fractional parts rounded.
inline void dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    out[0] = (half)((float)dequant_fp4_scalar((packed >>  0) & 0xF) * fscale);
    out[1] = (half)((float)dequant_fp4_scalar((packed >>  4) & 0xF) * fscale);
    out[2] = (half)((float)dequant_fp4_scalar((packed >>  8) & 0xF) * fscale);
    out[3] = (half)((float)dequant_fp4_scalar((packed >> 12) & 0xF) * fscale);
    out[4] = (half)((float)dequant_fp4_scalar((packed >> 16) & 0xF) * fscale);
    out[5] = (half)((float)dequant_fp4_scalar((packed >> 20) & 0xF) * fscale);
    out[6] = (half)((float)dequant_fp4_scalar((packed >> 24) & 0xF) * fscale);
    out[7] = (half)((float)dequant_fp4_scalar((packed >> 28) & 0xF) * fscale);
}

// --- INT4 U4 magic-bias dequant ---
//
// Magic-bias trick for branchless INT4 -> FP16 conversion:
//   1. OR 4-bit nibble into FP16 mantissa of 1024.0 (0x6400)
//   2. Subtract 1024.0 to recover nibble as FP16 float
//   3. Apply asymmetric dequant: output = (nibble - zero_point) * scale
//
// FUSED_MAGIC_BIAS = 0x64006400 = two FP16 1024.0 values packed in uint32
// FUSED_LO_MASK = 0x000F000F = extract bits [3:0] from each 16-bit lane
//
// Nibble packing convention (LSB-first):
//   nibble i = (packed >> (i * 4)) & 0xF
//   This matches pack_u4_weights() in metal_marlin.py
//
// NOTE: Uses float intermediates to work around Metal compiler bug where
// half parameters in inline functions have fractional parts rounded.
// See docs/metal_half_precision_bug.md for details.
constant constexpr uint32_t FUSED_MAGIC_BIAS = 0x64006400u;
constant constexpr uint32_t FUSED_LO_MASK    = 0x000F000Fu;

inline void dequant_u4x8(uint32_t packed, half scale, half zero_point, thread half *out) {
    // Cast to float to avoid Metal compiler rounding bug with half in inline functions
    float fscale = (float)scale;
    float fzero = (float)zero_point;

    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);

    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;

    uint32_t n1 = ((packed >> 4u) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v1 = as_type<half2>(n1) - bias;

    uint32_t n2 = ((packed >> 8u) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v2 = as_type<half2>(n2) - bias;

    uint32_t n3 = ((packed >> 12u) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v3 = as_type<half2>(n3) - bias;

    out[0] = (half)(((float)v0.x - fzero) * fscale);
    out[1] = (half)(((float)v1.x - fzero) * fscale);
    out[2] = (half)(((float)v2.x - fzero) * fscale);
    out[3] = (half)(((float)v3.x - fzero) * fscale);
    out[4] = (half)(((float)v0.y - fzero) * fscale);
    out[5] = (half)(((float)v1.y - fzero) * fscale);
    out[6] = (half)(((float)v2.y - fzero) * fscale);
    out[7] = (half)(((float)v3.y - fzero) * fscale);
}

// --- INT2 (2-bit) dequantization: 16 weights per uint32 ---
// Codes 0,1,2,3 map to {-1.0, -0.333, 0.333, 1.0} before scaling.
// Formula: dequant = (code - 1.5) * 0.667 * scale

inline void dequant_int2x16(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    for (uint i = 0; i < 16; i++) {
        uint code = (packed >> (i * 2u)) & 0x3u;
        float dequant = (float(code) - 1.5f) * 0.6667f;
        out[i] = (half)(dequant * fscale);
    }
}

// Unrolled variant for better performance (no loop)
inline void dequant_int2x16_unrolled(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    #define DEQUANT_INT2(idx) { \\
        uint code = (packed >> ((idx) * 2u)) & 0x3u; \\
        out[idx] = (half)((float(code) - 1.5f) * 0.6667f * fscale); \\
    }
    DEQUANT_INT2(0);  DEQUANT_INT2(1);  DEQUANT_INT2(2);  DEQUANT_INT2(3);
    DEQUANT_INT2(4);  DEQUANT_INT2(5);  DEQUANT_INT2(6);  DEQUANT_INT2(7);
    DEQUANT_INT2(8);  DEQUANT_INT2(9);  DEQUANT_INT2(10); DEQUANT_INT2(11);
    DEQUANT_INT2(12); DEQUANT_INT2(13); DEQUANT_INT2(14); DEQUANT_INT2(15);
    #undef DEQUANT_INT2
}

// Dequant first 8 of 16 INT2 values (for GEMM tile alignment)
inline void dequant_int2x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    for (uint i = 0; i < 8; i++) {
        uint code = (packed >> (i * 2u)) & 0x3u;
        float dequant = (float(code) - 1.5f) * 0.6667f;
        out[i] = (half)(dequant * fscale);
    }
}

// --- INT3 (3-bit) dequantization: 10 weights per uint32 (30 bits used) ---
// Codes 0-7 map to symmetric [-1.0, 1.0] range.
// Formula: dequant = (code - 3.5) / 3.5 * scale ≈ (code - 3.5) * 0.2857 * scale

inline void dequant_int3x10(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    for (uint i = 0; i < 10; i++) {
        uint code = (packed >> (i * 3u)) & 0x7u;
        float dequant = (float(code) - 3.5f) * 0.2857f;
        out[i] = (half)(dequant * fscale);
    }
}

// Unrolled variant for better performance
inline void dequant_int3x10_unrolled(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    #define DEQUANT_INT3(idx) { \\
        uint code = (packed >> ((idx) * 3u)) & 0x7u; \\
        out[idx] = (half)((float(code) - 3.5f) * 0.2857f * fscale); \\
    }
    DEQUANT_INT3(0); DEQUANT_INT3(1); DEQUANT_INT3(2); DEQUANT_INT3(3); DEQUANT_INT3(4);
    DEQUANT_INT3(5); DEQUANT_INT3(6); DEQUANT_INT3(7); DEQUANT_INT3(8); DEQUANT_INT3(9);
    #undef DEQUANT_INT3
}

// Dequant first 8 of 10 INT3 values (for GEMM tile alignment)
inline void dequant_int3x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    for (uint i = 0; i < 8; i++) {
        uint code = (packed >> (i * 3u)) & 0x7u;
        float dequant = (float(code) - 3.5f) * 0.2857f;
        out[i] = (half)(dequant * fscale);
    }
}

// GemmParams struct for kernel parameters
struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint group_size;
};
"""

# ---------------------------------------------------------------------------
# FP4 fused dequant-GEMM kernel (complete kernel, not body)
# ---------------------------------------------------------------------------

_FP4_GEMM_KERNEL = (
    _GEMM_HEADER
    + """
kernel void marlin_gemm_fp4(
    device const half* A [[buffer(0)]],
    device const uint* B_packed [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device half* out [[buffer(3)]],
    device const GemmParams* params [[buffer(4)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params->M;
    const uint N = params->N;
    const uint K = params->K;
    const uint GROUP_SIZE = params->group_size;

    const uint TILE_M = 64;
    const uint TILE_N = 64;
    const uint TILE_K = 32;
    const uint SIMDGROUPS_PER_TG = 4;
    const uint SG_M_TILES = 2;
    const uint SG_N_TILES = 4;
    const uint FP4_PER_UINT = 8;
    const uint NUM_BUFFERS = 2;

    // Thread identity
    uint3 tgid = threadgroup_position_in_grid;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint thread_idx = simd_id * 32 + simd_lane;

    // Double-buffered threadgroup memory
    threadgroup half A_tiles[NUM_BUFFERS][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    // Tile assignment
    uint tg_row = tgid.y * TILE_M;
    uint tg_col = tgid.x * TILE_N;
    uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    // Accumulators
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint k_packs = K / FP4_PER_UINT;
    uint k_tiles_count = TILE_K / 8;

    // Main K-reduction loop
    for (uint k_block = 0; k_block < K; k_block += TILE_K) {

        // Cooperative A tile load
        {
            const uint elems_per_thread = (TILE_M * TILE_K) / (SIMDGROUPS_PER_TG * 32);
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = k_block + col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tiles[0][row][col] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner loop: fused dequant + simdgroup MMA
        for (uint kt = 0; kt < k_tiles_count; ++kt) {
            uint k_sub_base = k_block + kt * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / GROUP_SIZE;

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[0][sg_row_offset + mi * 8][kt * 8],
                               TILE_K);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    // Fused B dequant: lanes 0-7 each handle one column
                    if (simd_lane < 8) {
                        uint b_col = b_col_base + simd_lane;
                        half dequant_vals[8];

                        if (b_col < N && k_pack_idx < k_packs) {
                            uint32_t packed = B_packed[k_pack_idx * N + b_col];
                            half scale = scales[group_idx * N + b_col];
                            dequant_fp4x8(packed, scale, dequant_vals);
                        } else {
                            for (uint v = 0; v < 8; ++v)
                                dequant_vals[v] = half(0.0h);
                        }

                        for (uint row = 0; row < 8; ++row)
                            B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }

                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], out + out_row * N + out_col, N);
            } else if (out_row < M && out_col < N) {
                threadgroup half out_staging[8][8];
                simdgroup_store(acc[mi][ni], &out_staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < M && out_col + c < N) {
                        out[(out_row + r) * N + out_col + c] = out_staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
"""
)

# ---------------------------------------------------------------------------
# INT4 fused dequant-GEMM kernel (with zero points)
# ---------------------------------------------------------------------------

_INT4_GEMM_KERNEL = (
    _GEMM_HEADER
    + """
struct GemmParamsInt4 {
    uint M;
    uint N;
    uint K;
    uint group_size;
};

kernel void marlin_gemm_int4(
    device const half* A [[buffer(0)]],
    device const uint* B_packed [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* zeros [[buffer(3)]],
    device half* out [[buffer(4)]],
    device const GemmParamsInt4* params [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params->M;
    const uint N = params->N;
    const uint K = params->K;
    const uint GROUP_SIZE = params->group_size;

    const uint TILE_M = 64;
    const uint TILE_N = 64;
    const uint TILE_K = 32;
    const uint SIMDGROUPS_PER_TG = 4;
    const uint SG_M_TILES = 2;
    const uint SG_N_TILES = 4;
    const uint FP4_PER_UINT = 8;

    uint3 tgid = threadgroup_position_in_grid;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint thread_idx = simd_id * 32 + simd_lane;

    threadgroup half A_tiles[TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    uint tg_row = tgid.y * TILE_M;
    uint tg_col = tgid.x * TILE_N;
    uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);
    uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint k_packs = K / FP4_PER_UINT;
    uint k_tiles_count = TILE_K / 8;

    for (uint k_block = 0; k_block < K; k_block += TILE_K) {
        {
            const uint elems_per_thread = (TILE_M * TILE_K) / (SIMDGROUPS_PER_TG * 32);
            for (uint i = 0; i < elems_per_thread; ++i) {
                uint flat_idx = thread_idx * elems_per_thread + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = k_block + col;

                half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : half(0.0h);
                A_tiles[row][col] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < k_tiles_count; ++kt) {
            uint k_sub_base = k_block + kt * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / GROUP_SIZE;

            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[sg_row_offset + mi * 8][kt * 8],
                               TILE_K);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    if (simd_lane < 8) {
                        uint b_col = b_col_base + simd_lane;
                        half dequant_vals[8];

                        if (b_col < N && k_pack_idx < k_packs) {
                            uint32_t packed = B_packed[k_pack_idx * N + b_col];
                            half scale = scales[group_idx * N + b_col];
                            half zero = zeros[group_idx * N + b_col];
                            dequant_u4x8(packed, scale, zero, dequant_vals);
                        } else {
                            for (uint v = 0; v < 8; ++v)
                                dequant_vals[v] = half(0.0h);
                        }

                        for (uint row = 0; row < 8; ++row)
                            B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }

                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], out + out_row * N + out_col, N);
            } else if (out_row < M && out_col < N) {
                threadgroup half out_staging[8][8];
                simdgroup_store(acc[mi][ni], &out_staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < M && out_col + c < N) {
                        out[(out_row + r) * N + out_col + c] = out_staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
"""
)

# ---------------------------------------------------------------------------
# Standalone dequant kernel (FP4 -> FP16)
# ---------------------------------------------------------------------------

_DEQUANT_FP4_KERNEL = (
    _GEMM_HEADER
    + """
struct DequantParams {
    uint K;
    uint N;
    uint group_size;
};

kernel void dequant_fp4(
    device const uint* B_packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device half* out [[buffer(2)]],
    device const DequantParams* params [[buffer(3)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    const uint K = params->K;
    const uint N = params->N;
    const uint GROUP_SIZE = params->group_size;
    const uint FP4_PER_UINT = 8;

    // Grid: (N, K/8, 1) - each thread handles one packed uint32 (8 FP4 values)
    uint n_idx = thread_position_in_grid.x;
    uint k_block = thread_position_in_grid.y;

    if (n_idx >= N || k_block * 8u >= K) return;

    uint packed_idx = k_block * N + n_idx;
    uint32_t packed = B_packed[packed_idx];

    uint k_start = k_block * 8u;
    uint group_idx = k_start / GROUP_SIZE;
    half scale = scales[group_idx * N + n_idx];

    half vals[8];
    dequant_fp4x8(packed, scale, vals);

    uint out_base = k_start * N + n_idx;
    uint k_remain = min(8u, K - k_start);
    for (uint i = 0; i < k_remain; i++) {
        out[out_base + i * N] = vals[i];
    }
}
"""
)

# ---------------------------------------------------------------------------
# Decode GEMV kernel (optimized for M=1)
# ---------------------------------------------------------------------------

_DECODE_GEMV_FP4_KERNEL = (
    _GEMM_HEADER
    + """
struct DecodeParams {
    uint K;
    uint N;
    uint group_size;
};

kernel void decode_gemv_fp4(
    device const half* A [[buffer(0)]],
    device const uint* B [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device half* out [[buffer(3)]],
    device const DecodeParams* params [[buffer(4)]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    const uint K = params->K;
    const uint N = params->N;
    const uint GROUP_SIZE = params->group_size;

    const uint TILE_N = 256;
    const uint COLS_PER_THREAD = 2;
    const uint FP4_PER_UINT = 8;

    uint tgid = threadgroup_position_in_grid;
    uint tid = thread_position_in_threadgroup;

    uint col_base = tgid * TILE_N + tid * COLS_PER_THREAD;
    uint col0 = col_base;
    uint col1 = col_base + 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    uint k_packs = K / FP4_PER_UINT;

    // Stream through K dimension
    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        uint pack_idx = k_base / FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;

        // Load 8 A values (shared across all columns)
        half a_vals[8];
        for (uint i = 0; i < 8; i++) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        // Process column 0
        if (col0 < N && pack_idx < k_packs) {
            uint packed0 = B[pack_idx * N + col0];
            half scale0 = scales[group_idx * N + col0];
            float fscale0 = (float)scale0;

            for (uint i = 0; i < 8; i++) {
                if ((k_base + i) < K) {
                    uint nibble = (packed0 >> (i * 4)) & 0xF;
                    half w_val = (half)((float)dequant_fp4_scalar(nibble) * fscale0);
                    acc0 += float(a_vals[i]) * float(w_val);
                }
            }
        }

        // Process column 1
        if (col1 < N && pack_idx < k_packs) {
            uint packed1 = B[pack_idx * N + col1];
            half scale1 = scales[group_idx * N + col1];
            float fscale1 = (float)scale1;

            for (uint i = 0; i < 8; i++) {
                if ((k_base + i) < K) {
                    uint nibble = (packed1 >> (i * 4)) & 0xF;
                    half w_val = (half)((float)dequant_fp4_scalar(nibble) * fscale1);
                    acc1 += float(a_vals[i]) * float(w_val);
                }
            }
        }
    }

    // Store results
    if (col0 < N) {
        out[col0] = half(acc0);
    }
    if (col1 < N) {
        out[col1] = half(acc1);
    }
}
"""
)

# ---------------------------------------------------------------------------
# Hadamard Transform kernels
# ---------------------------------------------------------------------------

_HADAMARD_KERNEL = """
#include <metal_stdlib>
using namespace metal;

/// Single butterfly step: computes (a + b, a - b) where b is shuffled from another lane.
inline float butterfly_step_f(float val, uint partner, bool is_upper) {
    float partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (val - partner_val) : (val + partner_val);
}

/// Butterfly step for float2 (two parallel values per thread)
inline float2 butterfly_step2_f(float2 val, uint partner, bool is_upper) {
    float2 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (val - partner_val) : (val + partner_val);
}

/// Butterfly step for float4 (four parallel values per thread)
inline float4 butterfly_step4_f(float4 val, uint partner, bool is_upper) {
    float4 partner_val = simd_shuffle(val, ushort(partner));
    return is_upper ? (val - partner_val) : (val + partner_val);
}

struct HadamardParams {
    uint N;
    uint normalize;
};

kernel void hadamard_32(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const HadamardParams* params [[buffer(2)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid;
    const uint N = params->N;
    const bool NORMALIZE = params->normalize != 0;

    if (tg_idx >= N) return;

    uint base = tg_idx * 32;
    float val = float(input[base + lane_id]);

    // 5 butterfly stages for size 32
    for (uint stage = 0; stage < 5; ++stage) {
        uint stride = 1u << stage;
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step_f(val, partner, is_upper);
    }

    // Normalize by 1/sqrt(32)
    if (NORMALIZE) {
        val *= 0.1767766953f;
    }

    out[base + lane_id] = half(val);
}

kernel void hadamard_64(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const HadamardParams* params [[buffer(2)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid;
    const uint N = params->N;
    const bool NORMALIZE = params->normalize != 0;

    if (tg_idx >= N) return;

    uint base = tg_idx * 64;
    float2 val;
    val.x = float(input[base + lane_id * 2]);
    val.y = float(input[base + lane_id * 2 + 1]);

    // Stage 0: stride 1 (within float2)
    {
        float sum = val.x + val.y;
        float diff = val.x - val.y;
        val.x = sum;
        val.y = diff;
    }

    // Stages 1-5: inter-lane shuffles
    for (uint stage = 1; stage < 6; ++stage) {
        uint stride = 1u << (stage - 1);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step2_f(val, partner, is_upper);
    }

    // Normalize by 1/sqrt(64) = 0.125
    if (NORMALIZE) {
        val *= 0.125f;
    }

    out[base + lane_id * 2] = half(val.x);
    out[base + lane_id * 2 + 1] = half(val.y);
}

kernel void hadamard_128(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const HadamardParams* params [[buffer(2)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid;
    const uint N = params->N;
    const bool NORMALIZE = params->normalize != 0;

    if (tg_idx >= N) return;

    uint base = tg_idx * 128;
    float4 val;
    val.x = float(input[base + lane_id * 4]);
    val.y = float(input[base + lane_id * 4 + 1]);
    val.z = float(input[base + lane_id * 4 + 2]);
    val.w = float(input[base + lane_id * 4 + 3]);

    // Stage 0: stride 1 (pairs within float4)
    {
        float sum0 = val.x + val.y;
        float diff0 = val.x - val.y;
        float sum1 = val.z + val.w;
        float diff1 = val.z - val.w;
        val = float4(sum0, diff0, sum1, diff1);
    }

    // Stage 1: stride 2 (swap pairs within float4)
    {
        float sum0 = val.x + val.z;
        float diff0 = val.x - val.z;
        float sum1 = val.y + val.w;
        float diff1 = val.y - val.w;
        val = float4(sum0, sum1, diff0, diff1);
    }

    // Stages 2-6: inter-lane shuffles
    for (uint stage = 2; stage < 7; ++stage) {
        uint stride = 1u << (stage - 2);
        uint partner = lane_id ^ stride;
        bool is_upper = (lane_id & stride) != 0;
        val = butterfly_step4_f(val, partner, is_upper);
    }

    // Normalize by 1/sqrt(128)
    if (NORMALIZE) {
        val *= 0.0883883476f;
    }

    out[base + lane_id * 4] = half(val.x);
    out[base + lane_id * 4 + 1] = half(val.y);
    out[base + lane_id * 4 + 2] = half(val.z);
    out[base + lane_id * 4 + 3] = half(val.w);
}
"""

# ---------------------------------------------------------------------------
# Kernel library management
# ---------------------------------------------------------------------------

_compiled_libraries: dict[str, bool] = {}


def _ensure_kernel_compiled(lib: MetalKernelLibrary, name: str, source: str) -> None:
    """Ensure a kernel source is compiled in the library."""
    if name not in _compiled_libraries:
        lib.compile_source(name, source)
        _compiled_libraries[name] = True


# ---------------------------------------------------------------------------
# INT2 and INT3 constants
# ---------------------------------------------------------------------------

INT2_PER_UINT = 16  # 32 bits / 2 bits = 16 values per uint32
INT3_PER_UINT = 10  # 30 bits used / 3 bits = 10 values per uint32 (2 bits unused)


# ---------------------------------------------------------------------------
# Public API (only defined when Metal/MPS is available)
# ---------------------------------------------------------------------------

if HAS_METAL and HAS_MPS:
    import Foundation
    import Metal

    from ._buffer_pool import MetalBufferPool

    _STAGING_POOLS: dict[int, MetalBufferPool] = {}
    _WEIGHT_BUFFER_CACHE: dict[int, Any] = {}  # data_ptr -> MTLBuffer

    class MarlinGemmDispatcher:
        """Compile and dispatch dense GEMM kernels with dtype-aware selection."""

        def __init__(self, lib: MetalKernelLibrary | None = None) -> None:
            require_mps()
            self._lib = lib or get_default_library()
            self._compiled: dict[tuple[str, tuple[str, ...] | None], Any | None] = {}
            self._gemm_fp16: Any | None = None
            self._gemm_bf16_fp32acc: Any | None = None
            self._compile_gemm_kernels()

        @staticmethod
        def _apply_compile_options(source: str, compile_options: list[str] | None) -> str:
            if not compile_options:
                return source
            defines: list[str] = []
            for opt in compile_options:
                if not opt.startswith("-D"):
                    continue
                define = opt[2:]
                if "=" in define:
                    name, value = define.split("=", 1)
                else:
                    name, value = define, "1"
                defines.append(f"#define {name} {value}")
            if not defines:
                return source
            return "\n".join(defines) + "\n" + source

        def _compile_kernel(
            self,
            function_name: str,
            *,
            source_name: str = "marlin_gemm",
            compile_options: list[str] | None = None,
        ) -> Any | None:
            key = (function_name, tuple(compile_options) if compile_options else None)
            if key in self._compiled:
                return self._compiled[key]

            try:
                if compile_options:
                    source = get_shader_source(source_name)
                    source = self._apply_compile_options(source, compile_options)
                    tag = "_".join(
                        opt.replace("-", "").replace("=", "_") for opt in compile_options
                    )
                    library_name = f"{source_name}:{function_name}:{tag}"
                    self._lib.compile_source(library_name, source)
                    pipeline = self._lib.get_pipeline(function_name, library_name)
                else:
                    try:
                        pipeline = self._lib.get_pipeline(function_name, source_name)
                    except KeyError:
                        source = get_shader_source(source_name)
                        self._lib.compile_source(source_name, source)
                        pipeline = self._lib.get_pipeline(function_name, source_name)
            except Exception:
                pipeline = None

            self._compiled[key] = pipeline
            return pipeline

        def _compile_gemm_kernels(self) -> None:
            self._gemm_fp16 = self._compile_kernel("marlin_gemm_fp16")
            if self._gemm_fp16 is None:
                self._gemm_fp16 = self._compile_kernel("marlin_gemm_fp16_pipelined")
            self._gemm_bf16_fp32acc = self._compile_kernel(
                "marlin_gemm_fp32acc",
                compile_options=["-DUSE_BF16_INPUTS=1"],
            )

        def dispatch_gemm(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            M: int,
            N: int,
            K: int,
            *,
            activation_dtype: str = "fp16",
        ) -> torch.Tensor:
            if activation_dtype == "bf16" and self._gemm_bf16_fp32acc is not None:
                return self._dispatch_bf16_path(A, B, M, N, K)
            return self._dispatch_fp16_path(A, B, M, N, K)

        def _dispatch_fp16_path(
            self, A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int
        ) -> torch.Tensor:
            return self._dispatch_kernel(self._gemm_fp16, A, B, M, N, K, torch.float16)

        def _dispatch_bf16_path(
            self, A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int
        ) -> torch.Tensor:
            return self._dispatch_kernel(self._gemm_bf16_fp32acc, A, B, M, N, K, torch.bfloat16)

        def _dispatch_kernel(
            self,
            kernel: Any | None,
            A: torch.Tensor,
            B: torch.Tensor,
            M: int,
            N: int,
            K: int,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            if kernel is None:
                raise RuntimeError("GEMM kernel pipeline is unavailable.")

            A_contig = A.to(dtype=dtype).contiguous()
            B_contig = B.to(dtype=dtype).contiguous()
            output = torch.empty((M, N), dtype=dtype, device="mps")

            device = self._lib.device
            A_buf = _private_buffer_from_tensor(A_contig, self._lib, device, cache=False)
            B_buf = _private_buffer_from_tensor(B_contig, self._lib, device, cache=True)
            C_buf = mps_tensor_to_metal_buffer(output, device)

            grid_m = (M + TILE_M - 1) // TILE_M
            grid_n = (N + TILE_N - 1) // TILE_N

            self._lib._dispatch(
                kernel,
                (grid_n, grid_m, 1),
                (THREADS_PER_TG, 1, 1),
                A_buf,
                B_buf,
                C_buf,
                M,
                N,
                K,
            )
            return output

    def _get_staging_pool(device: Any) -> MetalBufferPool:
        pool = _STAGING_POOLS.get(id(device))
        if pool is None:
            pool = MetalBufferPool(device, storage_mode=Metal.MTLResourceStorageModeManaged)
            _STAGING_POOLS[id(device)] = pool
        return pool

    def _blit_copy(lib: MetalKernelLibrary, source: Any, destination: Any, size: int) -> None:
        command_buffer = lib.command_queue.commandBuffer()
        blit = command_buffer.blitCommandEncoder()
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source, 0, destination, 0, size
        )
        blit.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _private_buffer_from_bytes(lib: MetalKernelLibrary, device: Any, data: bytes) -> Any:
        size = len(data)
        staging = _get_staging_pool(device).get(size)
        contents = staging.contents()
        view = memoryview(contents.as_buffer(staging.length()))
        view[:size] = data
        staging.didModifyRange_(Foundation.NSMakeRange(0, size))

        private_buf = device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModePrivate)
        _blit_copy(lib, staging, private_buf, size)
        _get_staging_pool(device).release(staging)
        return private_buf

    def _private_buffer_from_tensor(
        tensor: torch.Tensor,
        lib: MetalKernelLibrary,
        device: Any,
        *,
        cache: bool,
    ) -> Any:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        cache_key = tensor.data_ptr()
        if cache and cache_key in _WEIGHT_BUFFER_CACHE:
            return _WEIGHT_BUFFER_CACHE[cache_key]

        if tensor.is_mps:
            staging = mps_tensor_to_metal_buffer(tensor, device)
            size = staging.length()
            private_buf = device.newBufferWithLength_options_(
                size, Metal.MTLResourceStorageModePrivate
            )
            _blit_copy(lib, staging, private_buf, size)
        else:
            data = tensor.detach().cpu().numpy().tobytes()
            private_buf = _private_buffer_from_bytes(lib, device, data)

        if cache:
            _WEIGHT_BUFFER_CACHE[cache_key] = private_buf
        return private_buf

    def _params_buffer(lib: MetalKernelLibrary, device: Any, params: np.ndarray) -> Any:
        return _private_buffer_from_bytes(lib, device, params.tobytes())

    def pack_fp4_weights(
        weight: torch.Tensor,
        group_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pack FP16/BF16 weights into FP4 E2M1 format for the fused GEMM kernels.

        Weight layout: [K/8, N] as uint32, where each uint32 holds 8 FP4 values
        along the K dimension at a given output column. This matches the fused
        kernel's access pattern: B_packed[k_pack_idx * N + col].

        Scales layout: [K/group_size, N] with dtype float16.

        Args:
            weight: Weight matrix [N, K] (PyTorch convention: out_features first).
                    Transposed internally to [K, N] for the kernel layout.
            group_size: Number of K-dimension elements per quantization group.

        Returns:
            Tuple of (weight_packed [K/8, N] uint32, scales [K/group_size, N]).
        """
        w = weight.T.to(torch.float16)  # [K, N]
        K, N = w.shape

        if K % FP4_PER_UINT != 0:
            raise ValueError(f"K ({K}) must be divisible by {FP4_PER_UINT}")
        if K % group_size != 0:
            raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

        # Per-group scales along K: max abs value in each group
        w_grouped = w.reshape(K // group_size, group_size, N)
        scales = w_grouped.abs().amax(dim=1)
        scales = scales.clamp(min=1e-7)

        # E2M1 representable values (positive): 0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        # Max magnitude = 6.0
        MAX_E2M1 = 6.0

        # Normalize by group scale and clamp to E2M1 range
        scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]
        w_norm = w / scales_expanded
        w_norm = w_norm.clamp(-MAX_E2M1, MAX_E2M1)

        # Build E2M1 lookup table for nearest-value quantization
        e2m1_lut = np.zeros(16, dtype=np.float32)
        for nibble in range(16):
            s = (nibble >> 3) & 1
            e = (nibble >> 1) & 3
            m = nibble & 1
            if e == 0 and m == 0:
                val = 0.0
            elif e == 0 and m == 1:
                val = 0.25  # subnormal: m * 0.25
            else:
                val = (1.0 + m * 0.5) * (2.0 ** (e - 1))
            e2m1_lut[nibble] = -val if s else val

        # Quantize to nearest E2M1 nibble
        w_np = w_norm.cpu().float().numpy()
        k_packs = K // FP4_PER_UINT
        packed = np.zeros((k_packs, N), dtype=np.uint32)

        for k_pack in range(k_packs):
            k_base = k_pack * FP4_PER_UINT
            for bit_pos in range(FP4_PER_UINT):
                row_vals = w_np[k_base + bit_pos, :]  # [N]
                # Find nearest E2M1 nibble for each element
                dists = np.abs(row_vals[:, None] - e2m1_lut[None, :])  # [N, 16]
                nibbles = np.argmin(dists, axis=1).astype(np.uint32)  # [N]
                packed[k_pack, :] |= nibbles << (bit_pos * 4)

        weight_packed = torch.from_numpy(packed).to("mps")
        scales_out = scales.to("mps")
        return weight_packed, scales_out

    def pack_int2_weights(
        weight: torch.Tensor,
        group_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pack FP16/BF16 weights into INT2 format.

        Weight layout: [K/16, N] as uint32, where each uint32 holds 16 INT2 values.

        Args:
            weight: Weight matrix [N, K].
            group_size: Number of K-dimension elements per quantization group.

        Returns:
            Tuple of (weight_packed [K/16, N] uint32, scales [K/group_size, N]).
        """
        w = weight.T.to(torch.float16)  # [K, N]
        K, N = w.shape

        if K % INT2_PER_UINT != 0:
            raise ValueError(f"K ({K}) must be divisible by {INT2_PER_UINT}")
        if K % group_size != 0:
            raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

        # Per-group scales along K
        w_grouped = w.reshape(K // group_size, group_size, N)
        scales = w_grouped.abs().amax(dim=1)
        scales = scales.clamp(min=1e-7)

        # Normalize and clip
        scales_expanded = scales.repeat_interleave(group_size, dim=0)
        w_norm = w / scales_expanded
        w_norm = w_norm.clamp(-1.0, 1.0)

        # INT2 codebook: {-1.0, -0.333, 0.333, 1.0} -> codes {0, 1, 2, 3}
        int2_lut = np.array([-1.0, -0.333, 0.333, 1.0], dtype=np.float32)

        w_np = w_norm.cpu().float().numpy()
        k_packs = K // INT2_PER_UINT
        packed = np.zeros((k_packs, N), dtype=np.uint32)

        for k_pack in range(k_packs):
            k_base = k_pack * INT2_PER_UINT
            for bit_pos in range(INT2_PER_UINT):
                row_vals = w_np[k_base + bit_pos, :]
                dists = np.abs(row_vals[:, None] - int2_lut[None, :])
                codes = np.argmin(dists, axis=1).astype(np.uint32)
                packed[k_pack, :] |= codes << (bit_pos * 2)

        weight_packed = torch.from_numpy(packed).to("mps")
        scales_out = scales.to("mps")
        return weight_packed, scales_out

    def pack_int3_weights(
        weight: torch.Tensor,
        group_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pack FP16/BF16 weights into INT3 format.

        Weight layout: [K/10, N] as uint32, where each uint32 holds 10 INT3 values.

        Args:
            weight: Weight matrix [N, K].
            group_size: Number of K-dimension elements per quantization group.

        Returns:
            Tuple of (weight_packed [K/10, N] uint32, scales [K/group_size, N]).
        """
        w = weight.T.to(torch.float16)  # [K, N]
        K, N = w.shape

        if K % INT3_PER_UINT != 0:
            raise ValueError(f"K ({K}) must be divisible by {INT3_PER_UINT}")
        if K % group_size != 0:
            raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

        # Per-group scales
        w_grouped = w.reshape(K // group_size, group_size, N)
        scales = w_grouped.abs().amax(dim=1)
        scales = scales.clamp(min=1e-7)

        # Normalize and clip
        scales_expanded = scales.repeat_interleave(group_size, dim=0)
        w_norm = w / scales_expanded
        w_norm = w_norm.clamp(-1.0, 1.0)

        # INT3 codebook: 8 levels from -1.0 to 1.0
        int3_lut = np.array([(i - 3.5) / 3.5 for i in range(8)], dtype=np.float32)

        w_np = w_norm.cpu().float().numpy()
        k_packs = K // INT3_PER_UINT
        packed = np.zeros((k_packs, N), dtype=np.uint32)

        for k_pack in range(k_packs):
            k_base = k_pack * INT3_PER_UINT
            for bit_pos in range(INT3_PER_UINT):
                row_vals = w_np[k_base + bit_pos, :]
                dists = np.abs(row_vals[:, None] - int3_lut[None, :])
                codes = np.argmin(dists, axis=1).astype(np.uint32)
                packed[k_pack, :] |= codes << (bit_pos * 3)

        weight_packed = torch.from_numpy(packed).to("mps")
        scales_out = scales.to("mps")
        return weight_packed, scales_out

    def dispatch_gemm_fp4(
        lib: MetalKernelLibrary,
        A: torch.Tensor,
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        M: int,
        N: int,
        K: int,
        group_size: int,
    ) -> torch.Tensor:
        """Dispatch the FP4 fused dequant-GEMM Metal kernel.

        Args:
            lib: Metal kernel library with compiled marlin_gemm_fp4.
            A: Input activations [M, K] float16.
            B_packed: Packed FP4 weights [K/8, N] uint32.
            scales: Per-group scales [K/group_size, N] float16.
            M: Batch dimension (rows of A).
            N: Output features (cols of B).
            K: Reduction dimension.
            group_size: Quantization group size.

        Returns:
            Output tensor [M, N] float16.
        """
        device = lib.device

        # Ensure contiguous and correct dtype
        A_contig = A.contiguous()
        B_packed_contig = B_packed.contiguous()
        scales_half = (
            scales.half().contiguous() if scales.dtype != torch.float16 else scales.contiguous()
        )

        # Create output tensor
        C = torch.empty((M, N), dtype=torch.float16, device="mps")

        # Get Metal buffers
        A_buf = _private_buffer_from_tensor(A_contig, lib, device, cache=True)
        B_buf = _private_buffer_from_tensor(B_packed_contig, lib, device, cache=True)
        S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
        C_buf = mps_tensor_to_metal_buffer(C, device, copy_back=True)

        # Create separate param buffers (kernel expects buffers at indices 4, 5, 6, 7)
        M_buf = _params_buffer(lib, device, np.array([M], dtype=np.uint32))
        N_buf = _params_buffer(lib, device, np.array([N], dtype=np.uint32))
        K_buf = _params_buffer(lib, device, np.array([K], dtype=np.uint32))
        gs_buf = _params_buffer(lib, device, np.array([group_size], dtype=np.uint32))

        # Tile sizes matching marlin_gemm.metal
        TILE_M = 64
        TILE_N = 64
        THREADS_PER_TG = 128

        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        dispatch_kernel(
            lib,
            function_name="marlin_gemm_fused_fp4",
            grid=(grid_n, grid_m, 1),
            threadgroup=(THREADS_PER_TG, 1, 1),
            buffers=[A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf],
            wait=True,
        )

        return C

    def marlin_gemm_fp4(
        A: torch.Tensor,
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        """
        FP4 fused dequant-GEMM: C = A @ dequant(B_packed, scales).

        Uses simdgroup_multiply_accumulate with fused FP4 E2M1 dequantization
        in registers. No intermediate FP16 weight materialization in threadgroup
        memory; each simdgroup dequantizes only the 8x8 sub-tile it needs.

        Args:
            A: Input activations [*, K]. Arbitrary leading dims. MPS tensor.
            B_packed: Packed FP4 weights [K/8, N] as uint32. MPS tensor.
            scales: Per-group scales [K/group_size, N]. MPS tensor.
            group_size: Elements per quantization group (must divide K).

        Returns:
            Output [*, N] as float16 MPS tensor.
        """
        require_mps()

        lib = get_default_library()

        orig_shape = A.shape
        K = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d

        A_2d = A.reshape(M, K).half().contiguous()
        N = B_packed.shape[1]

        M_dispatch = round_up(M, 8)
        if M_dispatch != M:
            A_2d, _ = pad_torch_2d(A_2d, rows_multiple=8, cols_multiple=1)

        C = dispatch_gemm_fp4(lib, A_2d, B_packed, scales, M_dispatch, N, K, group_size)
        if M_dispatch != M:
            C = C[:M, :]
        out_shape = list(orig_shape[:-1]) + [N]
        return C.reshape(out_shape)

    def marlin_gemm_fp4_tuned(
        A: torch.Tensor,
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        """FP4 fused dequant-GEMM (alias for marlin_gemm_fp4)."""
        return marlin_gemm_fp4(A, B_packed, scales, group_size)

    def marlin_gemm_fused_fp4(
        A: torch.Tensor,
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        """Alias for marlin_gemm_fp4; preserves fused kernel naming."""
        return marlin_gemm_fp4(A, B_packed, scales, group_size)

    def marlin_gemm_int4(
        A: torch.Tensor,
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        """
        INT4 fused dequant-GEMM with asymmetric quantization (scale + zero point).

        Args:
            A: Input activations [*, K]. MPS tensor.
            B_packed: Packed INT4 weights [K/8, N] as uint32. MPS tensor.
            scales: Per-group scales [K/group_size, N]. MPS tensor.
            zeros: Per-group zero points [K/group_size, N]. MPS tensor.
            group_size: Elements per quantization group.

        Returns:
            Output [*, N] as float16 MPS tensor.
        """
        require_mps()

        lib = get_default_library()
        _ensure_kernel_compiled(lib, "int4_gemm", _INT4_GEMM_KERNEL)

        orig_shape = A.shape
        K = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d

        A_2d = A.reshape(M, K).half().contiguous()
        N = B_packed.shape[1]

        M_dispatch = round_up(M, 8)
        if M_dispatch != M:
            A_2d, _ = pad_torch_2d(A_2d, rows_multiple=8, cols_multiple=1)

        C = torch.empty((M_dispatch, N), dtype=torch.float16, device="mps")

        device = lib.device

        A_buf = _private_buffer_from_tensor(A_2d, lib, device, cache=False)
        B_packed_contig = B_packed.contiguous()
        B_buf = _private_buffer_from_tensor(B_packed_contig, lib, device, cache=True)
        scales_half = scales if scales.dtype == torch.float16 else scales.half()
        scales_half = scales_half.contiguous()
        S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
        zeros_half = zeros if zeros.dtype == torch.float16 else zeros.half()
        zeros_half = zeros_half.contiguous()
        Z_buf = _private_buffer_from_tensor(zeros_half, lib, device, cache=True)
        C_buf = mps_tensor_to_metal_buffer(C, device)

        params = np.array([M_dispatch, N, K, group_size], dtype=np.uint32)
        params_buf = _params_buffer(lib, device, params)

        grid_m = (M_dispatch + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        dispatch_kernel(
            lib,
            function_name="marlin_gemm_int4",
            grid=(grid_n, grid_m, 1),
            threadgroup=(THREADS_PER_TG, 1, 1),
            buffers=[A_buf, B_buf, S_buf, Z_buf, C_buf, params_buf],
            wait=True,
        )

        if M_dispatch != M:
            C = C[:M, :]

        out_shape = list(orig_shape[:-1]) + [N]
        return C.reshape(out_shape)

    def marlin_gemm_fused_u4(
        A: torch.Tensor,
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        """Alias for marlin_gemm_int4; preserves fused kernel naming."""
        return marlin_gemm_int4(A, B_packed, scales, zeros, group_size)

    def dequant_fp4(
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        K: int,
        N: int,
        group_size: int = 32,
    ) -> torch.Tensor:
        """
        Standalone FP4 dequantization: unpack FP4 weights to float.

        Args:
            B_packed: Packed FP4 weights [K/8, N] as uint32. MPS tensor.
            scales: Per-group scales [K/group_size, N]. MPS tensor.
            K: Reduction dimension size.
            N: Output feature dimension size.
            group_size: Elements per quantization group.

        Returns:
            Dequantized weights [K, N] as float16 MPS tensor.
        """
        require_mps()

        lib = get_default_library()
        _ensure_kernel_compiled(lib, "dequant_fp4", _DEQUANT_FP4_KERNEL)

        k_blocks = (K + 7) // 8

        out = torch.empty((K, N), dtype=torch.float16, device="mps")

        device = lib.device

        B_packed_contig = B_packed.contiguous()
        B_buf = _private_buffer_from_tensor(B_packed_contig, lib, device, cache=True)
        scales_half = scales if scales.dtype == torch.float16 else scales.half()
        scales_half = scales_half.contiguous()
        S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
        out_buf = mps_tensor_to_metal_buffer(out, device)

        params = np.array([K, N, group_size], dtype=np.uint32)
        params_buf = _params_buffer(lib, device, params)

        dispatch_kernel(
            lib,
            function_name="dequant_fp4",
            grid=(N, k_blocks, 1),
            threadgroup=(256, 1, 1),
            buffers=[B_buf, S_buf, out_buf, params_buf],
            wait=True,
        )

        return out

    def dequant_u4_standalone(packed: torch.Tensor) -> torch.Tensor:
        """
        Standalone U4 dequantization: unpack U4 nibbles to raw float integer values.

        Args:
            packed: 1-D tensor of uint8 values, each in [0, 15]. Length must be
                    a multiple of 8.

        Returns:
            1-D tensor of same length with float16, each element = float(input_element).
        """
        # Simple CPU implementation for debugging
        packed_np = packed.cpu().numpy().astype(np.uint8)
        out_np = packed_np.astype(np.float16)
        return torch.from_numpy(out_np).to("mps")

    def dequant_int2(
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        K: int,
        N: int,
        group_size: int = 32,
    ) -> torch.Tensor:
        """
        Standalone INT2 dequantization: unpack INT2 weights to float.

        Args:
            B_packed: Packed INT2 weights [K/16, N] as uint32. MPS tensor.
            scales: Per-group scales [K/group_size, N]. MPS tensor.
            K: Reduction dimension size.
            N: Output feature dimension size.
            group_size: Elements per quantization group.

        Returns:
            Dequantized weights [K, N] as float16 MPS tensor.
        """
        # CPU fallback implementation
        B_np = B_packed.cpu().numpy()
        S_np = scales.cpu().numpy()

        out = np.zeros((K, N), dtype=np.float16)
        int2_vals = np.array([-1.0, -0.333, 0.333, 1.0], dtype=np.float32)

        k_packs = K // INT2_PER_UINT
        for k_pack in range(k_packs):
            k_base = k_pack * INT2_PER_UINT
            group_idx = k_base // group_size
            for bit_pos in range(INT2_PER_UINT):
                for n in range(N):
                    code = (B_np[k_pack, n] >> (bit_pos * 2)) & 0x3
                    scale = S_np[group_idx, n]
                    out[k_base + bit_pos, n] = int2_vals[code] * scale

        return torch.from_numpy(out).to("mps")

    def dequant_int3(
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        K: int,
        N: int,
        group_size: int = 32,
    ) -> torch.Tensor:
        """
        Standalone INT3 dequantization: unpack INT3 weights to float.

        Args:
            B_packed: Packed INT3 weights [K/10, N] as uint32. MPS tensor.
            scales: Per-group scales [K/group_size, N]. MPS tensor.
            K: Reduction dimension size.
            N: Output feature dimension size.
            group_size: Elements per quantization group.

        Returns:
            Dequantized weights [K, N] as float16 MPS tensor.
        """
        # CPU fallback implementation
        B_np = B_packed.cpu().numpy()
        S_np = scales.cpu().numpy()

        out = np.zeros((K, N), dtype=np.float16)

        k_packs = K // INT3_PER_UINT
        for k_pack in range(k_packs):
            k_base = k_pack * INT3_PER_UINT
            group_idx = k_base // group_size
            for bit_pos in range(INT3_PER_UINT):
                for n in range(N):
                    code = (B_np[k_pack, n] >> (bit_pos * 3)) & 0x7
                    scale = S_np[group_idx, n]
                    dequant = (code - 3.5) / 3.5
                    out[k_base + bit_pos, n] = dequant * scale

        return torch.from_numpy(out).to("mps")

    def flash_attention_kv_fp4(
        Q: torch.Tensor,
        K_packed: torch.Tensor,
        V_packed: torch.Tensor,
        K_scales: torch.Tensor,
        V_scales: torch.Tensor,
        scale: float,
        num_heads_q: int | None = None,
        num_heads_k: int | None = None,
    ) -> torch.Tensor:
        """
        Flash Attention with FP4-quantized KV cache.

        Note: This is a placeholder that uses CPU fallback.
        Full Metal implementation would require additional kernel development.

        Args:
            Q: Query tensor [batch, num_heads_q, seq_q, head_dim]. MPS tensor.
            K_packed: Packed FP4 keys [batch, num_heads_k, seq_k, head_dim//8]. MPS tensor.
            V_packed: Packed FP4 values [batch, num_heads_k, seq_k, head_dim//8]. MPS tensor.
            K_scales: Per-row key scales [batch, num_heads_k, seq_k, 1]. MPS tensor.
            V_scales: Per-row value scales [batch, num_heads_k, seq_k, 1]. MPS tensor.
            scale: Attention scale factor.
            num_heads_q: Number of query heads.
            num_heads_k: Number of KV heads.

        Returns:
            Output tensor [batch, num_heads_q, seq_q, head_dim]. MPS tensor.
        """
        raise NotImplementedError(
            "Flash attention with FP4 KV cache requires full kernel implementation. "
            "Use standard attention with dequantized KV as a fallback."
        )

    def moe_expert_gemm_fp4(
        activations: torch.Tensor,
        expert_weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_probs: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """
        MoE expert GEMM with FP4-quantized expert weights.

        Note: This is a placeholder. Full implementation requires additional kernels.

        Args:
            activations: Input activations [batch, hidden_in]. MPS tensor.
            expert_weights: Packed FP4 expert weights. MPS tensor.
            scales: Per-group scales. MPS tensor.
            expert_ids: Expert indices [batch, top_k]. MPS tensor.
            expert_probs: Expert probabilities [batch, top_k]. MPS tensor.
            group_size: Elements per quantization group.

        Returns:
            Output tensor [batch, hidden_out]. MPS tensor.
        """
        raise NotImplementedError("MoE expert GEMM requires full kernel implementation.")

    def moe_router_topk(
        hidden: torch.Tensor,
        router_weights: torch.Tensor,
        top_k: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MoE router with top-k expert selection.

        Note: This is a placeholder. Full implementation requires additional kernels.

        Args:
            hidden: Hidden states [batch, hidden_dim]. MPS tensor.
            router_weights: Router weight matrix [hidden_dim, num_experts]. MPS tensor.
            top_k: Number of experts to select per token.

        Returns:
            Tuple of (expert_ids [batch, top_k], expert_probs [batch, top_k]).
        """
        raise NotImplementedError("MoE router requires full kernel implementation.")

    def hadamard_transform(
        x: torch.Tensor,
        block_size: int = 64,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Apply Walsh-Hadamard transform to input vectors.

        Args:
            x: Input tensor [..., block_size]. MPS tensor.
            block_size: Size of each Hadamard block. Must be 32, 64, or 128.
            normalize: If True, normalize by 1/sqrt(block_size) after transform.

        Returns:
            Transformed tensor with same shape as input. MPS tensor.
        """
        require_mps()

        if block_size not in (32, 64, 128):
            raise ValueError(f"block_size must be 32, 64, or 128, got {block_size}")

        if x.shape[-1] != block_size:
            raise ValueError(
                f"Last dimension of x ({x.shape[-1]}) must equal block_size ({block_size})"
            )

        lib = get_default_library()
        _ensure_kernel_compiled(lib, "hadamard", _HADAMARD_KERNEL)

        # Flatten to 2D: [N, block_size]
        orig_shape = x.shape
        n = 1
        for d in orig_shape[:-1]:
            n *= d

        x_2d = x.reshape(n, block_size).half().contiguous()
        out = torch.empty_like(x_2d)

        device = lib.device

        x_buf = _private_buffer_from_tensor(x_2d, lib, device, cache=False)
        out_buf = mps_tensor_to_metal_buffer(out, device)

        params = np.array([n, 1 if normalize else 0], dtype=np.uint32)
        params_buf = _params_buffer(lib, device, params)

        # Select kernel based on block size
        kernel_name = f"hadamard_{block_size}"

        dispatch_kernel(
            lib,
            function_name=kernel_name,
            grid=(n, 1, 1),
            threadgroup=(32, 1, 1),  # One simdgroup per vector
            buffers=[x_buf, out_buf, params_buf],
            wait=True,
        )

        return out.reshape(orig_shape)

    def decode_gemv_fp4(
        A: torch.Tensor,
        B_packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 128,
    ) -> torch.Tensor:
        """
        Decode GEMV for M=1: C[1,N] = A[1,K] @ dequant(B[K/8,N], scales).

        Optimized for single-token decode phase where M=1.

        Args:
            A: Activation vector [K] or [1, K]. MPS tensor.
            B_packed: Packed FP4 weights [K/8, N] as uint32. MPS tensor.
            scales: Per-group scales [K/group_size, N]. MPS tensor.
            group_size: Number of K-elements per quantization group.

        Returns:
            Output vector [N] or [1, N] depending on input shape. MPS tensor.
        """
        require_mps()

        lib = get_default_library()
        _ensure_kernel_compiled(lib, "decode_gemv", _DECODE_GEMV_FP4_KERNEL)

        # Normalize input shape
        squeeze_output = False
        if A.ndim == 1:
            A = A.reshape(1, -1)
            squeeze_output = True

        M, K = A.shape
        K_packed, N = B_packed.shape

        if K != K_packed * FP4_PER_UINT:
            raise ValueError(
                f"K mismatch: A has K={K}, B_packed implies K={K_packed * FP4_PER_UINT}"
            )

        if M > 1:
            # Fall back to full GEMM for M > 1
            return marlin_gemm_fp4(A, B_packed, scales, group_size)

        # Single-token decode
        A_flat = A.reshape(-1).half().contiguous()
        out = torch.empty((N,), dtype=torch.float16, device="mps")

        device = lib.device

        A_buf = _private_buffer_from_tensor(A_flat, lib, device, cache=False)
        B_packed_contig = B_packed.contiguous()
        B_buf = _private_buffer_from_tensor(B_packed_contig, lib, device, cache=True)
        scales_half = scales if scales.dtype == torch.float16 else scales.half()
        scales_half = scales_half.contiguous()
        S_buf = _private_buffer_from_tensor(scales_half, lib, device, cache=True)
        out_buf = mps_tensor_to_metal_buffer(out, device)

        params = np.array([K, N, group_size], dtype=np.uint32)
        params_buf = _params_buffer(lib, device, params)

        grid_x = (N + 255) // 256

        dispatch_kernel(
            lib,
            function_name="decode_gemv_fp4",
            grid=(grid_x, 1, 1),
            threadgroup=(128, 1, 1),
            buffers=[A_buf, B_buf, S_buf, out_buf, params_buf],
            wait=True,
        )

        if not squeeze_output:
            out = out.reshape(1, N)
        return out

    def select_decode_kernel(M: int, N: int, K: int) -> str:
        """
        Select optimal decode kernel based on problem dimensions.

        Args:
            M: Batch size (typically 1 for decode).
            N: Output dimension.
            K: Input dimension.

        Returns:
            Kernel name recommendation.
        """
        if M > 8:
            return "marlin_gemm_fp4"

        if M > 1:
            return "decode_gemv_fp4"  # batched handled internally

        # M == 1 decode
        return "decode_gemv_fp4"
