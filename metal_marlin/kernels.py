"""
MLX custom Metal kernel wrappers for Marlin GEMM and dequantization.

Provides mx.fast.metal_kernel interfaces to the fused dequant-GEMM kernels
(FP4 E2M1 and INT4 U4) and standalone dequant kernels from src/*.metal.

The mx.fast.metal_kernel API injects your source as the body of an
auto-generated kernel function. We put all helpers (dequant primitives,
tile loaders, simdgroup compute) in the header, and the main kernel
logic in the source body. Template parameters provide compile-time
constants (M, N, K, tile dimensions, etc.).

Usage:
    from kernels import marlin_gemm_fp4, marlin_gemm_int4, dequant_fp4

    # FP4 fused dequant-GEMM
    C = marlin_gemm_fp4(A, B_packed, scales, group_size=32)

    # INT4 fused dequant-GEMM with zero points
    C = marlin_gemm_int4(A, B_packed, scales, zeros, group_size=32)

    # Standalone FP4 dequantization
    W_fp16 = dequant_fp4(B_packed, scales, K, N, group_size=32)

Note:
    Metal kernel dispatch requires MLX. When MLX is not available, all kernel
    functions will raise ImportError with installation instructions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import HAS_MLX

# Conditional imports: MLX and dtypes require MLX at import time
if HAS_MLX:
    import mlx.core as mx

    from .dtypes import get_default_config
elif TYPE_CHECKING:
    # For static analysis only
    import mlx.core as mx

    from .autotuning import TileConfig
    from .dtypes import get_default_config


def _mlx_required(*args: Any, **kwargs: Any) -> Any:
    """Stub function that raises ImportError when MLX is unavailable."""
    raise ImportError(
        "Metal kernel dispatch requires MLX. "
        "Install with: pip install mlx"
    )


# When MLX is unavailable, export stubs for all public kernel functions
if not HAS_MLX:
    # GEMM kernels
    marlin_gemm_fp4 = _mlx_required
    marlin_gemm_fused_fp4 = _mlx_required
    marlin_gemm_fp4_tuned = _mlx_required
    marlin_gemm_int4 = _mlx_required
    marlin_gemm_fused_u4 = _mlx_required

    # Dequantization kernels
    dequant_fp4 = _mlx_required
    dequant_u4_standalone = _mlx_required
    dequant_int2 = _mlx_required
    dequant_int3 = _mlx_required

    # Weight packing functions
    pack_fp4_weights = _mlx_required
    pack_int2_weights = _mlx_required
    pack_int3_weights = _mlx_required

    # Flash attention
    flash_attention_kv_fp4 = _mlx_required

    # MoE kernels
    moe_expert_gemm_fp4 = _mlx_required
    moe_router_topk = _mlx_required

    # Hadamard transform
    hadamard_transform = _mlx_required

# Constants and Metal shader strings don't require MLX - defined unconditionally
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
# Kernel header: all helper functions, dequant primitives, tile logic
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
    #define DEQUANT_INT2(idx) { \
        uint code = (packed >> ((idx) * 2u)) & 0x3u; \
        out[idx] = (half)((float(code) - 1.5f) * 0.6667f * fscale); \
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
    #define DEQUANT_INT3(idx) { \
        uint code = (packed >> ((idx) * 3u)) & 0x7u; \
        out[idx] = (half)((float(code) - 3.5f) * 0.2857f * fscale); \
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
"""

# ---------------------------------------------------------------------------
# FP4 fused dequant-GEMM kernel body
#
# This is the body of the auto-generated kernel function. MLX provides
# the inputs as device pointers named per input_names/output_names:
#   device const half* A           (input)
#   device const uint* B_packed    (input)
#   device const half* scales      (input)
#   device half* out               (output)
#
# Template params provide compile-time uint constants:
#   M, N, K, GROUP_SIZE, TILE_M, TILE_N, TILE_K, SIMDGROUPS_PER_TG,
#   SG_M_TILES, SG_N_TILES, FP4_PER_UINT, NUM_BUFFERS
#
# Built-in Metal variables available:
#   threadgroup_position_in_grid, thread_position_in_threadgroup,
#   thread_index_in_simdgroup, simdgroup_index_in_threadgroup
# ---------------------------------------------------------------------------

_FP4_GEMM_SOURCE = """
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
"""

# ---------------------------------------------------------------------------
# FP4 fused dequant-GEMM kernel body with FP32 accumulators
#
# Identical to _FP4_GEMM_SOURCE but uses float accumulators instead of half
# to prevent overflow when K > ~8192 with large weight values.
# Output is still FP16 (cast from float on store).
# ---------------------------------------------------------------------------

_FP4_GEMM_FP32ACC_SOURCE = """
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

    // FP32 accumulators to prevent overflow for large K
    simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi)
        for (uint ni = 0; ni < SG_N_TILES; ++ni)
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

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

    // Store results (cast FP32 accumulators to FP16 output)
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                // Cast FP32 accumulator to FP16 for store
                simdgroup_matrix<half, 8, 8> out_frag;
                // Extract float values and convert to half via staging
                threadgroup float f32_staging[8][8];
                simdgroup_store(acc[mi][ni], &f32_staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup half h16_staging[8][8];
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    h16_staging[r][c] = half(f32_staging[r][c]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    out[(out_row + r) * N + out_col + c] = h16_staging[r][c];
                }
            } else if (out_row < M && out_col < N) {
                threadgroup float f32_staging[8][8];
                simdgroup_store(acc[mi][ni], &f32_staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < M && out_col + c < N) {
                        out[(out_row + r) * N + out_col + c] = half(f32_staging[r][c]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
"""

# ---------------------------------------------------------------------------
# INT4 fused dequant-GEMM kernel body (with zero points)
# ---------------------------------------------------------------------------

_INT4_GEMM_SOURCE = """
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
"""

# ---------------------------------------------------------------------------
# Single-stage (non-pipelined) FP4 GEMM kernel body
#
# Load → barrier → compute → barrier → loop
# No overlap, single-buffered. Useful as performance baseline and for debugging.
# Uses full B_tile[TILE_K][TILE_N] in threadgroup memory with cooperative
# dequantization (all 128 threads participate in loading/dequanting B).
# ---------------------------------------------------------------------------

_FP4_GEMM_SINGLE_STAGE_SOURCE = """
    // Thread identity
    uint3 tgid = threadgroup_position_in_grid;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint thread_idx = simd_id * 32 + simd_lane;

    // Single-buffered threadgroup memory
    threadgroup half A_tile[TILE_M][TILE_K];
    threadgroup half B_tile[TILE_K][TILE_N];

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

    uint k_tiles_count = TILE_K / 8;

    // Main K-reduction loop (single-stage: load, barrier, compute, barrier)
    for (uint k_block = 0; k_block < K; k_block += TILE_K) {

        // Cooperative A tile load (all 128 threads)
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
                A_tile[row][col] = val;
            }
        }

        // Cooperative B tile load with fused dequantization (all 128 threads)
        {
            const uint packed_per_thread = (TILE_K * TILE_N) / (SIMDGROUPS_PER_TG * 32 * FP4_PER_UINT);
            for (uint i = 0; i < packed_per_thread; ++i) {
                uint flat_packed_idx = thread_idx * packed_per_thread + i;
                uint n_idx = flat_packed_idx / (TILE_K / FP4_PER_UINT);
                uint k_group_in_tile = flat_packed_idx % (TILE_K / FP4_PER_UINT);

                uint global_n = tg_col + n_idx;
                uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

                uint scale_k = global_k_base / GROUP_SIZE;
                half s = half(1.0h);
                if (global_n < N && scale_k < ((K + GROUP_SIZE - 1) / GROUP_SIZE)) {
                    s = scales[scale_k * N + global_n];
                }

                uint32_t packed = 0;
                uint b_row = global_k_base / FP4_PER_UINT;
                if (global_n < N && b_row < ((K + FP4_PER_UINT - 1) / FP4_PER_UINT)) {
                    packed = B_packed[b_row * N + global_n];
                }

                uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
                half vals[8];
                dequant_fp4x8(packed, s, vals);
                for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < TILE_K; ++v) {
                    if (n_idx < TILE_N) {
                        B_tile[tile_k_base + v][n_idx] = vals[v];
                    }
                }
            }
        }

        // Wait for all loads to complete
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: each simdgroup does its portion via MMA
        for (uint kt = 0; kt < k_tiles_count; ++kt) {
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tile[sg_row_offset + mi * 8][kt * 8],
                               TILE_K);

                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag,
                                   &B_tile[kt * 8][sg_col_offset + ni * 8],
                                   TILE_N);

                    simdgroup_multiply_accumulate(acc[mi][ni],
                                                  a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        // Wait for compute to finish before next iteration overwrites tiles
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
"""

# ---------------------------------------------------------------------------
# Standalone dequant kernel body (FP4 -> FP16)
# ---------------------------------------------------------------------------

_DEQUANT_FP4_SOURCE = """
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
"""

# ---------------------------------------------------------------------------
# Standalone dequant kernel body (U4 -> FP16, raw integer recovery)
# ---------------------------------------------------------------------------

_DEQUANT_U4_STANDALONE_SOURCE = """
    // Grid: (num_packed, 1, 1) - each thread handles one packed uint32 (8 U4 values)
    uint tid = thread_position_in_grid.x;
    if (tid >= NUM_PACKED) return;

    uint32_t packed = input[tid];

    // Dequant 8 nibbles to raw integer values using magic bias trick
    // Scale=1.0, zero_point=0.0 -> recovers raw U4 integer as FP16
    half vals[8];
    dequant_u4x8(packed, half(1.0h), half(0.0h), vals);

    // Write 8 contiguous output values
    uint out_base = tid * 8u;
    for (uint i = 0; i < 8u; i++) {
        out[out_base + i] = vals[i];
    }
"""

# ---------------------------------------------------------------------------
# Standalone dequant kernel body (INT2 -> FP16)
# 16 weights per uint32 (2 bits each)
# ---------------------------------------------------------------------------

_DEQUANT_INT2_SOURCE = """
    // Grid: (N, K/16, 1) - each thread handles one packed uint32 (16 INT2 values)
    uint n_idx = thread_position_in_grid.x;
    uint k_block = thread_position_in_grid.y;

    if (n_idx >= N || k_block * 16u >= K) return;

    uint packed_idx = k_block * N + n_idx;
    uint32_t packed = B_packed[packed_idx];

    uint k_start = k_block * 16u;
    uint group_idx = k_start / GROUP_SIZE;
    half scale = scales[group_idx * N + n_idx];

    half vals[16];
    dequant_int2x16_unrolled(packed, scale, vals);

    uint out_base = k_start * N + n_idx;
    uint k_remain = min(16u, K - k_start);
    for (uint i = 0; i < k_remain; i++) {
        out[out_base + i * N] = vals[i];
    }
"""

# ---------------------------------------------------------------------------
# Standalone dequant kernel body (INT3 -> FP16)
# 10 weights per uint32 (3 bits each, 30 bits used, 2 bits unused)
# ---------------------------------------------------------------------------

_DEQUANT_INT3_SOURCE = """
    // Grid: (N, K/10, 1) - each thread handles one packed uint32 (10 INT3 values)
    uint n_idx = thread_position_in_grid.x;
    uint k_block = thread_position_in_grid.y;

    if (n_idx >= N || k_block * 10u >= K) return;

    uint packed_idx = k_block * N + n_idx;
    uint32_t packed = B_packed[packed_idx];

    uint k_start = k_block * 10u;
    uint group_idx = k_start / GROUP_SIZE;
    half scale = scales[group_idx * N + n_idx];

    half vals[10];
    dequant_int3x10_unrolled(packed, scale, vals);

    uint out_base = k_start * N + n_idx;
    uint k_remain = min(10u, K - k_start);
    for (uint i = 0; i < k_remain; i++) {
        out[out_base + i * N] = vals[i];
    }
"""

# ---------------------------------------------------------------------------
# Kernel builders (lazy cached)
# ---------------------------------------------------------------------------

_fp4_gemm_kernel: object | None = None
_fp4_gemm_fp32acc_kernel: object | None = None
_fp4_gemm_single_stage_kernel: object | None = None
_fp4_gemm_kernel_cache: dict[str, object] = {}
_int4_gemm_kernel: object | None = None
_dequant_fp4_kernel: object | None = None
_dequant_u4_standalone_kernel: object | None = None
_dequant_int2_kernel: object | None = None
_dequant_int3_kernel: object | None = None


def _get_fp4_gemm_kernel() -> object:
    global _fp4_gemm_kernel
    if _fp4_gemm_kernel is None:
        _fp4_gemm_kernel = mx.fast.metal_kernel(
            name="marlin_gemm_fused_fp4",
            input_names=["A", "B_packed", "scales"],
            output_names=["out"],
            source=_FP4_GEMM_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _fp4_gemm_kernel


def _get_fp4_gemm_kernel_for_config(name: str) -> object:
    kernel = _fp4_gemm_kernel_cache.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=["A", "B_packed", "scales"],
            output_names=["out"],
            source=_FP4_GEMM_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
        _fp4_gemm_kernel_cache[name] = kernel
    return kernel


def _get_fp4_gemm_fp32acc_kernel() -> object:
    global _fp4_gemm_fp32acc_kernel
    if _fp4_gemm_fp32acc_kernel is None:
        _fp4_gemm_fp32acc_kernel = mx.fast.metal_kernel(
            name="marlin_gemm_fused_fp4_fp32acc",
            input_names=["A", "B_packed", "scales"],
            output_names=["out"],
            source=_FP4_GEMM_FP32ACC_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _fp4_gemm_fp32acc_kernel


def _get_fp4_gemm_single_stage_kernel() -> object:
    global _fp4_gemm_single_stage_kernel
    if _fp4_gemm_single_stage_kernel is None:
        _fp4_gemm_single_stage_kernel = mx.fast.metal_kernel(
            name="marlin_gemm_fp4_single_stage",
            input_names=["A", "B_packed", "scales"],
            output_names=["out"],
            source=_FP4_GEMM_SINGLE_STAGE_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _fp4_gemm_single_stage_kernel


def _get_int4_gemm_kernel() -> object:
    global _int4_gemm_kernel
    if _int4_gemm_kernel is None:
        _int4_gemm_kernel = mx.fast.metal_kernel(
            name="marlin_gemm_fused_u4",
            input_names=["A", "B_packed", "scales", "zeros"],
            output_names=["out"],
            source=_INT4_GEMM_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _int4_gemm_kernel


def _get_dequant_fp4_kernel() -> object:
    global _dequant_fp4_kernel
    if _dequant_fp4_kernel is None:
        _dequant_fp4_kernel = mx.fast.metal_kernel(
            name="dequant_fp4",
            input_names=["B_packed", "scales"],
            output_names=["out"],
            source=_DEQUANT_FP4_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _dequant_fp4_kernel


def _get_dequant_u4_standalone_kernel() -> object:
    global _dequant_u4_standalone_kernel
    if _dequant_u4_standalone_kernel is None:
        _dequant_u4_standalone_kernel = mx.fast.metal_kernel(
            name="dequant_u4_standalone",
            input_names=["input"],
            output_names=["out"],
            source=_DEQUANT_U4_STANDALONE_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _dequant_u4_standalone_kernel


def _get_dequant_int2_kernel() -> object:
    global _dequant_int2_kernel
    if _dequant_int2_kernel is None:
        _dequant_int2_kernel = mx.fast.metal_kernel(
            name="dequant_int2",
            input_names=["B_packed", "scales"],
            output_names=["out"],
            source=_DEQUANT_INT2_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _dequant_int2_kernel


def _get_dequant_int3_kernel() -> object:
    global _dequant_int3_kernel
    if _dequant_int3_kernel is None:
        _dequant_int3_kernel = mx.fast.metal_kernel(
            name="dequant_int3",
            input_names=["B_packed", "scales"],
            output_names=["out"],
            source=_DEQUANT_INT3_SOURCE,
            header=_GEMM_HEADER,
            ensure_row_contiguous=True,
        )
    return _dequant_int3_kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack_fp4_weights(
    weight: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Pack FP16/BF16 weights into FP4 E2M1 format for the fused GEMM kernels.

    Weight layout: [K/8, N] as uint32, where each uint32 holds 8 FP4 values
    along the K dimension at a given output column. This matches the fused
    kernel's access pattern: B_packed[k_pack_idx * N + col].

    Scales layout: [K/group_size, N] with dtype matching config.

    Args:
        weight: Weight matrix [N, K] (PyTorch convention: out_features first).
                Transposed internally to [K, N] for the kernel layout.
        group_size: Number of K-dimension elements per quantization group.
        dtype: Compute dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Tuple of (weight_packed [K/8, N] uint32, scales [K/group_size, N]).
    """

    if dtype is None:
        dtype = get_default_config().mlx_weights

    w = weight.T.astype(dtype)  # [K, N]
    K, N = w.shape

    if K % FP4_PER_UINT != 0:
        raise ValueError(f"K ({K}) must be divisible by {FP4_PER_UINT}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    # Per-group scales along K: max abs value in each group
    w_grouped = w.reshape(K // group_size, group_size, N)
    scales = mx.max(mx.abs(w_grouped), axis=1)
    scales = mx.maximum(scales, mx.array(1e-7, dtype=dtype))
    mx.eval(scales)

    # E2M1 representable values (positive): 0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    # Max magnitude = 6.0
    MAX_E2M1 = 6.0

    # Normalize by group scale and clamp to E2M1 range
    scales_expanded = mx.repeat(scales, group_size, axis=0)  # [K, N]
    w_norm = w / scales_expanded
    w_norm = mx.clip(w_norm, -MAX_E2M1, MAX_E2M1)
    mx.eval(w_norm)

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
    w_np = np.array(w_norm.astype(mx.float32))
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

    weight_packed = mx.array(packed)
    return weight_packed, scales


def marlin_gemm_fp4(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
    autotune: bool = True,
    tile_config: TileConfig | None = None,
) -> mx.array:
    """
    FP4 fused dequant-GEMM: C = A @ dequant(B_packed, scales).

    Uses simdgroup_multiply_accumulate with fused FP4 E2M1 dequantization
    in registers. No intermediate FP16 weight materialization in threadgroup
    memory; each simdgroup dequantizes only the 8x8 sub-tile it needs.

    Args:
        A: Input activations [*, K]. Arbitrary leading dims.
        B_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        group_size: Elements per quantization group (must divide K).
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Output [*, N] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    orig_shape = A.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    A_2d = A.reshape(M, K).astype(dtype)
    N = B_packed.shape[1]

    if tile_config is None and autotune:
        try:
            from .autotuning import get_heuristic_config, get_tuned_config

            tile_config = get_tuned_config(M, N, K, group_size=group_size)
        except Exception:
            tile_config = get_heuristic_config(M, N, K, group_size=group_size)

    if tile_config is None:
        grid_x = (N + TILE_N - 1) // TILE_N
        grid_y = (M + TILE_M - 1) // TILE_M
        kernel = _get_fp4_gemm_kernel()
        template = [
            ("M", M),
            ("N", N),
            ("K", K),
            ("GROUP_SIZE", group_size),
            ("TILE_M", TILE_M),
            ("TILE_N", TILE_N),
            ("TILE_K", TILE_K),
            ("SIMDGROUPS_PER_TG", SIMDGROUPS_PER_TG),
            ("SG_M_TILES", SG_M_TILES),
            ("SG_N_TILES", SG_N_TILES),
            ("FP4_PER_UINT", FP4_PER_UINT),
            ("NUM_BUFFERS", NUM_BUFFERS),
        ]
        threadgroup = (THREADS_PER_TG, 1, 1)
    else:
        grid_x = (N + tile_config.tile_n - 1) // tile_config.tile_n
        grid_y = (M + tile_config.tile_m - 1) // tile_config.tile_m
        kernel_name = f"marlin_gemm_fp4_{tile_config.name}"
        kernel = _get_fp4_gemm_kernel_for_config(kernel_name)
        template = [
            ("M", M),
            ("N", N),
            ("K", K),
            ("GROUP_SIZE", group_size),
            ("TILE_M", tile_config.tile_m),
            ("TILE_N", tile_config.tile_n),
            ("TILE_K", tile_config.tile_k),
            ("SIMDGROUPS_PER_TG", tile_config.simdgroups_per_tg),
            ("SG_M_TILES", tile_config.sg_m_tiles),
            ("SG_N_TILES", tile_config.sg_n_tiles),
            ("FP4_PER_UINT", tile_config.fp4_per_uint),
            ("NUM_BUFFERS", tile_config.num_buffers),
        ]
        threadgroup = (tile_config.threads_per_tg, 1, 1)

    outputs = kernel(
        inputs=[A_2d, B_packed, scales],
        template=template,
        grid=(grid_x, grid_y, 1),
        threadgroup=threadgroup,
        output_shapes=[(M, N)],
        output_dtypes=[dtype],
    )

    out_shape = list(orig_shape[:-1]) + [N]
    return outputs[0].reshape(out_shape)


def marlin_gemm_fp4_tuned(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """FP4 fused dequant-GEMM using runtime autotuned tile parameters."""
    return marlin_gemm_fp4(
        A,
        B_packed,
        scales,
        group_size=group_size,
        dtype=dtype,
        autotune=True,
    )


def marlin_gemm_fused_fp4(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Alias for marlin_gemm_fp4; preserves fused kernel naming."""
    return marlin_gemm_fp4(
        A,
        B_packed,
        scales,
        group_size=group_size,
        dtype=dtype,
    )


def marlin_gemm_int4(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    zeros: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    INT4 fused dequant-GEMM with asymmetric quantization (scale + zero point).

    Uses magic-bias trick for INT4 U4 dequantization: OR nibble into FP16
    mantissa of 1024.0, subtract bias to recover integer value.

    Args:
        A: Input activations [*, K].
        B_packed: Packed INT4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        zeros: Per-group zero points [K/group_size, N].
        group_size: Elements per quantization group.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Output [*, N] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    orig_shape = A.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    A_2d = A.reshape(M, K).astype(dtype)
    N = B_packed.shape[1]

    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M

    kernel = _get_int4_gemm_kernel()
    outputs = kernel(
        inputs=[A_2d, B_packed, scales, zeros],
        template=[
            ("M", M),
            ("N", N),
            ("K", K),
            ("GROUP_SIZE", group_size),
            ("TILE_M", TILE_M),
            ("TILE_N", TILE_N),
            ("TILE_K", TILE_K),
            ("SIMDGROUPS_PER_TG", SIMDGROUPS_PER_TG),
            ("SG_M_TILES", SG_M_TILES),
            ("SG_N_TILES", SG_N_TILES),
            ("FP4_PER_UINT", FP4_PER_UINT),
            ("NUM_BUFFERS", NUM_BUFFERS),
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(THREADS_PER_TG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[dtype],
    )

    out_shape = list(orig_shape[:-1]) + [N]
    return outputs[0].reshape(out_shape)


def marlin_gemm_fused_u4(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    zeros: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Alias for marlin_gemm_int4; preserves fused kernel naming."""
    return marlin_gemm_int4(
        A,
        B_packed,
        scales,
        zeros,
        group_size=group_size,
        dtype=dtype,
    )


def dequant_fp4(
    B_packed: mx.array,
    scales: mx.array,
    K: int,
    N: int,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Standalone FP4 dequantization: unpack FP4 weights to float.

    Each thread handles one packed uint32 (8 FP4 values along K).
    Output is [K, N] row-major.

    Args:
        B_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        K: Reduction dimension size.
        N: Output feature dimension size.
        group_size: Elements per quantization group.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Dequantized weights [K, N] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_weights

    k_blocks = (K + 7) // 8
    # Grid: each thread = one (n, k_block) pair
    grid_x = N
    grid_y = k_blocks

    kernel = _get_dequant_fp4_kernel()
    outputs = kernel(
        inputs=[B_packed, scales],
        template=[
            ("K", K),
            ("N", N),
            ("GROUP_SIZE", group_size),
            ("FP4_PER_UINT", FP4_PER_UINT),
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(K, N)],
        output_dtypes=[dtype],
    )
    return outputs[0]


def dequant_u4_standalone(packed: mx.array, dtype: mx.Dtype | None = None) -> mx.array:
    """
    Standalone U4 dequantization: unpack U4 nibbles to raw float integer values.

    Each uint8 element in the input is treated as a single U4 value (only low
    4 bits are used). Values are packed into uint32 words (8 nibbles each) and
    run through the magic bias kernel (scale=1.0, zero=0.0) so the output is
    the raw integer [0, 15] as float.

    This function is for debugging/validation of the U4 magic bias trick in
    isolation, independent of the fused GEMM path.

    Args:
        packed: 1-D array of uint8 values, each in [0, 15]. Length must be
                a multiple of 8.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        1-D array of same length with specified dtype, each element = float(input_element).
    """
    if dtype is None:
        dtype = get_default_config().mlx_weights

    n = packed.size
    if n % 8 != 0:
        raise ValueError(f"Input length ({n}) must be a multiple of 8")

    # Pack 8 uint8 nibbles into each uint32: val0 in bits [3:0], val1 in [7:4], ...
    reshaped = packed.reshape(-1, 8).astype(mx.uint32)
    words = (
        (reshaped[:, 0])
        | (reshaped[:, 1] << 4)
        | (reshaped[:, 2] << 8)
        | (reshaped[:, 3] << 12)
        | (reshaped[:, 4] << 16)
        | (reshaped[:, 5] << 20)
        | (reshaped[:, 6] << 24)
        | (reshaped[:, 7] << 28)
    )
    num_packed = words.size

    kernel = _get_dequant_u4_standalone_kernel()
    outputs = kernel(
        inputs=[words],
        template=[("NUM_PACKED", num_packed)],
        grid=(num_packed, 1, 1),
        threadgroup=(min(num_packed, 256), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[dtype],
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# INT2 and INT3 constants
# ---------------------------------------------------------------------------

INT2_PER_UINT = 16  # 32 bits / 2 bits = 16 values per uint32
INT3_PER_UINT = 10  # 30 bits used / 3 bits = 10 values per uint32 (2 bits unused)


def pack_int2_weights(
    weight: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Pack FP16/BF16 weights into INT2 format for dequantization kernels.

    Weight layout: [K/16, N] as uint32, where each uint32 holds 16 INT2 values
    along the K dimension at a given output column.

    Quantization: values are mapped to codes 0,1,2,3 corresponding to
    {-1.0, -0.333, 0.333, 1.0} * scale. The scale is the max absolute value
    in each quantization group.

    Args:
        weight: Weight matrix [N, K] (PyTorch convention: out_features first).
                Transposed internally to [K, N] for the kernel layout.
        group_size: Number of K-dimension elements per quantization group.
        dtype: Compute dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Tuple of (weight_packed [K/16, N] uint32, scales [K/group_size, N]).
    """

    if dtype is None:
        dtype = get_default_config().mlx_weights

    w = weight.T.astype(dtype)  # [K, N]
    K, N = w.shape

    if K % INT2_PER_UINT != 0:
        raise ValueError(f"K ({K}) must be divisible by {INT2_PER_UINT}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    # Per-group scales along K: max abs value in each group
    w_grouped = w.reshape(K // group_size, group_size, N)
    scales = mx.max(mx.abs(w_grouped), axis=1)
    scales = mx.maximum(scales, mx.array(1e-7, dtype=dtype))
    mx.eval(scales)

    # Normalize by group scale and clip
    scales_expanded = mx.repeat(scales, group_size, axis=0)  # [K, N]
    w_norm = w / scales_expanded
    w_norm = mx.clip(w_norm, -1.0, 1.0)
    mx.eval(w_norm)

    # INT2 codebook: {-1.0, -0.333, 0.333, 1.0} -> codes {0, 1, 2, 3}
    # Reverse mapping: code = round((w_norm + 1) * 1.5)
    int2_lut = np.array([-1.0, -0.333, 0.333, 1.0], dtype=np.float32)

    w_np = np.array(w_norm.astype(mx.float32))
    k_packs = K // INT2_PER_UINT
    packed = np.zeros((k_packs, N), dtype=np.uint32)

    for k_pack in range(k_packs):
        k_base = k_pack * INT2_PER_UINT
        for bit_pos in range(INT2_PER_UINT):
            row_vals = w_np[k_base + bit_pos, :]  # [N]
            # Find nearest INT2 code
            dists = np.abs(row_vals[:, None] - int2_lut[None, :])  # [N, 4]
            codes = np.argmin(dists, axis=1).astype(np.uint32)  # [N]
            packed[k_pack, :] |= codes << (bit_pos * 2)

    weight_packed = mx.array(packed)
    return weight_packed, scales


def pack_int3_weights(
    weight: mx.array,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Pack FP16/BF16 weights into INT3 format for dequantization kernels.

    Weight layout: [K/10, N] as uint32, where each uint32 holds 10 INT3 values
    (30 bits used, 2 bits unused) along the K dimension.

    Quantization: values are mapped to codes 0-7 corresponding to
    symmetric [-1.0, 1.0] range: (code - 3.5) / 3.5 * scale.

    Args:
        weight: Weight matrix [N, K] (PyTorch convention: out_features first).
                Transposed internally to [K, N] for the kernel layout.
        group_size: Number of K-dimension elements per quantization group.
        dtype: Compute dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Tuple of (weight_packed [K/10, N] uint32, scales [K/group_size, N]).
    """

    if dtype is None:
        dtype = get_default_config().mlx_weights

    w = weight.T.astype(dtype)  # [K, N]
    K, N = w.shape

    if K % INT3_PER_UINT != 0:
        raise ValueError(f"K ({K}) must be divisible by {INT3_PER_UINT}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    # Per-group scales along K: max abs value in each group
    w_grouped = w.reshape(K // group_size, group_size, N)
    scales = mx.max(mx.abs(w_grouped), axis=1)
    scales = mx.maximum(scales, mx.array(1e-7, dtype=dtype))
    mx.eval(scales)

    # Normalize by group scale and clip
    scales_expanded = mx.repeat(scales, group_size, axis=0)  # [K, N]
    w_norm = w / scales_expanded
    w_norm = mx.clip(w_norm, -1.0, 1.0)
    mx.eval(w_norm)

    # INT3 codebook: 8 levels from -1.0 to 1.0
    # code -> (code - 3.5) / 3.5, so code = round(w_norm * 3.5 + 3.5)
    int3_lut = np.array([(i - 3.5) / 3.5 for i in range(8)], dtype=np.float32)

    w_np = np.array(w_norm.astype(mx.float32))
    k_packs = K // INT3_PER_UINT
    packed = np.zeros((k_packs, N), dtype=np.uint32)

    for k_pack in range(k_packs):
        k_base = k_pack * INT3_PER_UINT
        for bit_pos in range(INT3_PER_UINT):
            row_vals = w_np[k_base + bit_pos, :]  # [N]
            # Find nearest INT3 code
            dists = np.abs(row_vals[:, None] - int3_lut[None, :])  # [N, 8]
            codes = np.argmin(dists, axis=1).astype(np.uint32)  # [N]
            packed[k_pack, :] |= codes << (bit_pos * 3)

    weight_packed = mx.array(packed)
    return weight_packed, scales


def dequant_int2(
    B_packed: mx.array,
    scales: mx.array,
    K: int,
    N: int,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Standalone INT2 dequantization: unpack INT2 weights to float.

    Each thread handles one packed uint32 (16 INT2 values along K).
    Output is [K, N] row-major.

    INT2 mapping: codes 0,1,2,3 -> {-1.0, -0.333, 0.333, 1.0} * scale

    Args:
        B_packed: Packed INT2 weights [K/16, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        K: Reduction dimension size.
        N: Output feature dimension size.
        group_size: Elements per quantization group.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Dequantized weights [K, N] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_weights

    k_blocks = (K + 15) // 16
    # Grid: each thread = one (n, k_block) pair
    grid_x = N
    grid_y = k_blocks

    kernel = _get_dequant_int2_kernel()
    outputs = kernel(
        inputs=[B_packed, scales],
        template=[
            ("K", K),
            ("N", N),
            ("GROUP_SIZE", group_size),
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(K, N)],
        output_dtypes=[dtype],
    )
    return outputs[0]


def dequant_int3(
    B_packed: mx.array,
    scales: mx.array,
    K: int,
    N: int,
    group_size: int = 32,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Standalone INT3 dequantization: unpack INT3 weights to float.

    Each thread handles one packed uint32 (10 INT3 values along K).
    Output is [K, N] row-major.

    INT3 mapping: codes 0-7 -> (code - 3.5) / 3.5 * scale
                  giving symmetric range [-1.0, 1.0] with 8 levels

    Args:
        B_packed: Packed INT3 weights [K/10, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        K: Reduction dimension size.
        N: Output feature dimension size.
        group_size: Elements per quantization group.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Dequantized weights [K, N] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_weights

    k_blocks = (K + 9) // 10
    # Grid: each thread = one (n, k_block) pair
    grid_x = N
    grid_y = k_blocks

    kernel = _get_dequant_int3_kernel()
    outputs = kernel(
        inputs=[B_packed, scales],
        template=[
            ("K", K),
            ("N", N),
            ("GROUP_SIZE", group_size),
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(K, N)],
        output_dtypes=[dtype],
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# Flash Attention with FP4-quantized KV cache
# ---------------------------------------------------------------------------

_FLASH_ATTN_HEADER = """
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_KV = 64;
constant constexpr uint HEAD_DIM_MAX = 128;
constant constexpr uint THREADS_PER_ROW = 32;
constant constexpr uint ROWS_PER_TG = 4;
constant constexpr uint FA_THREADS_PER_TG = THREADS_PER_ROW * ROWS_PER_TG;
constant constexpr uint FA_FP4_PER_UINT = 8;

inline float simd_sum_f32_fa(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// NOTE: Uses float scale to work around Metal compiler bug where
// half parameters in inline functions have fractional parts rounded.
inline half dequant_fp4_fa(uint nibble, half scale) {
    float fscale = (float)scale;
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    float magnitude;
    if (exp_bits == 0) {
        magnitude = (float)(man_bit) * 0.25f;
    } else {
        float power = (float)(1u << (exp_bits - 1));
        float mantissa = 1.0f + (float)(man_bit) * 0.5f;
        magnitude = power * mantissa;
    }

    float result = sign_bit ? -magnitude : magnitude;
    return (half)(result * fscale);
}

inline void load_kv_fp4_tile_fa(
    device const uint* packed,
    device const half* scales,
    threadgroup half (&tile)[TILE_KV][HEAD_DIM_MAX],
    uint head_dim,
    uint seq_k,
    uint packed_head_dim,
    uint kv_packed_offset,
    uint scale_offset,
    uint tile_start,
    uint thread_idx
) {
    const uint packed_elems = TILE_KV * packed_head_dim;
    const uint packed_per_thread = (packed_elems + FA_THREADS_PER_TG - 1) / FA_THREADS_PER_TG;

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint idx = thread_idx + i * FA_THREADS_PER_TG;
        if (idx >= packed_elems) continue;

        uint kv_row = idx / packed_head_dim;
        uint packed_col = idx % packed_head_dim;
        uint src_row = tile_start + kv_row;
        uint base_col = packed_col * FA_FP4_PER_UINT;

        if (src_row < seq_k) {
            half s = scales[scale_offset + src_row];
            uint packed_word = packed[kv_packed_offset + src_row * packed_head_dim + packed_col];
            for (uint j = 0; j < FA_FP4_PER_UINT; ++j) {
                uint d = base_col + j;
                if (d < head_dim) {
                    uint nibble = (packed_word >> (j * 4)) & 0xFu;
                    tile[kv_row][d] = dequant_fp4_fa(nibble, s);
                }
            }
        } else {
            for (uint j = 0; j < FA_FP4_PER_UINT; ++j) {
                uint d = base_col + j;
                if (d < head_dim) {
                    tile[kv_row][d] = half(0);
                }
            }
        }
    }
}
"""

_FLASH_ATTN_KV_FP4_SOURCE = """
    // Template constants: BATCH, NUM_HEADS_Q, NUM_HEADS_K, SEQ_Q, SEQ_K, HEAD_DIM
    // SCALE_NUM and SCALE_DEN encode the attention scale as a rational to avoid
    // float template parameters: actual_scale = float(SCALE_NUM) / float(SCALE_DEN)

    uint3 tgid = threadgroup_position_in_grid;
    uint tid_in_tg = thread_index_in_threadgroup;
    uint lane_id = thread_index_in_simdgroup;
    uint sg_id = simdgroup_index_in_threadgroup;

    const uint head_q = tgid.x;
    const uint q_row_base = tgid.y * ROWS_PER_TG;
    const uint b = tgid.z;

    const uint head_k = head_q * NUM_HEADS_K / NUM_HEADS_Q;
    const uint q_row = q_row_base + sg_id;
    if (q_row >= SEQ_Q) return;

    const float attn_scale = float(SCALE_NUM) / float(SCALE_DEN);

    const uint q_stride_b = NUM_HEADS_Q * SEQ_Q * HEAD_DIM;
    const uint q_stride_h = SEQ_Q * HEAD_DIM;
    const uint q_stride_s = HEAD_DIM;

    const uint packed_head_dim = (HEAD_DIM + FA_FP4_PER_UINT - 1) / FA_FP4_PER_UINT;
    const uint k_packed_stride_b = NUM_HEADS_K * SEQ_K * packed_head_dim;
    const uint k_packed_stride_h = SEQ_K * packed_head_dim;

    const uint k_scale_stride_b = NUM_HEADS_K * SEQ_K;
    const uint k_scale_stride_h = SEQ_K;

    const uint q_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    const uint kv_packed_offset = b * k_packed_stride_b + head_k * k_packed_stride_h;
    const uint k_scale_offset = b * k_scale_stride_b + head_k * k_scale_stride_h;
    const uint v_scale_offset = k_scale_offset;

    const uint elems_per_lane = HEAD_DIM / THREADS_PER_ROW;
    float q_reg[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        q_reg[i] = (d < HEAD_DIM) ? float(Q[q_offset + d]) : 0.0f;
    }

    threadgroup half K_tile[2][TILE_KV][HEAD_DIM_MAX];
    threadgroup half V_tile[2][TILE_KV][HEAD_DIM_MAX];

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM_MAX / THREADS_PER_ROW];
    for (uint i = 0; i < elems_per_lane; ++i) {
        o_acc[i] = 0.0f;
    }

    const uint num_kv_tiles = (SEQ_K + TILE_KV - 1) / TILE_KV;

    load_kv_fp4_tile_fa(K_packed, K_scales, K_tile[0],
                         HEAD_DIM, SEQ_K, packed_head_dim,
                         kv_packed_offset, k_scale_offset, 0, tid_in_tg);
    load_kv_fp4_tile_fa(V_packed, V_scales, V_tile[0],
                         HEAD_DIM, SEQ_K, packed_head_dim,
                         kv_packed_offset, v_scale_offset, 0, tid_in_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;
    for (uint tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        uint buf_load = 1 - buf_compute;
        uint next_tile_start = (tile_idx + 1) * TILE_KV;
        if (tile_idx + 1 < num_kv_tiles) {
            load_kv_fp4_tile_fa(K_packed, K_scales, K_tile[buf_load],
                                 HEAD_DIM, SEQ_K, packed_head_dim,
                                 kv_packed_offset, k_scale_offset, next_tile_start, tid_in_tg);
            load_kv_fp4_tile_fa(V_packed, V_scales, V_tile[buf_load],
                                 HEAD_DIM, SEQ_K, packed_head_dim,
                                 kv_packed_offset, v_scale_offset, next_tile_start, tid_in_tg);
        }

        uint tile_start = tile_idx * TILE_KV;
        uint tile_end = min(tile_start + TILE_KV, SEQ_K);
        uint tile_len = tile_end - tile_start;

        float scores[TILE_KV];
        for (uint ki = 0; ki < tile_len; ++ki) {
            float dot = 0.0f;
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                dot += q_reg[i] * float(K_tile[buf_compute][ki][d]);
            }
            dot = simd_sum_f32_fa(dot);
            scores[ki] = dot * attn_scale;
        }
        for (uint ki = tile_len; ki < TILE_KV; ++ki) {
            scores[ki] = -INFINITY;
        }

        float m_tile = -INFINITY;
        for (uint ki = 0; ki < tile_len; ++ki) {
            m_tile = max(m_tile, scores[ki]);
        }

        float m_new = max(m_prev, m_tile);
        float correction = exp(m_prev - m_new);
        float l_new = l_prev * correction;
        for (uint ki = 0; ki < tile_len; ++ki) {
            l_new += exp(scores[ki] - m_new);
        }

        for (uint i = 0; i < elems_per_lane; ++i) {
            o_acc[i] *= correction;
        }
        for (uint ki = 0; ki < tile_len; ++ki) {
            float p = exp(scores[ki] - m_new);
            for (uint i = 0; i < elems_per_lane; ++i) {
                uint d = lane_id * elems_per_lane + i;
                o_acc[i] += p * float(V_tile[buf_compute][ki][d]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    const uint o_offset = b * q_stride_b + head_q * q_stride_h + q_row * q_stride_s;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = lane_id * elems_per_lane + i;
        if (d < HEAD_DIM) {
            O[o_offset + d] = half(o_acc[i] * inv_l);
        }
    }
"""

_flash_attn_kv_fp4_kernel: object | None = None


def _get_flash_attn_kv_fp4_kernel() -> object:
    global _flash_attn_kv_fp4_kernel
    if _flash_attn_kv_fp4_kernel is None:
        _flash_attn_kv_fp4_kernel = mx.fast.metal_kernel(
            name="flash_attention_kv_fp4",
            input_names=["Q", "K_packed", "V_packed", "K_scales", "V_scales"],
            output_names=["O"],
            source=_FLASH_ATTN_KV_FP4_SOURCE,
            header=_FLASH_ATTN_HEADER,
            ensure_row_contiguous=True,
        )
    return _flash_attn_kv_fp4_kernel


# Flash attention constants
_FA_ROWS_PER_TG = 4
_FA_THREADS_PER_TG = 32 * _FA_ROWS_PER_TG  # 128


def flash_attention_kv_fp4(
    Q: mx.array,
    K_packed: mx.array,
    V_packed: mx.array,
    K_scales: mx.array,
    V_scales: mx.array,
    scale: float,
    num_heads_q: int | None = None,
    num_heads_k: int | None = None,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Flash Attention with FP4-quantized KV cache.

    Uses online softmax to stream through KV tiles without materializing
    the full attention matrix. K and V are stored in packed FP4 format
    with per-row scales and dequantized on-the-fly in threadgroup memory.

    Args:
        Q: Query tensor [batch, num_heads_q, seq_q, head_dim].
        K_packed: Packed FP4 keys [batch, num_heads_k, seq_k, head_dim//8] as uint32.
        V_packed: Packed FP4 values [batch, num_heads_k, seq_k, head_dim//8] as uint32.
        K_scales: Per-row key scales [batch, num_heads_k, seq_k, 1].
        V_scales: Per-row value scales [batch, num_heads_k, seq_k, 1].
        scale: Attention scale factor (typically 1/sqrt(head_dim)).
        num_heads_q: Number of query heads. If None, inferred from Q.shape[1].
        num_heads_k: Number of KV heads. If None, inferred from K_packed.shape[1].
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Output tensor [batch, num_heads_q, seq_q, head_dim] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    Q = Q.astype(dtype)

    batch = Q.shape[0]
    nhq = num_heads_q or Q.shape[1]
    seq_q = Q.shape[2]
    head_dim = Q.shape[3]
    nhk = num_heads_k or K_packed.shape[1]
    seq_k = K_packed.shape[2]

    # Flatten scales: kernel expects flat [batch * num_heads_k * seq_k]
    k_scales_flat = K_scales.reshape(-1)
    v_scales_flat = V_scales.reshape(-1)

    # Encode scale as rational (numerator/denominator) for integer template params
    scale_den = 1000000
    scale_num = int(round(scale * scale_den))

    # Grid: [num_heads_q, ceil(seq_q / ROWS_PER_TG), batch]
    grid_x = nhq
    grid_y = (seq_q + _FA_ROWS_PER_TG - 1) // _FA_ROWS_PER_TG
    grid_z = batch

    output_size = batch * nhq * seq_q * head_dim

    kernel = _get_flash_attn_kv_fp4_kernel()
    outputs = kernel(
        inputs=[
            Q.reshape(-1),
            K_packed.reshape(-1),
            V_packed.reshape(-1),
            k_scales_flat,
            v_scales_flat,
        ],
        template=[
            ("BATCH", batch),
            ("NUM_HEADS_Q", nhq),
            ("NUM_HEADS_K", nhk),
            ("SEQ_Q", seq_q),
            ("SEQ_K", seq_k),
            ("HEAD_DIM", head_dim),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
        ],
        grid=(grid_x, grid_y, grid_z),
        threadgroup=(_FA_THREADS_PER_TG, 1, 1),
        output_shapes=[(output_size,)],
        output_dtypes=[dtype],
    )

    return outputs[0].reshape(batch, nhq, seq_q, head_dim)


# ---------------------------------------------------------------------------
# MoE (Mixture of Experts) kernels
# ---------------------------------------------------------------------------

_MOE_HEADER = """
#include <metal_stdlib>
using namespace metal;

// MoE constants
constant constexpr uint MOE_TILE_N = 64;
constant constexpr uint MOE_THREADS = 256;
constant constexpr uint MOE_FP4_PER_UINT = 8;

// FP4 E2M1 dequant for MoE (same as main GEMM header)
inline half moe_dequant_fp4_scalar(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.25h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

inline void moe_dequant_fp4x8(uint32_t packed, half scale, thread half *out) {
    float fscale = (float)scale;
    out[0] = (half)((float)moe_dequant_fp4_scalar((packed >>  0) & 0xF) * fscale);
    out[1] = (half)((float)moe_dequant_fp4_scalar((packed >>  4) & 0xF) * fscale);
    out[2] = (half)((float)moe_dequant_fp4_scalar((packed >>  8) & 0xF) * fscale);
    out[3] = (half)((float)moe_dequant_fp4_scalar((packed >> 12) & 0xF) * fscale);
    out[4] = (half)((float)moe_dequant_fp4_scalar((packed >> 16) & 0xF) * fscale);
    out[5] = (half)((float)moe_dequant_fp4_scalar((packed >> 20) & 0xF) * fscale);
    out[6] = (half)((float)moe_dequant_fp4_scalar((packed >> 24) & 0xF) * fscale);
    out[7] = (half)((float)moe_dequant_fp4_scalar((packed >> 28) & 0xF) * fscale);
}

// Parallel reduction for softmax
inline float simd_max_f32(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_f32_moe(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}
"""

# MoE expert GEMM: batched GEMM where each token routes to top_k experts
# Each thread block handles one (token, expert) pair
_MOE_EXPERT_GEMM_FP4_SOURCE = """
    // Template constants: BATCH, OUT_DIM, IN_DIM, TOP_K, NUM_EXPERTS, GROUP_SIZE
    // Grid: (ceil(OUT_DIM / MOE_TILE_N), BATCH * TOP_K, 1)

    uint3 tgid = threadgroup_position_in_grid;
    uint tid = thread_index_in_threadgroup;

    uint tile_n = tgid.x;
    uint token_expert_idx = tgid.y;
    uint token_idx = token_expert_idx / TOP_K;
    uint topk_slot = token_expert_idx % TOP_K;

    if (token_idx >= BATCH) return;

    // Get expert ID and probability for this token's k-th expert
    int expert_id = expert_ids[token_idx * TOP_K + topk_slot];
    half expert_prob = expert_probs[token_idx * TOP_K + topk_slot];

    if (expert_id < 0 || expert_id >= (int)NUM_EXPERTS) return;

    // Output column range for this tile
    uint col_start = tile_n * MOE_TILE_N;
    uint col_end = min(col_start + MOE_TILE_N, OUT_DIM);

    // Accumulator for this thread's output columns
    const uint cols_per_thread = MOE_TILE_N / MOE_THREADS;
    float acc[MOE_TILE_N / MOE_THREADS];
    for (uint i = 0; i < cols_per_thread; ++i) {
        acc[i] = 0.0f;
    }

    // Expert weight stride: [num_experts, out_dim, in_dim/8]
    uint packed_in_dim = IN_DIM / MOE_FP4_PER_UINT;
    uint expert_offset = (uint)expert_id * OUT_DIM * packed_in_dim;

    // Scale stride: [num_experts, num_groups, out_dim] where num_groups = in_dim / group_size
    uint num_groups = IN_DIM / GROUP_SIZE;
    uint scale_expert_offset = (uint)expert_id * num_groups * OUT_DIM;

    // Activation offset for this token
    uint act_offset = token_idx * IN_DIM;

    // K-reduction: iterate over input dimension in chunks of 8 (FP4 pack size)
    for (uint k_pack = 0; k_pack < packed_in_dim; ++k_pack) {
        uint k_base = k_pack * MOE_FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;

        // Load 8 activation values (shared across all output columns)
        half act_vals[MOE_FP4_PER_UINT];
        for (uint i = 0; i < MOE_FP4_PER_UINT; ++i) {
            act_vals[i] = (k_base + i < IN_DIM) ? activations[act_offset + k_base + i] : half(0);
        }

        // Each thread handles cols_per_thread output columns
        for (uint i = 0; i < cols_per_thread; ++i) {
            uint col = col_start + tid * cols_per_thread + i;
            if (col >= col_end) continue;

            // Load packed weight and scale for this expert/column
            uint weight_idx = expert_offset + col * packed_in_dim + k_pack;
            uint32_t packed = expert_weights[weight_idx];
            half scale = scales[scale_expert_offset + group_idx * OUT_DIM + col];

            // Dequant and accumulate
            half w_vals[MOE_FP4_PER_UINT];
            moe_dequant_fp4x8(packed, scale, w_vals);

            float dot = 0.0f;
            for (uint j = 0; j < MOE_FP4_PER_UINT; ++j) {
                dot += float(act_vals[j]) * float(w_vals[j]);
            }
            acc[i] += dot;
        }
    }

    // Write output: scale by expert probability and accumulate into output
    // Output layout: [batch, out_dim]
    uint out_base = token_idx * OUT_DIM;
    for (uint i = 0; i < cols_per_thread; ++i) {
        uint col = col_start + tid * cols_per_thread + i;
        if (col >= col_end) continue;

        half result = half(acc[i]) * expert_prob;
        // Atomic add to handle multiple experts contributing to same output
        // Note: For simplicity, we use non-atomic here assuming kernel is called
        // per expert and Python handles accumulation. For fused version, use atomic.
        out[out_base + col] += result;
    }
"""

# MoE router top-k: compute softmax(hidden @ router_weights) and select top-k experts
_MOE_ROUTER_TOPK_SOURCE = """
    // Template constants: BATCH, HIDDEN_DIM, NUM_EXPERTS, TOP_K
    // Grid: (BATCH, 1, 1) - one threadgroup per token

    uint token_idx = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;

    if (token_idx >= BATCH) return;

    // Shared memory for logits and top-k selection
    threadgroup float logits[256];  // Assume NUM_EXPERTS <= 256
    threadgroup int topk_ids[8];    // Assume TOP_K <= 8
    threadgroup float topk_vals[8];

    // Compute router logits: hidden[token] @ router_weights
    // hidden: [batch, hidden_dim], router_weights: [hidden_dim, num_experts]
    uint hidden_offset = token_idx * HIDDEN_DIM;

    // Each thread computes some experts' logits
    uint experts_per_thread = (NUM_EXPERTS + MOE_THREADS - 1) / MOE_THREADS;
    for (uint i = 0; i < experts_per_thread; ++i) {
        uint expert = tid * experts_per_thread + i;
        if (expert >= NUM_EXPERTS) {
            if (expert < 256) logits[expert] = -INFINITY;
            continue;
        }

        float dot = 0.0f;
        for (uint d = 0; d < HIDDEN_DIM; ++d) {
            dot += float(hidden[hidden_offset + d]) * float(router_weights[d * NUM_EXPERTS + expert]);
        }
        logits[expert] = dot;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax: find max, subtract, exp, sum
    if (tid == 0) {
        // Find max logit
        float max_logit = -INFINITY;
        for (uint e = 0; e < NUM_EXPERTS; ++e) {
            max_logit = max(max_logit, logits[e]);
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (uint e = 0; e < NUM_EXPERTS; ++e) {
            float exp_val = exp(logits[e] - max_logit);
            logits[e] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        float inv_sum = 1.0f / sum_exp;
        for (uint e = 0; e < NUM_EXPERTS; ++e) {
            logits[e] *= inv_sum;
        }

        // Top-k selection (simple O(k*n) for small k)
        for (uint k = 0; k < TOP_K; ++k) {
            float best_val = -INFINITY;
            int best_idx = -1;
            for (uint e = 0; e < NUM_EXPERTS; ++e) {
                if (logits[e] > best_val) {
                    best_val = logits[e];
                    best_idx = (int)e;
                }
            }
            topk_ids[k] = best_idx;
            topk_vals[k] = best_val;
            if (best_idx >= 0) {
                logits[best_idx] = -INFINITY;  // Mark as selected
            }
        }

        // Write outputs
        for (uint k = 0; k < TOP_K; ++k) {
            expert_ids_out[token_idx * TOP_K + k] = topk_ids[k];
            expert_probs_out[token_idx * TOP_K + k] = half(topk_vals[k]);
        }
    }
"""

_moe_expert_gemm_fp4_kernel: object | None = None
_moe_router_topk_kernel: object | None = None


def _get_moe_expert_gemm_fp4_kernel() -> object:
    global _moe_expert_gemm_fp4_kernel
    if _moe_expert_gemm_fp4_kernel is None:
        _moe_expert_gemm_fp4_kernel = mx.fast.metal_kernel(
            name="moe_expert_gemm_fp4",
            input_names=["activations", "expert_weights", "scales", "expert_ids", "expert_probs"],
            output_names=["out"],
            source=_MOE_EXPERT_GEMM_FP4_SOURCE,
            header=_MOE_HEADER,
            ensure_row_contiguous=True,
        )
    return _moe_expert_gemm_fp4_kernel


def _get_moe_router_topk_kernel() -> object:
    global _moe_router_topk_kernel
    if _moe_router_topk_kernel is None:
        _moe_router_topk_kernel = mx.fast.metal_kernel(
            name="moe_router_topk",
            input_names=["hidden", "router_weights"],
            output_names=["expert_ids_out", "expert_probs_out"],
            source=_MOE_ROUTER_TOPK_SOURCE,
            header=_MOE_HEADER,
            ensure_row_contiguous=True,
        )
    return _moe_router_topk_kernel


def moe_expert_gemm_fp4(
    activations: mx.array,
    expert_weights: mx.array,
    scales: mx.array,
    expert_ids: mx.array,
    expert_probs: mx.array,
    group_size: int = 128,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    MoE expert GEMM with FP4-quantized expert weights.

    Computes the weighted sum of expert outputs for each token:
        output[b] = sum_k(expert_probs[b,k] * activations[b] @ expert_weights[expert_ids[b,k]])

    Each expert's weights are stored in packed FP4 format. The kernel handles
    routing and weight dequantization on-the-fly.

    Args:
        activations: Input activations [batch, hidden_in].
        expert_weights: Packed FP4 expert weights [num_experts, hidden_out, hidden_in/8] as uint32.
        scales: Per-group scales [num_experts, num_groups, hidden_out],
                where num_groups = hidden_in / group_size.
        expert_ids: Expert indices for each token [batch, top_k] as int32.
        expert_probs: Expert probabilities [batch, top_k].
        group_size: Elements per quantization group (default 128).
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Output tensor [batch, hidden_out] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    batch = activations.shape[0]
    hidden_in = activations.shape[1]
    num_experts = expert_weights.shape[0]
    hidden_out = expert_weights.shape[1]
    top_k = expert_ids.shape[1]

    # Ensure correct dtypes
    activations = activations.astype(dtype)
    expert_probs = expert_probs.astype(dtype)
    expert_ids = expert_ids.astype(mx.int32)

    # Grid: (ceil(hidden_out / MOE_TILE_N), batch * top_k, 1)
    moe_tile_n = 64
    grid_x = (hidden_out + moe_tile_n - 1) // moe_tile_n
    grid_y = batch * top_k

    kernel = _get_moe_expert_gemm_fp4_kernel()
    outputs = kernel(
        inputs=[activations, expert_weights, scales, expert_ids, expert_probs],
        template=[
            ("BATCH", batch),
            ("OUT_DIM", hidden_out),
            ("IN_DIM", hidden_in),
            ("TOP_K", top_k),
            ("NUM_EXPERTS", num_experts),
            ("GROUP_SIZE", group_size),
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, hidden_out)],
        output_dtypes=[dtype],
    )

    return outputs[0]


def moe_router_topk(
    hidden: mx.array,
    router_weights: mx.array,
    top_k: int = 2,
    dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """
    MoE router with top-k expert selection.

    Computes softmax(hidden @ router_weights) and selects the top-k experts
    with highest probabilities for each token.

    Args:
        hidden: Hidden states [batch, hidden_dim].
        router_weights: Router weight matrix [hidden_dim, num_experts].
        top_k: Number of experts to select per token (default 2).
        dtype: Compute dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Tuple of:
            expert_ids: Selected expert indices [batch, top_k] as int32.
            expert_probs: Expert probabilities [batch, top_k] with specified dtype.
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    batch = hidden.shape[0]
    hidden_dim = hidden.shape[1]
    num_experts = router_weights.shape[1]

    # Ensure correct dtypes
    hidden = hidden.astype(dtype)
    router_weights = router_weights.astype(dtype)

    kernel = _get_moe_router_topk_kernel()
    outputs = kernel(
        inputs=[hidden, router_weights],
        template=[
            ("BATCH", batch),
            ("HIDDEN_DIM", hidden_dim),
            ("NUM_EXPERTS", num_experts),
            ("TOP_K", top_k),
        ],
        grid=(batch, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, top_k), (batch, top_k)],
        output_dtypes=[mx.int32, dtype],
    )

    return outputs[0], outputs[1]


# ---------------------------------------------------------------------------
# Hadamard Transform Metal Source
# ---------------------------------------------------------------------------

# The kernels use float for computation to handle both float16 and bfloat16 inputs.
# MLX's metal_kernel handles the dtype conversion at the input/output boundaries.

_HADAMARD_HEADER = """
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
"""

_HADAMARD_32_SOURCE = """
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid.x;

    if (tg_idx >= N) return;

    uint base = tg_idx * 32;
    float val = float(input[base + lane_id]);

    // 5 butterfly stages for size 32
    #pragma unroll
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

    out[base + lane_id] = val;
"""

_HADAMARD_64_SOURCE = """
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid.x;

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
    #pragma unroll
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

    out[base + lane_id * 2] = val.x;
    out[base + lane_id * 2 + 1] = val.y;
"""

_HADAMARD_128_SOURCE = """
    uint lane_id = thread_index_in_simdgroup;
    uint tg_idx = threadgroup_position_in_grid.x;

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
    #pragma unroll
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

    out[base + lane_id * 4] = val.x;
    out[base + lane_id * 4 + 1] = val.y;
    out[base + lane_id * 4 + 2] = val.z;
    out[base + lane_id * 4 + 3] = val.w;
"""

# Kernel cache for Hadamard transforms
_hadamard_32_kernel: object | None = None
_hadamard_64_kernel: object | None = None
_hadamard_128_kernel: object | None = None


def _get_hadamard_32_kernel() -> object:
    global _hadamard_32_kernel
    if _hadamard_32_kernel is None:
        _hadamard_32_kernel = mx.fast.metal_kernel(
            name="hadamard_32",
            input_names=["input"],
            output_names=["out"],
            source=_HADAMARD_32_SOURCE,
            header=_HADAMARD_HEADER,
            ensure_row_contiguous=True,
        )
    return _hadamard_32_kernel


def _get_hadamard_64_kernel() -> object:
    global _hadamard_64_kernel
    if _hadamard_64_kernel is None:
        _hadamard_64_kernel = mx.fast.metal_kernel(
            name="hadamard_64",
            input_names=["input"],
            output_names=["out"],
            source=_HADAMARD_64_SOURCE,
            header=_HADAMARD_HEADER,
            ensure_row_contiguous=True,
        )
    return _hadamard_64_kernel


def _get_hadamard_128_kernel() -> object:
    global _hadamard_128_kernel
    if _hadamard_128_kernel is None:
        _hadamard_128_kernel = mx.fast.metal_kernel(
            name="hadamard_128",
            input_names=["input"],
            output_names=["out"],
            source=_HADAMARD_128_SOURCE,
            header=_HADAMARD_HEADER,
            ensure_row_contiguous=True,
        )
    return _hadamard_128_kernel


def hadamard_transform(
    x: mx.array,
    block_size: int = 64,
    normalize: bool = True,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Apply Walsh-Hadamard transform to input vectors.

    For Hadamard-rotated weights (QuIP#, HQQ), activations need complementary
    rotation before the GEMM:
        y = (H @ W) @ (H^T @ x) = W @ x  (mathematically equivalent)

    The Walsh-Hadamard transform uses the butterfly pattern with O(n log n)
    complexity. This implementation uses simd_shuffle for register-only
    computation within a simdgroup (32 threads on Apple Silicon).

    Args:
        x: Input tensor [..., block_size]. The last dimension must equal block_size.
        block_size: Size of each Hadamard block. Must be 32, 64, or 128.
        normalize: If True, normalize by 1/sqrt(block_size) after transform.
                   For QuIP#/HQQ, typically True for forward, False for inverse.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).
               Supports mx.float16 and mx.bfloat16.

    Returns:
        Transformed tensor with same shape as input.

    Raises:
        ValueError: If block_size is not 32, 64, or 128.
        ValueError: If the last dimension of x doesn't match block_size.

    Example:
        >>> x = mx.random.normal((batch, hidden_dim))  # hidden_dim = 64
        >>> x_rotated = hadamard_transform(x, block_size=64)
        >>> # Use x_rotated with Hadamard-rotated quantized weights
    """
    if block_size not in (32, 64, 128):
        raise ValueError(f"block_size must be 32, 64, or 128, got {block_size}")

    if x.shape[-1] != block_size:
        raise ValueError(
            f"Last dimension of x ({x.shape[-1]}) must equal block_size ({block_size})"
        )

    if dtype is None:
        dtype = get_default_config().mlx_activations

    # Flatten to 2D: [N, block_size] where N = product of all dims except last
    orig_shape = x.shape
    n = 1
    for d in orig_shape[:-1]:
        n *= d

    x_2d = x.reshape(n, block_size).astype(dtype)

    # Select kernel based on block size
    if block_size == 32:
        kernel = _get_hadamard_32_kernel()
    elif block_size == 64:
        kernel = _get_hadamard_64_kernel()
    else:  # 128
        kernel = _get_hadamard_128_kernel()

    outputs = kernel(
        inputs=[x_2d],
        template=[
            ("N", n),
            ("NORMALIZE", 1 if normalize else 0),
        ],
        grid=(n, 1, 1),
        threadgroup=(32, 1, 1),  # One simdgroup per vector
        output_shapes=[(n, block_size)],
        output_dtypes=[dtype],
    )

    return outputs[0].reshape(orig_shape)


# ---------------------------------------------------------------------------
# Decode GEMV kernels (optimized for M=1)
# ---------------------------------------------------------------------------

# Decode kernel header - shared primitives
_DECODE_GEMV_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// FP4 E2M1 branchless dequant (same as main GEMM header)
inline half dequant_fp4_scalar_decode(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half sub_mag = half(man_bit) * half(0.25h);
    half norm_mag = half(1u << (exp_bits - 1)) * (half(1.0h) + half(man_bit) * half(0.5h));
    half magnitude = select(norm_mag, sub_mag, exp_bits == 0);
    return select(magnitude, -magnitude, bool(sign_bit));
}

inline half dequant_fp4_scaled_decode(uint nibble, float scale) {
    return (half)(float(dequant_fp4_scalar_decode(nibble)) * scale);
}

inline uint div_ceil_decode(uint a, uint b) {
    return (a + b - 1) / b;
}
"""

# Source for decode GEMV with 256-wide tiles (2 columns per thread)
_DECODE_GEMV_FP4_SOURCE = """
    // Decode GEMV kernel for M=1
    // C[1,N] = A[1,K] @ dequant(B[K/8,N], scales[K/group_size,N])
    //
    // Grid: (ceil(N / 256), 1, 1)
    // Threadgroup: 128 threads

    const uint TILE_N = 256;
    const uint COLS_PER_THREAD = 2;
    const uint FP4_PER_UINT = 8;

    uint tgid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

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
        #pragma unroll
        for (uint i = 0; i < 8; i++) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        // Process column 0
        if (col0 < N && pack_idx < k_packs) {
            uint packed0 = B[pack_idx * N + col0];
            half scale0 = scales[group_idx * N + col0];
            float fscale0 = (float)scale0;

            #pragma unroll
            for (uint i = 0; i < 8; i++) {
                if ((k_base + i) < K) {
                    uint nibble = (packed0 >> (i * 4)) & 0xF;
                    half w_val = dequant_fp4_scaled_decode(nibble, fscale0);
                    acc0 += float(a_vals[i]) * float(w_val);
                }
            }
        }

        // Process column 1
        if (col1 < N && pack_idx < k_packs) {
            uint packed1 = B[pack_idx * N + col1];
            half scale1 = scales[group_idx * N + col1];
            float fscale1 = (float)scale1;

            #pragma unroll
            for (uint i = 0; i < 8; i++) {
                if ((k_base + i) < K) {
                    uint nibble = (packed1 >> (i * 4)) & 0xF;
                    half w_val = dequant_fp4_scaled_decode(nibble, fscale1);
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
"""

# Source for decode GEMV with 512-wide tiles (4 columns per thread)
_DECODE_GEMV_FP4_WIDE_SOURCE = """
    // Wide decode GEMV kernel for M=1
    // Better memory coalescing with 4 columns per thread
    //
    // Grid: (ceil(N / 512), 1, 1)
    // Threadgroup: 128 threads

    const uint TILE_N = 512;
    const uint COLS_PER_THREAD = 4;
    const uint FP4_PER_UINT = 8;

    uint tgid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    uint col_base = tgid * TILE_N + tid * COLS_PER_THREAD;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k_packs = K / FP4_PER_UINT;

    // Stream through K dimension
    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        uint pack_idx = k_base / FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;

        // Load 8 A values
        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; i++) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        // Process 4 columns
        #pragma unroll
        for (uint c = 0; c < 4; c++) {
            uint col = col_base + c;
            if (col < N && pack_idx < k_packs) {
                uint packed = B[pack_idx * N + col];
                half scale = scales[group_idx * N + col];
                float fscale = (float)scale;

                #pragma unroll
                for (uint i = 0; i < 8; i++) {
                    if ((k_base + i) < K) {
                        uint nibble = (packed >> (i * 4)) & 0xF;
                        half w_val = dequant_fp4_scaled_decode(nibble, fscale);
                        acc[c] += float(a_vals[i]) * float(w_val);
                    }
                }
            }
        }
    }

    // Store 4 outputs
    #pragma unroll
    for (uint c = 0; c < 4; c++) {
        uint col = col_base + c;
        if (col < N) {
            out[col] = half(acc[c]);
        }
    }
"""

# Source for batched decode GEMV (M=1..8 sequences)
_DECODE_GEMV_FP4_BATCHED_SOURCE = """
    // Batched decode GEMV for M=1..8 sequences
    // Each thread handles one (row, col) output element
    //
    // Grid: (ceil(N / 32), M, 1)
    // Threadgroup: (32, 4, 1)

    const uint FP4_PER_UINT = 8;

    uint row = threadgroup_position_in_grid.y;
    uint col = threadgroup_position_in_grid.x * 32 + thread_position_in_threadgroup.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    uint k_packs = K / FP4_PER_UINT;

    // Base pointer for this row's activations
    device const half* A_row = A + row * K;

    // Stream through K
    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        uint pack_idx = k_base / FP4_PER_UINT;
        uint group_idx = k_base / GROUP_SIZE;

        if (pack_idx >= k_packs) break;

        uint packed = B[pack_idx * N + col];
        half scale = scales[group_idx * N + col];
        float fscale = (float)scale;

        #pragma unroll
        for (uint i = 0; i < 8; i++) {
            uint k_idx = k_base + i;
            if (k_idx < K) {
                half a_val = A_row[k_idx];
                uint nibble = (packed >> (i * 4)) & 0xF;
                half w_val = dequant_fp4_scaled_decode(nibble, fscale);
                acc += float(a_val) * float(w_val);
            }
        }
    }

    out[row * N + col] = half(acc);
"""

# Cached decode kernel objects
_decode_gemv_fp4_kernel: object | None = None
_decode_gemv_fp4_wide_kernel: object | None = None
_decode_gemv_fp4_batched_kernel: object | None = None


def _get_decode_gemv_fp4_kernel() -> object:
    """Get or create the standard decode GEMV kernel (256-wide tiles)."""
    global _decode_gemv_fp4_kernel
    if _decode_gemv_fp4_kernel is None:
        _decode_gemv_fp4_kernel = mx.fast.metal_kernel(
            name="decode_gemv_fp4",
            input_names=["A", "B", "scales"],
            output_names=["out"],
            source=_DECODE_GEMV_FP4_SOURCE,
            header=_DECODE_GEMV_HEADER,
            ensure_row_contiguous=True,
        )
    return _decode_gemv_fp4_kernel


def _get_decode_gemv_fp4_wide_kernel() -> object:
    """Get or create the wide decode GEMV kernel (512-wide tiles)."""
    global _decode_gemv_fp4_wide_kernel
    if _decode_gemv_fp4_wide_kernel is None:
        _decode_gemv_fp4_wide_kernel = mx.fast.metal_kernel(
            name="decode_gemv_fp4_wide",
            input_names=["A", "B", "scales"],
            output_names=["out"],
            source=_DECODE_GEMV_FP4_WIDE_SOURCE,
            header=_DECODE_GEMV_HEADER,
            ensure_row_contiguous=True,
        )
    return _decode_gemv_fp4_wide_kernel


def _get_decode_gemv_fp4_batched_kernel() -> object:
    """Get or create the batched decode GEMV kernel (M=1..8)."""
    global _decode_gemv_fp4_batched_kernel
    if _decode_gemv_fp4_batched_kernel is None:
        _decode_gemv_fp4_batched_kernel = mx.fast.metal_kernel(
            name="decode_gemv_fp4_batched",
            input_names=["A", "B", "scales"],
            output_names=["out"],
            source=_DECODE_GEMV_FP4_BATCHED_SOURCE,
            header=_DECODE_GEMV_HEADER,
            ensure_row_contiguous=True,
        )
    return _decode_gemv_fp4_batched_kernel


def decode_gemv_fp4(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    group_size: int = 128,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Decode GEMV for M=1: C[1,N] = A[1,K] @ dequant(B[K/8,N], scales).

    Optimized for single-token decode phase where M=1. Uses TILE_N=256
    with 2 columns per thread for good cache utilization.

    Expected speedup vs marlin_gemm_fp4 for M=1: ~3-4x.

    Args:
        A: Activation vector [K] or [1, K].
        B_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        group_size: Number of K-elements per quantization group.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).

    Returns:
        Output vector [N] or [1, N] depending on input shape.
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    # Normalize input shape
    squeeze_output = False
    if A.ndim == 1:
        A = A.reshape(1, -1)
        squeeze_output = True

    M, K = A.shape
    K_packed, N = B_packed.shape

    if K != K_packed * FP4_PER_UINT:
        raise ValueError(f"K mismatch: A has K={K}, B_packed implies K={K_packed * FP4_PER_UINT}")

    if M > 1:
        # Use batched kernel for M > 1
        return decode_gemv_fp4_batched(A, B_packed, scales, group_size, dtype)

    # Single-token decode
    A_flat = A.reshape(-1).astype(dtype)

    kernel = _get_decode_gemv_fp4_kernel()
    grid_x = (N + 255) // 256

    outputs = kernel(
        inputs=[A_flat, B_packed, scales],
        template=[
            ("K", K),
            ("N", N),
            ("GROUP_SIZE", group_size),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[dtype],
    )

    result = outputs[0]
    if not squeeze_output:
        result = result.reshape(1, N)
    return result


def decode_gemv_fp4_wide(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    group_size: int = 128,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Wide decode GEMV for M=1 with better memory coalescing.

    Uses TILE_N=512 with 4 columns per thread. Better for larger N
    where memory bandwidth is the bottleneck.

    Args:
        A: Activation vector [K] or [1, K].
        B_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        group_size: Number of K-elements per quantization group.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).

    Returns:
        Output vector [N] or [1, N] depending on input shape.
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    # Normalize input shape
    squeeze_output = False
    if A.ndim == 1:
        A = A.reshape(1, -1)
        squeeze_output = True

    M, K = A.shape
    K_packed, N = B_packed.shape

    if K != K_packed * FP4_PER_UINT:
        raise ValueError(f"K mismatch: A has K={K}, B_packed implies K={K_packed * FP4_PER_UINT}")

    if M > 1:
        return decode_gemv_fp4_batched(A, B_packed, scales, group_size, dtype)

    A_flat = A.reshape(-1).astype(dtype)

    kernel = _get_decode_gemv_fp4_wide_kernel()
    grid_x = (N + 511) // 512

    outputs = kernel(
        inputs=[A_flat, B_packed, scales],
        template=[
            ("K", K),
            ("N", N),
            ("GROUP_SIZE", group_size),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[dtype],
    )

    result = outputs[0]
    if not squeeze_output:
        result = result.reshape(1, N)
    return result


def decode_gemv_fp4_batched(
    A: mx.array,
    B_packed: mx.array,
    scales: mx.array,
    group_size: int = 128,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """
    Batched decode GEMV for M=1..8 sequences.

    For small batch decode where multiple sequences are generating
    in parallel. Each thread handles one (row, col) output element.

    Args:
        A: Activations [M, K] where M <= 8.
        B_packed: Packed FP4 weights [K/8, N] as uint32.
        scales: Per-group scales [K/group_size, N].
        group_size: Number of K-elements per quantization group.
        dtype: Output dtype. If None, uses DTypeConfig default (bf16).

    Returns:
        Output matrix [M, N].
    """
    if dtype is None:
        dtype = get_default_config().mlx_activations

    if A.ndim == 1:
        A = A.reshape(1, -1)

    M, K = A.shape
    K_packed, N = B_packed.shape

    if K != K_packed * FP4_PER_UINT:
        raise ValueError(f"K mismatch: A has K={K}, B_packed implies K={K_packed * FP4_PER_UINT}")

    if M > 8:
        # Fall back to full GEMM for larger batches
        return marlin_gemm_fp4(A, B_packed, scales, group_size, dtype)

    A_2d = A.astype(dtype)

    kernel = _get_decode_gemv_fp4_batched_kernel()
    grid_x = (N + 31) // 32

    outputs = kernel(
        inputs=[A_2d, B_packed, scales],
        template=[
            ("M", M),
            ("K", K),
            ("N", N),
            ("GROUP_SIZE", group_size),
        ],
        grid=(grid_x, M, 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(M, N)],
        output_dtypes=[dtype],
    )

    return outputs[0]


def select_decode_kernel(M: int, N: int, K: int) -> str:
    """
    Select optimal decode kernel based on problem dimensions.

    Args:
        M: Batch size (typically 1 for decode).
        N: Output dimension.
        K: Input dimension.

    Returns:
        Kernel name to use:
        - "decode_gemv_fp4": Standard M=1 decode (256-wide tiles)
        - "decode_gemv_fp4_wide": M=1 with larger N (512-wide tiles)
        - "decode_gemv_fp4_batched": M=2..8 small batch decode
        - "marlin_gemm_fp4": M > 8, fall back to full GEMM
    """
    if M > 8:
        return "marlin_gemm_fp4"

    if M > 1:
        return "decode_gemv_fp4_batched"

    # M == 1 decode
    if N >= 512:
        return "decode_gemv_fp4_wide"
    else:
        return "decode_gemv_fp4"
