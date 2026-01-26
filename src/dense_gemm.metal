// dense_gemm.metal - Optimized GEMM tiling for dense transformers
//
// Dense models have predictable, regular GEMM shapes:
//   - Q/K/V proj: [batch, seq, hidden] @ [hidden, n_heads * head_dim]
//   - MLP up:     [batch, seq, hidden] @ [hidden, intermediate]
//   - MLP down:   [batch, seq, intermediate] @ [intermediate, hidden]
//
// This file provides specialized kernels tuned for these shapes:
//   1. dense_gemm_prefill_*      - Large M (seq > 64), Split-K reduction
//   2. dense_gemm_decode_*       - M=1 strided GEMV
//   3. dense_gemm_small_batch_*  - M=1-16 batched decode
//   4. dense_gemm_fused_gate_up  - Fused gate*up for SwiGLU/GeGLU MLP
//
// Pre-computed optimal tile configurations:
//   Shape [M, N, K]           | Tile [TM, TN, TK] | Reason
//   --------------------------|-------------------|---------------------------
//   [1, N, K]                 | GEMV              | Decode, N-parallel
//   [1-16, N, K]              | [16, 128, 32]     | Small batch decode
//   [16-64, N, K]             | [64, 64, 32]      | Balanced (default)
//   [64-256, N, K]            | [128, 64, 16]     | Large M prefill
//   [256+, N, K] K>4096       | Split-K           | Very large, reduce K in parallel

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "bf16_compat.metal"

using namespace metal;

// ---------------------------------------------------------------------------
// Common tile configurations
// ---------------------------------------------------------------------------

// Default: Balanced for moderate M
constant constexpr uint DENSE_TILE_M = 64;
constant constexpr uint DENSE_TILE_N = 64;
constant constexpr uint DENSE_TILE_K = 32;

// Large M: Better for prefill with M > 64
constant constexpr uint PREFILL_TILE_M = 128;
constant constexpr uint PREFILL_TILE_N = 64;
constant constexpr uint PREFILL_TILE_K = 16;

// Small batch: Optimized for M=1-16 decode
constant constexpr uint SMALL_TILE_M = 16;
constant constexpr uint SMALL_TILE_N = 128;
constant constexpr uint SMALL_TILE_K = 32;

// Decode GEMV: Wide N coverage per threadgroup
constant constexpr uint DECODE_TILE_N = 512;
constant constexpr uint DECODE_COLS_PER_THREAD = 4;

// Common constants
constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 128
constant constexpr uint FP4_PER_UINT = 8;
constant constexpr uint NUM_BUFFERS = 2;

// Split-K configuration (used in host-side dispatch, documented here)
// constant constexpr uint MAX_SPLIT_K = 16;  // Maximum K-parallel slices

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

// FP4 E2M1 dequantization
inline half dequant_fp4_dense(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.25h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }

    return sign_bit ? -magnitude : magnitude;
}

inline half dequant_fp4_scaled(uint nibble, half scale) {
    float result = (float)dequant_fp4_dense(nibble) * (float)scale;
    return isfinite(result) ? (half)result : half(0.0h);
}

inline float dequant_fp4_scaled_fp32(uint nibble, half scale) {
    float result = float(dequant_fp4_dense(nibble)) * float(scale);
    return isfinite(result) ? result : 0.0f;
}

inline void unpack_fp4x8(uint packed, half scale, thread half* out) {
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = dequant_fp4_scaled(nibble, scale);
    }
}

inline void unpack_fp4x8_fp32(uint packed, half scale, thread float* out) {
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (packed >> (i * 4)) & 0xF;
        out[i] = dequant_fp4_scaled_fp32(nibble, scale);
    }
}

// ===========================================================================
// Section 1: Decode GEMV Kernels (M=1)
//
// For autoregressive decode, M=1 means traditional GEMM is wasteful.
// Instead, we maximize N-parallelism and stream through K.
// ===========================================================================

// ---------------------------------------------------------------------------
// dense_decode_gemv_fp4 - Optimized M=1 vector-matrix multiply
//
// Each threadgroup handles DECODE_TILE_N (512) output columns.
// Each thread handles 4 output columns, streaming through K.
// No threadgroup memory for B - register-resident dequantization.
//
// Dispatch: Grid [ceil(N / 512), 1, 1], Threadgroup [128, 1, 1]
// ---------------------------------------------------------------------------

kernel void dense_decode_gemv_fp4(
    device const half* A         [[buffer(0)]],  // [1, K]
    device const uint* B         [[buffer(1)]],  // [K/8, N] packed FP4
    device const half* scales    [[buffer(2)]],  // [K/group_size, N]
    device half* C               [[buffer(3)]],  // [1, N]
    constant uint& M             [[buffer(4)]],  // Always 1
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint tgid_x                  [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]]
) {
    const uint tg_col_base = tgid_x * DECODE_TILE_N;
    const uint col_base = tg_col_base + tid * DECODE_COLS_PER_THREAD;

    // 4 accumulators in FP32 for precision
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint k_packs = div_ceil(K, FP4_PER_UINT);

    // Stream through K dimension
    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        const uint pack_idx = k_base / FP4_PER_UINT;
        const uint group_idx = k_base / group_size;

        // Load A values (shared across all columns for M=1)
        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        // Process 4 columns with unrolled inner loop
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            uint col = col_base + c;
            if (col < N && pack_idx < k_packs) {
                uint packed = B[pack_idx * N + col];
                half scale = scales[group_idx * N + col];

                // Inline dequant + dot product
                #pragma unroll
                for (uint i = 0; i < 8; ++i) {
                    uint nibble = (packed >> (i * 4)) & 0xF;
                    half b_val = dequant_fp4_scaled(nibble, scale);
                    if ((k_base + i) < K) {
                        acc[c] += float(a_vals[i]) * float(b_val);
                    }
                }
            }
        }
    }

    // Store 4 outputs
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        uint col = col_base + c;
        if (col < N) {
            C[col] = half(acc[c]);
        }
    }
}

// ---------------------------------------------------------------------------
// dense_decode_gemv_fp4_tiled - With threadgroup memory for A
//
// For very large K, cache A in threadgroup memory to reduce global traffic.
// 128 threads load 256 A elements, reuse across all N columns.
//
// Dispatch: Grid [ceil(N / 512), 1, 1], Threadgroup [128, 1, 1]
// ---------------------------------------------------------------------------

constant constexpr uint DECODE_TILE_K = 256;

kernel void dense_decode_gemv_fp4_tiled(
    device const half* A         [[buffer(0)]],
    device const uint* B         [[buffer(1)]],
    device const half* scales    [[buffer(2)]],
    device half* C               [[buffer(3)]],
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint tgid_x                  [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]]
) {
    threadgroup half A_tile[DECODE_TILE_K];

    const uint tg_col_base = tgid_x * DECODE_TILE_N;
    const uint col_base = tg_col_base + tid * DECODE_COLS_PER_THREAD;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint k_packs = div_ceil(K, FP4_PER_UINT);

    // Outer loop: tile across K
    for (uint k_tile = 0; k_tile < K; k_tile += DECODE_TILE_K) {
        // Cooperative A tile load (128 threads load 256 elements = 2 per thread)
        const uint elems_per_thread = DECODE_TILE_K / THREADS_PER_TG;  // 2
        #pragma unroll
        for (uint i = 0; i < elems_per_thread; ++i) {
            uint k_idx = k_tile + tid * elems_per_thread + i;
            A_tile[tid * elems_per_thread + i] = (k_idx < K) ? A[k_idx] : half(0.0h);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner loop: process K tile in FP4 packs
        uint k_tile_end = min(k_tile + DECODE_TILE_K, K);
        for (uint k_base = k_tile; k_base < k_tile_end; k_base += FP4_PER_UINT) {
            const uint pack_idx = k_base / FP4_PER_UINT;
            const uint group_idx = k_base / group_size;
            const uint tile_offset = k_base - k_tile;

            // Load A from shared memory
            half a_vals[8];
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                uint local_k = tile_offset + i;
                a_vals[i] = (local_k < DECODE_TILE_K && (k_base + i) < K)
                            ? A_tile[local_k] : half(0.0h);
            }

            // Process 4 columns
            #pragma unroll
            for (uint c = 0; c < 4; ++c) {
                uint col = col_base + c;
                if (col < N && pack_idx < k_packs) {
                    uint packed = B[pack_idx * N + col];
                    half scale = scales[group_idx * N + col];

                    #pragma unroll
                    for (uint i = 0; i < 8; ++i) {
                        if ((k_base + i) < K) {
                            uint nibble = (packed >> (i * 4)) & 0xF;
                            half b_val = dequant_fp4_scaled(nibble, scale);
                            acc[c] += float(a_vals[i]) * float(b_val);
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        uint col = col_base + c;
        if (col < N) {
            C[col] = half(acc[c]);
        }
    }
}

// ===========================================================================
// Section 2: Small Batch Decode (M=1-16)
//
// For batched decode with small M, use [16, 128, 32] tiles.
// Wide N-tile (128) maximizes output coverage per threadgroup.
// ===========================================================================

// Tile loaders for small batch configuration
inline void load_A_tile_small(
    device const half* A,
    threadgroup half (&A_buf)[SMALL_TILE_M][SMALL_TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    // 16 * 32 = 512 elements, 128 threads = 4 per thread
    const uint elems_per_thread = (SMALL_TILE_M * SMALL_TILE_K) / THREADS_PER_TG;  // 4
    #pragma unroll
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / SMALL_TILE_K;
        uint col = flat_idx % SMALL_TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = 0.0h;
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_buf[row][col] = val;
    }
}

inline void load_B_tile_dequant_small(
    device const uint* B,
    device const half* scales,
    threadgroup half (&B_buf)[SMALL_TILE_K][SMALL_TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint num_groups = div_ceil(K, group_size);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    // 32 * 128 = 4096 elements, packed = 512, 128 threads = 4 packed per thread
    const uint packed_per_thread = (SMALL_TILE_K * SMALL_TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);  // 4

    #pragma unroll
    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (SMALL_TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (SMALL_TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = 1.0h;
        if (global_n < N && global_k_base < K && scale_k < num_groups) {
            s = scales[scale_k * N + global_n];
        }

        uint packed = 0;
        uint b_row = global_k_base / FP4_PER_UINT;
        if (global_n < N && b_row < k_packs && global_k_base < K) {
            packed = B[b_row * N + global_n];
        }

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        half vals[8];
        unpack_fp4x8(packed, s, vals);
        #pragma unroll
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < SMALL_TILE_K; ++v) {
            if (n_idx < SMALL_TILE_N) {
                uint global_k = global_k_base + v;
                B_buf[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : 0.0h;
            }
        }
    }
}

inline void load_A_tile_small_bf16_fp32(
    device const ushort* A,
    threadgroup float (&A_buf)[SMALL_TILE_M][SMALL_TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint blocks_per_row = SMALL_TILE_K / 8;
    const uint total_blocks = SMALL_TILE_M * blocks_per_row;
    if (thread_idx >= total_blocks) {
        return;
    }

    uint block_idx = thread_idx;
    uint row = block_idx / blocks_per_row;
    uint col = (block_idx % blocks_per_row) * 8;
    uint global_row = tg_row + row;
    uint global_col = k_block + col;

    float4 lo = float4(0.0f);
    float4 hi = float4(0.0f);
    if (global_row < M && (global_col + 7) < K) {
        bf16_load_as_float8(A, global_row * K + global_col, lo, hi);
    } else {
        for (uint v = 0; v < 8; ++v) {
            uint gcol = global_col + v;
            float val = 0.0f;
            if (global_row < M && gcol < K) {
                val = bf16_bits_to_float(A[global_row * K + gcol]);
            }
            A_buf[row][col + v] = val;
        }
        return;
    }

    A_buf[row][col + 0] = lo.x;
    A_buf[row][col + 1] = lo.y;
    A_buf[row][col + 2] = lo.z;
    A_buf[row][col + 3] = lo.w;
    A_buf[row][col + 4] = hi.x;
    A_buf[row][col + 5] = hi.y;
    A_buf[row][col + 6] = hi.z;
    A_buf[row][col + 7] = hi.w;
}

inline void load_B_tile_dequant_small_fp32(
    device const uint* B,
    device const half* scales,
    threadgroup float (&B_buf)[SMALL_TILE_K][SMALL_TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint num_groups = div_ceil(K, group_size);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    const uint packed_per_thread =
        (SMALL_TILE_K * SMALL_TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (SMALL_TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (SMALL_TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = half(1.0h);
        if (global_n < N && global_k_base < K && scale_k < num_groups) {
            s = scales[scale_k * N + global_n];
        }

        uint packed = 0;
        uint b_row = global_k_base / FP4_PER_UINT;
        if (global_n < N && b_row < k_packs && global_k_base < K) {
            packed = B[b_row * N + global_n];
        }

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        float vals[8];
        unpack_fp4x8_fp32(packed, s, vals);
        #pragma unroll
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < SMALL_TILE_K; ++v) {
            if (n_idx < SMALL_TILE_N) {
                uint global_k = global_k_base + v;
                B_buf[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : 0.0f;
            }
        }
    }
}

inline void store_small_batch_fp32_bf16(
    thread simdgroup_matrix<float, 8, 8> acc[SMALL_SG_M_TILES][SMALL_SG_N_TILES],
    device ushort* C,
    uint M, uint N,
    uint tg_row, uint tg_col,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane,
    threadgroup float (&staging)[8][8]
) {
    for (uint mi = 0; mi < SMALL_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SMALL_SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row >= M || out_col >= N) {
                continue;
            }

            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (out_row + 8 <= M && out_col + 8 <= N) {
                if (simd_lane < 8) {
                    float4 lo = float4(staging[simd_lane][0],
                                       staging[simd_lane][1],
                                       staging[simd_lane][2],
                                       staging[simd_lane][3]);
                    float4 hi = float4(staging[simd_lane][4],
                                       staging[simd_lane][5],
                                       staging[simd_lane][6],
                                       staging[simd_lane][7]);
                    bf16_store_from_float8(C, (out_row + simd_lane) * N + out_col, lo, hi);
                }
            } else {
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = bf16_from_float_rne(staging[r][c]).bits;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// dense_gemm_small_batch_fp4 - Optimized for M=1-16
//
// Uses [16, 128, 32] tiles with 4 simdgroups.
// SG partition: 2x2 grid, each SG handles 8x64 output
//
// Dispatch: Grid [ceil(N/128), ceil(M/16), 1], Threadgroup [128, 1, 1]
// ---------------------------------------------------------------------------

constant constexpr uint SMALL_SG_M_TILES = 1;  // 1 row of 8x8 = 8 rows per SG
constant constexpr uint SMALL_SG_N_TILES = 8;  // 8 cols of 8x8 = 64 cols per SG
constant constexpr uint SMALL_K_TILES = SMALL_TILE_K / 8;  // 4

kernel void dense_gemm_small_batch_fp4(
    device const half* A         [[buffer(0)]],
    device const uint* B         [[buffer(1)]],
    device const half* scales    [[buffer(2)]],
    device half* C               [[buffer(3)]],
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[NUM_BUFFERS][SMALL_TILE_M][SMALL_TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][SMALL_TILE_K][SMALL_TILE_N];

    const uint tg_row = tgid.y * SMALL_TILE_M;
    const uint tg_col = tgid.x * SMALL_TILE_N;

    // 2x2 simdgroup layout covering 16x128
    // Each SG: 8 rows x 64 cols
    const uint sg_row_offset = (simd_id / 2) * 8;
    const uint sg_col_offset = (simd_id % 2) * 64;

    // Each SG handles 1x8 = 8 tiles of 8x8
    simdgroup_matrix<half, 8, 8> acc[SMALL_SG_M_TILES][SMALL_SG_N_TILES];
    for (uint mi = 0; mi < SMALL_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SMALL_SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(K, SMALL_TILE_K);
    uint buf_compute = 0;

    // Prologue
    load_A_tile_small(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_dequant_small(B, scales, B_tiles[0], K, N, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * SMALL_TILE_K;
        uint next_k = k_offset + SMALL_TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_A_tile_small(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant_small(B, scales, B_tiles[buf_load], K, N, tg_col, next_k, group_size, thread_idx);
        }

        // Compute
        for (uint kst = 0; kst < SMALL_K_TILES; ++kst) {
            for (uint mi = 0; mi < SMALL_SG_M_TILES; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kst * 8],
                               SMALL_TILE_K);

                for (uint ni = 0; ni < SMALL_SG_N_TILES; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag,
                                   &B_tiles[buf_compute][kst * 8][sg_col_offset + ni * 8],
                                   SMALL_TILE_N);

                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store results
    for (uint mi = 0; mi < SMALL_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SMALL_SG_N_TILES; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
            } else if (out_row < M && out_col < N) {
                threadgroup half staging[8][8];
                simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < M && out_col + c < N) {
                        C[(out_row + r) * N + out_col + c] = staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// dense_gemm_small_batch_fp4_fp32acc - BF16 input/output with FP32 accumulation
//
// Uses float tiles and simdgroup_matrix<float> fragments to avoid FP16
// intermediates. Intended for BF16 benchmarking.
// ---------------------------------------------------------------------------

kernel void dense_gemm_small_batch_fp4_fp32acc(
    device const ushort* A       [[buffer(0)]],
    device const uint* B         [[buffer(1)]],
    device const half* scales    [[buffer(2)]],
    device ushort* C             [[buffer(3)]],
    constant uint& M             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float A_tiles[NUM_BUFFERS][SMALL_TILE_M][SMALL_TILE_K];
    threadgroup float B_tiles[NUM_BUFFERS][SMALL_TILE_K][SMALL_TILE_N];
    threadgroup float staging[8][8];

    const uint tg_row = tgid.y * SMALL_TILE_M;
    const uint tg_col = tgid.x * SMALL_TILE_N;

    const uint sg_row_offset = (simd_id / 2) * 8;
    const uint sg_col_offset = (simd_id % 2) * 64;

    simdgroup_matrix<float, 8, 8> acc[SMALL_SG_M_TILES][SMALL_SG_N_TILES];
    for (uint mi = 0; mi < SMALL_SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SMALL_SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(K, SMALL_TILE_K);
    uint buf_compute = 0;

    load_A_tile_small_bf16_fp32(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    load_B_tile_dequant_small_fp32(B, scales, B_tiles[0], K, N, tg_col, 0, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * SMALL_TILE_K;
        uint next_k = k_offset + SMALL_TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_A_tile_small_bf16_fp32(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant_small_fp32(B, scales, B_tiles[buf_load], K, N, tg_col, next_k, group_size, thread_idx);
        }

        for (uint kst = 0; kst < SMALL_K_TILES; ++kst) {
            for (uint mi = 0; mi < SMALL_SG_M_TILES; ++mi) {
                simdgroup_matrix<float, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kst * 8],
                               SMALL_TILE_K);

                for (uint ni = 0; ni < SMALL_SG_N_TILES; ++ni) {
                    simdgroup_matrix<float, 8, 8> b_frag;
                    simdgroup_load(b_frag,
                                   &B_tiles[buf_compute][kst * 8][sg_col_offset + ni * 8],
                                   SMALL_TILE_N);

                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    store_small_batch_fp32_bf16(acc, C, M, N, tg_row, tg_col,
                                sg_row_offset, sg_col_offset, simd_lane, staging);
}

// ===========================================================================
// Section 3: Prefill GEMM with Split-K
//
// For large M (prefill) with large K, use Split-K to parallelize K-reduction.
// Each threadgroup computes a partial sum for a slice of K, then a reduction
// kernel combines the results.
//
// Two-phase approach:
//   Phase 1: dense_gemm_prefill_splitk_fp4 - Compute partial sums
//   Phase 2: dense_splitk_reduce           - Sum partial results
// ===========================================================================

// ---------------------------------------------------------------------------
// Prefill tile loaders using [128, 64, 16] configuration
// ---------------------------------------------------------------------------

inline void load_A_tile_prefill(
    device const half* A,
    threadgroup half (&A_buf)[PREFILL_TILE_M][PREFILL_TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    // 128 * 16 = 2048 elements, 128 threads = 16 per thread
    const uint elems_per_thread = (PREFILL_TILE_M * PREFILL_TILE_K) / THREADS_PER_TG;
    #pragma unroll
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / PREFILL_TILE_K;
        uint col = flat_idx % PREFILL_TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = 0.0h;
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_buf[row][col] = val;
    }
}

inline void load_B_tile_dequant_prefill(
    device const uint* B,
    device const half* scales,
    threadgroup half (&B_buf)[PREFILL_TILE_K][PREFILL_TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint group_size,
    uint thread_idx
) {
    const uint num_groups = div_ceil(K, group_size);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    // 16 * 64 = 1024 elements, packed = 128, 128 threads = 1 packed per thread
    const uint packed_per_thread = (PREFILL_TILE_K * PREFILL_TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);

    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (PREFILL_TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (PREFILL_TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = 1.0h;
        if (global_n < N && global_k_base < K && scale_k < num_groups) {
            s = scales[scale_k * N + global_n];
        }

        uint packed = 0;
        uint b_row = global_k_base / FP4_PER_UINT;
        if (global_n < N && b_row < k_packs && global_k_base < K) {
            packed = B[b_row * N + global_n];
        }

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        half vals[8];
        unpack_fp4x8(packed, s, vals);
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < PREFILL_TILE_K; ++v) {
            if (n_idx < PREFILL_TILE_N) {
                uint global_k = global_k_base + v;
                B_buf[tile_k_base + v][n_idx] = (global_k < K) ? vals[v] : 0.0h;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// dense_gemm_prefill_splitk_fp4 - Split-K parallel reduction for large K
//
// Divides K dimension into `split_k` slices. Each threadgroup computes
// partial sums for one K-slice, writing to partial_out buffer.
//
// Uses [128, 64, 16] tiles optimized for large M.
// SG partition: 4x1 grid, each SG handles 32x64 output
//
// Dispatch: Grid [ceil(N/64), ceil(M/128), split_k], Threadgroup [128, 1, 1]
// ---------------------------------------------------------------------------

// Prefill SG config: 2x2 layout, each SG handles 64x32 (8x4 tiles of 8x8)
constant constexpr uint PREFILL_K_TILES = PREFILL_TILE_K / 8;  // 2

kernel void dense_gemm_prefill_splitk_fp4(
    device const half* A           [[buffer(0)]],
    device const uint* B           [[buffer(1)]],
    device const half* scales      [[buffer(2)]],
    device float* partial_out      [[buffer(3)]],  // [split_k, M, N] FP32 partials
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    constant uint& group_size      [[buffer(7)]],
    constant uint& split_k         [[buffer(8)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[NUM_BUFFERS][PREFILL_TILE_M][PREFILL_TILE_K];
    threadgroup half B_tiles[NUM_BUFFERS][PREFILL_TILE_K][PREFILL_TILE_N];

    const uint tg_row = tgid.y * PREFILL_TILE_M;
    const uint tg_col = tgid.x * PREFILL_TILE_N;
    const uint k_slice = tgid.z;

    // Compute K range for this slice
    const uint k_per_slice = div_ceil(K, split_k);
    const uint k_start = k_slice * k_per_slice;
    const uint k_end = min(k_start + k_per_slice, K);

    if (k_start >= K) return;

    // 4x1 simdgroup layout covering 128x64
    // Each SG: 32 rows x 64 cols (but only need 8 cols per SG for correct tiling)
    // Corrected: 2x2 layout, each SG handles 64x32
    const uint sg_row_offset = (simd_id / 2) * 64;
    const uint sg_col_offset = (simd_id % 2) * 32;

    // Each SG handles 8x4 = 32 tiles of 8x8
    simdgroup_matrix<float, 8, 8> acc[8][4];  // FP32 for Split-K precision
    for (uint mi = 0; mi < 8; ++mi) {
        for (uint ni = 0; ni < 4; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(k_end - k_start, PREFILL_TILE_K);
    uint buf_compute = 0;

    // Prologue
    load_A_tile_prefill(A, A_tiles[0], M, K, tg_row, k_start, thread_idx);
    load_B_tile_dequant_prefill(B, scales, B_tiles[0], K, N, tg_col, k_start, group_size, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop over K slice
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = k_start + kt * PREFILL_TILE_K;
        uint next_k = k_offset + PREFILL_TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < k_end) {
            load_A_tile_prefill(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            load_B_tile_dequant_prefill(B, scales, B_tiles[buf_load], K, N, tg_col, next_k, group_size, thread_idx);
        }

        // Compute with FP16 inputs, FP32 accumulation
        for (uint kst = 0; kst < PREFILL_K_TILES; ++kst) {
            for (uint mi = 0; mi < 8; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kst * 8],
                               PREFILL_TILE_K);

                for (uint ni = 0; ni < 4; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_frag;
                    simdgroup_load(b_frag,
                                   &B_tiles[buf_compute][kst * 8][sg_col_offset + ni * 8],
                                   PREFILL_TILE_N);

                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store FP32 partial results
    const uint partial_stride = M * N;
    device float* partial_base = partial_out + k_slice * partial_stride;

    for (uint mi = 0; mi < 8; ++mi) {
        for (uint ni = 0; ni < 4; ++ni) {
            uint out_row = tg_row + sg_row_offset + mi * 8;
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], partial_base + out_row * N + out_col, N);
            } else if (out_row < M && out_col < N) {
                threadgroup float staging[8][8];
                simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    if (out_row + r < M && out_col + c < N) {
                        partial_base[(out_row + r) * N + out_col + c] = staging[r][c];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// dense_splitk_reduce - Reduce Split-K partial sums
//
// Sums partial_in[0..split_k-1] into final FP16 output.
//
// Dispatch: Grid [ceil(N/256), ceil(M/1), 1], Threadgroup [256, 1, 1]
// ---------------------------------------------------------------------------

kernel void dense_splitk_reduce(
    device const float* partial_in  [[buffer(0)]],  // [split_k, M, N]
    device half* C                  [[buffer(1)]],  // [M, N]
    constant uint& M                [[buffer(2)]],
    constant uint& N                [[buffer(3)]],
    constant uint& split_k          [[buffer(4)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint3 tid3                      [[thread_position_in_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint col_base = tgid.x * 256 + tid;
    const uint row = tgid.y;

    if (row >= M || col_base >= N) return;

    const uint partial_stride = M * N;
    float sum = 0.0f;

    for (uint s = 0; s < split_k; ++s) {
        sum += partial_in[s * partial_stride + row * N + col_base];
    }

    C[row * N + col_base] = half(sum);
}

// ===========================================================================
// Section 4: Fused Gate * Up for SwiGLU/GeGLU MLP
//
// Llama/Qwen MLP uses: output = down_proj(silu(gate_proj(x)) * up_proj(x))
//
// The gate*up fusion computes both projections in one kernel:
//   - Load x once
//   - Compute gate = x @ W_gate, up = x @ W_up in parallel
//   - Apply activation and multiply: output = act(gate) * up
//
// Memory savings: 2x reduction in activation loads
// Compute: Same FLOPs but better cache utilization
// ===========================================================================

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
inline half silu_half(half x) {
    float fx = (float)x;
    return (half)(fx / (1.0f + exp(-fx)));
}

// GELU activation (approximate)
inline half gelu_half(half x) {
    float fx = (float)x;
    return (half)(0.5f * fx * (1.0f + tanh(0.7978845608f * fx * (1.0f + 0.044715f * fx * fx))));
}

// ---------------------------------------------------------------------------
// Tile loaders for fused gate*up kernel (64x64x32 tiling)
// ---------------------------------------------------------------------------

inline void load_A_tile_fused(
    device const half* x,
    threadgroup half (&A_tiles)[NUM_BUFFERS][DENSE_TILE_M][DENSE_TILE_K],
    uint M, uint K,
    uint tg_row, uint buf, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (DENSE_TILE_M * DENSE_TILE_K) / THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / DENSE_TILE_K;
        uint col = flat_idx % DENSE_TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;
        half val = (global_row < M && global_col < K) ? x[global_row * K + global_col] : half(0.0h);
        A_tiles[buf][row][col] = val;
    }
}

inline void load_B_gate_tile_fused(
    device const uint* W_gate,
    device const half* scales_gate,
    threadgroup half (&B_gate_tiles)[NUM_BUFFERS][DENSE_TILE_K][DENSE_TILE_N],
    uint K, uint intermediate, uint group_size,
    uint tg_col, uint buf, uint k_block,
    uint k_packs, uint num_groups,
    uint thread_idx
) {
    const uint packed_per_thread = (DENSE_TILE_K * DENSE_TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (DENSE_TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (DENSE_TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = (global_n < intermediate && global_k_base < K && scale_k < num_groups)
                 ? scales_gate[scale_k * intermediate + global_n] : half(1.0h);

        uint b_row = global_k_base / FP4_PER_UINT;
        uint packed = (global_n < intermediate && b_row < k_packs && global_k_base < K)
                      ? W_gate[b_row * intermediate + global_n] : 0u;

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        half vals[8];
        unpack_fp4x8(packed, s, vals);
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < DENSE_TILE_K; ++v) {
            if (n_idx < DENSE_TILE_N) {
                B_gate_tiles[buf][tile_k_base + v][n_idx] = vals[v];
            }
        }
    }
}

inline void load_B_up_tile_fused(
    device const uint* W_up,
    device const half* scales_up,
    threadgroup half (&B_up_tiles)[NUM_BUFFERS][DENSE_TILE_K][DENSE_TILE_N],
    uint K, uint intermediate, uint group_size,
    uint tg_col, uint buf, uint k_block,
    uint k_packs, uint num_groups,
    uint thread_idx
) {
    const uint packed_per_thread = (DENSE_TILE_K * DENSE_TILE_N) / (THREADS_PER_TG * FP4_PER_UINT);
    for (uint i = 0; i < packed_per_thread; ++i) {
        uint flat_packed_idx = thread_idx * packed_per_thread + i;
        uint n_idx = flat_packed_idx / (DENSE_TILE_K / FP4_PER_UINT);
        uint k_group_in_tile = flat_packed_idx % (DENSE_TILE_K / FP4_PER_UINT);

        uint global_n = tg_col + n_idx;
        uint global_k_base = k_block + k_group_in_tile * FP4_PER_UINT;

        uint scale_k = global_k_base / group_size;
        half s = (global_n < intermediate && global_k_base < K && scale_k < num_groups)
                 ? scales_up[scale_k * intermediate + global_n] : half(1.0h);

        uint b_row = global_k_base / FP4_PER_UINT;
        uint packed = (global_n < intermediate && b_row < k_packs && global_k_base < K)
                      ? W_up[b_row * intermediate + global_n] : 0u;

        uint tile_k_base = k_group_in_tile * FP4_PER_UINT;
        half vals[8];
        unpack_fp4x8(packed, s, vals);
        for (uint v = 0; v < FP4_PER_UINT && (tile_k_base + v) < DENSE_TILE_K; ++v) {
            if (n_idx < DENSE_TILE_N) {
                B_up_tiles[buf][tile_k_base + v][n_idx] = vals[v];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// dense_fused_gate_up_fp4 - Fused gate*up with SiLU activation
//
// Computes: output = silu(x @ W_gate) * (x @ W_up)
//
// W_gate and W_up are concatenated: W_combined[K, 2*intermediate]
// First half is W_gate, second half is W_up.
//
// Alternatively, two separate weight buffers can be used with double the
// dispatch to cover both projections (this variant uses concatenated).
//
// Dispatch: Grid [ceil(intermediate/64), ceil(M/64), 1], Threadgroup [128, 1, 1]
// ---------------------------------------------------------------------------

kernel void dense_fused_gate_up_fp4(
    device const half* x             [[buffer(0)]],  // [M, K]
    device const uint* W_gate        [[buffer(1)]],  // [K/8, intermediate] packed FP4
    device const half* scales_gate   [[buffer(2)]],  // [K/group_size, intermediate]
    device const uint* W_up          [[buffer(3)]],  // [K/8, intermediate] packed FP4
    device const half* scales_up     [[buffer(4)]],  // [K/group_size, intermediate]
    device half* out                 [[buffer(5)]],  // [M, intermediate]
    constant uint& M                 [[buffer(6)]],
    constant uint& K                 [[buffer(7)]],  // hidden_size
    constant uint& intermediate      [[buffer(8)]],  // intermediate_size
    constant uint& group_size        [[buffer(9)]],
    constant uint& activation_type   [[buffer(10)]],  // 0=silu, 1=gelu
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    // Use default 64x64x32 tiling
    threadgroup half A_tiles[NUM_BUFFERS][DENSE_TILE_M][DENSE_TILE_K];
    threadgroup half B_gate_tiles[NUM_BUFFERS][DENSE_TILE_K][DENSE_TILE_N];
    threadgroup half B_up_tiles[NUM_BUFFERS][DENSE_TILE_K][DENSE_TILE_N];

    const uint tg_row = tgid.y * DENSE_TILE_M;
    const uint tg_col = tgid.x * DENSE_TILE_N;

    const uint sg_row_offset = (simd_id / 2) * (2 * 8);  // SG_M_TILES=2
    const uint sg_col_offset = (simd_id % 2) * (4 * 8);  // SG_N_TILES=4

    // Two sets of accumulators: one for gate, one for up
    simdgroup_matrix<half, 8, 8> acc_gate[2][4];
    simdgroup_matrix<half, 8, 8> acc_up[2][4];
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint ni = 0; ni < 4; ++ni) {
            acc_gate[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            acc_up[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(K, DENSE_TILE_K);
    const uint k_packs = div_ceil(K, FP4_PER_UINT);
    const uint num_groups = div_ceil(K, group_size);
    uint buf_compute = 0;

    // Prologue
    load_A_tile_fused(x, A_tiles, M, K, tg_row, 0, 0, thread_idx);
    load_B_gate_tile_fused(W_gate, scales_gate, B_gate_tiles, K, intermediate, group_size, tg_col, 0, 0, k_packs, num_groups, thread_idx);
    load_B_up_tile_fused(W_up, scales_up, B_up_tiles, K, intermediate, group_size, tg_col, 0, 0, k_packs, num_groups, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * DENSE_TILE_K;
        uint next_k = k_offset + DENSE_TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_A_tile_fused(x, A_tiles, M, K, tg_row, buf_load, next_k, thread_idx);
            load_B_gate_tile_fused(W_gate, scales_gate, B_gate_tiles, K, intermediate, group_size, tg_col, buf_load, next_k, k_packs, num_groups, thread_idx);
            load_B_up_tile_fused(W_up, scales_up, B_up_tiles, K, intermediate, group_size, tg_col, buf_load, next_k, k_packs, num_groups, thread_idx);
        }

        // Compute both gate and up projections
        const uint K_TILES_INNER = DENSE_TILE_K / 8;
        for (uint kst = 0; kst < K_TILES_INNER; ++kst) {
            for (uint mi = 0; mi < 2; ++mi) {
                simdgroup_matrix<half, 8, 8> a_frag;
                simdgroup_load(a_frag,
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kst * 8],
                               DENSE_TILE_K);

                for (uint ni = 0; ni < 4; ++ni) {
                    simdgroup_matrix<half, 8, 8> b_gate_frag;
                    simdgroup_load(b_gate_frag,
                                   &B_gate_tiles[buf_compute][kst * 8][sg_col_offset + ni * 8],
                                   DENSE_TILE_N);
                    simdgroup_multiply_accumulate(acc_gate[mi][ni], a_frag, b_gate_frag, acc_gate[mi][ni]);

                    simdgroup_matrix<half, 8, 8> b_up_frag;
                    simdgroup_load(b_up_frag,
                                   &B_up_tiles[buf_compute][kst * 8][sg_col_offset + ni * 8],
                                   DENSE_TILE_N);
                    simdgroup_multiply_accumulate(acc_up[mi][ni], a_frag, b_up_frag, acc_up[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Apply activation and multiply: out = act(gate) * up
    // Store through staging for element-wise fusion
    threadgroup half gate_staging[16][32];
    threadgroup half up_staging[16][32];

    for (uint mi = 0; mi < 2; ++mi) {
        for (uint ni = 0; ni < 4; ++ni) {
            // Store both to staging
            simdgroup_store(acc_gate[mi][ni], &gate_staging[mi * 8][ni * 8], 32);
            simdgroup_store(acc_up[mi][ni], &up_staging[mi * 8][ni * 8], 32);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fused activation + multiply + store
    for (uint elem = simd_lane; elem < 16 * 32; elem += 32) {
        uint r = elem / 32;
        uint c = elem % 32;
        uint out_row = tg_row + sg_row_offset + r;
        uint out_col = tg_col + sg_col_offset + c;

        if (out_row < M && out_col < intermediate) {
            half gate_val = gate_staging[r][c];
            half up_val = up_staging[r][c];

            half activated;
            if (activation_type == 0) {
                activated = silu_half(gate_val);
            } else {
                activated = gelu_half(gate_val);
            }

            out[out_row * intermediate + out_col] = activated * up_val;
        }
    }
}

// ===========================================================================
// Section 5: Transformer-specific GEMM shapes (Common Model Configurations)
//
// Pre-computed optimal dispatches for popular models:
//
// Llama-7B:  hidden=4096, intermediate=11008, n_heads=32, head_dim=128
// Llama-13B: hidden=5120, intermediate=13824, n_heads=40, head_dim=128
// Llama-70B: hidden=8192, intermediate=28672, n_heads=64, head_dim=128
//
// Qwen2-7B:  hidden=3584, intermediate=18944, n_heads=28, head_dim=128
// Qwen2-72B: hidden=8192, intermediate=24576, n_heads=64, head_dim=128
//
// Attention projections: [batch*seq, hidden] @ [hidden, n_heads*head_dim]
// MLP projections:       [batch*seq, hidden] @ [hidden, intermediate]
//                        [batch*seq, intermediate] @ [intermediate, hidden]
// ===========================================================================

// Structure for kernel selection
struct DenseGEMMConfig {
    uint tile_m;
    uint tile_n;
    uint tile_k;
    uint split_k;     // 0 = no split, > 0 = split factor
    bool use_gemv;    // True for M=1 decode
};

// Host-side selection would use this logic (shown for documentation):
// inline DenseGEMMConfig select_config(uint M, uint N, uint K) {
//     if (M == 1) {
//         return {0, DECODE_TILE_N, 0, 0, true};  // GEMV
//     } else if (M <= 16) {
//         return {SMALL_TILE_M, SMALL_TILE_N, SMALL_TILE_K, 0, false};
//     } else if (M <= 64) {
//         return {DENSE_TILE_M, DENSE_TILE_N, DENSE_TILE_K, 0, false};
//     } else if (K <= 4096) {
//         return {PREFILL_TILE_M, PREFILL_TILE_N, PREFILL_TILE_K, 0, false};
//     } else {
//         // Large K: use split-K
//         uint split = min(MAX_SPLIT_K, (K + 4095) / 4096);
//         return {PREFILL_TILE_M, PREFILL_TILE_N, PREFILL_TILE_K, split, false};
//     }
// }

// ===========================================================================
// Section 6: Batched decode for multiple sequences
//
// For serving multiple sequences in parallel (M = num_sequences * 1),
// we can batch the GEMV operations.
// ===========================================================================

kernel void dense_batched_decode_gemv_fp4(
    device const half* A         [[buffer(0)]],  // [batch, K]
    device const uint* B         [[buffer(1)]],  // [K/8, N] packed FP4
    device const half* scales    [[buffer(2)]],  // [K/group_size, N]
    device half* C               [[buffer(3)]],  // [batch, N]
    constant uint& batch         [[buffer(4)]],  // Number of sequences
    constant uint& N             [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& group_size    [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint3 tid3                   [[thread_position_in_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint seq_idx = tgid.y;
    if (seq_idx >= batch) return;

    const uint tg_col_base = tgid.x * DECODE_TILE_N;
    const uint col_base = tg_col_base + tid * DECODE_COLS_PER_THREAD;

    // Point to this sequence's activation vector
    device const half* A_seq = A + seq_idx * K;
    device half* C_seq = C + seq_idx * N;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint k_packs = div_ceil(K, FP4_PER_UINT);

    for (uint k_base = 0; k_base < K; k_base += FP4_PER_UINT) {
        const uint pack_idx = k_base / FP4_PER_UINT;
        const uint group_idx = k_base / group_size;

        half a_vals[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint k_idx = k_base + i;
            a_vals[i] = (k_idx < K) ? A_seq[k_idx] : half(0.0h);
        }

        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            uint col = col_base + c;
            if (col < N && pack_idx < k_packs) {
                uint packed = B[pack_idx * N + col];
                half scale = scales[group_idx * N + col];

                #pragma unroll
                for (uint i = 0; i < 8; ++i) {
                    uint nibble = (packed >> (i * 4)) & 0xF;
                    half b_val = dequant_fp4_scaled(nibble, scale);
                    if ((k_base + i) < K) {
                        acc[c] += float(a_vals[i]) * float(b_val);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        uint col = col_base + c;
        if (col < N) {
            C_seq[col] = half(acc[c]);
        }
    }
}
