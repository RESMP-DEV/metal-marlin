// gemm_int8.metal - High-performance W8A16 GEMM kernel for INT8 weights with FP16 activations
//
// Optimized for Apple M4 Max GPU (40 cores, 32KB threadgroup memory, 546 GB/s bandwidth)
//
// Key optimizations:
//   1. Direct INT8 dequantization (no LUT needed - simple int-to-float conversion)
//   2. Double-buffered activation tiles, on-the-fly weight dequant
//   3. Simdgroup-local weight staging (256 bytes for 32x32 tiles)
//   4. 32x32x32 tiles with 4 simdgroups (128 threads) - optimal for M4 Max
//   5. Fused dequant-accumulate using dequant_s8x4_fused from dequant_int8.metal
//   6. Coalesced memory access patterns for both activations and packed weights
//
// Target: >65% of theoretical FP16 FLOPS on M4 Max (~18 TFLOPS of 27.8 TFLOPS peak)
//
// C[M,N] = activation(A[M,K] @ dequant(B[K/4,N], scales[K/group_size,N], zeros[K/group_size,N]) + bias)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ===========================================================================
// Tile dimensions - optimized for M4 Max
// ===========================================================================

constant constexpr uint TILE_M = 128;      // Output rows per threadgroup
constant constexpr uint TILE_N = 128;      // Output cols per threadgroup
constant constexpr uint TILE_K = 32;      // K-reduction per mainloop iteration
constant constexpr uint K_TILES = TILE_K / 4;  // 8 simdgroup MMA ops per K-block

// Simdgroup configuration: 4 simdgroups tile 128x128 output
// Each simdgroup handles 32x32 output (4x4 blocks of 8x8 tiles)
constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 128
constant constexpr uint SG_M_TILES = 4;   // 4 rows of 8x8 = 32 rows per simdgroup
constant constexpr uint SG_N_TILES = 4;   // 4 cols of 8x8 = 32 cols per simdgroup

// INT8 packing
constant constexpr uint INT8_PER_UINT = 4;

// Include INT8 dequantization primitives from dequant_int8.metal
// Using the same implementations to ensure consistency

/// Extract signed INT8 byte at position [0-3] from packed uint32 and sign-extend.
inline int8_t extract_s8(uint32_t packed, uint pos) {
    return int8_t((packed >> (pos * 8u)) & 0xFFu);
}

/// Dequantize 4 INT8 values from packed uint32 with symmetric quantization.
/// result = int8_value * scale
inline void dequant_s8x4_fused(uint32_t packed, half scale, thread half4 &out) {
    int8_t b0 = extract_s8(packed, 0);
    int8_t b1 = extract_s8(packed, 1);
    int8_t b2 = extract_s8(packed, 2);
    int8_t b3 = extract_s8(packed, 3);
    
    // Use float intermediate to avoid Metal compiler half-precision issues
    float fscale = float(scale);
    out = half4(float4(float(b0), float(b1), float(b2), float(b3)) * fscale);
}

/// Dequantize 4 INT8 values with asymmetric quantization.
/// result = (int8_value - zero_point) * scale
inline void dequant_s8x4_asym_fused(uint32_t packed, half scale, half zero_point, thread half4 &out) {
    int8_t b0 = extract_s8(packed, 0);
    int8_t b1 = extract_s8(packed, 1);
    int8_t b2 = extract_s8(packed, 2);
    int8_t b3 = extract_s8(packed, 3);
    
    float fscale = float(scale);
    float fzero = float(zero_point);
    out = half4((float4(float(b0), float(b1), float(b2), float(b3)) - fzero) * fscale);
}

/// Unpack and dequant 4 INT8 values to array for staging buffer
inline void unpack_dequant_s8x4(uint32_t packed, half scale, thread half* out) {
    half4 result;
    dequant_s8x4_fused(packed, scale, result);
    out[0] = result.x;
    out[1] = result.y;
    out[2] = result.z;
    out[3] = result.w;
}

/// Unpack and dequant 4 INT8 values with asymmetric quantization
inline void unpack_dequant_s8x4_asym(uint32_t packed, half scale, half zero_point, thread half* out) {
    half4 result;
    dequant_s8x4_asym_fused(packed, scale, zero_point, result);
    out[0] = result.x;
    out[1] = result.y;
    out[2] = result.z;
    out[3] = result.w;
}

// ===========================================================================
// Epilogue modes (matches gemm_epilogue.metal)
// ===========================================================================

enum EpilogueMode : uint {
    EPILOGUE_NONE      = 0,
    EPILOGUE_BIAS      = 1,
    EPILOGUE_GELU      = 2,
    EPILOGUE_SILU      = 3,
    EPILOGUE_BIAS_GELU = 4,
    EPILOGUE_BIAS_SILU = 5,
    EPILOGUE_RELU      = 6,
    EPILOGUE_BIAS_RELU = 7,
};

inline half gelu_fast(half x) {
    half kx = half(1.702h) * x;
    return x / (half(1.0h) + exp(-kx));
}

inline half silu_fast(half x) {
    return x / (half(1.0h) + exp(-x));
}

inline half apply_epilogue(half val, half bias_val, uint mode) {
    // Add bias if needed
    if (mode == EPILOGUE_BIAS || mode == EPILOGUE_BIAS_GELU ||
        mode == EPILOGUE_BIAS_SILU || mode == EPILOGUE_BIAS_RELU) {
        val += bias_val;
    }

    // Apply activation
    switch (mode) {
        case EPILOGUE_GELU:
        case EPILOGUE_BIAS_GELU:
            return gelu_fast(val);
        case EPILOGUE_SILU:
        case EPILOGUE_BIAS_SILU:
            return silu_fast(val);
        case EPILOGUE_RELU:
        case EPILOGUE_BIAS_RELU:
            return max(val, half(0.0h));
        default:
            return val;
    }
}

// ===========================================================================
// Cooperative activation tile loader (all 128 threads)
// ===========================================================================

inline void load_activation_tile(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    // 128 * 32 = 4096 elements / 128 threads = 32 elements per thread
    constexpr uint ELEMS_PER_THREAD = (TILE_M * TILE_K) / THREADS_PER_TG;

    #pragma unroll
    for (uint i = 0; i < ELEMS_PER_THREAD; ++i) {
        uint flat_idx = thread_idx * ELEMS_PER_THREAD + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        uint global_col = k_block + col;

        half val = half(0.0h);
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_buf[row][col] = val;
    }
}

// ===========================================================================
// Main Kernel: Fused INT8 GEMM with symmetric quantization
//
// Architecture:
//   - Double-buffered activation tiles in threadgroup memory (16KB per buffer)
//   - On-the-fly INT8 weight dequant with per-simdgroup staging
//   - Simdgroup MMA with half precision accumulation
//   - Fused bias + activation epilogue
//
// Dispatch: Grid ceil(N/128) x ceil(M/128), threadgroup 128 threads
// ===========================================================================

struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint group_size;
    uint epilogue_mode;
    uint asymmetric;  // 0 = symmetric, 1 = asymmetric
};

kernel void gemm_int8_tiled(
    device const half* A           [[buffer(0)]],  // [M, K] activations (row-major)
    device const uint32_t* B       [[buffer(1)]],  // [K/4, N] packed INT8 weights
    device const half* scales      [[buffer(2)]],  // [K/group_size, N] per-group scales
    device const half* zeros       [[buffer(3)]],  // [K/group_size, N] per-group zeros (optional)
    device half* C                 [[buffer(4)]],  // [M, N] output (row-major)
    device const half* bias        [[buffer(5)]],  // [N] bias (optional, can be nullptr)
    constant GemmParams& params    [[buffer(6)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;
    const uint group_size = params.group_size;
    const uint epilogue_mode = params.epilogue_mode;
    const uint asymmetric = params.asymmetric;

    // --- Threadgroup memory allocation ---
    // Double-buffered A: 2 * 128 * 32 * 2 = 16KB
    // Per-simdgroup B staging: 4 * 8 * 8 * 2 = 512 bytes
    // Total: 16.5KB (52% of 32KB budget - good for M4 Max)
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    // --- Tile assignment ---
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    // Early exit for out-of-bounds threadgroups
    if (tg_row >= M) return;

    // Simdgroup layout: 2x2 grid, each covering 32x32 of 128x128 output
    // SG 0: rows [0,31],   cols [0,31]
    // SG 1: rows [0,31],   cols [32,63]
    // SG 2: rows [32,63],  cols [0,31]
    // SG 3: rows [32,63],  cols [32,63]
    const uint sg_row_offset = (simd_id / 2) * 32;
    const uint sg_col_offset = (simd_id % 2) * 32;

    // --- Initialize accumulators ---
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint k_packs = (K + INT8_PER_UINT - 1) / INT8_PER_UINT;

    uint buf_compute = 0;

    // =========================================================================
    // Prologue: Load first activation tile
    // =========================================================================
    load_activation_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Main pipeline loop
    // =========================================================================
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        // --- Async load next activation tile while computing on current ---
        if (next_k < K) {
            load_activation_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }

        // --- Compute: Fused B dequant + MMA ---
        // For each K sub-tile (4 elements), dequant B on-the-fly and multiply
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 4;
            uint k_pack_idx = k_sub_base / INT8_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            // Load activation fragments for this K sub-tile (reused across N tiles)
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi],
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 4],
                               TILE_K);
            }

            // For each N sub-tile, dequant B and accumulate
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                // --- Fused B dequant: lanes 0-3 each load+dequant 4 values ---
                if (simd_lane < 4) {
                    uint b_col = b_col_base + simd_lane * 2;  // Each lane handles 2 cols
                    half4 dequant_vals[2];  // 2 half4s = 8 values

                    if (b_col + 1 < N && k_pack_idx < k_packs) {
                        // Load 2 packed uint32s for 8 INT8 values
                        uint32_t packed_lo = B[k_pack_idx * N + b_col];
                        uint32_t packed_hi = B[k_pack_idx * N + b_col + 1];
                        half scale = scales[group_idx * N + b_col];
                        half zero_point = zeros ? zeros[group_idx * N + b_col] : half(0.0h);

                        // INT8 dequantization
                        if (asymmetric) {
                            dequant_s8x4_asym_fused(packed_lo, scale, zero_point, dequant_vals[0]);
                            dequant_s8x4_asym_fused(packed_hi, scale, zero_point, dequant_vals[1]);
                        } else {
                            dequant_s8x4_fused(packed_lo, scale, dequant_vals[0]);
                            dequant_s8x4_fused(packed_hi, scale, dequant_vals[1]);
                        }
                    } else {
                        dequant_vals[0] = half4(0.0h);
                        dequant_vals[1] = half4(0.0h);
                    }

                    // Write to staging: B_staging[sg][k][n]
                    // Lane 0 handles cols 0,1 -> rows 0-3 and 4-7
                    // Lane 1 handles cols 2,3 -> rows 0-3 and 4-7
                    // Lane 2 handles cols 4,5 -> rows 0-3 and 4-7
                    // Lane 3 handles cols 6,7 -> rows 0-3 and 4-7
                    #pragma unroll
                    for (uint row = 0; row < 4; ++row) {
                        B_staging[simd_id][row][simd_lane * 2] = dequant_vals[0][row];
                        B_staging[simd_id][row][simd_lane * 2 + 1] = dequant_vals[1][row];
                    }
                }

                // Lightweight simdgroup sync (not full threadgroup barrier)
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Load B fragment and accumulate
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }

        // Wait for next tile load to complete before swapping buffers
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // =========================================================================
    // Epilogue: Apply bias + activation, then store
    // =========================================================================

    // Staging buffer for epilogue operations (shared across simdgroup)
    threadgroup half epilogue_staging[8][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            // Fast path: no epilogue, interior tile -> direct simdgroup_store
            if (epilogue_mode == EPILOGUE_NONE &&
                out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }

            // Store accumulator to staging
            simdgroup_store(acc[mi][ni], &epilogue_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Apply epilogue: each thread handles 2 elements (32 threads * 2 = 64)
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    half val = epilogue_staging[r][c];

                    // Apply bias + activation if mode requires it
                    if (epilogue_mode != EPILOGUE_NONE) {
                        half bias_val = (bias != nullptr && gc < N) ? bias[gc] : half(0.0h);
                        val = apply_epilogue(val, bias_val, epilogue_mode);
                    }

                    C[gr * N + gc] = val;
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ===========================================================================
// Variant: FP32 Accumulation (for numerical stability with large K)
// ===========================================================================

kernel void gemm_int8_tiled_fp32acc(
    device const half* A           [[buffer(0)]],
    device const uint32_t* B       [[buffer(1)]],
    device const half* scales      [[buffer(2)]],
    device const half* zeros       [[buffer(3)]],
    device half* C                 [[buffer(4)]],
    device const half* bias        [[buffer(5)]],
    constant GemmParams& params    [[buffer(6)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;
    const uint group_size = params.group_size;
    const uint epilogue_mode = params.epilogue_mode;
    const uint asymmetric = params.asymmetric;

    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    if (tg_row >= M) return;

    const uint sg_row_offset = (simd_id / 2) * 32;
    const uint sg_col_offset = (simd_id % 2) * 32;

    // FP32 accumulators for numerical stability
    simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint k_packs = (K + INT8_PER_UINT - 1) / INT8_PER_UINT;

    uint buf_compute = 0;

    load_activation_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_activation_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }

        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 4;
            uint k_pack_idx = k_sub_base / INT8_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi],
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 4],
                               TILE_K);
            }

            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane < 4) {
                    uint b_col = b_col_base + simd_lane * 2;
                    half4 dequant_vals[2];

                    if (b_col + 1 < N && k_pack_idx < k_packs) {
                        uint32_t packed_lo = B[k_pack_idx * N + b_col];
                        uint32_t packed_hi = B[k_pack_idx * N + b_col + 1];
                        half scale = scales[group_idx * N + b_col];
                        half zero_point = zeros ? zeros[group_idx * N + b_col] : half(0.0h);

                        if (asymmetric) {
                            dequant_s8x4_asym_fused(packed_lo, scale, zero_point, dequant_vals[0]);
                            dequant_s8x4_asym_fused(packed_hi, scale, zero_point, dequant_vals[1]);
                        } else {
                            dequant_s8x4_fused(packed_lo, scale, dequant_vals[0]);
                            dequant_s8x4_fused(packed_hi, scale, dequant_vals[1]);
                        }
                    } else {
                        dequant_vals[0] = half4(0.0h);
                        dequant_vals[1] = half4(0.0h);
                    }

                    #pragma unroll
                    for (uint row = 0; row < 4; ++row) {
                        B_staging[simd_id][row][simd_lane * 2] = dequant_vals[0][row];
                        B_staging[simd_id][row][simd_lane * 2 + 1] = dequant_vals[1][row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                // FP32 accumulation (mixed precision MMA)
                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store with FP32 -> FP16 conversion
    threadgroup float epilogue_staging[8][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            simdgroup_store(acc[mi][ni], &epilogue_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    float val = epilogue_staging[r][c];
                    half hval = half(val);

                    if (epilogue_mode != EPILOGUE_NONE) {
                        half bias_val = (bias != nullptr && gc < N) ? bias[gc] : half(0.0h);
                        hval = apply_epilogue(hval, bias_val, epilogue_mode);
                    }

                    C[gr * N + gc] = hval;
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ===========================================================================
// Variant: 64x32 tiles for better utilization on smaller workloads
// ===========================================================================

constant constexpr uint TILE_M_SMALL = 64;
constant constexpr uint TILE_N_SMALL = 128;
constant constexpr uint TILE_K_SMALL = 32;
constant constexpr uint K_TILES_SMALL = TILE_K_SMALL / 4;

kernel void gemm_int8_tiled_64x128(
    device const half* A           [[buffer(0)]],
    device const uint32_t* B       [[buffer(1)]],
    device const half* scales      [[buffer(2)]],
    device const half* zeros       [[buffer(3)]],
    device half* C                 [[buffer(4)]],
    device const half* bias        [[buffer(5)]],
    constant GemmParams& params    [[buffer(6)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;
    const uint group_size = params.group_size;
    const uint epilogue_mode = params.epilogue_mode;
    const uint asymmetric = params.asymmetric;

    // 64x128x32: A = 2*64*32*2 = 8KB, B_staging = 512B
    threadgroup half A_tiles[2][TILE_M_SMALL][TILE_K_SMALL];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M_SMALL;
    const uint tg_col = tgid.x * TILE_N_SMALL;

    if (tg_row >= M) return;

    // 4 simdgroups tile 64x128 as 4x4: each handles 16x32
    const uint sg_row_offset = (simd_id / 4) * 16;
    const uint sg_col_offset = (simd_id % 4) * 32;

    // Each simdgroup handles 2x4 = 8 sub-tiles of 8x8
    constexpr uint SG_M_TILES_S = 2;
    constexpr uint SG_N_TILES_S = 4;

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES_S][SG_N_TILES_S];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES_S; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES_S; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K_SMALL - 1) / TILE_K_SMALL;
    const uint k_packs = (K + INT8_PER_UINT - 1) / INT8_PER_UINT;

    // A tile load: 64 * 32 = 2048 / 128 = 16 per thread
    constexpr uint A_ELEMS = (TILE_M_SMALL * TILE_K_SMALL) / THREADS_PER_TG;

    // Prologue
    #pragma unroll
    for (uint i = 0; i < A_ELEMS; ++i) {
        uint flat_idx = thread_idx * A_ELEMS + i;
        uint row = flat_idx / TILE_K_SMALL;
        uint col = flat_idx % TILE_K_SMALL;
        uint global_row = tg_row + row;
        uint global_col = col;
        half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
        A_tiles[0][row][col] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K_SMALL;
        uint next_k = k_block + TILE_K_SMALL;
        uint buf_load = 1 - buf_compute;

        // Load next A tile
        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < A_ELEMS; ++i) {
                uint flat_idx = thread_idx * A_ELEMS + i;
                uint row = flat_idx / TILE_K_SMALL;
                uint col = flat_idx % TILE_K_SMALL;
                uint global_row = tg_row + row;
                uint global_col = next_k + col;
                half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
                A_tiles[buf_load][row][col] = val;
            }
        }

        // Compute
        #pragma unroll
        for (uint kk = 0; kk < K_TILES_SMALL; ++kk) {
            uint k_sub_base = k_block + kk * 4;
            uint k_pack_idx = k_sub_base / INT8_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES_S];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES_S; ++mi) {
                simdgroup_load(a_frag[mi],
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 4],
                               TILE_K_SMALL);
            }

            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES_S; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane < 4) {
                    uint b_col = b_col_base + simd_lane * 2;
                    half4 dequant_vals[2];

                    if (b_col + 1 < N && k_pack_idx < k_packs) {
                        uint32_t packed_lo = B[k_pack_idx * N + b_col];
                        uint32_t packed_hi = B[k_pack_idx * N + b_col + 1];
                        half scale = scales[group_idx * N + b_col];
                        half zero_point = zeros ? zeros[group_idx * N + b_col] : half(0.0h);

                        if (asymmetric) {
                            dequant_s8x4_asym_fused(packed_lo, scale, zero_point, dequant_vals[0]);
                            dequant_s8x4_asym_fused(packed_hi, scale, zero_point, dequant_vals[1]);
                        } else {
                            dequant_s8x4_fused(packed_lo, scale, dequant_vals[0]);
                            dequant_s8x4_fused(packed_hi, scale, dequant_vals[1]);
                        }
                    } else {
                        dequant_vals[0] = half4(0.0h);
                        dequant_vals[1] = half4(0.0h);
                    }

                    #pragma unroll
                    for (uint row = 0; row < 4; ++row) {
                        B_staging[simd_id][row][simd_lane * 2] = dequant_vals[0][row];
                        B_staging[simd_id][row][simd_lane * 2 + 1] = dequant_vals[1][row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES_S; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store
    threadgroup half epilogue_staging[8][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES_S; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES_S; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (epilogue_mode == EPILOGUE_NONE &&
                out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }

            simdgroup_store(acc[mi][ni], &epilogue_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    half val = epilogue_staging[r][c];
                    if (epilogue_mode != EPILOGUE_NONE) {
                        half bias_val = (bias != nullptr) ? bias[gc] : half(0.0h);
                        val = apply_epilogue(val, bias_val, epilogue_mode);
                    }
                    C[gr * N + gc] = val;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ===========================================================================
// Test kernels for validation
// ===========================================================================

kernel void test_int8_dequant_symmetric(
    device const uint32_t* packed_input [[buffer(0)]],
    device const half* scale          [[buffer(1)]],
    device half* output               [[buffer(2)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 result;
    dequant_s8x4_fused(packed_input[0], scale[0], result);

    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}

kernel void test_int8_dequant_asymmetric(
    device const uint32_t* packed_input [[buffer(0)]],
    device const half* scale          [[buffer(1)]],
    device const half* zero_point     [[buffer(2)]],
    device half* output               [[buffer(3)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid > 0u) return;

    half4 result;
    dequant_s8x4_asym_fused(packed_input[0], scale[0], zero_point[0], result);

    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
    output[3] = result.w;
}