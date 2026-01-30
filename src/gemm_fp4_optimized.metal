// gemm_fp4_optimized.metal - High-performance fused dequant+GEMM for FP4 weights
//
// Optimized for Apple M4 Max GPU (40 cores, 32KB threadgroup memory, 546 GB/s bandwidth)
//
// Key optimizations:
//   1. LUT-based FP4 dequantization (16-entry constant LUT, ~3 cycles vs ~8 for bitwise)
//   2. Double-buffered A tiles, on-the-fly B dequant (no B_tile materialization)
//   3. Simdgroup-local B staging (512 bytes vs 8KB for full B_tile)
//   4. 64x64x32 tiles with 4 simdgroups (128 threads) - optimal occupancy
//   5. Fused bias and activation epilogue (avoids extra kernel launch)
//   6. Coalesced memory access patterns for both A and packed B
//
// Target: >70% of theoretical FP16 FLOPS on M4 Max (~19.5 TFLOPS of 27.8 TFLOPS peak)
//
// C[M,N] = activation(A[M,K] @ dequant(B[K/8,N], scales[K/group_size,N]) + bias)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ===========================================================================
// Tile dimensions - optimized for M4 Max
// ===========================================================================

constant constexpr uint TILE_M = 128;      // Output rows per threadgroup
constant constexpr uint TILE_N = 128;      // Output cols per threadgroup
constant constexpr uint TILE_K = 32;      // K-reduction per mainloop iteration
constant constexpr uint K_TILES = TILE_K / 8;  // 4 simdgroup MMA ops per K-block

// Simdgroup configuration: 4 simdgroups tile the 64x64 output
// Each simdgroup handles 32x32 output (4x4 blocks of 8x8 tiles)
constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 128
constant constexpr uint SG_M_TILES = 4;   // 4 rows of 8x8 = 32 rows per simdgroup
constant constexpr uint SG_N_TILES = 4;   // 4 cols of 8x8 = 32 cols per simdgroup

// FP4 packing
constant constexpr uint FP4_PER_UINT = 8;

// ===========================================================================
// FP4 E2M1 Dequantization LUT
//
// Format: [sign(1) | exponent(2) | mantissa(1)]
//
// Values for positive numbers (bit3=0):
//   0000 -> 0.0    (exp=0, man=0 -> subnormal 0*0.25)
//   0001 -> 0.25   (exp=0, man=1 -> subnormal 1*0.25)
//   0010 -> 1.0    (exp=1, man=0 -> 2^0 * 1.0)
//   0011 -> 1.5    (exp=1, man=1 -> 2^0 * 1.5)
//   0100 -> 2.0    (exp=2, man=0 -> 2^1 * 1.0)
//   0101 -> 3.0    (exp=2, man=1 -> 2^1 * 1.5)
//   0110 -> 4.0    (exp=3, man=0 -> 2^2 * 1.0)
//   0111 -> 6.0    (exp=3, man=1 -> 2^2 * 1.5)
//
// Negative numbers (bit3=1) are negations of above.
// ===========================================================================

constant half FP4_LUT[16] = {
    half( 0.00h),  // 0000
    half( 0.25h),  // 0001
    half( 1.00h),  // 0010
    half( 1.50h),  // 0011
    half( 2.00h),  // 0100
    half( 3.00h),  // 0101
    half( 4.00h),  // 0110
    half( 6.00h),  // 0111
    half(-0.00h),  // 1000 (negative zero, same as zero)
    half(-0.25h),  // 1001
    half(-1.00h),  // 1010
    half(-1.50h),  // 1011
    half(-2.00h),  // 1100
    half(-3.00h),  // 1101
    half(-4.00h),  // 1110
    half(-6.00h),  // 1111
};

// Inline LUT lookup with scale multiplication
inline half dequant_fp4_lut(uint nibble, half scale) {
    return FP4_LUT[nibble & 0xF] * scale;
}

// Unpack and dequant 8 FP4 values from packed uint32
inline void unpack_dequant_fp4x8(uint32_t packed, half scale, thread half* out) {
    out[0] = FP4_LUT[(packed >>  0) & 0xF] * scale;
    out[1] = FP4_LUT[(packed >>  4) & 0xF] * scale;
    out[2] = FP4_LUT[(packed >>  8) & 0xF] * scale;
    out[3] = FP4_LUT[(packed >> 12) & 0xF] * scale;
    out[4] = FP4_LUT[(packed >> 16) & 0xF] * scale;
    out[5] = FP4_LUT[(packed >> 20) & 0xF] * scale;
    out[6] = FP4_LUT[(packed >> 24) & 0xF] * scale;
    out[7] = FP4_LUT[(packed >> 28) & 0xF] * scale;
}

// Unpack and dequant 4 FP4 values (half a uint32, for smaller K steps)
inline void unpack_dequant_fp4x4(uint32_t packed, uint offset, half scale, thread half* out) {
    uint shift = offset * 16;  // 0 or 16
    out[0] = FP4_LUT[(packed >> (shift +  0)) & 0xF] * scale;
    out[1] = FP4_LUT[(packed >> (shift +  4)) & 0xF] * scale;
    out[2] = FP4_LUT[(packed >> (shift +  8)) & 0xF] * scale;
    out[3] = FP4_LUT[(packed >> (shift + 12)) & 0xF] * scale;
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
    // Sigmoid approximation: x * sigmoid(1.702 * x)
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
// Cooperative A tile loader (all 128 threads)
// ===========================================================================

inline void load_A_tile(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    // 64 * 32 = 2048 elements / 128 threads = 16 elements per thread
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
// Main Kernel: Fused FP4 GEMM with optional epilogue
//
// Architecture:
//   - Double-buffered A tiles in threadgroup memory (8KB per buffer)
//   - On-the-fly B dequant with per-simdgroup staging (128 bytes per SG)
//   - Simdgroup MMA with half precision accumulation
//   - Fused bias + activation epilogue
//
// Dispatch: Grid ceil(N/64) x ceil(M/64), threadgroup 128 threads
// ===========================================================================

struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint group_size;
    uint epilogue_mode;
};

kernel void gemm_fp4_optimized(
    device const half* A           [[buffer(0)]],  // [M, K] activations (row-major)
    device const uint32_t* B       [[buffer(1)]],  // [K/8, N] packed FP4 weights
    device const half* scales      [[buffer(2)]],  // [K/group_size, N] per-group scales
    device half* C                 [[buffer(3)]],  // [M, N] output (row-major)
    device const half* bias        [[buffer(4)]],  // [N] bias (optional, can be nullptr)
    constant GemmParams& params    [[buffer(5)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;
    const uint group_size = params.group_size;
    const uint epilogue_mode = params.epilogue_mode;

    // --- Threadgroup memory allocation ---
    // Double-buffered A: 2 * 64 * 32 * 2 = 8KB
    // Per-simdgroup B staging: 4 * 8 * 8 * 2 = 512 bytes
    // Total: 8.5KB (26% of 32KB budget - excellent occupancy)
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    // --- Tile assignment ---
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    // Early exit for out-of-bounds threadgroups
    if (tg_row >= M) return;

    // Simdgroup layout: 2x2 grid, each covering 32x32 of the 64x64 output
    // SG 0: rows [0,31],  cols [0,31]
    // SG 1: rows [0,31],  cols [32,63]
    // SG 2: rows [32,63], cols [0,31]
    // SG 3: rows [32,63], cols [32,63]
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
    const uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;

    uint buf_compute = 0;

    // =========================================================================
    // Prologue: Load first A tile
    // =========================================================================
    load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Main pipeline loop
    // =========================================================================
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        // --- Async load next A tile while computing on current ---
        if (next_k < K) {
            load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }

        // --- Compute: Fused B dequant + MMA ---
        // For each K sub-tile (8 elements), dequant B on-the-fly and multiply
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            // Load A fragments for this K sub-tile (reused across N tiles)
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi],
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                               TILE_K);
            }

            // For each N sub-tile, dequant B and accumulate
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                // --- Fused B dequant: lanes 0-7 each load+dequant one column ---
                if (simd_lane < 8) {
                    uint b_col = b_col_base + simd_lane;
                    half dequant_vals[8];

                    if (b_col < N && k_pack_idx < k_packs) {
                        // Coalesced load: adjacent lanes read adjacent columns
                        uint32_t packed = B[k_pack_idx * N + b_col];
                        half scale = scales[group_idx * N + b_col];

                        // LUT-based dequant (unrolled for speed)
                        unpack_dequant_fp4x8(packed, scale, dequant_vals);
                    } else {
                        #pragma unroll
                        for (uint v = 0; v < 8; ++v) {
                            dequant_vals[v] = half(0.0h);
                        }
                    }

                    // Write to staging: B_staging[sg][k][n]
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }
                }

                // Lightweight simdgroup sync (not full threadgroup barrier)
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Load B fragment and accumulate
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    // simdgroup_multiply_accumulate is the core SIMD intrinsic that
                    // maps 8x8 tiles to Apple Silicon's matrix pipeline.
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

kernel void gemm_fp4_optimized_fp32acc(
    device const half* A           [[buffer(0)]],
    device const uint32_t* B       [[buffer(1)]],
    device const half* scales      [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    device const half* bias        [[buffer(4)]],
    constant GemmParams& params    [[buffer(5)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;
    const uint group_size = params.group_size;
    const uint epilogue_mode = params.epilogue_mode;

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
    const uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;

    uint buf_compute = 0;

    load_A_tile(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_A_tile(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }

        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi],
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                               TILE_K);
            }

            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane < 8) {
                    uint b_col = b_col_base + simd_lane;
                    half dequant_vals[8];

                    if (b_col < N && k_pack_idx < k_packs) {
                        uint32_t packed = B[k_pack_idx * N + b_col];
                        half scale = scales[group_idx * N + b_col];
                        unpack_dequant_fp4x8(packed, scale, dequant_vals);
                    } else {
                        #pragma unroll
                        for (uint v = 0; v < 8; ++v) {
                            dequant_vals[v] = half(0.0h);
                        }
                    }

                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_id][row][simd_lane] = dequant_vals[row];
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
// Variant: 128x64 tiles for large-M workloads (batch inference)
// ===========================================================================

constant constexpr uint TILE_M_LARGE = 128;
constant constexpr uint TILE_K_LARGE = 16;   // Shallower K for memory balance
constant constexpr uint K_TILES_LARGE = TILE_K_LARGE / 8;  // 2

kernel void gemm_fp4_optimized_large_m(
    device const half* A           [[buffer(0)]],
    device const uint32_t* B       [[buffer(1)]],
    device const half* scales      [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    device const half* bias        [[buffer(4)]],
    constant GemmParams& params    [[buffer(5)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;
    const uint group_size = params.group_size;
    const uint epilogue_mode = params.epilogue_mode;

    // 128x64x16 tiles: A = 2*128*16*2 = 8KB, B_staging = 512B
    threadgroup half A_tiles[2][TILE_M_LARGE][TILE_K_LARGE];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M_LARGE;
    const uint tg_col = tgid.x * TILE_N;

    if (tg_row >= M) return;

    // 4 simdgroups tile 128x64 as 4x1: each handles 32x64
    const uint sg_row_offset = simd_id * 32;
    const uint sg_col_offset = 0;

    // Each simdgroup handles 4x8 = 32 sub-tiles of 8x8
    constexpr uint SG_M_TILES_L = 4;
    constexpr uint SG_N_TILES_L = 8;

    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES_L][SG_N_TILES_L];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES_L; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES_L; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }

    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = (K + TILE_K_LARGE - 1) / TILE_K_LARGE;
    const uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;

    // A tile load: 128 * 16 = 2048 elements / 128 threads = 16 per thread
    constexpr uint A_ELEMS_PER_THREAD = (TILE_M_LARGE * TILE_K_LARGE) / THREADS_PER_TG;

    // Prologue
    #pragma unroll
    for (uint i = 0; i < A_ELEMS_PER_THREAD; ++i) {
        uint flat_idx = thread_idx * A_ELEMS_PER_THREAD + i;
        uint row = flat_idx / TILE_K_LARGE;
        uint col = flat_idx % TILE_K_LARGE;
        uint global_row = tg_row + row;
        uint global_col = col;
        half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
        A_tiles[0][row][col] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K_LARGE;
        uint next_k = k_block + TILE_K_LARGE;
        uint buf_load = 1 - buf_compute;

        // Load next A tile
        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < A_ELEMS_PER_THREAD; ++i) {
                uint flat_idx = thread_idx * A_ELEMS_PER_THREAD + i;
                uint row = flat_idx / TILE_K_LARGE;
                uint col = flat_idx % TILE_K_LARGE;
                uint global_row = tg_row + row;
                uint global_col = next_k + col;
                half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
                A_tiles[buf_load][row][col] = val;
            }
        }

        // Compute
        #pragma unroll
        for (uint kk = 0; kk < K_TILES_LARGE; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES_L];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES_L; ++mi) {
                simdgroup_load(a_frag[mi],
                               &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                               TILE_K_LARGE);
            }

            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES_L; ++ni) {
                uint b_col_base = tg_col + ni * 8;

                if (simd_lane < 8) {
                    uint b_col = b_col_base + simd_lane;
                    half dequant_vals[8];

                    if (b_col < N && k_pack_idx < k_packs) {
                        uint32_t packed = B[k_pack_idx * N + b_col];
                        half scale = scales[group_idx * N + b_col];
                        unpack_dequant_fp4x8(packed, scale, dequant_vals);
                    } else {
                        #pragma unroll
                        for (uint v = 0; v < 8; ++v) dequant_vals[v] = half(0.0h);
                    }

                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES_L; ++mi) {
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
    for (uint mi = 0; mi < SG_M_TILES_L; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES_L; ++ni) {
            uint out_col = tg_col + ni * 8;

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
// Variant: Decode-optimized (M=1-16, wide N tiles)
//
// For autoregressive decoding, M is small (typically 1 for single-token decode,
// up to 16 for speculative decoding). This variant uses 32x128 tiles to maximize
// N coverage per threadgroup, reducing the number of output tiles.
// ===========================================================================

constant constexpr uint TILE_M_DECODE = 32;
constant constexpr uint TILE_N_DECODE = 128;
constant constexpr uint K_TILES_DECODE = TILE_K / 8;

kernel void gemm_fp4_optimized_decode(
    device const half* A           [[buffer(0)]],
    device const uint32_t* B       [[buffer(1)]],
    device const half* scales      [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    device const half* bias        [[buffer(4)]],
    constant GemmParams& params    [[buffer(5)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;
    const uint group_size = params.group_size;
    const uint epilogue_mode = params.epilogue_mode;

    // 32x128x32: A = 2*32*32*2 = 4KB, B_staging = 512B
    threadgroup half A_tiles[2][TILE_M_DECODE][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * TILE_M_DECODE;
    const uint tg_col = tgid.x * TILE_N_DECODE;

    if (tg_row >= M) return;

    // 4 simdgroups tile 32x128 as 1x4: each handles 32x32
    const uint sg_row_offset = 0;
    const uint sg_col_offset = simd_id * 32;

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
    const uint k_packs = (K + FP4_PER_UINT - 1) / FP4_PER_UINT;

    // A tile load: 32 * 32 = 1024 / 128 = 8 per thread
    constexpr uint A_ELEMS = (TILE_M_DECODE * TILE_K) / THREADS_PER_TG;

    // Prologue
    #pragma unroll
    for (uint i = 0; i < A_ELEMS; ++i) {
        uint flat_idx = thread_idx * A_ELEMS + i;
        uint row = flat_idx / TILE_K;
        uint col = flat_idx % TILE_K;
        uint global_row = tg_row + row;
        half val = (global_row < M && col < K) ? A[global_row * K + col] : half(0.0h);
        A_tiles[0][row][col] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_compute = 0;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < A_ELEMS; ++i) {
                uint flat_idx = thread_idx * A_ELEMS + i;
                uint row = flat_idx / TILE_K;
                uint col = flat_idx % TILE_K;
                uint global_row = tg_row + row;
                uint global_col = next_k + col;
                half val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : half(0.0h);
                A_tiles[buf_load][row][col] = val;
            }
        }

        #pragma unroll
        for (uint kk = 0; kk < K_TILES_DECODE; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint k_pack_idx = k_sub_base / FP4_PER_UINT;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi], &A_tiles[buf_compute][mi * 8][kk * 8], TILE_K);
            }

            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane < 8) {
                    uint b_col = b_col_base + simd_lane;
                    half dequant_vals[8];

                    if (b_col < N && k_pack_idx < k_packs) {
                        uint32_t packed = B[k_pack_idx * N + b_col];
                        half scale = scales[group_idx * N + b_col];
                        unpack_dequant_fp4x8(packed, scale, dequant_vals);
                    } else {
                        #pragma unroll
                        for (uint v = 0; v < 8; ++v) dequant_vals[v] = half(0.0h);
                    }

                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_id][row][simd_lane] = dequant_vals[row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
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
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
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
