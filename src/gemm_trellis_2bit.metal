// gemm_trellis_2bit.metal - 2-Bit Specialized Fused Dequant+GEMM
// ============================================================================
//
// Specialized kernel for 2-bit trellis quantization (4 values per byte).
// This is a high-performance variant with optimizations specific to 2-bit packing.
//
// 2-bit quantization properties:
//   - n_levels = 4 (codebook indices: 0, 1, 2, 3)
//   - 4 indices packed per byte (bits [0:1], [2:3], [4:5], [6:7])
//   - 64 bytes per 16x16 trellis tile (256 * 2 / 8)
//   - Maximal packing density (same as 8-bit uncompressed but 4x smaller)
//
// Specializations:
//   - All switch statements eliminated (bits=2 is compile-time constant)
//   - No bounds checks on trellis_idx (always in [0,3])
//   - Codebook cache reduced to 4 elements
//   - Direct unpack_2bit_index calls (no runtime dispatch)
//   - Constant folded packed_bytes = 64
//
// Tile dimensions: TILE_M=128, TILE_N=128, TILE_K=32
// 8 simdgroups per threadgroup, 256 threads total
//
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// Tile Dimensions
// ============================================================================

constant constexpr uint TILE_M = 128;
constant constexpr uint TILE_N = 128;
constant constexpr uint TILE_K = 32;
constant constexpr uint K_TILES = TILE_K / 8;

constant constexpr uint SIMDGROUPS_PER_TG = 8;
constant constexpr uint THREADS_PER_TG = 256;
constant constexpr uint SG_M_TILES = 4;
constant constexpr uint SG_N_TILES = 8;

constant constexpr uint TRELLIS_TILE_DIM = 16;
constant constexpr uint TRELLIS_TILE_SIZE = 256;

// 2-bit constants
constant constexpr uint BITS_2 = 2;
constant constexpr uint N_LEVELS_2BIT = 4;
constant constexpr uint PACKED_BYTES_2BIT = (TRELLIS_TILE_SIZE * BITS_2 + 7) / 8;  // 64

// ============================================================================
// 2-Bit Unpacking Primitives
// ============================================================================

/// Unpack a single 2-bit trellis index from packed bytes.
/// Layout: 4 indices per byte (bits [0:1], [2:3], [4:5], [6:7])
/// @param packed      Pointer to packed byte data
/// @param idx_in_tile Index within the 256-element trellis tile [0, 255]
/// @return Index value [0, 3]
inline uint unpack_2bit_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 2;        // Divide by 4
    uint bit_offset = (idx_in_tile & 3) << 1; // (idx % 4) * 2
    return (packed[byte_idx] >> bit_offset) & 0x3;
}

/// Unpack 4 consecutive 2-bit indices from packed bytes using SIMD.
/// Since 4 indices fit in exactly 1 byte, this is a simple byte load + shifts.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices (values 0-3)
inline uint4 unpack_2bit_x4(device const uchar* packed, uint base_idx) {
    uchar byte_val = packed[base_idx >> 2];
    return uint4(
        (byte_val >> 0) & 0x3,
        (byte_val >> 2) & 0x3,
        (byte_val >> 4) & 0x3,
        (byte_val >> 6) & 0x3
    );
}

/// Unpack 8 consecutive 2-bit indices from packed bytes.
/// 8 indices = 2 bytes.
/// @param packed      Pointer to packed byte data
/// @param base_idx    Starting index in tile (must be aligned to 8)
/// @param indices_lo  Output: indices 0-3
/// @param indices_hi  Output: indices 4-7
inline void unpack_2bit_x8(
    device const uchar* packed,
    uint base_idx,
    thread uint4& indices_lo,
    thread uint4& indices_hi
) {
    uint byte_idx = base_idx >> 2;
    uchar2 bytes = uchar2(packed[byte_idx], packed[byte_idx + 1]);
    indices_lo = uint4(
        (bytes.x >> 0) & 0x3,
        (bytes.x >> 2) & 0x3,
        (bytes.x >> 4) & 0x3,
        (bytes.x >> 6) & 0x3
    );
    indices_hi = uint4(
        (bytes.y >> 0) & 0x3,
        (bytes.y >> 2) & 0x3,
        (bytes.y >> 4) & 0x3,
        (bytes.y >> 6) & 0x3
    );
}

/// Unpack 2-bit index from threadgroup memory.
inline uint unpack_2bit_index_tg(threadgroup uchar* packed, uint base_offset, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 2;
    uint bit_offset = (idx_in_tile & 3) << 1;
    return (packed[base_offset + byte_idx] >> bit_offset) & 0x3;
}

// ============================================================================
// Dequantization Primitives (2-bit specialized)
// ============================================================================

/// Dequantize a 2-bit trellis element using precomputed combined scale.
/// No bounds check needed (trellis_idx always in [0,3]).
inline half dequant_2bit_element(uint trellis_idx, float combined_scale, constant float* grid) {
    return half(grid[trellis_idx] * combined_scale);
}

// ============================================================================
// Cooperative A Tile Loader
// ============================================================================

inline void load_A_tile_cooperative(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
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

// ============================================================================
// Main 2-bit Specialized Kernel: gemm_trellis_2bit
// ============================================================================

/// Fused dequant + GEMM specialized for 2-bit trellis quantization.
///
/// Computes C[M,N] = A[M,K] @ dequant(W[K,N]) where W is 2-bit trellis-quantized.
/// 4 values per byte (maximal packing density).
///
/// @param A               Input activations [M, K] half (row-major)
/// @param packed_indices  Packed trellis indices [tiles_k, tiles_n, 64] uint8
/// @param scales          Per-group scales [K/group_size, N] float32
/// @param grid            Codebook grid values [4] float32
/// @param su              Row signs [K] float32
/// @param sv              Column signs [N] float32
/// @param C               Output matrix [M, N] half (row-major)
/// @param M               Number of rows in A and C
/// @param K               Number of columns in A / rows in W
/// @param N               Number of columns in W and C
/// @param group_size      Quantization group size
kernel void gemm_trellis_2bit(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales         [[buffer(2)]],
    constant float* grid           [[buffer(3)]],
    constant float* su             [[buffer(4)]],
    constant float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& group_size          [[buffer(10)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // Threadgroup memory allocation (2-bit optimized)
    // -------------------------------------------------------------------------
    // Double-buffered A tiles: 2 * 128 * 32 * 2 = 16KB
    // Per-simdgroup B staging: 8 * 8 * 8 * 2 = 1024B
    // Codebook cache: 4 * 4 = 16B (tiny - fits entirely in L1)
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup float grid_cache[N_LEVELS_2BIT];

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    if (tg_row >= M) return;

    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;

    // -------------------------------------------------------------------------
    // Initialize accumulators
    // -------------------------------------------------------------------------
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }

    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;

    uint buf_compute = 0;

    // -------------------------------------------------------------------------
    // Prologue: Load codebook and first A tile
    // -------------------------------------------------------------------------
    // Cooperative codebook load: all 4 threads load 1 value each
    if (thread_idx < N_LEVELS_2BIT) {
        grid_cache[thread_idx] = grid[thread_idx];
    }

    load_A_tile_cooperative(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Main pipeline loop
    // -------------------------------------------------------------------------
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Async load next A tile
        if (next_k < K) {
            load_A_tile_cooperative(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }

        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;

            uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
            uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;

            // Load A fragments for this K sub-tile
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(
                    a_frag[mi],
                    &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                    TILE_K
                );
            }

            // ILP: Prefetch sv values
            float sv_cached[8];
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                uint k_idx = k_sub_base + row;
                sv_cached[row] = (k_idx < K) ? sv[k_idx] : 0.0f;
            }

            // For each N sub-tile, dequant B (2-bit) and accumulate
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        uint scale_idx = group_idx * N + b_col;

                        float scale = scales[scale_idx];
                        float row_sign = su[b_col];
                        float scale_row = scale * row_sign;

                        // Precompute combined scales
                        float combined_scale[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            combined_scale[row] = scale_row * sv_cached[row];
                        }

                        // Dequant 8 elements using direct 2-bit unpack
                        // No bounds checks needed (trellis_idx always in [0,3])
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            if (k_idx >= K) {
                                dequant_vals[row] = half(0.0h);
                                continue;
                            }

                            uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                            uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM;

                            uint tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * PACKED_BYTES_2BIT;
                            uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;

                            // Direct 2-bit unpack - no switch, no bounds check
                            uint trellis_idx = unpack_2bit_index(
                                packed_indices + tile_offset, idx_in_tile);

                            // Grid lookup from cache (4 elements max)
                            dequant_vals[row] = half(grid_cache[trellis_idx] * combined_scale[row]);
                        }
                    } else {
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_vals[row] = half(0.0h);
                        }
                    }

                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // -------------------------------------------------------------------------
    // Epilogue: Store results
    // -------------------------------------------------------------------------
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }

            // Slow path for partial tiles
            threadgroup half epilogue_staging[8][8];
            simdgroup_store(acc[mi][ni], &epilogue_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem >> 3;
                uint c = elem & 7;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// 2-bit Decode-Optimized Variant: gemm_trellis_2bit_decode
// ============================================================================

/// Optimized variant for small M (autoregressive decode, M=1-16).
/// Uses 32x128 tiles to maximize N coverage per threadgroup.

constant constexpr uint DECODE_TILE_M = 32;
constant constexpr uint DECODE_TILE_N = 128;
constant constexpr uint DECODE_SG_M_TILES = 4;
constant constexpr uint DECODE_SG_N_TILES = 4;
constant constexpr uint DECODE_K_TILES = TILE_K / 8;

kernel void gemm_trellis_2bit_decode(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales         [[buffer(2)]],
    constant float* grid           [[buffer(3)]],
    constant float* su             [[buffer(4)]],
    constant float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& group_size          [[buffer(10)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[2][DECODE_TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];

    const uint tg_row = tgid.y * DECODE_TILE_M;
    const uint tg_col = tgid.x * DECODE_TILE_N;

    if (tg_row >= M) return;

    const uint sg_row_offset = 0;
    const uint sg_col_offset = simd_group_id * 32;

    simdgroup_matrix<half, 8, 8> acc[DECODE_SG_M_TILES][DECODE_SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0h));
        }
    }

    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;

    constexpr uint A_ELEMS = (DECODE_TILE_M * TILE_K) / THREADS_PER_TG;

    // Load first A tile
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
        for (uint kk = 0; kk < DECODE_K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[DECODE_SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                simdgroup_load(a_frag[mi], &A_tiles[buf_compute][mi * 8][kk * 8], TILE_K);
            }

            #pragma unroll
            for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;

                        uint scale_idx = group_idx * N + b_col;
                        float scale = scales[scale_idx];
                        float sign_n = sv[b_col];
                        float scale_sign_n = scale * sign_n;

                        // Precompute combined scales
                        float combined_scale[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            float su_val = (k_idx < K) ? su[k_idx] : 0.0f;
                            combined_scale[row] = scale_sign_n * su_val;
                        }

                        // Dequant 8 elements with direct 2-bit unpack
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            if (k_idx >= K) {
                                dequant_vals[row] = half(0.0h);
                                continue;
                            }

                            uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                            uint local_k = k_idx % TRELLIS_TILE_DIM;

                            uint tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * PACKED_BYTES_2BIT;
                            uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;

                            uint trellis_idx = unpack_2bit_index(
                                packed_indices + tile_offset, idx_in_tile);

                            dequant_vals[row] = half(grid[trellis_idx] * combined_scale[row]);
                        }
                    } else {
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_vals[row] = half(0.0h);
                        }
                    }

                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);

                #pragma unroll
                for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Store
    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        uint out_row = tg_row + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }

            threadgroup half epilogue_staging[8][8];
            simdgroup_store(acc[mi][ni], &epilogue_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem >> 3;
                uint c = elem & 7;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// 2-bit Fully-Fused Dequant+GEMM: gemm_trellis_2bit_fused_reg
// ============================================================================

/// Fully-fused variant with register-only dequantization.
/// No staging buffer - dequant and accumulate entirely in registers.
///
/// Key 2-bit optimizations:
/// - Direct unpack_2bit_index (no switch)
/// - No bounds checks on trellis_idx
/// - Grid lookup with 4-element constant array

inline void dequant_and_accumulate_2x_2bit(
    device const uchar* packed_indices,
    constant float* scales,
    constant float* grid,
    constant float* su,
    thread const float* sv_cached,
    uint tiles_n,
    uint n_levels,
    uint K,
    uint N,
    uint group_idx,
    uint k_sub_base,
    uint b_col_base,
    uint simd_lane_id,
    half a_vals[2][8],
    thread float acc[8]
) {
    #pragma unroll
    for (uint col = 0; col < 8; ++col) {
        uint b_col = b_col_base + col;
        if (b_col >= N) {
            acc[col] = 0.0f;
            continue;
        }

        float sum0 = 0.0f;
        float sum1 = 0.0f;

        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            uint k_idx = k_sub_base + k;
            if (k_idx >= K) continue;

            uint trellis_tile_k = k_idx / TRELLIS_TILE_DIM;
            uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
            uint local_k = k_idx % TRELLIS_TILE_DIM;
            uint local_n = b_col % TRELLIS_TILE_DIM;
            uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * PACKED_BYTES_2BIT;
            uint idx_in_tile = local_n * TRELLIS_TILE_DIM + local_k;

            // Direct 2-bit unpack - no bounds check needed
            uint trellis_idx = unpack_2bit_index(packed_indices + tile_offset, idx_in_tile);

            float scale = scales[group_idx * N + b_col];
            float row_sign = su[b_col];
            float col_sign = sv_cached[k];
            float b_val = grid[trellis_idx] * scale * row_sign * col_sign;

            sum0 += float(a_vals[0][k]) * b_val;
            sum1 += float(a_vals[1][k]) * b_val;
        }

        acc[col] = sum0 + sum1;
    }
}

kernel void gemm_trellis_2bit_fused_reg(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales         [[buffer(2)]],
    constant float* grid           [[buffer(3)]],
    constant float* su             [[buffer(4)]],
    constant float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& group_size          [[buffer(10)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    if (tg_row >= M) return;

    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;

    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;

    thread float tile_acc[SG_M_TILES][SG_N_TILES][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            #pragma unroll
            for (uint col = 0; col < 8; ++col) {
                tile_acc[mi][ni][col] = 0.0f;
            }
        }
    }

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;

        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;

            float sv_cached[8];
            {
                float my_sv = 0.0f;
                if (simd_lane_id < 8) {
                    uint k_idx = k_sub_base + simd_lane_id;
                    my_sv = (k_idx < K) ? sv[k_idx] : 0.0f;
                }
                #pragma unroll
                for (uint i = 0; i < 8; ++i) {
                    sv_cached[i] = simd_shuffle(my_sv, i);
                }
            }

            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                uint a_row_base = sg_row_offset + mi * 8;

                half a_frag[2][8];
                #pragma unroll
                for (uint k = 0; k < 8; ++k) {
                    uint k_idx = k_sub_base + k;
                    #pragma unroll
                    for (uint row = 0; row < 2; ++row) {
                        uint global_row = tg_row + a_row_base + row;
                        if (global_row < M && k_idx < K) {
                            a_frag[row][k] = A[global_row * K + k_idx];
                        } else {
                            a_frag[row][k] = half(0.0h);
                        }
                    }
                }

                #pragma unroll
                for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                    uint b_col_base = tg_col + sg_col_offset + ni * 8;

                    thread float col_acc[8];
                    dequant_and_accumulate_2x_2bit(
                        packed_indices, scales, grid, su, sv_cached,
                        tiles_n, N_LEVELS_2BIT, K, N,
                        group_idx, k_sub_base, b_col_base, simd_lane_id,
                        a_frag, col_acc
                    );

                    #pragma unroll
                    for (uint col = 0; col < 8; ++col) {
                        tile_acc[mi][ni][col] += col_acc[col];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row_base = tg_row + sg_row_offset + mi * 8;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col_base = tg_col + sg_col_offset + ni * 8;

            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem >> 3;
                uint c = elem & 7;
                uint global_row = out_row_base + r;
                uint global_col = out_col_base + c;

                if (global_row < M && global_col < N) {
                    C[global_row * N + global_col] = half(tile_acc[mi][ni][c]);
                }
            }
        }
    }
}
