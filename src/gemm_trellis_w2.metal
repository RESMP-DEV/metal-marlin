// gemm_trellis_w2.metal - Specialized 2-bit Trellis GEMM Kernel
// ============================================================================
//
// High-performance fused kernel specialized for 2-bit trellis quantization.
// This kernel eliminates runtime bit-width switching overhead by hardcoding
// all constants for 2-bit weights (4 values per byte, 4 codebook levels).
//
// 2-bit Trellis Quantization:
//   - Each weight is encoded as a 2-bit index into a 4-element codebook
//   - Packing: 4 indices per byte (bits [0:1], [2:3], [4:5], [6:7])
//   - 16x16 trellis tiles = 256 elements = 64 bytes packed
//   - Typical codebook: {-1.5, -0.5, 0.5, 1.5} or normalized Gaussian quantiles
//   - Dequant: w[k,n] = grid[idx] * scale * su[k] * sv[n]
//
// Performance Optimizations over generic kernel:
//   1. Compile-time constants for packed_bytes (64), n_levels (4)
//   2. Unrolled 2-bit unpacking without switch/case overhead
//   3. 4-element codebook fits entirely in 16 bytes (fits in registers)
//   4. Smaller prefetch buffers (64B per tile vs 128B for 4-bit)
//   5. Simplified index extraction (byte >> shift & 0x3)
//
// Expected speedup: ~10-15% over generic kernel due to reduced branch
// divergence and smaller working set.
//
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// Compile-time Constants for 2-bit Quantization
// ============================================================================

// 2-bit specific constants
constant constexpr uint W2_BITS = 2;
constant constexpr uint W2_LEVELS = 4;                    // 2^2 = 4 codebook levels
constant constexpr uint W2_VALUES_PER_BYTE = 4;           // 4 indices per byte
constant constexpr uint W2_TRELLIS_TILE_DIM = 16;
constant constexpr uint W2_TRELLIS_TILE_SIZE = 256;       // 16x16 elements
constant constexpr uint W2_PACKED_BYTES_PER_TILE = 64;    // 256 * 2 / 8 = 64 bytes

// GEMM tile dimensions (same as generic kernel for compatibility)
constant constexpr uint TILE_M = 128;
constant constexpr uint TILE_N = 128;
constant constexpr uint TILE_K = 32;
constant constexpr uint K_TILES = TILE_K / 8;  // 4 simdgroup MMA ops per K-block

// Simdgroup configuration
constant constexpr uint SIMDGROUPS_PER_TG = 8;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 256 threads
constant constexpr uint SG_M_TILES = 4;   // 4 rows of 8x8 = 32 rows per simdgroup
constant constexpr uint SG_N_TILES = 8;   // 8 cols of 8x8 = 64 cols per simdgroup

// 2-bit prefetch buffer sizing (smaller than generic)
constant constexpr uint W2_PREFETCH_TILES_K = (TILE_K + W2_TRELLIS_TILE_DIM - 1) / W2_TRELLIS_TILE_DIM;  // 2
constant constexpr uint W2_PREFETCH_N_TILES = (SG_N_TILES * 8 + W2_TRELLIS_TILE_DIM - 1) / W2_TRELLIS_TILE_DIM;  // 4
constant constexpr uint W2_PREFETCH_BUF_SIZE_PER_SG = W2_PREFETCH_TILES_K * W2_PREFETCH_N_TILES * W2_PACKED_BYTES_PER_TILE;  // 2*4*64=512 bytes

// ============================================================================
// 2-bit Index Unpacking Primitives
// ============================================================================
//
// Specialized unpacking for 2-bit indices. No switch/case, no bounds checking
// for bit width - everything is hardcoded for maximum performance.
//
// Byte layout: [idx0:2][idx1:2][idx2:2][idx3:2] = 4 indices per byte
// Index extraction: (byte >> (position * 2)) & 0x3
//
// ============================================================================

/// Unpack a single 2-bit index from packed bytes.
/// @param packed     Pointer to packed byte data
/// @param idx_in_tile Index within 256-element tile [0, 255]
/// @return Unpacked index value [0, 3]
inline uint unpack_w2_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 2;        // Divide by 4
    uint bit_offset = (idx_in_tile & 3) << 1; // (idx % 4) * 2
    return (packed[byte_idx] >> bit_offset) & 0x3;
}

/// Unpack a single 2-bit index from threadgroup memory.
/// @param packed     Threadgroup pointer to packed byte data
/// @param base_offset Byte offset into buffer for this tile
/// @param idx_in_tile Index within 256-element tile [0, 255]
/// @return Unpacked index value [0, 3]
inline uint unpack_w2_index_tg(threadgroup uchar* packed, uint base_offset, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 2;
    uint bit_offset = (idx_in_tile & 3) << 1;
    return (packed[base_offset + byte_idx] >> bit_offset) & 0x3;
}

/// Unpack 4 consecutive 2-bit indices from a single byte.
/// Vectorized extraction for SIMD-friendly processing.
/// @param byte_val Single byte containing 4 packed indices
/// @return uint4 containing 4 unpacked indices [0, 3]
inline uint4 unpack_w2_x4_from_byte(uchar byte_val) {
    return uint4(
        (byte_val >> 0) & 0x3,
        (byte_val >> 2) & 0x3,
        (byte_val >> 4) & 0x3,
        (byte_val >> 6) & 0x3
    );
}

/// Unpack 4 consecutive 2-bit indices from device memory.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 4)
/// @return uint4 containing 4 unpacked indices
inline uint4 unpack_w2_x4(device const uchar* packed, uint base_idx) {
    return unpack_w2_x4_from_byte(packed[base_idx >> 2]);
}

/// Unpack 8 consecutive 2-bit indices (spans 2 bytes).
/// @param packed     Pointer to packed byte data
/// @param base_idx   Starting index in tile (must be aligned to 8)
/// @param indices_lo Output: indices 0-3
/// @param indices_hi Output: indices 4-7
inline void unpack_w2_x8(
    device const uchar* packed,
    uint base_idx,
    thread uint4& indices_lo,
    thread uint4& indices_hi
) {
    uint byte_idx = base_idx >> 2;
    indices_lo = unpack_w2_x4_from_byte(packed[byte_idx]);
    indices_hi = unpack_w2_x4_from_byte(packed[byte_idx + 1]);
}

/// Unpack 16 consecutive 2-bit indices (spans 4 bytes).
/// Optimized for processing full rows of trellis tiles.
/// @param packed   Pointer to packed byte data
/// @param base_idx Starting index in tile (must be aligned to 16)
/// @param out      Output array of 16 indices
inline void unpack_w2_x16(device const uchar* packed, uint base_idx, thread uint* out) {
    uint byte_idx = base_idx >> 2;
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        uchar byte_val = packed[byte_idx + i];
        out[i * 4 + 0] = (byte_val >> 0) & 0x3;
        out[i * 4 + 1] = (byte_val >> 2) & 0x3;
        out[i * 4 + 2] = (byte_val >> 4) & 0x3;
        out[i * 4 + 3] = (byte_val >> 6) & 0x3;
    }
}

// ============================================================================
// A-tile Loading (identical to generic kernel)
// ============================================================================

/// Cooperative A-tile load using all threads in threadgroup.
/// Loads TILE_M x TILE_K half elements with bounds checking.
inline void load_A_tile_w2(
    device const half* A,
    threadgroup half A_tile[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    // Each thread loads multiple elements for better memory coalescing
    // Total elements: TILE_M * TILE_K = 128 * 32 = 4096
    // Threads: 256, so 16 elements per thread
    constexpr uint ELEMS_PER_THREAD = (TILE_M * TILE_K) / THREADS_PER_TG;  // 16

    #pragma unroll
    for (uint i = 0; i < ELEMS_PER_THREAD; ++i) {
        uint flat_idx = thread_idx * ELEMS_PER_THREAD + i;
        uint local_row = flat_idx / TILE_K;
        uint local_col = flat_idx % TILE_K;

        uint global_row = tg_row + local_row;
        uint global_col = k_block + local_col;

        half val = half(0.0h);
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_tile[local_row][local_col] = val;
    }
}

// ============================================================================
// 2-bit Weight Prefetching
// ============================================================================
//
// Prefetch 2-bit packed indices into threadgroup memory. With 64 bytes per
// tile (vs 128 for 4-bit), prefetch buffers are half the size, improving
// threadgroup memory utilization.
//
// ============================================================================

/// Cooperative prefetch of 2-bit packed indices for next K-block.
inline void prefetch_w2_weights(
    device const uchar* packed_indices,
    threadgroup uchar* prefetch_buf,
    uint k_block,
    uint tg_col,
    uint sg_col_offset,
    uint tiles_n,
    uint simd_lane_id,
    uint K
) {
    if (k_block >= K) return;

    uint trellis_tile_k_base = k_block / W2_TRELLIS_TILE_DIM;
    uint n_col_base = tg_col + sg_col_offset;
    uint trellis_tile_n_base = n_col_base / W2_TRELLIS_TILE_DIM;

    // Total bytes to prefetch: 2 K-tiles * 4 N-tiles * 64 bytes = 512 bytes
    // 32 lanes, so 16 bytes per lane
    constexpr uint TOTAL_BYTES = W2_PREFETCH_BUF_SIZE_PER_SG;
    constexpr uint BYTES_PER_LANE = (TOTAL_BYTES + 31) / 32;

    #pragma unroll
    for (uint i = 0; i < BYTES_PER_LANE; ++i) {
        uint buf_idx = simd_lane_id * BYTES_PER_LANE + i;
        if (buf_idx >= TOTAL_BYTES) break;

        // Decode which tile and byte this maps to
        uint tile_idx = buf_idx / W2_PACKED_BYTES_PER_TILE;
        uint byte_in_tile = buf_idx % W2_PACKED_BYTES_PER_TILE;

        uint tile_k_offset = tile_idx / W2_PREFETCH_N_TILES;
        uint tile_n_offset = tile_idx % W2_PREFETCH_N_TILES;

        uint actual_tile_k = trellis_tile_k_base + tile_k_offset;
        uint actual_tile_n = trellis_tile_n_base + tile_n_offset;

        // Bounds check and load
        if (actual_tile_k * W2_TRELLIS_TILE_DIM < K && actual_tile_n < tiles_n) {
            uint tile_offset = (actual_tile_k * tiles_n + actual_tile_n) * W2_PACKED_BYTES_PER_TILE;
            prefetch_buf[buf_idx] = packed_indices[tile_offset + byte_in_tile];
        } else {
            prefetch_buf[buf_idx] = 0;
        }
    }
}

// ============================================================================
// 2-bit Dequantization with Inlined Codebook
// ============================================================================
//
// For 2-bit quantization, the codebook has only 4 entries. These fit entirely
// in registers, eliminating threadgroup memory lookups for grid values.
//
// ============================================================================

/// Dequantize a 2-bit index using register-cached codebook.
/// @param idx             2-bit index [0, 3]
/// @param combined_scale  Precomputed scale * su * sv
/// @param grid            4-element codebook in registers
/// @return Dequantized weight value
inline half dequant_w2_element(uint idx, float combined_scale, thread float4& grid) {
    float grid_val;
    switch (idx) {
        case 0: grid_val = grid.x; break;
        case 1: grid_val = grid.y; break;
        case 2: grid_val = grid.z; break;
        default: grid_val = grid.w; break;
    }
    return half(grid_val * combined_scale);
}

/// Dequantize 4 consecutive 2-bit values using vectorized operations.
/// @param indices         uint4 containing 4 indices [0, 3]
/// @param combined_scales float4 containing 4 combined scale values
/// @param grid            4-element codebook
/// @return half4 containing 4 dequantized values
inline half4 dequant_w2_x4(uint4 indices, float4 combined_scales, thread float4& grid) {
    float4 grid_vals;
    grid_vals.x = (indices.x == 0) ? grid.x : (indices.x == 1) ? grid.y : (indices.x == 2) ? grid.z : grid.w;
    grid_vals.y = (indices.y == 0) ? grid.x : (indices.y == 1) ? grid.y : (indices.y == 2) ? grid.z : grid.w;
    grid_vals.z = (indices.z == 0) ? grid.x : (indices.z == 1) ? grid.y : (indices.z == 2) ? grid.z : grid.w;
    grid_vals.w = (indices.w == 0) ? grid.x : (indices.w == 1) ? grid.y : (indices.w == 2) ? grid.z : grid.w;
    return half4(grid_vals * combined_scales);
}

// ============================================================================
// Main 2-bit Specialized GEMM Kernel
// ============================================================================
//
// Computes C = A * dequant(W_packed) where W is 2-bit trellis quantized.
//
// This kernel is functionally identical to gemm_trellis_packed but with
// all 2-bit paths inlined and optimized. The dynamic bit-width parameter
// is removed entirely.
//
// ============================================================================

kernel void gemm_trellis_w2(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales             [[buffer(2)]],
    constant float* grid               [[buffer(3)]],
    constant float* su                 [[buffer(4)]],
    constant float* sv                 [[buffer(5)]],
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
    // Threadgroup memory allocation (smaller than generic due to 64B tiles)
    // -------------------------------------------------------------------------
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup uchar weight_prefetch[2][SIMDGROUPS_PER_TG][W2_PREFETCH_BUF_SIZE_PER_SG];

    // -------------------------------------------------------------------------
    // Load codebook into registers (only 4 values!)
    // -------------------------------------------------------------------------
    // This is the key optimization: 4 floats fit in a single float4 register
    float4 grid_reg = float4(grid[0], grid[1], grid[2], grid[3]);

    // -------------------------------------------------------------------------
    // Tile assignment
    // -------------------------------------------------------------------------
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
    const uint tiles_n = (N + W2_TRELLIS_TILE_DIM - 1) / W2_TRELLIS_TILE_DIM;

    uint buf_compute = 0;

    // -------------------------------------------------------------------------
    // Prologue: Load first A tile and prefetch first weight tile
    // -------------------------------------------------------------------------
    load_A_tile_w2(A, A_tiles[0], M, K, tg_row, 0, thread_idx);

    prefetch_w2_weights(
        packed_indices,
        weight_prefetch[0][simd_group_id],
        0,
        tg_col,
        sg_col_offset,
        tiles_n,
        simd_lane_id,
        K
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Main pipeline loop
    // -------------------------------------------------------------------------
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Async load next A tile and prefetch next weights
        if (next_k < K) {
            load_A_tile_w2(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);

            prefetch_w2_weights(
                packed_indices,
                weight_prefetch[buf_load][simd_group_id],
                next_k,
                tg_col,
                sg_col_offset,
                tiles_n,
                simd_lane_id,
                K
            );
        }

        // Compute: Fused B dequant + MMA
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;

            // Load A fragments
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(
                    a_frag[mi],
                    &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                    TILE_K
                );
            }

            // Cache sv values for this K sub-tile
            float sv_cached[8];
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                uint k_idx = k_sub_base + row;
                sv_cached[row] = (k_idx < K) ? sv[k_idx] : 0.0f;
            }

            // Process each N sub-tile
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                // Lanes 0-7 handle dequantization
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        uint trellis_tile_n = b_col / W2_TRELLIS_TILE_DIM;
                        uint local_n = b_col % W2_TRELLIS_TILE_DIM;
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

                        // Optimized 2-bit unpacking and dequantization
                        // Process 2 rows at a time for ILP
                        #pragma unroll(4)
                        for (uint row = 0; row < 8; row += 2) {
                            uint k_idx_0 = k_sub_base + row;
                            uint k_idx_1 = k_sub_base + row + 1;

                            uint actual_tile_k_0 = k_idx_0 / W2_TRELLIS_TILE_DIM;
                            uint actual_tile_k_1 = k_idx_1 / W2_TRELLIS_TILE_DIM;
                            uint local_k_0 = k_idx_0 % W2_TRELLIS_TILE_DIM;
                            uint local_k_1 = k_idx_1 % W2_TRELLIS_TILE_DIM;

                            uint tile_offset_0 = (actual_tile_k_0 * tiles_n + trellis_tile_n) * W2_PACKED_BYTES_PER_TILE;
                            uint tile_offset_1 = (actual_tile_k_1 * tiles_n + trellis_tile_n) * W2_PACKED_BYTES_PER_TILE;

                            // Column-major index within tile: local_n * 16 + local_k
                            uint idx_in_tile_0 = local_n * W2_TRELLIS_TILE_DIM + local_k_0;
                            uint idx_in_tile_1 = local_n * W2_TRELLIS_TILE_DIM + local_k_1;

                            // Direct 2-bit unpack (no switch/case)
                            uint trellis_idx_0 = (k_idx_0 < K) ?
                                unpack_w2_index(packed_indices + tile_offset_0, idx_in_tile_0) : 0;
                            uint trellis_idx_1 = (k_idx_1 < K) ?
                                unpack_w2_index(packed_indices + tile_offset_1, idx_in_tile_1) : 0;

                            // Register-based codebook lookup
                            dequant_vals[row] = (k_idx_0 < K) ?
                                dequant_w2_element(trellis_idx_0, combined_scale[row], grid_reg) : half(0.0h);
                            dequant_vals[row + 1] = (k_idx_1 < K) ?
                                dequant_w2_element(trellis_idx_1, combined_scale[row + 1], grid_reg) : half(0.0h);
                        }
                    } else {
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_vals[row] = half(0.0h);
                        }
                    }

                    // Write to staging
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Load B fragment and MMA
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
    threadgroup half epilogue_staging[SIMDGROUPS_PER_TG][8][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            // Fast path: full tile within bounds
            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }

            // Slow path: partial tile
            simdgroup_store(acc[mi][ni], &epilogue_staging[simd_group_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[simd_group_id][r][c];
                }
            }
        }
    }
}

// ============================================================================
// FP32 Accumulation Variant for Large K
// ============================================================================
//
// When K is very large (>4096), FP16 accumulation can lose precision.
// This variant uses FP32 accumulators internally, converting to FP16 on store.
//
// ============================================================================

kernel void gemm_trellis_w2_fp32acc(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales             [[buffer(2)]],
    constant float* grid               [[buffer(3)]],
    constant float* su                 [[buffer(4)]],
    constant float* sv                 [[buffer(5)]],
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
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup uchar weight_prefetch[2][SIMDGROUPS_PER_TG][W2_PREFETCH_BUF_SIZE_PER_SG];

    float4 grid_reg = float4(grid[0], grid[1], grid[2], grid[3]);

    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;

    if (tg_row >= M) return;

    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;

    // FP32 accumulators for better precision
    simdgroup_matrix<float, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const uint tiles_n = (N + W2_TRELLIS_TILE_DIM - 1) / W2_TRELLIS_TILE_DIM;

    uint buf_compute = 0;

    load_A_tile_w2(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    prefetch_w2_weights(
        packed_indices, weight_prefetch[0][simd_group_id],
        0, tg_col, sg_col_offset, tiles_n, simd_lane_id, K
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        if (next_k < K) {
            load_A_tile_w2(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
            prefetch_w2_weights(
                packed_indices, weight_prefetch[buf_load][simd_group_id],
                next_k, tg_col, sg_col_offset, tiles_n, simd_lane_id, K
            );
        }

        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(
                    a_frag[mi],
                    &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                    TILE_K
                );
            }

            float sv_cached[8];
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                uint k_idx = k_sub_base + row;
                sv_cached[row] = (k_idx < K) ? sv[k_idx] : 0.0f;
            }

            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        uint trellis_tile_n = b_col / W2_TRELLIS_TILE_DIM;
                        uint local_n = b_col % W2_TRELLIS_TILE_DIM;
                        uint scale_idx = group_idx * N + b_col;

                        float scale = scales[scale_idx];
                        float row_sign = su[b_col];
                        float scale_row = scale * row_sign;

                        float combined_scale[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            combined_scale[row] = scale_row * sv_cached[row];
                        }

                        #pragma unroll(4)
                        for (uint row = 0; row < 8; row += 2) {
                            uint k_idx_0 = k_sub_base + row;
                            uint k_idx_1 = k_sub_base + row + 1;

                            uint actual_tile_k_0 = k_idx_0 / W2_TRELLIS_TILE_DIM;
                            uint actual_tile_k_1 = k_idx_1 / W2_TRELLIS_TILE_DIM;
                            uint local_k_0 = k_idx_0 % W2_TRELLIS_TILE_DIM;
                            uint local_k_1 = k_idx_1 % W2_TRELLIS_TILE_DIM;

                            uint tile_offset_0 = (actual_tile_k_0 * tiles_n + trellis_tile_n) * W2_PACKED_BYTES_PER_TILE;
                            uint tile_offset_1 = (actual_tile_k_1 * tiles_n + trellis_tile_n) * W2_PACKED_BYTES_PER_TILE;

                            uint idx_in_tile_0 = local_n * W2_TRELLIS_TILE_DIM + local_k_0;
                            uint idx_in_tile_1 = local_n * W2_TRELLIS_TILE_DIM + local_k_1;

                            uint trellis_idx_0 = (k_idx_0 < K) ?
                                unpack_w2_index(packed_indices + tile_offset_0, idx_in_tile_0) : 0;
                            uint trellis_idx_1 = (k_idx_1 < K) ?
                                unpack_w2_index(packed_indices + tile_offset_1, idx_in_tile_1) : 0;

                            dequant_vals[row] = (k_idx_0 < K) ?
                                dequant_w2_element(trellis_idx_0, combined_scale[row], grid_reg) : half(0.0h);
                            dequant_vals[row + 1] = (k_idx_1 < K) ?
                                dequant_w2_element(trellis_idx_1, combined_scale[row + 1], grid_reg) : half(0.0h);
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
                    // FP32 accumulation
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }

    // Epilogue: convert FP32 accumulators to FP16 and store
    threadgroup half epilogue_staging[SIMDGROUPS_PER_TG][8][8];

    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            // Convert FP32 to FP16 for storage
            simdgroup_matrix<half, 8, 8> acc_h16;
            threadgroup float fp32_staging[8][8];
            simdgroup_store(acc[mi][ni], &fp32_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Convert in parallel
            if (simd_lane_id < 32) {
                uint idx = simd_lane_id;
                if (idx < 64) {
                    uint r = idx / 8;
                    uint c = idx % 8;
                    epilogue_staging[simd_group_id][r][c] = half(fp32_staging[r][c]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_load(acc_h16, &epilogue_staging[simd_group_id][0][0], 8);
                simdgroup_store(acc_h16, C + out_row * N + out_col, N);
                continue;
            }

            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[simd_group_id][r][c];
                }
            }
        }
    }
}

// ============================================================================
// Decode-optimized 2-bit GEMM (batch_size=1, long sequence)
// ============================================================================
//
// Optimized for autoregressive decode where M is very small (typically 1-8)
// but K and N are large. Uses smaller tile dimensions and prioritizes
// latency over throughput.
//
// ============================================================================

constant constexpr uint W2_DECODE_TILE_M = 32;
constant constexpr uint W2_DECODE_TILE_N = 128;
constant constexpr uint W2_DECODE_SIMDGROUPS = 4;

kernel void gemm_trellis_w2_decode(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    constant float* scales             [[buffer(2)]],
    constant float* grid               [[buffer(3)]],
    constant float* su                 [[buffer(4)]],
    constant float* sv                 [[buffer(5)]],
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
    // Smaller threadgroup memory for decode
    threadgroup half A_tiles[2][W2_DECODE_TILE_M][TILE_K];
    threadgroup half B_staging[W2_DECODE_SIMDGROUPS][8][8];

    float4 grid_reg = float4(grid[0], grid[1], grid[2], grid[3]);

    const uint tg_row = tgid.y * W2_DECODE_TILE_M;
    const uint tg_col = tgid.x * W2_DECODE_TILE_N;

    if (tg_row >= M) return;

    // 4 simdgroups: 2x2 grid covering 32x128 output
    const uint sg_row_offset = (simd_group_id / 2) * 16;
    const uint sg_col_offset = (simd_group_id % 2) * 64;

    // 2x8 MMA tiles per simdgroup (16x64 output)
    constexpr uint DECODE_SG_M_TILES = 2;
    constexpr uint DECODE_SG_N_TILES = 8;

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
    const uint tiles_n = (N + W2_TRELLIS_TILE_DIM - 1) / W2_TRELLIS_TILE_DIM;

    uint buf_compute = 0;

    // Simplified A-tile load for smaller decode tile
    constexpr uint DECODE_ELEMS = W2_DECODE_TILE_M * TILE_K;  // 32 * 32 = 1024
    constexpr uint DECODE_THREADS = W2_DECODE_SIMDGROUPS * 32;  // 128
    constexpr uint DECODE_ELEMS_PER_THREAD = DECODE_ELEMS / DECODE_THREADS;  // 8

    // Load first A tile
    #pragma unroll
    for (uint i = 0; i < DECODE_ELEMS_PER_THREAD; ++i) {
        uint flat_idx = thread_idx * DECODE_ELEMS_PER_THREAD + i;
        uint local_row = flat_idx / TILE_K;
        uint local_col = flat_idx % TILE_K;
        uint global_row = tg_row + local_row;
        uint global_col = local_col;

        half val = half(0.0h);
        if (global_row < M && global_col < K) {
            val = A[global_row * K + global_col];
        }
        A_tiles[0][local_row][local_col] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;

        // Prefetch next A tile
        if (next_k < K) {
            #pragma unroll
            for (uint i = 0; i < DECODE_ELEMS_PER_THREAD; ++i) {
                uint flat_idx = thread_idx * DECODE_ELEMS_PER_THREAD + i;
                uint local_row = flat_idx / TILE_K;
                uint local_col = flat_idx % TILE_K;
                uint global_row = tg_row + local_row;
                uint global_col = next_k + local_col;

                half val = half(0.0h);
                if (global_row < M && global_col < K) {
                    val = A[global_row * K + global_col];
                }
                A_tiles[buf_load][local_row][local_col] = val;
            }
        }

        // Compute
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;

            simdgroup_matrix<half, 8, 8> a_frag[DECODE_SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
                simdgroup_load(
                    a_frag[mi],
                    &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                    TILE_K
                );
            }

            float sv_cached[8];
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                uint k_idx = k_sub_base + row;
                sv_cached[row] = (k_idx < K) ? sv[k_idx] : 0.0f;
            }

            #pragma unroll
            for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;

                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];

                    if (b_col < N) {
                        uint trellis_tile_n = b_col / W2_TRELLIS_TILE_DIM;
                        uint local_n = b_col % W2_TRELLIS_TILE_DIM;
                        uint scale_idx = group_idx * N + b_col;

                        float scale = scales[scale_idx];
                        float row_sign = su[b_col];
                        float scale_row = scale * row_sign;

                        float combined_scale[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            combined_scale[row] = scale_row * sv_cached[row];
                        }

                        #pragma unroll(4)
                        for (uint row = 0; row < 8; row += 2) {
                            uint k_idx_0 = k_sub_base + row;
                            uint k_idx_1 = k_sub_base + row + 1;

                            uint actual_tile_k_0 = k_idx_0 / W2_TRELLIS_TILE_DIM;
                            uint actual_tile_k_1 = k_idx_1 / W2_TRELLIS_TILE_DIM;
                            uint local_k_0 = k_idx_0 % W2_TRELLIS_TILE_DIM;
                            uint local_k_1 = k_idx_1 % W2_TRELLIS_TILE_DIM;

                            uint tile_offset_0 = (actual_tile_k_0 * tiles_n + trellis_tile_n) * W2_PACKED_BYTES_PER_TILE;
                            uint tile_offset_1 = (actual_tile_k_1 * tiles_n + trellis_tile_n) * W2_PACKED_BYTES_PER_TILE;

                            uint idx_in_tile_0 = local_n * W2_TRELLIS_TILE_DIM + local_k_0;
                            uint idx_in_tile_1 = local_n * W2_TRELLIS_TILE_DIM + local_k_1;

                            uint trellis_idx_0 = (k_idx_0 < K) ?
                                unpack_w2_index(packed_indices + tile_offset_0, idx_in_tile_0) : 0;
                            uint trellis_idx_1 = (k_idx_1 < K) ?
                                unpack_w2_index(packed_indices + tile_offset_1, idx_in_tile_1) : 0;

                            dequant_vals[row] = (k_idx_0 < K) ?
                                dequant_w2_element(trellis_idx_0, combined_scale[row], grid_reg) : half(0.0h);
                            dequant_vals[row + 1] = (k_idx_1 < K) ?
                                dequant_w2_element(trellis_idx_1, combined_scale[row + 1], grid_reg) : half(0.0h);
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

    // Store results
    threadgroup half epilogue_staging[W2_DECODE_SIMDGROUPS][8][8];

    #pragma unroll
    for (uint mi = 0; mi < DECODE_SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;

        #pragma unroll
        for (uint ni = 0; ni < DECODE_SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;

            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }

            simdgroup_store(acc[mi][ni], &epilogue_staging[simd_group_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;

                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[simd_group_id][r][c];
                }
            }
        }
    }
}
