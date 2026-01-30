// gemm_trellis.metal - Fused Dequant+GEMM Kernel for Trellis-Quantized Weights
// ============================================================================
//
// High-performance fused kernel for trellis-quantized GEMM on Apple Metal.
// Performs on-the-fly dequantization of packed trellis indices during the
// GEMM computation without materializing the full FP16 weight matrix.
//
// Trellis Quantization Background:
//   - Weights are quantized using Viterbi algorithm through a codebook grid
//   - Each 16x16 tile (256 elements) shares quantization parameters
//   - Indices are packed into uint8: 2-bit (64B), 3-bit (96B), or 4-bit (128B)
//   - Dequant formula: w[k,n] = grid[idx] * scale * su[k] * sv[n]
//
// Kernel Architecture:
//   - Tile dimensions: TILE_M=64, TILE_N=64, TILE_K=32
//   - Threadgroup: 4 simdgroups (128 threads), each handling 32x32 output
//   - Double-buffered A tiles in threadgroup memory (8KB per buffer)
//   - On-the-fly B dequantization with per-simdgroup staging (512B)
//   - Never materializes full B matrix - dequant happens in registers
//
// Memory Layout:
//   - A: [M, K] half - input activations (row-major)
//   - packed_indices: [tiles_k, tiles_n, packed_bytes] uint8 - packed trellis
//   - scales: [K/group_size, N] float32 - per-group scale factors
//   - grid: [n_levels] float32 - codebook quantization centers
//   - su: [K] float32 - row signs for Hadamard inverse
//   - sv: [N] float32 - column signs for Hadamard inverse
//   - C: [M, N] half - output matrix (row-major)
//
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// Tile Dimensions - Optimized for Apple Silicon
// ============================================================================

// Main GEMM tile dimensions
constant constexpr uint TILE_M = 64;      // Output rows per threadgroup
constant constexpr uint TILE_N = 64;      // Output cols per threadgroup  
constant constexpr uint TILE_K = 32;      // K-reduction per mainloop iteration
constant constexpr uint K_TILES = TILE_K / 8;  // 4 simdgroup MMA ops per K-block

// Simdgroup configuration: 4 simdgroups tile the 64x64 output
// Each simdgroup handles 32x32 output (4x4 blocks of 8x8 tiles)
constant constexpr uint SIMDGROUPS_PER_TG = 4;
constant constexpr uint THREADS_PER_TG = SIMDGROUPS_PER_TG * 32;  // 128 threads
constant constexpr uint SG_M_TILES = 4;   // 4 rows of 8x8 = 32 rows per simdgroup
constant constexpr uint SG_N_TILES = 4;   // 4 cols of 8x8 = 32 cols per simdgroup

// Trellis tile dimensions (weights are stored in 16x16 tiles)
constant constexpr uint TRELLIS_TILE_DIM = 16;   // 16x16 tile dimension
constant constexpr uint TRELLIS_TILE_SIZE = 256; // Total elements per tile

// ============================================================================
// Packed Index Unpacking Primitives
// ============================================================================

/// Unpack a 2-bit trellis index from packed bytes.
/// Layout: 4 indices per byte (bits [0:1], [2:3], [4:5], [6:7])
inline uint unpack_2bit_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 2;        // Divide by 4
    uint bit_offset = (idx_in_tile & 3) << 1; // (idx % 4) * 2
    return (packed[byte_idx] >> bit_offset) & 0x3;
}

/// Unpack a 3-bit trellis index from packed bytes.
/// Layout: 8 indices per 3 bytes (packed across byte boundaries)
/// This is the most complex packing due to non-byte alignment.
inline uint unpack_3bit_index(device const uchar* packed, uint idx_in_tile) {
    uint bit_offset = idx_in_tile * 3;
    uint byte_idx = bit_offset >> 3;         // Divide by 8
    uint bit_in_byte = bit_offset & 7;       // Modulo 8
    
    // Read up to 2 bytes (indices may span byte boundary)
    uint packed_val = uint(packed[byte_idx]);
    if (bit_in_byte + 3 > 8) {
        packed_val |= uint(packed[byte_idx + 1]) << 8;
    }
    return (packed_val >> bit_in_byte) & 0x7;
}

/// Unpack a 4-bit trellis index from packed bytes.
/// Layout: 2 indices per byte (bits [0:3], [4:7])
inline uint unpack_4bit_index(device const uchar* packed, uint idx_in_tile) {
    uint byte_idx = idx_in_tile >> 1;        // Divide by 2
    uint shift = (idx_in_tile & 1) << 2;      // 0 or 4
    return (packed[byte_idx] >> shift) & 0xF;
}

/// Generic index unpack dispatcher.
/// Routes to the appropriate unpack function based on bit width.
inline uint unpack_trellis_index(device const uchar* packed, uint idx_in_tile, uint bits) {
    switch (bits) {
        case 2: return unpack_2bit_index(packed, idx_in_tile);
        case 3: return unpack_3bit_index(packed, idx_in_tile);
        case 4: return unpack_4bit_index(packed, idx_in_tile);
        default: return unpack_3bit_index(packed, idx_in_tile);  // Default to 3-bit
    }
}

/// Compute packed bytes per trellis tile based on bit width.
inline uint packed_bytes_per_trellis_tile(uint bits) {
    return (TRELLIS_TILE_SIZE * bits + 7) / 8;  // ceil(256 * bits / 8)
}

// ============================================================================
// Dequantization Primitives
// ============================================================================

/// Dequantize a single trellis element.
/// Formula: dequant = grid[idx] * scale * su * sv
inline half dequant_trellis_element(
    uint idx, 
    float scale, 
    float su, 
    float sv, 
    device const float* grid
) {
    return half(grid[idx] * scale * su * sv);
}

/// Dequantize 8 consecutive trellis elements for a single column.
/// Used during GEMM to load and dequant a column of B on-the-fly.
/// 
/// @param packed        Packed indices for the trellis tile
/// @param local_n       Column index within the 16x16 trellis tile [0, 15]
/// @param scale         Per-column scale factor
/// @param su_vec        Row signs for 16 consecutive K values
/// @param sv            Column sign
/// @param grid          Codebook grid
/// @param out           Output buffer for 8 dequantized values
/// @param bits          Quantization bit width (2, 3, or 4)
inline void dequant_trellis_column_8(
    device const uchar* packed,
    uint local_n,
    float scale,
    thread const float* su_vec,
    float sv,
    device const float* grid,
    thread half* out,
    uint bits
) {
    // Each column has 16 elements in the trellis tile
    // We process 8 at a time (called twice per tile)
    #pragma unroll
    for (uint row = 0; row < 8; ++row) {
        uint idx_in_tile = row * TRELLIS_TILE_DIM + local_n;
        uint trellis_idx = unpack_trellis_index(packed, idx_in_tile, bits);
        out[row] = dequant_trellis_element(trellis_idx, scale, su_vec[row], sv, grid);
    }
}

// ============================================================================
// Cooperative A Tile Loader (all 128 threads)
// ============================================================================

/// Load A tile from global memory to threadgroup memory.
/// Each thread loads TILE_M * TILE_K / THREADS_PER_TG elements.
inline void load_A_tile_cooperative(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    constexpr uint ELEMS_PER_THREAD = (TILE_M * TILE_K) / THREADS_PER_TG;  // 16
    
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
// Main Fused GEMM Kernel: gemm_trellis_packed
// ============================================================================

/// Fused dequant + GEMM for trellis-quantized weights.
///
/// Computes C[M,N] = A[M,K] @ dequant(W[K,N]) where W is trellis-quantized.
/// Weights are dequantized on-the-fly during the GEMM computation.
///
/// @param A               Input activations [M, K] half (row-major)
/// @param packed_indices  Packed trellis indices [tiles_k, tiles_n, packed_bytes] uint8
/// @param scales          Per-group scales [K/group_size, N] float32
/// @param grid            Codebook grid values [n_levels] float32
/// @param su              Row signs [K] float32
/// @param sv              Column signs [N] float32
/// @param C               Output matrix [M, N] half (row-major)
/// @param M               Number of rows in A and C
/// @param K               Number of columns in A / rows in W
/// @param N               Number of columns in W and C
/// @param bits            Quantization bit width (2, 3, or 4)
/// @param n_levels        Number of codebook levels (2^bits)
/// @param group_size      Quantization group size (typically 32 or 64)
kernel void gemm_trellis_packed(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    device const float* scales         [[buffer(2)]],
    device const float* grid           [[buffer(3)]],
    device const float* su             [[buffer(4)]],
    device const float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& bits                [[buffer(10)]],
    constant uint& n_levels            [[buffer(11)]],
    constant uint& group_size          [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // Threadgroup memory allocation
    // -------------------------------------------------------------------------
    // Double-buffered A tiles: 2 * 64 * 32 * 2 bytes = 8KB
    // Per-simdgroup B staging: 4 * 8 * 8 * 2 bytes = 512B
    // Total: ~8.5KB (well within 32KB budget for good occupancy)
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    
    // -------------------------------------------------------------------------
    // Tile assignment
    // -------------------------------------------------------------------------
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    
    // Early exit if this threadgroup is outside M bounds
    if (tg_row >= M) return;
    
    // Simdgroup layout: 2x2 grid covering the 64x64 output tile
    // SG 0: rows [0,31],  cols [0,31]   (simd_id=0)
    // SG 1: rows [0,31],  cols [32,63]  (simd_id=1)
    // SG 2: rows [32,63], cols [0,31]   (simd_id=2)
    // SG 3: rows [32,63], cols [32,63]  (simd_id=3)
    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;
    
    // -------------------------------------------------------------------------
    // Initialize accumulators
    // Each simdgroup accumulates a 32x32 output tile using 8x8 MMA blocks
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
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    uint buf_compute = 0;
    
    // -------------------------------------------------------------------------
    // Prologue: Load first A tile into buffer 0
    // -------------------------------------------------------------------------
    load_A_tile_cooperative(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // -------------------------------------------------------------------------
    // Main pipeline loop - Double buffered
    // -------------------------------------------------------------------------
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        
        // --- Async load next A tile while computing on current ---
        if (next_k < K) {
            load_A_tile_cooperative(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
        }
        
        // --- Compute: Fused B dequant + MMA ---
        // For each K sub-tile (8 elements), we:
        //   1. Load A fragments from threadgroup memory (reused across N)
        //   2. Dequant B on-the-fly for each N sub-tile
        //   3. Perform MMA accumulation
        
        #pragma unroll
        for (uint kk = 0; kk < K_TILES; ++kk) {
            uint k_sub_base = k_block + kk * 8;
            uint group_idx = k_sub_base / group_size;
            
            // Calculate trellis tile coordinates for this K sub-tile
            uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
            uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;
            
            // Load A fragments for this K sub-tile (reused across all N tiles)
            simdgroup_matrix<half, 8, 8> a_frag[SG_M_TILES];
            #pragma unroll
            for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                simdgroup_load(
                    a_frag[mi],
                    &A_tiles[buf_compute][sg_row_offset + mi * 8][kk * 8],
                    TILE_K
                );
            }
            
            // For each N sub-tile, dequant B and accumulate
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;
                
                // --- Fused B dequantization ---
                // Lanes 0-7 each handle one column of the 8x8 B tile
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];
                    
                    if (b_col < N) {
                        // Calculate trellis tile coordinates
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        
                        // Offset into packed_indices for this trellis tile
                        uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                        device const uchar* tile_packed = packed_indices + tile_offset;
                        
                        // Load scale for this column
                        uint scale_idx = group_idx * N + b_col;
                        float scale = scales[scale_idx];
                        
                        // Load column sign
                        float sign_n = sv[b_col];
                        
                        // Load row signs for 8 consecutive K values
                        float su_vec[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            su_vec[row] = (k_idx < K) ? su[k_idx] : 0.0f;
                        }
                        
                        // Dequantize 8 elements for this column
                        // We need to handle the case where k_sub_base spans trellis tile boundary
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM;
                            
                            if (k_idx < K) {
                                // Recalculate tile offset if we crossed tile boundary
                                uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                                uint actual_tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                                uint idx_in_tile = local_k * TRELLIS_TILE_DIM + local_n;
                                
                                uint trellis_idx = unpack_trellis_index(
                                    packed_indices + actual_tile_offset, idx_in_tile, bits);
                                if (trellis_idx >= n_levels) trellis_idx = 0;
                                
                                dequant_vals[row] = dequant_trellis_element(
                                    trellis_idx, scale, su_vec[row], sign_n, grid);
                            } else {
                                dequant_vals[row] = half(0.0h);
                            }
                        }
                    } else {
                        // Out of bounds - fill with zeros
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            dequant_vals[row] = half(0.0h);
                        }
                    }
                    
                    // Write dequantized values to staging buffer
                    // Layout: B_staging[simd_id][k][n] where k,n in [0,7]
                    #pragma unroll
                    for (uint row = 0; row < 8; ++row) {
                        B_staging[simd_group_id][row][simd_lane_id] = dequant_vals[row];
                    }
                }
                
                // Synchronize within simdgroup (lightweight barrier)
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Load B fragment and perform MMA
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_staging[simd_group_id][0][0], 8);
                
                #pragma unroll
                for (uint mi = 0; mi < SG_M_TILES; ++mi) {
                    // simdgroup_multiply_accumulate is the core SIMD intrinsic
                    // that maps 8x8 tiles to Apple Silicon's matrix pipeline.
                    // It performs: acc += A * B where A,B are 8x8 and acc accumulates.
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                }
            }
        }
        
        // Wait for next tile load before swapping buffers
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // -------------------------------------------------------------------------
    // Epilogue: Store results to global memory
    // -------------------------------------------------------------------------
    // Per-simdgroup staging to avoid race conditions between simdgroups
    threadgroup half epilogue_staging[SIMDGROUPS_PER_TG][8][8];
    
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;
        
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            
            // Fast path: full 8x8 tile within bounds -> direct simdgroup_store
            if (out_row + 8 <= M && out_col + 8 <= N) {
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
                continue;
            }
            
            // Slow path: partial tile or bounds checking needed
            simdgroup_store(acc[mi][ni], &epilogue_staging[simd_group_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Each thread handles 2 elements (32 threads * 2 = 64 elements)
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < M && gc < N) {
                    C[gr * N + gc] = epilogue_staging[simd_group_id][r][c];
                }
            }
            
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// FP32 Accumulation Variant (gemm_trellis_packed_fp32acc)
// ============================================================================

/// FP32 accumulation variant for numerical stability with large K.
/// Same algorithm as above but uses FP32 accumulators internally.
kernel void gemm_trellis_packed_fp32acc(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    device const float* scales         [[buffer(2)]],
    device const float* grid           [[buffer(3)]],
    device const float* su             [[buffer(4)]],
    device const float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& bits                [[buffer(10)]],
    constant uint& n_levels            [[buffer(11)]],
    constant uint& group_size          [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_tiles[2][TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    threadgroup float epilogue_staging_fp32[8][8];
    
    const uint tg_row = tgid.y * TILE_M;
    const uint tg_col = tgid.x * TILE_N;
    
    if (tg_row >= M) return;
    
    const uint sg_row_offset = (simd_group_id / 2) * 32;
    const uint sg_col_offset = (simd_group_id % 2) * 32;
    
    // FP32 accumulators for numerical stability
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
    const uint tiles_n = (N + TRELLIS_TILE_DIM - 1) / TRELLIS_TILE_DIM;
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    uint buf_compute = 0;
    
    load_A_tile_cooperative(A, A_tiles[0], M, K, tg_row, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_block = kt * TILE_K;
        uint next_k = k_block + TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < K) {
            load_A_tile_cooperative(A, A_tiles[buf_load], M, K, tg_row, next_k, thread_idx);
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
            
            #pragma unroll
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                uint b_col_base = tg_col + sg_col_offset + ni * 8;
                
                if (simd_lane_id < 8) {
                    uint b_col = b_col_base + simd_lane_id;
                    half dequant_vals[8];
                    
                    if (b_col < N) {
                        uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
                        uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        
                        uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                        
                        uint scale_idx = group_idx * N + b_col;
                        float scale = scales[scale_idx];
                        float sign_n = sv[b_col];
                        
                        float su_vec[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            su_vec[row] = (k_idx < K) ? su[k_idx] : 0.0f;
                        }
                        
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM;
                            
                            if (k_idx < K) {
                                uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                                uint actual_tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                                uint idx_in_tile = local_k * TRELLIS_TILE_DIM + local_n;
                                
                                uint trellis_idx = unpack_trellis_index(
                                    packed_indices + actual_tile_offset, idx_in_tile, bits);
                                if (trellis_idx >= n_levels) trellis_idx = 0;
                                
                                dequant_vals[row] = dequant_trellis_element(
                                    trellis_idx, scale, su_vec[row], sign_n, grid);
                            } else {
                                dequant_vals[row] = half(0.0h);
                            }
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
                
                // FP32 accumulation (mixed precision)
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
    #pragma unroll
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = tg_row + sg_row_offset + mi * 8;
        if (out_row >= M) continue;
        
        #pragma unroll
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = tg_col + sg_col_offset + ni * 8;
            
            simdgroup_store(acc[mi][ni], &epilogue_staging_fp32[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < M && gc < N) {
                    C[gr * N + gc] = half(epilogue_staging_fp32[r][c]);
                }
            }
            
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ============================================================================
// Decode-Optimized Variant (gemm_trellis_packed_decode)
// ============================================================================

/// Optimized variant for small M (autoregressive decode, M=1-16).
/// Uses 32x128 tiles to maximize N coverage per threadgroup.

constant constexpr uint DECODE_TILE_M = 32;
constant constexpr uint DECODE_TILE_N = 128;
constant constexpr uint DECODE_K_TILES = TILE_K / 8;  // 4

kernel void gemm_trellis_packed_decode(
    device const half* A               [[buffer(0)]],
    device const uchar* packed_indices [[buffer(1)]],
    device const float* scales         [[buffer(2)]],
    device const float* grid           [[buffer(3)]],
    device const float* su             [[buffer(4)]],
    device const float* sv             [[buffer(5)]],
    device half* C                     [[buffer(6)]],
    constant uint& M                   [[buffer(7)]],
    constant uint& K                   [[buffer(8)]],
    constant uint& N                   [[buffer(9)]],
    constant uint& bits                [[buffer(10)]],
    constant uint& n_levels            [[buffer(11)]],
    constant uint& group_size          [[buffer(12)]],
    uint3 tgid                         [[threadgroup_position_in_grid]],
    uint3 tid                          [[thread_position_in_threadgroup]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    // 32x128x32 tiles: A = 2*32*32*2 = 4KB, B_staging = 512B
    threadgroup half A_tiles[2][DECODE_TILE_M][TILE_K];
    threadgroup half B_staging[SIMDGROUPS_PER_TG][8][8];
    
    const uint tg_row = tgid.y * DECODE_TILE_M;
    const uint tg_col = tgid.x * DECODE_TILE_N;
    
    if (tg_row >= M) return;
    
    // 4 simdgroups tile 32x128 as 1x4: each handles 32x32
    const uint sg_row_offset = 0;
    const uint sg_col_offset = simd_group_id * 32;
    
    // Each simdgroup handles 4x4 = 16 sub-tiles of 8x8 within its 32x32 region
    constexpr uint DECODE_SG_M_TILES = 4;   // 32 rows / 8 = 4
    constexpr uint DECODE_SG_N_TILES = 4;   // 32 cols / 8 = 4
    
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
    const uint packed_bytes = packed_bytes_per_trellis_tile(bits);
    
    // A tile load: 32 * 32 = 1024 / 128 = 8 elements per thread
    constexpr uint A_ELEMS = (DECODE_TILE_M * TILE_K) / THREADS_PER_TG;
    
    // Prologue: Load first A tile
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
        
        // Load next A tile
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
        
        // Compute
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
                        uint trellis_tile_k = k_sub_base / TRELLIS_TILE_DIM;
                        uint local_k_base = k_sub_base % TRELLIS_TILE_DIM;
                        uint trellis_tile_n = b_col / TRELLIS_TILE_DIM;
                        uint local_n = b_col % TRELLIS_TILE_DIM;
                        
                        uint tile_offset = (trellis_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                        
                        uint scale_idx = group_idx * N + b_col;
                        float scale = scales[scale_idx];
                        float sign_n = sv[b_col];
                        
                        float su_vec[8];
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            su_vec[row] = (k_idx < K) ? su[k_idx] : 0.0f;
                        }
                        
                        #pragma unroll
                        for (uint row = 0; row < 8; ++row) {
                            uint k_idx = k_sub_base + row;
                            uint local_k = (local_k_base + row) % TRELLIS_TILE_DIM;
                            
                            if (k_idx < K) {
                                uint actual_tile_k = k_idx / TRELLIS_TILE_DIM;
                                uint actual_tile_offset = (actual_tile_k * tiles_n + trellis_tile_n) * packed_bytes;
                                uint idx_in_tile = local_k * TRELLIS_TILE_DIM + local_n;
                                
                                uint trellis_idx = unpack_trellis_index(
                                    packed_indices + actual_tile_offset, idx_in_tile, bits);
                                if (trellis_idx >= n_levels) trellis_idx = 0;
                                
                                dequant_vals[row] = dequant_trellis_element(
                                    trellis_idx, scale, su_vec[row], sign_n, grid);
                            } else {
                                dequant_vals[row] = half(0.0h);
                            }
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
    threadgroup half epilogue_staging[8][8];
    
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
            
            simdgroup_store(acc[mi][ni], &epilogue_staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane_id; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
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
