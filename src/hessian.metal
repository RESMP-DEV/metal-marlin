// hessian.metal - Hessian matrix computation for activations
//
// Computes H = 2 * X^T @ X where X is the activation matrix [n_samples, hidden_dim].
// The Hessian is used in quantization-aware training for optimal layer-wise scaling.
//
// Kernels:
//   1. hessian_compute      - Compute H = 2 * X^T @ X
//   2. hessian_accumulate   - Accumulate H += 2 * X^T @ X (streaming/layer-wise)
//   3. hessian_normalize    - Normalize H /= n_samples
//
// Numerical precision:
//   - BF16 compute with FP32 accumulation for numerical stability
//   - Final output in FP32 (can be downcast by caller if needed)
//
// Tiling strategy:
//   - Processes the symmetric Hessian in tiles to handle large hidden_dim (4096+)
//   - Each threadgroup computes a tile of the output Hessian
//   - Uses simdgroup matrix multiply-accumulate for efficiency

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "bf16_compat.metal"

using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions tuned for Apple Silicon
//
// Hessian is symmetric: H[i,j] = H[j,i]
// We only compute the upper triangle and mirror if needed.
//
// TILE_DIM = 64: Covers a 64x64 tile of the Hessian
// TILE_K = 32: Process 32 samples at a time (K dimension = n_samples)
//
// Threadgroup memory (double-buffered):
//   X_tiles: 2 * 32 * 64 * 2B = 8192 bytes (treating X^T as K x D)
//   Total < 32KB per threadgroup
// ---------------------------------------------------------------------------

constant constexpr uint HESSIAN_TILE_DIM = 64;
constant constexpr uint HESSIAN_TILE_K = 32;
constant constexpr uint HESSIAN_SIMDGROUPS_PER_TG = 4;
constant constexpr uint HESSIAN_THREADS_PER_TG = HESSIAN_SIMDGROUPS_PER_TG * 32;  // 128

// Number of 8x8 sub-tiles in each dimension
constant constexpr uint HESSIAN_SG_TILES = HESSIAN_TILE_DIM / 8;  // 8
constant constexpr uint HESSIAN_K_TILES = HESSIAN_TILE_K / 8;     // 4

// Sub-tiles per simdgroup (4 simdgroups cover 8x8 tile grid = 2x2 per simdgroup)
constant constexpr uint SG_TILES_PER_DIM = HESSIAN_SG_TILES / 2;  // 4

// Number of K tiles to process per mainloop iteration
constant constexpr uint HESSIAN_NUM_BUFFERS = 2;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Cooperative tile loaders for X (activations)
//
// X is [n_samples, hidden_dim] row-major
// We need X^T for the computation, so we load columns of X (rows of X^T)
// ---------------------------------------------------------------------------

/// Load a tile of X into threadgroup memory.
/// X is [n_samples, hidden_dim], we load X[k_offset : k_offset+TILE_K, d_offset : d_offset+TILE_DIM]
inline void load_X_tile_bf16(
    device const ushort* X,  // BF16 storage
    threadgroup float (&X_buf)[HESSIAN_TILE_K][HESSIAN_TILE_DIM],
    uint n_samples, uint hidden_dim,
    uint k_block, uint d_block,
    uint thread_idx
) {
    const uint elems_per_thread = (HESSIAN_TILE_K * HESSIAN_TILE_DIM) / HESSIAN_THREADS_PER_TG;  // 16
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / HESSIAN_TILE_DIM;  // k dimension (sample index)
        uint col = flat_idx % HESSIAN_TILE_DIM;  // d dimension (hidden dim)
        
        uint global_k = k_block + row;
        uint global_d = d_block + col;
        
        float val = 0.0f;
        if (global_k < n_samples && global_d < hidden_dim) {
            val = bf16_bits_to_float(X[global_k * hidden_dim + global_d]);
        }
        X_buf[row][col] = val;
    }
}

/// Load X tile for FP16 input
inline void load_X_tile_fp16(
    device const half* X,
    threadgroup float (&X_buf)[HESSIAN_TILE_K][HESSIAN_TILE_DIM],
    uint n_samples, uint hidden_dim,
    uint k_block, uint d_block,
    uint thread_idx
) {
    const uint elems_per_thread = (HESSIAN_TILE_K * HESSIAN_TILE_DIM) / HESSIAN_THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / HESSIAN_TILE_DIM;
        uint col = flat_idx % HESSIAN_TILE_DIM;
        
        uint global_k = k_block + row;
        uint global_d = d_block + col;
        
        float val = 0.0f;
        if (global_k < n_samples && global_d < hidden_dim) {
            val = float(X[global_k * hidden_dim + global_d]);
        }
        X_buf[row][col] = val;
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute for Hessian: X^T @ X
//
// For Hessian tile at (d_i, d_j):
//   H[d_i:d_i+8, d_j:d_j+8] += X[:, d_i:d_i+8]^T @ X[:, d_j:d_j+8]
//
// We process K in tiles, loading X[k:k+TILE_K, d_i:d_i+TILE_DIM] and
// X[k:k+TILE_K, d_j:d_j+TILE_DIM], then computing the matrix product.
// ---------------------------------------------------------------------------

__attribute__((always_inline))
inline void hessian_compute_from_tiles(
    threadgroup const float (&X_left)[HESSIAN_TILE_K][HESSIAN_TILE_DIM],
    threadgroup const float (&X_right)[HESSIAN_TILE_K][HESSIAN_TILE_DIM],
    thread simdgroup_matrix<float, 8, 8> (&acc)[SG_TILES_PER_DIM][SG_TILES_PER_DIM],
    uint sg_row_offset,
    uint sg_col_offset
) {
    // Process K_TILES sub-tiles in the K dimension
    for (uint kt = 0; kt < HESSIAN_K_TILES; ++kt) {
        for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
            simdgroup_matrix<float, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &X_left[kt * 8][sg_row_offset + mi * 8],
                           HESSIAN_TILE_DIM);
            
            for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
                simdgroup_matrix<float, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &X_right[kt * 8][sg_col_offset + ni * 8],
                               HESSIAN_TILE_DIM);
                
                simdgroup_multiply_accumulate(acc[mi][ni],
                                              a_frag,
                                              b_frag,
                                              acc[mi][ni]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Store accumulated results to Hessian output
//
// Hessian is symmetric: H[i,j] = H[j,i]
// We compute full tiles. The kernel handles diagonal tiles specially.
// ---------------------------------------------------------------------------

__attribute__((always_inline))
inline void hessian_store_results(
    thread simdgroup_matrix<float, 8, 8> (&acc)[SG_TILES_PER_DIM][SG_TILES_PER_DIM],
    device float* H,  // [hidden_dim, hidden_dim] row-major
    uint hidden_dim,
    uint d_i, uint d_j,
    uint sg_row_offset, uint sg_col_offset,
    uint simd_lane,
    threadgroup float (&staging)[8][8]
) {
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        uint out_row = d_i + sg_row_offset + mi * 8;
        if (out_row >= hidden_dim) continue;
        
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            uint out_col = d_j + sg_col_offset + ni * 8;
            if (out_col >= hidden_dim) continue;
            
            // Store to staging first
            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Scatter to global memory with bounds checking
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < hidden_dim && gc < hidden_dim) {
                    H[gr * hidden_dim + gc] = staging[r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Hessian compute
//
// Computes H = 2 * X^T @ X where X is [n_samples, hidden_dim]
// Output H is [hidden_dim, hidden_dim] symmetric matrix
//
// Dispatch: Grid ceil(hidden_dim/TILE_DIM) x ceil(hidden_dim/TILE_DIM)
//           Each threadgroup computes one tile of the Hessian
// ---------------------------------------------------------------------------

kernel void hessian_compute(
    device const ushort* X      [[buffer(0)]],  // [n_samples, hidden_dim] BF16
    device float* H             [[buffer(1)]],  // [hidden_dim, hidden_dim] FP32 output
    constant uint& n_samples    [[buffer(2)]],
    constant uint& hidden_dim   [[buffer(3)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    // Double-buffered threadgroup memory
    threadgroup float X_left[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    threadgroup float X_right[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    threadgroup float staging[8][8];
    
    // Tile position in Hessian output
    const uint d_i = tgid.y * HESSIAN_TILE_DIM;  // Row in Hessian
    const uint d_j = tgid.x * HESSIAN_TILE_DIM;  // Col in Hessian
    
    // Each simdgroup handles a sub-tile (4x4 8x8 tiles = 32x32 elements)
    const uint sg_row_offset = (simd_id / 2) * (SG_TILES_PER_DIM * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_TILES_PER_DIM * 8);
    
    // Initialize accumulators to zero
    simdgroup_matrix<float, 8, 8> acc[SG_TILES_PER_DIM][SG_TILES_PER_DIM];
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    
    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(n_samples, HESSIAN_TILE_K);
    
    if (num_k_tiles == 0) {
        return;
    }
    
    // For diagonal tiles (d_i == d_j), we only need to load X once
    const bool is_diagonal = (d_i == d_j);
    
    uint buf_compute = 0;
    
    // Prologue: Load first K-tile
    load_X_tile_bf16(X, X_left[0], n_samples, hidden_dim, 0, d_i, thread_idx);
    if (!is_diagonal) {
        load_X_tile_bf16(X, X_right[0], n_samples, hidden_dim, 0, d_j, thread_idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Main K-reduction loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * HESSIAN_TILE_K;
        uint next_k = k_offset + HESSIAN_TILE_K;
        uint buf_load = 1 - buf_compute;
        
        // Prefetch next tile
        if (next_k < n_samples) {
            load_X_tile_bf16(X, X_left[buf_load], n_samples, hidden_dim, next_k, d_i, thread_idx);
            if (!is_diagonal) {
                load_X_tile_bf16(X, X_right[buf_load], n_samples, hidden_dim, next_k, d_j, thread_idx);
            }
        }
        
        // Compute on current buffer
        if (is_diagonal) {
            // For diagonal: X_left^T @ X_left
            hessian_compute_from_tiles(X_left[buf_compute], X_left[buf_compute],
                                       acc, sg_row_offset, sg_col_offset);
        } else {
            // For off-diagonal: X_left^T @ X_right
            hessian_compute_from_tiles(X_left[buf_compute], X_right[buf_compute],
                                       acc, sg_row_offset, sg_col_offset);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store results with multiply by 2: H = 2 * X^T @ X
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        uint out_row = d_i + sg_row_offset + mi * 8;
        if (out_row >= hidden_dim) continue;
        
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            uint out_col = d_j + sg_col_offset + ni * 8;
            if (out_col >= hidden_dim) continue;
            
            // Multiply by 2 using simdgroup math
            // We do: result = acc * 2 by loading, scaling, and storing
            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Scale by 2
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                staging[r][c] *= 2.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Store to global memory
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < hidden_dim && gc < hidden_dim) {
                    H[gr * hidden_dim + gc] = staging[r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Hessian compute (FP16 input variant)
// ---------------------------------------------------------------------------

kernel void hessian_compute_fp16(
    device const half* X        [[buffer(0)]],  // [n_samples, hidden_dim] FP16
    device float* H             [[buffer(1)]],  // [hidden_dim, hidden_dim] FP32 output
    constant uint& n_samples    [[buffer(2)]],
    constant uint& hidden_dim   [[buffer(3)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float X_left[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    threadgroup float X_right[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    threadgroup float staging[8][8];
    
    const uint d_i = tgid.y * HESSIAN_TILE_DIM;
    const uint d_j = tgid.x * HESSIAN_TILE_DIM;
    
    const uint sg_row_offset = (simd_id / 2) * (SG_TILES_PER_DIM * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_TILES_PER_DIM * 8);
    
    simdgroup_matrix<float, 8, 8> acc[SG_TILES_PER_DIM][SG_TILES_PER_DIM];
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    
    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(n_samples, HESSIAN_TILE_K);
    
    if (num_k_tiles == 0) return;
    
    const bool is_diagonal = (d_i == d_j);
    uint buf_compute = 0;
    
    load_X_tile_fp16(X, X_left[0], n_samples, hidden_dim, 0, d_i, thread_idx);
    if (!is_diagonal) {
        load_X_tile_fp16(X, X_right[0], n_samples, hidden_dim, 0, d_j, thread_idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * HESSIAN_TILE_K;
        uint next_k = k_offset + HESSIAN_TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < n_samples) {
            load_X_tile_fp16(X, X_left[buf_load], n_samples, hidden_dim, next_k, d_i, thread_idx);
            if (!is_diagonal) {
                load_X_tile_fp16(X, X_right[buf_load], n_samples, hidden_dim, next_k, d_j, thread_idx);
            }
        }
        
        if (is_diagonal) {
            hessian_compute_from_tiles(X_left[buf_compute], X_left[buf_compute],
                                       acc, sg_row_offset, sg_col_offset);
        } else {
            hessian_compute_from_tiles(X_left[buf_compute], X_right[buf_compute],
                                       acc, sg_row_offset, sg_col_offset);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store results with multiply by 2
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        uint out_row = d_i + sg_row_offset + mi * 8;
        if (out_row >= hidden_dim) continue;
        
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            uint out_col = d_j + sg_col_offset + ni * 8;
            if (out_col >= hidden_dim) continue;
            
            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Scale by 2
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                staging[r][c] *= 2.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < hidden_dim && gc < hidden_dim) {
                    H[gr * hidden_dim + gc] = staging[r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Hessian accumulate
//
// Accumulates into existing Hessian: H += 2 * X^T @ X
// For memory-efficient layer-by-layer processing
// ---------------------------------------------------------------------------

kernel void hessian_accumulate(
    device const ushort* X      [[buffer(0)]],  // [n_samples, hidden_dim] BF16
    device float* H             [[buffer(1)]],  // [hidden_dim, hidden_dim] FP32 in/out
    constant uint& n_samples    [[buffer(2)]],
    constant uint& hidden_dim   [[buffer(3)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float X_left[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    threadgroup float X_right[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    threadgroup float staging[8][8];
    
    const uint d_i = tgid.y * HESSIAN_TILE_DIM;
    const uint d_j = tgid.x * HESSIAN_TILE_DIM;
    
    const uint sg_row_offset = (simd_id / 2) * (SG_TILES_PER_DIM * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_TILES_PER_DIM * 8);
    
    simdgroup_matrix<float, 8, 8> acc[SG_TILES_PER_DIM][SG_TILES_PER_DIM];
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    
    const uint thread_idx = simd_id * 32 + simd_lane;
    const uint num_k_tiles = div_ceil(n_samples, HESSIAN_TILE_K);
    
    if (num_k_tiles == 0) return;
    
    const bool is_diagonal = (d_i == d_j);
    uint buf_compute = 0;
    
    load_X_tile_bf16(X, X_left[0], n_samples, hidden_dim, 0, d_i, thread_idx);
    if (!is_diagonal) {
        load_X_tile_bf16(X, X_right[0], n_samples, hidden_dim, 0, d_j, thread_idx);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * HESSIAN_TILE_K;
        uint next_k = k_offset + HESSIAN_TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < n_samples) {
            load_X_tile_bf16(X, X_left[buf_load], n_samples, hidden_dim, next_k, d_i, thread_idx);
            if (!is_diagonal) {
                load_X_tile_bf16(X, X_right[buf_load], n_samples, hidden_dim, next_k, d_j, thread_idx);
            }
        }
        
        if (is_diagonal) {
            hessian_compute_from_tiles(X_left[buf_compute], X_left[buf_compute],
                                       acc, sg_row_offset, sg_col_offset);
        } else {
            hessian_compute_from_tiles(X_left[buf_compute], X_right[buf_compute],
                                       acc, sg_row_offset, sg_col_offset);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Load existing Hessian and accumulate: H += 2 * X^T @ X
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        uint out_row = d_i + sg_row_offset + mi * 8;
        if (out_row >= hidden_dim) continue;
        
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            uint out_col = d_j + sg_col_offset + ni * 8;
            if (out_col >= hidden_dim) continue;
            
            // Load existing value from H
            simdgroup_matrix<float, 8, 8> existing;
            if (out_row + 8 <= hidden_dim && out_col + 8 <= hidden_dim) {
                simdgroup_load(existing, H + out_row * hidden_dim + out_col, hidden_dim);
            } else {
                // Bounds-checked load
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    staging[r][c] = (gr < hidden_dim && gc < hidden_dim) ? 
                                    H[gr * hidden_dim + gc] : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(existing, &staging[0][0], 8);
            }
            
            // contribution = 2 * acc
            simdgroup_store(acc[mi][ni], &staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                staging[r][c] *= 2.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            simdgroup_matrix<float, 8, 8> contribution;
            simdgroup_load(contribution, &staging[0][0], 8);
            
            // H += contribution using multiply_accumulate with identity
            // result = existing + contribution
            // We use: result = 1 * existing + contribution via multiply_accumulate
            simdgroup_matrix<float, 8, 8> one = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
            simdgroup_matrix<float, 8, 8> result;
            simdgroup_multiply_accumulate(result, one, existing, contribution);
            
            // Store back
            simdgroup_store(result, &staging[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < hidden_dim && gc < hidden_dim) {
                    H[gr * hidden_dim + gc] = staging[r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Hessian normalize
//
// Simple elementwise division: H /= n_samples
// Called after all accumulations are complete
// ---------------------------------------------------------------------------

kernel void hessian_normalize(
    device float* H             [[buffer(0)]],  // [hidden_dim, hidden_dim] FP32 in/out
    constant uint& hidden_dim   [[buffer(1)]],
    constant uint& n_samples    [[buffer(2)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles a tile of the Hessian
    const uint d_i = tgid.y * HESSIAN_TILE_DIM;
    const uint d_j = tgid.x * HESSIAN_TILE_DIM;
    
    const uint sg_row_offset = (simd_id / 2) * (SG_TILES_PER_DIM * 8);
    const uint sg_col_offset = (simd_id % 2) * (SG_TILES_PER_DIM * 8);
    
    const float inv_n = 1.0f / float(n_samples);
    
    // Each simdgroup processes its sub-tile
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        uint out_row = d_i + sg_row_offset + mi * 8;
        if (out_row >= hidden_dim) continue;
        
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            uint out_col = d_j + sg_col_offset + ni * 8;
            if (out_col >= hidden_dim) continue;
            
            if (out_row + 8 <= hidden_dim && out_col + 8 <= hidden_dim) {
                // Fast path: full 8x8 tile
                simdgroup_matrix<float, 8, 8> tile;
                simdgroup_load(tile, H + out_row * hidden_dim + out_col, hidden_dim);
                
                // Multiply by inv_n: tile * inv_n
                // We need to scale each element - use staging
                threadgroup float staging_local[8][8];
                simdgroup_store(tile, &staging_local[0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    staging_local[r][c] *= inv_n;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                simdgroup_load(tile, &staging_local[0][0], 8);
                simdgroup_store(tile, H + out_row * hidden_dim + out_col, hidden_dim);
            } else {
                // Slow path: bounds checking
                threadgroup float staging[8][8];
                
                // Load
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    staging[r][c] = (gr < hidden_dim && gc < hidden_dim) ? 
                                    H[gr * hidden_dim + gc] : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Scale
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    staging[r][c] *= inv_n;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Store
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < hidden_dim && gc < hidden_dim) {
                        H[gr * hidden_dim + gc] = staging[r][c];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simple 1D elementwise normalize (alternative dispatch pattern)
// ---------------------------------------------------------------------------

kernel void hessian_normalize_1d(
    device float* H             [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]],
    constant uint& n_samples    [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid < num_elements) {
        H[tid] /= float(n_samples);
    }
}
