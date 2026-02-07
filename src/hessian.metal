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
// TILE_K = 16: Process 16 samples at a time (K dimension = n_samples)
//
// Threadgroup memory (double-buffered):
//   X_tiles: 2 * 16 * 64 * 2B = 4096 bytes (treating X^T as K x D)
//   Total < 16KB per threadgroup
// ---------------------------------------------------------------------------

constant constexpr uint HESSIAN_TILE_DIM = 64;
constant constexpr uint HESSIAN_TILE_K = 16;
constant constexpr uint HESSIAN_SIMDGROUPS_PER_TG = 4;
constant constexpr uint HESSIAN_THREADS_PER_TG = HESSIAN_SIMDGROUPS_PER_TG * 32;  // 128

// Number of 8x8 sub-tiles in each dimension
constant constexpr uint HESSIAN_SG_TILES = HESSIAN_TILE_DIM / 8;  // 8
constant constexpr uint HESSIAN_K_TILES = HESSIAN_TILE_K / 8;     // 2

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
// For computing H = X^T @ X, we need:
//   - X_left_T stored as [TILE_DIM, TILE_K] (transposed) for left operand
//   - X_right stored as [TILE_K, TILE_DIM] (normal) for right operand
// ---------------------------------------------------------------------------

/// Load a tile of X into threadgroup memory (normal layout for right operand).
/// X is [n_samples, hidden_dim], we load X[k_offset:k_offset+TILE_K, d_offset:d_offset+TILE_DIM]
inline void load_X_tile_bf16(
    device const ushort* X,  // BF16 storage
    threadgroup float (&X_buf)[HESSIAN_TILE_K][HESSIAN_TILE_DIM],
    uint n_samples, uint hidden_dim,
    uint k_block, uint d_block,
    uint thread_idx
) {
    const uint elems_per_thread = (HESSIAN_TILE_K * HESSIAN_TILE_DIM) / HESSIAN_THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / HESSIAN_TILE_DIM;  // k dimension
        uint col = flat_idx % HESSIAN_TILE_DIM;  // d dimension
        
        uint global_k = k_block + row;
        uint global_d = d_block + col;
        
        float val = 0.0f;
        if (global_k < n_samples && global_d < hidden_dim) {
            val = bf16_bits_to_float(X[global_k * hidden_dim + global_d]);
        }
        X_buf[row][col] = val;
    }
}

/// Load X tile TRANSPOSED into threadgroup memory (for left operand in X^T @ X).
/// Input X is [n_samples, hidden_dim], output is X_buf_T[TILE_DIM][TILE_K] = X^T
inline void load_X_tile_bf16_transposed(
    device const ushort* X,  // BF16 storage
    threadgroup float (&X_buf_T)[HESSIAN_TILE_DIM][HESSIAN_TILE_K],
    uint n_samples, uint hidden_dim,
    uint k_block, uint d_block,
    uint thread_idx
) {
    const uint elems_per_thread = (HESSIAN_TILE_K * HESSIAN_TILE_DIM) / HESSIAN_THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / HESSIAN_TILE_DIM;  // k dimension (will become col)
        uint col = flat_idx % HESSIAN_TILE_DIM;  // d dimension (will become row)
        
        uint global_k = k_block + row;
        uint global_d = d_block + col;
        
        float val = 0.0f;
        if (global_k < n_samples && global_d < hidden_dim) {
            val = bf16_bits_to_float(X[global_k * hidden_dim + global_d]);
        }
        // Store transposed: X_buf_T[d][k] = X[k][d]
        X_buf_T[col][row] = val;
    }
}

/// Load X tile for FP16 input (normal layout)
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

/// Load X tile TRANSPOSED for FP16 input
inline void load_X_tile_fp16_transposed(
    device const half* X,
    threadgroup float (&X_buf_T)[HESSIAN_TILE_DIM][HESSIAN_TILE_K],
    uint n_samples, uint hidden_dim,
    uint k_block, uint d_block,
    uint thread_idx
) {
    const uint elems_per_thread = (HESSIAN_TILE_K * HESSIAN_TILE_DIM) / HESSIAN_THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / HESSIAN_TILE_DIM;  // k dimension
        uint col = flat_idx % HESSIAN_TILE_DIM;  // d dimension
        
        uint global_k = k_block + row;
        uint global_d = d_block + col;
        
        float val = 0.0f;
        if (global_k < n_samples && global_d < hidden_dim) {
            val = float(X[global_k * hidden_dim + global_d]);
        }
        // Store transposed
        X_buf_T[col][row] = val;
    }
}

// ---------------------------------------------------------------------------
// Simdgroup compute for Hessian: H = X^T @ X
//
// X_left_T is stored TRANSPOSED as [TILE_DIM, TILE_K] (i.e., X^T layout)
// X_right is stored NORMAL as [TILE_K, TILE_DIM]
//
// For output tile H[d_i:d_i+64, d_j:d_j+64]:
//   We compute X_left_T @ X_right = [TILE_DIM, TILE_K] @ [TILE_K, TILE_DIM]
//   = [64, K] @ [K, 64] = [64, 64]
//
// simdgroup_multiply_accumulate(C, A, B, C):
//   C += A @ B where A is loaded with stride and B with stride
//   For 8x8 tiles: C[8,8] += A[8,K] @ B[K,8]
// ---------------------------------------------------------------------------

__attribute__((always_inline))
inline void hessian_compute_from_tiles_transpose(
    threadgroup const float (&X_left_T)[HESSIAN_TILE_DIM][HESSIAN_TILE_K],  // [D, K] = X^T
    threadgroup const float (&X_right)[HESSIAN_TILE_K][HESSIAN_TILE_DIM],   // [K, D] = X
    thread simdgroup_matrix<float, 8, 8> (&acc)[SG_TILES_PER_DIM][SG_TILES_PER_DIM],
    uint sg_row_offset,
    uint sg_col_offset
) {
    // For each 8-wide K block (HESSIAN_K_TILES = TILE_K / 8)
    for (uint kt = 0; kt < HESSIAN_K_TILES; ++kt) {
        // For each row-block mi in [0, SG_TILES_PER_DIM)
        for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
            // Load A from X_left_T: 8 rows of D, 8 cols of K
            // A[d, k] = X_left_T[sg_row_offset + mi*8 + d][kt*8 + k]
            simdgroup_matrix<float, 8, 8> a_frag;
            simdgroup_load(a_frag,
                           &X_left_T[sg_row_offset + mi * 8][kt * 8],
                           HESSIAN_TILE_K);  // stride = TILE_K (columns of X^T)
            
            for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
                // Load B from X_right: 8 rows of K, 8 cols of D  
                // B[k, d] = X_right[kt*8 + k][sg_col_offset + ni*8 + d]
                simdgroup_matrix<float, 8, 8> b_frag;
                simdgroup_load(b_frag,
                               &X_right[kt * 8][sg_col_offset + ni * 8],
                               HESSIAN_TILE_DIM);  // stride = TILE_DIM
                
                // acc[mi][ni] += A @ B = [8, 8] @ [8, 8] = [8, 8]
                // This computes: sum_k X^T[d_i, k] * X[k, d_j] = X^T @ X
                simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
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
    // X_left_T: transposed layout [TILE_DIM, TILE_K] for left operand (X^T)
    // X_right: normal layout [TILE_K, TILE_DIM] for right operand (X)
    threadgroup float X_left_T[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_DIM][HESSIAN_TILE_K];
    threadgroup float X_right[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    // Per-simdgroup staging (4 simdgroups, each needs 8x8)
    threadgroup float staging[HESSIAN_SIMDGROUPS_PER_TG][8][8];
    
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
    
    // For diagonal tiles (d_i == d_j), X_left_T and X_right have same source
    const bool is_diagonal = (d_i == d_j);
    
    uint buf_compute = 0;
    
    // Prologue: Load first K-tile
    // Left operand: load TRANSPOSED into X_left_T[D][K]
    load_X_tile_bf16_transposed(X, X_left_T[0], n_samples, hidden_dim, 0, d_i, thread_idx);
    // Right operand: load NORMAL into X_right[K][D]
    load_X_tile_bf16(X, X_right[0], n_samples, hidden_dim, 0, is_diagonal ? d_i : d_j, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Main K-reduction loop
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * HESSIAN_TILE_K;
        uint next_k = k_offset + HESSIAN_TILE_K;
        uint buf_load = 1 - buf_compute;
        
        // Prefetch next tile
        if (next_k < n_samples) {
            load_X_tile_bf16_transposed(X, X_left_T[buf_load], n_samples, hidden_dim, next_k, d_i, thread_idx);
            load_X_tile_bf16(X, X_right[buf_load], n_samples, hidden_dim, next_k, is_diagonal ? d_i : d_j, thread_idx);
        }
        
        // Compute: acc += X_left_T @ X_right = X^T @ X
        hessian_compute_from_tiles_transpose(X_left_T[buf_compute], X_right[buf_compute],
                                             acc, sg_row_offset, sg_col_offset);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store results with multiply by 2: H = 2 * X^T @ X
    // Each simdgroup stores its own tiles using its own staging memory
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        uint out_row = d_i + sg_row_offset + mi * 8;
        if (out_row >= hidden_dim) continue;
        
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            uint out_col = d_j + sg_col_offset + ni * 8;
            if (out_col >= hidden_dim) continue;
            
            // Store to this simdgroup's private staging area
            simdgroup_store(acc[mi][ni], &staging[simd_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Read and write to global memory with 2x scaling
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < hidden_dim && gc < hidden_dim) {
                    H[gr * hidden_dim + gc] = staging[simd_id][r][c] * 2.0f;
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
    // X_left_T: transposed [TILE_DIM, TILE_K], X_right: normal [TILE_K, TILE_DIM]
    threadgroup float X_left_T[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_DIM][HESSIAN_TILE_K];
    threadgroup float X_right[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    // Per-simdgroup staging to avoid race conditions (4 simdgroups)
    threadgroup float staging[4][8][8];
    
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
    
    // Load first K-tile
    load_X_tile_fp16_transposed(X, X_left_T[0], n_samples, hidden_dim, 0, d_i, thread_idx);
    load_X_tile_fp16(X, X_right[0], n_samples, hidden_dim, 0, is_diagonal ? d_i : d_j, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * HESSIAN_TILE_K;
        uint next_k = k_offset + HESSIAN_TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < n_samples) {
            load_X_tile_fp16_transposed(X, X_left_T[buf_load], n_samples, hidden_dim, next_k, d_i, thread_idx);
            load_X_tile_fp16(X, X_right[buf_load], n_samples, hidden_dim, next_k, is_diagonal ? d_i : d_j, thread_idx);
        }
        
        // Compute: acc += X_left_T @ X_right
        hessian_compute_from_tiles_transpose(X_left_T[buf_compute], X_right[buf_compute],
                                             acc, sg_row_offset, sg_col_offset);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Store results with multiply by 2: H = 2 * X^T @ X
    // Each simdgroup uses its own staging memory
    for (uint mi = 0; mi < SG_TILES_PER_DIM; ++mi) {
        uint out_row = d_i + sg_row_offset + mi * 8;
        if (out_row >= hidden_dim) continue;
        
        for (uint ni = 0; ni < SG_TILES_PER_DIM; ++ni) {
            uint out_col = d_j + sg_col_offset + ni * 8;
            if (out_col >= hidden_dim) continue;
            
            simdgroup_store(acc[mi][ni], &staging[simd_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Write to global memory with 2x scaling
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < hidden_dim && gc < hidden_dim) {
                    H[gr * hidden_dim + gc] = staging[simd_id][r][c] * 2.0f;
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
    // X_left_T: transposed [TILE_DIM, TILE_K], X_right: normal [TILE_K, TILE_DIM]
    threadgroup float X_left_T[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_DIM][HESSIAN_TILE_K];
    threadgroup float X_right[HESSIAN_NUM_BUFFERS][HESSIAN_TILE_K][HESSIAN_TILE_DIM];
    // Per-simdgroup staging to avoid race conditions (4 simdgroups)
    threadgroup float staging[4][8][8];
    
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
    
    load_X_tile_bf16_transposed(X, X_left_T[0], n_samples, hidden_dim, 0, d_i, thread_idx);
    load_X_tile_bf16(X, X_right[0], n_samples, hidden_dim, 0, is_diagonal ? d_i : d_j, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_offset = kt * HESSIAN_TILE_K;
        uint next_k = k_offset + HESSIAN_TILE_K;
        uint buf_load = 1 - buf_compute;
        
        if (next_k < n_samples) {
            load_X_tile_bf16_transposed(X, X_left_T[buf_load], n_samples, hidden_dim, next_k, d_i, thread_idx);
            load_X_tile_bf16(X, X_right[buf_load], n_samples, hidden_dim, next_k, is_diagonal ? d_i : d_j, thread_idx);
        }
        
        hessian_compute_from_tiles_transpose(X_left_T[buf_compute], X_right[buf_compute],
                                             acc, sg_row_offset, sg_col_offset);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_compute = buf_load;
    }
    
    // Load existing Hessian and accumulate: H += 2 * X^T @ X
    // Each simdgroup uses its own staging memory
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
                // Bounds-checked load using per-simdgroup staging
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    staging[simd_id][r][c] = (gr < hidden_dim && gc < hidden_dim) ? 
                                              H[gr * hidden_dim + gc] : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(existing, &staging[simd_id][0][0], 8);
            }
            
            // Store acc to staging, scale by 2, write back accumulated
            simdgroup_store(acc[mi][ni], &staging[simd_id][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            // Load existing into thread locals, compute accumulated value, write
            for (uint elem = simd_lane; elem < 64; elem += 32) {
                uint r = elem / 8;
                uint c = elem % 8;
                uint gr = out_row + r;
                uint gc = out_col + c;
                
                if (gr < hidden_dim && gc < hidden_dim) {
                    float existing_val = H[gr * hidden_dim + gc];
                    float new_contrib = staging[simd_id][r][c] * 2.0f;
                    H[gr * hidden_dim + gc] = existing_val + new_contrib;
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
    
    // Per-simdgroup staging to avoid race conditions (4 simdgroups)
    threadgroup float staging[4][8][8];
    
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
                simdgroup_store(tile, &staging[simd_id][0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    staging[simd_id][r][c] *= inv_n;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                simdgroup_load(tile, &staging[simd_id][0][0], 8);
                simdgroup_store(tile, H + out_row * hidden_dim + out_col, hidden_dim);
            } else {
                // Slow path: bounds checking
                
                // Load
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    staging[simd_id][r][c] = (gr < hidden_dim && gc < hidden_dim) ? 
                                    H[gr * hidden_dim + gc] : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Scale
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    staging[simd_id][r][c] *= inv_n;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                // Store
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < hidden_dim && gc < hidden_dim) {
                        H[gr * hidden_dim + gc] = staging[simd_id][r][c];
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
