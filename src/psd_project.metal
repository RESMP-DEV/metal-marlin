// psd_project.metal - Fast PSD projection via iterative diagonal regularization
//
// Instead of full eigendecomposition, we iteratively:
// 1. Attempt Cholesky: H = L @ L^T
// 2. On failure, add diagonal: H_reg = H + lambda * I
// 3. Repeat until Cholesky succeeds
//
// This is faster than eigh for well-conditioned matrices (common in Hessians).

#include <metal_stdlib>
using namespace metal;

constant constexpr uint PSD_TILE = 32;
constant constexpr uint PSD_THREADS = 256;

// In-place Cholesky decomposition (lower triangular)
// Returns true if successful, false if matrix is not PSD
// H is modified in-place to become L
kernel void cholesky_inplace(
    device float* H             [[buffer(0)]],  // [dim, dim] symmetric, becomes L
    device atomic_int* success  [[buffer(1)]],  // 1 if success, 0 if failed
    constant uint& dim          [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint lid                    [[thread_position_in_threadgroup]]
) {
    // Single-threaded Cholesky for now (GPU parallelization is complex)
    // This kernel is dispatched with 1 thread
    if (tid != 0) return;
    
    atomic_store_explicit(success, 1, memory_order_relaxed);
    
    for (uint j = 0; j < dim; ++j) {
        // Compute L[j,j] = sqrt(H[j,j] - sum(L[j,k]^2 for k<j))
        float sum = 0.0f;
        for (uint k = 0; k < j; ++k) {
            float val = H[j * dim + k];
            sum += val * val;
        }
        float diag = H[j * dim + j] - sum;
        
        if (diag <= 0.0f) {
            atomic_store_explicit(success, 0, memory_order_relaxed);
            return;
        }
        
        H[j * dim + j] = sqrt(diag);
        float inv_ljj = 1.0f / H[j * dim + j];
        
        // Compute L[i,j] for i > j
        for (uint i = j + 1; i < dim; ++i) {
            float sum_ij = 0.0f;
            for (uint k = 0; k < j; ++k) {
                sum_ij += H[i * dim + k] * H[j * dim + k];
            }
            H[i * dim + j] = (H[i * dim + j] - sum_ij) * inv_ljj;
        }
        
        // Zero upper triangle
        for (uint i = 0; i < j; ++i) {
            H[i * dim + j] = 0.0f;
        }
    }
}

// Add diagonal regularization: H += lambda * I
kernel void add_diagonal(
    device float* H          [[buffer(0)]],  // [dim, dim]
    constant float& lambda   [[buffer(1)]],
    constant uint& dim       [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    if (tid < dim) {
        H[tid * dim + tid] += lambda;
    }
}

// Reconstruct H_psd from Cholesky: H_psd = L @ L^T
kernel void reconstruct_from_cholesky(
    device const float* L    [[buffer(0)]],  // [dim, dim] lower triangular
    device float* H_psd      [[buffer(1)]],  // [dim, dim] output
    constant uint& dim       [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= dim || col >= dim) return;
    
    // H_psd[row, col] = sum_k L[row, k] * L[col, k] for k <= min(row, col)
    float sum = 0.0f;
    uint max_k = min(row, col) + 1;
    
    for (uint k = 0; k < max_k; ++k) {
        sum += L[row * dim + k] * L[col * dim + k];
    }
    
    H_psd[row * dim + col] = sum;
}

// Copy matrix
kernel void matrix_copy(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& size      [[buffer(2)]],  // total elements
    uint tid                 [[thread_position_in_grid]]
) {
    if (tid < size) {
        dst[tid] = src[tid];
    }
}
