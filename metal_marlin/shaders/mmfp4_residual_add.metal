#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// -----------------------------------------------------------------------------
// MMFP4 Optimized Residual Add Kernel
// -----------------------------------------------------------------------------
//
// This kernel performs element-wise addition of two matrices: C = A + B.
//
// Optimization Strategy:
// 1. **AMX Utilization**: We leverage the Apple AMX coprocessor (simdgroup_matrix)
//    to perform the addition. This is done by treating the operation as a 
//    Matrix Multiply-Accumulate (MMA): C = Identity * A + B.
//    This offloads the addition to the matrix units, potentially increasing throughput
//    for large matrices and allowing ALU operations to proceed in parallel if needed.
//
// 2. **Threadgroup Tiling & Padding**:
//    - The kernel operates on 32x32 tiles of data per threadgroup.
//    - Data is loaded from Global Memory -> Threadgroup Memory (Shared Memory).
//    - We add PADDING to the threadgroup memory allocation (e.g., [32][32+8])
//      to minimize bank conflicts when threads access columns or when the 
//      AMX unit loads/stores 8x8 blocks.
//
// 3. **Coalesced Global Access**:
//    - Threads collaborate to load data in 128-bit vectors (ulong2 = 8 halves) 
//      where possible, ensuring maximum global memory bandwidth efficiency.
//
// 4. **Identity Matrix Generation**:
//    - An 8x8 Identity matrix is constructed in threadgroup memory once and 
//      loaded into the AMX registers.
//
// Buffer Layout:
// - Buffer 0: Input A (half*)
// - Buffer 1: Input B (half*)
// - Buffer 2: Output C (half*)
// - Buffer 3: M (uint*) - Rows
// - Buffer 4: N (uint*) - Columns
// -----------------------------------------------------------------------------

constant uint TILE_DIM = 32;       // Threadgroup tile size (32x32)
constant uint SUB_TILE_DIM = 8;    // AMX tile size (8x8)
constant uint PAD = 8;             // Padding to avoid bank conflicts (32 + 8 = 40)
constant uint TG_STRIDE = TILE_DIM + PAD;

// Helper to load 32x32 tile from global to shared with padding
inline void load_tile_32x32(
    device const half* src,
    threadgroup half (*dest)[TG_STRIDE],
    uint rows,
    uint cols,
    uint tile_row_off,
    uint tile_col_off,
    uint tid
) {
    // Total elements = 32 * 32 = 1024
    // Threads per TG = 32 (assuming 1 simdgroup per TG)
    
    // Each thread handles 1 row of the 32x32 tile (32 elements).
    // We use vectorized loads (ulong2 = 8 halves).
    // 32 elements / 8 elements per vector = 4 vectors per thread.
    
    uint row_in_tile = tid; // 0..31
    uint global_r = tile_row_off + row_in_tile;
    
    if (global_r < rows) {
        // Load the row
        for (uint c = 0; c < TILE_DIM; c += 8) {
            uint global_c = tile_col_off + c;
            
            if (global_c + 8 <= cols) {
                // Vectorized Load
                device const ulong2* src_ptr = (device const ulong2*)(src + global_r * cols + global_c);
                threadgroup ulong2* dst_ptr = (threadgroup ulong2*)(&dest[row_in_tile][c]);
                *dst_ptr = *src_ptr;
            } else {
                // Scalar Fallback for boundary
                for (uint i = 0; i < 8; ++i) {
                    if (global_c + i < cols) {
                        dest[row_in_tile][c + i] = src[global_r * cols + global_c + i];
                    } else {
                        dest[row_in_tile][c + i] = 0.0h;
                    }
                }
            }
        }
    } else {
        // Zero out padding/out-of-bounds rows in shared mem
        for (uint c = 0; c < TILE_DIM; ++c) {
            dest[row_in_tile][c] = 0.0h;
        }
    }
}

// Helper to store 32x32 tile from shared to global
inline void store_tile_32x32(
    device half* dst,
    threadgroup const half (*src)[TG_STRIDE],
    uint rows,
    uint cols,
    uint tile_row_off,
    uint tile_col_off,
    uint tid
) {
    uint row_in_tile = tid; // 0..31
    uint global_r = tile_row_off + row_in_tile;
    
    if (global_r < rows) {
        for (uint c = 0; c < TILE_DIM; c += 8) {
            uint global_c = tile_col_off + c;
            
            if (global_c + 8 <= cols) {
                // Vectorized Store
                device ulong2* dst_ptr = (device ulong2*)(dst + global_r * cols + global_c);
                threadgroup const ulong2* src_ptr = (threadgroup const ulong2*)(&src[row_in_tile][c]);
                *dst_ptr = *src_ptr;
            } else {
                // Scalar Fallback
                for (uint i = 0; i < 8; ++i) {
                    if (global_c + i < cols) {
                        dst[global_r * cols + global_c + i] = src[row_in_tile][c + i];
                    }
                }
            }
        }
    }
}

kernel void residual_add(
    device const half* inputA [[buffer(0)]],
    device const half* inputB [[buffer(1)]],
    device half* output   [[buffer(2)]],
    device const uint* M_p [[buffer(3)]],
    device const uint* N_p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint M = *M_p;
    const uint N = *N_p;
    
    // Allocate Threadgroup Memory with Padding
    threadgroup half tile_A[TILE_DIM][TG_STRIDE];
    threadgroup half tile_B[TILE_DIM][TG_STRIDE];
    threadgroup half tile_C[TILE_DIM][TG_STRIDE];
    
    uint tile_row_off = tgid.y * TILE_DIM;
    uint tile_col_off = tgid.x * TILE_DIM;
    
    // 1. Load Data (Coalesced Global -> Shared)
    load_tile_32x32(inputA, tile_A, M, N, tile_row_off, tile_col_off, tid);
    load_tile_32x32(inputB, tile_B, M, N, tile_row_off, tile_col_off, tid);
    
    // Ensure all loads are done
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Prepare Identity Matrix (Collaborative Construction)
    // We use a small 8x8 shared buffer for the Identity tile.
    threadgroup half identity_8x8[64];
    
    // With 32 threads, each thread must fill 2 elements of the 64-element matrix.
    for (uint i = 0; i < 2; ++i) {
        uint idx = tid + i * 32;
        if (idx < 64) {
            uint r = idx / 8;
            uint c = idx % 8;
            identity_8x8[idx] = (r == c) ? 1.0h : 0.0h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    simdgroup_matrix<half, 8, 8> I;
    simdgroup_load(I, identity_8x8, 8);
    
    // 3. Process 32x32 Tile in 8x8 Sub-tiles using AMX
    // A single simdgroup (32 threads) can iterate over the 16 sub-tiles (4x4).
    
    // Unroll loops for better performance
    #pragma unroll(4)
    for (uint r = 0; r < TILE_DIM; r += 8) {
        #pragma unroll(4)
        for (uint c = 0; c < TILE_DIM; c += 8) {
            simdgroup_matrix<half, 8, 8> matA;
            simdgroup_matrix<half, 8, 8> matB;
            simdgroup_matrix<half, 8, 8> matC;
            
            // Load from Shared Memory (Strided)
            // Address of sub-tile: &tile[r][c]
            simdgroup_load(matA, &tile_A[r][c], TG_STRIDE);
            simdgroup_load(matB, &tile_B[r][c], TG_STRIDE);
            
            // C = I * A + B
            // The multiply_accumulate computes Result = A*B + C_in.
            // We want Res = 1*A + B.
            // So: multiply_accumulate(Result, I, A, B)
            simdgroup_multiply_accumulate(matC, I, matA, matB);
            
            // Store to Shared Memory
            simdgroup_store(matC, &tile_C[r][c], TG_STRIDE);
        }
    }
    
    // Wait for AMX stores to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 4. Store Data (Coalesced Shared -> Global)
    store_tile_32x32(output, tile_C, M, N, tile_row_off, tile_col_off, tid);
}