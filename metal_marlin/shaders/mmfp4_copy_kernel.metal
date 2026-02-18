#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// -----------------------------------------------------------------------------
// MMFP4 Copy Kernel (Synchronous Version)
// -----------------------------------------------------------------------------
//
// This kernel implements a synchronous memory copy for weights, compatible
// with older Metal versions that don't support simdgroup_async_copy.
//
// It performs a copy pipeline:
// 1. Device Memory -> Threadgroup Memory (via cooperative copy)
// 2. Threadgroup Memory -> Register File (via simdgroup_load)
// 3. Register File -> Device Memory (via simdgroup_store)
// -----------------------------------------------------------------------------

// Tile dimensions
constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint SIMDGROUP_SIZE = 32;

// Bank conflict avoidance:
constant uint PAD = 8;
constant uint SMEM_STRIDE = TILE_N + PAD;

kernel void mmfp4_copy_kernel(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    device const uint* M_p [[buffer(2)]],
    device const uint* N_p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint M = *M_p;
    const uint N = *N_p;

    // Threadgroup memory for staging
    threadgroup half tile_smem[TILE_M * SMEM_STRIDE];

    // Coordinates
    uint tile_row = tgid.y * TILE_M;
    uint tile_col = tgid.x * TILE_N;

    // Synchronous Copy: Device -> Threadgroup
    // Each thread copies multiple elements
    const uint total_elements = TILE_M * TILE_N;
    const uint num_threads = SIMDGROUP_SIZE;  // 32 threads
    
    if (tile_row < M && tile_col < N) {
        // Cooperative copy: each thread copies multiple elements
        for (uint i = tid; i < total_elements; i += num_threads) {
            uint row = i / TILE_N;
            uint col = i % TILE_N;
            
            uint src_row = tile_row + row;
            uint src_col = tile_col + col;
            
            half val = 0.0h;
            if (src_row < M && src_col < N) {
                val = src[src_row * N + src_col];
            }
            tile_smem[row * SMEM_STRIDE + col] = val;
        }
    }
    
    // Barrier to ensure all threads see the data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Load from Threadgroup -> Registers (Matrix) -> Store to Device
    // We use simdgroup_matrix to load 8x8 blocks from SMEM and store to DST.
    
    // Iterate over the 32x32 tile in 8x8 chunks
    // Each simdgroup can process multiple 8x8 blocks.
    // Since we have 1 simdgroup (32 threads), we can loop.
    
    for (uint row_offset = 0; row_offset < TILE_M; row_offset += 8) {
        for (uint col_offset = 0; col_offset < TILE_N; col_offset += 8) {
            // Check boundaries
            if ((tile_row + row_offset < M) && (tile_col + col_offset < N)) {
                
                simdgroup_matrix<half, 8, 8> mat;
                
                // Load from Threadgroup
                // Address: tile_smem + row_offset * SMEM_STRIDE + col_offset
                threadgroup half* smem_src = tile_smem + row_offset * SMEM_STRIDE + col_offset;
                
                simdgroup_load(
                    mat,
                    smem_src,
                    SMEM_STRIDE,
                    ulong2(TILE_M - row_offset, TILE_N - col_offset), // Max available in smem
                    false // col-major? No, defaults to row-major
                );
                
                // Store to Device (dst)
                device half* dst_ptr = dst + (tile_row + row_offset) * N + (tile_col + col_offset);
                
                simdgroup_store(
                    mat,
                    dst_ptr,
                    N,
                    ulong2(M - (tile_row + row_offset), N - (tile_col + col_offset)),
                    false
                );
            }
        }
    }
}
