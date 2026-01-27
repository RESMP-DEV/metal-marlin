#!/usr/bin/env python3
"""Test simdgroup MMA in a loop - closest to the real kernel pattern."""

from __future__ import annotations

import numpy as np
import torch

from metal_marlin.metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)

_TEST_KERNEL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_K = 32;
constant constexpr uint TILE_N = 64;
constant constexpr uint SG_M_TILES = 2;
constant constexpr uint SG_N_TILES = 4;

kernel void test_mma_in_loop(
    device half* out [[buffer(0)]],  // 2*4*64 = 512
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup half A_buf[TILE_M][TILE_K];
    threadgroup half B_buf[TILE_K][TILE_N];
    
    // Fill A_buf: A[m][k] = 1 (all ones)
    for (uint i = thread_idx; i < TILE_M * TILE_K; i += 128) {
        uint m = i / TILE_K;
        uint k = i % TILE_K;
        A_buf[m][k] = 1.0h;
    }
    
    // Fill B_buf: B[k][n] = n (column index)
    for (uint i = thread_idx; i < TILE_K * TILE_N; i += 128) {
        uint k = i / TILE_N;
        uint n = i % TILE_N;
        B_buf[k][n] = half(n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (thread_idx >= 32) return;
    
    // Initialize accumulators
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }
    
    // Compute - accumulate k=0..7 (one 8x8 tile of K)
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        simdgroup_matrix<half, 8, 8> a_frag;
        simdgroup_load(a_frag, &A_buf[mi * 8][0], TILE_K);
        
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            simdgroup_matrix<half, 8, 8> b_frag;
            simdgroup_load(b_frag, &B_buf[0][ni * 8], TILE_N);
            
            simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
        }
    }
    
    // Expected result: C[m][n] = K * n = 8 * n (since A=1, B=n, and we summed 8 k values)
    // So for each 8x8 tile at column ni*8:
    //   First column of tile = ni * 8 * 8 = ni * 64
    //   But actually, each lane in the 8x8 block has different n, so:
    //   tile[r][c] = sum_k(1 * (ni*8 + c)) = 8 * (ni*8 + c)
    
    // Store results
    uint offset = 0;
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            simdgroup_store(acc[mi][ni], out + offset, 8);
            offset += 64;
        }
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("test_mma", _TEST_KERNEL)

out_mps = torch.zeros(512, dtype=torch.float16, device="mps")
out_buf = mps_tensor_to_metal_buffer(out_mps, lib.device, copy_back=True)

dispatch_kernel(
    lib,
    function_name="test_mma_in_loop",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[out_buf],
    wait=True,
)

result = out_mps.cpu().numpy()

print("Testing simdgroup MMA in loop (like compute_from_tiles)...")
print("A = ones, B = column index, C = A @ B")
print()

for mi in range(2):
    for ni in range(4):
        idx = mi * 4 + ni
        start = idx * 64
        tile = result[start : start + 64].reshape(8, 8)

        # Expected: C[r][c] = 8 * (ni*8 + c) since sum_k(1 * (ni*8+c)) for k=0..7
        expected_first_col = 8 * (ni * 8)  # c=0
        actual_first_col = tile[:, 0]

        print(f"acc[{mi}][{ni}] (cols {ni * 8}-{ni * 8 + 7}):")
        print(f"  First column: {actual_first_col}")
        print(f"  Expected:     [{expected_first_col}]*8")

        if np.allclose(actual_first_col, expected_first_col):
            print("  PASS")
        else:
            print("  FAIL - got wrong values!")
        print()
