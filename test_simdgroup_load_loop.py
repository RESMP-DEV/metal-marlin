#!/usr/bin/env python3
"""Test if simdgroup_load works correctly INSIDE a loop."""

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

constant constexpr uint TILE_K = 32;
constant constexpr uint TILE_N = 64;
constant constexpr uint SG_N_TILES = 4;

kernel void test_simdgroup_load_in_loop(
    device half* out [[buffer(0)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup half B_buf[TILE_K][TILE_N];
    
    // Fill B_buf: B_buf[k][n] = n (column index)
    for (uint i = thread_idx; i < TILE_K * TILE_N; i += 128) {
        uint k = i / TILE_N;
        uint n = i % TILE_N;
        B_buf[k][n] = half(n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (thread_idx >= 32) return;
    
    // This is the problematic pattern - simdgroup_load inside a loop
    simdgroup_matrix<half, 8, 8> acc[SG_N_TILES];
    for (uint i = 0; i < SG_N_TILES; ++i) {
        acc[i] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }
    
    // Load different column tiles in a loop - like compute_from_tiles does
    for (uint ni = 0; ni < SG_N_TILES; ++ni) {
        simdgroup_matrix<half, 8, 8> b_frag;
        simdgroup_load(b_frag, &B_buf[0][ni * 8], TILE_N);
        
        // Store directly to acc to see what was loaded
        acc[ni] = b_frag;
    }
    
    // Store all acc matrices
    for (uint ni = 0; ni < SG_N_TILES; ++ni) {
        simdgroup_store(acc[ni], out + ni * 64, 8);
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("test_loop", _TEST_KERNEL)

out_mps = torch.zeros(256, dtype=torch.float16, device="mps")
out_buf = mps_tensor_to_metal_buffer(out_mps, lib.device, copy_back=True)

dispatch_kernel(
    lib,
    function_name="test_simdgroup_load_in_loop",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[out_buf],
    wait=True,
)

result = out_mps.cpu().numpy()

print("Testing simdgroup_load INSIDE a loop...")
print()

for ni in range(4):
    start = ni * 64
    tile = result[start : start + 64].reshape(8, 8)
    first_col = tile[:, 0]
    expected = ni * 8

    print(f"acc[{ni}] (should be cols {ni * 8}-{ni * 8 + 7}):")
    print(f"  First column: {first_col}")
    print(f"  Expected:     [{expected}]*8")

    if np.allclose(first_col, expected):
        print("  PASS")
    else:
        print("  FAIL - got wrong columns!")
    print()
