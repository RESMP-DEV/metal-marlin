#!/usr/bin/env python3
"""Test if simdgroup_load can read different columns from threadgroup memory."""

from __future__ import annotations

import numpy as np
import torch

from metal_marlin.metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)

# Test: Load 4 different 8x8 tiles from B_buf[32][64]
# Each tile should contain unique values
_TEST_KERNEL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_K = 32;
constant constexpr uint TILE_N = 64;

kernel void test_simdgroup_load_columns(
    device half* out [[buffer(0)]],  // Output: 4 * 64 = 256 FP16 values
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    // Create B_buf with known values: B_buf[k][n] = n (column index)
    threadgroup half B_buf[TILE_K][TILE_N];
    
    // Fill B_buf cooperatively
    for (uint i = thread_idx; i < TILE_K * TILE_N; i += 128) {
        uint k = i / TILE_N;
        uint n = i % TILE_N;
        B_buf[k][n] = half(n);  // Value = column index
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Only simd 0 does the test
    if (thread_idx >= 32) return;
    
    // Load 4 tiles at different column offsets and store them
    simdgroup_matrix<half, 8, 8> b_frag_0, b_frag_1, b_frag_2, b_frag_3;
    
    // Load from cols 0-7
    simdgroup_load(b_frag_0, &B_buf[0][0], TILE_N);
    
    // Load from cols 8-15
    simdgroup_load(b_frag_1, &B_buf[0][8], TILE_N);
    
    // Load from cols 16-23
    simdgroup_load(b_frag_2, &B_buf[0][16], TILE_N);
    
    // Load from cols 24-31
    simdgroup_load(b_frag_3, &B_buf[0][24], TILE_N);
    
    // Store results
    simdgroup_store(b_frag_0, out + 0, 8);
    simdgroup_store(b_frag_1, out + 64, 8);
    simdgroup_store(b_frag_2, out + 128, 8);
    simdgroup_store(b_frag_3, out + 192, 8);
}
"""

lib = MetalKernelLibrary()
lib.compile_source("test_load", _TEST_KERNEL)

out_mps = torch.zeros(256, dtype=torch.float16, device="mps")
out_buf = mps_tensor_to_metal_buffer(out_mps, lib.device, copy_back=True)

dispatch_kernel(
    lib,
    function_name="test_simdgroup_load_columns",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[out_buf],
    wait=True,
)

result = out_mps.cpu().numpy()

print("Testing simdgroup_load with different column offsets...")
print("B_buf[k][n] = n (column index)")
print()

for i, col_offset in enumerate([0, 8, 16, 24]):
    start = i * 64
    tile = result[start : start + 64].reshape(8, 8)
    # First column of each tile should be the column offset
    first_col = tile[:, 0]
    expected_first_col = col_offset

    print(f"Tile {i} (cols {col_offset}-{col_offset + 7}):")
    print(f"  First column values: {first_col}")
    print(f"  Expected first col:  [{expected_first_col}]*8")

    if np.allclose(first_col, expected_first_col):
        print("  PASS: simdgroup_load read correct columns")
    else:
        print("  FAIL: simdgroup_load read wrong columns!")
    print()
