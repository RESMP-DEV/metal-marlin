#!/usr/bin/env python3
"""Test with actual simdgroup layout (2x2 grid, sg_col_offset from simd_id)."""

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
constant constexpr uint SIMDGROUPS_PER_TG = 4;

kernel void test_layout(
    device half* out [[buffer(0)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup half A_buf[TILE_M][TILE_K];
    threadgroup half B_buf[TILE_K][TILE_N];
    
    // Fill A_buf: A[m][k] = 1 (all ones)
    for (uint i = thread_idx; i < TILE_M * TILE_K; i += 128) {
        A_buf[i / TILE_K][i % TILE_K] = 1.0h;
    }
    
    // Fill B_buf: B[k][n] = n (column index)
    for (uint i = thread_idx; i < TILE_K * TILE_N; i += 128) {
        B_buf[i / TILE_N][i % TILE_N] = half(i % TILE_N);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Same layout as real kernel
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);  // 0 or 16
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);  // 0 or 32
    
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }
    
    // Compute with actual offsets
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        simdgroup_matrix<half, 8, 8> a_frag;
        simdgroup_load(a_frag, &A_buf[sg_row_offset + mi * 8][0], TILE_K);
        
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            simdgroup_matrix<half, 8, 8> b_frag;
            simdgroup_load(b_frag, &B_buf[0][sg_col_offset + ni * 8], TILE_N);
            
            simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
        }
    }
    
    // Only store results from simd 0 for clarity
    // Note: In real kernel, all simdgroups write to different parts of C
    // But for this test, each simdgroup writes to its portion of out buffer
    uint base = simd_id * SG_M_TILES * SG_N_TILES * 64;  // 512 per simdgroup
    uint offset = 0;
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            simdgroup_store(acc[mi][ni], out + base + offset, 8);
            offset += 64;
        }
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("test_layout", _TEST_KERNEL)

# 4 simdgroups * 8 tiles * 64 = 2048
out_mps = torch.zeros(2048, dtype=torch.float16, device="mps")
out_buf = mps_tensor_to_metal_buffer(out_mps, lib.device, copy_back=True)

dispatch_kernel(
    lib,
    function_name="test_layout",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[out_buf],
    wait=True,
)

result = out_mps.cpu().numpy()

print("Testing with actual 2x2 simdgroup layout...")
print("simd 0,1: sg_row=0, simd 2,3: sg_row=16")
print("simd 0,2: sg_col=0, simd 1,3: sg_col=32")
print()

for simd_id in range(4):
    sg_row = (simd_id // 2) * 16
    sg_col = (simd_id % 2) * 32
    base = simd_id * 512

    print(f"=== Simdgroup {simd_id} (sg_row={sg_row}, sg_col={sg_col}) ===")

    for mi in range(2):
        for ni in range(4):
            idx = mi * 4 + ni
            start = base + idx * 64
            tile = result[start : start + 64].reshape(8, 8)

            # Expected: each element = 8 * (sg_col + ni*8 + c)
            expected_first_col = 8 * (sg_col + ni * 8)
            actual_first_col = tile[:, 0]

            status = "PASS" if np.allclose(actual_first_col, expected_first_col) else "FAIL"
            print(
                f"  acc[{mi}][{ni}]: col {sg_col + ni * 8}, first_col={actual_first_col[0]:.0f}, expected={expected_first_col} [{status}]"
            )
    print()
