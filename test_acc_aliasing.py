#!/usr/bin/env python3
"""Test if simdgroup_matrix 2D array elements alias each other."""

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

constant constexpr uint SG_M_TILES = 2;
constant constexpr uint SG_N_TILES = 4;

kernel void test_acc_aliasing(
    device half* out [[buffer(0)]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];

    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            half val = half(mi * 10 + ni);
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(val);
        }
    }

    uint base_offset = 0;
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            simdgroup_store(acc[mi][ni], out + base_offset, 8);
            base_offset += 64;
        }
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("test_acc", _TEST_KERNEL)

out_mps = torch.zeros(512, dtype=torch.float16, device="mps")
out_buf = mps_tensor_to_metal_buffer(out_mps, lib.device, copy_back=True)

dispatch_kernel(
    lib,
    function_name="test_acc_aliasing",
    grid=(1, 1, 1),
    threadgroup=(32, 1, 1),
    buffers=[out_buf],
    wait=True,
)

result = out_mps.cpu().numpy()

print("Testing 2D simdgroup_matrix array aliasing...")
print()
for mi in range(2):
    for ni in range(4):
        expected_val = mi * 10 + ni
        start = (mi * 4 + ni) * 64
        actual_vals = result[start : start + 64]
        unique_vals = np.unique(actual_vals)
        if len(unique_vals) == 1 and unique_vals[0] == expected_val:
            print(f"acc[{mi}][{ni}]: All 64 elements = {expected_val}")
        else:
            print(f"acc[{mi}][{ni}]: Expected {expected_val}, got {unique_vals}")
