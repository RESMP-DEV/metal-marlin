#!/usr/bin/env python3
"""Minimal simdgroup_load layout test.

Run from contrib/metal_marlin:
  uv run python test_simdgroup_load.py
"""

from __future__ import annotations

import numpy as np
import torch

from metal_marlin.metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_metal,
    require_mps,
)

_METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void test_simdgroup_load(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup half buf[8][8];

    // Fill threadgroup tile from input.
    for (uint i = simd_lane; i < 64; i += 32) {
        buf[i / 8][i % 8] = input[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half, 8, 8> mat;
    simdgroup_load(mat, &buf[0][0], 8);
    simdgroup_store(mat, output, 8);
}
"""


def _format_matrix(values: np.ndarray) -> str:
    rows = [" ".join(f"{v:5.0f}" for v in row) for row in values]
    return "\n".join(rows)


def main() -> None:
    require_metal()
    require_mps()

    lib = MetalKernelLibrary()
    lib.compile_source("test_simdgroup_load", _METAL_SOURCE)

    input_host = np.arange(64, dtype=np.float16)
    input_mps = torch.from_numpy(input_host).to("mps")
    output_mps = torch.zeros(64, dtype=torch.float16, device="mps")

    input_buf = mps_tensor_to_metal_buffer(input_mps, lib.device)
    output_buf = mps_tensor_to_metal_buffer(output_mps, lib.device, copy_back=True)

    dispatch_kernel(
        lib,
        function_name="test_simdgroup_load",
        grid=(1, 1, 1),
        threadgroup=(32, 1, 1),
        buffers=[input_buf, output_buf],
        wait=True,
    )

    output = output_mps.cpu().numpy().astype(np.float32)
    output_mat = output.reshape(8, 8)
    input_mat = input_host.reshape(8, 8).astype(np.float32)

    print("Input 8x8:")
    print(_format_matrix(input_mat))
    print("\nOutput 8x8:")
    print(_format_matrix(output_mat))

    if np.allclose(output_mat, input_mat):
        print("\nResult: simdgroup_load/store appears ROW-MAJOR with stride=8.")
        return

    if np.allclose(output_mat, input_mat.T):
        print("\nResult: simdgroup_load/store appears COLUMN-MAJOR (transpose) with stride=8.")
        return

    diff = np.abs(output_mat - input_mat)
    max_diff = float(diff.max())
    print(f"\nResult: unexpected layout (max abs diff vs row-major = {max_diff:.1f}).")


if __name__ == "__main__":
    main()
