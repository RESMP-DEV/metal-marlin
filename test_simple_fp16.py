#!/usr/bin/env python3
"""Simple FP16 GEMM test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from metal_marlin.metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)


def _make_scalar_buffer(lib: MetalKernelLibrary, value: int) -> object:
    """Create a Metal buffer containing a single uint32."""
    data = np.array([value], dtype=np.uint32)
    return lib.device.newBufferWithBytes_length_options_(data.tobytes(), 4, 0)


# Compile shader
lib = MetalKernelLibrary()
source = Path("src/marlin_gemm.metal").read_text()
lib.compile_source("marlin_gemm", source)

# Test: 8x16 A, 16x32 B = 8x32 C
M, K, N = 8, 16, 32

# Simple test: A = all ones, B = column index
A = np.ones((M, K), dtype=np.float16)
B = np.zeros((K, N), dtype=np.float16)
for n in range(N):
    B[:, n] = n  # Column n has all values = n

# Expected: C[i, n] = K * n
expected = A.astype(np.float32) @ B.astype(np.float32)
print(f"Expected row 0: {expected[0, :].astype(np.float16)}")

A_mps = torch.from_numpy(A).to("mps")
B_mps = torch.from_numpy(B).to("mps")
C_mps = torch.zeros(M, N, dtype=torch.float16, device="mps")

A_buf = mps_tensor_to_metal_buffer(A_mps, lib.device)
B_buf = mps_tensor_to_metal_buffer(B_mps, lib.device)
C_buf = mps_tensor_to_metal_buffer(C_mps, lib.device, copy_back=True)

# Create scalar buffers for M, N, K
M_buf = _make_scalar_buffer(lib, M)
N_buf = _make_scalar_buffer(lib, N)
K_buf = _make_scalar_buffer(lib, K)

dispatch_kernel(
    lib,
    function_name="marlin_gemm_fp16_single_stage",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[A_buf, B_buf, C_buf, M_buf, N_buf, K_buf],
    wait=True,
)

result = C_mps.cpu().numpy()
print(f"Actual row 0:   {result[0, :]}")
print()
print(f"Cols 0-7:   {result[0, :8]}")
print(f"Cols 8-15:  {result[0, 8:16]}")
print(f"Cols 16-23: {result[0, 16:24]}")
print(f"Cols 24-31: {result[0, 24:32]}")

# Check for column repetition
if np.allclose(result[0, :8], result[0, 8:16], rtol=1e-3):
    print("\n❌ BUG: Columns 0-7 repeat in 8-15!")
else:
    print("\n✅ Columns 0-7 and 8-15 are different")
