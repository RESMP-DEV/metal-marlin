#!/usr/bin/env python3
"""Debug: dump B_buf contents after load_B_tile_fp16."""

from __future__ import annotations

import numpy as np
import torch

from metal_marlin.metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)

# Modified kernel that dumps B_buf after loading
_DEBUG_KERNEL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_K = 32;
constant constexpr uint TILE_N = 64;
constant constexpr uint THREADS_PER_TG = 128;

inline uint div_ceil(uint a, uint b) { return (a + b - 1) / b; }

inline void load_B_tile_fp16(
    device const half* B,
    threadgroup half (&B_buf)[TILE_K][TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint thread_idx
) {
    const uint elems_per_thread = (TILE_K * TILE_N) / THREADS_PER_TG;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint flat_idx = thread_idx * elems_per_thread + i;
        uint row = flat_idx / TILE_N;
        uint col = flat_idx % TILE_N;
        uint global_row = k_block + row;
        uint global_col = tg_col + col;

        half val = 0.0h;
        if (global_row < K && global_col < N) {
            val = B[global_row * N + global_col];
        }
        B_buf[row][col] = val;
    }
}

kernel void debug_b_buf(
    device const half* B [[buffer(0)]],
    device half* B_buf_out [[buffer(1)]],  // [TILE_K * TILE_N]
    constant uint& K [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup half B_buf[TILE_K][TILE_N];
    
    load_B_tile_fp16(B, B_buf, K, N, 0, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Copy B_buf to output
    for (uint i = thread_idx; i < TILE_K * TILE_N; i += THREADS_PER_TG) {
        uint k = i / TILE_N;
        uint n = i % TILE_N;
        B_buf_out[k * TILE_N + n] = B_buf[k][n];
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("debug", _DEBUG_KERNEL)

# Test dimensions matching test_simple_fp16.py
K, N = 16, 32

# B[k][n] = n (column index)
B = np.zeros((K, N), dtype=np.float16)
for n in range(N):
    B[:, n] = n
print(f"B[0, :8] = {B[0, :8]}")
print(f"B[0, 8:16] = {B[0, 8:16]}")

B_mps = torch.from_numpy(B).to("mps")
B_buf_out_mps = torch.zeros(32 * 64, dtype=torch.float16, device="mps")

B_buf = mps_tensor_to_metal_buffer(B_mps, lib.device)
B_buf_out_buf = mps_tensor_to_metal_buffer(B_buf_out_mps, lib.device, copy_back=True)

# Create scalar buffers
K_np = np.array([K], dtype=np.uint32)
N_np = np.array([N], dtype=np.uint32)
K_buf = lib.device.newBufferWithBytes_length_options_(K_np.tobytes(), 4, 0)
N_buf = lib.device.newBufferWithBytes_length_options_(N_np.tobytes(), 4, 0)

dispatch_kernel(
    lib,
    function_name="debug_b_buf",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[B_buf, B_buf_out_buf, K_buf, N_buf],
    wait=True,
)

B_buf_result = B_buf_out_mps.cpu().numpy().reshape(32, 64)

print()
print("B_buf after load_B_tile_fp16:")
print(f"B_buf[0, :8] = {B_buf_result[0, :8]}")
print(f"B_buf[0, 8:16] = {B_buf_result[0, 8:16]}")
print(f"B_buf[0, 16:24] = {B_buf_result[0, 16:24]}")
print(f"B_buf[0, 24:32] = {B_buf_result[0, 24:32]}")
print(f"B_buf[0, 32:40] = {B_buf_result[0, 32:40]} (should be 0)")
print(f"B_buf[16, 0] = {B_buf_result[16, 0]} (should be 0, k >= K)")
