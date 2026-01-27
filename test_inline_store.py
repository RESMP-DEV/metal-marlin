#!/usr/bin/env python3
"""Test storing acc to C inline (no function call)."""

from __future__ import annotations

import numpy as np
import torch

from metal_marlin.metal_dispatch import (
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)

_DEBUG_KERNEL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_K = 32;
constant constexpr uint TILE_N = 64;
constant constexpr uint K_TILES = TILE_K / 8;
constant constexpr uint SG_M_TILES = 2;
constant constexpr uint SG_N_TILES = 4;
constant constexpr uint THREADS_PER_TG = 128;

inline void load_A_tile(
    device const half* A,
    threadgroup half (&A_buf)[TILE_M][TILE_K],
    uint M, uint K,
    uint tg_row, uint k_block,
    uint thread_idx
) {
    const uint elems = (TILE_M * TILE_K) / THREADS_PER_TG;
    for (uint i = 0; i < elems; ++i) {
        uint flat = thread_idx * elems + i;
        uint row = flat / TILE_K;
        uint col = flat % TILE_K;
        uint gr = tg_row + row;
        uint gc = k_block + col;
        A_buf[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0h;
    }
}

inline void load_B_tile_fp16(
    device const half* B,
    threadgroup half (&B_buf)[TILE_K][TILE_N],
    uint K, uint N,
    uint tg_col, uint k_block,
    uint thread_idx
) {
    const uint elems = (TILE_K * TILE_N) / THREADS_PER_TG;
    for (uint i = 0; i < elems; ++i) {
        uint flat = thread_idx * elems + i;
        uint row = flat / TILE_N;
        uint col = flat % TILE_N;
        uint gr = k_block + row;
        uint gc = tg_col + col;
        B_buf[row][col] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0h;
    }
}

kernel void test_inline_store(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup half A_buf[TILE_M][TILE_K];
    threadgroup half B_buf[TILE_K][TILE_N];
    
    // Only use simd 0
    if (simd_id != 0) return;
    
    const uint sg_row_offset = 0;
    const uint sg_col_offset = 0;
    
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }
    
    load_A_tile(A, A_buf, M, K, 0, 0, thread_idx);
    load_B_tile_fp16(B, B_buf, K, N, 0, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute
    for (uint kt = 0; kt < K_TILES; ++kt) {
        for (uint mi = 0; mi < SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag, &A_buf[sg_row_offset + mi * 8][kt * 8], TILE_K);
            
            {
                simdgroup_matrix<half, 8, 8> b_frag_0;
                simdgroup_load(b_frag_0, &B_buf[kt * 8][sg_col_offset + 0], TILE_N);
                simdgroup_multiply_accumulate(acc[mi][0], a_frag, b_frag_0, acc[mi][0]);
            }
            {
                simdgroup_matrix<half, 8, 8> b_frag_1;
                simdgroup_load(b_frag_1, &B_buf[kt * 8][sg_col_offset + 8], TILE_N);
                simdgroup_multiply_accumulate(acc[mi][1], a_frag, b_frag_1, acc[mi][1]);
            }
            {
                simdgroup_matrix<half, 8, 8> b_frag_2;
                simdgroup_load(b_frag_2, &B_buf[kt * 8][sg_col_offset + 16], TILE_N);
                simdgroup_multiply_accumulate(acc[mi][2], a_frag, b_frag_2, acc[mi][2]);
            }
            {
                simdgroup_matrix<half, 8, 8> b_frag_3;
                simdgroup_load(b_frag_3, &B_buf[kt * 8][sg_col_offset + 24], TILE_N);
                simdgroup_multiply_accumulate(acc[mi][3], a_frag, b_frag_3, acc[mi][3]);
            }
        }
    }
    
    // Store INLINE (no function call)
    // This mimics store_results but inline
    uint base_row = 0;
    uint base_col = 0;
    
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = base_row + mi * 8;
        if (out_row >= M) continue;
        
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = base_col + ni * 8;
            if (out_col >= N) continue;
            
            simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
        }
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("test_inline", _DEBUG_KERNEL)

M, K, N = 8, 16, 32

A = np.ones((M, K), dtype=np.float16)
B = np.zeros((K, N), dtype=np.float16)
for n in range(N):
    B[:, n] = n

A_mps = torch.from_numpy(A).to("mps")
B_mps = torch.from_numpy(B).to("mps")
C_mps = torch.zeros(M, N, dtype=torch.float16, device="mps")

A_buf = mps_tensor_to_metal_buffer(A_mps, lib.device)
B_buf = mps_tensor_to_metal_buffer(B_mps, lib.device)
C_buf = mps_tensor_to_metal_buffer(C_mps, lib.device, copy_back=True)


def make_scalar(val):
    return lib.device.newBufferWithBytes_length_options_(
        np.array([val], dtype=np.uint32).tobytes(), 4, 0
    )


dispatch_kernel(
    lib,
    function_name="test_inline_store",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[A_buf, B_buf, C_buf, make_scalar(M), make_scalar(N), make_scalar(K)],
    wait=True,
)

C = C_mps.cpu().numpy()

print("Testing inline store (no function call)...")
print(f"C[0, :8] = {C[0, :8]}  (should be [0, 16, 32, ...])")
print(f"C[0, 8:16] = {C[0, 8:16]}  (should be [128, 144, 160, ...])")
print(f"C[0, 16:24] = {C[0, 16:24]}  (should be [256, 272, ...])")
print(f"C[0, 24:32] = {C[0, 24:32]}  (should be [384, 400, ...])")

expected = np.array([16 * n for n in range(32)], dtype=np.float16)
if np.allclose(C[0, :], expected, atol=1):
    print("\nPASS - Inline store works correctly!")
else:
    print("\nFAIL - Still has the bug")
    print(f"Expected: {expected}")
    print(f"Actual:   {C[0, :]}")
