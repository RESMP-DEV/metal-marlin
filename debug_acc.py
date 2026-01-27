#!/usr/bin/env python3
"""Debug: dump acc[] contents BEFORE store_results."""

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

inline uint div_ceil(uint a, uint b) { return (a + b - 1) / b; }

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

kernel void debug_acc_contents(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* acc_out [[buffer(2)]],  // [SG_M_TILES * SG_N_TILES * 64]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup half A_buf[TILE_M][TILE_K];
    threadgroup half B_buf[TILE_K][TILE_N];
    
    // Only look at simd 0
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
    
    // Compute - mimicking compute_from_tiles with the unrolled loop
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
    
    // Dump acc from simd 0 only
    if (simd_id != 0) return;
    
    uint offset = 0;
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            simdgroup_store(acc[mi][ni], acc_out + offset, 8);
            offset += 64;
        }
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("debug_acc", _DEBUG_KERNEL)

M, K, N = 8, 16, 32

# A = all ones
A = np.ones((M, K), dtype=np.float16)

# B = column index
B = np.zeros((K, N), dtype=np.float16)
for n in range(N):
    B[:, n] = n

A_mps = torch.from_numpy(A).to("mps")
B_mps = torch.from_numpy(B).to("mps")
acc_out_mps = torch.zeros(8 * 64, dtype=torch.float16, device="mps")  # 2*4*64

A_buf = mps_tensor_to_metal_buffer(A_mps, lib.device)
B_buf = mps_tensor_to_metal_buffer(B_mps, lib.device)
acc_out_buf = mps_tensor_to_metal_buffer(acc_out_mps, lib.device, copy_back=True)


# Scalars
def make_scalar(val):
    return lib.device.newBufferWithBytes_length_options_(
        np.array([val], dtype=np.uint32).tobytes(), 4, 0
    )


dispatch_kernel(
    lib,
    function_name="debug_acc_contents",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[A_buf, B_buf, acc_out_buf, make_scalar(M), make_scalar(N), make_scalar(K)],
    wait=True,
)

acc = acc_out_mps.cpu().numpy()

print("Dumping acc[] contents BEFORE store_results")
print("A = ones, B = column index, expected: C[m][n] = K*n = 16*n")
print()

for mi in range(2):
    for ni in range(4):
        idx = mi * 4 + ni
        start = idx * 64
        tile = acc[start : start + 64].reshape(8, 8)

        # Expected: each element C[r][c] = K * (ni*8 + c) = 16 * (ni*8 + c)
        expected_row = [16 * (ni * 8 + c) for c in range(8)]
        actual_row = tile[0, :]

        print(f"acc[{mi}][{ni}] (covering cols {ni * 8} to {ni * 8 + 7}):")
        print(f"  Row 0: {actual_row}")
        print(f"  Expected: {expected_row}")

        if np.allclose(actual_row, expected_row, atol=1):
            print("  PASS")
        else:
            print("  FAIL - acc has wrong values!")
        print()
