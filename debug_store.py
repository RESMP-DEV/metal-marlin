#!/usr/bin/env python3
"""Debug: dump what ACTUALLY gets stored to C."""
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

// This kernel uses EXACTLY the same config as the real kernel:
// 4 simdgroups, 2x2 layout
// We'll print debug info for each simdgroup
kernel void debug_store(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device uint* dbg [[buffer(3)]],  // debug buffer
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half A_buf[TILE_M][TILE_K];
    threadgroup half B_buf[TILE_K][TILE_N];
    
    // Same layout as production kernel
    const uint sg_row_offset = (simd_id / 2) * (SG_M_TILES * 8);  // 0 or 16
    const uint sg_col_offset = (simd_id % 2) * (SG_N_TILES * 8);  // 0 or 32
    
    const uint thread_idx = simd_id * 32 + simd_lane;
    
    simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }
    
    load_A_tile(A, A_buf, M, K, 0, 0, thread_idx);
    load_B_tile_fp16(B, B_buf, K, N, 0, 0, thread_idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // K_TILES = 4 for K=32, but our K=16 so only 2 iterations needed
    const uint k_iters = K / 8;  // 16/8 = 2
    
    // Compute
    for (uint kt = 0; kt < k_iters; ++kt) {
        for (uint mi = 0; mi < SG_M_TILES; ++mi) {
            simdgroup_matrix<half, 8, 8> a_frag;
            simdgroup_load(a_frag, &A_buf[sg_row_offset + mi * 8][kt * 8], TILE_K);
            
            for (uint ni = 0; ni < SG_N_TILES; ++ni) {
                simdgroup_matrix<half, 8, 8> b_frag;
                simdgroup_load(b_frag, &B_buf[kt * 8][sg_col_offset + ni * 8], TILE_N);
                simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
            }
        }
    }
    
    // Log debug info (from thread 0 of each simdgroup)
    if (simd_lane == 0) {
        uint idx = simd_id * 4;
        dbg[idx + 0] = simd_id;
        dbg[idx + 1] = sg_row_offset;
        dbg[idx + 2] = sg_col_offset;
        dbg[idx + 3] = k_iters;
    }
    
    threadgroup half staging[8][8];
    
    // Store with proper bounds (matching production kernel)
    uint base_row = 0 + sg_row_offset;
    uint base_col = 0 + sg_col_offset;
    
    for (uint mi = 0; mi < SG_M_TILES; ++mi) {
        uint out_row = base_row + mi * 8;
        if (out_row >= M) continue;
        
        for (uint ni = 0; ni < SG_N_TILES; ++ni) {
            uint out_col = base_col + ni * 8;
            if (out_col >= N) continue;
            
            if (out_row + 8 <= M && out_col + 8 <= N) {
                // Fast path
                simdgroup_store(acc[mi][ni], C + out_row * N + out_col, N);
            } else {
                // Boundary path
                simdgroup_store(acc[mi][ni], &staging[0][0], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                for (uint elem = simd_lane; elem < 64; elem += 32) {
                    uint r = elem / 8;
                    uint c = elem % 8;
                    uint gr = out_row + r;
                    uint gc = out_col + c;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = staging[r][c];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
"""

lib = MetalKernelLibrary()
lib.compile_source("debug_store", _DEBUG_KERNEL)

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

# Debug buffer for 4 simdgroups * 4 values each
dbg = np.zeros(16, dtype=np.uint32)
dbg_mps = torch.from_numpy(dbg).to("mps")
dbg_buf = mps_tensor_to_metal_buffer(dbg_mps, lib.device, copy_back=True)

def make_scalar(val):
    return lib.device.newBufferWithBytes_length_options_(np.array([val], dtype=np.uint32).tobytes(), 4, 0)

dispatch_kernel(
    lib,
    function_name="debug_store",
    grid=(1, 1, 1),
    threadgroup=(128, 1, 1),
    buffers=[A_buf, B_buf, C_buf, dbg_buf, make_scalar(M), make_scalar(N), make_scalar(K)],
    wait=True,
)

C = C_mps.cpu().numpy()
dbg_out = dbg_mps.cpu().numpy()

print("Debug info for each simdgroup:")
for sid in range(4):
    idx = sid * 4
    print(f"  SG{sid}: sg_row_offset={dbg_out[idx+1]}, sg_col_offset={dbg_out[idx+2]}, k_iters={dbg_out[idx+3]}")
    
print()
print(f"M={M}, N={N}, K={K}")
print()
print("Output C:")
print(f"Row 0 cols 0-7:   {C[0, :8]}  (should be [0, 16, 32, ...])")
print(f"Row 0 cols 8-15:  {C[0, 8:16]}  (should be [128, 144, ...])")
print(f"Row 0 cols 16-23: {C[0, 16:24]}  (should be [256, ...])")
print(f"Row 0 cols 24-31: {C[0, 24:32]}  (should be [384, ...])")
print()

expected = np.array([16 * n for n in range(32)], dtype=np.float16)
if np.allclose(C[0, :], expected, atol=1):
    print("PASS")
else:
    print("FAIL")
    print(f"Expected: {expected}")
    print(f"Actual:   {C[0, :]}")
