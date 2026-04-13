import time

import torch
from metal_marlin.core import _marlin_gemm


def test_prefill():
    # Large prefill dimensions
    M = 2048
    K = 8192
    N = 8192
    group_size = 128
    n_groups = K // group_size

    # Prepare inputs
    A = torch.randn(M, K, dtype=torch.float16, device="mps")
    
    # B packed: shape (K/8, N) in uint32
    B_packed = torch.randint(0, 0xFFFFFFFF, (K // 8, N), dtype=torch.int32, device="mps")
    
    # Scales
    scales = torch.randn(n_groups, N, dtype=torch.float16, device="mps")

    # Output
    C = torch.empty(M, N, dtype=torch.float16, device="mps")
    
    workspace = torch.zeros(N // 128 * 16, device="mps", dtype=torch.int32)

    # Warmup
    for _ in range(3):
        _marlin_gemm(A, B_packed, C, scales, workspace, M, K, N)
    
    torch.mps.synchronize()

    # Time
    start = time.perf_counter()
    iters = 20
    for _ in range(iters):
        _marlin_gemm(A, B_packed, C, scales, workspace, M, K, N)
    torch.mps.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iters
    ops = 2 * M * N * K
    tflops = (ops / (avg_ms / 1000)) / 1e12

    print(f"Prefill 2048x8192x8192: {avg_ms:.2f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    test_prefill()
