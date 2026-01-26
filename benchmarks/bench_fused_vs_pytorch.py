import time

import torch

from metal_marlin.inference.pipeline import dequantize_fp4_torch
from metal_marlin.kernels import HAS_METAL, HAS_MPS, marlin_gemm_fp4


def benchmark_pytorch_path():
    '''Dequant then matmul (current pipeline.py approach)'''
    M, K, N = 1, 2560, 10240
    A = torch.randn(M, K, dtype=torch.float16, device='mps')
    packed = torch.randint(0, 255, (K // 8, N), dtype=torch.uint8, device='mps')
    scales = torch.randn(K // 128, N, dtype=torch.float16, device='mps')

    # Warmup
    for _ in range(5):
        W = dequantize_fp4_torch(packed, scales, K, N, 128)
        A @ W
        torch.mps.synchronize()

    # Benchmark
    iters = 100
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        W = dequantize_fp4_torch(packed, scales, K, N, 128)
        A @ W
    torch.mps.synchronize()
    pytorch_time = (time.perf_counter() - t0) / iters
    print(f'PyTorch (dequant+matmul): {pytorch_time * 1e6:.1f} us')
    return pytorch_time


def benchmark_fused_path():
    '''Fused dequant-matmul Metal kernel'''
    if not (HAS_METAL and HAS_MPS):
        print('Fused kernels not available')
        return None

    M, K, N = 1, 2560, 10240
    A = torch.randn(M, K, dtype=torch.float16, device='mps')
    packed = torch.randint(0, 255, (K // 8, N), dtype=torch.uint8, device='mps')
    scales = torch.randn(K // 128, N, dtype=torch.float16, device='mps')

    # Warmup
    for _ in range(5):
        marlin_gemm_fp4(A, packed, scales, group_size=128)
        torch.mps.synchronize()

    # Benchmark
    iters = 100
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        marlin_gemm_fp4(A, packed, scales, group_size=128)
    torch.mps.synchronize()
    fused_time = (time.perf_counter() - t0) / iters
    print(f'Fused Metal kernel: {fused_time * 1e6:.1f} us')
    return fused_time


if __name__ == '__main__':
    pt = benchmark_pytorch_path()
    fused = benchmark_fused_path()
    if fused:
        print(f'Speedup: {pt / fused:.1f}x')
