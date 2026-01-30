#!/usr/bin/env python3
"""Benchmark throughput of Metal trellis dequantization kernels."""

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.metal_dispatch import MetalKernelLibrary
from metal_marlin.quantization.trellis_codebook import TrellisCodebook
from metal_marlin.trellis_dispatch import dispatch_trellis_dequant_fused


def benchmark_dequant_kernel(K: int, N: int, bits: int, iterations: int = 100) -> dict:
    """Benchmark trellis dequantization kernel."""
    # Create test data
    tiles_k = (K + 15) // 16
    tiles_n = (N + 15) // 16
    n_levels = 2 ** bits

    indices = torch.randint(0, n_levels, (tiles_k, tiles_n, 256), dtype=torch.int16, device="mps")
    n_groups = (K + 127) // 128
    scales = torch.randn(n_groups, N, dtype=torch.float32, device="mps")
    codebook = TrellisCodebook(bits=bits)
    grid = torch.from_numpy(codebook.get_grid()).float().to("mps")
    su = torch.randn(K, dtype=torch.float32, device="mps")
    sv = torch.randn(N, dtype=torch.float32, device="mps")

    lib = MetalKernelLibrary.from_source_dir()

    # Warmup
    for _ in range(10):
        _ = dispatch_trellis_dequant_fused(lib, indices, scales, grid, su, sv, K, N)
    torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = dispatch_trellis_dequant_fused(lib, indices, scales, grid, su, sv, K, N)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    elements = K * N * iterations
    throughput_gelem_s = elements / elapsed / 1e9
    bandwidth_gb_s = elements * 2 / elapsed / 1e9  # FP16 output

    return {
        "K": K,
        "N": N,
        "bits": bits,
        "iterations": iterations,
        "elapsed_s": elapsed,
        "throughput_gelem_s": throughput_gelem_s,
        "bandwidth_gb_s": bandwidth_gb_s,
    }


if __name__ == "__main__":
    print("Trellis Dequantization Kernel Benchmark")
    print("=" * 60)

    # Test various sizes (typical MLP/attention dimensions)
    test_cases = [
        (2048, 5632, 3),   # Small MLP
        (2048, 27392, 3),  # Large MLP
        (2048, 2048, 3),   # Attention
        (4096, 11008, 3),  # Medium MLP
    ]

    for K, N, bits in test_cases:
        result = benchmark_dequant_kernel(K, N, bits)
        print(f"[{K:>5} x {N:>5}] {bits}bit: {result['throughput_gelem_s']:.2f} GElem/s, {result['bandwidth_gb_s']:.2f} GB/s")
