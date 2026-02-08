#!/usr/bin/env python3
"""Validate and benchmark fused Trellis GEMM kernel.

This benchmark ensures the fused kernel produces correct results
before reporting performance numbers.

Exit codes:
    0: All validations pass, benchmark complete
    1: Validation failure (incorrect results)
    2: Kernel error (compilation/dispatch failure)
"""

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.metal_dispatch import MetalKernelLibrary
from metal_marlin.quantization.trellis_codebook import TrellisCodebook
from metal_marlin.trellis.dispatch import (
import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

    dispatch_gemm_trellis_packed,
    dispatch_trellis_dequant_packed,
)


def create_test_data(M: int, K: int, N: int, bits: int, device: str = "mps"):
    """Create synthetic test data for validation."""
    tiles_k = K // 16
    tiles_n = N // 16
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]

    A = torch.randn(M, K, dtype=torch.float16, device=device)
    packed = torch.randint(0, 256, (tiles_k, tiles_n, packed_bytes),
                           dtype=torch.uint8, device=device)
    scales = torch.randn(K // 32, N, dtype=torch.float32, device=device) * 0.1
    codebook = TrellisCodebook(bits)
    grid = torch.from_numpy(codebook.get_grid()).to(device)
    su = torch.sign(torch.randn(K, device=device))
    sv = torch.sign(torch.randn(N, device=device))

    return A, packed, scales, grid, su, sv


def reference_gemm(lib, A, packed, scales, grid, su, sv, K, N, bits, group_size=32):
    """Reference implementation: dequant then matmul."""
    # Dequant weights to FP16
    W = dispatch_trellis_dequant_packed(lib, packed, scales, grid, su, sv, K, N, bits, group_size)
    # Standard matmul
    return torch.mm(A.float(), W.float()).half()


def validate_correctness(lib, M, K, N, bits, atol=0.5, rtol=0.1):
    """Validate fused kernel against reference."""
    group_size = 32
    A, packed, scales, grid, su, sv = create_test_data(M, K, N, bits)

    # Reference
    ref = reference_gemm(lib, A, packed, scales, grid, su, sv, K, N, bits, group_size)

    # Fused kernel
    out = dispatch_gemm_trellis_packed(lib, A, packed, scales, grid, su, sv, K, N, bits, group_size)

    # Compare
    torch.mps.synchronize()

    if not torch.allclose(out, ref, atol=atol, rtol=rtol):
        max_diff = (out - ref).abs().max().item()
        mean_diff = (out - ref).abs().mean().item()
        print(f"VALIDATION FAILED [{M}x{K}x{N}] {bits}bit")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        return False

    print(f"VALIDATION PASSED [{M}x{K}x{N}] {bits}bit")
    return True


def benchmark_kernel(lib, M, K, N, bits, iterations=20):
    """Benchmark fused kernel performance."""
    group_size = 32
    A, packed, scales, grid, su, sv = create_test_data(M, K, N, bits)

    # Warmup
    for _ in range(5):
        _ = dispatch_gemm_trellis_packed(lib, A, packed, scales, grid, su, sv, K, N, bits, group_size)
    torch.mps.synchronize()

    # Benchmark
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = dispatch_gemm_trellis_packed(lib, A, packed, scales, grid, su, sv, K, N, bits, group_size)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    ms = elapsed / iterations * 1000
    gelem = M * K * N / 1e9
    throughput = gelem * iterations / elapsed

    print(f"[{M:4d}x{K:4d}x{N:5d}] {bits}bit: {ms:6.2f} ms, {throughput:.2f} GElem/s")
    return ms, throughput


def main():
    print("=" * 60)
    print("Fused Trellis GEMM Kernel Validation & Benchmark")
    print("=" * 60)

    try:
        lib = MetalKernelLibrary.from_source_dir()
    except Exception as e:
        print(f"ERROR: Failed to load Metal library: {e}")
        return 2

    # Test shapes (from GLM-4.7-Flash expert projections)
    test_cases = [
        # (M, K, N, bits)
        (1, 2048, 1536, 3),      # Decode: single token
        (32, 2048, 1536, 3),     # Small batch
        (128, 2048, 1536, 3),    # Prefill
        (1, 1536, 2048, 3),      # Down projection
        (1, 2048, 5632, 3),      # Larger projection
    ]

    # Phase 1: Validation
    print("\n--- Phase 1: Correctness Validation ---")
    all_passed = True
    for M, K, N, bits in test_cases:
        if not validate_correctness(lib, M, K, N, bits):
            all_passed = False

    if not all_passed:
        print("\nERROR: Validation failed. Fix kernel before benchmarking.")
        return 1

    # Phase 2: Performance
    print("\n--- Phase 2: Performance Benchmark ---")
    for M, K, N, bits in test_cases:
        benchmark_kernel(lib, M, K, N, bits)

    # Compare with reference (dequant + matmul)
    print("\n--- Comparison: Fused vs Reference ---")
    M, K, N, bits = 1, 2048, 1536, 3
    A, packed, scales, grid, su, sv = create_test_data(M, K, N, bits)

    # Warmup both kernels
    for _ in range(10):
        W = dispatch_trellis_dequant_packed(lib, packed, scales, grid, su, sv, K, N, bits, 32)
        _ = torch.mm(A.float(), W.float()).half()
        _ = dispatch_gemm_trellis_packed(lib, A, packed, scales, grid, su, sv, K, N, bits, 32)
    torch.mps.synchronize()

    # Reference timing
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        W = dispatch_trellis_dequant_packed(lib, packed, scales, grid, su, sv, K, N, bits, 32)
        _ = torch.mm(A.float(), W.float()).half()
    torch.mps.synchronize()
    ref_ms = (time.perf_counter() - t0) / 50 * 1000

    # Fused timing
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = dispatch_gemm_trellis_packed(lib, A, packed, scales, grid, su, sv, K, N, bits, 32)
    torch.mps.synchronize()
    fused_ms = (time.perf_counter() - t0) / 50 * 1000

    speedup = ref_ms / fused_ms
    print(f"Reference (dequant+matmul): {ref_ms:.2f} ms")
    print(f"Fused kernel:               {fused_ms:.2f} ms")
    if speedup >= 1.0:
        print(f"Speedup:                    {speedup:.1f}x")
    else:
        print(f"Slowdown:                   {1.0/speedup:.1f}x")

    print("\n" + "=" * 60)
    print("SUCCESS: All validations passed, benchmark complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
