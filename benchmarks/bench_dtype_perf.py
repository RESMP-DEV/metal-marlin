#!/usr/bin/env python3
"""
Benchmark FP16/BF16/FP32 performance on Apple Silicon.

Apple Silicon has different performance characteristics than NVIDIA GPUs:
- M1/M2: FP16 is 2x faster than FP32 (dedicated FP16 ALUs)
- M3/M4: BF16 has dedicated hardware support
- Metal Performance Shaders may optimize differently per dtype

This script measures actual GEMM throughput for each dtype to determine
the optimal precision for "keeping layers in original format."

Usage:
    python benchmarks/bench_dtype_perf.py
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass

# Try to import torch for MPS benchmarks
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, skipping MPS benchmarks")


@dataclass
class DtypeBenchResult:
    """Benchmark result for a single dtype."""

    dtype_name: str
    gemm_tflops: float
    bandwidth_gbps: float
    supported: bool
    notes: str = ""


def get_gpu_info() -> dict[str, str]:
    """Get Apple Silicon GPU information."""
    info = {
        "chip": "Unknown",
        "gpu_cores": "Unknown",
        "memory": "Unknown",
    }

    if platform.system() != "Darwin":
        return info

    try:
        # Get chip info
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        )
        info["chip"] = result.stdout.strip()

        # Get GPU cores from system_profiler
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "Total Number of Cores" in line:
                info["gpu_cores"] = line.split(":")[-1].strip()
            elif "VRAM" in line or "Memory" in line:
                info["memory"] = line.split(":")[-1].strip()

    except Exception:
        pass

    return info


def bench_torch_mps_gemm(
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> tuple[float, float]:
    """
    Benchmark GEMM on MPS.

    Returns:
        (tflops, bandwidth_gbps)
    """
    if not HAS_TORCH:
        return 0.0, 0.0

    device = torch.device("mps")

    # Create matrices
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup_iters):
        C = torch.mm(A, B)
    torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        C = torch.mm(A, B)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    # Calculate metrics
    flops_per_gemm = 2 * M * N * K  # multiply-add = 2 ops
    total_flops = flops_per_gemm * bench_iters
    tflops = total_flops / elapsed / 1e12

    # Memory bandwidth (read A, B; write C)
    bytes_per_elem = A.element_size()
    bytes_per_gemm = (M * K + K * N + M * N) * bytes_per_elem
    total_bytes = bytes_per_gemm * bench_iters
    bandwidth_gbps = total_bytes / elapsed / 1e9

    return tflops, bandwidth_gbps


def bench_all_dtypes(
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
) -> list[DtypeBenchResult]:
    """Benchmark all supported dtypes."""
    results = []

    dtypes_to_test = [
        ("float32", torch.float32 if HAS_TORCH else None),
        ("float16", torch.float16 if HAS_TORCH else None),
        ("bfloat16", torch.bfloat16 if HAS_TORCH else None),
    ]

    for name, dtype in dtypes_to_test:
        if dtype is None:
            results.append(
                DtypeBenchResult(
                    dtype_name=name,
                    gemm_tflops=0.0,
                    bandwidth_gbps=0.0,
                    supported=False,
                    notes="PyTorch not available",
                )
            )
            continue

        try:
            tflops, bandwidth = bench_torch_mps_gemm(M, N, K, dtype)
            results.append(
                DtypeBenchResult(
                    dtype_name=name,
                    gemm_tflops=tflops,
                    bandwidth_gbps=bandwidth,
                    supported=True,
                )
            )
        except Exception as e:
            results.append(
                DtypeBenchResult(
                    dtype_name=name,
                    gemm_tflops=0.0,
                    bandwidth_gbps=0.0,
                    supported=False,
                    notes=str(e)[:50],
                )
            )

    return results


def bench_batch_sizes() -> dict[str, list[DtypeBenchResult]]:
    """Benchmark different matrix sizes (simulating different layer sizes)."""
    sizes = {
        "small (512x512)": (512, 512, 512),
        "medium (2048x2048)": (2048, 2048, 2048),
        "large (4096x4096)": (4096, 4096, 4096),
        "tall (8192x4096)": (8192, 4096, 4096),  # Like MLP up-proj
        "wide (4096x8192)": (4096, 8192, 4096),  # Like MLP down-proj
    }

    results = {}
    for name, (M, N, K) in sizes.items():
        print(f"  Benchmarking {name}...")
        results[name] = bench_all_dtypes(M, N, K)

    return results


def print_results(results: list[DtypeBenchResult], title: str = "Results"):
    """Print results table."""
    print(f"\n{title}")
    print("=" * 70)
    print(f"{'Dtype':<12} {'TFLOPS':>10} {'BW (GB/s)':>12} {'Supported':>10} {'Notes':<20}")
    print("-" * 70)

    for r in results:
        supported = "Yes" if r.supported else "No"
        print(
            f"{r.dtype_name:<12} {r.gemm_tflops:>10.2f} {r.bandwidth_gbps:>12.1f} {supported:>10} {r.notes:<20}"
        )


def main():
    print("=" * 70)
    print("APPLE SILICON DTYPE PERFORMANCE BENCHMARK")
    print("=" * 70)

    # GPU info
    gpu_info = get_gpu_info()
    print("\nHardware:")
    print(f"  Chip: {gpu_info['chip']}")
    print(f"  GPU Cores: {gpu_info['gpu_cores']}")
    print(f"  Memory: {gpu_info['memory']}")

    if not HAS_TORCH:
        print("\nError: PyTorch required for benchmarks")
        print("Install with: pip install torch")
        return

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("\nError: MPS not available")
        return

    print(f"\nPyTorch: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")

    # Quick single-size benchmark
    print("\n" + "=" * 70)
    print("SINGLE SIZE BENCHMARK (4096x4096)")
    print("=" * 70)

    results = bench_all_dtypes(4096, 4096, 4096)
    print_results(results, "4096x4096 GEMM Performance")

    # Find fastest
    supported = [r for r in results if r.supported and r.gemm_tflops > 0]
    if supported:
        fastest = max(supported, key=lambda x: x.gemm_tflops)
        print(f"\nFastest dtype: {fastest.dtype_name} ({fastest.gemm_tflops:.2f} TFLOPS)")

        # Calculate relative performance
        print("\nRelative Performance:")
        for r in supported:
            rel = r.gemm_tflops / fastest.gemm_tflops * 100
            print(f"  {r.dtype_name}: {rel:.1f}%")

    # Multi-size benchmark
    print("\n" + "=" * 70)
    print("MULTI-SIZE BENCHMARK")
    print("=" * 70)

    all_results = bench_batch_sizes()

    for name, results in all_results.items():
        print_results(results, f"\n{name}")

    # Summary recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR METAL MARLIN")
    print("=" * 70)

    # Analyze results
    large_results = all_results.get("large (4096x4096)", [])
    supported = {r.dtype_name: r for r in large_results if r.supported and r.gemm_tflops > 0}

    if "bfloat16" in supported and "float16" in supported:
        bf16 = supported["bfloat16"]
        fp16 = supported["float16"]

        if bf16.gemm_tflops > fp16.gemm_tflops * 0.95:  # Within 5%
            print("  - BF16 and FP16 have similar performance")
            print("  - Use BF16 for critical layers (larger dynamic range)")
        elif fp16.gemm_tflops > bf16.gemm_tflops:
            speedup = fp16.gemm_tflops / bf16.gemm_tflops
            print(f"  - FP16 is {speedup:.2f}x faster than BF16")
            print("  - Consider FP16 for performance-critical paths")
        else:
            speedup = bf16.gemm_tflops / fp16.gemm_tflops
            print(f"  - BF16 is {speedup:.2f}x faster than FP16")
            print("  - Use BF16 for all high-precision layers")

    if "float32" in supported:
        fp32 = supported["float32"]
        best_half = max(
            (
                supported.get("float16", DtypeBenchResult("", 0, 0, False)),
                supported.get("bfloat16", DtypeBenchResult("", 0, 0, False)),
            ),
            key=lambda x: x.gemm_tflops,
        )
        if best_half.gemm_tflops > 0:
            speedup = best_half.gemm_tflops / fp32.gemm_tflops
            print(f"  - {best_half.dtype_name.upper()} is {speedup:.2f}x faster than FP32")
            print("  - Avoid FP32 for inference unless necessary")


if __name__ == "__main__":
    main()
