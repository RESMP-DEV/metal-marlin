#!/usr/bin/env python3
"""Benchmark dequantization performance for various matrix shapes.

Measures dequantization throughput for M=1, 32, 128 shapes on Apple GPUs.
Supports FP4 and INT4 quantized weight formats.

Usage:
    cd contrib/metal_marlin
    uv run python scripts/benchmark_dequant.py
    uv run python scripts/benchmark_dequant.py --M 1 32 128 --N 4096 --K 4096
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.metal_dispatch import (  # noqa: E402
    HAS_METAL,
    HAS_MPS,
    MetalKernelLibrary,
    dispatch_kernel,
    get_shader_source,
    mps_tensor_to_metal_buffer,
)

try:
    from metal_marlin.kernels import pack_fp4_weights  # noqa: E402
    from metal_marlin.metal_marlin import pack_u4_weights  # noqa: E402
except Exception:  # pragma: no cover - optional imports
    pack_fp4_weights = None
    pack_u4_weights = None

if HAS_METAL:
    import Metal  # type: ignore

if TYPE_CHECKING:
    from typing import Any


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    M: int
    N: int
    K: int
    format: str
    mean_ms: float
    std_ms: float
    throughput_gbps: float
    elements_per_sec: float


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def _random_fp4_inputs(K: int, N: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random FP4 packed weights and scales."""
    k_blocks = _ceil_div(K, 8)
    packed = np.random.randint(0, 2**32 - 1, size=(k_blocks, N), dtype=np.uint32)
    scales = np.random.uniform(0.01, 1.0, size=(K // group_size, N)).astype(np.float16)
    packed_t = torch.from_numpy(packed).to("mps")
    scales_t = torch.from_numpy(scales).to("mps")
    return packed_t, scales_t


def _random_int4_inputs(
    K: int, N: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random INT4 packed weights, scales, and zeros."""
    k_blocks = _ceil_div(K, 8)
    packed = np.random.randint(0, 2**32 - 1, size=(k_blocks, N), dtype=np.uint32)
    scales = np.random.uniform(0.01, 1.0, size=(K // group_size, N)).astype(np.float16)
    zeros = np.random.uniform(0.0, 15.0, size=(K // group_size, N)).astype(np.float16)
    packed_t = torch.from_numpy(packed).to("mps")
    scales_t = torch.from_numpy(scales).to("mps")
    zeros_t = torch.from_numpy(zeros).to("mps")
    return packed_t, scales_t, zeros_t


def _setup_metal_dequant(
    lib: MetalKernelLibrary,
    device: Any,
    packed: torch.Tensor,
    scales: torch.Tensor,
    K: int,
    N: int,
    group_size: int,
    zeros: torch.Tensor | None = None,
) -> tuple[list[Any], tuple[int, int, int], tuple[int, int, int]]:
    """Setup Metal buffers and grid configuration for dequantization.

    Returns:
        Tuple of (buffers, grid, threadgroup)
    """
    k_blocks = _ceil_div(K, 8)
    tg_x, tg_y = 16, 16
    grid = (_ceil_div(N, tg_x), _ceil_div(k_blocks, tg_y), 1)
    threadgroup = (tg_x, tg_y, 1)

    packed_buf = mps_tensor_to_metal_buffer(packed.contiguous(), device)
    scales_buf = mps_tensor_to_metal_buffer(scales.contiguous(), device)

    k_param = np.array([K], dtype=np.uint32)
    n_param = np.array([N], dtype=np.uint32)
    gs_param = np.array([group_size], dtype=np.uint32)

    k_buf = device.newBufferWithBytes_length_options_(
        k_param.tobytes(), k_param.nbytes, Metal.MTLResourceStorageModeShared
    )
    n_buf = device.newBufferWithBytes_length_options_(
        n_param.tobytes(), n_param.nbytes, Metal.MTLResourceStorageModeShared
    )
    gs_buf = device.newBufferWithBytes_length_options_(
        gs_param.tobytes(), gs_param.nbytes, Metal.MTLResourceStorageModeShared
    )

    if zeros is not None:
        # INT4 path
        zeros_buf = mps_tensor_to_metal_buffer(zeros.contiguous(), device)
        output = torch.empty((K, N), dtype=torch.float16, device="mps")
        output_buf = mps_tensor_to_metal_buffer(output, device)
        buffers = [packed_buf, scales_buf, zeros_buf, output_buf, k_buf, n_buf, gs_buf]
    else:
        # FP4 path
        output = torch.empty((K, N), dtype=torch.float16, device="mps")
        output_buf = mps_tensor_to_metal_buffer(output, device)
        buffers = [packed_buf, scales_buf, output_buf, k_buf, n_buf, gs_buf]

    return buffers, grid, threadgroup


def _run_dequant_benchmark(
    lib: MetalKernelLibrary,
    device: Any,
    kernel_name: str,
    buffers: list[Any],
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    M: int,
    N: int,
    K: int,
    fmt: str,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    # Warmup
    for _ in range(warmup):
        dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers, wait=True)

    # Timed runs
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers, wait=True)
        torch.mps.synchronize() if HAS_MPS else None
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0

    # Calculate throughput
    elements = K * N
    total_bytes = elements * 2  # FP16 output
    throughput_gbps = (total_bytes / (mean_ms / 1000)) / 1e9
    elements_per_sec = elements / (mean_ms / 1000)

    return BenchmarkResult(
        M=M,
        N=N,
        K=K,
        format=fmt,
        mean_ms=mean_ms,
        std_ms=std_ms,
        throughput_gbps=throughput_gbps,
        elements_per_sec=elements_per_sec,
    )


def benchmark_fp4(
    lib: MetalKernelLibrary,
    device: Any,
    M: int,
    N: int,
    K: int,
    group_size: int,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark FP4 dequantization."""
    packed, scales = _random_fp4_inputs(K, N, group_size)
    buffers, grid, threadgroup = _setup_metal_dequant(
        lib, device, packed, scales, K, N, group_size
    )

    return _run_dequant_benchmark(
        lib,
        device,
        "dequant_fp4_bulk",
        buffers,
        grid,
        threadgroup,
        M,
        N,
        K,
        "FP4",
        warmup,
        iterations,
    )


def benchmark_int4(
    lib: MetalKernelLibrary,
    device: Any,
    M: int,
    N: int,
    K: int,
    group_size: int,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark INT4 dequantization."""
    packed, scales, zeros = _random_int4_inputs(K, N, group_size)
    buffers, grid, threadgroup = _setup_metal_dequant(
        lib, device, packed, scales, K, N, group_size, zeros
    )

    return _run_dequant_benchmark(
        lib,
        device,
        "dequant_int4_bulk",
        buffers,
        grid,
        threadgroup,
        M,
        N,
        K,
        "INT4",
        warmup,
        iterations,
    )


def format_result(result: BenchmarkResult) -> str:
    """Format benchmark result for display."""
    return (
        f"  {result.format:6s}: {result.mean_ms:7.3f} ± {result.std_ms:5.3f} ms "
        f"| {result.throughput_gbps:5.2f} GB/s | "
        f"{result.elements_per_sec / 1e9:.3f} GE/s"
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark dequantization performance")
    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=[1, 32, 128],
        help="Batch sizes to benchmark (default: 1 32 128)",
    )
    parser.add_argument("--N", type=int, default=4096, help="Output dimension (default: 4096)")
    parser.add_argument("--K", type=int, default=4096, help="Reduction dimension (default: 4096)")
    parser.add_argument(
        "--group-size", type=int, default=128, help="Quantization group size (default: 128)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="fp4,int4",
        help="Comma-separated formats to benchmark (default: fp4,int4)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument(
        "--iterations", type=int, default=50, help="Timed iterations (default: 50)"
    )
    args = parser.parse_args()

    if not (HAS_METAL and HAS_MPS):
        print("Error: Metal and MPS are required for this benchmark")
        print("Install with: pip install pyobjc-framework-Metal torch")
        return 1

    # Validate dimensions
    if args.K % 8 != 0:
        print(f"Error: K={args.K} must be divisible by 8")
        return 1
    if args.K % args.group_size != 0:
        print(f"Error: K={args.K} must be divisible by group_size={args.group_size}")
        return 1

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    # Initialize Metal
    lib = MetalKernelLibrary()
    source = get_shader_source("dequant")
    lib.compile_source("dequant", source)
    device = lib.device

    print("=" * 80)
    print("Dequantization Performance Benchmark")
    print("=" * 80)
    print(f"Configuration: N={args.N}, K={args.K}, group_size={args.group_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    all_results: list[BenchmarkResult] = []

    for M in args.M:
        print(f"M = {M} (batch size)")
        print("-" * 80)

        if "fp4" in formats:
            try:
                result = benchmark_fp4(
                    lib, device, M, args.N, args.K, args.group_size, args.warmup, args.iterations
                )
                all_results.append(result)
                print(format_result(result))
            except Exception as e:
                print(f"  FP4: ERROR - {e}")

        if "int4" in formats:
            try:
                result = benchmark_int4(
                    lib, device, M, args.N, args.K, args.group_size, args.warmup, args.iterations
                )
                all_results.append(result)
                print(format_result(result))
            except Exception as e:
                print(f"  INT4: ERROR - {e}")

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'M':>4s} | {'Format':>6s} | {'Time (ms)':>10s} | {'Throughput':>12s} | {'GE/s':>8s}")
    print("-" * 80)
    for r in all_results:
        print(f"{r.M:4d} | {r.format:>6s} | {r.mean_ms:10.3f} | {r.throughput_gbps:10.2f} GB/s | {r.elements_per_sec / 1e9:8.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
