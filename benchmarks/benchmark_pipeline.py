#!/usr/bin/env python3
"""Benchmark 2-stage vs 3-stage Marlin GEMM pipeline kernels."""

from __future__ import annotations

import argparse
import inspect
import statistics
import time
from collections.abc import Callable

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - MLX optional
    raise SystemExit("ERROR: mlx.core not available. Install MLX to run benchmarks.") from exc

try:
    from metal_marlin import marlin_gemm_2stage, marlin_gemm_3stage, pack_fp4_weights
except ImportError as exc:  # pragma: no cover - optional if bindings missing
    raise SystemExit(
        "ERROR: metal_marlin bindings not available. "
        "Build/install metal_marlin before running benchmarks."
    ) from exc


def pack_weights(weight_kn: mx.array, group_size: int) -> tuple[mx.array, mx.array]:
    """Pack FP16 weights [K, N] into Marlin FP4 packed format.

    pack_fp4_weights expects [N, K] (out_features, in_features), so transpose.
    """
    weight_nk = weight_kn.T
    packed, scales = pack_fp4_weights(weight_nk, group_size=group_size)
    mx.eval(packed, scales)
    return packed, scales


def _call_kernel(
    kernel: Callable[..., mx.array],
    A: mx.array,
    packed: mx.array,
    scales: mx.array,
    group_size: int,
) -> mx.array:
    """Call kernel with or without group_size based on signature support."""
    try:
        params = inspect.signature(kernel).parameters
    except (TypeError, ValueError):
        params = {}

    if "group_size" in params:
        return kernel(A, packed, scales, group_size=group_size)

    try:
        return kernel(A, packed, scales)
    except TypeError:
        return kernel(A, (packed, scales))


def benchmark_kernel(
    kernel: Callable[..., mx.array],
    A: mx.array,
    B: tuple[mx.array, mx.array],
    iters: int = 100,
    warmup: int = 10,
    group_size: int = 128,
) -> float:
    """Return median kernel time in milliseconds."""
    packed, scales = B
    mx.eval(A, packed, scales)

    for _ in range(warmup):
        out = _call_kernel(kernel, A, packed, scales, group_size)
        mx.eval(out)
        mx.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        out = _call_kernel(kernel, A, packed, scales, group_size)
        mx.eval(out)
        mx.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    return statistics.median(times_ms)


def benchmark_pipeline_stages(
    iters: int = 100,
    warmup: int = 10,
    group_size: int = 128,
) -> None:
    sizes = [
        (4096, 4096, 4096),  # Large square
        (1, 4096, 4096),      # Single token decode
        (32, 4096, 4096),     # Batched decode
        (4096, 14336, 4096),  # Llama-7B MLP
    ]

    for M, N, K in sizes:
        A = mx.random.normal((M, K), dtype=mx.float16)
        B = pack_weights(mx.random.normal((K, N), dtype=mx.float16), group_size)

        t2 = benchmark_kernel(marlin_gemm_2stage, A, B, iters=iters,
                              warmup=warmup, group_size=group_size)
        t3 = benchmark_kernel(marlin_gemm_3stage, A, B, iters=iters,
                              warmup=warmup, group_size=group_size)

        speedup = t2 / t3 if t3 > 0 else float("inf")
        print(
            f"{M}x{N}x{K}: 2-stage={t2:.2f}ms, 3-stage={t3:.2f}ms, "
            f"speedup={speedup:.2f}x"
        )


def benchmark_memory_throughput(
    iters: int = 100,
    warmup: int = 10,
    group_size: int = 128,
    theoretical_gbs: float = 400.0,
) -> None:
    """Measure achieved memory bandwidth vs theoretical."""
    M, N, K = 4096, 4096, 4096
    A = mx.random.normal((M, K), dtype=mx.float16)
    packed, scales = pack_weights(mx.random.normal((K, N), dtype=mx.float16), group_size)

    time_ms = benchmark_kernel(
        marlin_gemm_3stage,
        A,
        (packed, scales),
        iters=iters,
        warmup=warmup,
        group_size=group_size,
    )

    bytes_a = M * K * 2
    bytes_b = (K * N) // 2
    bytes_scales = (K // group_size) * N * 2
    bytes_c = M * N * 2
    bytes_total = bytes_a + bytes_b + bytes_scales + bytes_c

    gbs = (bytes_total / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
    pct = (gbs / theoretical_gbs * 100.0) if theoretical_gbs > 0 else 0.0

    print(
        f"Memory throughput: {gbs:.1f} GB/s "
        f"({pct:.1f}% of {theoretical_gbs:.0f} GB/s theoretical)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Marlin 2-stage vs 3-stage pipeline kernels"
    )
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--group-size", type=int, default=128,
                        help="FP4 quantization group size")
    parser.add_argument("--skip-memory", action="store_true",
                        help="Skip memory throughput benchmark")
    args = parser.parse_args()

    print("Benchmarking pipeline stages...")
    benchmark_pipeline_stages(
        iters=args.iters,
        warmup=args.warmup,
        group_size=args.group_size,
    )

    if not args.skip_memory:
        print("\nBenchmarking memory throughput...")
        benchmark_memory_throughput(
            iters=args.iters,
            warmup=args.warmup,
            group_size=args.group_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
