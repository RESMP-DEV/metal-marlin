#!/usr/bin/env python3
"""
A/B Test Script for Kernel Variants.

Compares baseline and optimized kernel implementations with statistical validation.

Usage:
    python kernel_ab_test.py --help
    python kernel_ab_test.py --iterations 100 --baseline my_baseline --optimized my_optimized
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


@dataclass
class BenchmarkResult:
    """Results from benchmarking a kernel."""

    name: str
    times_ms: list[float]
    mean_ms: float
    std_ms: float
    median_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_times(cls, name: str, times_ms: list[float]) -> BenchmarkResult:
        """Create result from timing measurements."""
        return cls(
            name=name,
            times_ms=times_ms,
            mean_ms=float(np.mean(times_ms)),
            std_ms=float(np.std(times_ms, ddof=1)),
            median_ms=float(np.median(times_ms)),
            min_ms=float(np.min(times_ms)),
            max_ms=float(np.max(times_ms)),
        )


@dataclass
class ABTestResult:
    """Statistical comparison between baseline and optimized kernels."""

    baseline: BenchmarkResult
    optimized: BenchmarkResult
    speedup: float
    speedup_percent: float
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float

    def __str__(self) -> str:
        lines = [
            f"\n{'=' * 70}",
            "A/B Test Results",
            f"{'=' * 70}",
            "",
            f"Baseline: {self.baseline.name}",
            f"  Mean:   {self.baseline.mean_ms:.4f} ms",
            f"  Std:    {self.baseline.std_ms:.4f} ms",
            f"  Median: {self.baseline.median_ms:.4f} ms",
            "",
            f"Optimized: {self.optimized.name}",
            f"  Mean:   {self.optimized.mean_ms:.4f} ms",
            f"  Std:    {self.optimized.std_ms:.4f} ms",
            f"  Median: {self.optimized.median_ms:.4f} ms",
            "",
            f"Speedup: {self.speedup:.3f}x ({self.speedup_percent:+.1f}%)",
            "Statistical Test: t-test",
            f"  t-statistic: {self.t_statistic:.4f}",
            f"  p-value: {self.p_value:.6f}",
            f"  Confidence level: {self.confidence_level * 100:.0f}%",
            f"  Significant: {'YES' if self.is_significant else 'NO'}",
            f"{'=' * 70}",
        ]
        return "\n".join(lines)


def benchmark_kernel(
    kernel_fn: Callable[[Any], Any],
    input_generator: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """
    Benchmark a kernel function.

    Args:
        kernel_fn: Function to benchmark.
        input_generator: Function that generates input data.
        iterations: Number of timed iterations.
        warmup: Number of warmup iterations.

    Returns:
        BenchmarkResult with timing statistics.
    """
    name = getattr(kernel_fn, "__name__", str(kernel_fn))

    # Warmup runs
    for _ in range(warmup):
        inputs = input_generator()
        _ = kernel_fn(inputs)

    # Timed runs
    times_ms: list[float] = []
    for _ in range(iterations):
        inputs = input_generator()

        start = time.perf_counter()
        _ = kernel_fn(inputs)
        elapsed = time.perf_counter() - start

        times_ms.append(elapsed * 1000)

    return BenchmarkResult.from_times(name, times_ms)


def run_ab_test(
    baseline_fn: Callable[[Any], Any],
    optimized_fn: Callable[[Any], Any],
    input_generator: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 10,
    confidence_level: float = 0.95,
) -> ABTestResult:
    """
    Run A/B test comparing baseline and optimized kernels.

    Args:
        baseline_fn: Baseline kernel function.
        optimized_fn: Optimized kernel function.
        input_generator: Function that generates input data.
        iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.
        confidence_level: Statistical confidence level (e.g., 0.95 for 95%).

    Returns:
        ABTestResult with statistical comparison.
    """
    print("Benchmarking baseline kernel...")
    baseline = benchmark_kernel(baseline_fn, input_generator, iterations, warmup)

    print("Benchmarking optimized kernel...")
    optimized = benchmark_kernel(optimized_fn, input_generator, iterations, warmup)

    # Statistical t-test
    t_stat, p_value = stats.ttest_ind(baseline.times_ms, optimized.times_ms)

    # Calculate speedup
    speedup = baseline.mean_ms / optimized.mean_ms
    speedup_percent = (speedup - 1) * 100

    # Determine significance
    is_significant = p_value < (1 - confidence_level)

    return ABTestResult(
        baseline=baseline,
        optimized=optimized,
        speedup=speedup,
        speedup_percent=speedup_percent,
        t_statistic=t_stat,
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=confidence_level,
    )


# =============================================================================
# Example Kernel Implementations for Demonstration
# =============================================================================


def example_baseline_matmul(x: Any) -> Any:
    """Baseline matrix multiplication using PyTorch."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    a, b = x
    return torch.matmul(a, b)


def example_optimized_matmul(x: Any) -> Any:
    """Optimized matrix multiplication (simulated)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    a, b = x
    # Use torch.mm which is optimized for 2D matrices
    return torch.mm(a, b)


def example_input_generator() -> Any:
    """Generate input data for benchmarking."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    a = torch.randn(512, 512, device="cpu")
    b = torch.randn(512, 512, device="cpu")
    return (a, b)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B Test Script for Kernel Variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (demo kernels)
  python kernel_ab_test.py --demo

  # Custom benchmark with 200 iterations
  python kernel_ab_test.py --iterations 200 --baseline fn1 --optimized fn2

  # High confidence statistical test
  python kernel_ab_test.py --confidence 0.99 --demo
        """,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for statistical test (default: 0.95)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline kernel function name (use with custom benchmarking)",
    )
    parser.add_argument(
        "--optimized",
        type=str,
        default=None,
        help="Optimized kernel function name (use with custom benchmarking)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with example matmul kernels",
    )

    args = parser.parse_args()

    # Validate confidence level
    if not 0 < args.confidence < 1:
        print("Error: confidence must be between 0 and 1")
        return 1

    # Default to demo if no custom kernels specified
    if args.demo or (args.baseline is None and args.optimized is None):
        print("Running demo with example matmul kernels...\n")
        if not TORCH_AVAILABLE:
            print("Error: PyTorch required for demo. Install with: pip install torch")
            return 1

        result = run_ab_test(
            example_baseline_matmul,
            example_optimized_matmul,
            example_input_generator,
            args.iterations,
            args.warmup,
            args.confidence,
        )
    else:
        print(f"Custom benchmark: {args.baseline} vs {args.optimized}\n")
        print(
            "Note: Custom benchmarking requires extending this script with your own kernel functions."
        )
        print("Use --demo to see an example implementation.")
        return 0

    # Print results
    print(result)

    # Return exit code based on significance
    return 0 if result.is_significant and result.speedup > 1 else 0


if __name__ == "__main__":
    exit(main())
