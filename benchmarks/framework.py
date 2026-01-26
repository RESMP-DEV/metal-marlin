"""
Benchmarking framework for Marlin kernels.

Features:
- Automatic warmup and iteration control
- Statistical analysis (mean, std, percentiles)
- GPU sync handling via mx.eval/mx.synchronize
- Result export to JSON/CSV
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx


@dataclass
class BenchmarkResult:
    """Result from a single kernel benchmark run."""

    name: str
    M: int
    N: int
    K: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    tflops: float
    memory_gb_s: float
    iterations: int


class Benchmark:
    """Reusable benchmarking harness with warmup, sync, and statistics.

    Args:
        warmup: Number of warmup iterations (discarded).
        iterations: Number of timed iterations.
        sync_gpu: Whether to call mx.synchronize() after each iteration
            to ensure GPU work completes before timing.
    """

    def __init__(
        self,
        warmup: int = 10,
        iterations: int = 100,
        sync_gpu: bool = True,
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.sync_gpu = sync_gpu
        self.results: list[BenchmarkResult] = []

    def run(
        self,
        name: str,
        fn: Callable[[], object],
        M: int,
        N: int,
        K: int,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            name: Human-readable label for this benchmark configuration.
            fn: Callable that executes the kernel. Return value is ignored.
                The function should NOT call mx.synchronize() internally;
                the harness handles sync based on the sync_gpu flag.
            M: Rows of the activation matrix (batch dimension).
            N: Columns of the output (output features).
            K: Shared dimension (input features).

        Returns:
            BenchmarkResult with timing statistics and throughput metrics.
        """
        # Warmup: run kernel and sync to ensure JIT compilation is done
        for _ in range(self.warmup):
            fn()
            if self.sync_gpu:
                mx.synchronize()

        # Timed iterations
        times: list[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            fn()
            if self.sync_gpu:
                mx.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times.append(elapsed_ms)

        # Statistics
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        sorted_times = sorted(times)
        n = len(sorted_times)

        # GEMM FLOPs: 2*M*N*K (one multiply + one add per output element per K step)
        flops = 2.0 * M * N * K
        # TFLOPS = FLOPs / (time_in_seconds) / 1e12
        tflops = (flops / (mean / 1000.0)) / 1e12 if mean > 0 else 0.0

        # Memory bandwidth estimate (bytes moved per operation):
        # Read A: M*K*2 (FP16)
        # Read B_packed: K*N/2 (4-bit packed)
        # Write C: M*N*2 (FP16)
        bytes_moved = M * K * 2 + K * N // 2 + M * N * 2
        # GB/s = bytes / (time_in_seconds) / 1e9
        memory_gb_s = (bytes_moved / (mean / 1000.0)) / 1e9 if mean > 0 else 0.0

        result = BenchmarkResult(
            name=name,
            M=M,
            N=N,
            K=K,
            mean_ms=mean,
            std_ms=std,
            min_ms=sorted_times[0],
            max_ms=sorted_times[-1],
            p50_ms=sorted_times[n // 2],
            p95_ms=sorted_times[int(n * 0.95)],
            p99_ms=sorted_times[min(int(n * 0.99), n - 1)],
            tflops=tflops,
            memory_gb_s=memory_gb_s,
            iterations=self.iterations,
        )

        self.results.append(result)
        return result

    def export_json(self, path: str | Path) -> None:
        """Export results to JSON."""
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

    def export_csv(self, path: str | Path) -> None:
        """Export results to CSV."""
        if not self.results:
            return

        fieldnames = list(asdict(self.results[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))

    def print_summary(self) -> None:
        """Print formatted summary table to stdout."""
        header = f"{'Name':<30} {'M':>6} {'N':>6} {'K':>6} {'Mean(ms)':>10} {'Std(ms)':>9} {'TFLOPS':>8} {'GB/s':>8}"
        print(header)
        print("-" * len(header))
        for r in self.results:
            print(
                f"{r.name:<30} {r.M:>6} {r.N:>6} {r.K:>6} "
                f"{r.mean_ms:>10.3f} {r.std_ms:>9.3f} "
                f"{r.tflops:>8.2f} {r.memory_gb_s:>8.1f}"
            )
