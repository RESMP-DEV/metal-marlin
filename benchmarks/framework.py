"""
Benchmarking framework for Metal kernels.

Features:
- Automatic warmup and iteration control
- Statistical analysis (mean, std, percentiles)
- GPU sync via PyTorch MPS + optional Metal profiling
- Result export to JSON/CSV
- Direct Metal kernel timing via command buffer timestamps

Requirements:
    - PyTorch with MPS backend
    - Optional: PyObjC Metal for command buffer timestamps
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

# Check MPS availability
if not torch.backends.mps.is_available():
    raise RuntimeError(
        "Metal benchmarking requires PyTorch MPS backend.\n"
        "Ensure you're on Apple Silicon with PyTorch >= 2.0"
    )

# Optional: PyObjC Metal for GPU timestamps
try:
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None


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
    # Optional GPU-side timing (requires Metal timestamps)
    gpu_mean_ms: float | None = None
    gpu_std_ms: float | None = None
    # Extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelConfig:
    """Configuration for a kernel benchmark."""

    name: str
    M: int
    N: int
    K: int
    dtype: torch.dtype = torch.float16
    group_size: int = 32
    bits: int = 4  # quantization bits (4, 8, etc.)


def mps_sync() -> None:
    """Synchronize MPS device and wait for all kernels to complete."""
    torch.mps.synchronize()


class Benchmark:
    """Reusable benchmarking harness with warmup, sync, and statistics.

    Uses PyTorch MPS for GPU operations with optional Metal timestamp
    support for accurate GPU-side timing.

    Args:
        warmup: Number of warmup iterations (discarded).
        iterations: Number of timed iterations.
        sync_gpu: Whether to synchronize MPS after each iteration.
        use_metal_timestamps: Use Metal command buffer timestamps for
            GPU-side timing (requires PyObjC Metal).
    """

    def __init__(
        self,
        warmup: int = 10,
        iterations: int = 100,
        sync_gpu: bool = True,
        use_metal_timestamps: bool = False,
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.sync_gpu = sync_gpu
        self.use_metal_timestamps = use_metal_timestamps and HAS_METAL
        self.results: list[BenchmarkResult] = []

        # Metal device for timestamps (if available)
        self._metal_device: Any = None
        self._command_queue: Any = None
        if self.use_metal_timestamps and Metal is not None:
            self._metal_device = Metal.MTLCreateSystemDefaultDevice()
            if self._metal_device is not None:
                self._command_queue = self._metal_device.newCommandQueue()

    def run(
        self,
        name: str,
        fn: Callable[[], object],
        M: int,
        N: int,
        K: int,
        bits: int = 4,
        group_size: int = 32,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            name: Human-readable label for this benchmark configuration.
            fn: Callable that executes the kernel. Return value is ignored.
                The function should NOT call torch.mps.synchronize() internally;
                the harness handles sync based on the sync_gpu flag.
            M: Rows of the activation matrix (batch dimension).
            N: Columns of the output (output features).
            K: Shared dimension (input features).
            bits: Quantization bit width (for memory bandwidth calculation).
            group_size: Quantization group size (for bandwidth calculation).
            metadata: Optional dict of extra info to store with result.

        Returns:
            BenchmarkResult with timing statistics and throughput metrics.
        """
        # Warmup: run kernel and sync to ensure compilation is done
        for _ in range(self.warmup):
            fn()
            if self.sync_gpu:
                mps_sync()

        # Timed iterations (CPU-side wall clock)
        times: list[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            fn()
            if self.sync_gpu:
                mps_sync()
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
        # Read B_packed: K*N*bits/8 (quantized)
        # Read scales: (K/group_size)*N*2 (FP16 scales)
        # Write C: M*N*2 (FP16)
        bytes_a = M * K * 2
        bytes_b = K * N * bits // 8
        bytes_scales = (K // group_size) * N * 2 if group_size > 0 else 0
        bytes_c = M * N * 2
        bytes_moved = bytes_a + bytes_b + bytes_scales + bytes_c
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
            gpu_mean_ms=None,
            gpu_std_ms=None,
            metadata=metadata or {},
        )

        self.results.append(result)
        return result

    def run_with_metal_timing(
        self,
        name: str,
        dispatch_fn: Callable[[Any, Any], None],
        M: int,
        N: int,
        K: int,
        bits: int = 4,
        group_size: int = 32,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Run benchmark with Metal command buffer timestamps.

        This provides accurate GPU-side timing by using Metal's
        GPUStartTime and GPUEndTime from command buffers.

        Args:
            name: Human-readable label for this benchmark.
            dispatch_fn: Function that takes (command_buffer, encoder) and
                encodes the kernel dispatch. Should NOT commit the buffer.
            M, N, K: Matrix dimensions for FLOPS calculation.
            bits: Quantization bits for bandwidth calculation.
            group_size: Quantization group size.
            metadata: Optional extra metadata.

        Returns:
            BenchmarkResult with both CPU and GPU timing.

        Raises:
            RuntimeError: If Metal timestamps are not available.
        """
        if not self.use_metal_timestamps or self._command_queue is None:
            raise RuntimeError(
                "Metal timestamps not available. Install PyObjC: pip install pyobjc-framework-Metal"
            )

        # Warmup
        for _ in range(self.warmup):
            cmd_buf = self._command_queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            dispatch_fn(cmd_buf, encoder)
            encoder.endEncoding()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()

        # Timed iterations with GPU timestamps
        cpu_times: list[float] = []
        gpu_times: list[float] = []

        for _ in range(self.iterations):
            cpu_start = time.perf_counter()

            cmd_buf = self._command_queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            dispatch_fn(cmd_buf, encoder)
            encoder.endEncoding()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()

            cpu_elapsed_ms = (time.perf_counter() - cpu_start) * 1000.0
            cpu_times.append(cpu_elapsed_ms)

            # GPU timestamps (in seconds)
            gpu_start = cmd_buf.GPUStartTime()
            gpu_end = cmd_buf.GPUEndTime()
            if gpu_start > 0 and gpu_end > gpu_start:
                gpu_times.append((gpu_end - gpu_start) * 1000.0)

        # CPU statistics
        cpu_mean = statistics.mean(cpu_times)
        cpu_std = statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0.0
        sorted_cpu = sorted(cpu_times)
        n = len(sorted_cpu)

        # GPU statistics (if available)
        gpu_mean = statistics.mean(gpu_times) if gpu_times else None
        gpu_std = statistics.stdev(gpu_times) if len(gpu_times) > 1 else None

        # Use GPU time for throughput if available, else CPU
        timing_ms = gpu_mean if gpu_mean else cpu_mean

        flops = 2.0 * M * N * K
        tflops = (flops / (timing_ms / 1000.0)) / 1e12 if timing_ms > 0 else 0.0

        bytes_a = M * K * 2
        bytes_b = K * N * bits // 8
        bytes_scales = (K // group_size) * N * 2 if group_size > 0 else 0
        bytes_c = M * N * 2
        bytes_moved = bytes_a + bytes_b + bytes_scales + bytes_c
        memory_gb_s = (bytes_moved / (timing_ms / 1000.0)) / 1e9 if timing_ms > 0 else 0.0

        result = BenchmarkResult(
            name=name,
            M=M,
            N=N,
            K=K,
            mean_ms=cpu_mean,
            std_ms=cpu_std,
            min_ms=sorted_cpu[0],
            max_ms=sorted_cpu[-1],
            p50_ms=sorted_cpu[n // 2],
            p95_ms=sorted_cpu[int(n * 0.95)],
            p99_ms=sorted_cpu[min(int(n * 0.99), n - 1)],
            tflops=tflops,
            memory_gb_s=memory_gb_s,
            iterations=self.iterations,
            gpu_mean_ms=gpu_mean,
            gpu_std_ms=gpu_std,
            metadata=metadata or {},
        )

        self.results.append(result)
        return result

    def run_kernel_sweep(
        self,
        name_prefix: str,
        fn_factory: Callable[[int, int, int], Callable[[], object]],
        configs: list[tuple[int, int, int]],
        bits: int = 4,
        group_size: int = 32,
    ) -> list[BenchmarkResult]:
        """Run benchmark across multiple (M, N, K) configurations.

        Args:
            name_prefix: Prefix for benchmark names.
            fn_factory: Function that takes (M, N, K) and returns the
                benchmark callable. Called once per config.
            configs: List of (M, N, K) tuples to benchmark.
            bits: Quantization bits.
            group_size: Quantization group size.

        Returns:
            List of BenchmarkResult for each config.
        """
        results = []
        for M, N, K in configs:
            fn = fn_factory(M, N, K)
            name = f"{name_prefix}_{M}x{N}x{K}"
            result = self.run(name, fn, M, N, K, bits=bits, group_size=group_size)
            results.append(result)
        return results

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
                row = asdict(r)
                # Flatten metadata dict to JSON string for CSV
                if "metadata" in row and row["metadata"]:
                    row["metadata"] = json.dumps(row["metadata"])
                writer.writerow(row)

    def print_summary(self, show_gpu_time: bool = True) -> None:
        """Print formatted summary table to stdout.

        Args:
            show_gpu_time: Include GPU timing columns if available.
        """
        has_gpu = show_gpu_time and any(r.gpu_mean_ms is not None for r in self.results)

        if has_gpu:
            header = (
                f"{'Name':<30} {'M':>6} {'N':>6} {'K':>6} "
                f"{'CPU(ms)':>9} {'GPU(ms)':>9} {'TFLOPS':>8} {'GB/s':>8}"
            )
        else:
            header = (
                f"{'Name':<30} {'M':>6} {'N':>6} {'K':>6} "
                f"{'Mean(ms)':>10} {'Std(ms)':>9} {'TFLOPS':>8} {'GB/s':>8}"
            )

        print(header)
        print("-" * len(header))

        for r in self.results:
            if has_gpu:
                gpu_str = f"{r.gpu_mean_ms:>9.3f}" if r.gpu_mean_ms else "     N/A"
                print(
                    f"{r.name:<30} {r.M:>6} {r.N:>6} {r.K:>6} "
                    f"{r.mean_ms:>9.3f} {gpu_str} "
                    f"{r.tflops:>8.2f} {r.memory_gb_s:>8.1f}"
                )
            else:
                print(
                    f"{r.name:<30} {r.M:>6} {r.N:>6} {r.K:>6} "
                    f"{r.mean_ms:>10.3f} {r.std_ms:>9.3f} "
                    f"{r.tflops:>8.2f} {r.memory_gb_s:>8.1f}"
                )

    def clear(self) -> None:
        """Clear all stored results."""
        self.results.clear()


# ---------------------------------------------------------------------------
# Convenience functions for common benchmark patterns
# ---------------------------------------------------------------------------


def create_gemm_inputs(
    M: int,
    N: int,
    K: int,
    bits: int = 4,
    group_size: int = 32,
    device: str = "mps",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random inputs for GEMM benchmarking.

    Args:
        M: Batch/sequence dimension.
        N: Output features.
        K: Input features.
        bits: Quantization bits (4, 8).
        group_size: Quantization group size.
        device: Target device.

    Returns:
        Tuple of (A, B_packed, scales) tensors on the target device.
    """
    # Activation matrix A: [M, K] in FP16
    A = torch.randn(M, K, dtype=torch.float16, device=device)

    # Packed weights B: [K * bits // 8 // 4, N] in uint32
    # For 4-bit: 8 values per uint32, so K/8 packed rows
    # For 8-bit: 4 values per uint32, so K/4 packed rows
    values_per_pack = 32 // bits
    packed_k = (K + values_per_pack - 1) // values_per_pack
    B_packed = torch.randint(0, 2**32, (packed_k, N), dtype=torch.int32, device=device).view(
        torch.int32
    )

    # Scales: [K // group_size, N] in FP16
    num_groups = (K + group_size - 1) // group_size
    scales = torch.randn(num_groups, N, dtype=torch.float16, device=device) * 0.1 + 0.5

    return A, B_packed, scales


def standard_gemm_configs() -> list[tuple[int, int, int]]:
    """Return standard GEMM configurations for benchmarking.

    Covers typical LLM shapes:
    - Small batch (1-4) for interactive inference
    - Medium batch (32-128) for batched inference
    - Large batch (512+) for prefill/throughput
    """
    return [
        # Decode (batch=1)
        (1, 4096, 4096),
        (1, 4096, 11008),
        (1, 11008, 4096),
        (1, 4096, 14336),
        (1, 14336, 4096),
        # Small batch
        (4, 4096, 4096),
        (4, 4096, 14336),
        # Medium batch
        (32, 4096, 4096),
        (32, 4096, 14336),
        (128, 4096, 4096),
        # Large batch (prefill)
        (512, 4096, 4096),
        (512, 4096, 14336),
        (2048, 4096, 4096),
    ]


def moe_gemm_configs(num_experts: int = 8) -> list[tuple[int, int, int]]:
    """Return MoE GEMM configurations.

    Args:
        num_experts: Number of experts (affects effective batch).
    """
    # Typical tokens per expert after routing
    return [
        (16, 4096, 14336),  # ~128 tokens / 8 experts
        (32, 4096, 14336),  # ~256 tokens / 8 experts
        (64, 4096, 14336),  # ~512 tokens / 8 experts
        (128, 4096, 14336),  # ~1024 tokens / 8 experts
        (256, 4096, 14336),  # High-throughput
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Metal Benchmark Framework")
    print("  PyTorch MPS: Available")
    print(f"  Metal timestamps: {HAS_METAL}")
    print()

    # Simple self-test
    bench = Benchmark(warmup=5, iterations=20)

    def dummy_kernel() -> None:
        x = torch.randn(1024, 1024, device="mps", dtype=torch.float16)
        y = torch.randn(1024, 1024, device="mps", dtype=torch.float16)
        _ = x @ y

    result = bench.run("matmul_1024x1024", dummy_kernel, 1024, 1024, 1024)
    bench.print_summary()
