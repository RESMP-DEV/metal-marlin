"""Memory bandwidth measurement and analysis for Metal kernels.

Provides tools to measure actual memory bandwidth utilization
and compare against theoretical peak bandwidth.

Apple Silicon Memory Bandwidth:
- M4 Max: 546 GB/s (unified memory)
- M3 Max: 400 GB/s
- M2 Max: 400 GB/s
- M1 Max: 400 GB/s

Bandwidth measurement approaches:
1. Timing-based: bytes_moved / elapsed_time
2. Counter-based: Direct read from GPU counters (when available)
3. Microbenchmark: Dedicated bandwidth probe kernels
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .._compat import HAS_TORCH, torch
from .occupancy import AppleSiliconGPU, detect_gpu
from .trace import TraceEvent


def _gpu_sync() -> None:
    """Synchronize GPU if torch MPS is available."""
    if HAS_TORCH and torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


@dataclass
class BandwidthMeasurement:
    """Result of a bandwidth measurement.

    Attributes:
        name: Measurement identifier.
        bytes_read: Total bytes read from memory.
        bytes_written: Total bytes written to memory.
        elapsed_ms: Elapsed time in milliseconds.
        read_bandwidth_gbs: Read bandwidth in GB/s.
        write_bandwidth_gbs: Write bandwidth in GB/s.
        total_bandwidth_gbs: Combined read + write bandwidth.
        efficiency_pct: Percentage of peak bandwidth achieved.
        peak_bandwidth_gbs: Hardware peak bandwidth.
        metadata: Additional measurement context.
    """

    name: str
    bytes_read: int
    bytes_written: int
    elapsed_ms: float
    read_bandwidth_gbs: float = 0.0
    write_bandwidth_gbs: float = 0.0
    total_bandwidth_gbs: float = 0.0
    efficiency_pct: float = 0.0
    peak_bandwidth_gbs: float = 546.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute derived fields."""
        if self.elapsed_ms > 0:
            elapsed_s = self.elapsed_ms / 1000.0
            self.read_bandwidth_gbs = (self.bytes_read / elapsed_s) / 1e9
            self.write_bandwidth_gbs = (self.bytes_written / elapsed_s) / 1e9
            self.total_bandwidth_gbs = self.read_bandwidth_gbs + self.write_bandwidth_gbs
            self.efficiency_pct = (self.total_bandwidth_gbs / self.peak_bandwidth_gbs) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "name": self.name,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "elapsed_ms": self.elapsed_ms,
            "read_bandwidth_gbs": self.read_bandwidth_gbs,
            "write_bandwidth_gbs": self.write_bandwidth_gbs,
            "total_bandwidth_gbs": self.total_bandwidth_gbs,
            "efficiency_pct": self.efficiency_pct,
            "bandwidth_utilization_pct": self.efficiency_pct,
            "peak_bandwidth_gbs": self.peak_bandwidth_gbs,
            "metadata": self.metadata,
        }

    def to_trace_event(
        self,
        *,
        timestamp_ns: int,
        pid: int = 0,
        tid: int = 0,
    ) -> TraceEvent:
        """Convert to a Chrome trace counter event."""
        return TraceEvent(
            name="memory_bandwidth",
            cat="memory",
            ph="C",
            ts=int(timestamp_ns / 1000),
            pid=pid,
            tid=tid,
            args={
                "read_gbs": self.read_bandwidth_gbs,
                "write_gbs": self.write_bandwidth_gbs,
                "total_gbs": self.total_bandwidth_gbs,
                "bandwidth_utilization_pct": self.efficiency_pct,
                "peak_bandwidth_gbs": self.peak_bandwidth_gbs,
            },
        )


class MemoryBandwidthProfiler:
    """Profile memory bandwidth for Metal kernels.

    Measures actual memory bandwidth achieved and compares to
    theoretical peak for the target hardware.

    Args:
        gpu: Target GPU variant. If None, auto-detects.
        peak_bandwidth_gbs: Override peak bandwidth.

    Example:
        profiler = MemoryBandwidthProfiler()

        # Measure bandwidth for a GEMM
        measurement = profiler.measure(
            name="gemm_fp4_4096",
            fn=lambda: marlin_gemm_fp4(A, B, scales),
            bytes_read=4096 * 4096 * 2 + 4096 * 4096 // 2,
            bytes_written=4096 * 4096 * 2,
        )
        print(f"Bandwidth: {measurement.total_bandwidth_gbs:.1f} GB/s")
        print(f"Efficiency: {measurement.efficiency_pct:.1f}%")
    """

    def __init__(
        self,
        gpu: AppleSiliconGPU | None = None,
        *,
        peak_bandwidth_gbs: float | None = None,
    ):
        self.gpu = gpu or detect_gpu()
        self.peak_bandwidth_gbs = peak_bandwidth_gbs or self.gpu.peak_bw_gbs
        self._measurements: list[BandwidthMeasurement] = []

    def measure(
        self,
        name: str,
        fn: Callable[[], Any],
        bytes_read: int,
        bytes_written: int,
        *,
        warmup: int = 5,
        iterations: int = 20,
        metadata: dict[str, Any] | None = None,
    ) -> BandwidthMeasurement:
        """Measure bandwidth for a function.

        Args:
            name: Measurement identifier.
            fn: Function to measure (no arguments).
            bytes_read: Total bytes read during one invocation.
            bytes_written: Total bytes written during one invocation.
            warmup: Warmup iterations (discarded).
            iterations: Timed iterations.
            metadata: Optional context.

        Returns:
            BandwidthMeasurement with statistics.
        """
        # Warmup
        for _ in range(warmup):
            fn()
            _gpu_sync()

        # Timed iterations
        times: list[float] = []
        for _ in range(iterations):
            _gpu_sync()

            start = time.perf_counter()
            fn()

            _gpu_sync()

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times.append(elapsed_ms)

        # Use median for robustness against outliers
        median_ms = statistics.median(times)

        measurement = BandwidthMeasurement(
            name=name,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            elapsed_ms=median_ms,
            peak_bandwidth_gbs=self.peak_bandwidth_gbs,
            metadata=metadata or {},
        )

        self._measurements.append(measurement)
        return measurement

    def measure_transfer(
        self,
        size_bytes: int,
        direction: str = "both",
        *,
        iterations: int = 20,
    ) -> BandwidthMeasurement:
        """Measure raw memory transfer bandwidth.

        Creates a dedicated bandwidth probe to measure actual
        memory transfer rates using PyTorch MPS.

        Args:
            size_bytes: Size of data to transfer.
            direction: "read", "write", or "both".
            iterations: Number of iterations.

        Returns:
            BandwidthMeasurement.
        """
        if not HAS_TORCH or torch is None or not torch.backends.mps.is_available():
            return BandwidthMeasurement(
                name=f"transfer_{direction}_{size_bytes}",
                bytes_read=size_bytes if direction in ("read", "both") else 0,
                bytes_written=size_bytes if direction in ("write", "both") else 0,
                elapsed_ms=0.0,
                peak_bandwidth_gbs=self.peak_bandwidth_gbs,
            )

        # Create test data
        elements = size_bytes // 4  # float32
        data = torch.randn(elements, device="mps", dtype=torch.float32)
        torch.mps.synchronize()

        def read_kernel() -> Any:
            return data.sum()

        def write_kernel() -> Any:
            return torch.zeros_like(data)

        def both_kernel() -> Any:
            return data + 1.0

        kernel = {"read": read_kernel, "write": write_kernel, "both": both_kernel}[direction]

        bytes_read = size_bytes if direction in ("read", "both") else 0
        bytes_written = size_bytes if direction in ("write", "both") else 0

        return self.measure(
            name=f"transfer_{direction}_{size_bytes}",
            fn=kernel,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            iterations=iterations,
        )

    @property
    def measurements(self) -> list[BandwidthMeasurement]:
        """All collected measurements."""
        return list(self._measurements)

    def clear(self) -> None:
        """Clear collected measurements."""
        self._measurements.clear()

    def print_summary(self) -> None:
        """Print formatted summary table."""
        if not self._measurements:
            print("No measurements collected")
            return

        header = (
            f"{'Name':<35} {'Read GB/s':>10} {'Write GB/s':>11} {'Total GB/s':>11} {'Eff %':>7}"
        )
        print(header)
        print("-" * len(header))

        for m in self._measurements:
            print(
                f"{m.name:<35} {m.read_bandwidth_gbs:>10.1f} "
                f"{m.write_bandwidth_gbs:>11.1f} {m.total_bandwidth_gbs:>11.1f} "
                f"{m.efficiency_pct:>7.1f}"
            )

        print(f"\nPeak bandwidth: {self.peak_bandwidth_gbs:.1f} GB/s ({self.gpu.name})")


def measure_bandwidth(
    fn: Callable[[], Any],
    bytes_read: int,
    bytes_written: int,
    *,
    warmup: int = 5,
    iterations: int = 20,
) -> BandwidthMeasurement:
    """Quick bandwidth measurement.

    Convenience function using default hardware detection.

    Args:
        fn: Function to measure.
        bytes_read: Bytes read per invocation.
        bytes_written: Bytes written per invocation.
        warmup: Warmup iterations.
        iterations: Timed iterations.

    Returns:
        BandwidthMeasurement.

    Example:
        bw = measure_bandwidth(
            lambda: marlin_gemm_fp4(A, B, scales),
            bytes_read=M*K*2 + K*N//2,  # A (fp16) + B (fp4)
            bytes_written=M*N*2,         # C (fp16)
        )
        print(f"Achieved: {bw.total_bandwidth_gbs:.1f} GB/s")
    """
    profiler = MemoryBandwidthProfiler()
    return profiler.measure(
        name="kernel",
        fn=fn,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        warmup=warmup,
        iterations=iterations,
    )


def estimate_gemm_bytes(
    M: int,
    N: int,
    K: int,
    *,
    a_dtype_bytes: int = 2,  # FP16
    b_dtype_bytes: float = 0.5,  # FP4
    c_dtype_bytes: int = 2,  # FP16
    scales_dtype_bytes: int = 2,  # FP16
    group_size: int = 128,
) -> tuple[int, int]:
    """Estimate bytes read and written for a quantized GEMM.

    Args:
        M: Rows of A and C.
        N: Columns of B and C.
        K: Shared dimension.
        a_dtype_bytes: Bytes per element in A.
        b_dtype_bytes: Bytes per element in B (0.5 for FP4).
        c_dtype_bytes: Bytes per element in C.
        scales_dtype_bytes: Bytes per scale value.
        group_size: Quantization group size.

    Returns:
        Tuple of (bytes_read, bytes_written).

    Example:
        bytes_read, bytes_written = estimate_gemm_bytes(
            M=4096, N=4096, K=4096,
            b_dtype_bytes=0.5,  # FP4 quantized
            group_size=128,
        )
    """
    # Bytes read
    a_bytes = M * K * a_dtype_bytes
    b_bytes = int(K * N * b_dtype_bytes)
    num_groups = (K + group_size - 1) // group_size
    scales_bytes = num_groups * N * scales_dtype_bytes

    bytes_read = a_bytes + b_bytes + scales_bytes

    # Bytes written
    bytes_written = M * N * c_dtype_bytes

    return bytes_read, bytes_written


def benchmark_peak_bandwidth(
    sizes_mb: list[int] | None = None,
    iterations: int = 50,
) -> dict[str, BandwidthMeasurement]:
    """Benchmark peak achievable bandwidth at various sizes.

    Useful for characterizing the memory system and finding
    optimal working set sizes.

    Args:
        sizes_mb: List of buffer sizes in MB to test.
        iterations: Iterations per size.

    Returns:
        Dictionary mapping size label to measurement.

    Example:
        results = benchmark_peak_bandwidth([1, 4, 16, 64, 256])
        for name, m in results.items():
            print(f"{name}: {m.total_bandwidth_gbs:.1f} GB/s")
    """
    if sizes_mb is None:
        sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    profiler = MemoryBandwidthProfiler()
    results: dict[str, BandwidthMeasurement] = {}

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        name = f"{size_mb}MB"

        measurement = profiler.measure_transfer(
            size_bytes=size_bytes,
            direction="both",
            iterations=iterations,
        )
        results[name] = measurement

    return results


def analyze_bandwidth_bottleneck(
    achieved_gbs: float,
    peak_gbs: float,
    arithmetic_intensity: float,
    peak_tflops: float = 32.0,
) -> dict[str, Any]:
    """Analyze whether kernel is compute or memory bound.

    Uses roofline model to determine the limiting factor.

    Args:
        achieved_gbs: Achieved memory bandwidth (GB/s).
        peak_gbs: Peak memory bandwidth (GB/s).
        arithmetic_intensity: FLOP/byte ratio.
        peak_tflops: Peak compute (TFLOPS).

    Returns:
        Dictionary with analysis results.
    """
    # Ridge point: where compute and memory intersect
    ridge_point = (peak_tflops * 1000) / peak_gbs  # GFLOP/GB = FLOP/byte

    if arithmetic_intensity < ridge_point:
        bound = "memory"
        utilization = (achieved_gbs / peak_gbs) * 100
        headroom_pct = 100 - utilization
    else:
        bound = "compute"
        # Estimate achieved TFLOPS from bandwidth and AI
        achieved_tflops = (achieved_gbs * arithmetic_intensity) / 1000
        utilization = (achieved_tflops / peak_tflops) * 100
        headroom_pct = 100 - utilization

    return {
        "bound": bound,
        "utilization_pct": utilization,
        "headroom_pct": headroom_pct,
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "achieved_bandwidth_gbs": achieved_gbs,
        "peak_bandwidth_gbs": peak_gbs,
        "peak_tflops": peak_tflops,
    }
