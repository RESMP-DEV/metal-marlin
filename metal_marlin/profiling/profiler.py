"""Core profiling infrastructure for Metal kernels.

Provides timing-based profiling with GPU synchronization and statistics.
This is the foundation layer that other profiling modules build upon.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .._compat import HAS_MLX, mx
from .trace import ChromeTrace, TraceEvent


@dataclass
class KernelProfile:
    """Profile data for a single kernel invocation.

    Attributes:
        name: Kernel identifier.
        wall_time_ms: Wall-clock time including GPU wait.
        gpu_time_ms: Actual GPU execution time (0 if unavailable).
        memory_allocated_bytes: Memory allocated during execution.
        metadata: Optional key-value pairs for context.
    """

    name: str
    wall_time_ms: float
    gpu_time_ms: float = 0.0
    memory_allocated_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_allocated_mb(self) -> float:
        """Memory allocated in megabytes."""
        return self.memory_allocated_bytes / (1024 * 1024)


@dataclass
class ProfileAggregate:
    """Aggregated statistics for multiple invocations of the same kernel.

    Attributes:
        name: Kernel identifier.
        count: Number of invocations.
        total_ms: Sum of all wall times.
        mean_ms: Mean wall time.
        std_ms: Standard deviation of wall times.
        min_ms: Minimum wall time.
        max_ms: Maximum wall time.
        p50_ms: 50th percentile (median).
        p95_ms: 95th percentile.
        p99_ms: 99th percentile.
    """

    name: str
    count: int
    total_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_profiles(cls, profiles: list[KernelProfile]) -> ProfileAggregate:
        """Compute aggregate statistics from a list of profiles."""
        if not profiles:
            raise ValueError("Cannot aggregate empty profile list")

        times = [p.wall_time_ms for p in profiles]
        sorted_times = sorted(times)
        n = len(sorted_times)

        return cls(
            name=profiles[0].name,
            count=n,
            total_ms=sum(times),
            mean_ms=statistics.mean(times),
            std_ms=statistics.stdev(times) if n > 1 else 0.0,
            min_ms=sorted_times[0],
            max_ms=sorted_times[-1],
            p50_ms=sorted_times[n // 2],
            p95_ms=sorted_times[min(int(n * 0.95), n - 1)],
            p99_ms=sorted_times[min(int(n * 0.99), n - 1)],
        )


class Profiler:
    """Kernel profiler with automatic GPU synchronization.

    Collects timing data for kernel invocations and provides
    aggregation and export capabilities.

    Args:
        sync_before: Synchronize GPU before timing (default True).
        sync_after: Synchronize GPU after timing (default True).
        collect_memory: Track memory allocation (default False).

    Example:
        profiler = Profiler()

        with profiler.profile("gemm_fp4"):
            result = marlin_gemm_fp4(A, B, scales)

        profiler.print_summary()
    """

    def __init__(
        self,
        *,
        sync_before: bool = True,
        sync_after: bool = True,
        collect_memory: bool = False,
    ):
        self.sync_before = sync_before
        self.sync_after = sync_after
        self.collect_memory = collect_memory
        self._profiles: list[KernelProfile] = []

    @contextmanager
    def profile(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[KernelProfile]:
        """Profile a kernel execution.

        Args:
            name: Identifier for this kernel invocation.
            metadata: Optional context (e.g., M, N, K dimensions).

        Yields:
            KernelProfile that will be populated after execution.

        Example:
            with profiler.profile("gemm_fp4", {"M": 4096, "N": 4096}):
                result = kernel(...)
        """
        if self.sync_before and HAS_MLX:
            mx.synchronize()

        start_time = time.perf_counter()

        profile = KernelProfile(
            name=name,
            wall_time_ms=0.0,
            metadata=metadata or {},
        )

        try:
            yield profile
        finally:
            if self.sync_after and HAS_MLX:
                mx.synchronize()

            elapsed = time.perf_counter() - start_time
            profile.wall_time_ms = elapsed * 1000.0
            self._profiles.append(profile)

    def add_profile(self, profile: KernelProfile) -> None:
        """Add a pre-constructed profile to the collection."""
        self._profiles.append(profile)

    @property
    def profiles(self) -> list[KernelProfile]:
        """All collected profiles."""
        return list(self._profiles)

    def get_profiles(self, name: str | None = None) -> list[KernelProfile]:
        """Get profiles, optionally filtered by name."""
        if name is None:
            return list(self._profiles)
        return [p for p in self._profiles if p.name == name]

    def aggregate(self, name: str | None = None) -> dict[str, ProfileAggregate]:
        """Aggregate profiles by kernel name.

        Args:
            name: If specified, only aggregate this kernel.

        Returns:
            Dictionary mapping kernel names to their aggregated stats.
        """
        from collections import defaultdict

        by_name: dict[str, list[KernelProfile]] = defaultdict(list)
        for p in self._profiles:
            if name is None or p.name == name:
                by_name[p.name].append(p)

        return {k: ProfileAggregate.from_profiles(v) for k, v in by_name.items()}

    def clear(self) -> None:
        """Clear all collected profiles."""
        self._profiles.clear()

    def print_summary(self) -> None:
        """Print formatted summary table to stdout."""
        aggregates = self.aggregate()
        if not aggregates:
            print("No profiles collected")
            return

        header = (
            f"{'Kernel':<35} {'Count':>6} {'Mean(ms)':>10} "
            f"{'Std(ms)':>9} {'P95(ms)':>9} {'Total(ms)':>10}"
        )
        print(header)
        print("-" * len(header))

        for agg in aggregates.values():
            print(
                f"{agg.name:<35} {agg.count:>6} {agg.mean_ms:>10.3f} "
                f"{agg.std_ms:>9.3f} {agg.p95_ms:>9.3f} {agg.total_ms:>10.1f}"
            )

    def to_dict(self) -> list[dict[str, Any]]:
        """Export profiles as list of dictionaries."""
        return [
            {
                "name": p.name,
                "wall_time_ms": p.wall_time_ms,
                "gpu_time_ms": p.gpu_time_ms,
                "memory_allocated_bytes": p.memory_allocated_bytes,
                "metadata": p.metadata,
            }
            for p in self._profiles
        ]


@dataclass
class KernelCapture:
    """Captured kernel profiling data with GPU metrics."""

    name: str
    start_ns: int
    end_ns: int
    wall_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    counters: Any | None = None
    occupancy: Any | None = None
    bandwidth: Any | None = None
    perf_state: Any | None = None

    def to_trace_events(self, *, pid: int = 0, tid: int = 0) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        events.append(
            TraceEvent(
                name=self.name,
                cat="kernel",
                ph="X",
                ts=int(self.start_ns / 1000),
                dur=int((self.end_ns - self.start_ns) / 1000),
                pid=pid,
                tid=tid,
                args={
                    "wall_time_ms": self.wall_time_ms,
                    **self.metadata,
                },
            )
        )

        if self.counters is not None:
            events.append(self.counters.to_trace_event(pid=pid, tid=tid))
        if self.occupancy is not None:
            events.append(
                self.occupancy.to_trace_event(
                    timestamp_ns=self.end_ns,
                    pid=pid,
                    tid=tid,
                )
            )
        if self.bandwidth is not None:
            events.append(
                self.bandwidth.to_trace_event(
                    timestamp_ns=self.end_ns,
                    pid=pid,
                    tid=tid,
                )
            )
        if self.perf_state is not None:
            events.append(self.perf_state.to_trace_event(pid=pid, tid=tid))

        return events


class MetalProfileSession:
    """Integrated Metal kernel profiling session with counters and trace export."""

    def __init__(
        self,
        *,
        pid: int = 0,
        tid: int = 0,
        enable_counters: bool = True,
    ) -> None:
        from .gpu_counters import GPUProfiler

        self._pid = pid
        self._tid = tid
        self._gpu_profiler = GPUProfiler(enable_counters=enable_counters)
        self._captures: list[KernelCapture] = []

    @contextmanager
    def capture(
        self,
        name: str,
        *,
        flops: float = 0,
        bytes_moved: float = 0,
        bytes_read: int | None = None,
        bytes_written: int | None = None,
        threadgroup_config: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[KernelCapture]:
        from .gpu_counters import read_gpu_performance_state
        from .memory_bandwidth import BandwidthMeasurement, MemoryBandwidthProfiler
        from .occupancy import OccupancyAnalyzer

        self._gpu_profiler.start_capture()
        start_ns = time.time_ns()
        start_time = time.perf_counter()

        capture = KernelCapture(
            name=name,
            start_ns=start_ns,
            end_ns=start_ns,
            wall_time_ms=0.0,
            metadata=metadata or {},
        )

        try:
            yield capture
        finally:
            end_ns = time.time_ns()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            capture.end_ns = end_ns
            capture.wall_time_ms = elapsed_ms
            capture.counters = self._gpu_profiler.stop_capture(
                flops=flops,
                bytes_moved=bytes_moved,
            )

            if threadgroup_config is not None:
                analyzer = OccupancyAnalyzer()
                capture.occupancy = analyzer.analyze(threadgroup_config)
                if capture.counters is not None:
                    capture.counters.threadgroup_occupancy = (
                        capture.occupancy.achieved_occupancy
                    )

            if bytes_read is not None or bytes_written is not None:
                profiler = MemoryBandwidthProfiler()
                capture.bandwidth = BandwidthMeasurement(
                    name=name,
                    bytes_read=bytes_read or 0,
                    bytes_written=bytes_written or 0,
                    elapsed_ms=elapsed_ms,
                    peak_bandwidth_gbs=profiler.peak_bandwidth_gbs,
                )

            capture.perf_state = read_gpu_performance_state()
            self._captures.append(capture)

    @property
    def captures(self) -> list[KernelCapture]:
        return list(self._captures)

    def clear(self) -> None:
        self._captures.clear()

    def to_trace(self) -> ChromeTrace:
        trace = ChromeTrace(pid=self._pid, tid=self._tid)
        for capture in self._captures:
            for event in capture.to_trace_events(pid=self._pid, tid=self._tid):
                trace.add_event(event)
        return trace

    def export_trace(self, output_path: str) -> None:
        trace = self.to_trace()
        trace.export_json(output_path)


# Global profiler instance for convenience API
_global_profiler = Profiler()


@contextmanager
def profile_kernel(
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Iterator[KernelProfile]:
    """Profile a kernel using the global profiler.

    Convenience wrapper around the global Profiler instance.

    Args:
        name: Kernel identifier.
        metadata: Optional context.

    Yields:
        KernelProfile populated after execution.

    Example:
        with profile_kernel("marlin_gemm_fp4"):
            result = marlin_gemm_fp4(A, B, scales)

        from metal_marlin.profiling import get_global_profiler
        get_global_profiler().print_summary()
    """
    with _global_profiler.profile(name, metadata) as profile:
        yield profile


def get_global_profiler() -> Profiler:
    """Get the global profiler instance."""
    return _global_profiler


def clear_global_profiles() -> None:
    """Clear all profiles from the global profiler."""
    _global_profiler.clear()


def bench(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    iterations: int = 100,
    metadata: dict[str, Any] | None = None,
) -> ProfileAggregate:
    """Benchmark a function with warmup and statistics.

    Args:
        name: Benchmark identifier.
        fn: Function to benchmark (no arguments).
        warmup: Number of warmup iterations (discarded).
        iterations: Number of timed iterations.
        metadata: Optional metadata to attach to each profile.

    Returns:
        Aggregated statistics for the benchmark.

    Example:
        stats = bench(
            "gemm_fp4_4096x4096",
            lambda: marlin_gemm_fp4(A, B, scales),
            warmup=20,
            iterations=100,
        )
        print(f"Mean: {stats.mean_ms:.3f} ms")
    """
    profiler = Profiler(sync_before=True, sync_after=True)

    # Warmup (still uses GPU sync, but results discarded)
    for _ in range(warmup):
        fn()
        if HAS_MLX:
            mx.synchronize()

    # Timed iterations
    for _ in range(iterations):
        with profiler.profile(name, metadata):
            fn()

    aggs = profiler.aggregate()
    return aggs[name]
