"""Profiling utilities for Metal kernel timing.

Provides the @profile_kernel decorator for automatic timing of functions
that dispatch Metal kernels. Supports both CPU wall-clock timing with MPS
synchronization and GPU-side timing via Metal command buffer timestamps.

Example:
    from metal_marlin.utils.profiling import profile_kernel, get_profile_stats

    @profile_kernel("gemm_fp4")
    def my_kernel(A, B):
        return marlin_gemm_fp4(A, B, scales)

    result = my_kernel(A, B)  # Automatically timed
    stats = get_profile_stats("gemm_fp4")
    print(f"Mean: {stats.mean_ms:.3f} ms, Count: {stats.count}")
"""

from __future__ import annotations

import functools
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

from .._compat import HAS_MPS, HAS_PYOBJC_METAL, torch

P = ParamSpec("P")
T = TypeVar("T")


def _gpu_sync() -> None:
    """Synchronize MPS device to ensure kernel completion."""
    if HAS_MPS and torch is not None:
        torch.mps.synchronize()


@dataclass
class ProfileRecord:
    """Single profiling measurement."""

    wall_time_ms: float
    gpu_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time.time_ns)


@dataclass
class ProfileStats:
    """Aggregated statistics for a profiled kernel."""

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
    gpu_mean_ms: float | None = None
    gpu_std_ms: float | None = None

    @classmethod
    def from_records(cls, name: str, records: list[ProfileRecord]) -> ProfileStats:
        """Compute statistics from a list of profile records."""
        if not records:
            raise ValueError("Cannot compute stats from empty record list")

        wall_times = [r.wall_time_ms for r in records]
        sorted_times = sorted(wall_times)
        n = len(sorted_times)

        # GPU timing stats if available
        gpu_times = [r.gpu_time_ms for r in records if r.gpu_time_ms is not None]
        gpu_mean = statistics.mean(gpu_times) if gpu_times else None
        gpu_std = statistics.stdev(gpu_times) if len(gpu_times) > 1 else None

        return cls(
            name=name,
            count=n,
            total_ms=sum(wall_times),
            mean_ms=statistics.mean(wall_times),
            std_ms=statistics.stdev(wall_times) if n > 1 else 0.0,
            min_ms=sorted_times[0],
            max_ms=sorted_times[-1],
            p50_ms=sorted_times[n // 2],
            p95_ms=sorted_times[min(int(n * 0.95), n - 1)],
            p99_ms=sorted_times[min(int(n * 0.99), n - 1)],
            gpu_mean_ms=gpu_mean,
            gpu_std_ms=gpu_std,
        )


class _ProfileRegistry:
    """Thread-safe registry for profile records.

    Stores timing data by kernel name and provides aggregation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, list[ProfileRecord]] = defaultdict(list)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable profiling globally."""
        self._enabled = value

    def add(self, name: str, record: ProfileRecord) -> None:
        """Add a profile record."""
        if not self._enabled:
            return
        with self._lock:
            self._records[name].append(record)

    def get_records(self, name: str) -> list[ProfileRecord]:
        """Get all records for a kernel."""
        with self._lock:
            return list(self._records.get(name, []))

    def get_stats(self, name: str) -> ProfileStats | None:
        """Get aggregated stats for a kernel."""
        records = self.get_records(name)
        if not records:
            return None
        return ProfileStats.from_records(name, records)

    def get_all_stats(self) -> dict[str, ProfileStats]:
        """Get stats for all profiled kernels."""
        with self._lock:
            names = list(self._records.keys())
        return {
            name: stats
            for name in names
            if (stats := self.get_stats(name)) is not None
        }

    def clear(self, name: str | None = None) -> None:
        """Clear profile records.

        Args:
            name: If specified, clear only this kernel. Otherwise clear all.
        """
        with self._lock:
            if name is not None:
                self._records.pop(name, None)
            else:
                self._records.clear()

    def print_summary(self) -> None:
        """Print formatted summary table to stdout."""
        all_stats = self.get_all_stats()
        if not all_stats:
            print("No profiles collected")
            return

        # Check if any GPU times are available
        has_gpu = any(s.gpu_mean_ms is not None for s in all_stats.values())

        if has_gpu:
            header = (
                f"{'Kernel':<35} {'Count':>6} {'Mean(ms)':>10} "
                f"{'GPU(ms)':>9} {'P95(ms)':>9} {'Total(ms)':>10}"
            )
        else:
            header = (
                f"{'Kernel':<35} {'Count':>6} {'Mean(ms)':>10} "
                f"{'Std(ms)':>9} {'P95(ms)':>9} {'Total(ms)':>10}"
            )

        print(header)
        print("-" * len(header))

        for stats in sorted(all_stats.values(), key=lambda s: s.total_ms, reverse=True):
            if has_gpu:
                gpu_str = f"{stats.gpu_mean_ms:.3f}" if stats.gpu_mean_ms else "N/A"
                print(
                    f"{stats.name:<35} {stats.count:>6} {stats.mean_ms:>10.3f} "
                    f"{gpu_str:>9} {stats.p95_ms:>9.3f} {stats.total_ms:>10.1f}"
                )
            else:
                print(
                    f"{stats.name:<35} {stats.count:>6} {stats.mean_ms:>10.3f} "
                    f"{stats.std_ms:>9.3f} {stats.p95_ms:>9.3f} {stats.total_ms:>10.1f}"
                )


# Global registry
_registry = _ProfileRegistry()


def profile_kernel(
    name: str | None = None,
    *,
    sync_before: bool = True,
    sync_after: bool = True,
    use_metal_timestamps: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator to profile Metal kernel execution time.

    Wraps a function to automatically measure its execution time,
    with optional MPS synchronization for accurate GPU timing.

    Args:
        name: Kernel identifier. Defaults to the function name.
        sync_before: Sync MPS before timing (default True).
        sync_after: Sync MPS after timing (default True).
        use_metal_timestamps: Use Metal command buffer GPU timestamps
            for more accurate GPU-side timing. Requires PyObjC Metal
            and the function to return a command buffer.
        metadata: Optional static metadata to attach to all records.

    Returns:
        Decorated function that profiles each call.

    Example:
        @profile_kernel("attention_forward")
        def attention(q, k, v):
            return flash_attention_v2(q, k, v)

        # Or with function name as kernel name
        @profile_kernel()
        def my_gemm(A, B):
            return matmul(A, B)

        # With Metal timestamps (function must return cmd_buf)
        @profile_kernel("fused_qkv", use_metal_timestamps=True)
        def fused_qkv_dispatch():
            cmd_buf = queue.commandBuffer()
            # ... encode kernel ...
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()
            return cmd_buf
    """

    def decorator(fn: Any) -> Any:
        kernel_name = name if name is not None else fn.__name__
        static_metadata = metadata or {}

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _registry.enabled:
                return fn(*args, **kwargs)

            if sync_before:
                _gpu_sync()

            start_time = time.perf_counter()
            result = fn(*args, **kwargs)

            if sync_after:
                _gpu_sync()

            wall_time_ms = (time.perf_counter() - start_time) * 1000.0

            # Try to get GPU timestamps from returned command buffer
            gpu_time_ms: float | None = None
            if use_metal_timestamps and HAS_PYOBJC_METAL:
                cmd_buf = result
                if cmd_buf is not None and hasattr(cmd_buf, "GPUStartTime"):
                    try:
                        gpu_start = cmd_buf.GPUStartTime()
                        gpu_end = cmd_buf.GPUEndTime()
                        if gpu_start > 0 and gpu_end > gpu_start:
                            gpu_time_ms = (gpu_end - gpu_start) * 1000.0
                    except Exception:
                        pass  # Fall back to wall time

            record = ProfileRecord(
                wall_time_ms=wall_time_ms,
                gpu_time_ms=gpu_time_ms,
                metadata=dict(static_metadata),
            )
            _registry.add(kernel_name, record)

            return result

        return wrapper

    return decorator


def get_profile_stats(name: str) -> ProfileStats | None:
    """Get aggregated statistics for a profiled kernel.

    Args:
        name: Kernel identifier used in @profile_kernel.

    Returns:
        ProfileStats with timing statistics, or None if no records exist.
    """
    return _registry.get_stats(name)


def get_all_profile_stats() -> dict[str, ProfileStats]:
    """Get statistics for all profiled kernels.

    Returns:
        Dictionary mapping kernel names to their ProfileStats.
    """
    return _registry.get_all_stats()


def get_profile_records(name: str) -> list[ProfileRecord]:
    """Get raw profile records for a kernel.

    Args:
        name: Kernel identifier used in @profile_kernel.

    Returns:
        List of ProfileRecord objects.
    """
    return _registry.get_records(name)


def clear_profiles(name: str | None = None) -> None:
    """Clear profile records.

    Args:
        name: If specified, clear only this kernel. Otherwise clear all.
    """
    _registry.clear(name)


def print_profile_summary() -> None:
    """Print a formatted summary of all profile statistics."""
    _registry.print_summary()


def enable_profiling() -> None:
    """Enable profiling globally."""
    _registry.enabled = True


def disable_profiling() -> None:
    """Disable profiling globally.

    When disabled, @profile_kernel decorated functions run without
    any timing overhead.
    """
    _registry.enabled = False


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled."""
    return _registry.enabled


# Convenience alias for the registry (for advanced use)
def get_profile_registry() -> _ProfileRegistry:
    """Get the global profile registry for advanced operations."""
    return _registry
