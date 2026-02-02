"""Prometheus-compatible metrics for Trellis MoE inference.

This module provides production monitoring metrics for the MoE forward pass,
including latency histograms, token counters, and path tracking.

Metrics are disabled by default. Enable with METAL_MARLIN_METRICS=1.

Example:
    >>> from metal_marlin.trellis.metrics import moe_metrics, metrics_enabled
    >>> if metrics_enabled():
    ...     with moe_metrics.time_forward():
    ...         # ... forward pass ...
    ...         moe_metrics.inc_tokens(batch_size * seq_len)
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def metrics_enabled() -> bool:
    """Check if metrics collection is enabled.

    Returns:
        True if METAL_MARLIN_METRICS=1, False otherwise.
    """
    return os.environ.get("METAL_MARLIN_METRICS", "0") == "1"


@dataclass
class HistogramBucket:
    """Single histogram bucket with upper bound and count."""

    le: float  # Upper bound (less than or equal)
    count: int = 0


def _default_buckets() -> list[HistogramBucket]:
    """Create default latency buckets: 1ms to 10s."""
    boundaries = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    buckets = [HistogramBucket(le=b) for b in boundaries]
    buckets.append(HistogramBucket(le=float("inf")))
    return buckets


@dataclass
class Histogram:
    """Prometheus-style histogram with configurable buckets.

    Tracks latency distributions with predefined bucket boundaries.
    Thread-safe via atomic operations on bucket counts.
    """

    name: str
    help_text: str
    buckets: list[HistogramBucket] = field(default_factory=_default_buckets)
    sum_value: float = 0.0
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float) -> None:
        """Record an observation.

        Args:
            value: The observed value (e.g., latency in seconds).
        """
        with self._lock:
            for bucket in self.buckets:
                if value <= bucket.le:
                    bucket.count += 1
            self.sum_value += value
            self.count += 1

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format.

        Returns:
            Prometheus-compatible metric string.
        """
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            cumulative = 0
            for bucket in self.buckets:
                cumulative += bucket.count
                le_str = "+Inf" if bucket.le == float("inf") else str(bucket.le)
                lines.append(f'{self.name}_bucket{{le="{le_str}"}} {cumulative}')
            lines.append(f"{self.name}_sum {self.sum_value}")
            lines.append(f"{self.name}_count {self.count}")
        return "\n".join(lines)


@dataclass
class Counter:
    """Prometheus-style counter (monotonically increasing).

    Thread-safe via lock.
    """

    name: str
    help_text: str
    value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: int = 1) -> None:
        """Increment the counter.

        Args:
            amount: Amount to increment (default 1).
        """
        with self._lock:
            self.value += amount

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format.

        Returns:
            Prometheus-compatible metric string.
        """
        with self._lock:
            return f"# HELP {self.name} {self.help_text}\n# TYPE {self.name} counter\n{self.name} {self.value}"


@dataclass
class Gauge:
    """Prometheus-style gauge (can go up or down).

    Thread-safe via lock.
    """

    name: str
    help_text: str
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float) -> None:
        """Set the gauge value.

        Args:
            value: New value to set.
        """
        with self._lock:
            self.value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge.

        Args:
            amount: Amount to increment (default 1.0).
        """
        with self._lock:
            self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge.

        Args:
            amount: Amount to decrement (default 1.0).
        """
        with self._lock:
            self.value -= amount

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format.

        Returns:
            Prometheus-compatible metric string.
        """
        with self._lock:
            return f"# HELP {self.name} {self.help_text}\n# TYPE {self.name} gauge\n{self.name} {self.value}"


class MoEMetrics:
    """Centralized metrics collector for MoE inference.

    Provides:
    - Forward pass latency histogram
    - Token counter
    - Fast path usage counter
    - Fallback counter
    - NaN detection counter
    - Memory usage gauge

    All operations are no-ops if metrics are disabled.
    """

    def __init__(self) -> None:
        """Initialize metrics collectors."""
        self.forward_latency = Histogram(
            name="moe_forward_seconds",
            help_text="MoE forward pass latency in seconds",
        )
        self.tokens_processed = Counter(
            name="moe_tokens_processed_total",
            help_text="Total number of tokens processed by MoE",
        )
        self.fast_path_used = Counter(
            name="moe_fast_path_used_total",
            help_text="Number of forward passes using fast Metal kernel",
        )
        self.fallback_used = Counter(
            name="moe_fallback_total",
            help_text="Number of forward passes using slow fallback path",
        )
        self.nan_detected = Counter(
            name="moe_nan_detected_total",
            help_text="Number of NaN detections in MoE output",
        )
        self.memory_bytes = Gauge(
            name="moe_memory_bytes",
            help_text="Current MoE memory usage in bytes",
        )
        self._forward_in_progress = Gauge(
            name="moe_forward_in_progress",
            help_text="Number of MoE forward passes currently in progress",
        )

    @contextmanager
    def time_forward(self) -> Iterator[None]:
        """Context manager to time a forward pass.

        Example:
            >>> with moe_metrics.time_forward():
            ...     output = model.forward(x)
        """
        if not metrics_enabled():
            yield
            return

        self._forward_in_progress.inc()
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.forward_latency.observe(elapsed)
            self._forward_in_progress.dec()

    def inc_tokens(self, count: int) -> None:
        """Increment tokens processed counter.

        Args:
            count: Number of tokens processed.
        """
        if metrics_enabled():
            self.tokens_processed.inc(count)

    def inc_fast_path(self) -> None:
        """Increment fast path usage counter."""
        if metrics_enabled():
            self.fast_path_used.inc()

    def inc_fallback(self) -> None:
        """Increment fallback path usage counter."""
        if metrics_enabled():
            self.fallback_used.inc()

    def inc_nan(self) -> None:
        """Increment NaN detection counter."""
        if metrics_enabled():
            self.nan_detected.inc()

    def set_memory(self, bytes_used: int) -> None:
        """Set current memory usage.

        Args:
            bytes_used: Memory usage in bytes.
        """
        if metrics_enabled():
            self.memory_bytes.set(float(bytes_used))

    def update_memory_from_mps(self) -> None:
        """Update memory gauge from MPS allocator if available."""
        if not metrics_enabled():
            return
        try:
            import torch

            if torch.backends.mps.is_available():
                self.memory_bytes.set(float(torch.mps.current_allocated_memory()))
        except Exception:
            pass

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus exposition format.

        Returns:
            Prometheus-compatible metrics string.
        """
        if not metrics_enabled():
            return "# Metrics disabled. Set METAL_MARLIN_METRICS=1 to enable.\n"

        sections = [
            self.forward_latency.to_prometheus(),
            self.tokens_processed.to_prometheus(),
            self.fast_path_used.to_prometheus(),
            self.fallback_used.to_prometheus(),
            self.nan_detected.to_prometheus(),
            self.memory_bytes.to_prometheus(),
            self._forward_in_progress.to_prometheus(),
        ]
        return "\n\n".join(sections) + "\n"

    def reset(self) -> None:
        """Reset all metrics to initial values.

        Useful for testing.
        """
        self.forward_latency = Histogram(
            name="moe_forward_seconds",
            help_text="MoE forward pass latency in seconds",
        )
        self.tokens_processed = Counter(
            name="moe_tokens_processed_total",
            help_text="Total number of tokens processed by MoE",
        )
        self.fast_path_used = Counter(
            name="moe_fast_path_used_total",
            help_text="Number of forward passes using fast Metal kernel",
        )
        self.fallback_used = Counter(
            name="moe_fallback_total",
            help_text="Number of forward passes using slow fallback path",
        )
        self.nan_detected = Counter(
            name="moe_nan_detected_total",
            help_text="Number of NaN detections in MoE output",
        )
        self.memory_bytes = Gauge(
            name="moe_memory_bytes",
            help_text="Current MoE memory usage in bytes",
        )
        self._forward_in_progress = Gauge(
            name="moe_forward_in_progress",
            help_text="Number of MoE forward passes currently in progress",
        )


# Global singleton instance
moe_metrics = MoEMetrics()


__all__ = [
    "metrics_enabled",
    "moe_metrics",
    "MoEMetrics",
    "Histogram",
    "Counter",
    "Gauge",
]
