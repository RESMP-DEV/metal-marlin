"""Expert usage profiler for MoE inference optimization.

Tracks per-expert metrics during MoE forward passes:
- Selection frequency (how often each expert is chosen)
- Load distribution (tokens per expert per batch)
- Compute time per expert dispatch
- Memory bandwidth utilization

This complements the routing analysis in analysis/moe_routing.py by focusing
on runtime performance metrics rather than routing pattern analysis.

Usage:
    from metal_marlin.profiling.expert_usage import ExpertUsageProfiler, get_global_expert_profiler

    # Enable profiling globally
    profiler = get_global_expert_profiler()
    profiler.enable()

    # Run inference
    for batch in data:
        output = model(batch)

    # Generate report
    profiler.print_summary()
    report = profiler.generate_report()

Integration with MoE dispatch:
    In dispatch_moe_trellis_swiglu or forward_fast, call:
        profiler.record_dispatch(
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            batch_size=batch_size,
            dispatch_time_ms=elapsed,
        )
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExpertDispatchRecord:
    """Single record of an expert dispatch operation.

    Attributes:
        expert_id: Expert index (0 to num_experts-1)
        batch_size: Number of tokens routed to this expert
        dispatch_time_ms: Wall-clock time for this dispatch
        bytes_read: Estimated bytes read (weights + activations)
        bytes_written: Estimated bytes written (output)
        timestamp_ns: Wall-clock timestamp when dispatch occurred
    """

    expert_id: int
    batch_size: int
    dispatch_time_ms: float
    bytes_read: int = 0
    bytes_written: int = 0
    timestamp_ns: int = 0


@dataclass
class ExpertStats:
    """Aggregated statistics for a single expert.

    Attributes:
        expert_id: Expert index
        total_selections: Total times this expert was selected
        total_tokens: Total tokens routed to this expert
        total_time_ms: Total compute time for this expert
        total_bytes_read: Total bytes read
        total_bytes_written: Total bytes written
        dispatch_count: Number of dispatch operations
        time_samples: List of individual dispatch times for percentile calculation
    """

    expert_id: int
    total_selections: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    dispatch_count: int = 0
    time_samples: list[float] = field(default_factory=list)

    @property
    def avg_tokens_per_selection(self) -> float:
        """Average tokens per selection."""
        if self.total_selections == 0:
            return 0.0
        return self.total_tokens / self.total_selections

    @property
    def avg_time_ms(self) -> float:
        """Average dispatch time in milliseconds."""
        if self.dispatch_count == 0:
            return 0.0
        return self.total_time_ms / self.dispatch_count

    @property
    def bandwidth_gbs(self) -> float:
        """Average memory bandwidth in GB/s."""
        if self.total_time_ms == 0:
            return 0.0
        total_bytes = self.total_bytes_read + self.total_bytes_written
        return (total_bytes / 1e9) / (self.total_time_ms / 1000.0)

    @property
    def p50_time_ms(self) -> float:
        """50th percentile dispatch time."""
        if not self.time_samples:
            return 0.0
        sorted_times = sorted(self.time_samples)
        return sorted_times[len(sorted_times) // 2]

    @property
    def p95_time_ms(self) -> float:
        """95th percentile dispatch time."""
        if not self.time_samples:
            return 0.0
        sorted_times = sorted(self.time_samples)
        idx = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
        return sorted_times[idx]


@dataclass
class LayerExpertStats:
    """Expert statistics for a specific MoE layer."""

    layer_idx: int
    expert_stats: dict[int, ExpertStats] = field(default_factory=dict)
    total_dispatches: int = 0
    total_time_ms: float = 0.0

    def get_or_create_expert(self, expert_id: int) -> ExpertStats:
        """Get or create stats for an expert."""
        if expert_id not in self.expert_stats:
            self.expert_stats[expert_id] = ExpertStats(expert_id=expert_id)
        return self.expert_stats[expert_id]


class ExpertUsageProfiler:
    """Profile expert usage during MoE inference.

    Thread-safe profiler that tracks:
    - Per-expert selection frequency and load
    - Dispatch timing per expert
    - Memory bandwidth utilization
    - Load imbalance metrics

    Args:
        num_experts: Total number of experts per layer
        num_layers: Number of MoE layers to track (None = auto-detect)
        enabled: Whether profiling is active (default False for performance)
        max_time_samples: Max dispatch times to store per expert (for memory)
    """

    def __init__(
        self,
        num_experts: int = 64,
        num_layers: int | None = None,
        enabled: bool = False,
        max_time_samples: int = 1000,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self._enabled = enabled
        self._max_time_samples = max_time_samples

        # Per-layer stats (layer_idx -> LayerExpertStats)
        self._layer_stats: dict[int, LayerExpertStats] = {}

        # Global stats (aggregated across layers)
        self._global_expert_stats: dict[int, ExpertStats] = {}

        # Dispatch records for detailed analysis (limited size)
        self._dispatch_records: list[ExpertDispatchRecord] = []
        self._max_dispatch_records = 10000

        # Thread safety
        self._lock = threading.Lock()

        # Timing context for current dispatch
        self._current_dispatch_start: float | None = None
        self._current_layer_idx: int = 0

        # Config for memory bandwidth estimation
        self.hidden_dim: int = 4096  # Updated by set_model_config
        self.intermediate_dim: int = 14336
        self.bits: int = 4

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    def set_model_config(
        self,
        num_experts: int,
        num_layers: int,
        hidden_dim: int,
        intermediate_dim: int,
        bits: int = 4,
    ) -> None:
        """Configure model dimensions for bandwidth estimation.

        Args:
            num_experts: Number of experts per MoE layer
            num_layers: Number of MoE layers
            hidden_dim: Hidden dimension
            intermediate_dim: FFN intermediate dimension
            bits: Weight quantization bits
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.bits = bits

    def _estimate_bytes_for_dispatch(self, batch_size: int) -> tuple[int, int]:
        """Estimate bytes read and written for a single expert dispatch.

        For Trellis MoE with SwiGLU:
        - Read: activations + gate_weights + up_weights + down_weights + scales
        - Write: output

        Returns:
            Tuple of (bytes_read, bytes_written)
        """
        # Activations: [batch, hidden_dim] fp16
        activation_bytes = batch_size * self.hidden_dim * 2

        # Expert weights (quantized):
        # gate: [hidden_dim, intermediate_dim] at bits per weight
        # up: [hidden_dim, intermediate_dim]
        # down: [intermediate_dim, hidden_dim]
        weight_bytes_per_element = self.bits / 8
        gate_bytes = int(self.hidden_dim * self.intermediate_dim * weight_bytes_per_element)
        up_bytes = gate_bytes
        down_bytes = gate_bytes

        # Scales: much smaller, roughly hidden_dim * groups * 2
        scale_bytes = self.hidden_dim * 8 * 2  # Approximate

        bytes_read = activation_bytes + gate_bytes + up_bytes + down_bytes + scale_bytes

        # Output: [batch, hidden_dim] fp16
        bytes_written = batch_size * self.hidden_dim * 2

        return bytes_read, bytes_written

    def record_routing(
        self,
        layer_idx: int,
        expert_ids: np.ndarray | Any,
        expert_probs: np.ndarray | Any | None = None,
    ) -> None:
        """Record routing decisions (lightweight, no timing).

        Called during router forward to track expert selection frequency.
        This is separate from dispatch timing.

        Args:
            layer_idx: MoE layer index
            expert_ids: [batch, top_k] selected expert indices
            expert_probs: [batch, top_k] routing probabilities (optional)
        """
        if not self._enabled:
            return

        # Convert to numpy if needed
        if hasattr(expert_ids, "cpu"):
            expert_ids = expert_ids.cpu().numpy()
        elif hasattr(expert_ids, "numpy"):
            expert_ids = expert_ids.numpy()

        with self._lock:
            layer_stats = self._get_or_create_layer_stats(layer_idx)

            # Count selections per expert
            unique, counts = np.unique(expert_ids.ravel(), return_counts=True)
            for expert_id, count in zip(unique, counts):
                stats = layer_stats.get_or_create_expert(int(expert_id))
                stats.total_selections += int(count)
                stats.total_tokens += int(count)

                # Update global stats
                if expert_id not in self._global_expert_stats:
                    self._global_expert_stats[int(expert_id)] = ExpertStats(expert_id=int(expert_id))
                global_stats = self._global_expert_stats[int(expert_id)]
                global_stats.total_selections += int(count)
                global_stats.total_tokens += int(count)

    def record_dispatch(
        self,
        layer_idx: int,
        expert_ids: np.ndarray | Any,
        batch_size: int,
        dispatch_time_ms: float,
    ) -> None:
        """Record a completed MoE dispatch with timing.

        Called after dispatch_moe_trellis_swiglu completes.

        Args:
            layer_idx: MoE layer index
            expert_ids: [batch, top_k] selected expert indices
            batch_size: Total tokens in batch
            dispatch_time_ms: Wall-clock dispatch time
        """
        if not self._enabled:
            return

        # Convert to numpy if needed
        if hasattr(expert_ids, "cpu"):
            expert_ids = expert_ids.cpu().numpy()
        elif hasattr(expert_ids, "numpy"):
            expert_ids = expert_ids.numpy()

        timestamp_ns = time.time_ns()

        with self._lock:
            layer_stats = self._get_or_create_layer_stats(layer_idx)
            layer_stats.total_dispatches += 1
            layer_stats.total_time_ms += dispatch_time_ms

            # Count tokens per expert and record timing
            unique, counts = np.unique(expert_ids.ravel(), return_counts=True)
            num_unique_experts = len(unique)

            # Estimate time per expert (proportional to tokens)
            total_tokens = counts.sum()
            time_per_token = dispatch_time_ms / total_tokens if total_tokens > 0 else 0

            for expert_id, count in zip(unique, counts):
                expert_id = int(expert_id)
                expert_time = time_per_token * count

                # Estimate bytes
                bytes_read, bytes_written = self._estimate_bytes_for_dispatch(int(count))

                # Update layer stats
                stats = layer_stats.get_or_create_expert(expert_id)
                stats.dispatch_count += 1
                stats.total_time_ms += expert_time
                stats.total_bytes_read += bytes_read
                stats.total_bytes_written += bytes_written

                if len(stats.time_samples) < self._max_time_samples:
                    stats.time_samples.append(expert_time)

                # Update global stats
                if expert_id not in self._global_expert_stats:
                    self._global_expert_stats[expert_id] = ExpertStats(expert_id=expert_id)
                global_stats = self._global_expert_stats[expert_id]
                global_stats.dispatch_count += 1
                global_stats.total_time_ms += expert_time
                global_stats.total_bytes_read += bytes_read
                global_stats.total_bytes_written += bytes_written
                if len(global_stats.time_samples) < self._max_time_samples:
                    global_stats.time_samples.append(expert_time)

                # Store dispatch record
                if len(self._dispatch_records) < self._max_dispatch_records:
                    self._dispatch_records.append(
                        ExpertDispatchRecord(
                            expert_id=expert_id,
                            batch_size=int(count),
                            dispatch_time_ms=expert_time,
                            bytes_read=bytes_read,
                            bytes_written=bytes_written,
                            timestamp_ns=timestamp_ns,
                        )
                    )

    @contextmanager
    def time_dispatch(self, layer_idx: int = 0) -> Iterator[None]:
        """Context manager to time a dispatch operation.

        Usage:
            with profiler.time_dispatch(layer_idx):
                output = dispatch_moe_trellis_swiglu(...)
            profiler.record_dispatch(layer_idx, expert_ids, batch_size, elapsed)

        Actually, this just times and stores the elapsed. Use record_dispatch_timed instead.
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        self._current_dispatch_start = start
        self._current_layer_idx = layer_idx
        try:
            yield
        finally:
            # Elapsed stored; caller should call record_dispatch
            pass

    def record_dispatch_timed(
        self,
        layer_idx: int,
        expert_ids: np.ndarray | Any,
        batch_size: int,
    ) -> None:
        """Record dispatch using time from time_dispatch context.

        Must be called within or immediately after time_dispatch context.
        """
        if not self._enabled or self._current_dispatch_start is None:
            return

        elapsed = (time.perf_counter() - self._current_dispatch_start) * 1000.0
        self._current_dispatch_start = None

        self.record_dispatch(layer_idx, expert_ids, batch_size, elapsed)

    def _get_or_create_layer_stats(self, layer_idx: int) -> LayerExpertStats:
        """Get or create stats for a layer."""
        if layer_idx not in self._layer_stats:
            self._layer_stats[layer_idx] = LayerExpertStats(layer_idx=layer_idx)
        return self._layer_stats[layer_idx]

    def get_expert_stats(self, expert_id: int, layer_idx: int | None = None) -> ExpertStats | None:
        """Get statistics for a specific expert.

        Args:
            expert_id: Expert index
            layer_idx: If provided, get layer-specific stats. If None, get global.

        Returns:
            ExpertStats or None if not found
        """
        with self._lock:
            if layer_idx is not None:
                if layer_idx in self._layer_stats:
                    return self._layer_stats[layer_idx].expert_stats.get(expert_id)
                return None
            return self._global_expert_stats.get(expert_id)

    def get_hot_experts(self, threshold: float = 1.5, layer_idx: int | None = None) -> list[int]:
        """Get experts with above-average load.

        Args:
            threshold: Multiplier above average to be considered "hot"
            layer_idx: If provided, analyze specific layer. If None, use global.

        Returns:
            List of expert IDs with load > threshold * average
        """
        with self._lock:
            if layer_idx is not None and layer_idx in self._layer_stats:
                stats_dict = self._layer_stats[layer_idx].expert_stats
            else:
                stats_dict = self._global_expert_stats

            if not stats_dict:
                return []

            total_selections = sum(s.total_selections for s in stats_dict.values())
            avg_selections = total_selections / max(len(stats_dict), 1)

            return [
                expert_id
                for expert_id, stats in stats_dict.items()
                if stats.total_selections > threshold * avg_selections
            ]

    def get_cold_experts(self, threshold: float = 0.5, layer_idx: int | None = None) -> list[int]:
        """Get experts with below-average load.

        Args:
            threshold: Multiplier below average to be considered "cold"
            layer_idx: If provided, analyze specific layer. If None, use global.

        Returns:
            List of expert IDs with load < threshold * average
        """
        with self._lock:
            if layer_idx is not None and layer_idx in self._layer_stats:
                stats_dict = self._layer_stats[layer_idx].expert_stats
            else:
                stats_dict = self._global_expert_stats

            if not stats_dict:
                return []

            total_selections = sum(s.total_selections for s in stats_dict.values())
            avg_selections = total_selections / max(len(stats_dict), 1)

            return [
                expert_id
                for expert_id, stats in stats_dict.items()
                if 0 < stats.total_selections < threshold * avg_selections
            ]

    def get_dead_experts(self, layer_idx: int | None = None) -> list[int]:
        """Get experts that were never selected.

        Args:
            layer_idx: If provided, check specific layer. If None, check global.

        Returns:
            List of expert IDs with zero selections
        """
        with self._lock:
            if layer_idx is not None and layer_idx in self._layer_stats:
                stats_dict = self._layer_stats[layer_idx].expert_stats
            else:
                stats_dict = self._global_expert_stats

            seen_experts = set(stats_dict.keys())
            all_experts = set(range(self.num_experts))
            return sorted(all_experts - seen_experts)

    def get_load_imbalance(self, layer_idx: int | None = None) -> dict[str, float]:
        """Compute load imbalance metrics.

        Args:
            layer_idx: If provided, analyze specific layer. If None, use global.

        Returns:
            Dictionary with:
            - cv: Coefficient of variation (lower = more balanced)
            - gini: Gini coefficient (0 = perfect equality)
            - max_min_ratio: Ratio of max to min load
        """
        with self._lock:
            if layer_idx is not None and layer_idx in self._layer_stats:
                stats_dict = self._layer_stats[layer_idx].expert_stats
            else:
                stats_dict = self._global_expert_stats

            if not stats_dict:
                return {"cv": 0.0, "gini": 0.0, "max_min_ratio": 1.0}

            loads = np.array([s.total_selections for s in stats_dict.values()])
            if len(loads) == 0 or loads.sum() == 0:
                return {"cv": 0.0, "gini": 0.0, "max_min_ratio": 1.0}

            # Coefficient of variation
            cv = float(np.std(loads) / np.mean(loads)) if np.mean(loads) > 0 else 0.0

            # Gini coefficient
            sorted_loads = np.sort(loads)
            n = len(sorted_loads)
            cumsum = np.cumsum(sorted_loads)
            gini = float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n) if cumsum[-1] > 0 else 0.0

            # Max/min ratio
            min_load = loads[loads > 0].min() if (loads > 0).any() else 1
            max_min_ratio = float(loads.max() / min_load)

            return {"cv": cv, "gini": gini, "max_min_ratio": max_min_ratio}

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive usage report.

        Returns:
            Dictionary with all profiling results.
        """
        with self._lock:
            total_dispatches = sum(ls.total_dispatches for ls in self._layer_stats.values())
            total_time_ms = sum(ls.total_time_ms for ls in self._layer_stats.values())

            # Global expert statistics
            expert_data = []
            for expert_id in sorted(self._global_expert_stats.keys()):
                stats = self._global_expert_stats[expert_id]
                expert_data.append({
                    "expert_id": expert_id,
                    "selections": stats.total_selections,
                    "tokens": stats.total_tokens,
                    "dispatches": stats.dispatch_count,
                    "total_time_ms": round(stats.total_time_ms, 3),
                    "avg_time_ms": round(stats.avg_time_ms, 3),
                    "p50_time_ms": round(stats.p50_time_ms, 3),
                    "p95_time_ms": round(stats.p95_time_ms, 3),
                    "bandwidth_gbs": round(stats.bandwidth_gbs, 2),
                })

            # Per-layer statistics
            layer_data = []
            for layer_idx in sorted(self._layer_stats.keys()):
                ls = self._layer_stats[layer_idx]
                imbalance = self.get_load_imbalance(layer_idx)
                layer_data.append({
                    "layer": layer_idx,
                    "dispatches": ls.total_dispatches,
                    "total_time_ms": round(ls.total_time_ms, 3),
                    "active_experts": len(ls.expert_stats),
                    "load_cv": round(imbalance["cv"], 4),
                    "load_gini": round(imbalance["gini"], 4),
                })

            # Global imbalance
            global_imbalance = self.get_load_imbalance()

            return {
                "summary": {
                    "num_experts": self.num_experts,
                    "num_layers_profiled": len(self._layer_stats),
                    "total_dispatches": total_dispatches,
                    "total_time_ms": round(total_time_ms, 3),
                    "dispatch_records_stored": len(self._dispatch_records),
                },
                "load_balance": {
                    "hot_experts": self.get_hot_experts(),
                    "cold_experts": self.get_cold_experts(),
                    "dead_experts": self.get_dead_experts(),
                    "cv": round(global_imbalance["cv"], 4),
                    "gini": round(global_imbalance["gini"], 4),
                    "max_min_ratio": round(global_imbalance["max_min_ratio"], 2),
                },
                "per_expert": expert_data,
                "per_layer": layer_data,
            }

    def print_summary(self) -> None:
        """Print human-readable summary to stdout."""
        report = self.generate_report()
        summary = report["summary"]
        balance = report["load_balance"]

        print("\n" + "=" * 70)
        print("EXPERT USAGE PROFILER REPORT")
        print("=" * 70)

        print("\nConfiguration:")
        print(f"  Experts: {summary['num_experts']}")
        print(f"  Layers profiled: {summary['num_layers_profiled']}")
        print(f"  Total dispatches: {summary['total_dispatches']:,}")
        print(f"  Total time: {summary['total_time_ms']:.1f} ms")

        print("\nLoad Balance:")
        print(f"  CV (coefficient of variation): {balance['cv']:.4f}")
        print(f"  Gini coefficient: {balance['gini']:.4f}")
        print(f"  Max/min ratio: {balance['max_min_ratio']:.1f}x")

        print("\nExpert Distribution:")
        hot = balance["hot_experts"]
        cold = balance["cold_experts"]
        dead = balance["dead_experts"]
        print(f"  Hot experts (>1.5x avg): {len(hot)} - {hot[:10]}{'...' if len(hot) > 10 else ''}")
        print(f"  Cold experts (<0.5x avg): {len(cold)} - {cold[:10]}{'...' if len(cold) > 10 else ''}")
        print(f"  Dead experts (never used): {len(dead)} - {dead[:10]}{'...' if len(dead) > 10 else ''}")

        # Top 10 experts by time
        if report["per_expert"]:
            print("\nTop 10 Experts by Compute Time:")
            sorted_experts = sorted(report["per_expert"], key=lambda x: x["total_time_ms"], reverse=True)
            print(f"  {'ID':>4} {'Selections':>10} {'Time(ms)':>10} {'Avg(ms)':>10} {'BW(GB/s)':>10}")
            print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for e in sorted_experts[:10]:
                print(f"  {e['expert_id']:>4} {e['selections']:>10} {e['total_time_ms']:>10.1f} "
                      f"{e['avg_time_ms']:>10.3f} {e['bandwidth_gbs']:>10.1f}")

        print()

    def save_report(self, path: str | Path) -> None:
        """Save report to JSON file."""
        report = self.generate_report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._layer_stats.clear()
            self._global_expert_stats.clear()
            self._dispatch_records.clear()


# Global singleton instance
_global_expert_profiler = ExpertUsageProfiler()


def get_global_expert_profiler() -> ExpertUsageProfiler:
    """Get the global expert usage profiler instance."""
    return _global_expert_profiler


def reset_global_expert_profiler() -> None:
    """Reset the global profiler."""
    _global_expert_profiler.clear()


__all__ = [
    "ExpertUsageProfiler",
    "ExpertStats",
    "ExpertDispatchRecord",
    "get_global_expert_profiler",
    "reset_global_expert_profiler",
]
