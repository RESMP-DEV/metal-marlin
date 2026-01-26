"""Metal GPU performance counter reading.

This module provides access to Metal GPU performance counters on macOS.
It uses pyobjc to access the Metal framework's MTLCounterSampleBuffer API
when available, with fallback to timing-only metrics.

Metal Performance Counters (Apple Silicon):
- GPU Core Utilization (%)
- Memory Bandwidth (read/write GB/s)
- Shader ALU Utilization (%)
- Cache Hit Rates (L1/L2)
- Threadgroup Occupancy

Note:
    Full counter access requires:
    1. macOS 11.0+ (Big Sur)
    2. pyobjc-framework-Metal installed
    3. Appropriate entitlements for system-level access

    Without these, the module provides estimated metrics based on timing
    and workload characteristics.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._compat import HAS_PYOBJC_METAL, Metal
from .occupancy import detect_gpu
from .trace import TraceEvent

HAS_METAL_COUNTERS = HAS_PYOBJC_METAL and platform.system() == "Darwin"
MTLDevice: Any = None
MTLCounterSet: Any = None


@dataclass
class GPUCounters:
    """GPU performance counter snapshot.

    All utilization values are percentages (0-100).
    Bandwidth values are in GB/s.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds.
        gpu_utilization: Overall GPU utilization (%).
        alu_utilization: Shader ALU utilization (%).
        memory_read_bandwidth: Memory read bandwidth (GB/s).
        memory_write_bandwidth: Memory write bandwidth (GB/s).
        l1_cache_hit_rate: L1 cache hit rate (%).
        l2_cache_hit_rate: L2 cache hit rate (%).
        threadgroup_occupancy: Threadgroup occupancy (%).
        active_simdgroups: Number of active SIMD groups.
        stall_memory: Memory stall percentage (%).
        stall_sync: Synchronization stall percentage (%).
        stall_alu: ALU stall percentage (%).
        raw_counters: Raw counter values from Metal API.
    """

    timestamp_ns: int = 0
    gpu_utilization: float = 0.0
    alu_utilization: float = 0.0
    memory_read_bandwidth: float = 0.0
    memory_write_bandwidth: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    l1_cache_hit_rate: float = 0.0
    l2_cache_hit_rate: float = 0.0
    threadgroup_occupancy: float = 0.0
    active_simdgroups: int = 0
    stall_memory: float = 0.0
    stall_sync: float = 0.0
    stall_alu: float = 0.0
    raw_counters: dict[str, Any] = field(default_factory=dict)

    @property
    def total_memory_bandwidth(self) -> float:
        """Combined read + write bandwidth (GB/s)."""
        return self.memory_read_bandwidth + self.memory_write_bandwidth

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "gpu_utilization": self.gpu_utilization,
            "alu_utilization": self.alu_utilization,
            "memory_read_bandwidth": self.memory_read_bandwidth,
            "memory_write_bandwidth": self.memory_write_bandwidth,
            "memory_bandwidth_utilization": self.memory_bandwidth_utilization,
            "l1_cache_hit_rate": self.l1_cache_hit_rate,
            "l2_cache_hit_rate": self.l2_cache_hit_rate,
            "threadgroup_occupancy": self.threadgroup_occupancy,
            "active_simdgroups": self.active_simdgroups,
            "stall_memory": self.stall_memory,
            "stall_sync": self.stall_sync,
            "stall_alu": self.stall_alu,
        }

    def to_trace_event(self, *, pid: int = 0, tid: int = 0) -> TraceEvent:
        """Convert to a Chrome trace counter event."""
        return TraceEvent(
            name="gpu_counters",
            cat="gpu",
            ph="C",
            ts=int(self.timestamp_ns / 1000),
            pid=pid,
            tid=tid,
            args={
                "gpu_utilization_pct": self.gpu_utilization,
                "alu_utilization_pct": self.alu_utilization,
                "memory_read_gbs": self.memory_read_bandwidth,
                "memory_write_gbs": self.memory_write_bandwidth,
                "memory_bandwidth_utilization_pct": self.memory_bandwidth_utilization,
                "l1_cache_hit_rate_pct": self.l1_cache_hit_rate,
                "l2_cache_hit_rate_pct": self.l2_cache_hit_rate,
                "threadgroup_occupancy_pct": self.threadgroup_occupancy,
                "stall_memory_pct": self.stall_memory,
                "stall_sync_pct": self.stall_sync,
                "stall_alu_pct": self.stall_alu,
            },
        )


@dataclass
class GPUPerformanceState:
    """GPU performance state snapshot from system telemetry."""

    timestamp_ns: int
    frequency_mhz: float = 0.0
    active_pct: float = 0.0
    power_w: float = 0.0
    temperature_c: float = 0.0

    def to_trace_event(self, *, pid: int = 0, tid: int = 0) -> TraceEvent:
        return TraceEvent(
            name="gpu_performance_state",
            cat="gpu_state",
            ph="C",
            ts=int(self.timestamp_ns / 1000),
            pid=pid,
            tid=tid,
            args={
                "frequency_mhz": self.frequency_mhz,
                "active_pct": self.active_pct,
                "power_w": self.power_w,
                "temperature_c": self.temperature_c,
            },
        )


def _get_powermetrics_gpu_stats() -> dict[str, float]:
    """Query GPU stats via powermetrics (requires sudo).

    Returns partial stats if available, empty dict on failure.
    """
    if platform.system() != "Darwin":
        return {}

    try:
        # powermetrics requires root, so this may fail
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "-s", "gpu_power", "-n", "1", "-i", "100"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        output = result.stdout

        stats: dict[str, float] = {}

        # Parse GPU utilization lines
        for line in output.split("\n"):
            if "GPU Active" in line:
                # Format: "GPU Active: 45%"
                try:
                    pct = float(line.split(":")[1].strip().rstrip("%"))
                    stats["gpu_utilization"] = pct
                except (IndexError, ValueError):
                    pass
            elif "GPU Frequency" in line:
                try:
                    freq = float(line.split(":")[1].strip().split()[0])
                    stats["gpu_frequency_mhz"] = freq
                except (IndexError, ValueError):
                    pass
            elif "GPU Power" in line or "GPU Average Power" in line:
                match = re.search(r"([0-9]+(?:\\.[0-9]+)?)\\s*mW", line)
                if match:
                    try:
                        stats["gpu_power_w"] = float(match.group(1)) / 1000.0
                    except ValueError:
                        pass
            elif "GPU Temperature" in line:
                match = re.search(r"([0-9]+(?:\\.[0-9]+)?)\\s*C", line)
                if match:
                    try:
                        stats["gpu_temperature_c"] = float(match.group(1))
                    except ValueError:
                        pass

        return stats

    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return {}


def _estimate_counters_from_timing(
    elapsed_ms: float,
    flops: float,
    bytes_moved: float,
    peak_tflops: float = 32.0,
    peak_bw_gbs: float = 546.0,
) -> GPUCounters:
    """Estimate GPU counters from timing and workload characteristics.

    This provides approximate metrics when direct counter access is unavailable.
    Assumes the kernel is either compute-bound or memory-bound.

    Args:
        elapsed_ms: Kernel execution time in milliseconds.
        flops: Total floating-point operations.
        bytes_moved: Total bytes read + written.
        peak_tflops: Hardware peak TFLOPS (M4 Max default: 32).
        peak_bw_gbs: Hardware peak memory bandwidth (M4 Max default: 546).

    Returns:
        GPUCounters with estimated values.
    """
    elapsed_s = elapsed_ms / 1000.0

    if elapsed_s <= 0:
        return GPUCounters(timestamp_ns=time.time_ns())

    achieved_tflops = (flops / elapsed_s) / 1e12
    achieved_bw = (bytes_moved / elapsed_s) / 1e9

    # Estimate utilization as fraction of peak
    alu_util = min(100.0, (achieved_tflops / peak_tflops) * 100.0)
    bw_util = min(100.0, (achieved_bw / peak_bw_gbs) * 100.0)

    # GPU utilization is the max of compute and memory utilization
    gpu_util = max(alu_util, bw_util)

    # Estimate stalls based on which is the bottleneck
    if alu_util > bw_util:
        # Compute-bound: likely memory stalls
        stall_memory = max(0, 100 - bw_util) * 0.5
        stall_alu = 0.0
    else:
        # Memory-bound: likely ALU stalls waiting for data
        stall_alu = max(0, 100 - alu_util) * 0.3
        stall_memory = 0.0

    return GPUCounters(
        timestamp_ns=time.time_ns(),
        gpu_utilization=gpu_util,
        alu_utilization=alu_util,
        memory_read_bandwidth=achieved_bw * 0.7,  # Estimate 70% reads
        memory_write_bandwidth=achieved_bw * 0.3,  # Estimate 30% writes
        memory_bandwidth_utilization=bw_util,
        stall_memory=stall_memory,
        stall_alu=stall_alu,
    )


def _extract_raw_counter_values(sample_buffer: Any) -> dict[str, float]:
    """Best-effort extraction of raw counter values from a Metal sample buffer."""
    if sample_buffer is None:
        return {}

    if hasattr(sample_buffer, "resolveCounterRange_"):
        try:
            sample_buffer.resolveCounterRange_((0, 1))
        except Exception:
            return {}

    values: dict[str, float] = {}
    if hasattr(sample_buffer, "counterData"):
        try:
            raw = sample_buffer.counterData()
            if isinstance(raw, dict):
                for key, value in raw.items():
                    try:
                        values[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
        except Exception:
            return {}
    return values


def _find_counter(raw: dict[str, float], *keywords: str) -> float | None:
    for key, value in raw.items():
        lowered = key.lower()
        if all(word in lowered for word in keywords):
            return value
    return None


def _compute_hit_rate(hit: float | None, miss: float | None) -> float:
    if hit is None or miss is None:
        return 0.0
    total = hit + miss
    if total <= 0:
        return 0.0
    return (hit / total) * 100.0


class GPUProfiler:
    """GPU profiler with Metal counter support.

    Provides access to GPU performance counters when available,
    with fallback to timing-based estimates.

    Args:
        device: Metal device to profile. If None, uses system default.
        enable_counters: Whether to attempt counter sampling (default True).

    Example:
        profiler = GPUProfiler()

        profiler.start_capture()
        result = marlin_gemm_fp4(A, B, scales)
        counters = profiler.stop_capture()

        print(f"GPU Utilization: {counters.gpu_utilization:.1f}%")
    """

    def __init__(
        self,
        device: Any = None,
        *,
        enable_counters: bool = True,
        peak_bandwidth_gbs: float | None = None,
    ):
        self._device = device
        self._enable_counters = enable_counters and HAS_METAL_COUNTERS
        self._peak_bandwidth_gbs = peak_bandwidth_gbs or detect_gpu().peak_bw_gbs
        self._counter_sets: list[Any] = []
        self._sample_buffer: Any = None
        self._capture_start_ns: int = 0
        self._captures: list[GPUCounters] = []

        # Initialize Metal device and counter sets
        if self._enable_counters:
            self._init_metal_counters()

    def _init_metal_counters(self) -> None:
        """Initialize Metal counter sampling infrastructure."""
        if not HAS_METAL_COUNTERS:
            return

        try:
            if self._device is None:
                if Metal is None:
                    self._enable_counters = False
                    return
                self._device = Metal.MTLCreateSystemDefaultDevice()

            if self._device is None:
                self._enable_counters = False
                return

            # Query available counter sets
            # This requires macOS 11+ and Metal 2.4+
            counter_sets = self._device.counterSets()
            if counter_sets:
                self._counter_sets = list(counter_sets)

        except Exception:
            self._enable_counters = False

    @property
    def counters_available(self) -> bool:
        """Whether direct GPU counter access is available."""
        return self._enable_counters and len(self._counter_sets) > 0

    def list_counter_sets(self) -> list[str]:
        """List available counter set names."""
        if not self._enable_counters:
            return []

        names: list[str] = []
        for cs in self._counter_sets:
            try:
                names.append(cs.name())
            except Exception:
                pass
        return names

    def start_capture(self) -> None:
        """Start GPU counter capture.

        Call stop_capture() after kernel execution to retrieve counters.
        """
        self._capture_start_ns = time.time_ns()

        if self._enable_counters and self._device is not None:
            # Create counter sample buffer for this capture
            # Note: Full implementation requires MTLCounterSampleBufferDescriptor
            # which may not be fully exposed via pyobjc
            pass

    def stop_capture(
        self,
        flops: float = 0,
        bytes_moved: float = 0,
    ) -> GPUCounters:
        """Stop capture and return GPU counters.

        Args:
            flops: Total FLOPs for estimate fallback.
            bytes_moved: Total bytes for estimate fallback.

        Returns:
            GPUCounters snapshot.
        """
        end_ns = time.time_ns()
        elapsed_ms = (end_ns - self._capture_start_ns) / 1e6

        counters: GPUCounters

        if self._enable_counters and self._sample_buffer is not None:
            # Read actual counters from Metal
            counters = self._read_metal_counters()
        else:
            # Fall back to timing-based estimates
            counters = _estimate_counters_from_timing(
                elapsed_ms=elapsed_ms,
                flops=flops,
                bytes_moved=bytes_moved,
            )

        # Try to augment with powermetrics data
        pm_stats = _get_powermetrics_gpu_stats()
        if "gpu_utilization" in pm_stats:
            counters.gpu_utilization = pm_stats["gpu_utilization"]
        if "gpu_frequency_mhz" in pm_stats:
            counters.raw_counters["gpu_frequency_mhz"] = pm_stats["gpu_frequency_mhz"]
        if "gpu_power_w" in pm_stats:
            counters.raw_counters["gpu_power_w"] = pm_stats["gpu_power_w"]
        if "gpu_temperature_c" in pm_stats:
            counters.raw_counters["gpu_temperature_c"] = pm_stats["gpu_temperature_c"]

        if counters.total_memory_bandwidth > 0 and self._peak_bandwidth_gbs > 0:
            counters.memory_bandwidth_utilization = min(
                100.0,
                (counters.total_memory_bandwidth / self._peak_bandwidth_gbs) * 100.0,
            )

        self._captures.append(counters)
        return counters

    def _read_metal_counters(self) -> GPUCounters:
        """Read counters from Metal sample buffer."""
        # This is a placeholder for full Metal counter implementation
        # The actual implementation would:
        # 1. Resolve the counter sample buffer
        # 2. Parse counter values by name
        # 3. Map to GPUCounters fields
        counters = GPUCounters(timestamp_ns=time.time_ns())
        raw = _extract_raw_counter_values(self._sample_buffer)
        if not raw:
            return counters

        counters.raw_counters = raw
        l1_hit = _find_counter(raw, "l1", "hit")
        l1_miss = _find_counter(raw, "l1", "miss")
        l2_hit = _find_counter(raw, "l2", "hit")
        l2_miss = _find_counter(raw, "l2", "miss")
        counters.l1_cache_hit_rate = _compute_hit_rate(l1_hit, l1_miss)
        counters.l2_cache_hit_rate = _compute_hit_rate(l2_hit, l2_miss)

        alu_util = _find_counter(raw, "alu", "util")
        if alu_util is not None:
            counters.alu_utilization = alu_util

        mem_read = _find_counter(raw, "memory", "read")
        mem_write = _find_counter(raw, "memory", "write")
        if mem_read is not None:
            counters.memory_read_bandwidth = mem_read
        if mem_write is not None:
            counters.memory_write_bandwidth = mem_write

        stall_mem = _find_counter(raw, "stall", "memory")
        stall_sync = _find_counter(raw, "stall", "sync")
        stall_alu = _find_counter(raw, "stall", "alu")
        counters.stall_memory = stall_mem or 0.0
        counters.stall_sync = stall_sync or 0.0
        counters.stall_alu = stall_alu or 0.0

        return counters

    @property
    def captures(self) -> list[GPUCounters]:
        """All captured counter snapshots."""
        return list(self._captures)

    def clear(self) -> None:
        """Clear captured data."""
        self._captures.clear()

    def print_summary(self) -> None:
        """Print summary of captured counters."""
        if not self._captures:
            print("No captures recorded")
            return

        print(f"{'Capture':<10} {'GPU%':>8} {'ALU%':>8} {'BW(GB/s)':>10} {'Stall%':>8}")
        print("-" * 50)

        for i, c in enumerate(self._captures):
            total_stall = c.stall_memory + c.stall_sync + c.stall_alu
            print(
                f"{i:<10} {c.gpu_utilization:>8.1f} {c.alu_utilization:>8.1f} "
                f"{c.total_memory_bandwidth:>10.1f} {total_stall:>8.1f}"
            )


def read_gpu_counters(
    flops: float = 0,
    bytes_moved: float = 0,
) -> GPUCounters:
    """Quick counter read using timing estimates.

    This is a convenience function for getting approximate GPU metrics
    without setting up a full GPUProfiler session.

    Args:
        flops: FLOPs if known (for compute utilization estimate).
        bytes_moved: Bytes if known (for bandwidth estimate).

    Returns:
        GPUCounters with available metrics.
    """
    counters = GPUCounters(timestamp_ns=time.time_ns())

    # Try powermetrics for GPU utilization
    pm_stats = _get_powermetrics_gpu_stats()
    if pm_stats:
        counters.gpu_utilization = pm_stats.get("gpu_utilization", 0.0)
        counters.raw_counters.update(pm_stats)

    return counters


def read_gpu_performance_state() -> GPUPerformanceState | None:
    """Read current GPU performance state from system telemetry."""
    stats = _get_powermetrics_gpu_stats()
    if not stats:
        return None

    return GPUPerformanceState(
        timestamp_ns=time.time_ns(),
        frequency_mhz=stats.get("gpu_frequency_mhz", 0.0),
        active_pct=stats.get("gpu_utilization", 0.0),
        power_w=stats.get("gpu_power_w", 0.0),
        temperature_c=stats.get("gpu_temperature_c", 0.0),
    )


class MetalSystemTraceRecorder:
    """Record Metal System Trace and GPU Performance State via xctrace."""

    def __init__(
        self,
        *,
        template: str = "Metal System Trace",
        output_dir: str | Path = "metal_traces",
    ) -> None:
        self.template = template
        self.output_dir = Path(output_dir)

    def record_command(
        self,
        command: list[str],
        *,
        duration_s: float = 2.0,
    ) -> Path:
        """Record a system trace while running the given command."""
        if platform.system() != "Darwin":
            raise RuntimeError("Metal system trace requires macOS")

        if shutil.which("xcrun") is None:
            raise RuntimeError("xcrun not found. Install Xcode command line tools.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / f"metal_trace_{int(time.time())}.trace"

        record_cmd = [
            "xcrun",
            "xctrace",
            "record",
            "--template",
            self.template,
            "--time-limit",
            f"{duration_s}s",
            "--output",
            str(trace_path),
            "--launch",
        ]
        record_cmd.extend(command)

        subprocess.run(record_cmd, check=True)
        return trace_path

    def record_with_template(
        self,
        command: list[str],
        *,
        template: str,
        duration_s: float = 2.0,
    ) -> Path:
        """Record using a specific Instruments template."""
        original = self.template
        try:
            self.template = template
            return self.record_command(command, duration_s=duration_s)
        finally:
            self.template = original

    def record_gpu_performance_state(
        self,
        command: list[str],
        *,
        duration_s: float = 2.0,
    ) -> Path:
        """Record GPU Performance State trace while running a command."""
        return self.record_with_template(
            command,
            template="GPU Performance State",
            duration_s=duration_s,
        )

    def export_trace_json(self, trace_path: str | Path) -> Path:
        """Export an xctrace recording to JSON for external analysis."""
        if shutil.which("xcrun") is None:
            raise RuntimeError("xcrun not found. Install Xcode command line tools.")

        trace_path = Path(trace_path)
        output_json = trace_path.with_suffix(".json")

        export_cmd = [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            str(trace_path),
            "--output",
            str(output_json),
            "--format",
            "json",
        ]
        subprocess.run(export_cmd, check=True)
        return output_json

    def export_trace_events(self, trace_path: str | Path) -> list[dict[str, Any]]:
        """Load exported trace JSON if available."""
        json_path = self.export_trace_json(trace_path)
        with open(json_path) as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "traceEvents" in payload:
            return payload["traceEvents"]
        if isinstance(payload, list):
            return payload
        return []


def get_gpu_info() -> dict[str, Any]:
    """Get information about the GPU.

    Returns:
        Dictionary with GPU name, memory, core count, etc.
    """
    info: dict[str, Any] = {
        "platform": platform.system(),
        "counters_available": HAS_METAL_COUNTERS,
    }

    if platform.system() == "Darwin":
        # Query GPU info via system_profiler
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            import json

            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu = displays[0]
                info["gpu_name"] = gpu.get("sppci_model", "Unknown")
                info["gpu_cores"] = gpu.get("sppci_cores", "Unknown")
                info["metal_family"] = gpu.get("spmetal_family", "Unknown")
                info["vram_mb"] = gpu.get("spdisplays_vram", "Unified")
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        # Query memory bandwidth via sysctl
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            mem_bytes = int(result.stdout.strip())
            info["unified_memory_gb"] = mem_bytes / (1024**3)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

    return info
