"""Threadgroup occupancy analysis for Metal kernels.

Calculates theoretical and achievable occupancy based on:
- Threadgroup size and SIMD group configuration
- Threadgroup memory usage
- Register pressure
- Hardware limits (Apple Silicon GPU specs)

Apple Silicon GPU Architecture:
- SIMD width: 32 threads
- Max threads per threadgroup: 1024
- Max threadgroups per execution unit varies by chip
- Threadgroup memory limit: 32KB (typical)
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .trace import TraceEvent


class AppleSiliconGPU(Enum):
    """Apple Silicon GPU variants with their specs."""

    # Format: (gpu_cores, max_tg_per_eu, peak_tflops_fp16, peak_bw_gbs)
    M1 = (8, 32, 2.6, 68.3)
    M1_PRO = (16, 32, 5.2, 200)
    M1_MAX = (32, 32, 10.4, 400)
    M1_ULTRA = (64, 32, 20.8, 800)
    M2 = (10, 32, 3.6, 100)
    M2_PRO = (19, 32, 6.8, 200)
    M2_MAX = (38, 32, 13.6, 400)
    M2_ULTRA = (76, 32, 27.2, 800)
    M3 = (10, 32, 4.1, 100)
    M3_PRO = (18, 32, 7.4, 150)
    M3_MAX = (40, 32, 16.4, 400)
    M3_ULTRA = (80, 32, 32.8, 800)
    M4 = (10, 32, 4.3, 120)
    M4_PRO = (20, 32, 8.6, 273)
    M4_MAX = (40, 32, 32.0, 546)
    UNKNOWN = (16, 32, 5.0, 200)  # Conservative default

    @property
    def gpu_cores(self) -> int:
        return self.value[0]

    @property
    def max_tg_per_eu(self) -> int:
        return self.value[1]

    @property
    def peak_tflops_fp16(self) -> float:
        return self.value[2]

    @property
    def peak_bw_gbs(self) -> float:
        return self.value[3]


@dataclass(frozen=True)
class ThreadgroupConfig:
    """Metal kernel threadgroup configuration.

    Attributes:
        threads_per_tg: Total threads per threadgroup.
        simdgroups_per_tg: Number of SIMD groups (threads_per_tg / 32).
        threadgroup_memory_bytes: Shared memory per threadgroup.
        registers_per_thread: Estimated registers per thread.
    """

    threads_per_tg: int
    simdgroups_per_tg: int = 0
    threadgroup_memory_bytes: int = 0
    registers_per_thread: int = 32  # Conservative estimate

    def __post_init__(self) -> None:
        # Auto-compute simdgroups if not specified
        if self.simdgroups_per_tg == 0:
            object.__setattr__(
                self, "simdgroups_per_tg", (self.threads_per_tg + 31) // 32
            )


@dataclass
class OccupancyMetrics:
    """Occupancy analysis results.

    Attributes:
        theoretical_occupancy: Maximum possible occupancy (%).
        achieved_occupancy: Estimated achievable occupancy (%).
        limiting_factor: What's limiting occupancy.
        max_tg_per_eu: Max threadgroups per execution unit.
        max_threads_per_eu: Max concurrent threads per EU.
        threads_per_eu: Actual threads per EU with this config.
        recommendations: Suggestions for improving occupancy.
        details: Additional analysis details.
    """

    theoretical_occupancy: float
    achieved_occupancy: float
    limiting_factor: str
    max_tg_per_eu: int
    max_threads_per_eu: int
    threads_per_eu: int
    recommendations: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "theoretical_occupancy": self.theoretical_occupancy,
            "achieved_occupancy": self.achieved_occupancy,
            "limiting_factor": self.limiting_factor,
            "max_tg_per_eu": self.max_tg_per_eu,
            "max_threads_per_eu": self.max_threads_per_eu,
            "threads_per_eu": self.threads_per_eu,
            "recommendations": self.recommendations,
            "details": self.details,
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
            name="threadgroup_occupancy",
            cat="occupancy",
            ph="C",
            ts=int(timestamp_ns / 1000),
            pid=pid,
            tid=tid,
            args={
                "theoretical_occupancy_pct": self.theoretical_occupancy,
                "achieved_occupancy_pct": self.achieved_occupancy,
                "limiting_factor": self.limiting_factor,
                "max_tg_per_eu": self.max_tg_per_eu,
                "threads_per_eu": self.threads_per_eu,
            },
        )


def detect_gpu() -> AppleSiliconGPU:
    """Detect the current Apple Silicon GPU.

    Returns:
        AppleSiliconGPU enum variant.
    """
    if platform.system() != "Darwin":
        return AppleSiliconGPU.UNKNOWN

    try:
        # Query chip name via sysctl
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        brand = result.stdout.strip().lower()

        # Match GPU variant
        if "m4 max" in brand:
            return AppleSiliconGPU.M4_MAX
        elif "m4 pro" in brand:
            return AppleSiliconGPU.M4_PRO
        elif "m4" in brand:
            return AppleSiliconGPU.M4
        elif "m3 ultra" in brand:
            return AppleSiliconGPU.M3_ULTRA
        elif "m3 max" in brand:
            return AppleSiliconGPU.M3_MAX
        elif "m3 pro" in brand:
            return AppleSiliconGPU.M3_PRO
        elif "m3" in brand:
            return AppleSiliconGPU.M3
        elif "m2 ultra" in brand:
            return AppleSiliconGPU.M2_ULTRA
        elif "m2 max" in brand:
            return AppleSiliconGPU.M2_MAX
        elif "m2 pro" in brand:
            return AppleSiliconGPU.M2_PRO
        elif "m2" in brand:
            return AppleSiliconGPU.M2
        elif "m1 ultra" in brand:
            return AppleSiliconGPU.M1_ULTRA
        elif "m1 max" in brand:
            return AppleSiliconGPU.M1_MAX
        elif "m1 pro" in brand:
            return AppleSiliconGPU.M1_PRO
        elif "m1" in brand:
            return AppleSiliconGPU.M1
        else:
            return AppleSiliconGPU.UNKNOWN

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return AppleSiliconGPU.UNKNOWN


class OccupancyAnalyzer:
    """Analyze threadgroup occupancy for Metal kernels.

    Computes theoretical and achievable occupancy based on kernel
    configuration and hardware limits.

    Args:
        gpu: Target GPU variant. If None, auto-detects.
        max_threads_per_eu: Override max threads per execution unit.
        max_tg_memory_kb: Max threadgroup memory in KB (default 32).

    Example:
        analyzer = OccupancyAnalyzer()

        config = ThreadgroupConfig(
            threads_per_tg=128,
            simdgroups_per_tg=4,
            threadgroup_memory_bytes=16 * 1024,
        )

        metrics = analyzer.analyze(config)
        print(f"Occupancy: {metrics.achieved_occupancy:.1f}%")
    """

    # Metal hardware limits
    SIMD_WIDTH = 32
    MAX_THREADS_PER_TG = 1024
    MAX_TG_MEMORY_BYTES = 32 * 1024  # 32 KB
    MAX_REGISTERS_PER_THREAD = 256  # Estimate

    def __init__(
        self,
        gpu: AppleSiliconGPU | None = None,
        *,
        max_threads_per_eu: int | None = None,
        max_tg_memory_kb: int = 32,
    ):
        self.gpu = gpu or detect_gpu()
        self.max_tg_memory_bytes = max_tg_memory_kb * 1024

        # Max threads per EU depends on GPU variant
        # This is an approximation; actual value varies
        if max_threads_per_eu is not None:
            self.max_threads_per_eu = max_threads_per_eu
        else:
            # Estimate: 32 TGs * 32 threads (1024) is common max
            self.max_threads_per_eu = self.gpu.max_tg_per_eu * self.SIMD_WIDTH

    def analyze(self, config: ThreadgroupConfig) -> OccupancyMetrics:
        """Analyze occupancy for a given threadgroup configuration.

        Args:
            config: Kernel threadgroup configuration.

        Returns:
            OccupancyMetrics with analysis results.
        """
        recommendations: list[str] = []
        details: dict[str, Any] = {}

        # Check basic validity
        if config.threads_per_tg > self.MAX_THREADS_PER_TG:
            return OccupancyMetrics(
                theoretical_occupancy=0.0,
                achieved_occupancy=0.0,
                limiting_factor="threads_per_tg exceeds hardware limit",
                max_tg_per_eu=0,
                max_threads_per_eu=self.max_threads_per_eu,
                threads_per_eu=0,
                recommendations=["Reduce threads_per_tg to <= 1024"],
                details={"error": "Invalid configuration"},
            )

        # Calculate limits from each resource

        # 1. Threadgroup limit
        max_tg_by_threads = self.gpu.max_tg_per_eu
        details["max_tg_by_threads"] = max_tg_by_threads

        # 2. Threadgroup memory limit
        if config.threadgroup_memory_bytes > 0:
            # Assume 32KB total TG memory per EU
            total_tg_memory = self.max_tg_memory_bytes
            max_tg_by_memory = total_tg_memory // config.threadgroup_memory_bytes
            max_tg_by_memory = max(1, min(max_tg_by_memory, self.gpu.max_tg_per_eu))
        else:
            max_tg_by_memory = self.gpu.max_tg_per_eu
        details["max_tg_by_memory"] = max_tg_by_memory

        # 3. Register limit (approximate)
        # Assume 65536 registers per EU
        total_registers = 65536
        registers_per_tg = config.threads_per_tg * config.registers_per_thread
        if registers_per_tg > 0:
            max_tg_by_registers = total_registers // registers_per_tg
            max_tg_by_registers = max(1, min(max_tg_by_registers, self.gpu.max_tg_per_eu))
        else:
            max_tg_by_registers = self.gpu.max_tg_per_eu
        details["max_tg_by_registers"] = max_tg_by_registers

        # Find the limiting factor
        limits = {
            "threadgroup_count": max_tg_by_threads,
            "threadgroup_memory": max_tg_by_memory,
            "registers": max_tg_by_registers,
        }
        limiting_factor = min(limits.keys(), key=lambda k: limits[k])
        max_tg_per_eu = limits[limiting_factor]

        # Calculate occupancy
        threads_per_eu = max_tg_per_eu * config.threads_per_tg
        theoretical_occupancy = (threads_per_eu / self.max_threads_per_eu) * 100.0

        # Achieved occupancy accounts for practical inefficiencies
        # Typically 80-90% of theoretical due to scheduling overhead
        efficiency_factor = 0.85
        achieved_occupancy = theoretical_occupancy * efficiency_factor

        # Generate recommendations
        if limiting_factor == "threadgroup_memory":
            recommendations.append(
                f"Reduce threadgroup memory from {config.threadgroup_memory_bytes} bytes"
            )
            recommendations.append(
                "Consider using device memory instead of threadgroup memory"
            )

        if limiting_factor == "registers":
            recommendations.append("Reduce register usage by simplifying kernel logic")
            recommendations.append("Consider using function constants instead of locals")

        if config.threads_per_tg < 128:
            recommendations.append(
                "Increase threads_per_tg to at least 128 for better utilization"
            )

        if config.threads_per_tg % self.SIMD_WIDTH != 0:
            recommendations.append(
                f"threads_per_tg ({config.threads_per_tg}) should be multiple of "
                f"SIMD width ({self.SIMD_WIDTH})"
            )

        # Check for power-of-2 sizing
        if config.threads_per_tg & (config.threads_per_tg - 1) != 0:
            recommendations.append(
                "Consider power-of-2 threadgroup size for better coalescing"
            )

        return OccupancyMetrics(
            theoretical_occupancy=theoretical_occupancy,
            achieved_occupancy=achieved_occupancy,
            limiting_factor=limiting_factor,
            max_tg_per_eu=max_tg_per_eu,
            max_threads_per_eu=self.max_threads_per_eu,
            threads_per_eu=threads_per_eu,
            recommendations=recommendations,
            details=details,
        )

    @classmethod
    def analyze_quick(
        cls,
        threads_per_tg: int,
        simdgroups_per_tg: int = 0,
        threadgroup_memory_bytes: int = 0,
    ) -> OccupancyMetrics:
        """Quick occupancy analysis with default hardware detection.

        Args:
            threads_per_tg: Threads per threadgroup.
            simdgroups_per_tg: SIMD groups per TG (auto-computed if 0).
            threadgroup_memory_bytes: Shared memory usage.

        Returns:
            OccupancyMetrics.

        Example:
            metrics = OccupancyAnalyzer.analyze_quick(
                threads_per_tg=128,
                threadgroup_memory_bytes=8192,
            )
        """
        analyzer = cls()
        config = ThreadgroupConfig(
            threads_per_tg=threads_per_tg,
            simdgroups_per_tg=simdgroups_per_tg,
            threadgroup_memory_bytes=threadgroup_memory_bytes,
        )
        return analyzer.analyze(config)


def estimate_optimal_config(
    work_items: int,
    threadgroup_memory_per_item: int = 0,
    gpu: AppleSiliconGPU | None = None,
) -> ThreadgroupConfig:
    """Estimate optimal threadgroup configuration for a workload.

    Args:
        work_items: Total number of work items to process.
        threadgroup_memory_per_item: Shared memory needed per work item.
        gpu: Target GPU (auto-detect if None).

    Returns:
        Recommended ThreadgroupConfig.

    Example:
        # For a GEMM with M=4096, N=4096
        config = estimate_optimal_config(work_items=4096 * 4096)
    """
    gpu = gpu or detect_gpu()
    analyzer = OccupancyAnalyzer(gpu)

    # Try common threadgroup sizes and pick the one with best occupancy
    candidates = [64, 128, 256, 512, 1024]

    best_config = ThreadgroupConfig(threads_per_tg=128)
    best_occupancy = 0.0

    for threads in candidates:
        if threads > work_items:
            continue

        tg_memory = threads * threadgroup_memory_per_item
        if tg_memory > analyzer.max_tg_memory_bytes:
            continue

        config = ThreadgroupConfig(
            threads_per_tg=threads,
            threadgroup_memory_bytes=tg_memory,
        )
        metrics = analyzer.analyze(config)

        if metrics.achieved_occupancy > best_occupancy:
            best_occupancy = metrics.achieved_occupancy
            best_config = config

    return best_config


def print_occupancy_report(config: ThreadgroupConfig) -> None:
    """Print a formatted occupancy report.

    Args:
        config: Threadgroup configuration to analyze.
    """
    analyzer = OccupancyAnalyzer()
    metrics = analyzer.analyze(config)

    print("\n" + "=" * 60)
    print("THREADGROUP OCCUPANCY REPORT")
    print("=" * 60)
    print(f"\nGPU: {analyzer.gpu.name}")
    print(f"GPU Cores: {analyzer.gpu.gpu_cores}")
    print(f"Peak FP16 TFLOPS: {analyzer.gpu.peak_tflops_fp16}")

    print("\nConfiguration:")
    print(f"  Threads per TG: {config.threads_per_tg}")
    print(f"  SIMD groups per TG: {config.simdgroups_per_tg}")
    print(f"  TG memory: {config.threadgroup_memory_bytes} bytes")

    print("\nOccupancy:")
    print(f"  Theoretical: {metrics.theoretical_occupancy:.1f}%")
    print(f"  Achieved (est.): {metrics.achieved_occupancy:.1f}%")
    print(f"  Limiting factor: {metrics.limiting_factor}")

    print("\nLimits:")
    print(f"  Max TG per EU: {metrics.max_tg_per_eu}")
    print(f"  Max threads per EU: {metrics.max_threads_per_eu}")
    print(f"  Threads per EU (this config): {metrics.threads_per_eu}")

    if metrics.recommendations:
        print("\nRecommendations:")
        for rec in metrics.recommendations:
            print(f"  - {rec}")

    print("=" * 60 + "\n")
