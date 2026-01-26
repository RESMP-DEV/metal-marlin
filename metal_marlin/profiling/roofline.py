"""Roofline model analysis and visualization for Metal kernels.

The roofline model visualizes the relationship between:
- Arithmetic intensity (FLOP/byte)
- Achieved performance (FLOP/s or TFLOP/s)
- Hardware limits (peak compute and memory bandwidth)

This helps identify whether kernels are compute-bound or memory-bound
and how close they are to hardware limits.

Apple Silicon Peak Performance:
- M4 Max: 32 TFLOPS FP16, 546 GB/s
- M3 Max: 16.4 TFLOPS FP16, 400 GB/s
- M2 Max: 13.6 TFLOPS FP16, 400 GB/s
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._compat import HAS_MATPLOTLIB, plt
from .occupancy import AppleSiliconGPU, detect_gpu
from .trace import TraceEvent


@dataclass
class RooflineConfig:
    """Hardware configuration for roofline model.

    Attributes:
        peak_tflops: Peak compute throughput (TFLOPS FP16).
        peak_bw_gbs: Peak memory bandwidth (GB/s).
        gpu_name: GPU identifier.
        additional_ceilings: Extra ceilings (e.g., "FP32": 16.0).
    """

    peak_tflops: float
    peak_bw_gbs: float
    gpu_name: str = "Unknown"
    additional_ceilings: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_gpu(cls, gpu: AppleSiliconGPU | None = None) -> RooflineConfig:
        """Create config from detected or specified GPU."""
        gpu = gpu or detect_gpu()
        return cls(
            peak_tflops=gpu.peak_tflops_fp16,
            peak_bw_gbs=gpu.peak_bw_gbs,
            gpu_name=gpu.name,
        )

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at ridge point (FLOP/byte)."""
        return (self.peak_tflops * 1000) / self.peak_bw_gbs


@dataclass
class KernelPoint:
    """A kernel measurement for the roofline plot.

    Attributes:
        name: Kernel identifier.
        tflops: Achieved TFLOPS.
        arithmetic_intensity: FLOP/byte ratio.
        label: Display label (defaults to name).
        color: Plot color (None for auto).
        marker: Plot marker (None for default).
        metadata: Additional kernel info.
    """

    name: str
    tflops: float
    arithmetic_intensity: float
    label: str | None = None
    color: str | None = None
    marker: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_measurement(
        cls,
        name: str,
        elapsed_ms: float,
        flops: float,
        bytes_moved: float,
        **metadata: Any,
    ) -> KernelPoint:
        """Create from raw measurement data.

        Args:
            name: Kernel identifier.
            elapsed_ms: Execution time in milliseconds.
            flops: Total floating-point operations.
            bytes_moved: Total bytes read + written.
            **metadata: Additional context.

        Returns:
            KernelPoint with computed metrics.
        """
        if elapsed_ms <= 0 or bytes_moved <= 0:
            return cls(
                name=name,
                tflops=0.0,
                arithmetic_intensity=0.0,
                metadata=metadata,
            )

        elapsed_s = elapsed_ms / 1000.0
        tflops = (flops / elapsed_s) / 1e12
        ai = flops / bytes_moved

        return cls(
            name=name,
            tflops=tflops,
            arithmetic_intensity=ai,
            metadata=metadata,
        )

    @classmethod
    def from_gemm(
        cls,
        name: str,
        M: int,
        N: int,
        K: int,
        elapsed_ms: float,
        *,
        a_bytes: int = 2,
        b_bytes: float = 0.5,
        c_bytes: int = 2,
        scale_bytes: int = 2,
        group_size: int = 128,
    ) -> KernelPoint:
        """Create from GEMM parameters.

        Args:
            name: Kernel identifier.
            M: Rows of A and C.
            N: Columns of B and C.
            K: Shared dimension.
            elapsed_ms: Execution time.
            a_bytes: Bytes per A element (default 2 for FP16).
            b_bytes: Bytes per B element (default 0.5 for FP4).
            c_bytes: Bytes per C element (default 2 for FP16).
            scale_bytes: Bytes per scale value.
            group_size: Quantization group size.

        Returns:
            KernelPoint.
        """
        # GEMM FLOPs: 2*M*N*K (multiply + accumulate)
        flops = 2.0 * M * N * K

        # Memory traffic
        bytes_a = M * K * a_bytes
        bytes_b = int(K * N * b_bytes)
        num_groups = (K + group_size - 1) // group_size
        bytes_scales = num_groups * N * scale_bytes
        bytes_c = M * N * c_bytes

        bytes_read = bytes_a + bytes_b + bytes_scales
        bytes_written = bytes_c
        bytes_total = bytes_read + bytes_written

        return cls.from_measurement(
            name=name,
            elapsed_ms=elapsed_ms,
            flops=flops,
            bytes_moved=bytes_total,
            M=M,
            N=N,
            K=K,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "name": self.name,
            "tflops": self.tflops,
            "arithmetic_intensity": self.arithmetic_intensity,
            "label": self.label or self.name,
            "metadata": self.metadata,
        }

    def to_trace_event(
        self,
        *,
        timestamp_ns: int,
        pid: int = 0,
        tid: int = 0,
    ) -> TraceEvent:
        return TraceEvent(
            name="roofline_point",
            cat="roofline",
            ph="C",
            ts=int(timestamp_ns / 1000),
            pid=pid,
            tid=tid,
            args={
                "kernel": self.name,
                "tflops": self.tflops,
                "arithmetic_intensity": self.arithmetic_intensity,
            },
        )


class RooflineAnalyzer:
    """Roofline model analysis and visualization.

    Manages hardware configuration and kernel measurements,
    generates roofline plots and analysis reports.

    Args:
        config: Hardware configuration. If None, auto-detects.

    Example:
        analyzer = RooflineAnalyzer()

        # Add kernel measurements
        analyzer.add_kernel("gemm_fp4_4096", tflops=15.2, memory_gbs=320)
        analyzer.add_kernel("attention_fp4", tflops=8.5, memory_gbs=180)

        # Generate plot
        analyzer.plot("roofline.png")

        # Print analysis
        analyzer.print_analysis()
    """

    def __init__(self, config: RooflineConfig | None = None):
        self.config = config or RooflineConfig.from_gpu()
        self._kernels: list[KernelPoint] = []

    def add_kernel(
        self,
        name: str,
        *,
        tflops: float | None = None,
        arithmetic_intensity: float | None = None,
        memory_gbs: float | None = None,
        elapsed_ms: float | None = None,
        flops: float | None = None,
        bytes_moved: float | None = None,
        **metadata: Any,
    ) -> KernelPoint:
        """Add a kernel measurement.

        Accepts various input combinations:
        1. tflops + arithmetic_intensity (direct)
        2. tflops + memory_gbs (compute AI)
        3. elapsed_ms + flops + bytes_moved (compute all)

        Args:
            name: Kernel identifier.
            tflops: Achieved TFLOPS.
            arithmetic_intensity: FLOP/byte ratio.
            memory_gbs: Achieved memory bandwidth (GB/s).
            elapsed_ms: Execution time (ms).
            flops: Total FLOPs.
            bytes_moved: Total bytes.
            **metadata: Additional context.

        Returns:
            Created KernelPoint.

        Raises:
            ValueError: If insufficient parameters provided.
        """
        if tflops is not None and arithmetic_intensity is not None:
            # Direct specification
            point = KernelPoint(
                name=name,
                tflops=tflops,
                arithmetic_intensity=arithmetic_intensity,
                metadata=metadata,
            )
        elif tflops is not None and memory_gbs is not None:
            # Compute AI from bandwidth
            # AI = TFLOPS * 1000 / BW(GB/s)
            ai = (tflops * 1000) / memory_gbs if memory_gbs > 0 else 0
            point = KernelPoint(
                name=name,
                tflops=tflops,
                arithmetic_intensity=ai,
                metadata=metadata,
            )
        elif elapsed_ms is not None and flops is not None and bytes_moved is not None:
            # Compute from raw measurements
            point = KernelPoint.from_measurement(
                name=name,
                elapsed_ms=elapsed_ms,
                flops=flops,
                bytes_moved=bytes_moved,
                **metadata,
            )
        else:
            raise ValueError(
                "Must provide one of: "
                "(tflops + arithmetic_intensity), "
                "(tflops + memory_gbs), or "
                "(elapsed_ms + flops + bytes_moved)"
            )

        self._kernels.append(point)
        return point

    def add_gemm(
        self,
        name: str,
        M: int,
        N: int,
        K: int,
        elapsed_ms: float,
        **kwargs: Any,
    ) -> KernelPoint:
        """Add a GEMM kernel measurement.

        Convenience method for GEMM kernels with standard memory layout.

        Args:
            name: Kernel identifier.
            M, N, K: GEMM dimensions.
            elapsed_ms: Execution time.
            **kwargs: Additional params for KernelPoint.from_gemm.

        Returns:
            Created KernelPoint.
        """
        point = KernelPoint.from_gemm(name, M, N, K, elapsed_ms, **kwargs)
        self._kernels.append(point)
        return point

    @property
    def kernels(self) -> list[KernelPoint]:
        """All added kernel measurements."""
        return list(self._kernels)

    def clear(self) -> None:
        """Clear all kernel measurements."""
        self._kernels.clear()

    def analyze_kernel(self, kernel: KernelPoint) -> dict[str, Any]:
        """Analyze a single kernel against the roofline.

        Args:
            kernel: Kernel to analyze.

        Returns:
            Analysis dictionary with bound type, utilization, etc.
        """
        ridge = self.config.ridge_point

        if kernel.arithmetic_intensity < ridge:
            # Memory-bound region
            bound = "memory"
            # Max achievable at this AI
            max_tflops = (kernel.arithmetic_intensity * self.config.peak_bw_gbs) / 1000
            attainment = (kernel.tflops / max_tflops) * 100 if max_tflops > 0 else 0
            headroom_tflops = max_tflops - kernel.tflops
        else:
            # Compute-bound region
            bound = "compute"
            max_tflops = self.config.peak_tflops
            attainment = (kernel.tflops / max_tflops) * 100
            headroom_tflops = max_tflops - kernel.tflops

        return {
            "name": kernel.name,
            "bound": bound,
            "tflops": kernel.tflops,
            "arithmetic_intensity": kernel.arithmetic_intensity,
            "max_achievable_tflops": max_tflops,
            "attainment_pct": attainment,
            "headroom_tflops": headroom_tflops,
            "distance_to_ridge": abs(kernel.arithmetic_intensity - ridge),
        }

    def print_analysis(self) -> None:
        """Print analysis of all kernels."""
        if not self._kernels:
            print("No kernels added")
            return

        print("\n" + "=" * 80)
        print(f"ROOFLINE ANALYSIS - {self.config.gpu_name}")
        print(f"Peak: {self.config.peak_tflops} TFLOPS, {self.config.peak_bw_gbs} GB/s")
        print(f"Ridge point: {self.config.ridge_point:.1f} FLOP/byte")
        print("=" * 80)

        header = (
            f"{'Kernel':<30} {'TFLOPS':>8} {'AI':>10} {'Bound':>8} {'Attain%':>8} {'Max TF':>8}"
        )
        print(header)
        print("-" * len(header))

        for kernel in self._kernels:
            analysis = self.analyze_kernel(kernel)
            print(
                f"{kernel.name:<30} {kernel.tflops:>8.2f} "
                f"{kernel.arithmetic_intensity:>10.1f} {analysis['bound']:>8} "
                f"{analysis['attainment_pct']:>8.1f} "
                f"{analysis['max_achievable_tflops']:>8.2f}"
            )

        print("=" * 80 + "\n")

    def plot(
        self,
        output_path: str | Path = "roofline.png",
        *,
        title: str | None = None,
        figsize: tuple[float, float] = (12, 8),
        dpi: int = 150,
        show_labels: bool = True,
        ai_range: tuple[float, float] = (0.1, 1000),
    ) -> None:
        """Generate roofline plot.

        Args:
            output_path: Path for output image.
            title: Plot title (default auto-generated).
            figsize: Figure size in inches.
            dpi: Output resolution.
            show_labels: Whether to show kernel labels.
            ai_range: Arithmetic intensity range for x-axis.

        Raises:
            ImportError: If matplotlib not available.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib required for plotting. Install with: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Generate roofline envelope
        ai = np.logspace(np.log10(ai_range[0]), np.log10(ai_range[1]), 500)
        memory_roof = (ai * self.config.peak_bw_gbs) / 1000  # Convert to TFLOPS
        compute_roof = np.full_like(ai, self.config.peak_tflops)
        roof = np.minimum(memory_roof, compute_roof)

        # Plot roofline
        ax.loglog(ai, roof, "b-", linewidth=2.5, label=f"Roofline ({self.config.gpu_name})")

        # Plot memory and compute ceilings with different styles
        memory_region = ai < self.config.ridge_point
        ax.loglog(
            ai[memory_region],
            memory_roof[memory_region],
            "b--",
            linewidth=1,
            alpha=0.5,
        )
        ax.loglog(
            ai[~memory_region],
            compute_roof[~memory_region],
            "b--",
            linewidth=1,
            alpha=0.5,
        )

        # Ridge point annotation
        ax.axvline(
            self.config.ridge_point,
            color="gray",
            linestyle=":",
            alpha=0.7,
            linewidth=1,
        )
        ax.annotate(
            f"Ridge: {self.config.ridge_point:.1f} F/B",
            xy=(self.config.ridge_point, self.config.peak_tflops * 0.3),
            fontsize=9,
            color="gray",
        )

        # Plot additional ceilings if any
        for name, ceiling_tflops in self.config.additional_ceilings.items():
            ceiling_roof = np.minimum(memory_roof, ceiling_tflops)
            ax.loglog(ai, ceiling_roof, "--", linewidth=1, alpha=0.7, label=name)

        # Plot kernel points
        colors = plt.cm.Set1.colors  # type: ignore[attr-defined]
        for i, kernel in enumerate(self._kernels):
            color = kernel.color or colors[i % len(colors)]
            marker = kernel.marker or "o"
            label = kernel.label or kernel.name

            ax.plot(
                kernel.arithmetic_intensity,
                kernel.tflops,
                marker,
                color=color,
                markersize=10,
                label=label,
            )

            if show_labels:
                ax.annotate(
                    label,
                    (kernel.arithmetic_intensity, kernel.tflops),
                    textcoords="offset points",
                    xytext=(8, 5),
                    fontsize=8,
                )

        # Labels and formatting
        ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
        ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
        ax.set_title(
            title or f"Roofline Model - {self.config.gpu_name}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=9)

        # Set axis limits
        ax.set_xlim(ai_range)
        ax.set_ylim(0.1, self.config.peak_tflops * 1.5)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved roofline plot to {output_path}")

    def export_json(self, output_path: str | Path) -> None:
        """Export analysis to JSON.

        Args:
            output_path: Path for JSON output.
        """
        data = {
            "config": {
                "peak_tflops": self.config.peak_tflops,
                "peak_bw_gbs": self.config.peak_bw_gbs,
                "gpu_name": self.config.gpu_name,
                "ridge_point": self.config.ridge_point,
            },
            "kernels": [
                {
                    **k.to_dict(),
                    "analysis": self.analyze_kernel(k),
                }
                for k in self._kernels
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported analysis to {output_path}")

    def export_trace(self, output_path: str | Path) -> None:
        """Export kernel points as Chrome trace counters."""
        from .trace import ChromeTrace

        trace = ChromeTrace()
        ts_ns = 0
        for kernel in self._kernels:
            ts_ns += 1000
            trace.add_event(kernel.to_trace_event(timestamp_ns=ts_ns))

        trace.export_json(output_path)


def quick_roofline(
    measurements: list[dict[str, Any]],
    output_path: str | Path = "roofline.png",
) -> RooflineAnalyzer:
    """Generate roofline plot from measurement dictionaries.

    Convenience function for quick visualization.

    Args:
        measurements: List of dicts with kernel measurements.
            Each dict should have 'name' and one of:
            - 'tflops' + 'arithmetic_intensity'
            - 'tflops' + 'memory_gbs'
            - 'elapsed_ms' + 'flops' + 'bytes_moved'
        output_path: Path for output image.

    Returns:
        RooflineAnalyzer with added measurements.

    Example:
        measurements = [
            {"name": "gemm_4096", "tflops": 15.2, "arithmetic_intensity": 85.3},
            {"name": "attention", "tflops": 8.5, "memory_gbs": 280},
        ]
        analyzer = quick_roofline(measurements, "my_roofline.png")
    """
    analyzer = RooflineAnalyzer()

    for m in measurements:
        name = m.pop("name")
        analyzer.add_kernel(name, **m)

    analyzer.plot(output_path)
    return analyzer
