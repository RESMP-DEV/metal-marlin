"""Roofline model analysis for Metal Marlin kernels.

M4 Max specifications (consistent with benchmark_gemm.py):
- Peak compute: ~32 TFLOPS FP16 (16 cores * 2 TFLOPS/core)
- Peak memory BW: ~546 GB/s
- Ridge point: 32000 GFLOPS / 546 GB/s = ~58.6 FLOP/byte
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# M4 Max specs (matching benchmark_gemm.py constants)
PEAK_TFLOPS = 32.0  # FP16
PEAK_BW_GBS = 546.0  # GB/s
RIDGE_POINT = PEAK_TFLOPS * 1000 / PEAK_BW_GBS  # FLOP/byte (~58.6)


@dataclass
class KernelMetrics:
    """Metrics for a single GEMM kernel invocation."""

    name: str
    M: int
    N: int
    K: int
    time_ms: float

    @property
    def flops(self) -> float:
        """Total floating-point operations (2*M*N*K for GEMM)."""
        return 2.0 * self.M * self.N * self.K

    @property
    def bytes_read(self) -> float:
        """Total bytes read from memory.

        A: M*K*2 (FP16)
        B_packed: K*N/2 (4-bit quantized)
        scales: (K/group_size)*N*2 (FP16, group_size=128)
        """
        return self.M * self.K * 2 + self.K * self.N / 2 + (self.K / 128) * self.N * 2

    @property
    def bytes_written(self) -> float:
        """Total bytes written (C matrix in FP16)."""
        return self.M * self.N * 2

    @property
    def total_bytes(self) -> float:
        return self.bytes_read + self.bytes_written

    @property
    def arithmetic_intensity(self) -> float:
        """FLOP/byte ratio."""
        return self.flops / self.total_bytes

    @property
    def achieved_tflops(self) -> float:
        return self.flops / self.time_ms / 1e9

    @property
    def achieved_bw_gbs(self) -> float:
        return self.total_bytes / self.time_ms / 1e6


def plot_roofline(
    metrics: list[KernelMetrics],
    output_path: str | Path = "roofline.png",
) -> None:
    """Generate roofline plot with kernel measurements overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline envelope
    ai = np.logspace(-1, 3, 200)
    roof = np.minimum(PEAK_TFLOPS, ai * PEAK_BW_GBS / 1000)
    ax.loglog(ai, roof, "b-", linewidth=2, label="Roofline (M4 Max)")

    # Ridge point annotation
    ax.axvline(RIDGE_POINT, color="gray", linestyle="--", alpha=0.5)
    ax.text(
        RIDGE_POINT * 1.1,
        PEAK_TFLOPS * 0.5,
        f"Ridge: {RIDGE_POINT:.1f} FLOP/byte",
        fontsize=8,
    )

    # Plot kernel measurements
    for m in metrics:
        ax.plot(m.arithmetic_intensity, m.achieved_tflops, "ro", markersize=8)
        ax.annotate(
            m.name,
            (m.arithmetic_intensity, m.achieved_tflops),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Marlin Kernel Roofline Analysis (M4 Max)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.1, 50)

    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved roofline plot to {output_path}")


def analyze_bottleneck(metrics: KernelMetrics) -> str:
    """Determine if kernel is compute-bound or memory-bound."""
    if metrics.arithmetic_intensity < RIDGE_POINT:
        efficiency = metrics.achieved_bw_gbs / PEAK_BW_GBS * 100
        return (
            f"Memory bound: {efficiency:.1f}% of peak BW "
            f"({metrics.achieved_bw_gbs:.0f} GB/s)"
        )
    else:
        efficiency = metrics.achieved_tflops / PEAK_TFLOPS * 100
        return (
            f"Compute bound: {efficiency:.1f}% of peak "
            f"({metrics.achieved_tflops:.1f} TFLOPS)"
        )


def print_analysis(metrics: list[KernelMetrics]) -> None:
    """Print bottleneck analysis table for all kernels."""
    print(f"\n{'Kernel':<30} {'AI (F/B)':>10} {'TFLOPS':>8} {'BW (GB/s)':>10} {'Bottleneck'}")
    print("-" * 90)
    for m in metrics:
        bottleneck = "MEM" if m.arithmetic_intensity < RIDGE_POINT else "COMPUTE"
        print(
            f"{m.name:<30} {m.arithmetic_intensity:>10.1f} "
            f"{m.achieved_tflops:>8.2f} {m.achieved_bw_gbs:>10.1f} "
            f"{bottleneck}"
        )
    print()
    for m in metrics:
        print(f"  {m.name}: {analyze_bottleneck(m)}")
