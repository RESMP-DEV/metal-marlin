#!/usr/bin/env python3
"""
Theoretical Memory Bandwidth Analysis for Attention Operations

Analyzes memory bandwidth requirements for different attention configurations
and hardware platforms without requiring actual execution.

Key Outputs:
- Bytes loaded/stored per operation
- Arithmetic intensity (FLOP/byte ratio)
- Memory vs compute bound classification
- Bandwidth utilization at target efficiency

Usage:
    uv run python analyze_attention_bandwidth.py
    uv run python analyze_attention_bandwidth.py --seq-lens 512 1024 2048 4096
    uv run python analyze_attention_bandwidth.py --hardware A100 --roofline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# Hardware Specifications
# =============================================================================

@dataclass(frozen=True)
class HardwareSpec:
    """Hardware specifications for bandwidth analysis."""
    name: str
    peak_bw_gbs: float          # Peak memory bandwidth (GB/s)
    peak_tflops_fp16: float    # Peak FP16 compute (TFLOPS)
    peak_tflops_fp32: float    # Peak FP32 compute (TFLOPS)
    memory_gb: float           # Memory capacity

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at ridge point (FLOP/byte).
        
        The ridge point is where compute and memory bounds intersect.
        Below this: memory-bound. Above this: compute-bound.
        """
        return (self.peak_tflops_fp16 * 1000) / self.peak_bw_gbs

    def max_tflops_at_ai(self, ai: float) -> float:
        """Maximum achievable TFLOPS at given arithmetic intensity."""
        if ai < self.ridge_point:
            # Memory bound
            return (ai * self.peak_bw_gbs) / 1000
        else:
            # Compute bound
            return self.peak_tflops_fp16


HARDWARE_SPECS = {
    "M1_MAX": HardwareSpec("Apple M1 Max", 400.0, 10.4, 5.2, 64.0),
    "M2_MAX": HardwareSpec("Apple M2 Max", 400.0, 13.6, 6.8, 96.0),
    "M3_MAX": HardwareSpec("Apple M3 Max", 400.0, 16.4, 8.2, 128.0),
    "M4_MAX": HardwareSpec("Apple M4 Max", 546.0, 32.0, 16.0, 128.0),
    "V100": HardwareSpec("NVIDIA V100", 900.0, 31.4, 15.7, 32.0),
    "A100_40": HardwareSpec("NVIDIA A100 40GB", 1555.0, 78.0, 19.5, 40.0),
    "A100_80": HardwareSpec("NVIDIA A100 80GB", 2039.0, 78.0, 19.5, 80.0),
    "H100": HardwareSpec("NVIDIA H100 SXM", 3352.0, 197.9, 51.0, 80.0),
}


# =============================================================================
# Attention Operation Analysis
# =============================================================================

@dataclass
class AttentionConfig:
    """Configuration for attention operation."""
    batch_size: int
    num_heads: int
    seq_len_q: int
    seq_len_kv: int
    head_dim: int
    causal: bool = False
    dtype_bytes: int = 2  # FP16 default

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim


@dataclass
class AttentionBandwidthProfile:
    """Complete bandwidth profile for attention operation."""

    # Configuration
    config: AttentionConfig
    impl_type: str  # "standard", "flash", "memory_efficient"

    # Memory traffic breakdown
    q_bytes: int
    k_bytes: int
    v_bytes: int
    output_bytes: int
    intermediate_bytes: int

    # FLOPs breakdown
    qk_flops: int
    softmax_flops: int
    av_flops: int

    # Derived metrics (computed in __post_init__)
    total_bytes_loaded: int = field(init=False)
    total_bytes_stored: int = field(init=False)
    total_bytes_moved: int = field(init=False)
    total_flops: int = field(init=False)
    arithmetic_intensity: float = field(init=False)

    def __post_init__(self):
        self.total_bytes_loaded = self.q_bytes + self.k_bytes + self.v_bytes
        self.total_bytes_stored = self.output_bytes + self.intermediate_bytes
        self.total_bytes_moved = self.total_bytes_loaded + self.total_bytes_stored
        self.total_flops = self.qk_flops + self.softmax_flops + self.av_flops
        self.arithmetic_intensity = (
            self.total_flops / self.total_bytes_moved if self.total_bytes_moved > 0 else 0.0
        )

    def analyze(self, hardware: HardwareSpec) -> dict[str, Any]:
        """Analyze profile against hardware constraints."""
        # Determine bound type
        if self.arithmetic_intensity < hardware.ridge_point:
            bound_type = "memory"
            max_theoretical_tflops = (self.arithmetic_intensity * hardware.peak_bw_gbs) / 1000
        else:
            bound_type = "compute"
            max_theoretical_tflops = hardware.peak_tflops_fp16

        # Calculate memory bandwidth required for different efficiency levels
        results = {
            "bound_type": bound_type,
            "ridge_point": hardware.ridge_point,
            "arithmetic_intensity": self.arithmetic_intensity,
            "max_theoretical_tflops": max_theoretical_tflops,
            "bytes_loaded": self.total_bytes_loaded,
            "bytes_stored": self.total_bytes_stored,
            "total_mb": self.total_bytes_moved / (1024 * 1024),
            "total_gb": self.total_bytes_moved / (1024 * 1024 * 1024),
        }

        # Add bandwidth requirements at different efficiency levels
        for efficiency in [0.5, 0.7, 0.8, 0.9]:
            time_at_efficiency = self.total_flops / (max_theoretical_tflops * efficiency * 1e12)
            bw_required = (self.total_bytes_moved / time_at_efficiency) / 1e9
            results[f"bw_required_{int(efficiency*100)}pct_gbs"] = bw_required
            results[f"time_at_{int(efficiency*100)}pct_us"] = time_at_efficiency * 1e6

        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "batch": self.config.batch_size,
                "heads": self.config.num_heads,
                "seq_q": self.config.seq_len_q,
                "seq_kv": self.config.seq_len_kv,
                "head_dim": self.config.head_dim,
                "causal": self.config.causal,
                "dtype_bytes": self.config.dtype_bytes,
            },
            "implementation": self.impl_type,
            "memory": {
                "q_bytes": self.q_bytes,
                "k_bytes": self.k_bytes,
                "v_bytes": self.v_bytes,
                "output_bytes": self.output_bytes,
                "intermediate_bytes": self.intermediate_bytes,
                "total_bytes_loaded": self.total_bytes_loaded,
                "total_bytes_stored": self.total_bytes_stored,
                "total_bytes_moved": self.total_bytes_moved,
            },
            "compute": {
                "qk_flops": self.qk_flops,
                "softmax_flops": self.softmax_flops,
                "av_flops": self.av_flops,
                "total_flops": self.total_flops,
            },
            "metrics": {
                "arithmetic_intensity": self.arithmetic_intensity,
            },
        }


def analyze_standard_attention(config: AttentionConfig) -> AttentionBandwidthProfile:
    """
    Analyze standard (non-fused) attention.
    
    Standard attention:
    1. Load Q, K, V from HBM
    2. Compute Q @ K^T, write to HBM (materialize NxN)
    3. Load QK^T, apply softmax, write back
    4. Load softmax output, compute @ V, write output
    """
    B, H, Sq, Skv, D = (
        config.batch_size, config.num_heads,
        config.seq_len_q, config.seq_len_kv, config.head_dim
    )
    dtype = config.dtype_bytes
    causal = config.causal

    # Memory: Q, K, V read once each
    q_bytes = B * H * Sq * D * dtype
    k_bytes = B * H * Skv * D * dtype
    v_bytes = B * H * Skv * D * dtype
    output_bytes = B * H * Sq * D * dtype

    # Intermediate: QK^T matrix materialized (read + written)
    # Softmax output also materialized
    qk_bytes = B * H * Sq * Skv * dtype
    softmax_bytes = qk_bytes

    # Total intermediate traffic (read and write)
    # QK^T: written once, read once = 2x
    # Softmax: written once, read once = 2x
    intermediate_bytes = 2 * (qk_bytes + softmax_bytes)

    # FLOPs
    causal_factor = 0.5 if causal else 1.0
    qk_flops = int(2 * B * H * Sq * Skv * D * causal_factor)
    softmax_flops = int(5 * B * H * Sq * Skv * causal_factor)
    av_flops = int(2 * B * H * Sq * Skv * D * causal_factor)

    return AttentionBandwidthProfile(
        config=config,
        impl_type="standard",
        q_bytes=q_bytes,
        k_bytes=k_bytes,
        v_bytes=v_bytes,
        output_bytes=output_bytes,
        intermediate_bytes=intermediate_bytes,
        qk_flops=qk_flops,
        softmax_flops=softmax_flops,
        av_flops=av_flops,
    )


def analyze_flash_attention(config: AttentionConfig) -> AttentionBandwidthProfile:
    """
    Analyze Flash Attention (tiled, fused).
    
    Flash Attention never materializes the full QK^T matrix.
    Reduces memory traffic from O(N^2) to O(N).
    """
    B, H, Sq, Skv, D = (
        config.batch_size, config.num_heads,
        config.seq_len_q, config.seq_len_kv, config.head_dim
    )
    dtype = config.dtype_bytes
    causal = config.causal

    # Input/output same as standard
    q_bytes = B * H * Sq * D * dtype
    k_bytes = B * H * Skv * D * dtype
    v_bytes = B * H * Skv * D * dtype
    output_bytes = B * H * Sq * D * dtype

    # Flash Attention: minimal intermediate storage
    # Only O(block_size) instead of O(N^2)
    # Approximate as ~1% of standard intermediate for tile buffers
    qk_bytes = B * H * Sq * Skv * dtype
    intermediate_bytes = int(qk_bytes * 0.01)  # ~1% for tile buffers

    # FLOPs same as standard
    causal_factor = 0.5 if causal else 1.0
    qk_flops = int(2 * B * H * Sq * Skv * D * causal_factor)
    softmax_flops = int(5 * B * H * Sq * Skv * causal_factor)
    av_flops = int(2 * B * H * Sq * Skv * D * causal_factor)

    return AttentionBandwidthProfile(
        config=config,
        impl_type="flash",
        q_bytes=q_bytes,
        k_bytes=k_bytes,
        v_bytes=v_bytes,
        output_bytes=output_bytes,
        intermediate_bytes=intermediate_bytes,
        qk_flops=qk_flops,
        softmax_flops=softmax_flops,
        av_flops=av_flops,
    )


def analyze_memory_efficient_attention(config: AttentionConfig) -> AttentionBandwidthProfile:
    """
    Analyze memory-efficient attention (checkpointing, tiling without fusion).
    
    Similar to flash attention but may have different tradeoffs.
    """
    # For this analysis, treat similar to flash attention
    # but with slightly higher intermediate storage
    profile = analyze_flash_attention(config)
    profile.impl_type = "memory_efficient"
    # 5% intermediate instead of 1%
    profile.intermediate_bytes = int(profile.intermediate_bytes * 5)
    return profile


# =============================================================================
# Reporting and Visualization
# =============================================================================

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable."""
    if bytes_val >= 1024**4:
        return f"{bytes_val / 1024**4:.2f} TB"
    elif bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def format_flops(flops: int) -> str:
    """Format FLOPs to human readable."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops} FLOPs"


def print_profile(profile: AttentionBandwidthProfile, hardware: HardwareSpec) -> None:
    """Print detailed bandwidth profile."""
    c = profile.config
    analysis = profile.analyze(hardware)

    print(f"\n{'─' * 100}")
    print(f"Attention Bandwidth Profile: {profile.impl_type.upper()}")
    print(f"{'─' * 100}")
    print(f"Configuration: B={c.batch_size}, H={c.num_heads}, Sq={c.seq_len_q}, "
          f"Skv={c.seq_len_kv}, D={c.head_dim}, Causal={c.causal}")
    print(f"Hardware: {hardware.name} (Peak BW: {hardware.peak_bw_gbs:.0f} GB/s)")
    print(f"{'─' * 100}")

    print("\n📊 MEMORY TRAFFIC BREAKDOWN")
    print("  Bytes Loaded:")
    print(f"    Q:           {format_bytes(profile.q_bytes):>15} ({profile.q_bytes/profile.total_bytes_loaded*100:.1f}%)")
    print(f"    K:           {format_bytes(profile.k_bytes):>15} ({profile.k_bytes/profile.total_bytes_loaded*100:.1f}%)")
    print(f"    V:           {format_bytes(profile.v_bytes):>15} ({profile.v_bytes/profile.total_bytes_loaded*100:.1f}%)")
    print("    ─────────────────────────────────────")
    print(f"    Total:       {format_bytes(profile.total_bytes_loaded):>15}")
    print("\n  Bytes Stored:")
    print(f"    Output:      {format_bytes(profile.output_bytes):>15}")
    print(f"    Intermediate:{format_bytes(profile.intermediate_bytes):>15}")
    print("    ─────────────────────────────────────")
    print(f"    Total:       {format_bytes(profile.total_bytes_stored):>15}")
    print(f"\n  TOTAL BYTES MOVED: {format_bytes(profile.total_bytes_moved):>15}")

    print("\n⚡ COMPUTE BREAKDOWN")
    print(f"  QK matmul:   {format_flops(profile.qk_flops):>20}")
    print(f"  Softmax:     {format_flops(profile.softmax_flops):>20}")
    print(f"  AV matmul:   {format_flops(profile.av_flops):>20}")
    print("  ─────────────────────────────────────")
    print(f"  Total:       {format_flops(profile.total_flops):>20}")

    print("\n📈 ANALYSIS")
    print(f"  Arithmetic Intensity: {profile.arithmetic_intensity:.2f} FLOP/byte")
    print(f"  Ridge Point:          {hardware.ridge_point:.2f} FLOP/byte")
    print(f"  Classification:       {'🔴 MEMORY-BOUND' if analysis['bound_type'] == 'memory' else '🔵 COMPUTE-BOUND'}")
    print(f"  Max Theoretical:      {analysis['max_theoretical_tflops']:.2f} TFLOPS")

    print("\n💾 BANDWIDTH REQUIREMENTS (at target efficiency)")
    for eff in [50, 70, 80, 90]:
        bw_key = f"bw_required_{eff}pct_gbs"
        time_key = f"time_at_{eff}pct_us"
        bw = analysis.get(bw_key, 0)
        t = analysis.get(time_key, 0)
        pct_of_peak = (bw / hardware.peak_bw_gbs) * 100
        print(f"  {eff}% efficiency: {bw:>7.1f} GB/s ({pct_of_peak:>5.1f}% of peak) → {t:>10.2f} μs")

    print(f"{'─' * 100}\n")


def print_comparison_table(
    profiles: list[AttentionBandwidthProfile],
    hardware: HardwareSpec
) -> None:
    """Print comparison table of different implementations."""
    print(f"\n{'=' * 120}")
    print(f"IMPLEMENTATION COMPARISON - {hardware.name}")
    print(f"{'=' * 120}")

    header = (
        f"{'Implementation':<20} {'Bytes Loaded':>14} {'Bytes Stored':>14} "
        f"{'Total':>12} {'AI':>10} {'Bound':>10} {'BW@80%':>12} {'Time@80%':>12}"
    )
    print(header)
    print("-" * len(header))

    for profile in profiles:
        analysis = profile.analyze(hardware)
        print(
            f"{profile.impl_type:<20} "
            f"{format_bytes(profile.total_bytes_loaded):>14} "
            f"{format_bytes(profile.total_bytes_stored):>14} "
            f"{format_bytes(profile.total_bytes_moved):>12} "
            f"{profile.arithmetic_intensity:>10.1f} "
            f"{analysis['bound_type']:>10} "
            f"{analysis.get('bw_required_80pct_gbs', 0):>10.1f} GB/s "
            f"{analysis.get('time_at_80pct_us', 0):>10.2f} μs"
        )

    print(f"{'=' * 120}\n")


def generate_roofline_plot(
    profiles_map: dict[str, list[AttentionBandwidthProfile]],
    hardware: HardwareSpec,
    output_path: Path,
) -> None:
    """Generate roofline plot for attention analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    fig, ax = plt.subplots(figsize=(14, 9))

    # Generate roofline envelope
    ai_range = np.logspace(-1, 3, 500)
    memory_roof = (ai_range * hardware.peak_bw_gbs) / 1000  # Convert to TFLOPS
    compute_roof = np.full_like(ai_range, hardware.peak_tflops_fp16)
    roof = np.minimum(memory_roof, compute_roof)

    # Plot roofline
    ax.loglog(ai_range, roof, 'b-', linewidth=3, label=f'Roofline ({hardware.name})')
    ax.loglog(
        ai_range[ai_range < hardware.ridge_point],
        memory_roof[ai_range < hardware.ridge_point],
        'b--', linewidth=1.5, alpha=0.5, label=f'Memory Ceiling ({hardware.peak_bw_gbs:.0f} GB/s)'
    )
    ax.loglog(
        ai_range[ai_range >= hardware.ridge_point],
        compute_roof[ai_range >= hardware.ridge_point],
        'r--', linewidth=1.5, alpha=0.5, label=f'Compute Ceiling ({hardware.peak_tflops_fp16:.0f} TFLOPS)'
    )

    # Ridge point line
    ax.axvline(hardware.ridge_point, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    ax.annotate(
        f'Ridge Point\n{hardware.ridge_point:.1f} FLOP/byte',
        xy=(hardware.ridge_point, hardware.peak_tflops_fp16 * 0.1),
        xytext=(hardware.ridge_point * 3, hardware.peak_tflops_fp16 * 0.05),
        fontsize=10,
        color='gray',
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
    )

    # Shade regions
    ax.axvspan(0.1, hardware.ridge_point, alpha=0.1, color='blue')
    ax.axvspan(hardware.ridge_point, 1000, alpha=0.1, color='red')
    ax.text(2, hardware.peak_tflops_fp16 * 0.02, 'Memory-Bound\nRegion',
            fontsize=12, ha='center', color='blue', alpha=0.7)
    ax.text(200, hardware.peak_tflops_fp16 * 0.7, 'Compute-Bound\nRegion',
            fontsize=12, ha='center', color='red', alpha=0.7)

    # Plot profiles
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, (config_name, profiles) in enumerate(profiles_map.items()):
        for j, profile in enumerate(profiles):
            max_tflops = hardware.max_tflops_at_ai(profile.arithmetic_intensity)

            color = colors[i % len(colors)]
            marker = markers[j % len(markers)]

            ax.plot(
                profile.arithmetic_intensity,
                max_tflops,
                marker,
                color=color,
                markersize=12,
                markeredgecolor='black',
                markeredgewidth=1,
                label=f'{config_name} ({profile.impl_type})',
            )

            # Add annotation
            seq_len = profile.config.seq_len_q
            ax.annotate(
                f'S={seq_len}',
                (profile.arithmetic_intensity, max_tflops),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
            )

    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (TFLOPS)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Attention Roofline Analysis - {hardware.name}\n' +
        f'Peak: {hardware.peak_bw_gbs:.0f} GB/s, {hardware.peak_tflops_fp16:.0f} TFLOPS',
        fontsize=16,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    ax.set_xlim(0.5, 500)
    ax.set_ylim(0.1, hardware.peak_tflops_fp16 * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Roofline plot saved to {output_path}")


def generate_bandwidth_sweep_plot(
    seq_lengths: list[int],
    hardware_list: list[HardwareSpec],
    output_path: Path,
) -> None:
    """Generate plot showing bandwidth requirements across sequence lengths."""
    try:
        import matplotlib.pyplot as plt
import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Config: B=1, H=32, D=64
    config_base = AttentionConfig(1, 32, 512, 512, 64, False, 2)

    for hw in hardware_list:
        std_bytes = []
        flash_bytes = []
        std_ai = []
        flash_ai = []
        std_bound = []
        flash_bound = []

        for seq_len in seq_lengths:
            config = AttentionConfig(1, 32, seq_len, seq_len, 64, False, 2)

            std = analyze_standard_attention(config)
            flash = analyze_flash_attention(config)

            std_bytes.append(std.total_bytes_moved / (1024**3))  # GB
            flash_bytes.append(flash.total_bytes_moved / (1024**3))
            std_ai.append(std.arithmetic_intensity)
            flash_ai.append(flash.arithmetic_intensity)
            std_bound.append(std.analyze(hw)['bound_type'])
            flash_bound.append(flash.analyze(hw)['bound_type'])

        # Plot 1: Total bytes moved
        axes[0, 0].semilogy(seq_lengths, std_bytes, 'o-', label=f'{hw.name} (Standard)')
        axes[0, 0].semilogy(seq_lengths, flash_bytes, 's--', label=f'{hw.name} (Flash)')

        # Plot 2: Arithmetic intensity
        axes[0, 1].semilogx(seq_lengths, std_ai, 'o-', label=f'{hw.name} (Standard)')
        axes[0, 1].semilogx(seq_lengths, flash_ai, 's--', label=f'{hw.name} (Flash)')
        axes[0, 1].axhline(hw.ridge_point, linestyle=':', alpha=0.5)

        # Plot 3: Memory saved by Flash Attention
        savings = [(s - f) / s * 100 for s, f in zip(std_bytes, flash_bytes)]
        axes[1, 0].plot(seq_lengths, savings, 'o-', label=hw.name)

        # Plot 4: Bound type transitions
        # Convert to numeric: 0=memory, 1=compute
        std_numeric = [0 if b == 'memory' else 1 for b in std_bound]
        axes[1, 1].plot(seq_lengths, std_numeric, 'o-', label=f'{hw.name} (Standard)')

    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Total Bytes Moved (GB)')
    axes[0, 0].set_title('Memory Traffic vs Sequence Length')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Arithmetic Intensity (FLOP/byte)')
    axes[0, 1].set_title('Arithmetic Intensity vs Sequence Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Memory Saved (%)')
    axes[1, 0].set_title('Flash Attention Memory Savings vs Standard')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].set_ylabel('Bound Type (0=Memory, 1=Compute)')
    axes[1, 1].set_title('Compute vs Memory Bound Classification')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bandwidth sweep plot saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Theoretical memory bandwidth analysis for attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default config
  uv run python analyze_attention_bandwidth.py
  
  # Analyze specific configuration
  uv run python analyze_attention_bandwidth.py --batch 4 --seq-len 2048 --heads 32
  
  # Compare across multiple sequence lengths and generate plots
  uv run python analyze_attention_bandwidth.py --sweep --roofline --hardware A100_80
  
  # Compare different hardware platforms
  uv run python analyze_attention_bandwidth.py --compare-hardware --sweep
        """,
    )

    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--seq-len-kv", type=int, default=None, help="KV sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--hardware", type=str, default="M3_MAX",
                        choices=list(HARDWARE_SPECS.keys()),
                        help="Target hardware platform")

    parser.add_argument("--sweep", action="store_true",
                        help="Sweep across sequence lengths")
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192, 16384],
                        help="Sequence lengths for sweep")
    parser.add_argument("--compare", action="store_true",
                        help="Compare standard vs flash attention")
    parser.add_argument("--compare-hardware", action="store_true",
                        help="Compare across different hardware platforms")

    parser.add_argument("--roofline", action="store_true",
                        help="Generate roofline plot")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Setup
    hardware = HARDWARE_SPECS[args.hardware]
    seq_kv = args.seq_len_kv or args.seq_len

    # Collect profiles
    all_profiles: dict[str, list[AttentionBandwidthProfile]] = {}

    if args.sweep:
        print(f"\n🔍 Sweeping across sequence lengths: {args.seq_lens}")
        print(f"   Hardware: {hardware.name}")
        print(f"   Config: B={args.batch}, H={args.heads}, D={args.head_dim}")

        profiles = []
        for seq_len in args.seq_lens:
            config = AttentionConfig(
                args.batch, args.heads, seq_len, seq_len,
                args.head_dim, args.causal, 2
            )
            profile = analyze_standard_attention(config) if not args.compare else analyze_flash_attention(config)
            profiles.append(profile)

            # Print brief summary
            analysis = profile.analyze(hardware)
            print(f"  Seq {seq_len:5d}: {format_bytes(profile.total_bytes_moved):>12}, "
                  f"AI={profile.arithmetic_intensity:>6.1f}, "
                  f"Bound={analysis['bound_type']}")

        all_profiles[f"B={args.batch},H={args.heads}"] = profiles

        # Print detailed report for last config
        if profiles:
            print_profile(profiles[-1], hardware)

    elif args.compare:
        config = AttentionConfig(
            args.batch, args.heads, args.seq_len, seq_kv,
            args.head_dim, args.causal, 2
        )

        profiles = [
            analyze_standard_attention(config),
            analyze_flash_attention(config),
            analyze_memory_efficient_attention(config),
        ]

        for p in profiles:
            print_profile(p, hardware)

        print_comparison_table(profiles, hardware)
        all_profiles["comparison"] = profiles

    else:
        config = AttentionConfig(
            args.batch, args.heads, args.seq_len, seq_kv,
            args.head_dim, args.causal, 2
        )
        profile = analyze_standard_attention(config)
        print_profile(profile, hardware)
        all_profiles["single"] = [profile]

    # Generate plots
    if args.roofline and all_profiles:
        plot_path = args.output.with_suffix('.png') if args.output else Path("attention_roofline.png")
        generate_roofline_plot(all_profiles, hardware, plot_path)

    if args.compare_hardware:
        hardware_list = [HARDWARE_SPECS["M3_MAX"], HARDWARE_SPECS["A100_80"], HARDWARE_SPECS["H100"]]
        plot_path = Path("attention_hardware_comparison.png")
        generate_bandwidth_sweep_plot(args.seq_lens, hardware_list, plot_path)

    # Save JSON output
    if args.output:
        output_data = {
            "hardware": {
                "name": hardware.name,
                "peak_bw_gbs": hardware.peak_bw_gbs,
                "peak_tflops_fp16": hardware.peak_tflops_fp16,
                "ridge_point": hardware.ridge_point,
            },
            "profiles": {
                name: [p.to_dict() for p in profiles]
                for name, profiles in all_profiles.items()
            },
        }
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
