#!/usr/bin/env python3
"""
Memory Bandwidth Profiler for Attention Operations

Measures detailed memory bandwidth usage for attention mechanisms:
- Bytes loaded per operation
- Bytes stored per operation  
- Arithmetic intensity (FLOPs/byte)
- Bandwidth utilization percentage
- Compute vs memory-bound classification

Supports multiple attention implementations:
- Standard SDPA (PyTorch)
- Flash Attention V2 (metal_marlin)
- Fused MPS Graph attention

Usage:
    uv run python profile_attention_bandwidth.py
    uv run python profile_attention_bandwidth.py --seq-len 4096 --batch 4 --flash
    uv run python profile_attention_bandwidth.py --compare --roofline --output report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_TORCH, torch  # noqa: E402

if HAS_TORCH and torch is not None:
    import torch.nn.functional as F  # noqa: E402


# =============================================================================
# Hardware Specifications
# =============================================================================

@dataclass(frozen=True)
class HardwareSpec:
    """Hardware specifications for bandwidth calculations."""
    name: str
    peak_bw_gbs: float          # Peak memory bandwidth (GB/s)
    peak_tflops_fp16: float    # Peak FP16 compute (TFLOPS)
    peak_tflops_fp32: float    # Peak FP32 compute (TFLOPS)

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at ridge point (FLOP/byte)."""
        return (self.peak_tflops_fp16 * 1000) / self.peak_bw_gbs


# Common hardware specs
HARDWARE_SPECS = {
    "M4_MAX": HardwareSpec("Apple M4 Max", 546.0, 32.0, 16.0),
    "M3_MAX": HardwareSpec("Apple M3 Max", 400.0, 16.4, 8.2),
    "M2_MAX": HardwareSpec("Apple M2 Max", 400.0, 13.6, 6.8),
    "M1_MAX": HardwareSpec("Apple M1 Max", 400.0, 10.4, 5.2),
    "V100": HardwareSpec("NVIDIA V100", 900.0, 31.4, 15.7),
    "A100": HardwareSpec("NVIDIA A100", 2039.0, 78.0, 19.5),
    "H100": HardwareSpec("NVIDIA H100", 3352.0, 197.9, 51.0),
}


def detect_hardware() -> HardwareSpec:
    """Detect current hardware spec."""
    if HAS_TORCH and torch is not None:
        if torch.backends.mps.is_available():
            # Try to detect specific Apple Silicon variant
            # Default to M3 Max as middle ground
            return HARDWARE_SPECS["M3_MAX"]
        elif torch.cuda.is_available():
            name = torch.cuda.get_device_name().lower()
            if "h100" in name:
                return HARDWARE_SPECS["H100"]
            elif "a100" in name:
                return HARDWARE_SPECS["A100"]
            elif "v100" in name:
                return HARDWARE_SPECS["V100"]
    return HARDWARE_SPECS["M3_MAX"]  # Default


# =============================================================================
# Attention Memory Analysis
# =============================================================================

@dataclass
class AttentionOpMetrics:
    """Memory and compute metrics for a single attention operation."""

    # Operation identification
    name: str
    batch_size: int
    num_heads: int
    seq_len_q: int
    seq_len_kv: int
    head_dim: int
    causal: bool

    # Memory traffic (bytes)
    q_bytes_loaded: int
    k_bytes_loaded: int
    v_bytes_loaded: int
    output_bytes_stored: int
    intermediate_bytes: int          # QK^T, softmax intermediates

    # FLOPs
    qk_flops: int                    # Q @ K^T
    softmax_flops: int               # Softmax computation
    av_flops: int                    # Attention @ V
    total_flops: int

    # Derived metrics
    total_bytes_loaded: int = field(init=False)
    total_bytes_stored: int = field(init=False)
    total_bytes_moved: int = field(init=False)
    arithmetic_intensity: float = field(init=False)

    def __post_init__(self):
        self.total_bytes_loaded = self.q_bytes_loaded + self.k_bytes_loaded + self.v_bytes_loaded
        self.total_bytes_stored = self.output_bytes_stored + self.intermediate_bytes
        self.total_bytes_moved = self.total_bytes_loaded + self.total_bytes_stored
        self.arithmetic_intensity = (
            self.total_flops / self.total_bytes_moved if self.total_bytes_moved > 0 else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": {
                "batch": self.batch_size,
                "heads": self.num_heads,
                "seq_q": self.seq_len_q,
                "seq_kv": self.seq_len_kv,
                "head_dim": self.head_dim,
            },
            "causal": self.causal,
            "memory": {
                "q_bytes_loaded": self.q_bytes_loaded,
                "k_bytes_loaded": self.k_bytes_loaded,
                "v_bytes_loaded": self.v_bytes_loaded,
                "output_bytes_stored": self.output_bytes_stored,
                "intermediate_bytes": self.intermediate_bytes,
                "total_bytes_loaded": self.total_bytes_loaded,
                "total_bytes_stored": self.total_bytes_stored,
                "total_bytes_moved": self.total_bytes_moved,
                "total_mb": self.total_bytes_moved / (1024 * 1024),
            },
            "compute": {
                "qk_flops": self.qk_flops,
                "softmax_flops": self.softmax_flops,
                "av_flops": self.av_flops,
                "total_flops": self.total_flops,
                "total_gflops": self.total_flops / 1e9,
            },
            "arithmetic_intensity": self.arithmetic_intensity,
        }


def analyze_standard_attention(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    head_dim: int,
    dtype_bytes: int = 2,
    causal: bool = False,
) -> AttentionOpMetrics:
    """
    Analyze memory traffic for standard (non-fused) attention.
    
    Standard attention materializes the full QK^T matrix.
    """
    # Memory traffic - each element read/written
    q_bytes = batch_size * num_heads * seq_len_q * head_dim * dtype_bytes
    k_bytes = batch_size * num_heads * seq_len_kv * head_dim * dtype_bytes
    v_bytes = batch_size * num_heads * seq_len_kv * head_dim * dtype_bytes
    output_bytes = batch_size * num_heads * seq_len_q * head_dim * dtype_bytes

    # Intermediate matrices (materialized)
    qk_intermediate = batch_size * num_heads * seq_len_q * seq_len_kv * dtype_bytes
    softmax_intermediate = qk_intermediate  # softmax output

    total_intermediate = qk_intermediate + softmax_intermediate

    # FLOPs calculation
    causal_factor = 0.5 if causal else 1.0

    # Q @ K^T: 2 * B * H * Sq * Sk * D
    qk_flops = int(2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim * causal_factor)

    # Softmax: ~5 * B * H * Sq * Sk (subtract max, exp, sum, divide)
    softmax_flops = int(5 * batch_size * num_heads * seq_len_q * seq_len_kv * causal_factor)

    # Attn @ V: 2 * B * H * Sq * Sk * D
    av_flops = int(2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim * causal_factor)

    return AttentionOpMetrics(
        name="standard_attention",
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        head_dim=head_dim,
        causal=causal,
        q_bytes_loaded=q_bytes,
        k_bytes_loaded=k_bytes,
        v_bytes_loaded=v_bytes,
        output_bytes_stored=output_bytes,
        intermediate_bytes=total_intermediate,
        qk_flops=qk_flops,
        softmax_flops=softmax_flops,
        av_flops=av_flops,
        total_flops=qk_flops + softmax_flops + av_flops,
    )


def analyze_flash_attention(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    head_dim: int,
    dtype_bytes: int = 2,
    causal: bool = False,
    block_size_q: int = 64,
    block_size_kv: int = 64,
) -> AttentionOpMetrics:
    """
    Analyze memory traffic for Flash Attention (tiled, fused).
    
    Flash Attention never materializes the full QK^T matrix.
    It processes in tiles that fit in SRAM.
    """
    # Input/output memory same as standard
    q_bytes = batch_size * num_heads * seq_len_q * head_dim * dtype_bytes
    k_bytes = batch_size * num_heads * seq_len_kv * head_dim * dtype_bytes
    v_bytes = batch_size * num_heads * seq_len_kv * head_dim * dtype_bytes
    output_bytes = batch_size * num_heads * seq_len_q * head_dim * dtype_bytes

    # Flash Attention reduces intermediate memory dramatically
    # Only needs O(block_size) intermediate storage, not O(seq^2)
    # We approximate this as a small fraction of the full intermediate
    num_blocks_q = (seq_len_q + block_size_q - 1) // block_size_q
    num_blocks_kv = (seq_len_kv + block_size_kv - 1) // block_size_kv

    # Each block processes a tile
    tile_intermediate = block_size_q * block_size_kv * dtype_bytes
    # Flash attention loads K/V tiles multiple times
    kv_load_factor = num_blocks_q if not causal else num_blocks_q // 2 + 1

    # Minimal intermediate - just tile buffers in SRAM
    total_intermediate = batch_size * num_heads * tile_intermediate * 2

    # FLOPs same as standard attention
    causal_factor = 0.5 if causal else 1.0
    qk_flops = int(2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim * causal_factor)
    softmax_flops = int(5 * batch_size * num_heads * seq_len_q * seq_len_kv * causal_factor)
    av_flops = int(2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim * causal_factor)

    return AttentionOpMetrics(
        name="flash_attention",
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        head_dim=head_dim,
        causal=causal,
        q_bytes_loaded=q_bytes,
        k_bytes_loaded=k_bytes * kv_load_factor,  # K loaded multiple times
        v_bytes_loaded=v_bytes * kv_load_factor,  # V loaded multiple times
        output_bytes_stored=output_bytes,
        intermediate_bytes=total_intermediate,
        qk_flops=qk_flops,
        softmax_flops=softmax_flops,
        av_flops=av_flops,
        total_flops=qk_flops + softmax_flops + av_flops,
    )


# =============================================================================
# Bandwidth Profiling
# =============================================================================

@dataclass
class BandwidthProfile:
    """Complete bandwidth profile for an attention operation."""

    # Theoretical metrics
    theoretical: AttentionOpMetrics

    # Timing
    elapsed_ms: float
    elapsed_std_ms: float

    # Achieved metrics
    achieved_bandwidth_gbs: float = field(init=False)
    achieved_tflops: float = field(init=False)

    # Utilization
    bandwidth_util_pct: float = 0.0
    compute_util_pct: float = 0.0

    # Classification
    bound_type: str = "unknown"
    efficiency_score: float = 0.0

    def __post_init__(self):
        if self.elapsed_ms > 0:
            elapsed_s = self.elapsed_ms / 1000.0
            self.achieved_bandwidth_gbs = (self.theoretical.total_bytes_moved / elapsed_s) / 1e9
            self.achieved_tflops = (self.theoretical.total_flops / elapsed_s) / 1e12

    def analyze(self, hardware: HardwareSpec) -> None:
        """Analyze against hardware limits."""
        # Calculate utilization percentages
        self.bandwidth_util_pct = (self.achieved_bandwidth_gbs / hardware.peak_bw_gbs) * 100
        self.compute_util_pct = (self.achieved_tflops / hardware.peak_tflops_fp16) * 100

        # Determine if compute or memory bound
        ridge = hardware.ridge_point

        if self.theoretical.arithmetic_intensity < ridge:
            self.bound_type = "memory"
            # Max achievable at this AI
            max_tflops_at_ai = (self.theoretical.arithmetic_intensity * hardware.peak_bw_gbs) / 1000
            self.efficiency_score = (
                (self.achieved_tflops / max_tflops_at_ai) * 100 if max_tflops_at_ai > 0 else 0
            )
        else:
            self.bound_type = "compute"
            self.efficiency_score = self.compute_util_pct

    def to_dict(self) -> dict[str, Any]:
        return {
            "theoretical": self.theoretical.to_dict(),
            "timing": {
                "elapsed_ms": self.elapsed_ms,
                "elapsed_std_ms": self.elapsed_std_ms,
            },
            "achieved": {
                "bandwidth_gbs": self.achieved_bandwidth_gbs,
                "tflops": self.achieved_tflops,
            },
            "utilization": {
                "bandwidth_pct": self.bandwidth_util_pct,
                "compute_pct": self.compute_util_pct,
            },
            "analysis": {
                "bound_type": self.bound_type,
                "efficiency_score": self.efficiency_score,
            },
        }


def profile_attention_bandwidth(
    name: str,
    fn: Callable[[], torch.Tensor],
    metrics: AttentionOpMetrics,
    hardware: HardwareSpec,
    warmup: int = 5,
    iterations: int = 20,
) -> BandwidthProfile:
    """Profile bandwidth for an attention implementation."""

    def _sync():
        if HAS_TORCH and torch is not None:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = fn()
        _sync()

    # Timed iterations
    times: list[float] = []
    for _ in range(iterations):
        _sync()
        start = time.perf_counter()
        _ = fn()
        _sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

    median_ms = statistics.median(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0

    profile = BandwidthProfile(
        theoretical=metrics,
        elapsed_ms=median_ms,
        elapsed_std_ms=std_ms,
    )
    profile.analyze(hardware)
    return profile


# =============================================================================
# Implementation Benchmarks
# =============================================================================

def benchmark_pytorch_sdpa(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    hardware: HardwareSpec,
) -> BandwidthProfile:
    """Benchmark PyTorch's scaled_dot_product_attention."""
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch not available")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    q = torch.randn(batch, heads, seq_q, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_kv, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_kv, head_dim, device=device, dtype=torch.float16)
    scale = head_dim ** -0.5

    # PyTorch SDPA uses fused kernels when available
    metrics = analyze_flash_attention(batch, heads, seq_q, seq_kv, head_dim, causal=causal)
    metrics.name = "pytorch_sdpa"

    def fn():
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    return profile_attention_bandwidth("pytorch_sdpa", fn, metrics, hardware)


def benchmark_naive_attention(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    hardware: HardwareSpec,
) -> BandwidthProfile:
    """Benchmark naive (non-fused) attention implementation."""
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch not available")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    q = torch.randn(batch, heads, seq_q, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_kv, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_kv, head_dim, device=device, dtype=torch.float16)
    scale = head_dim ** -0.5

    # Naive implementation materializes intermediates
    metrics = analyze_standard_attention(batch, heads, seq_q, seq_kv, head_dim, causal=causal)
    metrics.name = "naive_attention"

    def fn():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(seq_q, seq_kv, device=device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    # Ensure fn is defined even if torch not available
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch not available")

    return profile_attention_bandwidth("naive_attention", fn, metrics, hardware)


def benchmark_flash_attention(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    hardware: HardwareSpec,
) -> BandwidthProfile | None:
    """Benchmark Flash Attention V2 if available."""
    try:
        from metal_marlin.flash_attention_v2 import flash_attention_v2
    except ImportError:
        return None

    if not HAS_TORCH or torch is None:
        return None

    if not torch.backends.mps.is_available():
        return None

    device = "mps"

    q = torch.randn(batch, heads, seq_q, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_kv, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_kv, head_dim, device=device, dtype=torch.float16)
    scale = head_dim ** -0.5

    metrics = analyze_flash_attention(batch, heads, seq_q, seq_kv, head_dim, causal=causal)
    metrics.name = "flash_attention_v2"

    try:
        # Test if flash attention works
        _ = flash_attention_v2(q, k, v, scale=scale, causal=causal)
    except Exception as e:
        print(f"  Flash Attention not available: {e}")
        return None

    def fn():
        return flash_attention_v2(q, k, v, scale=scale, causal=causal)

    return profile_attention_bandwidth("flash_attention_v2", fn, metrics, hardware)


# =============================================================================
# Reporting
# =============================================================================

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def print_bandwidth_report(profiles: list[BandwidthProfile], hardware: HardwareSpec) -> None:
    """Print detailed bandwidth analysis report."""
    print("\n" + "=" * 100)
    print(f"ATTENTION MEMORY BANDWIDTH PROFILE - {hardware.name}")
    print("=" * 100)
    print(f"Hardware: Peak BW = {hardware.peak_bw_gbs:.1f} GB/s, "
          f"Peak FP16 = {hardware.peak_tflops_fp16:.1f} TFLOPS")
    print(f"Ridge Point: {hardware.ridge_point:.1f} FLOP/byte")
    print("=" * 100)

    for profile in profiles:
        m = profile.theoretical
        print(f"\n┌{'─' * 98}┐")
        print(f"│ {m.name.upper():<96} │")
        print(f"├{'─' * 98}┤")
        print(f"│ Shape: B={m.batch_size}, H={m.num_heads}, Sq={m.seq_len_q}, Sk={m.seq_len_kv}, D={m.head_dim}, Causal={m.causal}")
        print(f"├{'─' * 98}┤")
        print("│ MEMORY TRAFFIC")
        print("│   Bytes Loaded:")
        print(f"│     Q:        {format_bytes(m.q_bytes_loaded):>12} ({m.q_bytes_loaded / m.total_bytes_moved * 100:.1f}%)")
        print(f"│     K:        {format_bytes(m.k_bytes_loaded):>12} ({m.k_bytes_loaded / m.total_bytes_moved * 100:.1f}%)")
        print(f"│     V:        {format_bytes(m.v_bytes_loaded):>12} ({m.v_bytes_loaded / m.total_bytes_moved * 100:.1f}%)")
        print(f"│     Subtotal: {format_bytes(m.total_bytes_loaded):>12}")
        print("│   Bytes Stored:")
        print(f"│     Output:       {format_bytes(m.output_bytes_stored):>12}")
        print(f"│     Intermediate: {format_bytes(m.intermediate_bytes):>12}")
        print(f"│     Subtotal:     {format_bytes(m.total_bytes_stored):>12}")
        print(f"│   TOTAL: {format_bytes(m.total_bytes_moved):>12}")
        print(f"├{'─' * 98}┤")
        print("│ COMPUTE")
        print(f"│   QK matmul:   {m.qk_flops / 1e9:.2f} GFLOPs")
        print(f"│   Softmax:     {m.softmax_flops / 1e9:.2f} GFLOPs")
        print(f"│   AV matmul:   {m.av_flops / 1e9:.2f} GFLOPs")
        print(f"│   TOTAL:       {m.total_flops / 1e9:.2f} GFLOPs")
        print(f"├{'─' * 98}┤")
        print("│ METRICS")
        print(f"│   Arithmetic Intensity: {m.arithmetic_intensity:.2f} FLOP/byte")
        print(f"│   Time: {profile.elapsed_ms:.3f} ± {profile.elapsed_std_ms:.3f} ms")
        print(f"│   Achieved BW: {profile.achieved_bandwidth_gbs:.1f} GB/s ({profile.bandwidth_util_pct:.1f}% of peak)")
        print(f"│   Achieved Compute: {profile.achieved_tflops:.2f} TFLOPS ({profile.compute_util_pct:.1f}% of peak)")
        print(f"│   BOUND TYPE: {profile.bound_type.upper()}")
        print(f"│   Efficiency: {profile.efficiency_score:.1f}%")
        print(f"└{'─' * 98}┘")


def print_comparison_table(profiles: list[BandwidthProfile], hardware: HardwareSpec) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)

    header = (
        f"{'Implementation':<25} {'Bytes Loaded':>14} {'Bytes Stored':>14} "
        f"{'Total MB':>12} {'AI':>8} {'BW %':>8} {'Compute %':>10} {'Bound':>10}"
    )
    print(header)
    print("-" * len(header))

    for profile in profiles:
        m = profile.theoretical
        print(
            f"{m.name:<25} "
            f"{format_bytes(m.total_bytes_loaded):>14} "
            f"{format_bytes(m.total_bytes_stored):>14} "
            f"{m.total_bytes_moved / 1024 / 1024:>10.1f} MB "
            f"{m.arithmetic_intensity:>8.1f} "
            f"{profile.bandwidth_util_pct:>7.1f}% "
            f"{profile.compute_util_pct:>9.1f}% "
            f"{profile.bound_type:>10}"
        )

    print("=" * 120)


def generate_roofline_plot(
    profiles: list[BandwidthProfile],
    hardware: HardwareSpec,
    output_path: Path,
) -> None:
    """Generate roofline plot showing memory vs compute bound."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot roofline envelope
    ai_range = np.logspace(-1, 3, 500)
    memory_roof = (ai_range * hardware.peak_bw_gbs) / 1000  # Convert to TFLOPS
    compute_roof = np.full_like(ai_range, hardware.peak_tflops_fp16)
    roof = np.minimum(memory_roof, compute_roof)

    ax.loglog(ai_range, roof, 'b-', linewidth=2.5, label=f'Roofline ({hardware.name})')

    # Ridge point
    ridge = hardware.ridge_point
    ax.axvline(ridge, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.annotate(
        f'Ridge: {ridge:.1f} F/B',
        xy=(ridge, hardware.peak_tflops_fp16 * 0.3),
        fontsize=9,
        color='gray',
    )

    # Plot each profile
    colors = plt.cm.Set1.colors
    for i, profile in enumerate(profiles):
        m = profile.theoretical
        color = colors[i % len(colors)]

        ax.plot(
            m.arithmetic_intensity,
            profile.achieved_tflops,
            'o',
            color=color,
            markersize=12,
            label=m.name,
        )

        # Color code by bound type
        if profile.bound_type == "memory":
            ax.plot(
                m.arithmetic_intensity,
                profile.achieved_tflops,
                's',
                color=color,
                markersize=8,
                markerfacecolor='none',
                markeredgewidth=2,
            )

        ax.annotate(
            f"{m.name}\n{profile.bandwidth_util_pct:.1f}% BW",
            (m.arithmetic_intensity, profile.achieved_tflops),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
        )

    # Shade memory-bound and compute-bound regions
    ax.axvspan(ai_range[0], ridge, alpha=0.1, color='blue', label='Memory-Bound Region')
    ax.axvspan(ridge, ai_range[-1], alpha=0.1, color='red', label='Compute-Bound Region')

    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
    ax.set_title(f'Attention Roofline Analysis - {hardware.name}', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=9)

    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.1, hardware.peak_tflops_fp16 * 1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nRoofline plot saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Profile memory bandwidth usage in attention operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling with defaults
  uv run python profile_attention_bandwidth.py
  
  # Profile specific configuration
  uv run python profile_attention_bandwidth.py --batch 4 --seq-len 2048 --heads 32
  
  # Compare implementations and generate roofline plot
  uv run python profile_attention_bandwidth.py --compare --roofline
  
  # Full sweep across sequence lengths
  uv run python profile_attention_bandwidth.py --sweep --output results.json
        """,
    )

    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--seq-len-kv", type=int, default=None, help="KV sequence length (defaults to seq-len)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--hardware", type=str, default=None, help="Hardware spec (M4_MAX, V100, etc.)")

    parser.add_argument("--compare", action="store_true", help="Compare all implementations")
    parser.add_argument("--roofline", action="store_true", help="Generate roofline plot")
    parser.add_argument("--sweep", action="store_true", help="Sweep across sequence lengths")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Setup
    seq_kv = args.seq_len_kv or args.seq_len
    hardware = HARDWARE_SPECS.get(args.hardware, detect_hardware())

    profiles: list[BandwidthProfile] = []

    if args.sweep:
        # Sweep across sequence lengths
        seq_lengths = [512, 1024, 2048, 4096, 8192]
        print(f"\nSweeping across sequence lengths: {seq_lengths}")

        for seq_len in seq_lengths:
            print(f"\n--- Seq Len: {seq_len} ---")
            try:
                profile = benchmark_pytorch_sdpa(
                    args.batch, args.heads, seq_len, seq_len, args.head_dim,
                    args.causal, hardware
                )
                profiles.append(profile)
                print(f"  PyTorch SDPA: {profile.elapsed_ms:.3f} ms, "
                      f"BW={profile.bandwidth_util_pct:.1f}%, "
                      f"Bound={profile.bound_type}")
            except Exception as e:
                print(f"  Failed: {e}")

    elif args.compare:
        # Compare different implementations
        implementations = [
            ("PyTorch SDPA", benchmark_pytorch_sdpa),
            ("Naive Attention", benchmark_naive_attention),
        ]

        for name, bench_fn in implementations:
            print(f"\nProfiling {name}...")
            try:
                profile = bench_fn(
                    args.batch, args.heads, args.seq_len, seq_kv, args.head_dim,
                    args.causal, hardware
                )
                profiles.append(profile)
            except Exception as e:
                print(f"  Failed: {e}")

        # Try Flash Attention
        profile = benchmark_flash_attention(
            args.batch, args.heads, args.seq_len, seq_kv, args.head_dim,
            args.causal, hardware
        )
        if profile:
            profiles.append(profile)

    else:
        # Single implementation
        profile = benchmark_pytorch_sdpa(
            args.batch, args.heads, args.seq_len, seq_kv, args.head_dim,
            args.causal, hardware
        )
        profiles.append(profile)

    # Reporting
    if profiles:
        print_bandwidth_report(profiles, hardware)

        if len(profiles) > 1:
            print_comparison_table(profiles, hardware)

        if args.roofline:
            plot_path = args.output.with_suffix('.png') if args.output else Path("roofline_attention.png")
            generate_roofline_plot(profiles, hardware, plot_path)

        if args.output:
            output_data = {
                "hardware": {
                    "name": hardware.name,
                    "peak_bw_gbs": hardware.peak_bw_gbs,
                    "peak_tflops_fp16": hardware.peak_tflops_fp16,
                    "ridge_point": hardware.ridge_point,
                },
                "profiles": [p.to_dict() for p in profiles],
            }
            args.output.write_text(json.dumps(output_data, indent=2))
            print(f"\nResults saved to {args.output}")
    else:
        print("No successful profiles collected")


if __name__ == "__main__":
    main()
