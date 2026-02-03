#!/usr/bin/env python3
"""
Benchmark Flash Attention v3 vs v2 across context lengths 1K to 32K.

This script compares the performance of Flash Attention V3 (optimized causal
masking, improved register usage) against Flash Attention V2 (baseline)
across various context lengths relevant for LLM inference.

Usage:
    cd contrib/metal_marlin
    uv run python scripts/benchmark_fa3.py

    # With specific batch size and head configuration
    uv run python scripts/benchmark_fa3.py --batch 4 --heads 32 --head-dim 128

    # Export results to JSON
    uv run python scripts/benchmark_fa3.py --export results/fa3_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "benchmarks"))

import torch
from framework import mps_sync  # noqa: E402

from metal_marlin._compat import HAS_MPS  # noqa: E402
from metal_marlin.flash_attention_v2 import (  # noqa: E402
    flash_attention_v2,
    flash_attention_v2_decode,
)

WARMUP_ITERS = 10
BENCH_ITERS = 50


@dataclass
class BenchmarkResult:
    """Result from a single attention benchmark configuration."""

    # Configuration
    batch: int
    num_heads: int
    head_dim: int
    seq_len: int
    context_len: int
    mode: str  # "prefill" or "decode"

    # Timing (ms)
    fa2_time_ms: float = 0.0
    fa2_std_ms: float = 0.0
    fa3_time_ms: float = 0.0
    fa3_std_ms: float = 0.0

    # Throughput metrics
    fa2_tflops: float = 0.0
    fa3_tflops: float = 0.0
    speedup: float = 0.0

    # Memory
    memory_mb: float = 0.0

    # Metadata
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _time_kernel(
    fn: Callable[[], torch.Tensor],
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> tuple[float, float]:
    """Time a kernel with warmup and MPS synchronization.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = fn()
        mps_sync()

    # Timed iterations
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = fn()
        mps_sync()
        times.append((time.perf_counter() - start) * 1000.0)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def benchmark_fa2_prefill(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = True,
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> tuple[float, float]:
    """Benchmark Flash Attention V2 for prefill phase.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length (context length for prefill)
        head_dim: Head dimension
        causal: Whether to use causal masking

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="mps")
    mps_sync()

    scale = 1.0 / math.sqrt(head_dim)

    def fn() -> torch.Tensor:
        return flash_attention_v2(q, k, v, scale=scale, causal=causal)

    return _time_kernel(fn, warmup=warmup, iterations=iterations)


def benchmark_fa2_decode(
    batch: int,
    num_heads: int,
    context_len: int,
    head_dim: int,
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> tuple[float, float]:
    """Benchmark Flash Attention V2 for decode phase (single token).

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        context_len: Context length (seq_k)
        head_dim: Head dimension

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Decode: seq_q=1, seq_k=context_len
    q = torch.randn(batch, num_heads, 1, head_dim, dtype=torch.float16, device="mps")
    k = torch.randn(batch, num_heads, context_len, head_dim, dtype=torch.float16, device="mps")
    v = torch.randn(batch, num_heads, context_len, head_dim, dtype=torch.float16, device="mps")
    mps_sync()

    scale = 1.0 / math.sqrt(head_dim)

    def fn() -> torch.Tensor:
        return flash_attention_v2_decode(q, k, v, scale=scale)

    return _time_kernel(fn, warmup=warmup, iterations=iterations)


def benchmark_fa3_prefill(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = True,
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> tuple[float, float, str]:
    """Benchmark Flash Attention V3 for prefill phase.

    Note: Flash Attention V3 Python interface is not yet implemented.
    This function will return placeholder values and a note.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length (context length for prefill)
        head_dim: Head dimension
        causal: Whether to use causal masking

    Returns:
        Tuple of (mean_time_ms, std_time_ms, notes)
    """
    # TODO: Implement FA3 Python binding when available
    # For now, return placeholder values
    return 0.0, 0.0, "FA3 Python interface not yet implemented"


def benchmark_fa3_decode(
    batch: int,
    num_heads: int,
    context_len: int,
    head_dim: int,
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> tuple[float, float, str]:
    """Benchmark Flash Attention V3 for decode phase (single token).

    Note: Flash Attention V3 Python interface is not yet implemented.
    This function will return placeholder values and a note.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        context_len: Context length (seq_k)
        head_dim: Head dimension

    Returns:
        Tuple of (mean_time_ms, std_time_ms, notes)
    """
    # TODO: Implement FA3 Python binding when available
    # For now, return placeholder values
    return 0.0, 0.0, "FA3 Python interface not yet implemented"


def calculate_attention_flops(
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    causal: bool = False,
) -> int:
    """Calculate FLOPs for attention computation.

    For causal attention, we compute roughly half the FLOPs.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        seq_q: Query sequence length
        seq_k: Key/value sequence length
        head_dim: Head dimension
        causal: Whether causal masking is applied

    Returns:
        Total FLOPs
    """
    # Q @ K^T: 2 * batch * heads * seq_q * seq_k * head_dim
    qk_flops = 2 * batch * num_heads * seq_q * seq_k * head_dim

    # Softmax + attn @ V: 2 * batch * heads * seq_q * seq_k * head_dim
    # (simplified, ignoring softmax overhead)
    av_flops = 2 * batch * num_heads * seq_q * seq_k * head_dim

    total_flops = qk_flops + av_flops

    # Causal reduces FLOPs by approximately half for square attention
    if causal and seq_q == seq_k:
        total_flops = total_flops // 2

    return total_flops


def calculate_memory_usage(
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> float:
    """Calculate memory usage in MB for attention tensors.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        seq_q: Query sequence length
        seq_k: Key/value sequence length
        head_dim: Head dimension

    Returns:
        Memory usage in MB
    """
    # Q: batch * num_heads * seq_q * head_dim * 2 bytes (FP16)
    q_bytes = batch * num_heads * seq_q * head_dim * 2

    # K: batch * num_heads * seq_k * head_dim * 2 bytes
    k_bytes = batch * num_heads * seq_k * head_dim * 2

    # V: batch * num_heads * seq_k * head_dim * 2 bytes
    v_bytes = batch * num_heads * seq_k * head_dim * 2

    # Output: batch * num_heads * seq_q * head_dim * 2 bytes
    o_bytes = batch * num_heads * seq_q * head_dim * 2

    total_bytes = q_bytes + k_bytes + v_bytes + o_bytes
    return total_bytes / (1024 * 1024)


def run_prefill_benchmarks(
    batch: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    context_lengths: list[int] | None = None,
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> list[BenchmarkResult]:
    """Run prefill benchmarks across context lengths.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        head_dim: Head dimension
        context_lengths: List of context lengths to benchmark

    Returns:
        List of BenchmarkResult objects
    """
    if context_lengths is None:
        context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    results: list[BenchmarkResult] = []

    print("\n" + "=" * 80)
    print(f"Prefill Benchmark: FA2 vs FA3 (B={batch}, H={num_heads}, D={head_dim})")
    print("=" * 80)
    print(f"{'Context':>10} {'FA2 (ms)':>12} {'FA3 (ms)':>12} {'Speedup':>10} {'FA2 TFLOPS':>12} {'FA3 TFLOPS':>12}")
    print("-" * 80)

    for ctx_len in context_lengths:
        # Skip if context length might cause OOM
        try:
            fa2_mean, fa2_std = benchmark_fa2_prefill(
                batch, num_heads, ctx_len, head_dim, causal=True, warmup=warmup, iterations=iterations
            )
            fa3_mean, fa3_std, notes = benchmark_fa3_prefill(
                batch, num_heads, ctx_len, head_dim, causal=True, warmup=warmup, iterations=iterations
            )

            # Calculate metrics
            flops = calculate_attention_flops(batch, num_heads, ctx_len, ctx_len, head_dim, causal=True)
            fa2_tflops = (flops / (fa2_mean / 1000.0)) / 1e12 if fa2_mean > 0 else 0.0
            fa3_tflops = (flops / (fa3_mean / 1000.0)) / 1e12 if fa3_mean > 0 else 0.0
            speedup = fa2_mean / fa3_mean if fa3_mean > 0 else 0.0
            memory_mb = calculate_memory_usage(batch, num_heads, ctx_len, ctx_len, head_dim)

            result = BenchmarkResult(
                batch=batch,
                num_heads=num_heads,
                head_dim=head_dim,
                seq_len=ctx_len,
                context_len=ctx_len,
                mode="prefill",
                fa2_time_ms=fa2_mean,
                fa2_std_ms=fa2_std,
                fa3_time_ms=fa3_mean,
                fa3_std_ms=fa3_std,
                fa2_tflops=fa2_tflops,
                fa3_tflops=fa3_tflops,
                speedup=speedup,
                memory_mb=memory_mb,
                notes=notes,
            )
            results.append(result)

            # Print progress
            fa3_display = f"{fa3_mean:.3f}" if fa3_mean > 0 else "N/A"
            speedup_display = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            fa3_tflops_display = f"{fa3_tflops:.2f}" if fa3_tflops > 0 else "N/A"

            print(f"{ctx_len:>10} {fa2_mean:>12.3f} {fa3_display:>12} {speedup_display:>10} {fa2_tflops:>12.2f} {fa3_tflops_display:>12}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                print(f"{ctx_len:>10} {'OOM':>12} {'OOM':>12} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
                results.append(BenchmarkResult(
                    batch=batch,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    seq_len=ctx_len,
                    context_len=ctx_len,
                    mode="prefill",
                    notes="OOM",
                ))
            else:
                raise

    return results


def run_decode_benchmarks(
    batch: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    context_lengths: list[int] | None = None,
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> list[BenchmarkResult]:
    """Run decode benchmarks across context lengths.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        head_dim: Head dimension
        context_lengths: List of context lengths to benchmark

    Returns:
        List of BenchmarkResult objects
    """
    if context_lengths is None:
        context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    results: list[BenchmarkResult] = []

    print("\n" + "=" * 80)
    print(f"Decode Benchmark: FA2 vs FA3 (B={batch}, H={num_heads}, D={head_dim})")
    print("=" * 80)
    print(f"{'Context':>10} {'FA2 (ms)':>12} {'FA3 (ms)':>12} {'Speedup':>10} {'FA2 TFLOPS':>12} {'FA3 TFLOPS':>12}")
    print("-" * 80)

    for ctx_len in context_lengths:
        try:
            fa2_mean, fa2_std = benchmark_fa2_decode(
                batch, num_heads, ctx_len, head_dim, warmup=warmup, iterations=iterations
            )
            fa3_mean, fa3_std, notes = benchmark_fa3_decode(
                batch, num_heads, ctx_len, head_dim, warmup=warmup, iterations=iterations
            )

            # Calculate metrics
            flops = calculate_attention_flops(batch, num_heads, 1, ctx_len, head_dim, causal=True)
            fa2_tflops = (flops / (fa2_mean / 1000.0)) / 1e12 if fa2_mean > 0 else 0.0
            fa3_tflops = (flops / (fa3_mean / 1000.0)) / 1e12 if fa3_mean > 0 else 0.0
            speedup = fa2_mean / fa3_mean if fa3_mean > 0 else 0.0
            memory_mb = calculate_memory_usage(batch, num_heads, 1, ctx_len, head_dim)

            result = BenchmarkResult(
                batch=batch,
                num_heads=num_heads,
                head_dim=head_dim,
                seq_len=1,
                context_len=ctx_len,
                mode="decode",
                fa2_time_ms=fa2_mean,
                fa2_std_ms=fa2_std,
                fa3_time_ms=fa3_mean,
                fa3_std_ms=fa3_std,
                fa2_tflops=fa2_tflops,
                fa3_tflops=fa3_tflops,
                speedup=speedup,
                memory_mb=memory_mb,
                notes=notes,
            )
            results.append(result)

            # Print progress
            fa3_display = f"{fa3_mean:.3f}" if fa3_mean > 0 else "N/A"
            speedup_display = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            fa3_tflops_display = f"{fa3_tflops:.2f}" if fa3_tflops > 0 else "N/A"

            print(f"{ctx_len:>10} {fa2_mean:>12.3f} {fa3_display:>12} {speedup_display:>10} {fa2_tflops:>12.2f} {fa3_tflops_display:>12}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                print(f"{ctx_len:>10} {'OOM':>12} {'OOM':>12} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
                results.append(BenchmarkResult(
                    batch=batch,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    seq_len=1,
                    context_len=ctx_len,
                    mode="decode",
                    notes="OOM",
                ))
            else:
                raise

    return results


def run_gqa_benchmarks(
    batch: int = 1,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    context_lengths: list[int] | None = None,
    warmup: int = WARMUP_ITERS,
    iterations: int = BENCH_ITERS,
) -> list[BenchmarkResult]:
    """Run GQA (Grouped Query Attention) benchmarks across context lengths.

    Args:
        batch: Batch size
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Head dimension
        context_lengths: List of context lengths to benchmark

    Returns:
        List of BenchmarkResult objects
    """
    if context_lengths is None:
        context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    results: list[BenchmarkResult] = []

    print("\n" + "=" * 80)
    print(f"GQA Decode Benchmark: FA2 vs FA3 (B={batch}, Q={num_q_heads}, KV={num_kv_heads}, D={head_dim})")
    print("=" * 80)
    print(f"{'Context':>10} {'FA2 (ms)':>12} {'FA3 (ms)':>12} {'Speedup':>10} {'FA2 TFLOPS':>12} {'FA3 TFLOPS':>12}")
    print("-" * 80)

    for ctx_len in context_lengths:
        try:
            # For GQA, K and V have fewer heads
            q = torch.randn(batch, num_q_heads, 1, head_dim, dtype=torch.float16, device="mps")
            k = torch.randn(batch, num_kv_heads, ctx_len, head_dim, dtype=torch.float16, device="mps")
            v = torch.randn(batch, num_kv_heads, ctx_len, head_dim, dtype=torch.float16, device="mps")
            mps_sync()

            scale = 1.0 / math.sqrt(head_dim)

            def fa2_fn() -> torch.Tensor:
                return flash_attention_v2_decode(q, k, v, scale=scale)

            fa2_mean, fa2_std = _time_kernel(fa2_fn, warmup=warmup, iterations=iterations)

            # FA3 placeholder
            fa3_mean, fa3_std, notes = 0.0, 0.0, "FA3 GQA Python interface not yet implemented"

            # Calculate metrics
            flops = calculate_attention_flops(batch, num_q_heads, 1, ctx_len, head_dim, causal=True)
            fa2_tflops = (flops / (fa2_mean / 1000.0)) / 1e12 if fa2_mean > 0 else 0.0
            fa3_tflops = 0.0
            speedup = fa2_mean / fa3_mean if fa3_mean > 0 else 0.0
            memory_mb = calculate_memory_usage(batch, num_q_heads, 1, ctx_len, head_dim)

            result = BenchmarkResult(
                batch=batch,
                num_heads=num_q_heads,
                head_dim=head_dim,
                seq_len=1,
                context_len=ctx_len,
                mode="decode_gqa",
                fa2_time_ms=fa2_mean,
                fa2_std_ms=fa2_std,
                fa3_time_ms=fa3_mean,
                fa3_std_ms=fa3_std,
                fa2_tflops=fa2_tflops,
                fa3_tflops=fa3_tflops,
                speedup=speedup,
                memory_mb=memory_mb,
                notes=notes,
                metadata={"num_kv_heads": num_kv_heads, "gqa_ratio": num_q_heads // num_kv_heads},
            )
            results.append(result)

            fa3_display = f"{fa3_mean:.3f}" if fa3_mean > 0 else "N/A"
            speedup_display = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            fa3_tflops_display = f"{fa3_tflops:.2f}" if fa3_tflops > 0 else "N/A"

            print(f"{ctx_len:>10} {fa2_mean:>12.3f} {fa3_display:>12} {speedup_display:>10} {fa2_tflops:>12.2f} {fa3_tflops_display:>12}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{ctx_len:>10} {'OOM':>12} {'OOM':>12} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
                results.append(BenchmarkResult(
                    batch=batch,
                    num_heads=num_q_heads,
                    head_dim=head_dim,
                    seq_len=1,
                    context_len=ctx_len,
                    mode="decode_gqa",
                    notes="OOM",
                ))
            else:
                raise

    return results


def export_results(results: list[BenchmarkResult], output_path: str | Path) -> None:
    """Export benchmark results to JSON.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark": "flash_attention_v3_vs_v2",
        "device": "mps" if HAS_MPS else "unknown",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults exported to: {output_path}")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print summary of benchmark results.

    Args:
        results: List of BenchmarkResult objects
    """
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)

    # Group by mode
    prefill_results = [r for r in results if r.mode == "prefill"]
    decode_results = [r for r in results if r.mode == "decode"]
    gqa_results = [r for r in results if r.mode == "decode_gqa"]

    if prefill_results:
        print(f"\nPrefill Results ({len(prefill_results)} configurations):")
        print(f"  Context lengths: {[r.context_len for r in prefill_results]}")
        avg_fa2 = statistics.mean([r.fa2_time_ms for r in prefill_results if r.fa2_time_ms > 0])
        print(f"  Average FA2 time: {avg_fa2:.3f} ms")

    if decode_results:
        print(f"\nDecode Results ({len(decode_results)} configurations):")
        print(f"  Context lengths: {[r.context_len for r in decode_results]}")
        avg_fa2 = statistics.mean([r.fa2_time_ms for r in decode_results if r.fa2_time_ms > 0])
        print(f"  Average FA2 time: {avg_fa2:.3f} ms")

    if gqa_results:
        print(f"\nGQA Decode Results ({len(gqa_results)} configurations):")
        print(f"  Context lengths: {[r.context_len for r in gqa_results]}")
        avg_fa2 = statistics.mean([r.fa2_time_ms for r in gqa_results if r.fa2_time_ms > 0])
        print(f"  Average FA2 time: {avg_fa2:.3f} ms")

    print("\nNote: FA3 results show as 'N/A' until the Python interface is implemented.")
    print("      The Metal kernel exists at: src/flash_attention_v3.metal")


def main() -> int:
    """Main entry point for the benchmark script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Benchmark Flash Attention v3 vs v2 across context lengths 1K to 32K"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=32,
        help="Number of attention heads (default: 32)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension (default: 128)",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768],
        help="Context lengths to benchmark (default: 1K 2K 4K 8K 16K 32K)",
    )
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help="Skip prefill benchmarks",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip decode benchmarks",
    )
    parser.add_argument(
        "--skip-gqa",
        action="store_true",
        help="Skip GQA benchmarks",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_ITERS,
        help=f"Number of warmup iterations (default: {WARMUP_ITERS})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=BENCH_ITERS,
        help=f"Number of benchmark iterations (default: {BENCH_ITERS})",
    )

    args = parser.parse_args()

    # Check MPS availability
    if not HAS_MPS:
        print("ERROR: MPS backend not available. This benchmark requires Apple Silicon.")
        return 1

    # Use local variables for warmup/iterations
    warmup_iters = args.warmup
    bench_iters = args.iterations

    print("Flash Attention V3 vs V2 Benchmark")
    print("=" * 80)
    print("Device: MPS (Apple Silicon)")
    print(f"PyTorch: {torch.__version__}")
    print(f"Configuration: batch={args.batch}, heads={args.heads}, head_dim={args.head_dim}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")

    all_results: list[BenchmarkResult] = []

    # Run benchmarks
    if not args.skip_prefill:
        prefill_results = run_prefill_benchmarks(
            batch=args.batch,
            num_heads=args.heads,
            head_dim=args.head_dim,
            context_lengths=args.context_lengths,
        )
        all_results.extend(prefill_results)

    if not args.skip_decode:
        decode_results = run_decode_benchmarks(
            batch=args.batch,
            num_heads=args.heads,
            head_dim=args.head_dim,
            context_lengths=args.context_lengths,
        )
        all_results.extend(decode_results)

    if not args.skip_gqa:
        gqa_results = run_gqa_benchmarks(
            batch=args.batch,
            num_q_heads=args.heads,
            num_kv_heads=args.heads // 4,  # 4:1 GQA ratio
            head_dim=args.head_dim,
            context_lengths=args.context_lengths,
        )
        all_results.extend(gqa_results)

    # Print summary
    print_summary(all_results)

    # Export results if requested
    if args.export:
        export_results(all_results, args.export)

    return 0


if __name__ == "__main__":
    sys.exit(main())
