#!/usr/bin/env python3
"""
Visualize attention benchmark results.

Usage:
    uv run python benchmarks/visualize_attention_results.py benchmarks/results/attention_benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

def load_results(filepath: Path) -> list[dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 80)
    print("ATTENTION BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by configuration
    configs = {}
    for r in results:
        key = (r["batch_size"], r["seq_len"], r["num_heads"], r["head_dim"])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    # Print for each configuration
    for (batch, seq, heads, dim), impls in sorted(configs.items()):
        print(f"\n📊 Configuration: Batch={batch}, Seq={seq}, Heads={heads}, Dim={dim}")
        print("-" * 80)

        # Sort by mean time
        impls_sorted = sorted(impls, key=lambda x: x["mean_time_ms"])
        fastest = impls_sorted[0]

        for impl in impls_sorted:
            speedup = fastest["mean_time_ms"] / impl["mean_time_ms"]
            efficiency = "🥇" if impl == fastest else f"{speedup:.2f}x"

            print(
                f"  {impl['implementation']:<30} "
                f"{impl['mean_time_ms']:>8.3f} ms ± {impl['std_time_ms']:<6.3f}  "
                f"({impl['throughput_tokens_per_sec']/1000:>6.1f}k tok/s)  "
                f"{efficiency}"
            )


def print_scaling_analysis(results: list[dict[str, Any]]) -> None:
    """Analyze how implementations scale with sequence length."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS (Sequence Length)")
    print("=" * 80)

    # Group by implementation and batch size
    impls = {}
    for r in results:
        key = (r["implementation"], r["batch_size"])
        if key not in impls:
            impls[key] = []
        impls[key].append(r)

    # Print scaling table
    for (impl_name, batch), data in sorted(impls.items()):
        print(f"\n{impl_name} (Batch={batch}):")
        data_sorted = sorted(data, key=lambda x: x["seq_len"])

        print(f"  {'Seq Len':<12} {'Time (ms)':<12} {'Throughput':<15} {'TFLOPS':<10}")
        print(f"  {'-'*50}")

        for d in data_sorted:
            print(
                f"  {d['seq_len']:<12} {d['mean_time_ms']:<12.3f} "
                f"{d['throughput_tokens_per_sec']/1000:<15.1f}k {d['tflops']:<10.1f}"
            )


def print_memory_analysis(results: list[dict[str, Any]]) -> None:
    """Analyze memory usage patterns."""
    print("\n" + "=" * 80)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 80)

    # Group by configuration
    configs = {}
    for r in results:
        key = (r["batch_size"], r["seq_len"], r["num_heads"], r["head_dim"])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    print(f"\n  {'Config':<25} {'Implementation':<30} {'Memory (MB)':<12}")
    print(f"  {'-'*70}")

    for (batch, seq, heads, dim), impls in sorted(configs.items()):
        config_str = f"B={batch}, N={seq}, H={heads}"
        for impl in impls:
            print(f"  {config_str:<25} {impl['implementation']:<30} {impl['memory_mb']:<12.1f}")
            config_str = ""  # Only show config once


def print_accuracy_analysis(results: list[dict[str, Any]]) -> None:
    """Print accuracy comparison (max error vs reference)."""
    print("\n" + "=" * 80)
    print("ACCURACY ANALYSIS")
    print("=" * 80)

    # Check if we have accuracy data
    has_accuracy = any(r.get("max_error", 0) > 0 for r in results)

    if not has_accuracy:
        print("\n  ⚠️  All implementations show 0.0 max error (computed in same precision)")
        print("  For accurate comparison, reference should be computed in float64")
    else:
        for r in results:
            status = "✅" if r["max_error"] < 0.01 else "⚠️"
            print(f"  {r['implementation']:<30} {status} Max Error: {r['max_error']:.8f}")


def print_theoretical_analysis(results: list[dict[str, Any]]) -> None:
    """Print theoretical analysis of attention mechanisms."""
    print("\n" + "=" * 80)
    print("THEORETICAL COMPLEXITY COMPARISON")
    print("=" * 80)

    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ IMPLEMENTATION        │ TIME      │ MEMORY     │ BEST FOR              │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Standard Attention    │ O(N²·D)   │ O(N²)      │ Baseline, debugging   │
  │ Flash Attention       │ O(N²·D)   │ O(N)       │ Long sequences        │
  │ Fused QKV             │ O(N²·D)   │ O(N²)      │ Reduced kernel launches│
  │ GQA                   │ O(N²·D·g) │ O(N·g)     │ Memory constrained    │
  └─────────────────────────────────────────────────────────────────────────┘

  Where:
    N = sequence length
    D = head dimension
    g = ratio of KV heads to Q heads (for GQA)

  Key Insights:
  • Flash Attention avoids materializing the N×N attention matrix
  • GQA reduces KV cache memory by factor of g
  • Fused QKV reduces GPU kernel launch overhead
    """)


def print_recommendations(results: list[dict[str, Any]]) -> None:
    """Print recommendations based on benchmark results."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Group by config
    configs = {}
    for r in results:
        key = (r["batch_size"], r["seq_len"])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    for (batch, seq), impls in sorted(configs.items()):
        fastest = min(impls, key=lambda x: x["mean_time_ms"])
        highest_throughput = max(impls, key=lambda x: x["throughput_tokens_per_sec"])

        print(f"\n  Config: Batch={batch}, Seq={seq}")
        print(f"    🏎️  Fastest: {fastest['implementation']} ({fastest['mean_time_ms']:.3f} ms)")
        print(f"    ⚡ Highest Throughput: {highest_throughput['implementation']}")

    print("""
  General Guidelines:
  • Use Fused QKV for best overall performance (fewer kernel launches)
  • Use Flash Attention for very long sequences (>4k tokens)
  • Use GQA when KV cache memory is a bottleneck
  • Use Standard Attention only for debugging or when others unavailable
    """)


def main():
    parser = argparse.ArgumentParser(description="Visualize attention benchmark results")
    parser.add_argument("results_file", type=Path, help="Path to JSON results file")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["summary", "scaling", "memory", "accuracy", "theoretical", "recommendations", "all"],
        default=["all"],
        help="Which sections to display",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Error: File not found: {args.results_file}")
        return 1

    results = load_results(args.results_file)
    sections = args.sections
    if "all" in sections:
        sections = ["summary", "scaling", "memory", "accuracy", "theoretical", "recommendations"]

    if "summary" in sections:
        print_summary(results)
    if "scaling" in sections:
        print_scaling_analysis(results)
    if "memory" in sections:
        print_memory_analysis(results)
    if "accuracy" in sections:
        print_accuracy_analysis(results)
    if "theoretical" in sections:
        print_theoretical_analysis(results)
    if "recommendations" in sections:
        print_recommendations(results)

    return 0


if __name__ == "__main__":
    exit(main())
