#!/usr/bin/env python3
"""Benchmark expert routing overhead for MoE layers.

Measures the individual components of MoE routing:
1. Router forward pass (linear layer)
2. Top-k selection
3. Weight normalization
4. Expert ID tensor creation

Compares routing overhead vs. expert computation to ensure routing
is <5% of total MoE forward time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F


@dataclass
class TimingStats:
    """Statistics for a timed operation."""

    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    p50_us: float
    p95_us: float


def _sync_device(device: str) -> None:
    """Synchronize GPU device."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _compute_stats(times_s: list[float]) -> TimingStats:
    """Compute statistics from list of times in seconds."""
    times_us = [t * 1e6 for t in times_s]
    sorted_times = sorted(times_us)
    n = len(sorted_times)
    return TimingStats(
        mean_us=statistics.mean(times_us),
        std_us=statistics.stdev(times_us) if n > 1 else 0.0,
        min_us=sorted_times[0],
        max_us=sorted_times[-1],
        p50_us=sorted_times[n // 2],
        p95_us=sorted_times[int(n * 0.95)],
    )


def _time_operation(
    device: str,
    fn: Any,
    warmup: int,
    iterations: int,
) -> tuple[TimingStats, Any]:
    """Time an operation with warmup and return stats + last result."""
    result = None

    # Warmup
    for _ in range(warmup):
        result = fn()
        _sync_device(device)

    # Timed runs
    times: list[float] = []
    for _ in range(iterations):
        _sync_device(device)
        start = time.perf_counter()
        result = fn()
        _sync_device(device)
        times.append(time.perf_counter() - start)

    return _compute_stats(times), result


def benchmark_routing_components(
    batch_size: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark individual routing components.

    Args:
        batch_size: Number of tokens.
        hidden_dim: Hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        device: Device to run on.
        dtype: Data type for computations.
        warmup: Warmup iterations.
        iterations: Timed iterations.

    Returns:
        Dictionary with timing stats for each component.
    """
    torch.manual_seed(42)  # type: ignore[no-untyped-call]

    # Create inputs
    hidden_states = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    router_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=dtype)

    results: dict[str, Any] = {
        "config": {
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "device": device,
            "dtype": str(dtype),
        }
    }

    # 1. Router forward pass (linear layer)
    def router_forward() -> torch.Tensor:
        return F.linear(hidden_states, router_weight)

    router_stats, router_logits = _time_operation(device, router_forward, warmup, iterations)
    results["router_forward"] = router_stats

    # 2. Softmax
    def softmax_op() -> torch.Tensor:
        return F.softmax(router_logits, dim=-1, dtype=torch.float)

    softmax_stats, probs = _time_operation(device, softmax_op, warmup, iterations)
    results["softmax"] = softmax_stats

    # 3. Top-k selection
    def topk_op() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(probs, k=top_k, dim=-1)

    topk_stats, (top_probs, top_ids) = _time_operation(device, topk_op, warmup, iterations)
    results["topk"] = topk_stats

    # 4. Weight normalization
    def normalize_op() -> torch.Tensor:
        return top_probs / top_probs.sum(dim=-1, keepdim=True)

    normalize_stats, _ = _time_operation(device, normalize_op, warmup, iterations)
    results["normalize"] = normalize_stats

    # 5. Expert ID tensor creation (casting + contiguous)
    # Use a lambda that captures top_ids to measure the cast + contiguous operation
    top_ids_for_cast = top_ids  # Capture for closure

    def expert_id_creation() -> torch.Tensor:
        return top_ids_for_cast.int().contiguous()

    expert_id_stats, _ = _time_operation(device, expert_id_creation, warmup, iterations)
    results["expert_id_creation"] = expert_id_stats

    # Combined routing (all steps together)
    def full_routing() -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states, router_weight)
        probs_full = F.softmax(logits, dim=-1, dtype=torch.float)
        top_probs_full, top_ids_full = torch.topk(probs_full, k=top_k, dim=-1)
        normalized = top_probs_full / top_probs_full.sum(dim=-1, keepdim=True)
        expert_ids = top_ids_full.int().contiguous()
        return normalized, expert_ids

    full_routing_stats, _ = _time_operation(device, full_routing, warmup, iterations)
    results["full_routing"] = full_routing_stats

    # Calculate total from components
    component_total = (
        router_stats.mean_us
        + softmax_stats.mean_us
        + topk_stats.mean_us
        + normalize_stats.mean_us
        + expert_id_stats.mean_us
    )
    results["component_total_us"] = component_total

    return results


def benchmark_routing_vs_expert(
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, Any]:
    """Compare routing overhead vs expert computation.

    Args:
        batch_size: Number of tokens.
        hidden_dim: Hidden dimension.
        intermediate_dim: FFN intermediate dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        device: Device to run on.
        dtype: Data type for computations.
        warmup: Warmup iterations.
        iterations: Timed iterations.

    Returns:
        Dictionary with routing vs expert timing comparison.
    """
    torch.manual_seed(42)  # type: ignore[no-untyped-call]

    # Create inputs
    hidden_states = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    router_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=dtype)

    # Simulate expert weights (gate/up/down projections for one expert)
    gate_weight = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype)
    up_weight = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype)
    down_weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype)

    results: dict[str, Any] = {
        "config": {
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "device": device,
            "dtype": str(dtype),
        }
    }

    # Full routing
    def full_routing() -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states, router_weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
        normalized = top_probs / top_probs.sum(dim=-1, keepdim=True)
        expert_ids = top_ids.int().contiguous()
        return normalized, expert_ids

    routing_stats, _ = _time_operation(device, full_routing, warmup, iterations)
    results["routing"] = routing_stats

    # Single expert forward (SwiGLU)
    def expert_forward() -> torch.Tensor:
        gate = F.linear(hidden_states, gate_weight)
        up = F.linear(hidden_states, up_weight)
        activated = F.silu(gate) * up
        output = F.linear(activated, down_weight)
        return output

    expert_stats, _ = _time_operation(device, expert_forward, warmup, iterations)
    results["single_expert"] = expert_stats

    # Simulated MoE forward (routing + top_k expert calls)
    # In practice, experts are fused/batched, but this shows worst case
    def moe_forward_sequential() -> torch.Tensor:
        logits = F.linear(hidden_states, router_weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        top_probs, _ = torch.topk(probs, k=top_k, dim=-1)
        normalized = top_probs / top_probs.sum(dim=-1, keepdim=True)

        # Simulate top_k expert calls (sequential worst case)
        output = torch.zeros_like(hidden_states)
        for k in range(top_k):
            gate = F.linear(hidden_states, gate_weight)
            up = F.linear(hidden_states, up_weight)
            activated = F.silu(gate) * up
            expert_out = F.linear(activated, down_weight)
            output = output + expert_out * normalized[:, k:k+1]

        return output

    moe_stats, _ = _time_operation(device, moe_forward_sequential, warmup, iterations)
    results["moe_sequential"] = moe_stats

    # Calculate percentages
    routing_us = routing_stats.mean_us
    moe_us = moe_stats.mean_us
    expert_total_us = expert_stats.mean_us * top_k

    results["routing_pct_of_moe"] = (routing_us / moe_us * 100) if moe_us > 0 else 0.0
    results["routing_pct_of_expert_only"] = (
        (routing_us / expert_total_us * 100) if expert_total_us > 0 else 0.0
    )
    results["expert_total_us"] = expert_total_us

    return results


def run_top_k_sweep(
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k_values: list[int],
    device: str,
    dtype: torch.dtype,
    warmup: int = 10,
    iterations: int = 100,
) -> list[dict[str, Any]]:
    """Run benchmarks across different top_k values.

    Args:
        batch_size: Number of tokens.
        hidden_dim: Hidden dimension.
        intermediate_dim: FFN intermediate dimension.
        num_experts: Total number of experts.
        top_k_values: List of top_k values to test.
        device: Device to run on.
        dtype: Data type for computations.
        warmup: Warmup iterations.
        iterations: Timed iterations.

    Returns:
        List of benchmark results for each top_k value.
    """
    results: list[dict[str, Any]] = []
    for top_k in top_k_values:
        result = benchmark_routing_vs_expert(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            dtype=dtype,
            warmup=warmup,
            iterations=iterations,
        )
        results.append(result)
    return results


def print_component_breakdown(results: dict[str, Any]) -> None:
    """Print detailed component timing breakdown."""
    cfg = results["config"]
    print("\nROUTING COMPONENT BREAKDOWN")
    print("=" * 60)
    print(f"Config: batch={cfg['batch_size']}, hidden={cfg['hidden_dim']}, "
          f"experts={cfg['num_experts']}, top_k={cfg['top_k']}")
    print()

    components = [
        ("Router linear", "router_forward"),
        ("Softmax", "softmax"),
        ("Top-k selection", "topk"),
        ("Weight normalize", "normalize"),
        ("Expert ID creation", "expert_id_creation"),
    ]

    total = results["component_total_us"]
    print(f"{'Component':<25} {'Mean (µs)':>12} {'Std (µs)':>12} {'% of Total':>12}")
    print("-" * 60)

    for name, key in components:
        stats = results[key]
        pct = (stats.mean_us / total * 100) if total > 0 else 0.0
        print(f"{name:<25} {stats.mean_us:>12.2f} {stats.std_us:>12.2f} {pct:>11.1f}%")

    print("-" * 60)
    full_stats = results["full_routing"]
    print(f"{'Full routing (fused)':<25} {full_stats.mean_us:>12.2f} {full_stats.std_us:>12.2f}")
    print(f"{'Component sum':<25} {total:>12.2f}")


def print_routing_vs_expert(results: dict[str, Any]) -> None:
    """Print routing vs expert computation comparison."""
    cfg = results["config"]
    print(f"\nROUTING vs EXPERT COMPUTATION (top_k={cfg['top_k']})")
    print("=" * 60)

    routing = results["routing"]
    expert = results["single_expert"]
    moe = results["moe_sequential"]

    print(f"{'Operation':<30} {'Mean (µs)':>12} {'Std (µs)':>12}")
    print("-" * 60)
    print(f"{'Routing (total)':<30} {routing.mean_us:>12.2f} {routing.std_us:>12.2f}")
    print(f"{'Single expert forward':<30} {expert.mean_us:>12.2f} {expert.std_us:>12.2f}")
    print(f"{'Expert × top_k':<30} {results['expert_total_us']:>12.2f}")
    print(f"{'Full MoE (sequential)':<30} {moe.mean_us:>12.2f} {moe.std_us:>12.2f}")
    print("-" * 60)
    print(f"Routing as % of MoE total:    {results['routing_pct_of_moe']:.2f}%")
    print(f"Routing as % of expert-only:  {results['routing_pct_of_expert_only']:.2f}%")

    if results["routing_pct_of_moe"] < 5.0:
        print("\n✓ PASS: Routing overhead < 5% of MoE forward time")
    else:
        print("\n✗ FAIL: Routing overhead >= 5% of MoE forward time")


def print_top_k_sweep_summary(results: list[dict[str, Any]]) -> None:
    """Print summary table for top_k sweep."""
    print("\nTOP-K SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'top_k':>8} {'Routing (µs)':>15} {'Expert (µs)':>15} "
          f"{'MoE (µs)':>15} {'Routing %':>12}")
    print("-" * 80)

    for r in results:
        cfg = r["config"]
        print(f"{cfg['top_k']:>8} {r['routing'].mean_us:>15.2f} "
              f"{r['expert_total_us']:>15.2f} {r['moe_sequential'].mean_us:>15.2f} "
              f"{r['routing_pct_of_moe']:>11.2f}%")

    print("-" * 80)

    all_pass = all(r["routing_pct_of_moe"] < 5.0 for r in results)
    if all_pass:
        print("\n✓ ALL PASS: Routing overhead < 5% for all top_k values")
    else:
        failed = [r["config"]["top_k"] for r in results if r["routing_pct_of_moe"] >= 5.0]
        print(f"\n✗ FAIL: Routing overhead >= 5% for top_k values: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MoE routing overhead",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Number of tokens")
    parser.add_argument("--hidden-dim", type=int, default=3584, help="Hidden dimension")
    parser.add_argument("--intermediate-dim", type=int, default=18944, help="FFN intermediate dim")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of experts")
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 4, 8],
                        help="Top-k values to test")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Data type")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    parser.add_argument("--component-breakdown", action="store_true",
                        help="Show detailed component breakdown")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Parse dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("MoE Routing Overhead Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Intermediate dim: {args.intermediate_dim}")
    print(f"Num experts: {args.num_experts}")
    print(f"Top-k values: {args.top_k}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")

    # Component breakdown for first top_k value
    if args.component_breakdown:
        component_results = benchmark_routing_components(
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_experts=args.num_experts,
            top_k=args.top_k[0],
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print_component_breakdown(component_results)

    # Run top_k sweep
    sweep_results = run_top_k_sweep(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_experts=args.num_experts,
        top_k_values=args.top_k,
        device=device,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    # Print individual results
    for r in sweep_results:
        print_routing_vs_expert(r)

    # Print summary
    print_top_k_sweep_summary(sweep_results)

    # Export JSON if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert TimingStats to dicts
        def stats_to_dict(stats: TimingStats) -> dict[str, float]:
            return {
                "mean_us": stats.mean_us,
                "std_us": stats.std_us,
                "min_us": stats.min_us,
                "max_us": stats.max_us,
                "p50_us": stats.p50_us,
                "p95_us": stats.p95_us,
            }

        export_data: list[dict[str, Any]] = []
        for r in sweep_results:
            entry = {
                "config": r["config"],
                "routing": stats_to_dict(r["routing"]),
                "single_expert": stats_to_dict(r["single_expert"]),
                "moe_sequential": stats_to_dict(r["moe_sequential"]),
                "routing_pct_of_moe": r["routing_pct_of_moe"],
                "routing_pct_of_expert_only": r["routing_pct_of_expert_only"],
                "expert_total_us": r["expert_total_us"],
            }
            export_data.append(entry)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
