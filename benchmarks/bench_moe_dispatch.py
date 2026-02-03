#!/usr/bin/env python3
"""Benchmark MoE dispatch implementations.

Tests sequential, batched, sorted, and parallel dispatch strategies
with varying batch sizes and top-k values to measure performance.
"""

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Add parent directory to path for imports
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import after path setup
from metal_marlin._compat import HAS_TORCH, torch  # noqa: E402

if not HAS_TORCH or torch is None:
    print("Error: PyTorch is required for MoE dispatch benchmarks")
    sys.exit(1)

# Import dispatch implementations
try:
    from metal_marlin.moe.token_dispatcher import group_tokens_by_expert  # noqa: E402
    HAS_DISPATCHER = True
except ImportError:
    HAS_DISPATCHER = False

# Try importing Metal implementation
try:
    from metal_marlin.moe_dispatch_metal import group_tokens_by_expert_full_metal  # noqa: E402
    HAS_METAL_IMPL = True
except (ImportError, ModuleNotFoundError):
    HAS_METAL_IMPL = False


def _sync_device(device: str) -> None:
    """Synchronize device operations."""
    assert torch is not None
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_function(
    name: str,
    func: Callable,
    args: tuple,
    device: str = "cpu",
    num_iters: int = 100,
    warmup: int = 10,
) -> tuple[float, list[float]]:
    """Benchmark a function and return (avg_time_ms, per_iteration_times_ms)."""
    times = []

    # Warmup
    for _ in range(warmup):
        func(*args)

    _sync_device(device)

    # Collect individual iteration times
    for _ in range(num_iters):
        start = time.perf_counter()
        func(*args)
        _sync_device(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = statistics.mean(times)
    return avg_time, times


def sequential_dispatch(expert_ids: torch.Tensor, num_experts: int) -> list[torch.Tensor]:
    """Simulate sequential dispatch (per-expert masking).
    
    This represents a naive approach where we iterate over experts
    and mask the batch for each expert.
    """
    batch_size, top_k = expert_ids.shape
    expert_masks = []

    for i in range(num_experts):
        # Create mask for this expert
        mask = (expert_ids == i)
        expert_masks.append(mask)

    return expert_masks


def batched_sorted_dispatch(expert_ids: torch.Tensor, num_experts: int):
    """Benchmark the PyTorch implementation with sorting."""
    if HAS_DISPATCHER:
        return group_tokens_by_expert(expert_ids, num_experts)
    # Fallback to simple sorting approach
    batch_size, top_k = expert_ids.shape
    flat_ids = expert_ids.reshape(-1)
    sorted_indices = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[sorted_indices]

    # Count tokens per expert
    expert_counts = torch.bincount(sorted_ids, minlength=num_experts)
    offsets = torch.cat([torch.tensor([0], device=expert_ids.device),
                         expert_counts.cumsum(0)])

    token_indices = sorted_indices // top_k
    expert_slot_indices = sorted_indices % top_k

    return token_indices, offsets, None


def parallel_dispatch_metal(expert_ids: torch.Tensor, num_experts: int):
    """Benchmark the Metal implementation."""
    if HAS_METAL_IMPL:
        return group_tokens_by_expert_full_metal(expert_ids, num_experts)
    return None


def format_stats(times: list[float]) -> str:
    """Format timing statistics."""
    if not times:
        return "N/A"
    return (f"mean={statistics.mean(times):.4f}ms "
            f"p50={statistics.median(times):.4f}ms "
            f"p95={sorted(times)[int(len(times) * 0.95)]:.4f}ms")


def run_benchmark():
    assert torch is not None
    parser = argparse.ArgumentParser(description="Benchmark MoE Dispatch Implementations")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16,64,256",
                       help="Comma-separated batch sizes")
    parser.add_argument("--top-k-values", type=str, default="2,4,8",
                       help="Comma-separated top-k values")
    parser.add_argument("--num-experts", type=int, default=8,
                       help="Number of experts")
    parser.add_argument("--device", type=str,
                       default="mps" if torch.backends.mps.is_available() else "cpu",
                       help="Device to run on (cpu/mps/cuda)")
    parser.add_argument("--num-iters", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--show-stats", action="store_true",
                       help="Show detailed timing statistics (mean/p50/p95)")

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    top_k_vals = [int(x) for x in args.top_k_values.split(",")]
    num_experts = args.num_experts
    device = args.device

    print(f"\n{'='*80}")
    print("MoE Dispatch Benchmark")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Experts: {num_experts}")
    print(f"Iterations: {args.num_iters} (warmup: {args.warmup})")
    print(f"{'='*80}\n")

    if args.show_stats:
        header = f"{'Batch':<8} {'TopK':<6} {'Method':<20} {'Stats':<50}"
    else:
        header = f"{'Batch':<8} {'TopK':<6} {'Method':<20} {'Time (ms)':<12} {'Speedup':<10} {'Tokens/sec':<12}"

    print(header)
    print("-" * len(header))

    for b in batch_sizes:
        for k in top_k_vals:
            # Generate random expert assignments
            expert_ids = torch.randint(0, num_experts, (b, k), device=device)
            total_assignments = b * k

            # 1. Sequential (Naive baseline)
            seq_time, seq_times = benchmark_function(
                "Sequential",
                sequential_dispatch,
                (expert_ids, num_experts),
                device=device,
                num_iters=args.num_iters,
                warmup=args.warmup,
            )

            if args.show_stats:
                print(f"{b:<8} {k:<6} {'Sequential':<20} {format_stats(seq_times):<50}")
            else:
                tokens_per_sec = (total_assignments / seq_time * 1000) if seq_time > 0 else 0
                print(f"{b:<8} {k:<6} {'Sequential':<20} {seq_time:<12.4f} {'1.0x':<10} {tokens_per_sec:<12.0f}")

            # 2. Batched/Sorted (PyTorch)
            sorted_time, sorted_times = benchmark_function(
                "Batched (PyTorch)",
                batched_sorted_dispatch,
                (expert_ids, num_experts),
                device=device,
                num_iters=args.num_iters,
                warmup=args.warmup,
            )
            speedup_sorted = seq_time / sorted_time if sorted_time > 0 else 0

            if args.show_stats:
                print(f"{b:<8} {k:<6} {'Batched (Sorted)':<20} {format_stats(sorted_times):<50}")
            else:
                tokens_per_sec = (total_assignments / sorted_time * 1000) if sorted_time > 0 else 0
                print(f"{b:<8} {k:<6} {'Batched (Sorted)':<20} {sorted_time:<12.4f} "
                      f"{f'{speedup_sorted:.2f}x':<10} {tokens_per_sec:<12.0f}")

            # 3. Parallel (Metal) - only on MPS
            if HAS_METAL_IMPL and device == "mps":
                try:
                    metal_time, metal_times = benchmark_function(
                        "Parallel (Metal)",
                        parallel_dispatch_metal,
                        (expert_ids, num_experts),
                        device=device,
                        num_iters=args.num_iters,
                        warmup=args.warmup,
                    )
                    speedup_metal = seq_time / metal_time if metal_time > 0 else 0

                    if args.show_stats:
                        print(f"{b:<8} {k:<6} {'Parallel (Metal)':<20} {format_stats(metal_times):<50}")
                    else:
                        tokens_per_sec = (total_assignments / metal_time * 1000) if metal_time > 0 else 0
                        print(f"{b:<8} {k:<6} {'Parallel (Metal)':<20} {metal_time:<12.4f} "
                              f"{f'{speedup_metal:.2f}x':<10} {tokens_per_sec:<12.0f}")
                except Exception as e:
                    print(f"{b:<8} {k:<6} {'Parallel (Metal)':<20} {'ERROR':<12} {'-':<10} (error: {str(e)[:30]})")

            print("-" * len(header))

        # Add blank line between batch size groups
        if b != batch_sizes[-1]:
            print()

    print(f"\n{'='*80}")
    print("Benchmark complete")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_benchmark()
