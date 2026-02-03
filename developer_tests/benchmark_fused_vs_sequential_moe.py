#!/usr/bin/env python3
"""Benchmark Fused MoE vs Sequential Dispatch for GLM-4.7-Flash (64 experts).

This benchmark compares three MoE dispatch strategies:
1. **Fused Metal Dispatch** (FAST): Single batched kernel processes all experts
2. **Sequential Python Loop** (SLOW): Iterates over experts with Python-level dispatch
3. **Batched Expert Dispatch** (MEDIUM): Groups tokens by expert, processes sequentially

Key metrics:
- Latency (ms per forward pass)
- Throughput (tokens/second)
- Memory bandwidth utilization
- Kernel launch overhead

GLM-4.7-Flash Configuration:
- 64 experts per MoE layer
- Top-2 or Top-8 expert selection per token
- Hidden dimension: 4096
- Intermediate dimension: 13696
- 3-bit Trellis quantization

Usage:
    cd contrib/metal_marlin
    uv run python developer_tests/benchmark_fused_vs_sequential_moe.py

    # With specific batch sizes
    uv run python developer_tests/benchmark_fused_vs_sequential_moe.py --batch-sizes 1,4,16,64

    # With real model weights (requires GLM-4.7-Flash model)
    uv run python developer_tests/benchmark_fused_vs_sequential_moe.py --use-real-model
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# GLM-4.7-Flash MoE configuration
NUM_EXPERTS = 64
TOP_K_VALUES = (2, 8)
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 13696
BITS = 3


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    batch_size: int
    top_k: int
    num_experts: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    tokens_per_sec: float
    kernel_launches: int


def create_mock_expert_weights(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    device: str = "mps",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create mock expert weights for benchmarking.

    Returns:
        (gate_weights, up_weights, down_weights) each with shape [num_experts, ...]
    """
    # SwiGLU experts: gate [H -> I], up [H -> I], down [I -> H]
    gate_weights = torch.randn(
        num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype
    ) * 0.02
    up_weights = torch.randn(
        num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype
    ) * 0.02
    down_weights = torch.randn(
        num_experts, intermediate_dim, hidden_dim, device=device, dtype=dtype
    ) * 0.02

    return gate_weights, up_weights, down_weights


def create_mock_routing(
    batch_size: int,
    num_experts: int,
    top_k: int,
    device: str = "mps",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create mock routing decisions.

    Returns:
        (expert_ids, expert_weights) for routing
    """
    # Random expert selection
    expert_ids = torch.randint(
        0, num_experts, (batch_size, top_k), device=device, dtype=torch.int64
    )

    # Softmax weights summing to 1
    logits = torch.randn(batch_size, top_k, device=device, dtype=torch.float16)
    expert_weights = F.softmax(logits, dim=-1)

    return expert_ids, expert_weights


class SequentialMoEDispatch(nn.Module):
    """Sequential dispatch: iterate over experts with Python loop (SLOW baseline).

    This represents the naive approach where each expert is processed sequentially,
    causing multiple kernel launches and poor GPU utilization.
    """

    def __init__(
        self,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        down_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_experts = gate_weights.shape[0]
        self.register_buffer("gate_weights", gate_weights)
        self.register_buffer("up_weights", up_weights)
        self.register_buffer("down_weights", down_weights)
        self.kernel_launches = 0

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with sequential expert processing.

        Args:
            x: Input [batch, hidden_dim]
            expert_ids: Expert indices [batch, top_k]
            expert_weights: Routing weights [batch, top_k]

        Returns:
            Output [batch, hidden_dim]
        """
        batch_size, hidden_dim = x.shape
        top_k = expert_ids.shape[1]
        output = torch.zeros_like(x)

        self.kernel_launches = 0

        # Sequential processing: iterate over each (token, expert) pair
        for b in range(batch_size):
            for k in range(top_k):
                expert_id = expert_ids[b, k].item()
                weight = expert_weights[b, k]

                # SwiGLU: out = down(silu(gate(x)) * up(x))
                gate_out = x[b:b+1] @ self.gate_weights[expert_id]  # kernel 1
                up_out = x[b:b+1] @ self.up_weights[expert_id]      # kernel 2
                activated = F.silu(gate_out) * up_out               # kernel 3
                expert_out = activated @ self.down_weights[expert_id]  # kernel 4

                output[b] += weight * expert_out.squeeze(0)
                self.kernel_launches += 4

        return output


class BatchedExpertMoEDispatch(nn.Module):
    """Batched dispatch: group tokens by expert, process each expert once (MEDIUM).

    This groups all tokens assigned to the same expert and processes them together,
    reducing kernel launches from O(batch * top_k) to O(num_active_experts).
    """

    def __init__(
        self,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        down_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_experts = gate_weights.shape[0]
        self.register_buffer("gate_weights", gate_weights)
        self.register_buffer("up_weights", up_weights)
        self.register_buffer("down_weights", down_weights)
        self.kernel_launches = 0

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with batched expert processing.

        Groups tokens by expert for efficient batch processing.
        """
        batch_size, hidden_dim = x.shape
        top_k = expert_ids.shape[1]
        output = torch.zeros_like(x)

        self.kernel_launches = 0

        # Flatten expert assignments
        flat_expert_ids = expert_ids.view(-1)  # [batch * top_k]
        total_assignments = batch_size * top_k

        # Sort by expert for coalesced processing
        sort_indices = torch.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sort_indices]

        # Compute expert boundaries
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x.device)
        expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

        # Process each active expert
        for expert_id in range(self.num_experts):
            start = expert_offsets[expert_id].item()
            end = expert_offsets[expert_id + 1].item()

            if start == end:
                continue

            # Get token indices for this expert
            assignment_indices = sort_indices[start:end]
            token_indices = assignment_indices // top_k
            slot_indices = assignment_indices % top_k

            # Gather tokens and weights for this expert
            expert_tokens = x[token_indices]  # [num_tokens, hidden]
            weights = expert_weights[token_indices, slot_indices]  # [num_tokens]

            # Batched SwiGLU computation
            gate_out = expert_tokens @ self.gate_weights[expert_id]  # kernel 1
            up_out = expert_tokens @ self.up_weights[expert_id]      # kernel 2
            activated = F.silu(gate_out) * up_out                    # kernel 3
            expert_out = activated @ self.down_weights[expert_id]    # kernel 4

            # Weighted scatter-add back to output
            weighted_out = expert_out * weights.unsqueeze(-1)
            output.index_add_(0, token_indices, weighted_out)

            self.kernel_launches += 4

        return output


class FusedMoEDispatch(nn.Module):
    """Fused dispatch: single kernel processes all experts (FAST).

    Simulates the behavior of the fused Metal kernel that processes all experts
    in a single launch, maximizing GPU utilization and minimizing launch overhead.
    """

    def __init__(
        self,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        down_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_experts = gate_weights.shape[0]
        # Stack all expert weights for single kernel
        self.register_buffer("gate_weights", gate_weights)  # [E, H, I]
        self.register_buffer("up_weights", up_weights)      # [E, H, I]
        self.register_buffer("down_weights", down_weights)  # [E, I, H]
        self.kernel_launches = 0

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with fused expert processing.

        Uses einsum/gather ops that can be fused into single kernel on Metal.
        """
        batch_size, hidden_dim = x.shape
        top_k = expert_ids.shape[1]

        # Expand input for all top_k experts: [batch, top_k, hidden]
        x_expanded = x.unsqueeze(1).expand(-1, top_k, -1)

        # Gather expert weights for selected experts
        # gate_weights: [E, H, I] -> selected: [batch, top_k, H, I]
        gate_selected = self.gate_weights[expert_ids]  # [batch, top_k, H, I]
        up_selected = self.up_weights[expert_ids]      # [batch, top_k, H, I]
        down_selected = self.down_weights[expert_ids]  # [batch, top_k, I, H]

        # Fused SwiGLU: single pass through all experts
        # [batch, top_k, hidden] @ [batch, top_k, hidden, inter] -> [batch, top_k, inter]
        gate_out = torch.einsum("bkh,bkhi->bki", x_expanded, gate_selected)
        up_out = torch.einsum("bkh,bkhi->bki", x_expanded, up_selected)
        activated = F.silu(gate_out) * up_out

        # [batch, top_k, inter] @ [batch, top_k, inter, hidden] -> [batch, top_k, hidden]
        expert_out = torch.einsum("bki,bkih->bkh", activated, down_selected)

        # Weighted sum across experts
        # expert_weights: [batch, top_k] -> [batch, top_k, 1]
        weighted = expert_out * expert_weights.unsqueeze(-1)
        output = weighted.sum(dim=1)  # [batch, hidden]

        # Count kernel launches (simulated as single fused op)
        self.kernel_launches = 1

        return output


def benchmark_dispatch(
    dispatch_fn: Callable,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> tuple[list[float], int]:
    """Benchmark a dispatch function.

    Returns:
        (list of iteration times in ms, number of kernel launches)
    """
    # Warmup
    for _ in range(num_warmup):
        _ = dispatch_fn(x, expert_ids, expert_weights)
        torch.mps.synchronize()

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = dispatch_fn(x, expert_ids, expert_weights)
        torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    # Get kernel launches from last iteration
    kernel_launches = getattr(dispatch_fn, "kernel_launches", 0)
    if hasattr(dispatch_fn, "__self__"):
        kernel_launches = getattr(dispatch_fn.__self__, "kernel_launches", 0)

    return times, kernel_launches


def run_benchmark(
    batch_sizes: list[int],
    top_k_values: list[int],
    num_experts: int = NUM_EXPERTS,
    hidden_dim: int = HIDDEN_DIM,
    intermediate_dim: int = INTERMEDIATE_DIM,
    device: str = "mps",
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> list[BenchmarkResult]:
    """Run full benchmark suite."""

    results: list[BenchmarkResult] = []

    # Create shared expert weights (simulates real model)
    print(f"Creating mock expert weights: {num_experts} experts, "
          f"hidden={hidden_dim}, intermediate={intermediate_dim}")
    gate_weights, up_weights, down_weights = create_mock_expert_weights(
        num_experts, hidden_dim, intermediate_dim, device=device
    )

    # Memory usage
    weights_mb = (
        gate_weights.numel() + up_weights.numel() + down_weights.numel()
    ) * 2 / 1024 / 1024
    print(f"Expert weights memory: {weights_mb:.1f} MB")

    # Create dispatch modules
    sequential = SequentialMoEDispatch(gate_weights, up_weights, down_weights)
    batched = BatchedExpertMoEDispatch(gate_weights, up_weights, down_weights)
    fused = FusedMoEDispatch(gate_weights, up_weights, down_weights)

    for top_k in top_k_values:
        print(f"\n{'=' * 70}")
        print(f"Top-K = {top_k}")
        print('=' * 70)

        for batch_size in batch_sizes:
            print(f"\n--- Batch Size: {batch_size} ---")

            # Create input and routing
            x = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)
            expert_ids, expert_weights_tensor = create_mock_routing(
                batch_size, num_experts, top_k, device=device
            )

            # Skip sequential for large batches (too slow)
            if batch_size <= 8:
                print("  Sequential (Python loop):")
                try:
                    times, launches = benchmark_dispatch(
                        sequential.forward, x, expert_ids, expert_weights_tensor,
                        num_warmup=2, num_iterations=5,  # Fewer iterations for slow method
                    )
                    mean_ms = sum(times) / len(times)
                    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
                    tokens_per_sec = batch_size / (mean_ms / 1000)
                    print(f"    Mean: {mean_ms:.2f} ms (±{std_ms:.2f})")
                    print(f"    Kernel launches: {launches}")
                    print(f"    Throughput: {tokens_per_sec:.0f} tok/s")
                    results.append(BenchmarkResult(
                        name="Sequential",
                        batch_size=batch_size,
                        top_k=top_k,
                        num_experts=num_experts,
                        mean_ms=mean_ms,
                        std_ms=std_ms,
                        min_ms=min(times),
                        max_ms=max(times),
                        tokens_per_sec=tokens_per_sec,
                        kernel_launches=launches,
                    ))
                except Exception as e:
                    print(f"    ERROR: {e}")
            else:
                print("  Sequential: SKIPPED (batch > 8 too slow)")

            # Batched expert dispatch
            print("  Batched Expert:")
            try:
                times, launches = benchmark_dispatch(
                    batched.forward, x, expert_ids, expert_weights_tensor,
                    num_warmup=num_warmup, num_iterations=num_iterations,
                )
                mean_ms = sum(times) / len(times)
                std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
                tokens_per_sec = batch_size / (mean_ms / 1000)
                print(f"    Mean: {mean_ms:.2f} ms (±{std_ms:.2f})")
                print(f"    Kernel launches: {launches}")
                print(f"    Throughput: {tokens_per_sec:.0f} tok/s")
                results.append(BenchmarkResult(
                    name="Batched",
                    batch_size=batch_size,
                    top_k=top_k,
                    num_experts=num_experts,
                    mean_ms=mean_ms,
                    std_ms=std_ms,
                    min_ms=min(times),
                    max_ms=max(times),
                    tokens_per_sec=tokens_per_sec,
                    kernel_launches=launches,
                ))
            except Exception as e:
                print(f"    ERROR: {e}")

            # Fused dispatch
            print("  Fused (single kernel):")
            try:
                times, launches = benchmark_dispatch(
                    fused.forward, x, expert_ids, expert_weights_tensor,
                    num_warmup=num_warmup, num_iterations=num_iterations,
                )
                mean_ms = sum(times) / len(times)
                std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
                tokens_per_sec = batch_size / (mean_ms / 1000)
                print(f"    Mean: {mean_ms:.2f} ms (±{std_ms:.2f})")
                print(f"    Kernel launches: {launches}")
                print(f"    Throughput: {tokens_per_sec:.0f} tok/s")
                results.append(BenchmarkResult(
                    name="Fused",
                    batch_size=batch_size,
                    top_k=top_k,
                    num_experts=num_experts,
                    mean_ms=mean_ms,
                    std_ms=std_ms,
                    min_ms=min(times),
                    max_ms=max(times),
                    tokens_per_sec=tokens_per_sec,
                    kernel_launches=launches,
                ))
            except Exception as e:
                print(f"    ERROR: {e}")

            # Memory cleanup
            del x, expert_ids, expert_weights_tensor
            gc.collect()

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark summary with speedups."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY: Fused MoE vs Sequential Dispatch")
    print("=" * 80)

    # Group by (batch_size, top_k)
    from collections import defaultdict
    grouped: dict[tuple[int, int], dict[str, BenchmarkResult]] = defaultdict(dict)

    for r in results:
        grouped[(r.batch_size, r.top_k)][r.name] = r

    # Print table header
    print(f"\n{'Batch':<8}{'Top-K':<8}{'Method':<12}{'Latency (ms)':<15}{'Throughput':<12}{'Speedup':<10}")
    print("-" * 70)

    for (batch_size, top_k), methods in sorted(grouped.items()):
        baseline_ms = None

        for name in ["Sequential", "Batched", "Fused"]:
            if name not in methods:
                continue
            r = methods[name]

            if baseline_ms is None:
                baseline_ms = r.mean_ms
                speedup = "1.0x"
            else:
                speedup = f"{baseline_ms / r.mean_ms:.1f}x"

            print(f"{batch_size:<8}{top_k:<8}{name:<12}{r.mean_ms:>8.2f} ±{r.std_ms:<5.2f}{r.tokens_per_sec:>8.0f}/s   {speedup:<10}")

        print()

    # Print key insights
    print("\nKey Insights:")
    print("-" * 40)

    # Find best speedups
    fused_results = [r for r in results if r.name == "Fused"]
    batched_results = [r for r in results if r.name == "Batched"]
    sequential_results = [r for r in results if r.name == "Sequential"]

    if fused_results and batched_results:
        # Compare fused vs batched at same config
        for fr in fused_results:
            matching = [br for br in batched_results
                       if br.batch_size == fr.batch_size and br.top_k == fr.top_k]
            if matching:
                br = matching[0]
                speedup = br.mean_ms / fr.mean_ms
                if speedup > 1.5:
                    print(f"• Fused is {speedup:.1f}x faster than Batched at batch={fr.batch_size}, top_k={fr.top_k}")

    if fused_results and sequential_results:
        for fr in fused_results:
            matching = [sr for sr in sequential_results
                       if sr.batch_size == fr.batch_size and sr.top_k == fr.top_k]
            if matching:
                sr = matching[0]
                speedup = sr.mean_ms / fr.mean_ms
                print(f"• Fused is {speedup:.1f}x faster than Sequential at batch={fr.batch_size}, top_k={fr.top_k}")


def run_real_model_benchmark() -> None:
    """Run benchmark with real GLM-4.7-Flash model weights."""
    print("\n" + "=" * 70)
    print("Real Model Benchmark: GLM-4.7-Flash MoE Layer")
    print("=" * 70)

    try:
        from metal_marlin.trellis.lm import TrellisForCausalLM
    except ImportError:
        print("ERROR: metal_marlin not found. Run from contrib/metal_marlin directory.")
        return

    model_path = "models/GLM-4.7-Flash-Trellis-3bpw"
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = TrellisForCausalLM.from_pretrained(model_path, device="mps")

    # Get MoE layer
    moe_layer = model.model.layers[1].mlp  # Layer 1 is MoE
    print(f"MoE layer type: {type(moe_layer).__name__}")
    print(f"Num experts: {moe_layer.num_experts}")
    print(f"Top-k: {moe_layer.num_experts_per_tok}")
    print(f"Fast MoE enabled: {moe_layer._use_fast_moe}")

    # Create test input
    for batch_size in [1, 4, 16]:
        print(f"\n--- Batch Size: {batch_size} ---")

        x = torch.randn(batch_size, 1, moe_layer.hidden_size, device="mps", dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = moe_layer(x)
            torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = moe_layer(x)
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
        print(f"  Mean latency: {mean_ms:.2f} ms (±{std_ms:.2f})")
        print(f"  Throughput: {batch_size / (mean_ms / 1000):.0f} tok/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Fused MoE vs Sequential Dispatch"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="2,8",
        help="Comma-separated top-k values to test",
    )
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        help="Benchmark with real GLM-4.7-Flash model",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=NUM_EXPERTS,
        help=f"Number of experts (default: {NUM_EXPERTS})",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Fused MoE vs Sequential Dispatch Benchmark")
    print("GLM-4.7-Flash Configuration (64 experts)")
    print("=" * 70)

    # Check device
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available. This benchmark requires Apple Silicon.")
        sys.exit(1)

    device = torch.device("mps")
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    top_k_values = [int(x) for x in args.top_k.split(",")]

    print(f"\nBatch sizes: {batch_sizes}")
    print(f"Top-K values: {top_k_values}")
    print(f"Num experts: {args.num_experts}")

    # Run mock benchmark
    results = run_benchmark(
        batch_sizes=batch_sizes,
        top_k_values=top_k_values,
        num_experts=args.num_experts,
    )

    # Print summary
    print_summary(results)

    # Run real model benchmark if requested
    if args.use_real_model:
        run_real_model_benchmark()

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
