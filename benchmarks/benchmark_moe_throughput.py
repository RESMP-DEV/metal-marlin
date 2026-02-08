#!/usr/bin/env python3
"""Comprehensive MoE throughput benchmarks for Metal Marlin.

Measures tokens/second across various configurations:
- Batch sizes: 1, 4, 8, 16, 32, 64, 128
- Sequence lengths: 1 (decode), 128, 512, 2048 (prefill)
- Model configs: 64 experts top-8, 8 experts top-2

Compares fast MoE kernel vs slow PyTorch path with proper warmup
and statistical analysis.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

@dataclass
class MoEBenchmarkResult:
    """Result from a single MoE benchmark configuration."""

    name: str
    batch_size: int
    seq_len: int
    num_experts: int
    top_k: int
    hidden_dim: int
    intermediate_dim: int

    # Timing statistics (milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    # Throughput
    tokens_per_sec: float

    # Memory
    peak_memory_mb: float

    # Metadata
    path: str  # "fast" or "slow"
    iterations: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class MoEComparisonResult:
    """Comparison between fast and slow MoE paths."""

    config_name: str
    batch_size: int
    seq_len: int
    num_experts: int
    top_k: int

    fast_mean_ms: float
    slow_mean_ms: float
    speedup: float

    fast_memory_mb: float
    slow_memory_mb: float
    memory_ratio: float

    fast_tokens_per_sec: float
    slow_tokens_per_sec: float


def get_mps_memory_mb() -> float:
    """Get current MPS memory allocation in MB."""
    if hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def reset_memory_stats() -> None:
    """Reset memory tracking."""
    torch.mps.synchronize()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


class MockTrellisLinear(nn.Module):
    """Mock TrellisLinear for benchmarking without loading actual weights.

    Uses standard FP16 linear but matches the interface expected by MoE layers.
    """

    def __init__(self, in_features: int, out_features: int, device: str = "mps"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = 3
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.float16, device=device) * 0.02
        )

        # Mock trellis attributes
        self.register_buffer(
            "packed_indices",
            torch.zeros(1, 1, 96, dtype=torch.uint8, device=device)
        )
        self.register_buffer("scales", torch.ones(1, out_features, dtype=torch.float32, device=device))
        self.register_buffer("su", torch.ones(in_features, dtype=torch.float32, device=device))
        self.register_buffer("sv", torch.ones(out_features, dtype=torch.float32, device=device))
        self.register_buffer("grid", torch.linspace(-1, 1, 8, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MockTrellisDenseMLP(nn.Module):
    """Mock dense MLP matching TrellisDenseMLP interface."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, device: str = "mps"):
        super().__init__()
        self.gate_proj = MockTrellisLinear(hidden_dim, intermediate_dim, device)
        self.up_proj = MockTrellisLinear(hidden_dim, intermediate_dim, device)
        self.down_proj = MockTrellisLinear(intermediate_dim, hidden_dim, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockMoELayer(nn.Module):
    """Mock MoE layer for benchmarking both fast and slow paths.

    Simulates the structure of TrellisMoEMLP without actual trellis quantization.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        device: str = "mps",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.device = device

        # Router
        self.router = nn.Linear(hidden_dim, num_experts, bias=False, device=device)

        # Experts
        self.experts = nn.ModuleList([
            MockTrellisDenseMLP(hidden_dim, intermediate_dim, device)
            for _ in range(num_experts)
        ])

        # Shared expert
        self.shared_expert = MockTrellisDenseMLP(hidden_dim, intermediate_dim, device)

        # Control fast vs slow path - public for benchmark configuration
        self.use_fast_path = True

    def forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Simulated fast path using batched expert computation.

        In the real implementation, this uses a fused Metal kernel.
        Here we simulate it with grouped matmuls.
        """
        orig_dtype = x.dtype
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.hidden_dim)
        num_tokens = x_flat.shape[0]

        # Get router scores
        router_logits = self.router(x_flat.float())
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float),
            k=self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Batched expert computation - group tokens by expert
        output = torch.zeros(num_tokens, self.hidden_dim, dtype=torch.float16, device=self.device)

        for k in range(self.top_k):
            experts_for_slot = selected_experts[:, k]
            weights_for_slot = routing_weights[:, k:k+1]

            for expert_id in range(self.num_experts):
                mask = (experts_for_slot == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_output * weights_for_slot[mask]

        # Add shared expert
        output = output + self.shared_expert(x_flat)

        return output.view(*batch_shape, self.hidden_dim).to(orig_dtype)

    def forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        """Slow path - sequential expert iteration."""
        # Get router scores
        router_logits = self.router(x.float())
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float),
            k=self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Find unique experts
        unique_experts: list[int] = selected_experts.unique().tolist()

        # Initialize output
        output = torch.zeros_like(x)

        # Process each unique expert
        for expert_id in unique_experts:
            expert_mask: torch.Tensor = selected_experts == expert_id
            weights_for_expert = torch.where(
                expert_mask,
                routing_weights,
                torch.zeros_like(routing_weights),
            ).sum(dim=-1)

            expert_output = self.experts[expert_id](x)
            output += expert_output * weights_for_expert.unsqueeze(-1)

            del expert_output, weights_for_expert, expert_mask

        # Add shared expert
        output = output + self.shared_expert(x)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fast_path:
            return self.forward_fast(x)
        return self.forward_slow(x)


class MoEBenchmark:
    """Benchmark harness for MoE throughput testing."""

    def __init__(
        self,
        warmup: int = 5,
        iterations: int = 20,
        device: str = "mps",
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.device = device
        self.results: list[MoEBenchmarkResult] = []
        self.comparisons: list[MoEComparisonResult] = []

    def _run_single(
        self,
        name: str,
        model: nn.Module,
        x: torch.Tensor,
        path: str,
        batch_size: int,
        seq_len: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> MoEBenchmarkResult:
        """Run benchmark for a single configuration."""
        reset_memory_stats()

        # Warmup
        for _ in range(self.warmup):
            with torch.inference_mode():
                _ = model(x)
            torch.mps.synchronize()

        # Timed runs
        times_ms: list[float] = []
        peak_memory = 0.0

        for _ in range(self.iterations):
            torch.mps.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                _ = model(x)

            torch.mps.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)

            current_mem = get_mps_memory_mb()
            peak_memory = max(peak_memory, current_mem)

        # Statistics
        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_ms = min(times_ms)
        max_ms = max(times_ms)

        # Throughput
        total_tokens = batch_size * seq_len
        tokens_per_sec = (total_tokens / mean_ms) * 1000.0

        result = MoEBenchmarkResult(
            name=name,
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            tokens_per_sec=tokens_per_sec,
            peak_memory_mb=peak_memory,
            path=path,
            iterations=self.iterations,
        )

        self.results.append(result)
        return result

    def run_comparison(
        self,
        config_name: str,
        batch_size: int,
        seq_len: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int = 3584,
        intermediate_dim: int = 18944,
    ) -> MoEComparisonResult:
        """Run comparison between fast and slow paths."""
        # Create mock MoE layer
        model = MockMoELayer(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            device=self.device,
        )

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=self.device)

        # Benchmark fast path
        model._use_fast_path = True
        fast_result = self._run_single(
            name=f"{config_name}_fast",
            model=model,
            x=x,
            path="fast",
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

        # Benchmark slow path
        model._use_fast_path = False
        slow_result = self._run_single(
            name=f"{config_name}_slow",
            model=model,
            x=x,
            path="slow",
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

        # Calculate comparison metrics
        speedup = slow_result.mean_ms / fast_result.mean_ms if fast_result.mean_ms > 0 else 0.0
        memory_ratio = slow_result.peak_memory_mb / fast_result.peak_memory_mb if fast_result.peak_memory_mb > 0 else 0.0

        comparison = MoEComparisonResult(
            config_name=config_name,
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=num_experts,
            top_k=top_k,
            fast_mean_ms=fast_result.mean_ms,
            slow_mean_ms=slow_result.mean_ms,
            speedup=speedup,
            fast_memory_mb=fast_result.peak_memory_mb,
            slow_memory_mb=slow_result.peak_memory_mb,
            memory_ratio=memory_ratio,
            fast_tokens_per_sec=fast_result.tokens_per_sec,
            slow_tokens_per_sec=slow_result.tokens_per_sec,
        )

        self.comparisons.append(comparison)
        return comparison

    def run_sweep(
        self,
        batch_sizes: list[int] | None = None,
        seq_lengths: list[int] | None = None,
        model_configs: list[tuple[int, int]] | None = None,
    ) -> list[MoEComparisonResult]:
        """Run full benchmark sweep across configurations."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        if seq_lengths is None:
            seq_lengths = [1, 128, 512, 2048]
        if model_configs is None:
            # (num_experts, top_k)
            model_configs = [(64, 8), (8, 2)]

        results = []
        total = len(batch_sizes) * len(seq_lengths) * len(model_configs)
        current = 0

        for num_experts, top_k in model_configs:
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    current += 1
                    config_name = f"e{num_experts}_k{top_k}_b{batch_size}_s{seq_len}"
                    print(f"[{current}/{total}] Running {config_name}...")

                    try:
                        result = self.run_comparison(
                            config_name=config_name,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            num_experts=num_experts,
                            top_k=top_k,
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        continue

        return results

    def export_json(self, path: str | Path) -> None:
        """Export results to JSON."""
        data = {
            "results": [asdict(r) for r in self.results],
            "comparisons": [asdict(c) for c in self.comparisons],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_markdown_table(self) -> str:
        """Generate markdown table of comparison results."""
        if not self.comparisons:
            return "No results available."

        lines = [
            "| Config | Batch | SeqLen | Experts | TopK | Fast (ms) | Slow (ms) | Speedup | Fast (tok/s) | Slow (tok/s) |",
            "|--------|-------|--------|---------|------|-----------|-----------|---------|--------------|--------------|",
        ]

        for c in self.comparisons:
            lines.append(
                f"| {c.config_name} | {c.batch_size} | {c.seq_len} | {c.num_experts} | {c.top_k} | "
                f"{c.fast_mean_ms:.2f} | {c.slow_mean_ms:.2f} | {c.speedup:.2f}x | "
                f"{c.fast_tokens_per_sec:.0f} | {c.slow_tokens_per_sec:.0f} |"
            )

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print("\n" + "=" * 80)
        print("MoE Throughput Benchmark Results")
        print("=" * 80)

        if self.comparisons:
            # Group by model config
            configs: dict[tuple[int, int], list[MoEComparisonResult]] = {}
            for c in self.comparisons:
                key = (c.num_experts, c.top_k)
                if key not in configs:
                    configs[key] = []
                configs[key].append(c)

            for (num_experts, top_k), results in configs.items():
                print(f"\n### {num_experts} Experts, Top-{top_k}")
                print("-" * 70)
                print(f"{'Batch':>6} {'SeqLen':>8} {'Fast(ms)':>10} {'Slow(ms)':>10} {'Speedup':>8} {'Tok/s':>12}")
                print("-" * 70)

                for c in results:
                    print(
                        f"{c.batch_size:>6} {c.seq_len:>8} "
                        f"{c.fast_mean_ms:>10.2f} {c.slow_mean_ms:>10.2f} "
                        f"{c.speedup:>7.2f}x {c.fast_tokens_per_sec:>12.0f}"
                    )

        print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE Throughput Benchmark")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of timed iterations"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=None,
        help="Batch sizes to test (default: 1,4,8,16,32,64,128)"
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+", default=None,
        help="Sequence lengths to test (default: 1,128,512,2048)"
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--output-md", type=str, default=None,
        help="Path to save markdown table"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer configurations for fast testing"
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend required for MoE benchmarking")

    print("MoE Throughput Benchmark")
    print("  Device: MPS (Apple Silicon)")
    print(f"  Warmup: {args.warmup} iterations")
    print(f"  Measurement: {args.iterations} iterations")
    print()

    bench = MoEBenchmark(
        warmup=args.warmup,
        iterations=args.iterations,
    )

    # Determine configurations
    if args.quick:
        batch_sizes = [1, 16, 64]
        seq_lengths = [1, 512]
        model_configs = [(64, 8)]
    else:
        batch_sizes = args.batch_sizes or [1, 4, 8, 16, 32, 64, 128]
        seq_lengths = args.seq_lengths or [1, 128, 512, 2048]
        model_configs = [(64, 8), (8, 2)]

    # Run sweep
    bench.run_sweep(
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        model_configs=model_configs,
    )

    # Print summary
    bench.print_summary()

    # Export results
    if args.output_json:
        bench.export_json(args.output_json)
        print(f"\nJSON results saved to: {args.output_json}")

    if args.output_md:
        md_table = bench.print_markdown_table()
        with open(args.output_md, "w") as f:
            f.write("# MoE Throughput Benchmark Results\n\n")
            f.write(md_table)
        print(f"Markdown table saved to: {args.output_md}")


if __name__ == "__main__":
    main()
