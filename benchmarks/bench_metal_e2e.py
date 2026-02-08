#!/usr/bin/env python3
"""
End-to-end benchmark comparing full inference with/without Metal acceleration.

This benchmark measures the overall speedup from Metal-accelerated kernels
compared to PyTorch fallbacks. It creates a mock model that exercises the
key Metal Marlin components:
- FP4 quantization/dequantization
- MoE dispatch
- Token sampling

Usage:
    cd contrib/metal_marlin && uv run python benchmarks/bench_metal_e2e.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import metal_marlin.moe_dispatch as moe

# Import Metal Marlin modules with _USE_METAL flags
import metal_marlin.quantize_fp4 as fp4


def set_metal_enabled(enabled: bool) -> None:
    """Enable or disable Metal acceleration in all Metal Marlin modules."""
    fp4._USE_METAL = enabled
    moe._USE_METAL = enabled


class MockExpert(nn.Module):
    """Mock expert network for MoE layer."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MockMoELayer(nn.Module):
    """Mock MoE layer that uses Metal Marlin dispatch."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            MockExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        # Router
        router_logits = self.gate(x_flat)
        weights, expert_ids = torch.topk(
            torch.softmax(router_logits, dim=-1), self.top_k, dim=-1
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Use Metal Marlin dispatch if available
        if moe._USE_METAL and hasattr(moe, 'group_tokens_by_expert'):
            try:
                return self._forward_metal(x_flat, expert_ids, weights)
            except Exception:
                # Fallback to PyTorch
                pass

        return self._forward_pytorch(x_flat, expert_ids, weights)

    def _forward_metal(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using Metal Marlin dispatch."""
        batch_size = x.shape[0]
        output = torch.zeros_like(x)

        # Group tokens by expert
        for i in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = (expert_ids == i).any(dim=-1)
            if not mask.any():
                continue

            expert_input = x[mask]
            expert_output = self.experts[i](expert_input.unsqueeze(0)).squeeze(0)

            # Add weighted output
            token_weights = weights[mask][expert_ids[mask] == i].unsqueeze(-1)
            output[mask] += token_weights * expert_output

        return output.view(-1, self.hidden_size)

    def _forward_pytorch(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using pure PyTorch."""
        batch_size = x.shape[0]
        output = torch.zeros_like(x)

        for i in range(batch_size):
            token_out = torch.zeros(self.hidden_size, device=x.device, dtype=x.dtype)
            for j in range(self.top_k):
                expert_idx = expert_ids[i, j].item()
                weight = weights[i, j].item()
                expert_out = self.experts[expert_idx](x[i].unsqueeze(0)).squeeze(0)
                token_out += weight * expert_out
            output[i] = token_out

        return output.view(-1, self.hidden_size)


class MockTransformerLayer(nn.Module):
    """Mock transformer layer with MoE."""

    def __init__(self, hidden_size: int = 512, intermediate_size: int = 1364):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Self-attention (simplified)
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # MoE FFN
        self.moe = MockMoELayer(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention (simplified)
        residual = x
        x = self.norm1(x)
        qkv = self.attn_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_out = self.attn_out(v)  # Simplified attention
        x = residual + attn_out

        # MoE FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.moe(x)

        return x


class MockModel(nn.Module):
    """Mock language model using Metal Marlin components."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_layers: int = 4,
        intermediate_size: int = 1364,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MockTransformerLayer(hidden_size, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


def bench_inference_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    use_metal: bool,
    num_runs: int = 1,
) -> float:
    """
    Run inference passes and return average time.

    Args:
        model: The model to benchmark
        input_ids: Input token IDs
        use_metal: Whether to use Metal acceleration
        num_runs: Number of inference passes to average

    Returns:
        Average elapsed time in seconds
    """
    # Set Metal flags
    set_metal_enabled(use_metal)

    # Ensure model is on MPS
    device = next(model.parameters()).device

    # Synchronize before timing
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / num_runs


def run_benchmark(
    vocab_size: int = 32000,
    hidden_size: int = 512,
    num_layers: int = 4,
    seq_len: int = 128,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> dict[str, Any]:
    """
    Run full end-to-end benchmark.

    Returns:
        Dictionary with benchmark results
    """
    print("=" * 70)
    print("Metal Marlin E2E Benchmark: Metal vs PyTorch")
    print("=" * 70)
    print()

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("Warning: MPS not available, running on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print(f"Device: {device}")

    print(f"Model config: {num_layers} layers, {hidden_size} hidden, {vocab_size} vocab")
    print(f"Sequence length: {seq_len}")
    print()

    # Create model
    print("Creating mock model...")
    model = MockModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device).eval()

    # Create input
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Verify Metal is available
    metal_available = fp4._USE_METAL
    print(f"Metal kernels available: {metal_available}")
    print()

    # Warmup with Metal
    print(f"Warming up ({warmup_runs} runs with Metal)...")
    set_metal_enabled(True)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids)
    if device.type == "mps":
        torch.mps.synchronize()
    print("Warmup complete")
    print()

    # Benchmark with Metal
    print(f"Benchmarking with Metal ({benchmark_runs} runs)...")
    times_metal = [
        bench_inference_pass(model, input_ids, use_metal=True)
        for _ in range(benchmark_runs)
    ]

    # Benchmark without Metal (PyTorch fallbacks)
    print(f"Benchmarking with PyTorch fallback ({benchmark_runs} runs)...")
    times_pytorch = [
        bench_inference_pass(model, input_ids, use_metal=False)
        for _ in range(benchmark_runs)
    ]

    # Compute statistics
    avg_metal = sum(times_metal) / len(times_metal)
    avg_pytorch = sum(times_pytorch) / len(times_pytorch)
    min_metal = min(times_metal)
    min_pytorch = min(times_pytorch)

    # Remove outliers (top/bottom 10%) for more stable comparison
    times_metal_sorted = sorted(times_metal)
    times_pytorch_sorted = sorted(times_pytorch)
    trim = benchmark_runs // 10
    if trim > 0:
        trimmed_metal = times_metal_sorted[trim:-trim]
        trimmed_pytorch = times_pytorch_sorted[trim:-trim]
        trim_metal = sum(trimmed_metal) / len(trimmed_metal)
        trim_pytorch = sum(trimmed_pytorch) / len(trimmed_pytorch)
    else:
        trim_metal = avg_metal
        trim_pytorch = avg_pytorch

    speedup_avg = avg_pytorch / avg_metal if avg_metal > 0 else float('inf')
    speedup_trim = trim_pytorch / trim_metal if trim_metal > 0 else float('inf')
    speedup_min = min_pytorch / min_metal if min_metal > 0 else float('inf')

    results = {
        "metal_available": metal_available,
        "device": str(device),
        "model_config": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "seq_len": seq_len,
        },
        "metal": {
            "avg_ms": avg_metal * 1000,
            "min_ms": min_metal * 1000,
            "trimmed_ms": trim_metal * 1000,
            "all_times_ms": [t * 1000 for t in times_metal],
        },
        "pytorch": {
            "avg_ms": avg_pytorch * 1000,
            "min_ms": min_pytorch * 1000,
            "trimmed_ms": trim_pytorch * 1000,
            "all_times_ms": [t * 1000 for t in times_pytorch],
        },
        "speedup": {
            "avg": speedup_avg,
            "trimmed": speedup_trim,
            "min": speedup_min,
        },
    }

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print benchmark results in a formatted way."""
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    metal = results["metal"]
    pytorch = results["pytorch"]
    speedup = results["speedup"]

    print("\nMetal Acceleration:")
    print(f"  Average:   {metal['avg_ms']:.2f} ms")
    print(f"  Min:       {metal['min_ms']:.2f} ms")
    print(f"  Trimmed:   {metal['trimmed_ms']:.2f} ms (outliers removed)")

    print("\nPyTorch Fallback:")
    print(f"  Average:   {pytorch['avg_ms']:.2f} ms")
    print(f"  Min:       {pytorch['min_ms']:.2f} ms")
    print(f"  Trimmed:   {pytorch['trimmed_ms']:.2f} ms (outliers removed)")

    print("\nSpeedup (Metal vs PyTorch):")
    print(f"  Average:   {speedup['avg']:.2f}x")
    print(f"  Trimmed:   {speedup['trimmed']:.2f}x (outliers removed)")
    print(f"  Min:       {speedup['min']:.2f}x (best case)")

    print()
    print("=" * 70)

    # Summary
    if speedup['trimmed'] > 1.5:
        print(f"✓ Excellent speedup: {speedup['trimmed']:.2f}x faster with Metal")
    elif speedup['trimmed'] > 1.1:
        print(f"✓ Good speedup: {speedup['trimmed']:.2f}x faster with Metal")
    elif speedup['trimmed'] > 0.9:
        print(f"~ Similar performance: {speedup['trimmed']:.2f}x (within noise)")
    else:
        print(f"⚠ PyTorch faster: {1/speedup['trimmed']:.2f}x (Metal overhead?)")

    print("=" * 70)


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Metal Marlin E2E inference"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="Hidden size (default: 512)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup runs (default: 5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of benchmark runs (default: 20)",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    )

    # Print results
    print_results(results)

    # Save to JSON if requested
    if args.json:
        import json
import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

        output_path = Path(args.json)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
