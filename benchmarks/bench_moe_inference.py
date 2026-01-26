#!/usr/bin/env python3
"""
Inference benchmarks for MoE models with standard vs layer-aware quantization.

Measures:
- Prefill throughput (tok/s) at various context lengths
- Decode throughput (tok/s) for token generation
- Memory usage

Usage:
    python bench_moe_inference.py --model Qwen/Qwen3-30B-A3B --max-layers 200
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin import quantized_linear
from metal_marlin.quantize import pack_fp4_weights


@dataclass
class BenchmarkResult:
    """Results from inference benchmark."""

    method: str
    context_length: int
    prefill_tok_s: float
    decode_tok_s: float
    prefill_time_ms: float
    decode_time_ms: float
    memory_mb: float


class QuantizedMLPBenchmark(nn.Module):
    """Simple MLP benchmark module with quantized weights."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        group_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers

        # Pre-quantize weights
        self.layers: list[dict[str, Any]] = []

        for i in range(num_layers):
            # Gate projection
            gate_w = mx.random.normal((hidden_size, intermediate_size)).astype(mx.float16)
            gate_packed, gate_scales, _ = pack_fp4_weights(gate_w, group_size=group_size)

            # Up projection
            up_w = mx.random.normal((hidden_size, intermediate_size)).astype(mx.float16)
            up_packed, up_scales, _ = pack_fp4_weights(up_w, group_size=group_size)

            # Down projection
            down_w = mx.random.normal((intermediate_size, hidden_size)).astype(mx.float16)
            down_packed, down_scales, _ = pack_fp4_weights(down_w, group_size=group_size)

            self.layers.append(
                {
                    "gate_packed": gate_packed,
                    "gate_scales": gate_scales,
                    "up_packed": up_packed,
                    "up_scales": up_scales,
                    "down_packed": down_packed,
                    "down_scales": down_scales,
                }
            )

        mx.eval([v for layer in self.layers for v in layer.values()])

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through all layers."""
        h = x
        for layer in self.layers:
            # SwiGLU: down(silu(gate(x)) * up(x))
            gate = quantized_linear(h, layer["gate_packed"], layer["gate_scales"])
            up = quantized_linear(h, layer["up_packed"], layer["up_scales"])
            h_mlp = nn.silu(gate) * up
            h = quantized_linear(h_mlp, layer["down_packed"], layer["down_scales"])
        return h


def benchmark_prefill(
    model: QuantizedMLPBenchmark,
    batch_size: int,
    seq_length: int,
    warmup: int = 3,
    iterations: int = 10,
) -> tuple[float, float]:
    """Benchmark prefill (forward pass on full sequence)."""
    x = mx.random.normal((batch_size, seq_length, model.hidden_size)).astype(mx.float16)
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        _ = model(x)
        mx.eval(model(x))
        mx.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter()
        out = model(x)
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    tokens = batch_size * seq_length
    tok_s = tokens / avg_time

    return tok_s, avg_time * 1000


def benchmark_decode(
    model: QuantizedMLPBenchmark,
    batch_size: int,
    num_tokens: int,
    warmup: int = 3,
    iterations: int = 10,
) -> tuple[float, float]:
    """Benchmark decode (sequential single-token forward passes)."""
    x = mx.random.normal((batch_size, 1, model.hidden_size)).astype(mx.float16)
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        for _ in range(min(num_tokens, 10)):
            _ = model(x)
            mx.eval(model(x))
        mx.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter()
        for _ in range(num_tokens):
            out = model(x)
            mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    tok_s = (batch_size * num_tokens) / avg_time

    return tok_s, avg_time * 1000


def run_benchmark_suite(
    hidden_size: int = 2048,
    intermediate_size: int = 5632,
    num_layers: int = 8,
    group_sizes: list[int] | None = None,
    context_lengths: list[int] | None = None,
    decode_tokens: int = 64,
) -> list[BenchmarkResult]:
    """Run full benchmark suite comparing different group sizes."""
    if group_sizes is None:
        group_sizes = [32, 64, 128]
    if context_lengths is None:
        context_lengths = [128, 512, 1024, 2048]

    results: list[BenchmarkResult] = []

    for group_size in group_sizes:
        method = f"group_size={group_size}"
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {method}")
        print(f"  Hidden: {hidden_size}, Intermediate: {intermediate_size}")
        print(f"  Layers: {num_layers}")
        print(f"{'=' * 60}")

        # Create model
        model = QuantizedMLPBenchmark(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            group_size=group_size,
        )

        for ctx_len in context_lengths:
            # Prefill
            prefill_tok_s, prefill_ms = benchmark_prefill(model, batch_size=1, seq_length=ctx_len)

            # Decode
            decode_tok_s, decode_ms = benchmark_decode(
                model, batch_size=1, num_tokens=decode_tokens
            )

            result = BenchmarkResult(
                method=method,
                context_length=ctx_len,
                prefill_tok_s=prefill_tok_s,
                decode_tok_s=decode_tok_s,
                prefill_time_ms=prefill_ms,
                decode_time_ms=decode_ms,
                memory_mb=0,  # TODO: measure actual memory
            )
            results.append(result)

            print(
                f"  ctx={ctx_len:<5}: prefill={prefill_tok_s:>7.0f} tok/s, decode={decode_tok_s:>6.1f} tok/s"
            )

        # Cleanup
        del model
        gc.collect()
        mx.synchronize()

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Context':<10} {'Prefill (tok/s)':<18} {'Decode (tok/s)':<15}")
    print("-" * 80)

    for r in results:
        print(
            f"{r.method:<20} {r.context_length:<10} {r.prefill_tok_s:<18.0f} {r.decode_tok_s:<15.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="MoE Inference Benchmark")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden size")
    parser.add_argument("--intermediate-size", type=int, default=5632, help="FFN intermediate size")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of MLP layers")
    parser.add_argument(
        "--group-sizes", type=int, nargs="+", default=[32, 64, 128], help="Group sizes to benchmark"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[128, 512, 1024],
        help="Context lengths to benchmark",
    )
    parser.add_argument("--decode-tokens", type=int, default=64, help="Tokens to decode")
    args = parser.parse_args()

    print("MoE Inference Benchmark")
    print("  Device: Apple Silicon (Metal via MLX)")
    print(
        f"  Config: hidden={args.hidden_size}, ffn={args.intermediate_size}, layers={args.num_layers}"
    )

    results = run_benchmark_suite(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        group_sizes=args.group_sizes,
        context_lengths=args.context_lengths,
        decode_tokens=args.decode_tokens,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
