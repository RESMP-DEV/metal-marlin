#!/usr/bin/env python
"""Benchmark MoE kernel across multiple use cases.

Tests: decode (bs=1), prefill (bs=4,8), batch inference (bs=16,32)
Memory tracking to verify we stay under budget.
"""

import resource
import time

import torch


def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

def benchmark_moe_forward(moe_layer, batch_sizes, hidden_dim, warmup=3, trials=10):
    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, hidden_dim, dtype=torch.float16, device='mps')
        torch.mps.synchronize()

        # Warmup
        for _ in range(warmup):
            _ = moe_layer(x)
            torch.mps.synchronize()

        # Timed
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            _ = moe_layer(x)
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)

        results[bs] = {
            'mean_ms': sum(times) / len(times) * 1000,
            'tokens_per_sec': bs / (sum(times) / len(times)),
            'memory_mb': get_rss_mb(),
        }

    return results

def main():
    from metal_marlin.trellis.model import TrellisForCausalLM

import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

    print("Loading model...")
    model = TrellisForCausalLM.from_pretrained(
        'models/GLM-4.7-Flash-Trellis-3bpw', device='mps'
    )

    moe = model.model.layers[1].mlp  # Use layer 1's MoE

    print("\nBenchmarking MoE forward pass:")
    print("-" * 60)
    print(f"{'Batch':>8} {'Time (ms)':>12} {'TPS':>10} {'Memory (MB)':>12}")
    print("-" * 60)

    results = benchmark_moe_forward(
        moe,
        batch_sizes=[1, 4, 8, 16, 32],
        hidden_dim=moe.hidden_dim
    )

    for bs, r in results.items():
        print(f"{bs:>8} {r['mean_ms']:>12.2f} {r['tokens_per_sec']:>10.1f} {r['memory_mb']:>12.0f}")

    # Memory budget check
    final_rss = get_rss_mb()
    assert final_rss < 20000, f"Memory exceeded 20GB budget: {final_rss:.0f} MB"
    print(f"\nMemory budget: PASS ({final_rss:.0f} MB < 20000 MB)")

if __name__ == '__main__':
    main()
