#!/usr/bin/env python3
"""Benchmark Metal Hessian kernel throughput at various context lengths."""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_hessian_compute


def main():
    print("Initializing Metal kernel library...")
    metal_lib = MetalKernelLibrary.from_source_dir()
    print(f"  Loaded {len(metal_lib._pipelines)} cached pipelines")

    print()
    print("Benchmarking Hessian throughput at various context lengths")
    print("=" * 60)

    hidden_size = 2048  # GLM-4.7-Flash hidden size
    context_lengths = [2048, 8192, 16384, 32768]

    results = {}
    for ctx_len in context_lengths:
        print(f"Context {ctx_len:>6}...", end=" ", flush=True)

        # Generate test data
        X = torch.randn(ctx_len, hidden_size, device="mps")

        # Warmup
        for _ in range(3):
            H = dispatch_hessian_compute(metal_lib, X)
            torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            H = dispatch_hessian_compute(metal_lib, X)
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        tokens_per_sec = ctx_len / avg_time
        flops = 2 * ctx_len * hidden_size * hidden_size  # H = X^T @ X
        tflops = flops / avg_time / 1e12

        print(f"{tokens_per_sec / 1e6:.2f}M tok/s | {avg_time * 1000:.1f}ms | {tflops:.2f} TFLOPS")

        results[ctx_len] = {
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": avg_time * 1000,
            "tflops": tflops,
        }

        del X, H
        gc.collect()
        torch.mps.empty_cache()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Context':<10} {'Throughput':<15} {'Latency':<12} {'TFLOPS':<10}")
    print("-" * 47)
    for ctx, r in results.items():
        print(
            f"{ctx:<10} {r['tokens_per_sec'] / 1e6:.2f}M tok/s    {r['latency_ms']:.1f}ms       {r['tflops']:.2f}"
        )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
