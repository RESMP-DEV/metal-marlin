#!/usr/bin/env python3
"""Attention throughput benchmarks.

Benchmarks PyTorch scaled_dot_product_attention on MPS against a naive
reference implementation across common LLM shapes.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure framework is importable from benchmarks directory
sys.path.insert(0, str(Path(__file__).parent))
from framework import mps_sync  # noqa: E402

WARMUP_ITERS = 10
BENCH_ITERS = 50


def _check_mps() -> bool:
    """Check if MPS backend is available."""
    if not torch.backends.mps.is_available():
        print("ERROR: MPS backend not available.")
        print("This benchmark requires Apple Silicon with PyTorch MPS support.")
        return False
    return True


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Naive attention for reference timing."""
    # Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    # Softmax
    probs = F.softmax(scores, dim=-1)
    # Attention @ V
    return torch.matmul(probs, v)


def _time_kernel(fn: Callable[[], torch.Tensor]) -> float:
    """Time a kernel with warmup and MPS synchronization."""
    for _ in range(WARMUP_ITERS):
        out = fn()
        mps_sync()

    times: list[float] = []
    for _ in range(BENCH_ITERS):
        start = time.perf_counter()
        _out = fn()  # Result discarded, timing the call
        mps_sync()
        times.append((time.perf_counter() - start) * 1000.0)

    return statistics.mean(times)


def benchmark_sdpa(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> float:
    """Benchmark PyTorch scaled_dot_product_attention on MPS."""
    q = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float16, device="mps")
    k = torch.randn(batch, heads, seq_k, head_dim, dtype=torch.float16, device="mps")
    v = torch.randn(batch, heads, seq_k, head_dim, dtype=torch.float16, device="mps")
    mps_sync()

    scale = 1.0 / math.sqrt(head_dim)

    def fn() -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    return _time_kernel(fn)


def benchmark_reference_attention(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> float:
    """Benchmark naive reference attention on MPS."""
    q = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float16, device="mps")
    k = torch.randn(batch, heads, seq_k, head_dim, dtype=torch.float16, device="mps")
    v = torch.randn(batch, heads, seq_k, head_dim, dtype=torch.float16, device="mps")
    mps_sync()

    scale = 1.0 / math.sqrt(head_dim)

    def fn() -> torch.Tensor:
        return _reference_attention(q, k, v, scale)

    return _time_kernel(fn)


def benchmark_sdpa_gqa(
    batch: int,
    heads_q: int,
    heads_kv: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> float:
    """Benchmark GQA attention where K/V have fewer heads than Q."""
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float16, device="mps")
    k = torch.randn(batch, heads_kv, seq_k, head_dim, dtype=torch.float16, device="mps")
    v = torch.randn(batch, heads_kv, seq_k, head_dim, dtype=torch.float16, device="mps")
    mps_sync()

    scale = 1.0 / math.sqrt(head_dim)
    gqa_ratio = heads_q // heads_kv

    def fn() -> torch.Tensor:
        # Expand K/V to match Q heads
        k_expanded = k.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v.repeat_interleave(gqa_ratio, dim=1)
        return F.scaled_dot_product_attention(q, k_expanded, v_expanded, scale=scale)

    return _time_kernel(fn)


def benchmark_attention() -> None:
    """Run attention benchmarks across common LLM configurations."""
    if not _check_mps():
        sys.exit(1)

    configs = [
        # (batch, heads, seq_q, seq_k, head_dim)
        (1, 32, 1, 512, 128),  # Decode, short context
        (1, 32, 1, 4096, 128),  # Decode, medium context
        (1, 32, 1, 32768, 128),  # Decode, long context
        (1, 32, 512, 512, 128),  # Prefill
        (32, 32, 1, 512, 128),  # Batched decode
    ]

    print("=" * 70)
    print("PyTorch SDPA vs Reference Attention (MPS)")
    print("=" * 70)

    for batch, heads, seq_q, seq_k, head_dim in configs:
        t_sdpa = benchmark_sdpa(batch, heads, seq_q, seq_k, head_dim)
        t_ref = benchmark_reference_attention(batch, heads, seq_q, seq_k, head_dim)

        # FLOPs: 4 * batch * heads * seq_q * seq_k * head_dim
        # (2 for Q@K^T, 2 for attn@V)
        flops = 4 * batch * heads * seq_q * seq_k * head_dim
        tflops_sdpa = flops / (t_sdpa / 1000.0) / 1e12
        tflops_ref = flops / (t_ref / 1000.0) / 1e12
        speedup = t_ref / t_sdpa

        print(
            f"B={batch} H={heads} Sq={seq_q:>5} Sk={seq_k:>5} D={head_dim}: "
            f"SDPA={t_sdpa:.3f}ms ({tflops_sdpa:.2f} TFLOPS), "
            f"Ref={t_ref:.3f}ms ({tflops_ref:.2f} TFLOPS), "
            f"Speedup={speedup:.2f}x"
        )


def benchmark_gqa() -> None:
    """Run GQA-specific benchmarks."""
    if not _check_mps():
        sys.exit(1)

    gqa_configs = [
        # (batch, heads_q, heads_kv, seq_q, seq_k, head_dim, model)
        (1, 32, 8, 1, 4096, 128, "Llama-3-8B GQA"),
        (1, 32, 2, 1, 4096, 64, "GLM-4.7-Flash"),
        (1, 64, 8, 1, 8192, 128, "Llama-3-70B GQA"),
        (1, 32, 1, 1, 4096, 128, "MQA (single KV)"),
    ]

    print()
    print("=" * 70)
    print("Grouped Query Attention (GQA) Benchmarks")
    print("=" * 70)

    for batch, heads_q, heads_kv, seq_q, seq_k, head_dim, model in gqa_configs:
        t_gqa = benchmark_sdpa_gqa(batch, heads_q, heads_kv, seq_q, seq_k, head_dim)

        flops = 4 * batch * heads_q * seq_q * seq_k * head_dim
        tflops = flops / (t_gqa / 1000.0) / 1e12
        gqa_ratio = heads_q // heads_kv

        print(
            f"{model:<20} (Q={heads_q}, KV={heads_kv}, ratio={gqa_ratio:>2}): "
            f"{t_gqa:.3f}ms ({tflops:.2f} TFLOPS)"
        )


def benchmark_long_context() -> None:
    """Benchmark attention with very long context lengths."""
    if not _check_mps():
        sys.exit(1)

    long_configs = [
        # (batch, heads, seq_q, seq_k, head_dim)
        (1, 32, 1, 8192, 128),
        (1, 32, 1, 16384, 128),
        (1, 32, 1, 32768, 128),
        (1, 32, 1, 65536, 128),
        (1, 32, 1, 131072, 128),
    ]

    print()
    print("=" * 70)
    print("Long Context Attention Benchmarks")
    print("=" * 70)

    for batch, heads, seq_q, seq_k, head_dim in long_configs:
        try:
            t_sdpa = benchmark_sdpa(batch, heads, seq_q, seq_k, head_dim)
            flops = 4 * batch * heads * seq_q * seq_k * head_dim
            tflops = flops / (t_sdpa / 1000.0) / 1e12

            # Memory: Q + K + V + O, all float16
            mem_mb = batch * heads * (seq_q + 2 * seq_k + seq_q) * head_dim * 2 / 1e6

            print(f"Context={seq_k:>6}: {t_sdpa:.3f}ms ({tflops:.2f} TFLOPS), ~{mem_mb:.1f} MB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Context={seq_k:>6}: OOM")
            else:
                raise


def main() -> None:
    """Run all attention benchmarks."""
    benchmark_attention()
    benchmark_gqa()
    benchmark_long_context()


if __name__ == "__main__":
    main()
