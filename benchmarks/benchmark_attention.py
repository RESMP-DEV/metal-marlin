#!/usr/bin/env python3
"""Attention throughput benchmarks.

Benchmarks Metal-accelerated scaled dot-product attention (if available)
against a simple MLX reference implementation across common LLM shapes.
"""

from __future__ import annotations

import math
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

WARMUP_ITERS = 10
BENCH_ITERS = 50


def _reference_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
) -> mx.array:
    """Naive attention for reference timing."""
    k_t = mx.transpose(k, (0, 1, 3, 2))
    scores = mx.matmul(q, k_t) * scale
    if hasattr(mx, "softmax"):
        probs = mx.softmax(scores, axis=-1)
    else:
        scores = scores - mx.max(scores, axis=-1, keepdims=True)
        probs = mx.exp(scores)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
    return mx.matmul(probs, v)


def _resolve_fast_attention() -> callable | None:
    if hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention"):
        return mx.fast.scaled_dot_product_attention
    if hasattr(nn, "scaled_dot_product_attention"):
        return nn.scaled_dot_product_attention
    return None


def _time_kernel(fn: callable) -> float:
    for _ in range(WARMUP_ITERS):
        out = fn()
        mx.eval(out)
        mx.synchronize()

    times: list[float] = []
    for _ in range(BENCH_ITERS):
        start = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    return statistics.mean(times)


def benchmark_metal_flash_attn(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> float | None:
    fast_attn = _resolve_fast_attention()
    if fast_attn is None:
        return None

    q = mx.random.normal((batch, heads, seq_q, head_dim), dtype=mx.float16)
    k = mx.random.normal((batch, heads, seq_k, head_dim), dtype=mx.float16)
    v = mx.random.normal((batch, heads, seq_k, head_dim), dtype=mx.float16)
    mx.eval(q, k, v)

    scale = 1.0 / math.sqrt(head_dim)

    def fn(q=q, k=k, v=v, scale=scale):
        return fast_attn(q, k, v, scale=scale)

    return _time_kernel(fn)


def benchmark_mlx_attention(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> float:
    q = mx.random.normal((batch, heads, seq_q, head_dim), dtype=mx.float16)
    k = mx.random.normal((batch, heads, seq_k, head_dim), dtype=mx.float16)
    v = mx.random.normal((batch, heads, seq_k, head_dim), dtype=mx.float16)
    mx.eval(q, k, v)

    scale = 1.0 / math.sqrt(head_dim)

    def fn(q=q, k=k, v=v, scale=scale):
        return _reference_attention(q, k, v, scale)

    return _time_kernel(fn)


def benchmark_attention() -> None:
    configs = [
        # (batch, heads, seq_q, seq_k, head_dim)
        (1, 32, 1, 512, 128),
        (1, 32, 1, 4096, 128),
        (1, 32, 1, 32768, 128),
        (1, 32, 512, 512, 128),
        (32, 32, 1, 512, 128),
    ]

    print("Metal Flash Attention vs MLX reference")

    for batch, heads, seq_q, seq_k, head_dim in configs:
        t_metal = benchmark_metal_flash_attn(batch, heads, seq_q, seq_k, head_dim)
        t_mlx = benchmark_mlx_attention(batch, heads, seq_q, seq_k, head_dim)

        if t_metal is None:
            print(
                f"B={batch} H={heads} Sq={seq_q} Sk={seq_k} D={head_dim}: "
                f"Metal: unavailable, MLX: {t_mlx:.3f}ms"
            )
            continue

        flops = 4 * batch * heads * seq_q * seq_k * head_dim
        tflops = flops / (t_metal / 1000.0) / 1e12
        print(
            f"B={batch} H={heads} Sq={seq_q} Sk={seq_k} D={head_dim}: "
            f"{t_metal:.3f}ms ({tflops:.1f} TFLOPS), "
            f"MLX: {t_mlx:.3f}ms"
        )


if __name__ == "__main__":
    benchmark_attention()
