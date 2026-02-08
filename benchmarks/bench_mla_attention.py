#!/usr/bin/env python3
"""Benchmark MLA attention kernel vs PyTorch SDPA on Apple Silicon.

This benchmark times the attention kernel used by MLA (standard SDPA) using:
- Metal fused attention (MPSGraph/Flash Attention V2 via fused_attention)
- PyTorch scaled_dot_product_attention fallback

Example:
  cd contrib/metal_marlin && uv run python benchmarks/bench_mla_attention.py --seq-len 2048 --iters 50
"""

from __future__ import annotations

import argparse
import time

import torch

from metal_marlin.fused_attention_mps import fused_attention


import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

def _synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _time_fn(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _synchronize()
    return (time.perf_counter() - start) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLA attention kernels")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--causal", dest="causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("MPS is not available. Run this benchmark on Apple Silicon with MPS.")
        return 1

    device = torch.device("mps")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    q = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / (args.head_dim**0.5)

    def _run_fused() -> None:
        fused_attention(q, k, v, mask=None, scale=scale, causal=args.causal)

    def _run_torch() -> None:
        torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=args.causal,
            scale=scale,
        )

    fused_sec = _time_fn(_run_fused, args.iters, args.warmup)
    torch_sec = _time_fn(_run_torch, args.iters, args.warmup)

    fused_ms = fused_sec * 1000.0
    torch_ms = torch_sec * 1000.0
    speedup = torch_sec / fused_sec if fused_sec > 0 else float("inf")

    print("MLA Attention Benchmark")
    print(f"  batch={args.batch} seq_len={args.seq_len} heads={args.heads} head_dim={args.head_dim} dtype={args.dtype}")
    print(f"  fused_attention: {fused_ms:.3f} ms")
    print(f"  torch sdpa    : {torch_ms:.3f} ms")
    print(f"  speedup       : {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
