#!/usr/bin/env python3
"""
Profile attention layer memory traffic and kernel launches on Apple Silicon.

This benchmark compares:
  1) metal_marlin Flash Attention V2
  2) PyTorch SDPA (MPS)
  3) MPSGraph fused SDPA (when available)

It reports:
  - Estimated materialization of Q*K^T and softmax intermediates
  - Kernel launch proxy counts
  - Memory traffic vs theoretical minimum
  - KV cache read pattern for decode (seq_q=1)

Usage:
  uv run python benchmarks/profile_attention.py
  uv run python benchmarks/profile_attention.py --seq-q 1 --seq-k 4096 --heads 32 --head-dim 128
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_MPS, HAS_MPSGRAPH, HAS_TORCH, torch  # noqa: E402
from metal_marlin.flash_attention_v2 import flash_attention_v2  # noqa: E402
from metal_marlin.fused_attention_mps import fused_scaled_dot_product_attention  # noqa: E402
from metal_marlin.profiling.gpu_counters import GPUProfiler  # noqa: E402
from metal_marlin.profiling.occupancy import detect_gpu  # noqa: E402

if HAS_TORCH and torch is not None:
    import torch.nn.functional as F  # noqa: E402


WARMUP_ITERS = 5
PROFILE_ITERS = 20


@dataclass(frozen=True)
class AttentionShape:
    batch: int
    heads_q: int
    heads_kv: int
    seq_q: int
    seq_k: int
    head_dim: int
    dtype: str = "float16"


@dataclass
class KernelEstimate:
    kernel_launches: int | None = None
    cpu_op_count_proxy: int | None = None
    note: str = ""


@dataclass
class MemoryEstimate:
    theoretical_min_bytes: int
    theoretical_min_bytes_per_head: int
    qk_bytes: int
    softmax_bytes: int
    output_bytes: int
    extra_alloc_bytes: int
    qk_materialized: bool
    softmax_materialized: bool
    counter_mode: str
    measured_total_bytes: float | None = None
    measured_total_bytes_per_head: float | None = None
    traffic_ratio: float | None = None


@dataclass
class ProfileResult:
    name: str
    shape: AttentionShape
    causal: bool
    mean_ms: float
    std_ms: float
    kernel_estimate: KernelEstimate
    memory_estimate: MemoryEstimate
    metadata: dict[str, Any] = field(default_factory=dict)


def _check_mps() -> None:
    if not (HAS_TORCH and torch is not None and HAS_MPS):
        raise RuntimeError("This script requires PyTorch with MPS backend on Apple Silicon.")


def _mps_sync() -> None:
    if HAS_TORCH and torch is not None and HAS_MPS:
        torch.mps.synchronize()


def _mps_memory_bytes() -> tuple[int, int]:
    if not (HAS_TORCH and torch is not None and HAS_MPS):
        return 0, 0
    current = 0
    driver = 0
    if hasattr(torch.mps, "current_allocated_memory"):
        current = int(torch.mps.current_allocated_memory())
    if hasattr(torch.mps, "driver_allocated_memory"):
        driver = int(torch.mps.driver_allocated_memory())
    return current, driver


def _bytes_per_element(dtype: torch.dtype) -> int:
    return int(torch.tensor([], dtype=dtype).element_size())


def _theoretical_bytes(shape: AttentionShape, dtype: torch.dtype) -> dict[str, int]:
    elem = _bytes_per_element(dtype)
    q = shape.batch * shape.heads_q * shape.seq_q * shape.head_dim * elem
    k = shape.batch * shape.heads_kv * shape.seq_k * shape.head_dim * elem
    v = shape.batch * shape.heads_kv * shape.seq_k * shape.head_dim * elem
    o = shape.batch * shape.heads_q * shape.seq_q * shape.head_dim * elem
    qk = shape.batch * shape.heads_q * shape.seq_q * shape.seq_k * elem
    softmax = qk
    return {
        "q": q,
        "k": k,
        "v": v,
        "o": o,
        "qk": qk,
        "softmax": softmax,
        "min_total": q + k + v + o,
    }


def _estimate_flops(shape: AttentionShape, causal: bool) -> float:
    full = 4.0 * shape.batch * shape.heads_q * shape.seq_q * shape.seq_k * shape.head_dim
    if not causal:
        return full
    # Triangular attention reduces QK and PV roughly by half.
    return full * 0.5


def _count_cpu_ops(fn: Callable[[], torch.Tensor]) -> int | None:
    if not (HAS_TORCH and torch is not None):
        return None
    try:
        from torch.profiler import ProfilerActivity, profile
import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

    except Exception:
        return None
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _ = fn()
        _mps_sync()
    return int(sum(item.count for item in prof.key_averages()))


def _time_fn(fn: Callable[[], torch.Tensor], iters: int) -> tuple[float, float]:
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = fn()
        _mps_sync()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def _profile_impl(
    name: str,
    shape: AttentionShape,
    dtype: torch.dtype,
    causal: bool,
    fn: Callable[[], torch.Tensor],
    kernel_launches: int | None,
    kernel_note: str,
) -> ProfileResult:
    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = fn()
        _mps_sync()

    mem_before, _ = _mps_memory_bytes()
    mean_ms, std_ms = _time_fn(fn, PROFILE_ITERS)
    mem_after, _ = _mps_memory_bytes()

    output = fn()
    _mps_sync()

    mem_after_final, _ = _mps_memory_bytes()
    output_bytes = int(output.numel() * output.element_size())
    extra_alloc = max(0, mem_after_final - mem_before - output_bytes)

    theory = _theoretical_bytes(shape, dtype)
    qk_bytes = theory["qk"]
    softmax_bytes = theory["softmax"]
    min_total = theory["min_total"]

    profiler = GPUProfiler()
    profiler.start_capture()
    start = time.perf_counter()
    _ = fn()
    _mps_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    counters = profiler.stop_capture(
        flops=_estimate_flops(shape, causal),
        bytes_moved=min_total,
    )

    counter_mode = "hardware" if profiler.counters_available else "estimate"
    measured_total = None
    measured_per_head = None
    traffic_ratio = None
    if counters.total_memory_bandwidth > 0:
        measured_total = counters.total_memory_bandwidth * 1e9 * (elapsed_ms / 1000.0)
        measured_per_head = measured_total / max(1, shape.heads_q)
        traffic_ratio = measured_total / min_total if min_total > 0 else None

    qk_materialized = extra_alloc >= 0.5 * qk_bytes
    softmax_materialized = extra_alloc >= 0.5 * (qk_bytes + softmax_bytes)

    mem_estimate = MemoryEstimate(
        theoretical_min_bytes=min_total,
        theoretical_min_bytes_per_head=min_total // max(1, shape.heads_q),
        qk_bytes=qk_bytes,
        softmax_bytes=softmax_bytes,
        output_bytes=output_bytes,
        extra_alloc_bytes=extra_alloc,
        qk_materialized=qk_materialized,
        softmax_materialized=softmax_materialized,
        counter_mode=counter_mode,
        measured_total_bytes=measured_total,
        measured_total_bytes_per_head=measured_per_head,
        traffic_ratio=traffic_ratio,
    )

    cpu_ops = _count_cpu_ops(fn)
    kernel_estimate = KernelEstimate(
        kernel_launches=kernel_launches,
        cpu_op_count_proxy=cpu_ops,
        note=kernel_note,
    )

    return ProfileResult(
        name=name,
        shape=shape,
        causal=causal,
        mean_ms=mean_ms,
        std_ms=std_ms,
        kernel_estimate=kernel_estimate,
        memory_estimate=mem_estimate,
    )


def _format_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if value < 1024:
        return f"{value:.0f} B"
    if value < 1024**2:
        return f"{value / 1024:.2f} KB"
    if value < 1024**3:
        return f"{value / 1024**2:.2f} MB"
    return f"{value / 1024**3:.2f} GB"


def _print_result(result: ProfileResult) -> None:
    shape = result.shape
    mem = result.memory_estimate
    kernel = result.kernel_estimate
    print("-" * 80)
    print(f"{result.name} (causal={result.causal})")
    print(
        f"B={shape.batch} Hq={shape.heads_q} Hkv={shape.heads_kv} "
        f"Sq={shape.seq_q} Sk={shape.seq_k} D={shape.head_dim} dtype={shape.dtype}"
    )
    print(f"time: {result.mean_ms:.3f} ms ± {result.std_ms:.3f} ms")
    print(
        "kernel launches: "
        f"{kernel.kernel_launches if kernel.kernel_launches is not None else 'n/a'} "
        f"(cpu op proxy={kernel.cpu_op_count_proxy if kernel.cpu_op_count_proxy is not None else 'n/a'})"
    )
    if kernel.note:
        print(f"kernel note: {kernel.note}")
    print(
        "theoretical min traffic: "
        f"{_format_bytes(mem.theoretical_min_bytes)} total, "
        f"{_format_bytes(mem.theoretical_min_bytes_per_head)} per head"
    )
    print(
        "measured traffic: "
        f"{_format_bytes(mem.measured_total_bytes)} total "
        f"({mem.counter_mode})"
    )
    if mem.traffic_ratio is not None:
        print(f"traffic ratio (measured/min): {mem.traffic_ratio:.2f}x")
    print(
        "materialization: "
        f"QK={'yes' if mem.qk_materialized else 'no'}, "
        f"softmax={'yes' if mem.softmax_materialized else 'no'}, "
        f"extra alloc={_format_bytes(mem.extra_alloc_bytes)}"
    )


def _build_impls(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
) -> list[tuple[str, Callable[[], torch.Tensor], int | None, str]]:
    impls: list[tuple[str, Callable[[], torch.Tensor], int | None, str]] = []

    def flash() -> torch.Tensor:
        return flash_attention_v2(q, k, v, scale=scale, causal=causal)

    impls.append(("flash_attention_v2", flash, 1, "single Metal kernel dispatch"))

    if HAS_TORCH and torch is not None:
        def sdpa() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale
            )

        impls.append(("torch_sdpa", sdpa, None, "kernel count unavailable; CPU op proxy reported"))

    if HAS_MPSGRAPH:
        def mpsgraph_sdpa() -> torch.Tensor:
            return fused_scaled_dot_product_attention(q, k, v, causal=causal, scale=scale)

        impls.append(("mpsgraph_sdpa", mpsgraph_sdpa, None, "graph execution; kernel count unknown"))

    return impls


def _profile_case(shape: AttentionShape, causal: bool) -> list[ProfileResult]:
    if not (HAS_TORCH and torch is not None):
        raise RuntimeError("PyTorch is required for attention profiling.")

    dtype = torch.float16 if shape.dtype == "float16" else torch.float32
    q = torch.randn(
        shape.batch,
        shape.heads_q,
        shape.seq_q,
        shape.head_dim,
        device="mps",
        dtype=dtype,
    )
    k = torch.randn(
        shape.batch,
        shape.heads_kv,
        shape.seq_k,
        shape.head_dim,
        device="mps",
        dtype=dtype,
    )
    v = torch.randn(
        shape.batch,
        shape.heads_kv,
        shape.seq_k,
        shape.head_dim,
        device="mps",
        dtype=dtype,
    )

    scale = 1.0 / math.sqrt(shape.head_dim)

    results: list[ProfileResult] = []
    for name, fn, kernel_launches, note in _build_impls(q, k, v, scale, causal):
        try:
            result = _profile_impl(name, shape, dtype, causal, fn, kernel_launches, note)
            results.append(result)
        except Exception as exc:
            print(f"{name}: failed ({exc})")

    return results


def _compare_causal(results: list[ProfileResult]) -> None:
    by_name: dict[str, dict[bool, ProfileResult]] = {}
    for res in results:
        by_name.setdefault(res.name, {})[res.causal] = res

    print("-" * 80)
    print("Causal vs Non-causal Comparison")
    for name, variants in by_name.items():
        if True in variants and False in variants:
            causal = variants[True]
            non = variants[False]
            time_ratio = causal.mean_ms / non.mean_ms if non.mean_ms > 0 else 0.0
            print(f"{name}: causal/non time ratio {time_ratio:.2f}x")


def _profile_kv_cache(results: list[ProfileResult]) -> None:
    print("-" * 80)
    print("KV Cache Read Pattern (decode)")
    for res in results:
        shape = res.shape
        if shape.seq_q != 1:
            continue
        mem = res.memory_estimate
        kv_min = (mem.theoretical_min_bytes - mem.output_bytes) / max(1, shape.heads_q)
        kv_measured = mem.measured_total_bytes_per_head
        if kv_measured is not None:
            ratio = kv_measured / kv_min if kv_min > 0 else None
            print(f"{res.name}: per-head KV traffic ratio {ratio:.2f}x")
        else:
            print(f"{res.name}: per-head KV traffic ratio n/a (no counters)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile attention memory traffic and kernel launches.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32, dest="heads_q")
    parser.add_argument("--heads-kv", type=int, default=None)
    parser.add_argument("--seq-q", type=int, default=None)
    parser.add_argument("--seq-k", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--json", type=str, default=None, help="Optional path to write JSON results.")
    return parser.parse_args()


def main() -> None:
    _check_mps()

    args = _parse_args()
    heads_kv = args.heads_kv if args.heads_kv is not None else args.heads_q

    if args.seq_q is not None and args.seq_k is not None:
        configs = [
            AttentionShape(
                batch=args.batch,
                heads_q=args.heads_q,
                heads_kv=heads_kv,
                seq_q=args.seq_q,
                seq_k=args.seq_k,
                head_dim=args.head_dim,
                dtype=args.dtype,
            )
        ]
    else:
        configs = [
            AttentionShape(
                batch=args.batch,
                heads_q=args.heads_q,
                heads_kv=heads_kv,
                seq_q=128,
                seq_k=128,
                head_dim=args.head_dim,
                dtype=args.dtype,
            ),
            AttentionShape(
                batch=args.batch,
                heads_q=args.heads_q,
                heads_kv=heads_kv,
                seq_q=1,
                seq_k=4096,
                head_dim=args.head_dim,
                dtype=args.dtype,
            ),
        ]

    gpu = detect_gpu()
    print("=" * 80)
    print("Attention Memory/Kernel Profiling")
    print("=" * 80)
    print(f"GPU: {gpu.name} (peak bandwidth {gpu.peak_bw_gbs:.1f} GB/s)")
    print(f"MPSGraph available: {HAS_MPSGRAPH}")

    all_results: list[ProfileResult] = []

    for shape in configs:
        results_non_causal = _profile_case(shape, causal=False)
        results_causal = _profile_case(shape, causal=True)
        all_results.extend(results_non_causal)
        all_results.extend(results_causal)

        for res in results_non_causal + results_causal:
            _print_result(res)

        _compare_causal(results_non_causal + results_causal)

        if shape.seq_q == 1:
            _profile_kv_cache(results_non_causal + results_causal)

    if args.json:
        output_path = Path(args.json)
        payload = [asdict(res) for res in all_results]
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON results to {output_path}")


if __name__ == "__main__":
    main()
