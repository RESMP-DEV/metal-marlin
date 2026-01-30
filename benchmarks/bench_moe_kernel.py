#!/usr/bin/env python3
"""Benchmark Metal MoE expert kernel variants.

Compares three implementations for GLM-4.7-Flash style shapes:
- python_loop: metal_marlin.kernels.moe_expert_gemm_fp4 (per-expert Python loop)
- moe_expert_gemm_fp4: per-token dispatch kernel
- moe_expert_gemm_fp4_grouped: token-grouped dispatch kernel

Measures latency (ms), throughput (tokens/sec), and an estimated memory
bandwidth utilization. Also prints guidance for profiling with Metal System
Trace (Xcode Instruments).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import (  # noqa: E402
    HAS_MPS,
    HAS_PYOBJC_METAL,
    HAS_TORCH,
    Metal,
    torch,
)
from metal_marlin.kernels import moe_expert_gemm_fp4  # noqa: E402
from metal_marlin.metal_dispatch import (  # noqa: E402
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
)
from metal_marlin.moe_dispatch import group_tokens_by_expert_full  # noqa: E402

MOE_TILE_M = 64
MOE_TILE_N = 64
MOE_THREADS = 128
FP4_PER_UINT = 8


@dataclass
class TimeStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass
class BenchResult:
    variant: str
    batch_size: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    tok_s: float
    memory_gb_s: float
    bandwidth_util: float | None


def _require_deps() -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for this benchmark")
    if not HAS_MPS:
        raise RuntimeError("MPS is required. Run on Apple Silicon with MPS enabled.")
    if not HAS_PYOBJC_METAL or Metal is None:
        raise RuntimeError(
            "PyObjC Metal is required. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )


def _sync_mps() -> None:
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(q * (len(sorted_vals) - 1))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def _time_fn(fn: Callable[[], None], warmup: int, iters: int, sync: Callable[[], None] | None) -> TimeStats:
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - start) * 1000.0)

    sorted_times = sorted(times)
    return TimeStats(
        mean_ms=statistics.mean(times),
        p50_ms=_percentile(sorted_times, 0.50),
        p95_ms=_percentile(sorted_times, 0.95),
        min_ms=min(sorted_times),
        max_ms=max(sorted_times),
    )


def _make_packed_weights(
    num_experts: int,
    hidden_dim: int,
    out_dim: int,
    group_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_dim % FP4_PER_UINT != 0:
        raise ValueError("hidden_dim must be divisible by 8 for FP4 packing")
    if hidden_dim % group_size != 0:
        raise ValueError("hidden_dim must be divisible by group_size")

    rng = np.random.default_rng(seed)
    k_packed = hidden_dim // FP4_PER_UINT
    scale_groups = hidden_dim // group_size

    packed = rng.integers(
        0, 2**32, size=(num_experts, k_packed, out_dim), dtype=np.uint32
    )
    scales = rng.uniform(0.01, 1.0, size=(num_experts, scale_groups, out_dim)).astype(
        np.float16
    )

    # Use int32 for MPS compatibility; bit patterns are preserved.
    packed_t = torch.from_numpy(packed.view(np.int32)).to(device="mps")
    scales_t = torch.from_numpy(scales).to(device="mps")
    return packed_t, scales_t


def _make_routing(
    batch_size: int,
    num_experts: int,
    top_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    expert_ids = torch.randint(
        0, num_experts, (batch_size, top_k), device="mps", dtype=torch.int32
    )
    expert_probs = torch.rand(
        (batch_size, top_k), device="mps", dtype=torch.float16
    )
    expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)
    return expert_ids, expert_probs


def _estimate_bytes(
    *,
    batch_size: int,
    hidden_dim: int,
    out_dim: int,
    num_experts: int,
    top_k: int,
    group_size: int,
    variant: str,
    expert_ids: torch.Tensor,
) -> int:
    bytes_a = batch_size * hidden_dim * 2
    bytes_c = batch_size * out_dim * 2
    weights_per_expert = (hidden_dim // FP4_PER_UINT) * out_dim * 4
    scales_per_expert = (hidden_dim // group_size) * out_dim * 2

    if variant == "moe_expert_gemm_fp4_grouped":
        unique_experts = int(torch.unique(expert_ids).numel())
        bytes_weights = unique_experts * weights_per_expert
        bytes_scales = unique_experts * scales_per_expert
    else:
        assignments = batch_size * top_k
        bytes_weights = assignments * weights_per_expert
        bytes_scales = assignments * scales_per_expert

    return bytes_a + bytes_c + bytes_weights + bytes_scales


def _make_params_buffer(
    device: object,
    batch_size: int,
    hidden_dim: int,
    out_dim: int,
    num_experts: int,
    top_k: int,
    group_size: int,
) -> object:
    params = np.array(
        [batch_size, hidden_dim, out_dim, num_experts, top_k, group_size, 0, 0],
        dtype=np.uint32,
    )
    return device.newBufferWithBytes_length_options_(
        params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
    )


def _build_kernel_buffers(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    output: torch.Tensor,
    params_buf: object,
) -> list[object]:
    device = lib.device
    act_buf = mps_tensor_to_metal_buffer(activations.half().contiguous(), device)
    w_buf = mps_tensor_to_metal_buffer(expert_weights.contiguous(), device)
    s_buf = mps_tensor_to_metal_buffer(expert_scales.half().contiguous(), device)
    ids_buf = mps_tensor_to_metal_buffer(expert_ids.int().contiguous(), device)
    probs_buf = mps_tensor_to_metal_buffer(expert_probs.half().contiguous(), device)
    out_buf = mps_tensor_to_metal_buffer(output, device)
    return [act_buf, w_buf, s_buf, ids_buf, probs_buf, out_buf, params_buf]


def _build_grouped_buffers(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_probs: torch.Tensor,
    output: torch.Tensor,
    params_buf: object,
) -> list[object]:
    device = lib.device
    dispatch_info = group_tokens_by_expert_full(expert_ids, expert_weights.shape[0])
    sorted_token_ids = dispatch_info.sorted_token_indices.to(torch.int32)
    expert_offsets = dispatch_info.expert_offsets.to(torch.int32)

    expert_probs_sorted = expert_probs[
        dispatch_info.sorted_token_indices, dispatch_info.sorted_expert_indices
    ].contiguous()

    act_buf = mps_tensor_to_metal_buffer(activations.half().contiguous(), device)
    w_buf = mps_tensor_to_metal_buffer(expert_weights.contiguous(), device)
    s_buf = mps_tensor_to_metal_buffer(expert_scales.half().contiguous(), device)
    token_buf = mps_tensor_to_metal_buffer(sorted_token_ids.contiguous(), device)
    offset_buf = mps_tensor_to_metal_buffer(expert_offsets.contiguous(), device)
    probs_buf = mps_tensor_to_metal_buffer(expert_probs_sorted.half().contiguous(), device)
    out_buf = mps_tensor_to_metal_buffer(output, device)
    return [act_buf, w_buf, s_buf, token_buf, offset_buf, probs_buf, out_buf, params_buf]


def _run_benchmarks(
    *,
    lib: MetalKernelLibrary,
    batch_sizes: list[int],
    hidden_dim: int,
    out_dim: int,
    num_experts: int,
    top_k: int,
    group_size: int,
    warmup: int,
    iters: int,
    seed: int,
    peak_bandwidth: float | None,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    device = torch.device("mps")

    expert_weights, expert_scales = _make_packed_weights(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        group_size=group_size,
        seed=seed,
    )

    for batch_size in batch_sizes:
        activations = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)
        expert_ids, expert_probs = _make_routing(batch_size, num_experts, top_k, seed)

        # Baseline: python loop over experts
        def _run_python_loop() -> None:
            _ = moe_expert_gemm_fp4(
                activations,
                expert_weights,
                expert_scales,
                expert_ids,
                expert_probs,
                group_size=group_size,
            )

        _sync_mps()
        stats = _time_fn(_run_python_loop, warmup, iters, _sync_mps)
        bytes_est = _estimate_bytes(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_experts=num_experts,
            top_k=top_k,
            group_size=group_size,
            variant="python_loop",
            expert_ids=expert_ids,
        )
        tok_s = batch_size / (stats.mean_ms / 1000.0)
        mem_gb_s = (bytes_est / (stats.mean_ms / 1000.0)) / 1e9
        bw_util = mem_gb_s / peak_bandwidth if peak_bandwidth else None
        results.append(
            BenchResult(
                variant="python_loop",
                batch_size=batch_size,
                mean_ms=stats.mean_ms,
                p50_ms=stats.p50_ms,
                p95_ms=stats.p95_ms,
                tok_s=tok_s,
                memory_gb_s=mem_gb_s,
                bandwidth_util=bw_util,
            )
        )

        # Per-token kernel
        params_buf = _make_params_buffer(
            lib.device, batch_size, hidden_dim, out_dim, num_experts, top_k, group_size
        )
        output = torch.empty((batch_size, out_dim), device=device, dtype=torch.float16)
        buffers = _build_kernel_buffers(
            lib,
            activations,
            expert_weights,
            expert_scales,
            expert_ids,
            expert_probs,
            output,
            params_buf,
        )
        grid_n = (out_dim + MOE_TILE_N - 1) // MOE_TILE_N
        grid_m = (batch_size + MOE_TILE_M - 1) // MOE_TILE_M

        def _run_per_token() -> None:
            dispatch_kernel(
                lib,
                function_name="moe_expert_gemm_fp4",
                grid=(grid_n, grid_m, 1),
                threadgroup=(MOE_THREADS, 1, 1),
                buffers=buffers,
                wait=True,
            )

        stats = _time_fn(_run_per_token, warmup, iters, None)
        bytes_est = _estimate_bytes(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_experts=num_experts,
            top_k=top_k,
            group_size=group_size,
            variant="moe_expert_gemm_fp4",
            expert_ids=expert_ids,
        )
        tok_s = batch_size / (stats.mean_ms / 1000.0)
        mem_gb_s = (bytes_est / (stats.mean_ms / 1000.0)) / 1e9
        bw_util = mem_gb_s / peak_bandwidth if peak_bandwidth else None
        results.append(
            BenchResult(
                variant="moe_expert_gemm_fp4",
                batch_size=batch_size,
                mean_ms=stats.mean_ms,
                p50_ms=stats.p50_ms,
                p95_ms=stats.p95_ms,
                tok_s=tok_s,
                memory_gb_s=mem_gb_s,
                bandwidth_util=bw_util,
            )
        )

        # Grouped kernel
        output_grouped = torch.zeros((batch_size, out_dim), device=device, dtype=torch.float16)
        grouped_buffers = _build_grouped_buffers(
            lib,
            activations,
            expert_weights,
            expert_scales,
            expert_ids,
            expert_probs,
            output_grouped,
            params_buf,
        )
        grid_g = (grid_n, num_experts, 1)

        def _run_grouped() -> None:
            output_grouped.zero_()
            dispatch_kernel(
                lib,
                function_name="moe_expert_gemm_fp4_grouped",
                grid=grid_g,
                threadgroup=(MOE_THREADS, 1, 1),
                buffers=grouped_buffers,
                wait=True,
            )

        stats = _time_fn(_run_grouped, warmup, iters, None)
        bytes_est = _estimate_bytes(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_experts=num_experts,
            top_k=top_k,
            group_size=group_size,
            variant="moe_expert_gemm_fp4_grouped",
            expert_ids=expert_ids,
        )
        tok_s = batch_size / (stats.mean_ms / 1000.0)
        mem_gb_s = (bytes_est / (stats.mean_ms / 1000.0)) / 1e9
        bw_util = mem_gb_s / peak_bandwidth if peak_bandwidth else None
        results.append(
            BenchResult(
                variant="moe_expert_gemm_fp4_grouped",
                batch_size=batch_size,
                mean_ms=stats.mean_ms,
                p50_ms=stats.p50_ms,
                p95_ms=stats.p95_ms,
                tok_s=tok_s,
                memory_gb_s=mem_gb_s,
                bandwidth_util=bw_util,
            )
        )

    return results


def _print_results(results: list[BenchResult], peak_bandwidth: float | None) -> None:
    if not results:
        return

    results_by_batch: dict[int, list[BenchResult]] = {}
    for res in results:
        results_by_batch.setdefault(res.batch_size, []).append(res)

    print("Metal MoE Kernel Benchmark (GLM-4.7-Flash configs)")
    if peak_bandwidth:
        print(f"  Peak bandwidth: {peak_bandwidth:.1f} GB/s")
    print("  Memory bandwidth is estimated; validate with Metal System Trace.")
    print("")

    for batch_size in sorted(results_by_batch):
        group = results_by_batch[batch_size]
        print(f"Batch size {batch_size}:")
        for res in group:
            bw = f"{res.memory_gb_s:.1f} GB/s"
            util = f"{res.bandwidth_util * 100.0:.1f}%" if res.bandwidth_util else "n/a"
            print(
                f"  {res.variant:28s} "
                f"mean {res.mean_ms:7.3f} ms  "
                f"p50 {res.p50_ms:7.3f} ms  "
                f"p95 {res.p95_ms:7.3f} ms  "
                f"{res.tok_s:8.1f} tok/s  "
                f"bw {bw:>12s}  util {util}"
            )
        best = min(group, key=lambda r: r.mean_ms)
        print(f"  -> recommend: {best.variant} (lowest mean latency)")
        print("")

    print("Metal System Trace tips:")
    print("  - Use Xcode Instruments > Metal System Trace")
    print("  - Filter kernels by: moe_expert_gemm_fp4, moe_expert_gemm_fp4_grouped")
    print("  - Inspect GPU utilization, memory read/write throughput, and occupancy")
    print("  - Look for cache misses or low occupancy on small batch sizes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Metal MoE expert kernels")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,128",
        help="Comma-separated batch sizes (tokens)",
    )
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        default=None,
        help="Peak device bandwidth in GB/s (for utilization estimate)",
    )
    args = parser.parse_args()

    _require_deps()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    lib = MetalKernelLibrary()
    src_path = _ROOT / "src" / "moe_expert_gemm.metal"
    lib.compile_source("moe_expert_gemm", src_path.read_text())

    results = _run_benchmarks(
        lib=lib,
        batch_sizes=batch_sizes,
        hidden_dim=args.hidden_size,
        out_dim=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        group_size=args.group_size,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        peak_bandwidth=args.peak_bandwidth,
    )

    _print_results(results, args.peak_bandwidth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
