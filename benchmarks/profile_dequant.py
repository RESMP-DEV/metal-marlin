#!/usr/bin/env python3
"""Profile quantized weight dequantization overhead on Apple GPUs.

Measures:
- Kernel time for standalone dequantization (FP4 + INT4)
- Effective read bandwidth (bits_read / time)
- Total traffic (read + write) vs theoretical bandwidth
- Memory traffic model for fused vs upfront dequantization

Usage:
  uv run python benchmarks/profile_dequant.py --K 4096 --N 4096 --group-size 128
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(_ROOT))

from metal_marlin.metal_dispatch import (  # noqa: E402
    HAS_METAL,
    HAS_MPS,
    MetalKernelLibrary,
    dispatch_kernel,
    get_shader_source,
    mps_tensor_to_metal_buffer,
)

try:
    from metal_marlin.kernels import HAS_METAL as HAS_METAL_KERNELS  # noqa: E402
    from metal_marlin.kernels import HAS_MPS as HAS_MPS_KERNELS  # noqa: E402
    from metal_marlin.kernels import (
        marlin_gemm_fp4,  # noqa: F401, E402
        marlin_gemm_int4,  # noqa: F401, E402
        pack_fp4_weights,  # noqa: E402
    )
    from metal_marlin.metal_marlin import pack_u4_weights  # noqa: E402
except Exception:  # pragma: no cover - optional imports
    HAS_METAL_KERNELS = False
    HAS_MPS_KERNELS = False
    pack_u4_weights = None
    pack_fp4_weights = None

if HAS_METAL:
    import Metal  # type: ignore


@dataclass
class DequantResult:
    name: str
    mean_ms: float
    read_gb_s: float
    total_gb_s: float
    bits_read: int
    bytes_read: int
    bytes_written: int
    timing_source: str


@dataclass
class TrafficBreakdown:
    weights_bytes: int
    scales_bytes: int
    zeros_bytes: int
    output_bytes: int

    @property
    def read_bytes(self) -> int:
        return self.weights_bytes + self.scales_bytes + self.zeros_bytes

    @property
    def write_bytes(self) -> int:
        return self.output_bytes

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


class DequantProfiler:
    def __init__(self, use_gpu_timestamps: bool) -> None:
        if not (HAS_METAL and HAS_MPS):
            raise RuntimeError(
                "Metal dequant profiling requires PyObjC Metal and PyTorch MPS.\n"
                "Install with: pip install pyobjc-framework-Metal torch"
            )

        self.lib = MetalKernelLibrary()
        source = get_shader_source("dequant")
        self.lib.compile_source("dequant", source)
        self.device = self.lib.device
        self.use_gpu_timestamps = use_gpu_timestamps

    def _dispatch_timed(
        self,
        function_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[object],
        warmup: int,
        iterations: int,
    ) -> tuple[float, str]:
        pipeline = self.lib.get_pipeline(function_name, library_name="dequant")

        for _ in range(warmup):
            dispatch_kernel(self.lib, function_name, grid, threadgroup, buffers, wait=True)

        if self.use_gpu_timestamps:
            gpu_times: list[float] = []
            for _ in range(iterations):
                cmd_buf = self.lib.command_queue.commandBuffer()
                encoder = cmd_buf.computeCommandEncoder()
                encoder.setComputePipelineState_(pipeline)
                for i, buf in enumerate(buffers):
                    encoder.setBuffer_offset_atIndex_(buf, 0, i)

                grid_size = Metal.MTLSizeMake(*grid)
                tg_size = Metal.MTLSizeMake(*threadgroup)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
                encoder.endEncoding()
                cmd_buf.commit()
                cmd_buf.waitUntilCompleted()

                gpu_start = cmd_buf.GPUStartTime()
                gpu_end = cmd_buf.GPUEndTime()
                if gpu_end > gpu_start:
                    gpu_times.append(gpu_end - gpu_start)

            if gpu_times:
                return statistics.mean(gpu_times), "metal_gpu_time"

        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            dispatch_kernel(self.lib, function_name, grid, threadgroup, buffers, wait=True)
            times.append(time.perf_counter() - start)

        return statistics.mean(times), "wall_time"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _grid_for_matrix(n: int, k_blocks: int, tg_x: int = 16, tg_y: int = 16) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    grid = (_ceil_div(n, tg_x), _ceil_div(k_blocks, tg_y), 1)
    threadgroup = (tg_x, tg_y, 1)
    return grid, threadgroup


def _random_fp4_inputs(
    K: int, N: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k_blocks = _ceil_div(K, 8)
    packed = np.random.randint(0, 2**32 - 1, size=(k_blocks, N), dtype=np.uint32)
    scales = np.random.uniform(0.01, 1.0, size=(K // group_size, N)).astype(np.float16)
    packed_t = torch.from_numpy(packed).to("mps")
    scales_t = torch.from_numpy(scales).to("mps")
    return packed_t, scales_t


def _random_int4_inputs(
    K: int, N: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_blocks = _ceil_div(K, 8)
    packed = np.random.randint(0, 2**32 - 1, size=(k_blocks, N), dtype=np.uint32)
    scales = np.random.uniform(0.01, 1.0, size=(K // group_size, N)).astype(np.float16)
    zeros = np.random.uniform(0.0, 15.0, size=(K // group_size, N)).astype(np.float16)
    packed_t = torch.from_numpy(packed).to("mps")
    scales_t = torch.from_numpy(scales).to("mps")
    zeros_t = torch.from_numpy(zeros).to("mps")
    return packed_t, scales_t, zeros_t


def _maybe_pack_fp4(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if pack_fp4_weights is None:
        return _random_fp4_inputs(weight.shape[1], weight.shape[0], group_size)
    return pack_fp4_weights(weight, group_size=group_size)


def _maybe_pack_int4(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pack_u4_weights is None:
        return _random_int4_inputs(weight.shape[1], weight.shape[0], group_size)
    return pack_u4_weights(weight, group_size=group_size)


def _traffic_breakdown(K: int, N: int, group_size: int, bits: int, has_zeros: bool) -> TrafficBreakdown:
    weights_bytes = (K * N * bits) // 8
    scales_bytes = (K // group_size) * N * 2
    zeros_bytes = scales_bytes if has_zeros else 0
    output_bytes = K * N * 2
    return TrafficBreakdown(
        weights_bytes=weights_bytes,
        scales_bytes=scales_bytes,
        zeros_bytes=zeros_bytes,
        output_bytes=output_bytes,
    )


def _format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _print_strategy_analysis(K: int, N: int, M: int, group_size: int) -> None:
    fused_available = HAS_METAL_KERNELS and HAS_MPS_KERNELS
    print("\nDequant Strategy Analysis")
    print("-" * 72)
    if fused_available:
        print("Fused dequant-GEMM kernels: AVAILABLE (marlin_gemm_fp4/int4)")
    else:
        print("Fused dequant-GEMM kernels: NOT AVAILABLE (Metal/MPS missing)")

    try:
        import inspect

        from metal_marlin.inference import decode as decode_module

        src = inspect.getsource(decode_module.quantized_linear_torch)
        if "dequant" in src and "@" in src:
            print("Reference decode path: dequantizes full weights then matmul (upfront)")
    except Exception:
        print("Reference decode path: unable to inspect")

    try:
        import inspect

        from metal_marlin.inference import pipeline as pipeline_module

        src = inspect.getsource(pipeline_module.MetalMarlinModel.__init__)
        if "_dequant_cache" in src:
            print("Pipeline path: caches dequantized weights (first call upfront, then reuse)")
    except Exception:
        print("Pipeline path: unable to inspect")

    traffic = _traffic_breakdown(K, N, group_size, bits=4, has_zeros=False)
    scale_overhead = traffic.scales_bytes / max(1, traffic.weights_bytes)
    print(f"Scale metadata overhead (FP4): {scale_overhead:.2f}x weight bytes")

    upfront_total = traffic.total_bytes + traffic.output_bytes
    fused_ideal = traffic.read_bytes
    print(
        "Upfront per-forward traffic (dequant + GEMM read): "
        f"{_format_bytes(upfront_total)}"
    )
    print(
        "Fused ideal traffic (no materialized weights): "
        f"{_format_bytes(fused_ideal)}"
    )


def _run_dequant_fp4(
    profiler: DequantProfiler,
    K: int,
    N: int,
    group_size: int,
    warmup: int,
    iterations: int,
) -> DequantResult:
    weight = torch.randn(N, K, dtype=torch.float16, device="mps")
    packed, scales = _maybe_pack_fp4(weight, group_size)

    output = torch.empty((K, N), dtype=torch.float16, device="mps")

    k_blocks = _ceil_div(K, 8)
    grid, threadgroup = _grid_for_matrix(N, k_blocks)

    packed_buf = mps_tensor_to_metal_buffer(packed.contiguous(), profiler.device)
    scales_buf = mps_tensor_to_metal_buffer(scales.contiguous(), profiler.device)
    output_buf = mps_tensor_to_metal_buffer(output, profiler.device)

    k_param = np.array([K], dtype=np.uint32)
    n_param = np.array([N], dtype=np.uint32)
    gs_param = np.array([group_size], dtype=np.uint32)

    k_buf = profiler.device.newBufferWithBytes_length_options_(
        k_param.tobytes(), k_param.nbytes, Metal.MTLResourceStorageModeShared
    )
    n_buf = profiler.device.newBufferWithBytes_length_options_(
        n_param.tobytes(), n_param.nbytes, Metal.MTLResourceStorageModeShared
    )
    gs_buf = profiler.device.newBufferWithBytes_length_options_(
        gs_param.tobytes(), gs_param.nbytes, Metal.MTLResourceStorageModeShared
    )

    buffers: list[Any] = [packed_buf, scales_buf, output_buf, k_buf, n_buf, gs_buf]

    mean_s, source = profiler._dispatch_timed(
        "dequant_fp4_bulk", grid, threadgroup, buffers, warmup, iterations
    )

    traffic = _traffic_breakdown(K, N, group_size, bits=4, has_zeros=False)
    read_gb_s = (traffic.read_bytes / mean_s) / 1e9
    total_gb_s = (traffic.total_bytes / mean_s) / 1e9
    return DequantResult(
        name="FP4",
        mean_ms=mean_s * 1000.0,
        read_gb_s=read_gb_s,
        total_gb_s=total_gb_s,
        bits_read=traffic.read_bytes * 8,
        bytes_read=traffic.read_bytes,
        bytes_written=traffic.write_bytes,
        timing_source=source,
    )


def _run_dequant_int4(
    profiler: DequantProfiler,
    K: int,
    N: int,
    group_size: int,
    warmup: int,
    iterations: int,
) -> DequantResult:
    weight = torch.randn(N, K, dtype=torch.float16, device="mps")
    packed, scales, zeros = _maybe_pack_int4(weight, group_size)

    output = torch.empty((K, N), dtype=torch.float16, device="mps")

    k_blocks = _ceil_div(K, 8)
    grid, threadgroup = _grid_for_matrix(N, k_blocks)

    packed_buf = mps_tensor_to_metal_buffer(packed.contiguous(), profiler.device)
    scales_buf = mps_tensor_to_metal_buffer(scales.contiguous(), profiler.device)
    zeros_buf = mps_tensor_to_metal_buffer(zeros.contiguous(), profiler.device)
    output_buf = mps_tensor_to_metal_buffer(output, profiler.device)

    k_param = np.array([K], dtype=np.uint32)
    n_param = np.array([N], dtype=np.uint32)
    gs_param = np.array([group_size], dtype=np.uint32)

    k_buf = profiler.device.newBufferWithBytes_length_options_(
        k_param.tobytes(), k_param.nbytes, Metal.MTLResourceStorageModeShared
    )
    n_buf = profiler.device.newBufferWithBytes_length_options_(
        n_param.tobytes(), n_param.nbytes, Metal.MTLResourceStorageModeShared
    )
    gs_buf = profiler.device.newBufferWithBytes_length_options_(
        gs_param.tobytes(), gs_param.nbytes, Metal.MTLResourceStorageModeShared
    )

    buffers: list[Any] = [packed_buf, scales_buf, zeros_buf, output_buf, k_buf, n_buf, gs_buf]

    mean_s, source = profiler._dispatch_timed(
        "dequant_int4_bulk", grid, threadgroup, buffers, warmup, iterations
    )

    traffic = _traffic_breakdown(K, N, group_size, bits=4, has_zeros=True)
    read_gb_s = (traffic.read_bytes / mean_s) / 1e9
    total_gb_s = (traffic.total_bytes / mean_s) / 1e9
    return DequantResult(
        name="INT4",
        mean_ms=mean_s * 1000.0,
        read_gb_s=read_gb_s,
        total_gb_s=total_gb_s,
        bits_read=traffic.read_bytes * 8,
        bytes_read=traffic.read_bytes,
        bytes_written=traffic.write_bytes,
        timing_source=source,
    )


def _print_result(result: DequantResult, theoretical_gb_s: float) -> None:
    ratio = 0.0
    if theoretical_gb_s > 0:
        ratio = (result.read_gb_s / theoretical_gb_s) * 100.0

    print(f"{result.name} dequant kernel")
    print(f"  mean time: {result.mean_ms:.3f} ms ({result.timing_source})")
    print(f"  read bandwidth: {result.read_gb_s:.2f} GB/s")
    print(f"  total bandwidth (read+write): {result.total_gb_s:.2f} GB/s")
    print(f"  bits read: {result.bits_read / 1e9:.3f} Gb")
    print(f"  bytes read: {_format_bytes(result.bytes_read)}")
    print(f"  bytes written: {_format_bytes(result.bytes_written)}")
    if theoretical_gb_s > 0:
        print(f"  % of theoretical: {ratio:.1f}% (theoretical {theoretical_gb_s:.1f} GB/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FP4/INT4 dequantization kernels")
    parser.add_argument("--K", type=int, default=4096, help="Reduction dimension (rows)")
    parser.add_argument("--N", type=int, default=4096, help="Output dimension (columns)")
    parser.add_argument("--M", type=int, default=1, help="Batch size for strategy analysis")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument(
        "--formats",
        type=str,
        default="fp4,int4",
        help="Comma-separated formats to profile (fp4,int4)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations")
    parser.add_argument(
        "--theoretical-gb-s",
        type=float,
        default=0.0,
        help="Theoretical memory bandwidth for comparison",
    )
    parser.add_argument(
        "--gpu-timestamps",
        action="store_true",
        help="Use Metal GPU timestamps for kernel timing",
    )
    args = parser.parse_args()

    if not (HAS_METAL and HAS_MPS):
        raise RuntimeError(
            "Metal profiling requires PyObjC Metal and PyTorch MPS.\n"
            "Install with: pip install pyobjc-framework-Metal torch"
        )

    K = int(args.K)
    N = int(args.N)
    M = int(args.M)
    group_size = int(args.group_size)

    if K % 8 != 0:
        raise ValueError(f"K={K} must be divisible by 8")
    if K % group_size != 0:
        raise ValueError(f"K={K} must be divisible by group_size={group_size}")

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    profiler = DequantProfiler(use_gpu_timestamps=args.gpu_timestamps)

    print("Quantized Dequant Profiling")
    print("-" * 72)
    print(f"K={K}, N={N}, group_size={group_size}, warmup={args.warmup}, iters={args.iterations}")

    results: list[DequantResult] = []

    if "fp4" in formats:
        results.append(_run_dequant_fp4(profiler, K, N, group_size, args.warmup, args.iterations))

    if "int4" in formats:
        results.append(_run_dequant_int4(profiler, K, N, group_size, args.warmup, args.iterations))

    print()
    for result in results:
        _print_result(result, args.theoretical_gb_s)

    _print_strategy_analysis(K, N, M, group_size)

    print("\nMemory Traffic Model")
    print("-" * 72)
    for result in results:
        has_zeros = result.name == "INT4"
        traffic = _traffic_breakdown(K, N, group_size, bits=4, has_zeros=has_zeros)
        fused_ideal = traffic.read_bytes
        upfront = traffic.total_bytes + traffic.output_bytes
        print(f"{result.name} traffic")
        print(f"  weights: {_format_bytes(traffic.weights_bytes)}")
        print(f"  scales:  {_format_bytes(traffic.scales_bytes)}")
        if has_zeros:
            print(f"  zeros:   {_format_bytes(traffic.zeros_bytes)}")
        print(f"  output:  {_format_bytes(traffic.output_bytes)}")
        print(f"  total read:  {_format_bytes(traffic.read_bytes)}")
        print(f"  total write: {_format_bytes(traffic.write_bytes)}")
        print(f"  total traffic (dequant only): {_format_bytes(traffic.total_bytes)}")
        print(f"  fused ideal (no weight materialization): {_format_bytes(fused_ideal)}")
        print(f"  upfront + GEMM read: {_format_bytes(upfront)}")


if __name__ == "__main__":
    main()
