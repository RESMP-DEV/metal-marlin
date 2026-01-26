#!/usr/bin/env python3
"""
Microbenchmark for BF16 conversion overhead on Apple Silicon.

Measures conversion paths at multiple sizes (1K, 64K, 1M, 16M elements):
  1) BF16 -> FP16 -> FP32 (current path)
  2) BF16 -> FP32 direct (proposed)
  3) FP32 -> FP16 -> BF16 (current store path)
  4) FP32 -> BF16 direct (proposed)
  5) Bulk kernel: bf16_to_half_kernel vs direct conversion
  6) Fused load+convert (bf16_to_float_kernel) vs separate kernels

Reports:
  - GB/s (bytes moved / time)
  - ns/element
  - Estimated time breakdown: memory vs conversion (derived from copy bandwidth)

Usage:
  uv run python benchmarks/bench_bf16_conversion.py
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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

if HAS_METAL:
    import Metal  # type: ignore


SIZES = [1024, 65_536, 1_048_576, 16_777_216]


@dataclass
class BenchResult:
    name: str
    elements: int
    mean_s: float
    ns_per_elem: float
    gb_s: float
    bytes_per_elem: int
    baseline_gb_s: float
    mem_ns_per_elem: float
    conv_ns_per_elem: float
    timing_source: str


def _mps_sync() -> None:
    torch.mps.synchronize()


def _time_torch(
    fn: Callable[[], None],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    _mps_sync()

    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        _mps_sync()
        times.append(time.perf_counter() - start)

    return statistics.mean(times)


def _bytes_per_elem_copy(dtype: torch.dtype) -> int:
    return 2 * torch.tensor([], dtype=dtype).element_size()


def _estimate_breakdown(
    mean_s: float,
    elements: int,
    bytes_per_elem: int,
    baseline_gb_s: float,
) -> tuple[float, float]:
    if baseline_gb_s <= 0:
        return mean_s * 1e9 / elements, 0.0

    bytes_total = bytes_per_elem * elements
    mem_time_s = bytes_total / (baseline_gb_s * 1e9)
    mem_ns = mem_time_s * 1e9 / elements
    total_ns = mean_s * 1e9 / elements
    conv_ns = max(0.0, total_ns - mem_ns)
    return mem_ns, conv_ns


class Bf16KernelRunner:
    def __init__(self, use_gpu_timestamps: bool = True) -> None:
        if not (HAS_METAL and HAS_MPS):
            raise RuntimeError("Metal kernels require PyObjC Metal + MPS")

        self.lib = MetalKernelLibrary()
        source = get_shader_source("bf16_compat")
        self.lib.compile_source("bf16_compat", source)
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
        pipeline = self.lib.get_pipeline(function_name, library_name="bf16_compat")

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

    def _dispatch_once(
        self,
        function_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[object],
    ) -> None:
        dispatch_kernel(self.lib, function_name, grid, threadgroup, buffers, wait=True)

    def prepare_bf16_to_half(
        self, inp: torch.Tensor, out: torch.Tensor
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], list[object]]:
        num_elements = inp.numel()
        num_threads = math.ceil(num_elements / 4)
        tg_size = 256
        grid = (math.ceil(num_threads / tg_size), 1, 1)
        threadgroup = (tg_size, 1, 1)

        inp_buf = mps_tensor_to_metal_buffer(inp.contiguous(), self.device)
        out_buf = mps_tensor_to_metal_buffer(out, self.device)
        params = np.array([num_elements], dtype=np.uint32)
        params_buf = self.device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        return grid, threadgroup, [inp_buf, out_buf, params_buf]

    def bf16_to_half(self, inp: torch.Tensor, out: torch.Tensor, warmup: int, iterations: int) -> float:
        num_elements = inp.numel()
        num_threads = math.ceil(num_elements / 4)
        tg_size = 256
        grid = (math.ceil(num_threads / tg_size), 1, 1)
        threadgroup = (tg_size, 1, 1)

        inp_buf = mps_tensor_to_metal_buffer(inp.contiguous(), self.device)
        out_buf = mps_tensor_to_metal_buffer(out, self.device)
        params = np.array([num_elements], dtype=np.uint32)
        params_buf = self.device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        mean_s, _ = self._dispatch_timed(
            "bf16_to_half_kernel",
            grid,
            threadgroup,
            [inp_buf, out_buf, params_buf],
            warmup,
            iterations,
        )
        return mean_s

    def bf16_to_half_once(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        grid, threadgroup, buffers = self.prepare_bf16_to_half(inp, out)
        self._dispatch_once("bf16_to_half_kernel", grid, threadgroup, buffers)

    def bf16_to_float(self, inp: torch.Tensor, out: torch.Tensor, warmup: int, iterations: int) -> float:
        num_elements = inp.numel()
        num_threads = math.ceil(num_elements / 4)
        tg_size = 256
        grid = (math.ceil(num_threads / tg_size), 1, 1)
        threadgroup = (tg_size, 1, 1)

        inp_buf = mps_tensor_to_metal_buffer(inp.contiguous(), self.device)
        out_buf = mps_tensor_to_metal_buffer(out, self.device)
        params = np.array([num_elements], dtype=np.uint32)
        params_buf = self.device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        mean_s, _ = self._dispatch_timed(
            "bf16_to_float_kernel",
            grid,
            threadgroup,
            [inp_buf, out_buf, params_buf],
            warmup,
            iterations,
        )
        return mean_s

    def bf16_to_float_once(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        num_elements = inp.numel()
        num_threads = math.ceil(num_elements / 4)
        tg_size = 256
        grid = (math.ceil(num_threads / tg_size), 1, 1)
        threadgroup = (tg_size, 1, 1)

        inp_buf = mps_tensor_to_metal_buffer(inp.contiguous(), self.device)
        out_buf = mps_tensor_to_metal_buffer(out, self.device)
        params = np.array([num_elements], dtype=np.uint32)
        params_buf = self.device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        self._dispatch_once(
            "bf16_to_float_kernel",
            grid,
            threadgroup,
            [inp_buf, out_buf, params_buf],
        )


def _bench_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    def fn() -> None:
        dst.copy_(src)

    return _time_torch(fn, warmup, iterations)


def _bench_op(
    name: str,
    fn: Callable[[], None],
    elements: int,
    bytes_per_elem: int,
    baseline_gb_s: float,
    mean_s: float | None = None,
    timing_source: str = "wall_time",
) -> BenchResult:
    if mean_s is None:
        mean_s = _time_torch(fn, warmup=args.warmup, iterations=args.iterations)

    ns_per_elem = mean_s * 1e9 / elements
    gb_s = (bytes_per_elem * elements) / mean_s / 1e9 if mean_s > 0 else 0.0
    mem_ns, conv_ns = _estimate_breakdown(mean_s, elements, bytes_per_elem, baseline_gb_s)

    return BenchResult(
        name=name,
        elements=elements,
        mean_s=mean_s,
        ns_per_elem=ns_per_elem,
        gb_s=gb_s,
        bytes_per_elem=bytes_per_elem,
        baseline_gb_s=baseline_gb_s,
        mem_ns_per_elem=mem_ns,
        conv_ns_per_elem=conv_ns,
        timing_source=timing_source,
    )


def _print_results(size: int, results: list[BenchResult]) -> None:
    print(f"\nSize: {size:,} elements")
    header = (
        f"{'Op':<36} {'ns/elem':>10} {'GB/s':>8} {'mem ns':>9} {'conv ns':>9} {'util':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        util = r.gb_s / r.baseline_gb_s if r.baseline_gb_s > 0 else 0.0
        print(
            f"{r.name:<36} {r.ns_per_elem:>10.2f} {r.gb_s:>8.1f} "
            f"{r.mem_ns_per_elem:>9.2f} {r.conv_ns_per_elem:>9.2f} {util:>6.2f}"
        )


def _ensure_bf16_supported() -> None:
    try:
        _ = torch.empty(1, device="mps", dtype=torch.bfloat16)
    except Exception as exc:
        raise RuntimeError("BF16 not supported on this MPS device") from exc


def _bytes_per_elem_for_path(path: str) -> int:
    if path in {"bf16_fp16_fp32", "fp32_fp16_bf16", "bf16_fp16_kernel_fp32"}:
        return 10
    if path in {"bf16_fp32", "fp32_bf16", "bf16_to_float_kernel"}:
        return 6
    if path in {"bf16_to_half_kernel", "bf16_to_half"}:
        return 4
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BF16 conversion microbenchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    parser.add_argument("--sizes", type=int, nargs="*", default=SIZES, help="Element counts")
    parser.add_argument(
        "--no-metal",
        action="store_true",
        help="Skip Metal kernel timings even if available",
    )
    parser.add_argument(
        "--no-metal-timestamps",
        action="store_true",
        help="Disable Metal GPU timestamps even if available",
    )
    args = parser.parse_args()

    if not HAS_MPS:
        raise RuntimeError("MPS backend is required for this benchmark")

    _ensure_bf16_supported()

    print("BF16 Conversion Microbenchmark")
    print(f"  MPS: {HAS_MPS}")
    print(f"  Metal kernels: {HAS_METAL and not args.no_metal}")
    use_metal_timestamps = HAS_METAL and not args.no_metal_timestamps
    print(f"  Metal GPU timestamps: {use_metal_timestamps}")

    kernel_runner: Bf16KernelRunner | None = None
    if HAS_METAL and not args.no_metal:
        kernel_runner = Bf16KernelRunner(use_gpu_timestamps=use_metal_timestamps)

    for size in args.sizes:
        bf16 = torch.randn(size, device="mps", dtype=torch.bfloat16)
        fp32 = torch.randn(size, device="mps", dtype=torch.float32)
        fp16_tmp = torch.empty(size, device="mps", dtype=torch.float16)
        fp32_out = torch.empty(size, device="mps", dtype=torch.float32)
        bf16_out = torch.empty(size, device="mps", dtype=torch.bfloat16)

        # Baseline copy bandwidth per dtype (for breakdown estimation)
        bf16_copy_time = _bench_copy(bf16, bf16_out, args.warmup, args.iterations)
        bf16_copy_bytes = _bytes_per_elem_copy(torch.bfloat16)
        bf16_copy_gb_s = (bf16_copy_bytes * size) / bf16_copy_time / 1e9

        fp32_copy_time = _bench_copy(fp32, fp32_out, args.warmup, args.iterations)
        fp32_copy_bytes = _bytes_per_elem_copy(torch.float32)
        fp32_copy_gb_s = (fp32_copy_bytes * size) / fp32_copy_time / 1e9

        results: list[BenchResult] = []

        # 1) BF16 -> FP16 -> FP32 (current)
        def bf16_fp16_fp32() -> None:
            fp16_tmp.copy_(bf16)
            fp32_out.copy_(fp16_tmp)

        results.append(
            _bench_op(
                "BF16->FP16->FP32",
                bf16_fp16_fp32,
                size,
                _bytes_per_elem_for_path("bf16_fp16_fp32"),
                bf16_copy_gb_s,
            )
        )

        # 2) BF16 -> FP32 direct
        def bf16_fp32() -> None:
            fp32_out.copy_(bf16)

        results.append(
            _bench_op(
                "BF16->FP32 direct",
                bf16_fp32,
                size,
                _bytes_per_elem_for_path("bf16_fp32"),
                bf16_copy_gb_s,
            )
        )

        # 3) FP32 -> FP16 -> BF16 (current)
        def fp32_fp16_bf16() -> None:
            fp16_tmp.copy_(fp32)
            bf16_out.copy_(fp16_tmp)

        results.append(
            _bench_op(
                "FP32->FP16->BF16",
                fp32_fp16_bf16,
                size,
                _bytes_per_elem_for_path("fp32_fp16_bf16"),
                fp32_copy_gb_s,
            )
        )

        # 4) FP32 -> BF16 direct
        def fp32_bf16() -> None:
            bf16_out.copy_(fp32)

        results.append(
            _bench_op(
                "FP32->BF16 direct",
                fp32_bf16,
                size,
                _bytes_per_elem_for_path("fp32_bf16"),
                fp32_copy_gb_s,
            )
        )

        # 5) bf16_to_half_kernel vs direct conversion
        if kernel_runner is not None:
            mean_s = kernel_runner.bf16_to_half(bf16, fp16_tmp, args.warmup, args.iterations)
            results.append(
                _bench_op(
                    "bf16_to_half_kernel",
                    bf16_fp32,
                    size,
                    _bytes_per_elem_for_path("bf16_to_half_kernel"),
                    bf16_copy_gb_s,
                    mean_s=mean_s,
                    timing_source="metal",
                )
            )

        def bf16_to_half_direct() -> None:
            fp16_tmp.copy_(bf16)

        results.append(
            _bench_op(
                "BF16->FP16 direct",
                bf16_to_half_direct,
                size,
                _bytes_per_elem_for_path("bf16_to_half"),
                bf16_copy_gb_s,
            )
        )

        # 6) Fused load+convert vs separate kernels
        if kernel_runner is not None:
            mean_s = kernel_runner.bf16_to_float(bf16, fp32_out, args.warmup, args.iterations)
            results.append(
                _bench_op(
                    "bf16_to_float_kernel",
                    bf16_fp32,
                    size,
                    _bytes_per_elem_for_path("bf16_to_float_kernel"),
                    bf16_copy_gb_s,
                    mean_s=mean_s,
                    timing_source="metal",
                )
            )

            grid, threadgroup, buffers = kernel_runner.prepare_bf16_to_half(bf16, fp16_tmp)

            def bf16_kernel_then_half_to_float() -> None:
                dispatch_kernel(kernel_runner.lib, "bf16_to_half_kernel", grid, threadgroup, buffers, wait=True)
                fp32_out.copy_(fp16_tmp)

            results.append(
                _bench_op(
                    "kernel BF16->FP16 + FP32",
                    bf16_kernel_then_half_to_float,
                    size,
                    _bytes_per_elem_for_path("bf16_fp16_kernel_fp32"),
                    bf16_copy_gb_s,
                )
            )

        _print_results(size, results)

    print("\nLegend: util = GB/s / baseline copy GB/s")
