#!/usr/bin/env python3
"""Microbenchmark: device reads vs threadgroup cache on Apple GPUs.

Compares three access patterns for a half-precision buffer:
1) Copy to threadgroup then read from threadgroup (cache-like).
2) Direct device reads (rely on L1/L2 cache).
3) Hybrid: device read once, threadgroup for reuse.

Intended to highlight M3/A17+ behavior where direct device reads can
outperform threadgroup staging for cache-like usage.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_PYOBJC_METAL, Metal  # noqa: E402
from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_kernel  # noqa: E402

_KERNEL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint kTGSize = 256;

kernel void tg_copy_read(
    device const volatile half* A [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    constant uint& iters [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint idx = tg_id * kTGSize + tid;
    if (idx >= elements) {
        return;
    }
    threadgroup volatile half shared[kTGSize];
    shared[tid] = A[idx];
    threadgroup_barrier(mem_threadgroup);
    float acc = 0.0f;
    for (uint i = 0; i < iters; ++i) {
        acc += float(shared[tid]);
    }
    out[idx] = acc;
}

kernel void direct_read(
    device const volatile half* A [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    constant uint& iters [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint idx = tg_id * kTGSize + tid;
    if (idx >= elements) {
        return;
    }
    float acc = 0.0f;
    for (uint i = 0; i < iters; ++i) {
        acc += float(A[idx]);
    }
    out[idx] = acc;
}

kernel void hybrid_read(
    device const volatile half* A [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    constant uint& iters [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint idx = tg_id * kTGSize + tid;
    if (idx >= elements) {
        return;
    }
    float acc = float(A[idx]);
    threadgroup volatile half shared[kTGSize];
    shared[tid] = half(acc);
    threadgroup_barrier(mem_threadgroup);
    for (uint i = 1; i < iters; ++i) {
        acc += float(shared[tid]);
    }
    out[idx] = acc;
}
"""


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    device_gb_s: float


def _parse_iters(raw: str) -> list[int]:
    if "," in raw:
        return [int(part) for part in raw.split(",") if part.strip()]
    return [int(raw)]


def _make_uint_buffer(device: object, value: int) -> object:
    arr = np.array([value], dtype=np.uint32)
    return device.newBufferWithBytes_length_options_(
        arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
    )


def _benchmark(
    *,
    lib: MetalKernelLibrary,
    kernel_name: str,
    buffers: list[object],
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    warmup: int,
    repeats: int,
) -> BenchResult:
    for _ in range(warmup):
        dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers, wait=True)

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers, wait=True)
        times.append((time.perf_counter() - start) * 1000.0)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return BenchResult(
        name=kernel_name,
        mean_ms=mean,
        std_ms=std,
        min_ms=min(times),
        max_ms=max(times),
        device_gb_s=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Device vs threadgroup memory access benchmark")
    parser.add_argument("--elements", type=int, default=1 << 20, help="Number of half elements")
    parser.add_argument(
        "--iters",
        type=str,
        default="1,4,16,64",
        help="Reuse iterations per element (comma-separated)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=50, help="Timed iterations")
    args = parser.parse_args()

    if not HAS_PYOBJC_METAL or Metal is None:
        raise RuntimeError(
            "PyObjC Metal is required. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )

    elements = int(args.elements)
    iters_list = _parse_iters(args.iters)

    lib = MetalKernelLibrary()
    lib.compile_source("memory_access_bench", _KERNEL_SOURCE)

    device = lib.device
    tg_size = 256
    grid_x = (elements + tg_size - 1) // tg_size

    rng = np.random.default_rng(0)
    data = rng.standard_normal(elements, dtype=np.float32).astype(np.float16)
    input_buf = device.newBufferWithBytes_length_options_(
        data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
    )
    output_buf = device.newBufferWithLength_options_(
        elements * 4, Metal.MTLResourceStorageModeShared
    )

    kernels = [
        ("tg_copy_read", True),
        ("direct_read", False),
        ("hybrid_read", True),
    ]

    print("Metal Memory Access Microbenchmark")
    print(f"Elements: {elements}  Threadgroup: {tg_size}")
    print("Legend: device GB/s estimates only count device reads (threadgroup reads excluded).")
    print()

    for reuse in iters_list:
        elements_buf = _make_uint_buffer(device, elements)
        iters_buf = _make_uint_buffer(device, reuse)
        buffers = [input_buf, output_buf, elements_buf, iters_buf]

        print(f"Reuse iters: {reuse}")
        for kernel_name, device_once in kernels:
            result = _benchmark(
                lib=lib,
                kernel_name=kernel_name,
                buffers=buffers,
                grid=(grid_x, 1, 1),
                threadgroup=(tg_size, 1, 1),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            device_reads = elements * 2 * (1 if device_once else reuse)
            seconds = result.mean_ms / 1000.0 if result.mean_ms > 0 else 0.0
            device_gb_s = (device_reads / seconds) / 1e9 if seconds > 0 else 0.0
            print(
                f"  {kernel_name:12s}  "
                f"{result.mean_ms:8.3f} ms  "
                f"(std {result.std_ms:6.3f})  "
                f"device {device_gb_s:8.2f} GB/s"
            )
        print()


if __name__ == "__main__":
    main()
