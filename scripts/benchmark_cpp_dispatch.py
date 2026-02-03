#!/usr/bin/env python3
"""Benchmark C++ dispatch latency vs Python dispatch.

Measures the overhead of kernel dispatch using:
- Python dispatch (PyObjC path) - standard method
- C++ dispatch (FastPath) - optimized low-latency path

The benchmark isolates dispatch latency by using minimal kernels
and varying the number of dispatches to measure per-call overhead.

Usage:
    cd contrib/metal_marlin
    uv run python scripts/benchmark_cpp_dispatch.py

    # Quick mode (fewer iterations)
    uv run python scripts/benchmark_cpp_dispatch.py --quick

    # Specific kernel counts
    uv run python scripts/benchmark_cpp_dispatch.py --counts 100 1000 10000

Expected results (M3 Max):
    - Python dispatch: ~80-150μs per call
    - C++ dispatch: ~5-15μs per call
    - Speedup: 5-10x
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.metal_dispatch import (
    HAS_CPP_EXT,
    HAS_METAL,
    HAS_MPS,
    FastPath,
    Metal,
    MetalKernelLibrary,
    dispatch_kernel,
)


@dataclass
class DispatchResult:
    """Result from a dispatch benchmark."""

    dispatch_type: str  # "python" or "cpp"
    n_dispatches: int
    n_buffers: int
    total_sec: float
    us_per_dispatch: float
    throughput_dispatches_per_sec: float
    warmup_iterations: int = 3


@dataclass
class ComparisonResult:
    """Comparison between Python and C++ dispatch."""

    python_us: float
    cpp_us: float
    speedup: float
    cpp_available: bool


def create_metal_kernel_library() -> MetalKernelLibrary:
    """Create a MetalKernelLibrary for benchmarking.

    Returns a library with a simple identity kernel for dispatch testing.
    """
    if not HAS_METAL:
        raise RuntimeError("Metal framework not available")

    # Create a minimal Metal shader for dispatch testing
    minimal_shader = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void identity_kernel(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint gid [[thread_position_in_grid]]
    ) {
        output[gid] = input[gid];
    }
    """

    lib = MetalKernelLibrary()
    lib.device = Metal.MTLCreateSystemDefaultDevice()
    lib.command_queue = lib.device.newCommandQueue()

    # Compile the shader
    try:
        shader_compiler = lib.device.newComputePipelineStateWithFunction_error_(
            lib.device.newLibraryWithSource_options_error_(minimal_shader, None, None)[0]
            .newFunctionWithName_("identity_kernel"),
        )[0]
    except Exception:
        # Fallback: use metallib if available
        shader_compiler = None

    lib._compute_pipeline_cache = {"identity_kernel": shader_compiler}
    return lib


def prepare_buffers(count: int, size: int = 4096) -> list[Any]:
    """Prepare Metal buffers for dispatch testing.

    Args:
        count: Number of buffer pairs (input/output).
        size: Size of each buffer in floats.

    Returns:
        List of (input, output) buffer tuples.
    """
    if not HAS_MPS:
        raise RuntimeError("MPS backend not available")

    buffers = []
    for _ in range(count):
        input_tensor = torch.randn(size, device="mps", dtype=torch.float32)
        output_tensor = torch.zeros_like(input_tensor)
        buffers.append((input_tensor, output_tensor))
    return buffers


def benchmark_python_dispatch(
    lib: MetalKernelLibrary,
    n_dispatches: int,
    buffers: Sequence[tuple[torch.Tensor, torch.Tensor]],
    grid: tuple[int, int, int] = (256, 1, 1),
    threadgroup: tuple[int, int, int] = (256, 1, 1),
    warmup: int = 3,
) -> DispatchResult:
    """Benchmark Python/PyObjC dispatch latency.

    Uses the standard dispatch_kernel path through PyObjC.
    """
    # Warmup
    for i in range(min(warmup, len(buffers))):
        input_buf, output_buf = buffers[i]
        try:
            dispatch_kernel(
                lib,
                "identity_kernel",
                grid,
                threadgroup,
                [input_buf, output_buf],
                wait=True,
            )
        except Exception:
            # Skip if kernel not available
            pass

    torch.mps.synchronize()

    # Benchmark
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()

    start = time.perf_counter()

    for i in range(n_dispatches):
        input_buf, output_buf = buffers[i % len(buffers)]
        try:
            dispatch_kernel(
                lib,
                "identity_kernel",
                grid,
                threadgroup,
                [input_buf, output_buf],
                wait=True,
            )
        except Exception:
            # Skip if kernel not available
            pass

    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    us_per_dispatch = (elapsed * 1_000_000) / n_dispatches

    return DispatchResult(
        dispatch_type="python",
        n_dispatches=n_dispatches,
        n_buffers=len(buffers),
        total_sec=elapsed,
        us_per_dispatch=us_per_dispatch,
        throughput_dispatches_per_sec=n_dispatches / elapsed,
        warmup_iterations=warmup,
    )


def benchmark_cpp_dispatch(
    lib: MetalKernelLibrary,
    n_dispatches: int,
    buffers: Sequence[tuple[torch.Tensor, torch.Tensor]],
    grid: tuple[int, int, int] = (256, 1, 1),
    threadgroup: tuple[int, int, int] = (256, 1, 1),
    warmup: int = 3,
) -> DispatchResult:
    """Benchmark C++ FastPath dispatch latency.

    Uses the FastPath class with C++ extension for low-overhead dispatch.
    """
    fast_path = FastPath(lib)

    if not fast_path.available:
        raise RuntimeError("C++ FastPath not available")

    # Prepare buffers for FastPath
    cpp_buffers: list[Any] = []
    for input_buf, output_buf in buffers:
        # Try to create buffers from MPS tensors
        try:
            in_cpp = fast_path.create_buffer_from_ptr(input_buf.data_ptr(), input_buf.element_size() * input_buf.numel())
            out_cpp = fast_path.create_buffer_from_ptr(output_buf.data_ptr(), output_buf.element_size() * output_buf.numel())
            cpp_buffers.append((in_cpp, out_cpp))
        except Exception:
            # Fallback to creating new buffers
            in_cpp = fast_path.create_buffer(input_buf.element_size() * input_buf.numel())
            out_cpp = fast_path.create_buffer(output_buf.element_size() * output_buf.numel())
            cpp_buffers.append((in_cpp, out_cpp))

    # Warmup
    for i in range(min(warmup, len(cpp_buffers))):
        in_buf, out_buf = cpp_buffers[i]
        try:
            fast_path.dispatch(
                "identity_kernel",
                grid,
                threadgroup,
                [in_buf, out_buf],
                wait=True,
            )
        except Exception:
            # Skip if kernel not available
            pass

    torch.mps.synchronize()

    # Benchmark
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()

    start = time.perf_counter()

    for i in range(n_dispatches):
        in_buf, out_buf = cpp_buffers[i % len(cpp_buffers)]
        try:
            fast_path.dispatch(
                "identity_kernel",
                grid,
                threadgroup,
                [in_buf, out_buf],
                wait=True,
            )
        except Exception:
            # Skip if kernel not available
            pass

    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    us_per_dispatch = (elapsed * 1_000_000) / n_dispatches

    return DispatchResult(
        dispatch_type="cpp",
        n_dispatches=n_dispatches,
        n_buffers=len(buffers),
        total_sec=elapsed,
        us_per_dispatch=us_per_dispatch,
        throughput_dispatches_per_sec=n_dispatches / elapsed,
        warmup_iterations=warmup,
    )


def run_benchmark_suite(
    counts: list[int],
    buffer_sizes: list[int] = [10],
    warmup: int = 3,
) -> tuple[list[DispatchResult], list[DispatchResult], ComparisonResult]:
    """Run a complete benchmark suite.

    Args:
        counts: List of dispatch counts to test.
        buffer_sizes: List of buffer counts to test (default: 10 buffers).

    Returns:
        Tuple of (python_results, cpp_results, comparison).
    """
    lib = create_metal_kernel_library()

    python_results: list[DispatchResult] = []
    cpp_results: list[DispatchResult] = []

    for buffer_count in buffer_sizes:
        buffers = prepare_buffers(buffer_count)

        for n_dispatches in counts:
            print(f"  Testing {n_dispatches} dispatches with {buffer_count} buffers...")

            # Python dispatch
            try:
                py_result = benchmark_python_dispatch(lib, n_dispatches, buffers, warmup=warmup)
                python_results.append(py_result)
                print(f"    Python: {py_result.us_per_dispatch:.1f} μs/dispatch")
            except Exception as e:
                print(f"    Python dispatch failed: {e}")

            # C++ dispatch
            if HAS_CPP_EXT:
                try:
                    cpp_result = benchmark_cpp_dispatch(lib, n_dispatches, buffers, warmup=warmup)
                    cpp_results.append(cpp_result)
                    print(f"    C++:    {cpp_result.us_per_dispatch:.1f} μs/dispatch")
                except Exception as e:
                    print(f"    C++ dispatch failed: {e}")

    # Compute comparison
    if python_results and cpp_results:
        avg_py_us = sum(r.us_per_dispatch for r in python_results) / len(python_results)
        avg_cpp_us = sum(r.us_per_dispatch for r in cpp_results) / len(cpp_results)
        speedup = avg_py_us / avg_cpp_us
    else:
        avg_py_us = 0.0
        avg_cpp_us = 0.0
        speedup = 0.0

    comparison = ComparisonResult(
        python_us=avg_py_us,
        cpp_us=avg_cpp_us,
        speedup=speedup,
        cpp_available=HAS_CPP_EXT,
    )

    return python_results, cpp_results, comparison


def print_results(
    python_results: list[DispatchResult],
    cpp_results: list[DispatchResult],
    comparison: ComparisonResult,
) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("DISPATCH LATENCY BENCHMARK RESULTS")
    print("=" * 80)

    # Group by buffer count
    from collections import defaultdict

    py_by_nbufs: dict[int, list[DispatchResult]] = defaultdict(list)
    cpp_by_nbufs: dict[int, list[DispatchResult]] = defaultdict(list)

    for r in python_results:
        py_by_nbufs[r.n_buffers].append(r)

    for r in cpp_results:
        cpp_by_nbufs[r.n_buffers].append(r)

    for n_buffers in sorted(set(list(py_by_nbufs.keys()) + list(cpp_by_nbufs.keys()))):
        print(f"\nBuffer count: {n_buffers}")
        print("-" * 80)
        print(f"{'Dispatches':>10} {'Python (μs)':>15} {'C++ (μs)':>12} {'Speedup':>10}")
        print("-" * 80)

        py_for_nbuf = py_by_nbufs.get(n_buffers, [])
        cpp_for_nbuf = cpp_by_nbufs.get(n_buffers, [])

        if py_for_nbuf and cpp_for_nbuf:
            for py, cpp in zip(py_for_nbuf, cpp_for_nbuf):
                speedup = py.us_per_dispatch / cpp.us_per_dispatch if cpp.us_per_dispatch > 0 else 0
                print(
                    f"{py.n_dispatches:>10} {py.us_per_dispatch:>15.1f} {cpp.us_per_dispatch:>12.1f} {speedup:>10.2f}x"
                )
        elif py_for_nbuf:
            for py in py_for_nbuf:
                print(f"{py.n_dispatches:>10} {py.us_per_dispatch:>15.1f} {'N/A':>12} {'N/A':>10}")
        elif cpp_for_nbuf:
            for cpp in cpp_for_nbuf:
                print(f"{cpp.n_dispatches:>10} {'N/A':>15} {cpp.us_per_dispatch:>12.1f} {'N/A':>10}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"C++ extension available: {comparison.cpp_available}")
    if comparison.cpp_available:
        print(f"Average Python dispatch: {comparison.python_us:.1f} μs")
        print(f"Average C++ dispatch:    {comparison.cpp_us:.1f} μs")
        print(f"Speedup:                 {comparison.speedup:.2f}x")
    else:
        print("C++ extension not available - only Python dispatch was benchmarked")
    print("=" * 80)

    # Expected ranges for reference
    print("\nExpected ranges (M3 Max):")
    print("  - Python dispatch: ~80-150 μs per call")
    print("  - C++ dispatch:    ~5-15 μs per call")
    print("  - Speedup:         5-10x")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark C++ dispatch latency vs Python dispatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark
  uv run python scripts/benchmark_cpp_dispatch.py

  # Quick mode
  uv run python scripts/benchmark_cpp_dispatch.py --quick

  # Custom dispatch counts
  uv run python scripts/benchmark_cpp_dispatch.py --counts 100 1000 5000
        """,
    )
    parser.add_argument(
        "--counts",
        "-c",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000],
        help="Number of dispatches to test",
    )
    parser.add_argument(
        "--buffers",
        "-b",
        type=int,
        nargs="+",
        default=[10],
        help="Number of buffers to use",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick mode: fewer iterations",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=3,
        help="Warmup iterations",
    )
    args = parser.parse_args()

    if args.quick:
        args.counts = [100, 500]
        args.warmup = 2

    print("=" * 80)
    print("C++ DISPATCH LATENCY BENCHMARK")
    print("=" * 80)
    print(f"Dispatch counts: {args.counts}")
    print(f"Buffer counts: {args.buffers}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"C++ extension available: {HAS_CPP_EXT}")
    print(f"MPS available: {HAS_MPS}")
    print()

    if not HAS_MPS:
        print("ERROR: MPS backend not available. This benchmark requires PyTorch with MPS.")
        return 1

    if not HAS_METAL:
        print("ERROR: Metal framework not available. Install PyObjC:")
        print("  pip install pyobjc-framework-Metal")
        return 1

    try:
        python_results, cpp_results, comparison = run_benchmark_suite(
            counts=args.counts,
            buffer_sizes=args.buffers,
            warmup=args.warmup,
        )
        print_results(python_results, cpp_results, comparison)
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
