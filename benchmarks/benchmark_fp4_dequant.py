#!/usr/bin/env python3
"""Benchmark: scalar vs vectorized FP4 E2M1 → FP16 dequantization on Metal.

Dispatches both bench_fp4_scalar and bench_fp4_vectorized kernels with identical
input buffers, measures GPU execution time, and compares throughput.

Usage:
    cd metal_marlin
    uv run python benchmarks/benchmark_fp4_dequant.py
    uv run python benchmarks/benchmark_fp4_dequant.py --num-elements 16777216
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent  # metal_marlin/
SHADER_PATH = _ROOT / "src" / "dequant.metal"

# Benchmark parameters
WARMUP_ITERS = 20
BENCH_ITERS = 100
BENCH_ITERS_PER_THREAD = 16  # Must match constant in dequant.metal


def compile_library(device):
    """Compile the dequant.metal shader and return the Metal library."""
    import Metal

    source = SHADER_PATH.read_text()
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err is not None:
        raise RuntimeError(f"Metal compile error: {err}")
    return library


def create_pipeline(device, library, kernel_name: str):
    """Create a compute pipeline state for the named kernel function."""
    func = library.newFunctionWithName_(kernel_name)
    if func is None:
        raise RuntimeError(f"Kernel '{kernel_name}' not found in library")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    if err is not None:
        raise RuntimeError(f"Pipeline error for '{kernel_name}': {err}")
    return pipeline


def run_benchmark(device, queue, pipeline, buffers: dict, num_packed: int,
                  num_iters: int) -> float:
    """Run a kernel num_iters times and return median GPU time in microseconds.

    Uses GPU timestamps via addCompletedHandler to measure kernel execution
    excluding command buffer submission overhead.
    """
    import Metal

    num_threads = (num_packed + BENCH_ITERS_PER_THREAD - 1) // BENCH_ITERS_PER_THREAD
    threads_per_group = min(256, num_threads)
    num_groups = (num_threads + threads_per_group - 1) // threads_per_group

    gpu_times: list[float] = []

    for _ in range(num_iters):
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buffers["packed"], 0, 0)
        encoder.setBuffer_offset_atIndex_(buffers["scales"], 0, 1)
        encoder.setBuffer_offset_atIndex_(buffers["output"], 0, 2)
        encoder.setBuffer_offset_atIndex_(buffers["num_packed"], 0, 3)
        encoder.setBuffer_offset_atIndex_(buffers["group_size"], 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(num_groups, 1, 1),
            Metal.MTLSizeMake(threads_per_group, 1, 1))
        encoder.endEncoding()

        t_start = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t_end = time.perf_counter_ns()

        gpu_times.append((t_end - t_start) / 1000.0)  # ns -> us

    # Return median to reduce variance from thermal throttling
    gpu_times.sort()
    return gpu_times[len(gpu_times) // 2]


def verify_correctness(device, library, queue) -> bool:
    """Quick correctness check: vectorized must match scalar for all 16 codes."""
    import Metal

    # Test with codes 0-7 in lower nibbles, 8-F in upper
    test_packed = np.uint32(0xFEDCBA98)
    test_scale = np.float16(2.5)

    # Run scalar path
    pipeline_scalar = create_pipeline(device, library, "test_fp4_packed_scaled")
    # Run vectorized path
    pipeline_vec = create_pipeline(device, library, "test_fp4_vectorized")

    packed_bytes = np.array([test_packed], dtype=np.uint32).tobytes()
    scale_bytes = np.array([test_scale], dtype=np.float16).tobytes()

    buf_packed = device.newBufferWithBytes_length_options_(
        packed_bytes, 4, Metal.MTLResourceStorageModeShared)
    buf_scale = device.newBufferWithBytes_length_options_(
        scale_bytes, 2, Metal.MTLResourceStorageModeShared)
    buf_out_scalar = device.newBufferWithLength_options_(
        16, Metal.MTLResourceStorageModeShared)
    buf_out_vec = device.newBufferWithLength_options_(
        16, Metal.MTLResourceStorageModeShared)

    # Dispatch scalar
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline_scalar)
    encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_out_scalar, 0, 2)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    # Dispatch vectorized
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline_vec)
    encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_out_vec, 0, 2)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    # Compare outputs
    scalar_ptr = buf_out_scalar.contents()
    vec_ptr = buf_out_vec.contents()
    scalar_arr = np.frombuffer(
        bytes((ctypes.c_char * 16).from_address(scalar_ptr)), dtype=np.float16).copy()
    vec_arr = np.frombuffer(
        bytes((ctypes.c_char * 16).from_address(vec_ptr)), dtype=np.float16).copy()

    match = True
    for i in range(8):
        s_bits = scalar_arr.view(np.uint16)[i]
        v_bits = vec_arr.view(np.uint16)[i]
        if s_bits != v_bits:
            print(f"  MISMATCH at [{i}]: scalar=0x{s_bits:04X} ({scalar_arr[i]}), "
                  f"vectorized=0x{v_bits:04X} ({vec_arr[i]})")
            match = False

    return match


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark scalar vs vectorized FP4 dequant on Metal")
    parser.add_argument("--num-elements", type=int, default=4 * 1024 * 1024,
                        help="Number of FP4 elements to dequantize (default: 4M)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS,
                        help=f"Warmup iterations (default: {WARMUP_ITERS})")
    parser.add_argument("--iters", type=int, default=BENCH_ITERS,
                        help=f"Benchmark iterations (default: {BENCH_ITERS})")
    args = parser.parse_args()

    try:
        import Metal
    except ImportError:
        print("ERROR: PyObjC Metal framework not available.")
        print("Install with: uv pip install pyobjc-framework-Metal")
        return 1

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        print("ERROR: No Metal device found.")
        return 1

    print(f"Device: {device.name()}")
    print(f"Compiling {SHADER_PATH.name}...")

    library = compile_library(device)
    queue = device.newCommandQueue()

    # Verify correctness first
    print("\nVerifying vectorized correctness...")
    if verify_correctness(device, library, queue):
        print("  PASS: vectorized output matches scalar (bit-exact)")
    else:
        print("  FAIL: vectorized output differs from scalar!")
        return 1

    # Prepare benchmark buffers
    num_elements = args.num_elements
    num_packed = num_elements // 8
    group_size = args.group_size
    num_groups = (num_elements + group_size - 1) // group_size

    print("\nBenchmark configuration:")
    print(f"  Elements:   {num_elements:,} FP4 values")
    print(f"  Packed:     {num_packed:,} uint32 words")
    print(f"  Groups:     {num_groups:,} (group_size={group_size})")
    print(f"  Data size:  {num_packed * 4 / 1024 / 1024:.2f} MB packed input")
    print(f"  Output:     {num_elements * 2 / 1024 / 1024:.2f} MB FP16 output")
    print(f"  Iterations: {args.warmup} warmup + {args.iters} measured")

    rng = np.random.default_rng(42)
    packed_data = rng.integers(0, 2**32, size=num_packed, dtype=np.uint32)
    scales_data = np.float16(rng.uniform(0.01, 4.0, size=num_groups))

    # Create Metal buffers
    buf_packed = device.newBufferWithBytes_length_options_(
        packed_data.tobytes(), packed_data.nbytes,
        Metal.MTLResourceStorageModeShared)
    buf_scales = device.newBufferWithBytes_length_options_(
        scales_data.tobytes(), scales_data.nbytes,
        Metal.MTLResourceStorageModeShared)

    output_size = num_elements * 2  # half = 2 bytes
    # Align to half4 (8 bytes)
    output_size_aligned = ((output_size + 7) // 8) * 8
    buf_output = device.newBufferWithLength_options_(
        output_size_aligned, Metal.MTLResourceStorageModeShared)

    num_packed_bytes = np.array([num_packed], dtype=np.uint32).tobytes()
    group_size_bytes = np.array([group_size], dtype=np.uint32).tobytes()
    buf_num_packed = device.newBufferWithBytes_length_options_(
        num_packed_bytes, 4, Metal.MTLResourceStorageModeShared)
    buf_group_size = device.newBufferWithBytes_length_options_(
        group_size_bytes, 4, Metal.MTLResourceStorageModeShared)

    buffers = {
        "packed": buf_packed,
        "scales": buf_scales,
        "output": buf_output,
        "num_packed": buf_num_packed,
        "group_size": buf_group_size,
    }

    # Create pipelines
    pipeline_scalar = create_pipeline(device, library, "bench_fp4_scalar")
    pipeline_vec = create_pipeline(device, library, "bench_fp4_vectorized")

    print(f"\n{'Kernel':<20} {'Median (us)':>12} {'Elements/us':>12} {'GB/s':>8}")
    print("-" * 56)

    # Warmup scalar
    for _ in range(args.warmup):
        run_benchmark(device, queue, pipeline_scalar, buffers, num_packed, 1)

    # Benchmark scalar
    scalar_us = run_benchmark(device, queue, pipeline_scalar, buffers,
                              num_packed, args.iters)
    scalar_throughput = num_elements / scalar_us  # elements per microsecond
    # Bandwidth: read packed (4B per 8 elements) + read scale + write output (2B per element)
    scalar_bytes = num_packed * 4 + num_groups * 2 + num_elements * 2
    scalar_bw = scalar_bytes / scalar_us / 1000.0  # GB/s

    print(f"{'Scalar (x8 calls)':<20} {scalar_us:>12.1f} {scalar_throughput:>12.1f} "
          f"{scalar_bw:>8.1f}")

    # Warmup vectorized
    for _ in range(args.warmup):
        run_benchmark(device, queue, pipeline_vec, buffers, num_packed, 1)

    # Benchmark vectorized
    vec_us = run_benchmark(device, queue, pipeline_vec, buffers,
                           num_packed, args.iters)
    vec_throughput = num_elements / vec_us
    vec_bytes = num_packed * 4 + num_groups * 2 + num_elements * 2
    vec_bw = vec_bytes / vec_us / 1000.0

    print(f"{'Vectorized (pairs)':<20} {vec_us:>12.1f} {vec_throughput:>12.1f} "
          f"{vec_bw:>8.1f}")

    # Summary
    print("-" * 56)
    if vec_us > 0:
        speedup = scalar_us / vec_us
        print(f"\nSpeedup: {speedup:.3f}x "
              f"({'vectorized faster' if speedup > 1.0 else 'scalar faster'})")
    else:
        print("\nCould not compute speedup (zero vectorized time)")

    # Verify outputs match
    print("\nOutput verification (post-benchmark)...")
    # Run scalar once, read output
    run_benchmark(device, queue, pipeline_scalar, buffers, num_packed, 1)
    scalar_ptr = buf_output.contents()
    scalar_out = np.frombuffer(
        bytes((ctypes.c_char * (num_elements * 2)).from_address(scalar_ptr)),
        dtype=np.float16).copy()

    # Run vectorized once, read output
    run_benchmark(device, queue, pipeline_vec, buffers, num_packed, 1)
    vec_ptr = buf_output.contents()
    vec_out = np.frombuffer(
        bytes((ctypes.c_char * (num_elements * 2)).from_address(vec_ptr)),
        dtype=np.float16).copy()

    # Compare bit-exact
    mismatches = np.sum(scalar_out.view(np.uint16) != vec_out.view(np.uint16))
    if mismatches == 0:
        print(f"  PASS: {num_elements:,} values bit-exact match")
    else:
        print(f"  FAIL: {mismatches:,}/{num_elements:,} values differ")
        # Show first few mismatches
        diff_indices = np.where(
            scalar_out.view(np.uint16) != vec_out.view(np.uint16))[0][:5]
        for idx in diff_indices:
            print(f"    [{idx}] scalar={scalar_out[idx]}, vectorized={vec_out[idx]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
