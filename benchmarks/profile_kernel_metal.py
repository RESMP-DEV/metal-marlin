#!/usr/bin/env python3
"""Metal kernel occupancy and utilization profiler for Apple Silicon GPUs.

This script profiles Metal compute kernels to measure:
- Threadgroup utilization (threads/simdgroups per threadgroup)
- Memory bandwidth utilization (achieved vs theoretical)
- ALU utilization (TFLOPS achieved vs peak)
- Occupancy (active threads / max threads)
- Bottleneck identification (memory vs compute bound)

Metal does not expose hardware performance counters programmatically like CUDA's
cupti or AMD's ROCm profiler. Instead, we use:
1. Command buffer GPU timestamps (GPUStartTime/GPUEndTime) for accurate timing
2. Pipeline state introspection (maxTotalThreadsPerThreadgroup, threadExecutionWidth)
3. Device capabilities (recommendedMaxWorkingSetSize, memory bandwidth estimates)
4. Roofline model analysis to infer bottlenecks

For detailed hardware profiling, use Instruments.app with:
- Metal System Trace
- GPU Counters
- Metal Memory Graph

Usage:
    uv run python benchmarks/profile_kernel_metal.py
    uv run python benchmarks/profile_kernel_metal.py --kernel gemm_trellis --bits 3
    uv run python benchmarks/profile_kernel_metal.py --all-kernels --export-json results/profile.json

Requirements:
    - PyObjC: pip install pyobjc-framework-Metal
    - PyTorch with MPS backend
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure metal_marlin is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Check dependencies
try:
    import torch

    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None

try:
    import Foundation
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None
    Foundation = None


@dataclass
class DeviceCapabilities:
    """Metal device hardware capabilities."""

    name: str
    gpu_family: int  # 7=M1, 8=M2, 9=M3/M4
    max_threads_per_threadgroup: int
    thread_execution_width: int  # SIMD width (32 on Apple Silicon)
    max_threadgroup_memory_bytes: int
    recommended_working_set_gb: float
    # Estimated peak performance (architecture-dependent)
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    memory_bandwidth_gb_s: float


@dataclass
class PipelineProfile:
    """Compute pipeline state properties."""

    function_name: str
    max_total_threads: int
    thread_execution_width: int
    static_threadgroup_memory_bytes: int
    # Derived
    max_simdgroups: int
    max_occupancy_threads: int


@dataclass
class KernelExecutionProfile:
    """Profile of a single kernel execution."""

    name: str
    M: int
    N: int
    K: int
    bits: int
    group_size: int
    # Timing
    gpu_time_ms: float
    cpu_time_ms: float
    timing_source: str
    # Throughput
    tflops: float
    memory_gb_s: float
    # Efficiency
    compute_utilization_pct: float  # tflops / peak_tflops
    memory_utilization_pct: float  # memory_gb_s / peak_memory_bw
    # Occupancy
    threadgroup_size: tuple[int, int, int]
    grid_size: tuple[int, int, int]
    total_threads: int
    total_threadgroups: int
    threads_per_threadgroup: int
    simdgroups_per_threadgroup: int
    occupancy_pct: float  # threads_per_tg / max_threads_per_tg
    # Bottleneck analysis
    arithmetic_intensity: float  # FLOP / byte
    bottleneck: str  # "memory", "compute", or "balanced"
    roofline_attainment_pct: float
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelProfileSummary:
    """Summary across multiple kernel configurations."""

    kernel_name: str
    device: DeviceCapabilities
    pipeline: PipelineProfile
    profiles: list[KernelExecutionProfile]
    # Aggregate statistics
    avg_compute_utilization: float
    avg_memory_utilization: float
    avg_occupancy: float
    bottleneck_distribution: dict[str, int]  # count per bottleneck type


class MetalKernelProfiler:
    """Profiler for Metal compute kernels with occupancy and utilization analysis."""

    # Known peak performance for Apple Silicon chips (rough estimates)
    # These vary by chip variant (M4, M4 Pro, M4 Max, etc.)
    _DEVICE_SPECS: dict[str, dict[str, float]] = {
        # M1 family
        "Apple M1": {"fp16_tflops": 5.5, "fp32_tflops": 2.6, "mem_bw": 68.25},
        "Apple M1 Pro": {"fp16_tflops": 8.9, "fp32_tflops": 4.2, "mem_bw": 200},
        "Apple M1 Max": {"fp16_tflops": 17.8, "fp32_tflops": 8.4, "mem_bw": 400},
        "Apple M1 Ultra": {"fp16_tflops": 35.6, "fp32_tflops": 16.8, "mem_bw": 800},
        # M2 family
        "Apple M2": {"fp16_tflops": 7.0, "fp32_tflops": 3.6, "mem_bw": 100},
        "Apple M2 Pro": {"fp16_tflops": 13.6, "fp32_tflops": 6.8, "mem_bw": 200},
        "Apple M2 Max": {"fp16_tflops": 27.2, "fp32_tflops": 13.6, "mem_bw": 400},
        "Apple M2 Ultra": {"fp16_tflops": 54.4, "fp32_tflops": 27.2, "mem_bw": 800},
        # M3 family
        "Apple M3": {"fp16_tflops": 8.4, "fp32_tflops": 4.2, "mem_bw": 100},
        "Apple M3 Pro": {"fp16_tflops": 14.0, "fp32_tflops": 7.0, "mem_bw": 150},
        "Apple M3 Max": {"fp16_tflops": 28.0, "fp32_tflops": 14.0, "mem_bw": 400},
        # M4 family
        "Apple M4": {"fp16_tflops": 8.4, "fp32_tflops": 4.2, "mem_bw": 120},
        "Apple M4 Pro": {"fp16_tflops": 16.8, "fp32_tflops": 8.4, "mem_bw": 273},
        "Apple M4 Max": {"fp16_tflops": 33.6, "fp32_tflops": 16.8, "mem_bw": 546},
    }

    def __init__(
        self,
        warmup: int = 10,
        iterations: int = 50,
        use_gpu_timestamps: bool = True,
    ):
        if not HAS_METAL:
            raise RuntimeError(
                "Metal profiling requires PyObjC. Install with:\n"
                "  pip install pyobjc-framework-Metal"
            )
        if not HAS_MPS:
            raise RuntimeError(
                "Metal profiling requires PyTorch MPS backend.\n"
                "Ensure you're on Apple Silicon with PyTorch >= 2.0"
            )

        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal device available")

        self._command_queue = self._device.newCommandQueue()
        self._warmup = warmup
        self._iterations = iterations
        self._use_gpu_timestamps = use_gpu_timestamps

        # Compile shader library
        from metal_marlin.metal_dispatch import MetalKernelLibrary

        self._lib = MetalKernelLibrary(self._device)
        # Compile all shaders from src directory
        src_dir = _ROOT / "src"
        for metal_file in sorted(src_dir.glob("*.metal")):
            try:
                self._lib.compile_source(metal_file.stem, metal_file.read_text())
            except Exception as e:
                print(f"Warning: Failed to compile {metal_file.name}: {e}")

        self._capabilities = self._get_device_capabilities()

    def _get_device_capabilities(self) -> DeviceCapabilities:
        """Query Metal device capabilities."""
        device = self._device
        name = str(device.name())

        # GPU family detection
        gpu_family = 7  # Default to M1 era
        for family_num in range(9, 6, -1):  # Check 9, 8, 7
            family_attr = f"MTLGPUFamilyApple{family_num}"
            if hasattr(Metal, family_attr):
                if device.supportsFamily_(getattr(Metal, family_attr)):
                    gpu_family = family_num
                    break

        # Device limits
        max_threads_per_tg = device.maxThreadsPerThreadgroup()
        # max_threads_per_tg is MTLSize; extract width
        if hasattr(max_threads_per_tg, "width"):
            max_threads = max_threads_per_tg.width
        else:
            max_threads = 1024  # Default for Apple Silicon

        # SIMD width (always 32 on Apple Silicon)
        thread_execution_width = 32

        # Threadgroup memory
        max_tg_memory = device.maxThreadgroupMemoryLength() if hasattr(device, "maxThreadgroupMemoryLength") else 32768

        # Working set size
        working_set = device.recommendedMaxWorkingSetSize()
        working_set_gb = working_set / (1024**3)

        # Look up performance specs
        specs = self._DEVICE_SPECS.get(name, {})
        peak_fp16 = specs.get("fp16_tflops", 10.0)  # Conservative default
        peak_fp32 = specs.get("fp32_tflops", 5.0)
        mem_bw = specs.get("mem_bw", 200.0)

        return DeviceCapabilities(
            name=name,
            gpu_family=gpu_family,
            max_threads_per_threadgroup=max_threads,
            thread_execution_width=thread_execution_width,
            max_threadgroup_memory_bytes=max_tg_memory,
            recommended_working_set_gb=working_set_gb,
            peak_fp16_tflops=peak_fp16,
            peak_fp32_tflops=peak_fp32,
            memory_bandwidth_gb_s=mem_bw,
        )

    def _get_pipeline_profile(self, function_name: str) -> PipelineProfile:
        """Get compute pipeline properties."""
        try:
            pipeline = self._lib.get_pipeline(function_name)
        except KeyError as e:
            raise KeyError(f"Kernel function '{function_name}' not found") from e

        max_threads = pipeline.maxTotalThreadsPerThreadgroup()
        thread_width = pipeline.threadExecutionWidth()
        static_tg_mem = pipeline.staticThreadgroupMemoryLength()

        max_simdgroups = max_threads // thread_width

        return PipelineProfile(
            function_name=function_name,
            max_total_threads=max_threads,
            thread_execution_width=thread_width,
            static_threadgroup_memory_bytes=static_tg_mem,
            max_simdgroups=max_simdgroups,
            max_occupancy_threads=max_threads,
        )

    def _dispatch_timed(
        self,
        function_name: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        buffers: list[Any],
    ) -> tuple[float, float, str]:
        """Dispatch kernel with timing.

        Returns:
            (gpu_time_ms, cpu_time_ms, timing_source)
        """
        pipeline = self._lib.get_pipeline(function_name)

        # Warmup
        for _ in range(self._warmup):
            cmd_buf = self._command_queue.commandBuffer()
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

        # Timed iterations
        cpu_times: list[float] = []
        gpu_times: list[float] = []

        for _ in range(self._iterations):
            cpu_start = time.perf_counter()

            cmd_buf = self._command_queue.commandBuffer()
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

            cpu_end = time.perf_counter()
            cpu_times.append((cpu_end - cpu_start) * 1000.0)

            if self._use_gpu_timestamps:
                gpu_start = cmd_buf.GPUStartTime()
                gpu_end = cmd_buf.GPUEndTime()
                if gpu_end > gpu_start:
                    gpu_times.append((gpu_end - gpu_start) * 1000.0)

        cpu_mean = statistics.mean(cpu_times)

        if gpu_times:
            gpu_mean = statistics.mean(gpu_times)
            return gpu_mean, cpu_mean, "metal_gpu_timestamps"
        else:
            return cpu_mean, cpu_mean, "wall_clock"

    def _compute_arithmetic_intensity(
        self,
        M: int,
        N: int,
        K: int,
        bits: int,
        group_size: int,
    ) -> tuple[float, float, float]:
        """Compute arithmetic intensity for quantized GEMM.

        Returns:
            (arithmetic_intensity, flops, bytes_moved)
        """
        # FLOPS: 2 * M * N * K (multiply-accumulate)
        flops = 2.0 * M * N * K

        # Bytes moved:
        # - Read A: M * K * 2 (FP16)
        # - Read B_packed: K * N * bits / 8
        # - Read scales: (K / group_size) * N * 2 (FP16)
        # - Write C: M * N * 2 (FP16)
        bytes_a = M * K * 2
        bytes_b = (K * N * bits) // 8
        bytes_scales = ((K // group_size) * N * 2) if group_size > 0 else 0
        bytes_c = M * N * 2
        bytes_moved = bytes_a + bytes_b + bytes_scales + bytes_c

        ai = flops / bytes_moved if bytes_moved > 0 else 0.0
        return ai, flops, bytes_moved

    def _analyze_bottleneck(
        self,
        arithmetic_intensity: float,
        achieved_tflops: float,
        achieved_memory_gb_s: float,
    ) -> tuple[str, float]:
        """Analyze whether kernel is memory or compute bound.

        Uses roofline model: performance is limited by
        min(peak_compute, arithmetic_intensity * memory_bandwidth)

        Returns:
            (bottleneck_type, roofline_attainment_pct)
        """
        caps = self._capabilities

        # Ridge point: where memory and compute roofs intersect
        ridge_point = (caps.peak_fp16_tflops * 1000.0) / caps.memory_bandwidth_gb_s

        if arithmetic_intensity < ridge_point:
            # Memory bound region
            max_achievable_tflops = (arithmetic_intensity * caps.memory_bandwidth_gb_s) / 1000.0
            bottleneck = "memory"
        else:
            # Compute bound region
            max_achievable_tflops = caps.peak_fp16_tflops
            bottleneck = "compute"

        # Check if we're close to the ridge point (within 20%)
        if 0.8 * ridge_point <= arithmetic_intensity <= 1.2 * ridge_point:
            bottleneck = "balanced"

        # Roofline attainment
        attainment = (achieved_tflops / max_achievable_tflops) * 100 if max_achievable_tflops > 0 else 0.0

        return bottleneck, attainment

    def profile_gemm_kernel(
        self,
        function_name: str,
        M: int,
        N: int,
        K: int,
        bits: int = 4,
        group_size: int = 128,
        tile_m: int = 64,
        tile_n: int = 64,
        threads_per_tg: int = 128,
    ) -> KernelExecutionProfile:
        """Profile a GEMM kernel with full occupancy and utilization analysis.

        Args:
            function_name: Metal kernel function name
            M, N, K: GEMM dimensions
            bits: Weight quantization bits
            group_size: Quantization group size
            tile_m, tile_n: Output tile dimensions
            threads_per_tg: Threads per threadgroup

        Returns:
            KernelExecutionProfile with detailed metrics
        """
        from metal_marlin.metal_dispatch import mps_tensor_to_metal_buffer

        # Create test tensors
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        C = torch.zeros(M, N, dtype=torch.float16, device="mps")

        # Packed weights (random for profiling)
        k_blocks = (K * bits + 31) // 32
        B_packed = torch.randint(0, 2**31, (k_blocks, N), dtype=torch.int32, device="mps")

        # Scales
        num_groups = (K + group_size - 1) // group_size
        scales = torch.randn(num_groups, N, dtype=torch.float16, device="mps") * 0.1 + 0.5

        # Convert to Metal buffers
        A_buf = mps_tensor_to_metal_buffer(A.contiguous(), self._device)
        B_buf = mps_tensor_to_metal_buffer(B_packed.contiguous(), self._device)
        scales_buf = mps_tensor_to_metal_buffer(scales.contiguous(), self._device)
        C_buf = mps_tensor_to_metal_buffer(C.contiguous(), self._device)

        # Parameters buffer
        params = np.array([M, N, K, group_size], dtype=np.uint32)
        params_buf = self._device.newBufferWithBytes_length_options_(
            params.tobytes(), params.nbytes, Metal.MTLResourceStorageModeShared
        )

        buffers = [A_buf, B_buf, scales_buf, C_buf, params_buf]

        # Grid and threadgroup dimensions
        grid_x = (N + tile_n - 1) // tile_n
        grid_y = (M + tile_m - 1) // tile_m
        grid = (grid_x, grid_y, 1)

        # Threadgroup dimensions (typically 128 threads = 4 simdgroups)
        simdgroups = threads_per_tg // 32
        threadgroup = (threads_per_tg, 1, 1)

        # Get pipeline profile
        pipeline_profile = self._get_pipeline_profile(function_name)

        # Dispatch with timing
        gpu_time_ms, cpu_time_ms, timing_source = self._dispatch_timed(
            function_name, grid, threadgroup, buffers
        )

        # Compute metrics
        ai, flops, bytes_moved = self._compute_arithmetic_intensity(M, N, K, bits, group_size)

        timing_s = gpu_time_ms / 1000.0
        tflops = (flops / timing_s) / 1e12 if timing_s > 0 else 0.0
        memory_gb_s = (bytes_moved / timing_s) / 1e9 if timing_s > 0 else 0.0

        # Utilization percentages
        caps = self._capabilities
        compute_util = (tflops / caps.peak_fp16_tflops) * 100 if caps.peak_fp16_tflops > 0 else 0.0
        memory_util = (memory_gb_s / caps.memory_bandwidth_gb_s) * 100 if caps.memory_bandwidth_gb_s > 0 else 0.0

        # Occupancy
        occupancy = (threads_per_tg / pipeline_profile.max_total_threads) * 100

        # Bottleneck analysis
        bottleneck, roofline_attainment = self._analyze_bottleneck(ai, tflops, memory_gb_s)

        total_threadgroups = grid_x * grid_y
        total_threads = total_threadgroups * threads_per_tg

        return KernelExecutionProfile(
            name=function_name,
            M=M,
            N=N,
            K=K,
            bits=bits,
            group_size=group_size,
            gpu_time_ms=gpu_time_ms,
            cpu_time_ms=cpu_time_ms,
            timing_source=timing_source,
            tflops=tflops,
            memory_gb_s=memory_gb_s,
            compute_utilization_pct=compute_util,
            memory_utilization_pct=memory_util,
            threadgroup_size=threadgroup,
            grid_size=grid,
            total_threads=total_threads,
            total_threadgroups=total_threadgroups,
            threads_per_threadgroup=threads_per_tg,
            simdgroups_per_threadgroup=simdgroups,
            occupancy_pct=occupancy,
            arithmetic_intensity=ai,
            bottleneck=bottleneck,
            roofline_attainment_pct=roofline_attainment,
        )

    def profile_kernel_sweep(
        self,
        function_name: str,
        configs: list[tuple[int, int, int]],
        bits: int = 4,
        group_size: int = 128,
        tile_m: int = 64,
        tile_n: int = 64,
        threads_per_tg: int = 128,
    ) -> KernelProfileSummary:
        """Profile a kernel across multiple (M, N, K) configurations.

        Args:
            function_name: Metal kernel function name
            configs: List of (M, N, K) tuples
            bits: Weight quantization bits
            group_size: Quantization group size
            tile_m, tile_n: Output tile dimensions
            threads_per_tg: Threads per threadgroup

        Returns:
            KernelProfileSummary with aggregate statistics
        """
        pipeline_profile = self._get_pipeline_profile(function_name)
        profiles: list[KernelExecutionProfile] = []

        for M, N, K in configs:
            try:
                profile = self.profile_gemm_kernel(
                    function_name=function_name,
                    M=M,
                    N=N,
                    K=K,
                    bits=bits,
                    group_size=group_size,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    threads_per_tg=threads_per_tg,
                )
                profiles.append(profile)
            except Exception as e:
                print(f"Warning: Failed to profile {function_name} M={M} N={N} K={K}: {e}")

        # Aggregate statistics
        if profiles:
            avg_compute = statistics.mean(p.compute_utilization_pct for p in profiles)
            avg_memory = statistics.mean(p.memory_utilization_pct for p in profiles)
            avg_occupancy = statistics.mean(p.occupancy_pct for p in profiles)
        else:
            avg_compute = avg_memory = avg_occupancy = 0.0

        bottleneck_dist: dict[str, int] = {"memory": 0, "compute": 0, "balanced": 0}
        for p in profiles:
            bottleneck_dist[p.bottleneck] = bottleneck_dist.get(p.bottleneck, 0) + 1

        return KernelProfileSummary(
            kernel_name=function_name,
            device=self._capabilities,
            pipeline=pipeline_profile,
            profiles=profiles,
            avg_compute_utilization=avg_compute,
            avg_memory_utilization=avg_memory,
            avg_occupancy=avg_occupancy,
            bottleneck_distribution=bottleneck_dist,
        )

    def print_device_info(self) -> None:
        """Print device capabilities."""
        caps = self._capabilities
        print("=" * 72)
        print("Metal Device Capabilities")
        print("=" * 72)
        print(f"  Device:                    {caps.name}")
        print(f"  GPU Family:                Apple{caps.gpu_family}")
        print(f"  Max Threads/Threadgroup:   {caps.max_threads_per_threadgroup}")
        print(f"  Thread Execution Width:    {caps.thread_execution_width} (SIMD)")
        print(f"  Max Threadgroup Memory:    {caps.max_threadgroup_memory_bytes / 1024:.1f} KB")
        print(f"  Recommended Working Set:   {caps.recommended_working_set_gb:.1f} GB")
        print(f"  Peak FP16 (estimated):     {caps.peak_fp16_tflops:.1f} TFLOPS")
        print(f"  Peak FP32 (estimated):     {caps.peak_fp32_tflops:.1f} TFLOPS")
        print(f"  Memory Bandwidth (est):    {caps.memory_bandwidth_gb_s:.0f} GB/s")
        print()

    def print_pipeline_info(self, function_name: str) -> None:
        """Print pipeline state properties."""
        try:
            p = self._get_pipeline_profile(function_name)
        except KeyError:
            print(f"Kernel '{function_name}' not found")
            return

        print(f"Pipeline: {p.function_name}")
        print(f"  Max Total Threads:         {p.max_total_threads}")
        print(f"  Thread Execution Width:    {p.thread_execution_width}")
        print(f"  Static Threadgroup Memory: {p.static_threadgroup_memory_bytes} bytes")
        print(f"  Max Simdgroups:            {p.max_simdgroups}")
        print()

    def print_profile(self, profile: KernelExecutionProfile) -> None:
        """Print a single kernel execution profile."""
        p = profile
        print(f"\n--- {p.name} [{p.M}x{p.N}x{p.K}] {p.bits}-bit ---")
        print(f"  Timing:            {p.gpu_time_ms:.3f} ms (GPU), {p.cpu_time_ms:.3f} ms (CPU)")
        print(f"  Throughput:        {p.tflops:.2f} TFLOPS, {p.memory_gb_s:.1f} GB/s")
        print(f"  Compute Util:      {p.compute_utilization_pct:.1f}%")
        print(f"  Memory Util:       {p.memory_utilization_pct:.1f}%")
        print(f"  Occupancy:         {p.occupancy_pct:.1f}% ({p.threads_per_threadgroup}/{self._capabilities.max_threads_per_threadgroup} threads)")
        print(f"  Simdgroups/TG:     {p.simdgroups_per_threadgroup}")
        print(f"  Grid:              {p.grid_size[0]}x{p.grid_size[1]}x{p.grid_size[2]} ({p.total_threadgroups} threadgroups)")
        print(f"  Arith. Intensity:  {p.arithmetic_intensity:.2f} FLOP/byte")
        print(f"  Bottleneck:        {p.bottleneck.upper()} bound")
        print(f"  Roofline Attain:   {p.roofline_attainment_pct:.1f}%")

    def print_summary(self, summary: KernelProfileSummary) -> None:
        """Print kernel profile summary."""
        s = summary
        print("\n" + "=" * 72)
        print(f"Kernel Summary: {s.kernel_name}")
        print("=" * 72)
        print(f"  Avg Compute Utilization:   {s.avg_compute_utilization:.1f}%")
        print(f"  Avg Memory Utilization:    {s.avg_memory_utilization:.1f}%")
        print(f"  Avg Occupancy:             {s.avg_occupancy:.1f}%")
        print("  Bottleneck Distribution:")
        for bn_type, count in s.bottleneck_distribution.items():
            pct = (count / len(s.profiles)) * 100 if s.profiles else 0
            print(f"    {bn_type:12s}: {count:3d} ({pct:.0f}%)")

        # Identify optimization opportunities
        print("\n  Optimization Opportunities:")
        if s.avg_compute_utilization < 50:
            print("    - Low compute utilization suggests insufficient parallelism")
            print("      Consider larger tile sizes or more simdgroups per threadgroup")
        if s.avg_memory_utilization < 50:
            print("    - Low memory utilization suggests memory access inefficiency")
            print("      Check coalescing, consider prefetching or double-buffering")
        if s.avg_occupancy < 50:
            print("    - Low occupancy limits latency hiding")
            print("      Reduce register pressure or threadgroup memory usage")

        # Print detailed results table
        print("\n  Detailed Results:")
        print(f"  {'M':>6} {'N':>6} {'K':>6} {'Time(ms)':>10} {'TFLOPS':>8} "
              f"{'GB/s':>8} {'Comp%':>6} {'Mem%':>6} {'Occ%':>6} {'Bound':>8}")
        print("  " + "-" * 80)
        for p in s.profiles:
            print(f"  {p.M:>6} {p.N:>6} {p.K:>6} {p.gpu_time_ms:>10.3f} "
                  f"{p.tflops:>8.2f} {p.memory_gb_s:>8.1f} "
                  f"{p.compute_utilization_pct:>6.1f} {p.memory_utilization_pct:>6.1f} "
                  f"{p.occupancy_pct:>6.1f} {p.bottleneck:>8}")


def _standard_gemm_configs() -> list[tuple[int, int, int]]:
    """Standard GEMM configurations for LLM inference."""
    return [
        # Decode (M=1)
        (1, 4096, 4096),
        (1, 4096, 14336),
        (1, 14336, 4096),
        # Small batch
        (4, 4096, 4096),
        (4, 4096, 14336),
        # Medium batch
        (32, 4096, 4096),
        (32, 4096, 14336),
        # Prefill
        (128, 4096, 4096),
        (512, 4096, 4096),
        (2048, 4096, 4096),
    ]


def _available_gemm_kernels() -> list[str]:
    """List available GEMM kernel function names."""
    return [
        "marlin_gemm_fp4",
        "marlin_gemm_fp4_double_buffered",
        "marlin_gemm_fp4_pipelined",
        "marlin_gemm_u4",
        "gemm_trellis_packed",
        "gemm_trellis_packed_decode",
    ]


def export_profiles_json(
    summary: KernelProfileSummary,
    output_path: Path,
) -> None:
    """Export profile summary to JSON."""

    def serialize(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: serialize(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serialize(summary), f, indent=2)
    print(f"\nExported profile to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Metal kernel occupancy and utilization"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="marlin_gemm_fp4",
        help="Kernel function name to profile",
    )
    parser.add_argument(
        "--all-kernels",
        action="store_true",
        help="Profile all available GEMM kernels",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Quantization bits",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size",
    )
    parser.add_argument(
        "--tile-m",
        type=int,
        default=64,
        help="Output tile M dimension",
    )
    parser.add_argument(
        "--tile-n",
        type=int,
        default=64,
        help="Output tile N dimension",
    )
    parser.add_argument(
        "--threads-per-tg",
        type=int,
        default=128,
        help="Threads per threadgroup",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Timed iterations",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--single-config",
        type=str,
        default=None,
        help="Single M,N,K config (e.g., '128,4096,4096')",
    )
    args = parser.parse_args()

    if not HAS_METAL or not HAS_MPS:
        print("Error: Metal profiling requires PyObjC Metal and PyTorch MPS")
        print("Install with: pip install pyobjc-framework-Metal torch")
        sys.exit(1)

    profiler = MetalKernelProfiler(
        warmup=args.warmup,
        iterations=args.iterations,
        use_gpu_timestamps=True,
    )

    profiler.print_device_info()

    # Determine configs
    if args.single_config:
        M, N, K = map(int, args.single_config.split(","))
        configs = [(M, N, K)]
    else:
        configs = _standard_gemm_configs()

    # Profile kernels
    kernels = _available_gemm_kernels() if args.all_kernels else [args.kernel]

    for kernel_name in kernels:
        profiler.print_pipeline_info(kernel_name)

        try:
            summary = profiler.profile_kernel_sweep(
                function_name=kernel_name,
                configs=configs,
                bits=args.bits,
                group_size=args.group_size,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                threads_per_tg=args.threads_per_tg,
            )
            profiler.print_summary(summary)

            if args.export_json and not args.all_kernels:
                export_profiles_json(summary, args.export_json)
            elif args.export_json and args.all_kernels:
                # Export each kernel separately
                output_path = args.export_json.parent / f"{kernel_name}_profile.json"
                export_profiles_json(summary, output_path)

        except KeyError as e:
            print(f"Kernel '{kernel_name}' not found: {e}")
            continue
        except Exception as e:
            print(f"Failed to profile '{kernel_name}': {e}")
            import traceback
import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

            traceback.print_exc()
            continue

    # Print Instruments usage guide
    print("\n" + "=" * 72)
    print("For detailed hardware profiling, use Instruments.app:")
    print("=" * 72)
    print("""
1. Metal System Trace (GPU timeline, kernel dispatch):
   xcrun xctrace record --template 'Metal System Trace' --output trace.trace \\
       --launch -- python your_script.py

2. GPU Counters (hardware performance counters):
   - Open Instruments.app
   - Choose 'GPU Counters' template
   - Select metrics: ALU Utilization, Memory Bandwidth, Occupancy
   - Profile your application

3. Metal Debugger (Xcode GPU Frame Capture):
   - Set METAL_DEVICE_WRAPPER_TYPE=1 environment variable
   - Run app from Xcode with GPU Frame Capture enabled
   - Inspect per-kernel statistics

4. Metal Performance HUD (real-time overlay):
   export MTL_HUD_ENABLED=1
   python your_script.py

Key Instruments metrics for kernel optimization:
- ALU Utilization: Should be >80% for compute-bound kernels
- Memory Bandwidth: Compare to theoretical peak
- Occupancy: Active threads vs max threads
- Thread Utilization: Work distribution across simdgroups
- Shader Invocations: Total threads dispatched
""")


if __name__ == "__main__":
    main()
