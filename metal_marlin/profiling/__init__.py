"""Profiling and memory analysis tools for Metal Marlin kernels.

This package provides GPU profiling capabilities including:
- MetalProfiler: GPU timestamp-based kernel profiling with Chrome trace export
- GPUCounters: Hardware performance counter access
- MemoryAuditor: Memory access pattern analysis
- OccupancyAnalyzer: GPU occupancy optimization
"""

from .gpu_counters import (
    GPUCounters,
    GPUPerformanceState,
    GPUProfiler,
    MetalSystemTraceRecorder,
    get_gpu_info,
    read_gpu_counters,
    read_gpu_performance_state,
)
from .memory_audit import (
    AccessPattern,
    MemoryAuditor,
    MemoryAuditReport,
    ShaderAnalysis,
    analyze_shader_file,
    run_coalescing_benchmark,
)
from .memory_bandwidth import (
    BandwidthMeasurement,
    MemoryBandwidthProfiler,
    analyze_bandwidth_bottleneck,
    benchmark_peak_bandwidth,
    estimate_gemm_bytes,
    measure_bandwidth,
)
from .occupancy import (
    AppleSiliconGPU,
    OccupancyAnalyzer,
    OccupancyMetrics,
    ThreadgroupConfig,
    detect_gpu,
    estimate_optimal_config,
)
from .profiler import (
    KernelCapture,
    MetalProfileSession,
    Profiler,
    bench,
    clear_global_profiles,
    get_global_profiler,
    profile_kernel,
)
from .roofline import (
    KernelPoint,
    RooflineAnalyzer,
    RooflineConfig,
    quick_roofline,
)
from .metal_profiler import MetalProfiler, ProfileEvent, ProfileRegion
from .trace import ChromeTrace, TraceEvent

__all__ = [
    "AccessPattern",
    "MetalProfiler",
    "ProfileEvent",
    "ProfileRegion",
    "AppleSiliconGPU",
    "BandwidthMeasurement",
    "ChromeTrace",
    "GPUCounters",
    "GPUPerformanceState",
    "GPUProfiler",
    "KernelCapture",
    "KernelPoint",
    "MemoryAuditReport",
    "MemoryAuditor",
    "MemoryBandwidthProfiler",
    "MetalProfileSession",
    "MetalSystemTraceRecorder",
    "OccupancyAnalyzer",
    "OccupancyMetrics",
    "Profiler",
    "RooflineAnalyzer",
    "RooflineConfig",
    "ShaderAnalysis",
    "ThreadgroupConfig",
    "TraceEvent",
    "analyze_bandwidth_bottleneck",
    "analyze_shader_file",
    "bench",
    "benchmark_peak_bandwidth",
    "clear_global_profiles",
    "detect_gpu",
    "estimate_gemm_bytes",
    "estimate_optimal_config",
    "get_global_profiler",
    "get_gpu_info",
    "measure_bandwidth",
    "profile_kernel",
    "quick_roofline",
    "read_gpu_counters",
    "read_gpu_performance_state",
    "run_coalescing_benchmark",
]
