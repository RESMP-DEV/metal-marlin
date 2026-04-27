"""Benchmark utilities for metal_marlin mixed-precision optimization."""
import logging

from .mixed_precision_bench import (
    BenchmarkConfig,
    BenchmarkResult,
    MixedPrecisionBenchmark,
    run_quick_benchmark,
)


logger = logging.getLogger(__name__)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "MixedPrecisionBenchmark",
    "run_quick_benchmark",
]
