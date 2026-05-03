"""Test fixtures for metal_marlin."""
import logging

from tests.fixtures.synthetic_mixed_moe import (
    BenchmarkResult,
    SyntheticConfig,
    SyntheticMixedMoE,
    benchmark_forward,
    create_synthetic_model,
)


logger = logging.getLogger(__name__)

__all__ = [
    "BenchmarkResult",
    "SyntheticConfig",
    "SyntheticMixedMoE",
    "benchmark_forward",
    "create_synthetic_model",
]
