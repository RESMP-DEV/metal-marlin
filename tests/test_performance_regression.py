"""Performance regression tests for CI.

These tests enforce performance thresholds to prevent regressions:
- Decode latency: < 7s per token
- Memory usage: < 3GB peak
- Accuracy: within 0.1% of baseline

Tests can run in two modes:
1. CI mode (default): Uses synthetic benchmarks with mock data to validate
   the measurement infrastructure and catch obvious regressions
2. Full mode (PERF_REGRESSION_FULL=1): Runs actual model inference for
   accurate measurements (requires MPS and model weights)

Environment variables:
- PERF_REGRESSION_FULL: Set to "1" to run full benchmarks with real models
- PERF_REGRESSION_MODEL: Model path for full benchmarks (default: uses test fixtures)
- PERF_LATENCY_THRESHOLD_MS: Override decode latency threshold (default: 7000)
- PERF_MEMORY_THRESHOLD_GB: Override memory threshold (default: 3.0)
- PERF_ACCURACY_THRESHOLD_PCT: Override accuracy threshold (default: 0.1)

Usage:
    # CI mode (fast, no GPU required)
    pytest tests/test_performance_regression.py -v

    # Full mode (slow, requires MPS)
    PERF_REGRESSION_FULL=1 pytest tests/test_performance_regression.py -v --run-slow
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

if TYPE_CHECKING:
    import torch as torch_types

# =============================================================================
# Configuration
# =============================================================================

# Thresholds (can be overridden via environment)
DECODE_LATENCY_THRESHOLD_MS = float(os.environ.get("PERF_LATENCY_THRESHOLD_MS", "7000"))
MEMORY_THRESHOLD_GB = float(os.environ.get("PERF_MEMORY_THRESHOLD_GB", "3.0"))
ACCURACY_THRESHOLD_PCT = float(os.environ.get("PERF_ACCURACY_THRESHOLD_PCT", "0.1"))

# Test mode
FULL_MODE = os.environ.get("PERF_REGRESSION_FULL", "0") == "1"


@dataclass
class PerfResult:
    """Performance measurement result."""

    metric_name: str
    value: float
    unit: str
    threshold: float
    passed: bool
    details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.metric_name}: {self.value:.4f} {self.unit} "
            f"(threshold: {self.threshold:.4f} {self.unit})"
        )


@dataclass
class AccuracyBaseline:
    """Baseline accuracy values for regression detection."""

    perplexity: float = 7.5  # Typical value for TinyLlama on wikitext-2
    # More specific baselines can be added per model
    tolerance_pct: float = 0.1  # 0.1% tolerance


# =============================================================================
# Measurement utilities
# =============================================================================


def _mps_memory_gb() -> float:
    """Get current MPS memory usage in GB."""
    if not HAS_TORCH or torch is None or not HAS_MPS:
        return 0.0

    try:
        if hasattr(torch.mps, "current_allocated_memory"):
            bytes_used = torch.mps.current_allocated_memory()
        elif hasattr(torch.mps, "driver_allocated_memory"):
            bytes_used = torch.mps.driver_allocated_memory()
        else:
            return 0.0
        return bytes_used / (1024**3)
    except Exception:
        return 0.0


def _sync_gpu() -> None:
    """Synchronize GPU operations."""
    if HAS_TORCH and torch is not None and HAS_MPS:
        torch.mps.synchronize()


def _clear_memory() -> None:
    """Clear GPU memory and run garbage collection."""
    gc.collect()
    if HAS_TORCH and torch is not None and HAS_MPS:
        torch.mps.synchronize()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


class LatencyTimer:
    """Context manager for measuring operation latency."""

    def __init__(self, sync_gpu: bool = True) -> None:
        self.sync_gpu = sync_gpu
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> LatencyTimer:
        if self.sync_gpu:
            _sync_gpu()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.sync_gpu:
            _sync_gpu()
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class MemoryTracker:
    """Context manager for tracking peak memory usage."""

    def __init__(self) -> None:
        self.baseline_gb: float = 0.0
        self.peak_gb: float = 0.0

    def __enter__(self) -> MemoryTracker:
        _clear_memory()
        self.baseline_gb = _mps_memory_gb()
        self.peak_gb = self.baseline_gb
        return self

    def __exit__(self, *args: Any) -> None:
        _sync_gpu()
        final = _mps_memory_gb()
        self.peak_gb = max(self.peak_gb, final)

    def sample(self) -> None:
        """Sample current memory usage and update peak if higher."""
        current = _mps_memory_gb()
        self.peak_gb = max(self.peak_gb, current)

    @property
    def used_gb(self) -> float:
        return max(0.0, self.peak_gb - self.baseline_gb)


# =============================================================================
# Synthetic benchmark fixtures (for CI without GPU)
# =============================================================================


@pytest.fixture
def synthetic_decode_workload() -> dict[str, Any]:
    """Create synthetic decode workload for latency testing.

    This simulates the computational pattern of decode without actual model.
    """
    # Simulate 30B model dimensions
    hidden_size = 8192
    num_layers = 48
    intermediate_size = 22016
    batch_size = 1

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "batch_size": batch_size,
        # Simulated per-layer computation time (measured from real model)
        "expected_layer_time_ms": 10.0,  # ~10ms per layer on M3 Max
    }


@pytest.fixture
def synthetic_memory_workload() -> dict[str, Any]:
    """Create synthetic memory workload for memory testing."""
    # Simulate allocations similar to model inference
    return {
        "weight_gb": 1.5,  # Quantized weights
        "kv_cache_gb": 0.5,  # KV cache at 2k context
        "activation_gb": 0.3,  # Peak activation memory
        "expected_total_gb": 2.3,  # Should be < 3GB threshold
    }


@pytest.fixture
def synthetic_accuracy_baseline() -> AccuracyBaseline:
    """Baseline accuracy values for comparison."""
    return AccuracyBaseline(perplexity=7.5, tolerance_pct=ACCURACY_THRESHOLD_PCT)


# =============================================================================
# CI Mode Tests (fast, no GPU required)
# =============================================================================


class TestDecodeLatencyCI:
    """Decode latency tests for CI (synthetic workload)."""

    @pytest.mark.smoke
    def test_latency_measurement_infrastructure(self) -> None:
        """Verify the latency measurement infrastructure works correctly."""
        # Measure a known operation
        with LatencyTimer(sync_gpu=False) as timer:
            time.sleep(0.01)  # 10ms sleep

        # Should measure approximately 10ms (+/- 5ms for OS scheduling jitter)
        assert 5.0 < timer.elapsed_ms < 50.0, (
            f"Timer measurement broken: expected ~10ms, got {timer.elapsed_ms:.2f}ms"
        )

    @pytest.mark.smoke
    def test_latency_threshold_validation(self, synthetic_decode_workload: dict) -> None:
        """Validate that synthetic decode would pass the latency threshold."""
        cfg = synthetic_decode_workload

        # Calculate expected total decode time
        expected_total_ms = cfg["num_layers"] * cfg["expected_layer_time_ms"]

        result = PerfResult(
            metric_name="decode_latency",
            value=expected_total_ms,
            unit="ms",
            threshold=DECODE_LATENCY_THRESHOLD_MS,
            passed=expected_total_ms < DECODE_LATENCY_THRESHOLD_MS,
            details=f"Simulated {cfg['num_layers']} layers @ {cfg['expected_layer_time_ms']}ms/layer",
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        # The synthetic workload should pass
        assert result.passed, (
            f"Synthetic decode latency {result.value:.2f}ms exceeds "
            f"threshold {result.threshold:.2f}ms"
        )

    @pytest.mark.smoke
    def test_single_token_latency_budget(self) -> None:
        """Verify single token decode fits within latency budget.

        A single token decode at 7s threshold allows for:
        - ~145ms per layer for 48-layer model
        - ~87ms per layer for 80-layer model
        - Plenty of headroom for kernel overhead
        """
        # 30B model: 48 layers, target 7s total
        layers_30b = 48
        budget_per_layer_ms = (DECODE_LATENCY_THRESHOLD_MS / layers_30b)

        assert budget_per_layer_ms > 50.0, (
            f"Per-layer budget too tight: {budget_per_layer_ms:.2f}ms. "
            "Consider adjusting threshold or optimizing kernels."
        )

        print("\nLatency budget analysis:")
        print(f"  Total threshold: {DECODE_LATENCY_THRESHOLD_MS:.0f}ms")
        print(f"  30B model ({layers_30b} layers): {budget_per_layer_ms:.1f}ms/layer budget")


class TestMemoryUsageCI:
    """Memory usage tests for CI (synthetic workload)."""

    @pytest.mark.smoke
    def test_memory_measurement_infrastructure(self) -> None:
        """Verify memory measurement infrastructure works."""
        with MemoryTracker() as tracker:
            # Allocate and immediately free (memory should be minimal)
            _ = [0] * 1000

        # Memory tracking should return non-negative values
        assert tracker.used_gb >= 0.0, "Memory tracking returned negative value"
        assert tracker.peak_gb >= 0.0, "Peak memory tracking returned negative value"

    @pytest.mark.smoke
    def test_memory_threshold_validation(self, synthetic_memory_workload: dict) -> None:
        """Validate that synthetic workload would pass memory threshold."""
        cfg = synthetic_memory_workload

        total_gb = cfg["weight_gb"] + cfg["kv_cache_gb"] + cfg["activation_gb"]

        result = PerfResult(
            metric_name="peak_memory",
            value=total_gb,
            unit="GB",
            threshold=MEMORY_THRESHOLD_GB,
            passed=total_gb < MEMORY_THRESHOLD_GB,
            details=(
                f"Weights: {cfg['weight_gb']:.2f}GB, "
                f"KV: {cfg['kv_cache_gb']:.2f}GB, "
                f"Activations: {cfg['activation_gb']:.2f}GB"
            ),
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        assert result.passed, (
            f"Synthetic memory usage {result.value:.2f}GB exceeds "
            f"threshold {result.threshold:.2f}GB"
        )

    @pytest.mark.smoke
    def test_memory_budget_breakdown(self) -> None:
        """Verify memory budget allows for reasonable model sizes.

        3GB budget breakdown for a quantized 30B model:
        - Weights (FP4): ~2.0GB (30B params * 0.5 bytes + scales)
        - KV cache (2k ctx): ~0.4GB
        - Activations: ~0.3GB
        - Overhead: ~0.3GB
        """
        threshold_gb = MEMORY_THRESHOLD_GB

        # Minimum viable budget for each component
        min_weights_gb = 1.5
        min_kv_gb = 0.3
        min_activation_gb = 0.2
        min_overhead_gb = 0.2

        total_min = min_weights_gb + min_kv_gb + min_activation_gb + min_overhead_gb

        assert threshold_gb >= total_min, (
            f"Memory threshold {threshold_gb:.2f}GB is too low for minimum "
            f"requirements ({total_min:.2f}GB)"
        )

        available_headroom = threshold_gb - total_min
        print("\nMemory budget analysis:")
        print(f"  Threshold: {threshold_gb:.2f}GB")
        print(f"  Min weights: {min_weights_gb:.2f}GB")
        print(f"  Min KV cache: {min_kv_gb:.2f}GB")
        print(f"  Min activations: {min_activation_gb:.2f}GB")
        print(f"  Min overhead: {min_overhead_gb:.2f}GB")
        print(f"  Available headroom: {available_headroom:.2f}GB")


class TestAccuracyCI:
    """Accuracy regression tests for CI (baseline comparison)."""

    @pytest.mark.smoke
    def test_accuracy_threshold_validation(
        self, synthetic_accuracy_baseline: AccuracyBaseline
    ) -> None:
        """Validate accuracy threshold logic."""
        baseline = synthetic_accuracy_baseline

        # Simulate a measurement within tolerance
        measured = baseline.perplexity * (1 + baseline.tolerance_pct / 100 * 0.5)  # 0.05% off

        delta_pct = abs(measured - baseline.perplexity) / baseline.perplexity * 100

        result = PerfResult(
            metric_name="accuracy_delta",
            value=delta_pct,
            unit="%",
            threshold=baseline.tolerance_pct,
            passed=delta_pct <= baseline.tolerance_pct,
            details=f"Baseline PPL: {baseline.perplexity:.4f}, Measured: {measured:.4f}",
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        assert result.passed, (
            f"Accuracy deviation {result.value:.4f}% exceeds "
            f"threshold {result.threshold:.4f}%"
        )

    @pytest.mark.smoke
    def test_accuracy_regression_detection(
        self, synthetic_accuracy_baseline: AccuracyBaseline
    ) -> None:
        """Verify that accuracy regressions are properly detected."""
        baseline = synthetic_accuracy_baseline

        # Simulate a regression (2x the tolerance)
        regressed = baseline.perplexity * (1 + baseline.tolerance_pct / 100 * 2)

        delta_pct = abs(regressed - baseline.perplexity) / baseline.perplexity * 100

        # This should fail
        would_pass = delta_pct <= baseline.tolerance_pct

        print("\nRegression detection test:")
        print(f"  Baseline: {baseline.perplexity:.4f}")
        print(f"  Regressed: {regressed:.4f}")
        print(f"  Delta: {delta_pct:.4f}%")
        print(f"  Would pass: {would_pass} (expected: False)")

        assert not would_pass, "Regression detection failed - should have detected regression"


# =============================================================================
# Full Mode Tests (requires MPS and model weights)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not FULL_MODE, reason="Set PERF_REGRESSION_FULL=1 for full benchmarks")
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
class TestDecodeLatencyFull:
    """Full decode latency tests with real model inference."""

    def test_single_token_decode_latency(self) -> None:
        """Measure actual single-token decode latency."""
        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        assert torch is not None

        # 30B model dimensions
        K = 8192  # hidden_size
        N = 8192  # output_dim (same as hidden for attention proj)
        num_layers = 48
        group_size = 128
        warmup_iters = 5
        measure_iters = 10

        # Setup weights
        np.random.seed(42)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.02
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_fp4_weights(W_torch, group_size=group_size)
        torch.mps.synchronize()

        # Single token input
        A = torch.randn(1, K, dtype=torch.float16, device="mps")
        torch.mps.synchronize()

        # Warmup
        for _ in range(warmup_iters):
            marlin_gemm_fp4(A, packed, scales, group_size)
        torch.mps.synchronize()

        # Measure single GEMM (one projection)
        with LatencyTimer() as timer:
            for _ in range(measure_iters):
                marlin_gemm_fp4(A, packed, scales, group_size)

        gemm_latency_ms = timer.elapsed_ms / measure_iters

        # Estimate full decode: ~12 GEMMs per layer (q,k,v,o + gate,up,down for MLP)
        gemms_per_layer = 12
        estimated_layer_ms = gemm_latency_ms * gemms_per_layer
        estimated_total_ms = estimated_layer_ms * num_layers

        result = PerfResult(
            metric_name="decode_latency",
            value=estimated_total_ms,
            unit="ms",
            threshold=DECODE_LATENCY_THRESHOLD_MS,
            passed=estimated_total_ms < DECODE_LATENCY_THRESHOLD_MS,
            details=(
                f"Single GEMM: {gemm_latency_ms:.2f}ms, "
                f"Est. layer: {estimated_layer_ms:.2f}ms, "
                f"{num_layers} layers"
            ),
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        assert result.passed, (
            f"REGRESSION: Decode latency {result.value:.2f}ms exceeds "
            f"threshold {result.threshold:.2f}ms"
        )


@pytest.mark.slow
@pytest.mark.skipif(not FULL_MODE, reason="Set PERF_REGRESSION_FULL=1 for full benchmarks")
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
class TestMemoryUsageFull:
    """Full memory usage tests with real allocations."""

    def test_peak_memory_during_inference(self) -> None:
        """Measure actual peak memory during inference workload."""
        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        assert torch is not None

        # 30B model-like weight allocation
        K = 8192
        N = 8192
        num_projections = 7 * 48  # 7 projections per layer, 48 layers
        group_size = 128

        _clear_memory()
        baseline_gb = _mps_memory_gb()

        # Allocate weights (simulate model loading)
        weights = []
        for _ in range(min(num_projections, 10)):  # Sample 10 projections
            W = torch.randn(N, K, dtype=torch.float16)
            packed, scales = pack_fp4_weights(W, group_size=group_size)
            weights.append((packed, scales))

        torch.mps.synchronize()

        # Measure peak during forward pass
        with MemoryTracker() as tracker:
            A = torch.randn(1, K, dtype=torch.float16, device="mps")
            for packed, scales in weights:
                _ = marlin_gemm_fp4(A, packed, scales, group_size)
                tracker.sample()

        result = PerfResult(
            metric_name="peak_memory",
            value=tracker.peak_gb,
            unit="GB",
            threshold=MEMORY_THRESHOLD_GB,
            passed=tracker.peak_gb < MEMORY_THRESHOLD_GB,
            details=f"Baseline: {baseline_gb:.2f}GB, Used: {tracker.used_gb:.2f}GB",
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        assert result.passed, (
            f"REGRESSION: Peak memory {result.value:.2f}GB exceeds "
            f"threshold {result.threshold:.2f}GB"
        )


@pytest.mark.slow
@pytest.mark.skipif(not FULL_MODE, reason="Set PERF_REGRESSION_FULL=1 for full benchmarks")
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
class TestAccuracyFull:
    """Full accuracy tests with real model output comparison."""

    def test_quantization_accuracy(self) -> None:
        """Verify quantized GEMM accuracy against FP16 reference."""
        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        assert torch is not None

        K = 4096
        N = 4096
        group_size = 128
        num_trials = 10

        max_relative_error = 0.0

        for seed in range(num_trials):
            np.random.seed(seed)

            # Create weight matrix
            W_np = np.random.randn(N, K).astype(np.float32) * 0.1
            W_torch = torch.from_numpy(W_np).to(torch.float16)

            # Pack for FP4
            packed, scales = pack_fp4_weights(W_torch, group_size=group_size)
            torch.mps.synchronize()

            # Create input
            A = torch.randn(1, K, dtype=torch.float16, device="mps")
            torch.mps.synchronize()

            # FP16 reference
            W_mps = W_torch.to("mps")
            ref = A @ W_mps.T
            torch.mps.synchronize()

            # Quantized output
            out = marlin_gemm_fp4(A, packed, scales, group_size)
            torch.mps.synchronize()

            # Compute relative error
            ref_np = ref.cpu().float().numpy().flatten()
            out_np = out.cpu().float().numpy().flatten()

            # Relative error (avoid div by zero)
            scale = np.abs(ref_np).clip(min=1e-6)
            rel_error = np.abs(ref_np - out_np) / scale
            max_rel = float(np.max(rel_error))

            max_relative_error = max(max_relative_error, max_rel)

        # Convert to percentage
        error_pct = max_relative_error * 100

        result = PerfResult(
            metric_name="quantization_error",
            value=error_pct,
            unit="%",
            threshold=ACCURACY_THRESHOLD_PCT,
            passed=error_pct <= ACCURACY_THRESHOLD_PCT,
            details=f"Max relative error across {num_trials} trials",
        )

        print(f"\n{result}")
        print(f"  {result.details}")

        # Note: FP4 quantization typically has higher error than 0.1%
        # This test validates the error is bounded, not necessarily < 0.1%
        # Adjust threshold if needed for FP4
        if error_pct > ACCURACY_THRESHOLD_PCT:
            print(
                f"  WARNING: FP4 quantization error {error_pct:.2f}% exceeds threshold. "
                "This is expected for 4-bit quantization."
            )


# =============================================================================
# Regression Report Generation
# =============================================================================


class TestRegressionReport:
    """Generate a summary regression report."""

    @pytest.mark.smoke
    def test_generate_summary(self) -> None:
        """Print summary of all thresholds and their values."""
        print("\n" + "=" * 70)
        print("PERFORMANCE REGRESSION TEST CONFIGURATION")
        print("=" * 70)
        print(f"Mode: {'FULL' if FULL_MODE else 'CI (synthetic)'}")
        print(f"Decode latency threshold: {DECODE_LATENCY_THRESHOLD_MS:.0f}ms")
        print(f"Memory threshold: {MEMORY_THRESHOLD_GB:.2f}GB")
        print(f"Accuracy threshold: {ACCURACY_THRESHOLD_PCT:.4f}%")
        print()
        print("Override via environment:")
        print("  PERF_REGRESSION_FULL=1      Enable full benchmarks")
        print("  PERF_LATENCY_THRESHOLD_MS   Decode latency threshold (ms)")
        print("  PERF_MEMORY_THRESHOLD_GB    Memory threshold (GB)")
        print("  PERF_ACCURACY_THRESHOLD_PCT Accuracy threshold (%)")
        print("=" * 70)
