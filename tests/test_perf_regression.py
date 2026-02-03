"""Performance regression test that fails if latency increases >5%.

This test tracks baseline latency values and detects regressions by comparing
current measurements against the baseline. Regressions >5% cause test failure.

Baseline storage:
- File: .perf_baseline.json (git-tracked)
- Format: {"test_name": {"baseline_ms": float, "last_updated": str}}

Environment variables:
- PERF_REGRESSION_THRESHOLD_PCT: Regression threshold (default: 5.0)
- PERF_REGRESSION_UPDATE_BASELINE: Set to "1" to update baseline

Usage:
    # Run test (fails on >5% regression)
    pytest tests/test_perf_regression.py -v

    # Update baseline after intentional optimizations
    PERF_REGRESSION_UPDATE_BASELINE=1 pytest tests/test_perf_regression.py -v
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

if TYPE_CHECKING:
    import torch as torch_types

# Configuration
REGRESSION_THRESHOLD_PCT = float(os.environ.get("PERF_REGRESSION_THRESHOLD_PCT", "5.0"))
UPDATE_BASELINE = os.environ.get("PERF_REGRESSION_UPDATE_BASELINE", "0") == "1"

# Baseline file path (repo root to share across runs)
BASELINE_PATH = Path(__file__).parent.parent.parent / ".perf_baseline.json"


@dataclass
class LatencyBaseline:
    """Baseline latency value for a test."""

    baseline_ms: float
    last_updated: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_ms": self.baseline_ms,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyBaseline:
        return cls(baseline_ms=data["baseline_ms"], last_updated=data["last_updated"])


class BaselineManager:
    """Manages performance baseline storage and retrieval."""

    def __init__(self, path: Path = BASELINE_PATH) -> None:
        self.path = path
        self._baselines: dict[str, LatencyBaseline] = {}
        self._load()

    def _load(self) -> None:
        """Load baselines from file."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self._baselines[key] = LatencyBaseline.from_dict(value)
            except Exception:
                self._baselines = {}

    def _save(self) -> None:
        """Save baselines to file."""
        data = {key: value.to_dict() for key, value in self._baselines.items()}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, test_name: str) -> LatencyBaseline | None:
        """Get baseline for a test."""
        return self._baselines.get(test_name)

    def set(self, test_name: str, latency_ms: float) -> None:
        """Set baseline for a test."""
        from datetime import datetime

        self._baselines[test_name] = LatencyBaseline(
            baseline_ms=latency_ms,
            last_updated=datetime.now().isoformat(),
        )
        self._save()

    def check_regression(
        self, test_name: str, current_ms: float
    ) -> tuple[bool, float, str]:
        """Check if current latency exceeds baseline by threshold.

        Returns:
            (is_regression, increase_pct, message)
        """
        baseline = self.get(test_name)

        if baseline is None:
            # No baseline exists, create it
            self.set(test_name, current_ms)
            return (
                False,
                0.0,
                f"No baseline found. Created baseline: {current_ms:.2f}ms",
            )

        increase_pct = ((current_ms - baseline.baseline_ms) / baseline.baseline_ms) * 100

        if increase_pct > REGRESSION_THRESHOLD_PCT:
            message = (
                f"REGRESSION: Latency increased by {increase_pct:.2f}% "
                f"({baseline.baseline_ms:.2f}ms -> {current_ms:.2f}ms). "
                f"Threshold: {REGRESSION_THRESHOLD_PCT}%"
            )
            return True, increase_pct, message

        if increase_pct < -REGRESSION_THRESHOLD_PCT:
            # Significant improvement - offer to update baseline
            message = (
                f"IMPROVEMENT: Latency decreased by {abs(increase_pct):.2f}% "
                f"({baseline.baseline_ms:.2f}ms -> {current_ms:.2f}ms). "
                f"Consider updating baseline with PERF_REGRESSION_UPDATE_BASELINE=1"
            )
        else:
            message = (
                f"PASS: Latency {current_ms:.2f}ms within "
                f"{abs(increase_pct):.2f}% of baseline {baseline.baseline_ms:.2f}ms. "
                f"Threshold: +/-{REGRESSION_THRESHOLD_PCT}%"
            )

        return False, increase_pct, message


class LatencyTimer:
    """Simple timer for measuring operation latency."""

    def __init__(self, sync_gpu: bool = False) -> None:
        self.sync_gpu = sync_gpu
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> LatencyTimer:
        if self.sync_gpu and HAS_TORCH and torch is not None and HAS_MPS:
            torch.mps.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.sync_gpu and HAS_TORCH and torch is not None and HAS_MPS:
            torch.mps.synchronize()
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@pytest.fixture
def baseline_manager() -> BaselineManager:
    """Provide baseline manager for tests."""
    return BaselineManager()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestLatencyRegression:
    """Performance regression tests that fail on >5% latency increase."""

    @pytest.fixture(autouse=True)
    def setup(self, baseline_manager: BaselineManager) -> None:
        """Auto-use fixture to provide baseline manager."""
        self.manager = baseline_manager

    def test_fp4_gemm_latency_regression(self) -> None:
        """Test FP4 GEMM latency regression.

        Measures FP4 quantized GEMM performance and compares to baseline.
        Fails if latency increases >5%.
        """
        if torch is None:
            pytest.skip("PyTorch not available")

        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        # Test configuration
        K = 4096
        N = 4096
        group_size = 128
        batch_size = 1
        warmup_iters = 5
        measure_iters = 20

        # Setup weights (fixed seed for consistency)
        np.random.seed(42)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_fp4_weights(W_torch, group_size=group_size)

        # Setup input
        A = torch.randn(batch_size, K, dtype=torch.float16, device="mps")

        if HAS_MPS:
            torch.mps.synchronize()

        # Warmup
        for _ in range(warmup_iters):
            marlin_gemm_fp4(A, packed, scales, group_size)

        if HAS_MPS:
            torch.mps.synchronize()

        # Measure
        with LatencyTimer(sync_gpu=True) as timer:
            for _ in range(measure_iters):
                marlin_gemm_fp4(A, packed, scales, group_size)

        avg_latency_ms = timer.elapsed_ms / measure_iters

        # Check regression
        test_name = "fp4_gemm_latency"
        is_regression, increase_pct, message = self.manager.check_regression(
            test_name, avg_latency_ms
        )

        print(f"\n{message}")
        print(f"  Measured: {avg_latency_ms:.2f}ms over {measure_iters} iterations")

        if UPDATE_BASELINE and not is_regression:
            self.manager.set(test_name, avg_latency_ms)
            print(f"  Updated baseline to {avg_latency_ms:.2f}ms")

        if is_regression:
            pytest.fail(message)

    def test_int4_gemm_latency_regression(self) -> None:
        """Test INT4 GEMM latency regression.

        Measures INT4 quantized GEMM performance and compares to baseline.
        Fails if latency increases >5%.
        """
        if torch is None:
            pytest.skip("PyTorch not available")

        from metal_marlin.kernels import marlin_gemm_int4, pack_int4_weights

        # Test configuration
        K = 4096
        N = 4096
        group_size = 128
        batch_size = 1
        warmup_iters = 5
        measure_iters = 20

        # Setup weights
        np.random.seed(42)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_int4_weights(W_torch, group_size=group_size)

        # Setup input
        A = torch.randn(batch_size, K, dtype=torch.float16, device="mps")

        if HAS_MPS:
            torch.mps.synchronize()

        # Warmup
        for _ in range(warmup_iters):
            marlin_gemm_int4(A, packed, scales, group_size)

        if HAS_MPS:
            torch.mps.synchronize()

        # Measure
        with LatencyTimer(sync_gpu=True) as timer:
            for _ in range(measure_iters):
                marlin_gemm_int4(A, packed, scales, group_size)

        avg_latency_ms = timer.elapsed_ms / measure_iters

        # Check regression
        test_name = "int4_gemm_latency"
        is_regression, increase_pct, message = self.manager.check_regression(
            test_name, avg_latency_ms
        )

        print(f"\n{message}")
        print(f"  Measured: {avg_latency_ms:.2f}ms over {measure_iters} iterations")

        if UPDATE_BASELINE and not is_regression:
            self.manager.set(test_name, avg_latency_ms)
            print(f"  Updated baseline to {avg_latency_ms:.2f}ms")

        if is_regression:
            pytest.fail(message)

    def test_large_matrix_gemm_latency_regression(self) -> None:
        """Test large matrix GEMM latency regression.

        Tests a larger matrix size (8192x8192) typical of attention projections
        in 30B+ parameter models. Fails if latency increases >5%.
        """
        if torch is None:
            pytest.skip("PyTorch not available")

        from metal_marlin.kernels import marlin_gemm_fp4, pack_fp4_weights

        # Large matrix size (attention projection in 30B+ models)
        K = 8192
        N = 8192
        group_size = 128
        batch_size = 1
        warmup_iters = 5
        measure_iters = 10

        # Setup weights
        np.random.seed(42)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.02
        W_torch = torch.from_numpy(W_np).to(torch.float16)
        packed, scales = pack_fp4_weights(W_torch, group_size=group_size)

        # Setup input
        A = torch.randn(batch_size, K, dtype=torch.float16, device="mps")

        if HAS_MPS:
            torch.mps.synchronize()

        # Warmup
        for _ in range(warmup_iters):
            marlin_gemm_fp4(A, packed, scales, group_size)

        if HAS_MPS:
            torch.mps.synchronize()

        # Measure
        with LatencyTimer(sync_gpu=True) as timer:
            for _ in range(measure_iters):
                marlin_gemm_fp4(A, packed, scales, group_size)

        avg_latency_ms = timer.elapsed_ms / measure_iters

        # Check regression
        test_name = "large_matrix_gemm_latency"
        is_regression, increase_pct, message = self.manager.check_regression(
            test_name, avg_latency_ms
        )

        print(f"\n{message}")
        print(f"  Matrix size: {K}x{N}")
        print(f"  Measured: {avg_latency_ms:.2f}ms over {measure_iters} iterations")

        if UPDATE_BASELINE and not is_regression:
            self.manager.set(test_name, avg_latency_ms)
            print(f"  Updated baseline to {avg_latency_ms:.2f}ms")

        if is_regression:
            pytest.fail(message)


class TestRegressionConfiguration:
    """Test configuration and utilities."""

    @pytest.mark.smoke
    def test_threshold_configuration(self) -> None:
        """Verify threshold is correctly configured."""
        print(f"\nRegression threshold: {REGRESSION_THRESHOLD_PCT}%")
        assert 0 < REGRESSION_THRESHOLD_PCT < 100, (
            f"Invalid threshold: {REGRESSION_THRESHOLD_PCT}%. "
            "Must be between 0 and 100."
        )

    @pytest.mark.smoke
    def test_baseline_manager(self, tmp_path: Path) -> None:
        """Test baseline manager functionality."""
        baseline_file = tmp_path / "test_baseline.json"
        manager = BaselineManager(baseline_file)

        # Test initial state
        assert manager.get("test") is None, "Should have no initial baseline"

        # Test setting baseline
        manager.set("test", 100.0)
        baseline = manager.get("test")
        assert baseline is not None
        assert baseline.baseline_ms == 100.0

        # Test regression detection
        is_regression, inc_pct, msg = manager.check_regression("test", 100.0)
        assert not is_regression
        assert inc_pct == 0.0
        assert "PASS" in msg

        # Test regression (6% increase)
        is_regression, inc_pct, msg = manager.check_regression("test", 106.0)
        assert is_regression
        assert inc_pct == 6.0
        assert "REGRESSION" in msg

        # Test improvement (10% decrease)
        is_regression, inc_pct, msg = manager.check_regression("test", 90.0)
        assert not is_regression
        assert inc_pct == -10.0
        assert "IMPROVEMENT" in msg

    @pytest.mark.smoke
    def test_latency_timer(self) -> None:
        """Test latency timer accuracy."""
        with LatencyTimer(sync_gpu=False) as timer:
            time.sleep(0.01)

        elapsed = timer.elapsed_ms
        # 10ms sleep +/- 5ms tolerance
        assert 5.0 < elapsed < 50.0, f"Timer inaccurate: {elapsed:.2f}ms"

        print(f"\nLatency timer: {elapsed:.2f}ms for 10ms sleep")
