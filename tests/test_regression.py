"""Regression tests for Metal kernel performance.

These tests ensure that Metal kernel performance doesn't regress after changes.
Specifically targets the MoE dispatch slowness bug that caused 20s+ per layer
before the fix.

Requirements:
- MPS device (Apple Silicon)
- TrellisForCausalLM model loaded from checkpoint

Usage:
    cd contrib/metal_marlin && uv run pytest tests/test_regression.py -v

    # Run with specific model path:
    MODEL_PATH=models/GLM-4.7-Flash-Trellis-3bpw uv run pytest tests/test_regression.py -v
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

from metal_marlin._compat import HAS_MPS, HAS_TORCH

# Default model path (can be overridden via environment)
DEFAULT_MODEL_PATH = "models/GLM-4.7-Flash-Trellis-3bpw"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# Skip if model doesn't exist
MODEL_EXISTS = Path(MODEL_PATH).exists()

# Check for TrellisForCausalLM availability
try:
    from metal_marlin.trellis import TrellisForCausalLM

    HAS_TRELLIS = True
except ImportError:
    HAS_TRELLIS = False


requires_model = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason=f"Model not found at {MODEL_PATH}. Set MODEL_PATH env var or download model.",
)
requires_mps = pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
requires_trellis = pytest.mark.skipif(
    not HAS_TRELLIS, reason="TrellisForCausalLM not available"
)


@pytest.fixture(scope="module")
def model():
    """Load the Trellis model for regression testing.

    This fixture is module-scoped to avoid reloading for each test.
    """
    if not HAS_TRELLIS or not MODEL_EXISTS or not HAS_MPS:
        pytest.skip("Model, Trellis module, or MPS not available")

    return TrellisForCausalLM.from_pretrained(MODEL_PATH, device="mps")


@requires_mps
@requires_trellis
@requires_model
class TestForwardPassRegression:
    """Regression tests for forward pass latency."""

    def test_forward_16_tokens_under_500ms(self, model):
        """Forward pass with 16 tokens should complete in <500ms."""
        input_ids = torch.randint(0, 1000, (1, 16), device="mps")

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        torch.mps.synchronize()

        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Forward 16 tokens took {elapsed_ms:.1f}ms (>500ms)"

    def test_forward_128_tokens_under_2000ms(self, model):
        """Forward pass with 128 tokens should complete in <2000ms."""
        input_ids = torch.randint(0, 1000, (1, 128), device="mps")

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        torch.mps.synchronize()

        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 2000, f"Forward 128 tokens took {elapsed_ms:.1f}ms (>2000ms)"


@requires_mps
@requires_trellis
@requires_model
class TestMoEDispatchRegression:
    """Regression tests for MoE dispatch performance.

    These tests verify the fix for the MoE dispatch slowness bug that caused
    20,000ms+ latency per layer (47 layers = catastrophic slowness).
    """

    def test_no_moe_dispatch_slowness(self, model):
        """MoE dispatch should not exhibit 20s+ slowness.

        Before fix: 20,000ms+ per layer (47 layers = catastrophic)
        After fix: should be <1000ms total for a forward pass
        """
        input_ids = torch.randint(0, 1000, (1, 64), device="mps")

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        torch.mps.synchronize()

        # Run multiple times to get stable measurement
        times = []
        for _ in range(3):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)

        # Should be well under 1000ms - the bug caused 20,000ms+ per layer
        assert avg_ms < 1000, (
            f"MoE dispatch still slow: {avg_ms:.1f}ms avg "
            f"(min={min_ms:.1f}ms, max={max_ms:.1f}ms)"
        )

    def test_consistent_latency(self, model):
        """Forward pass latency should be consistent across calls.

        Large variance indicates buffer creation or other per-call overhead.
        """
        input_ids = torch.randint(0, 1000, (1, 32), device="mps")

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids)
            torch.mps.synchronize()

        # Measure multiple times
        times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

        # Coefficient of variation should be low (< 50%)
        cv = std_ms / avg_ms if avg_ms > 0 else 0

        assert cv < 0.5, (
            f"Latency too variable: avg={avg_ms:.1f}ms, std={std_ms:.1f}ms, CV={cv:.2%}"
        )


@requires_mps
@requires_trellis
@requires_model
class TestDecodeRegression:
    """Regression tests for decode (single token) performance."""

    def test_single_token_decode_latency(self, model):
        """Single token decode should be fast after prefill."""
        # Prefill with initial prompt
        input_ids = torch.randint(0, 1000, (1, 32), device="mps")

        with torch.no_grad():
            _ = model(input_ids)
        torch.mps.synchronize()

        # Now decode single tokens (simulating generation)
        single_token = torch.randint(0, 1000, (1, 1), device="mps")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(single_token)
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)

        # Single token decode should be fast (< 100ms for well-optimized path)
        # Being generous with 200ms threshold to account for variance
        assert avg_ms < 200, f"Single token decode too slow: {avg_ms:.1f}ms avg"
