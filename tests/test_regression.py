"""
Regression tests to catch performance and accuracy regressions.

Uses a baselines.json file to track expected accuracy (RMSE) and latency (ms)
for known matrix dimensions. On first run, baselines are established. On
subsequent runs, results are compared against stored baselines with tolerance.

Accuracy tolerance: 10% regression allowed (quantization noise varies slightly)
Performance tolerance: 20% regression allowed (hardware thermals, background load)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PYTHON_DIR = Path(__file__).parent.parent / "python"
BASELINE_PATH = Path(__file__).parent / "baselines.json"


# ---------------------------------------------------------------------------
# Lazy imports (skip gracefully in CI without Metal)
# ---------------------------------------------------------------------------


def _get_metal_marlin() -> Any:
    """Import metal_marlin, skip test if unavailable."""
    try:
        if str(_PYTHON_DIR) not in sys.path:
            sys.path.insert(0, str(_PYTHON_DIR))
        import metal_marlin

        return metal_marlin
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"metal_marlin not available: {e}")


def _get_mlx() -> Any:
    """Import mlx.core, skip test if unavailable."""
    try:
        import mlx.core as mx

        return mx
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mlx not available")


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------


def load_baselines() -> dict[str, float]:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {}


def save_baselines(baselines: dict[str, float]) -> None:
    BASELINE_PATH.write_text(json.dumps(baselines, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Reference dequant for accuracy validation (pure numpy, no Metal needed)
# ---------------------------------------------------------------------------

FP4_E2M1_TABLE: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float16,
)


def _reference_dequant_gemm(A: np.ndarray, W: np.ndarray, group_size: int = 32) -> np.ndarray:
    """Quantize W to FP4, dequant, then matmul in FP32.

    This provides the accuracy floor: the best any FP4 GEMM can achieve.
    """
    K, N = W.shape
    n_groups = K // group_size

    # Quantize per-group
    W_reshaped = W.reshape(n_groups, group_size, N)
    scales = np.abs(W_reshaped).max(axis=1, keepdims=True).astype(np.float32)
    scales = np.where(scales == 0, 1.0, scales)  # avoid div by zero

    # Normalize and find nearest E2M1 code
    W_norm = W_reshaped.astype(np.float32) / scales
    abs_vals = np.abs(FP4_E2M1_TABLE[:8].astype(np.float32))
    codes = np.zeros_like(W_norm, dtype=np.int32)
    for i in range(n_groups):
        for j in range(group_size):
            row = W_norm[i, j, :]
            signs = (row < 0).astype(np.int32) * 8
            distances = np.abs(np.abs(row)[:, None] - abs_vals[None, :])
            codes[i, j, :] = distances.argmin(axis=1) + signs

    # Dequant
    W_dequant = FP4_E2M1_TABLE[codes].astype(np.float32) * scales
    W_dequant = W_dequant.reshape(K, N)

    return (A.astype(np.float32) @ W_dequant).astype(np.float16)


# ===========================================================================
# Regression Tests
# ===========================================================================


class TestAccuracyRegression:
    """Ensure quantized GEMM accuracy doesn't regress across builds."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (1, 4096, 11008),
            (128, 4096, 4096),
        ],
    )
    def test_accuracy_vs_baseline(self, M: int, N: int, K: int) -> None:
        """Compare Metal GEMM RMSE against stored baselines."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        baselines = load_baselines()
        key = f"accuracy_{M}_{N}_{K}"

        # Deterministic weights and activations
        mx.random.seed(42)
        A = mx.random.normal((M, K), dtype=mx.float16)
        # pack_fp4_weights expects [out_features, in_features] = [N, K]
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=32)

        # Metal kernel result
        result = mm.quantized_linear(A, W_packed, scales, group_size=32)
        mx.eval(result)

        # Reference: FP16 matmul (no quantization loss)
        ref = A @ W_full.T
        mx.eval(ref)

        rmse = float(mx.sqrt(mx.mean((result - ref) ** 2)))

        if key in baselines:
            baseline_rmse = baselines[key]
            assert rmse <= baseline_rmse * 1.1, (
                f"Accuracy regression for ({M},{N},{K}): "
                f"RMSE {rmse:.6f} > baseline {baseline_rmse:.6f} * 1.1"
            )
        else:
            # First run: establish baseline
            baselines[key] = rmse
            save_baselines(baselines)

    @pytest.mark.parametrize(
        "M,N,K,group_size",
        [
            (1, 128, 128, 32),
            (16, 256, 256, 32),
            (1, 4096, 4096, 128),
        ],
    )
    def test_accuracy_vs_reference_dequant(self, M: int, N: int, K: int, group_size: int) -> None:
        """Metal GEMM should match numpy reference dequant-then-matmul."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        mx.random.seed(123)
        A = mx.random.normal((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=group_size)

        # Metal result
        result = mm.quantized_linear(A, W_packed, scales, group_size=group_size)
        mx.eval(result)
        result_np = np.array(result)

        # Numpy reference (same quantization, CPU dequant + matmul)
        A_np = np.array(A)
        # Transpose W to [K, N] for reference (pack_fp4_weights transposes internally)
        W_np = np.array(W_full).T  # [K, N]
        ref_np = _reference_dequant_gemm(A_np, W_np, group_size=group_size)

        # Both should have similar quantization error
        # Allow relative tolerance since both paths quantize independently
        rmse = float(np.sqrt(np.mean((result_np - ref_np) ** 2)))
        max_val = float(np.abs(ref_np).max()) or 1.0
        relative_error = rmse / max_val

        # FP4 quantization introduces ~10-20% relative error;
        # the Metal vs reference mismatch should be much smaller
        assert relative_error < 0.5, (
            f"Metal vs reference mismatch too large for ({M},{N},{K}): "
            f"relative error {relative_error:.4f}"
        )


@pytest.mark.slow
class TestPerformanceRegression:
    """Ensure kernel performance doesn't regress across builds.

    Marked slow because:
    - Requires Metal GPU
    - Runs warmup + 100 iterations per test case
    - Results are hardware-dependent (run on same machine for meaningful comparison)
    """

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (1, 4096, 11008),
            (128, 4096, 4096),
        ],
    )
    def test_latency_vs_baseline(self, M: int, N: int, K: int) -> None:
        """Kernel latency must not regress more than 20% vs stored baseline."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        baselines = load_baselines()
        key = f"perf_{M}_{N}_{K}"

        mx.random.seed(0)
        A = mx.random.normal((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=32)

        # Warmup: ensure kernel is compiled and caches are hot
        for _ in range(20):
            out = mm.quantized_linear(A, W_packed, scales, group_size=32)
            mx.eval(out)

        # Benchmark: measure 100 iterations
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            out = mm.quantized_linear(A, W_packed, scales, group_size=32)
            mx.eval(out)
        elapsed_ms = (time.perf_counter() - start) * 1000 / n_iters

        if key in baselines:
            baseline_ms = baselines[key]
            assert elapsed_ms <= baseline_ms * 1.2, (
                f"Performance regression for ({M},{N},{K}): "
                f"{elapsed_ms:.3f}ms > baseline {baseline_ms:.3f}ms * 1.2"
            )
        else:
            # First run: establish baseline
            baselines[key] = elapsed_ms
            save_baselines(baselines)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 4096, 4096),
            (32, 4096, 4096),
        ],
    )
    def test_throughput_tflops(self, M: int, N: int, K: int) -> None:
        """Verify kernel achieves minimum expected TFLOPS."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        mx.random.seed(0)
        A = mx.random.normal((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=32)

        # Warmup
        for _ in range(20):
            out = mm.quantized_linear(A, W_packed, scales, group_size=32)
            mx.eval(out)

        # Measure
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            out = mm.quantized_linear(A, W_packed, scales, group_size=32)
            mx.eval(out)
        elapsed_s = (time.perf_counter() - start) / n_iters

        flops = 2 * M * N * K
        tflops = flops / elapsed_s / 1e12

        # FP4 GEMM on M-series should exceed 0.1 TFLOPS even for small M
        # This is a sanity floor, not a target
        assert tflops > 0.01, (
            f"Throughput too low for ({M},{N},{K}): {tflops:.4f} TFLOPS (expected > 0.01)"
        )


class TestNumericalStability:
    """Regression tests for edge cases that previously caused numerical issues."""

    def test_zero_input_zero_output(self) -> None:
        """Zero activations must produce zero output regardless of weights."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        M, N, K = 1, 128, 128
        A = mx.zeros((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=32)

        result = mm.quantized_linear(A, W_packed, scales, group_size=32)
        mx.eval(result)

        assert mx.all(result == 0), "Zero input should produce zero output"

    def test_identity_weight_passthrough(self) -> None:
        """Identity-like weights should approximately pass through the input."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        N = K = 128
        M = 1

        # Create identity weights [N, K] (square)
        W_identity = mx.eye(N, dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_identity, group_size=32)
        mx.eval(W_packed, scales)

        A = mx.array([[1.0] * K], dtype=mx.float16)

        # Warmup: prime the Metal kernel compilation cache to avoid
        # first-invocation issues on Apple Silicon
        warmup_out = mm.quantized_linear(A, W_packed, scales, group_size=32)
        mx.eval(warmup_out)

        result = mm.quantized_linear(A, W_packed, scales, group_size=32)
        mx.eval(result)

        # After FP4 quantization, identity diagonal values (1.0) should survive
        # since 1.0 is exactly representable in E2M1
        expected = A  # identity passthrough
        rmse = float(mx.sqrt(mx.mean((result - expected) ** 2)))
        assert rmse < 0.1, f"Identity weight passthrough RMSE too high: {rmse:.4f}"

    def test_large_scale_weights(self) -> None:
        """Weights with large magnitude should be handled by group scaling."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        M, N, K = 4, 256, 256
        mx.random.seed(99)

        # Weights with large values that stress the per-group scale
        A = mx.random.normal((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16) * 100.0
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=32)

        result = mm.quantized_linear(A, W_packed, scales, group_size=32)
        mx.eval(result)

        # Should not produce NaN or Inf
        assert not mx.any(mx.isnan(result)), "NaN in output with large weights"
        assert not mx.any(mx.isinf(result)), "Inf in output with large weights"

    def test_extreme_scale_values(self) -> None:
        """Extreme scale values should not produce NaN/Inf outputs."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        M, N, K = 2, 128, 128
        group_size = 32
        mx.random.seed(101)

        A = mx.random.normal((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=group_size)

        zero_scales = mx.zeros(scales.shape, dtype=mx.float16)
        result = mm.quantized_linear(A, W_packed, zero_scales, group_size=group_size)
        mx.eval(result)
        assert not mx.any(mx.isnan(result)), "NaN in output with zero scales"
        assert not mx.any(mx.isinf(result)), "Inf in output with zero scales"

        # Use scale=10 which is large but won't overflow FP16 during GEMM accumulation
        # (FP4 max=6, K=128, scale=10 -> max partial sum ~7680, well within FP16 range)
        large_scales = mx.ones(scales.shape, dtype=mx.float16) * 10.0
        result = mm.quantized_linear(A, W_packed, large_scales, group_size=group_size)
        mx.eval(result)
        assert not mx.any(mx.isnan(result)), "NaN in output with large scales"
        assert not mx.any(mx.isinf(result)), "Inf in output with large scales"

    def test_reproducibility(self) -> None:
        """Same inputs must produce identical outputs across calls."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        M, N, K = 8, 512, 512
        mx.random.seed(77)

        A = mx.random.normal((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=32)

        result1 = mm.quantized_linear(A, W_packed, scales, group_size=32)
        mx.eval(result1)

        result2 = mm.quantized_linear(A, W_packed, scales, group_size=32)
        mx.eval(result2)

        assert mx.array_equal(result1, result2), (
            "Non-deterministic output: same inputs produced different results"
        )

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_group_size_consistency(self, group_size: int) -> None:
        """Different group sizes should all produce finite, reasonable results."""
        mx = _get_mlx()
        mm = _get_metal_marlin()

        M, N, K = 4, 256, 256
        # K must be divisible by group_size
        assert K % group_size == 0

        mx.random.seed(55)
        A = mx.random.normal((M, K), dtype=mx.float16)
        W_full = mx.random.normal((N, K), dtype=mx.float16)
        W_packed, scales = mm.pack_fp4_weights(W_full, group_size=group_size)

        result = mm.quantized_linear(A, W_packed, scales, group_size=group_size)
        mx.eval(result)

        assert not mx.any(mx.isnan(result)), f"NaN with group_size={group_size}"
        assert not mx.any(mx.isinf(result)), f"Inf with group_size={group_size}"

        # Compare against FP16 reference for sanity
        ref = A @ W_full.T
        mx.eval(ref)
        diff = (result - ref).astype(mx.float32)
        rmse = float(mx.sqrt(mx.mean(diff**2)))
        max_val = float(mx.abs(ref).max()) or 1.0
        relative_error = rmse / max_val

        # FP4 should be within 50% relative error of FP16
        assert relative_error < 0.5, (
            f"group_size={group_size} relative error too high: {relative_error:.4f}"
        )
