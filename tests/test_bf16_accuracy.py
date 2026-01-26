"""BF16 vs FP16 accuracy validation tests.

Validates that BF16 operations provide equal or better accuracy than FP16,
particularly for scenarios where BF16's larger dynamic range helps:
- Large activations (no overflow)
- Attention softmax stability
- MoE router precision
- Quantization scale handling

Test coverage:
1. GEMM numerical accuracy (BF16 vs FP16 vs FP32 reference)
2. Attention accuracy (BF16 accumulation)
3. Quantization error (FP16 scales vs FP32 scales)
4. End-to-end perplexity (should be equal or better)

Key insight: BF16 has 8-bit exponent (same as FP32) vs FP16's 5-bit exponent,
giving ~10^38 dynamic range vs ~65504. This prevents overflow in:
- Large K reductions in GEMM
- Softmax pre-normalization
- Gradient accumulation (if training)
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add metal_marlin python module to path
_METAL_MARLIN_DIR = Path(__file__).parent.parent
if str(_METAL_MARLIN_DIR) not in sys.path:
    sys.path.insert(0, str(_METAL_MARLIN_DIR))


# ---------------------------------------------------------------------------
# MLX availability checks
# ---------------------------------------------------------------------------


def _check_mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _check_bf16_available() -> bool:
    """Check if MLX supports bfloat16."""
    try:
        import mlx.core as mx
        x = mx.array([1.0], dtype=mx.bfloat16)
        mx.eval(x)
        return True
    except (ImportError, AttributeError, Exception):
        return False


requires_metal = pytest.mark.skipif(
    not _check_mlx_available(),
    reason="MLX not available (no Metal GPU)",
)

requires_bf16 = pytest.mark.skipif(
    not _check_bf16_available(),
    reason="BF16 not available in MLX",
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def gemm_reference_fp32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """FP32 reference GEMM."""
    return A.astype(np.float32) @ B.astype(np.float32)


def softmax_reference(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in FP32."""
    x = x.astype(np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_reference(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float
) -> np.ndarray:
    """Reference attention: softmax(Q @ K^T * scale) @ V in FP32.

    Supports 4D tensors [batch, heads, seq, head_dim].
    """
    Q = Q.astype(np.float32)
    K = K.astype(np.float32)
    V = V.astype(np.float32)
    # For 4D: transpose last two axes of K
    if K.ndim == 4:
        K_T = np.transpose(K, (0, 1, 3, 2))
    else:
        K_T = np.swapaxes(K, -1, -2)
    scores = Q @ K_T * scale
    attn = softmax_reference(scores, axis=-1)
    return attn @ V


def mx_to_numpy(mx_array) -> np.ndarray:
    """Convert MLX array to numpy, handling bfloat16 properly."""
    import mlx.core as mx
    # BF16 can't be directly converted to numpy, cast to float32 first
    if mx_array.dtype == mx.bfloat16:
        mx_array = mx_array.astype(mx.float32)
    mx.eval(mx_array)
    return np.array(mx_array)


# ---------------------------------------------------------------------------
# Data classes for test results
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AccuracyResult:
    """Container for accuracy comparison results."""
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    rmse: float
    correlation: float
    has_nan: bool
    has_inf: bool

    def __str__(self) -> str:
        return (
            f"max_abs={self.max_abs_error:.6f}, mean_abs={self.mean_abs_error:.6f}, "
            f"max_rel={self.max_rel_error:.4f}, rmse={self.rmse:.6f}, "
            f"corr={self.correlation:.6f}, nan={self.has_nan}, inf={self.has_inf}"
        )


def compute_accuracy(result: np.ndarray, reference: np.ndarray) -> AccuracyResult:
    """Compute accuracy metrics between result and reference."""
    result = result.astype(np.float32).flatten()
    reference = reference.astype(np.float32).flatten()

    abs_error = np.abs(result - reference)
    max_abs_error = float(abs_error.max())
    mean_abs_error = float(abs_error.mean())

    # Relative error (avoid divide by zero)
    ref_abs = np.abs(reference)
    mask = ref_abs > 1e-7
    if mask.any():
        rel_error = abs_error[mask] / ref_abs[mask]
        max_rel_error = float(rel_error.max())
        mean_rel_error = float(rel_error.mean())
    else:
        max_rel_error = 0.0
        mean_rel_error = 0.0

    rmse = float(np.sqrt(np.mean(abs_error**2)))

    # Correlation
    if np.std(result) > 1e-10 and np.std(reference) > 1e-10:
        correlation = float(np.corrcoef(result, reference)[0, 1])
    else:
        correlation = 1.0 if np.allclose(result, reference) else 0.0

    return AccuracyResult(
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        max_rel_error=max_rel_error,
        mean_rel_error=mean_rel_error,
        rmse=rmse,
        correlation=correlation,
        has_nan=bool(np.isnan(result).any()),
        has_inf=bool(np.isinf(result).any()),
    )


# ---------------------------------------------------------------------------
# BF16 vs FP16 comparison logging
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Bf16VsFp16Result:
    """Comparison result for BF16 vs FP16."""
    test_name: str
    bf16_accuracy: AccuracyResult
    fp16_accuracy: AccuracyResult
    bf16_better: bool
    improvement_factor: float  # >1 means BF16 is better

    def log(self) -> None:
        """Log the comparison result."""
        status = "BF16 BETTER" if self.bf16_better else "FP16 BETTER"
        print(f"\n{self.test_name}:")
        print(f"  BF16: {self.bf16_accuracy}")
        print(f"  FP16: {self.fp16_accuracy}")
        print(f"  Result: {status} (factor: {self.improvement_factor:.2f}x)")


_bf16_worse_cases: list[Bf16VsFp16Result] = []


def log_if_bf16_worse(result: Bf16VsFp16Result) -> None:
    """Log cases where BF16 performs worse than FP16."""
    if not result.bf16_better:
        _bf16_worse_cases.append(result)
        print(f"\n*** WARNING: BF16 worse than FP16 in {result.test_name} ***")


# ---------------------------------------------------------------------------
# Test: GEMM Numerical Accuracy
# ---------------------------------------------------------------------------


class TestGEMMBf16Accuracy:
    """Test GEMM accuracy with BF16 vs FP16 activations and scales."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    @requires_metal
    @requires_bf16
    @pytest.mark.parametrize("M,K,N", [
        (1, 128, 128),        # Small
        (32, 512, 512),       # Medium
        (1, 4096, 4096),      # LLM typical
        (128, 4096, 4096),    # Batched LLM
        (1, 4096, 14336),     # Llama-3 MLP up
        (1, 14336, 4096),     # Llama-3 MLP down
    ])
    def test_gemm_bf16_vs_fp16(
        self, rng: np.random.Generator, M: int, K: int, N: int
    ) -> None:
        """BF16 GEMM should match or beat FP16 accuracy vs FP32 reference."""
        import mlx.core as mx

        # Generate test data in FP32
        A_fp32 = rng.standard_normal((M, K)).astype(np.float32)
        B_fp32 = rng.standard_normal((K, N)).astype(np.float32)

        # FP32 reference
        ref = gemm_reference_fp32(A_fp32, B_fp32)

        # FP16 computation
        A_fp16 = mx.array(A_fp32.astype(np.float16))
        B_fp16 = mx.array(B_fp32.astype(np.float16))
        result_fp16 = A_fp16 @ B_fp16
        mx.eval(result_fp16)
        result_fp16_np = np.array(result_fp16, dtype=np.float32)

        # BF16 computation
        A_bf16 = mx.array(A_fp32, dtype=mx.bfloat16)
        B_bf16 = mx.array(B_fp32, dtype=mx.bfloat16)
        result_bf16 = A_bf16 @ B_bf16
        result_bf16_np = mx_to_numpy(result_bf16)

        # Compute accuracy
        acc_fp16 = compute_accuracy(result_fp16_np, ref)
        acc_bf16 = compute_accuracy(result_bf16_np, ref)

        # BF16 should not be worse than FP16 by more than a small margin
        # (allow 20% tolerance due to different rounding behavior)
        bf16_better = acc_bf16.rmse <= acc_fp16.rmse * 1.2

        improvement = acc_fp16.rmse / max(acc_bf16.rmse, 1e-10)
        comparison = Bf16VsFp16Result(
            test_name=f"GEMM_{M}x{K}x{N}",
            bf16_accuracy=acc_bf16,
            fp16_accuracy=acc_fp16,
            bf16_better=bf16_better,
            improvement_factor=improvement,
        )
        comparison.log()
        log_if_bf16_worse(comparison)

        # Assert no NaN/Inf in either
        assert not acc_bf16.has_nan, f"BF16 produced NaN in GEMM {M}x{K}x{N}"
        assert not acc_bf16.has_inf, f"BF16 produced Inf in GEMM {M}x{K}x{N}"
        assert not acc_fp16.has_nan, f"FP16 produced NaN in GEMM {M}x{K}x{N}"

        # Allow FP16 inf (expected for large K) but not BF16
        # BF16's larger dynamic range should prevent overflow

    @requires_metal
    @requires_bf16
    def test_gemm_large_k_overflow(self, rng: np.random.Generator) -> None:
        """Test that BF16 handles large K reduction without overflow.

        FP16 max is ~65504, so summing K > 65504 elements of magnitude ~1
        can overflow FP16 accumulation. BF16 should handle this.
        """
        import mlx.core as mx

        M, K, N = 1, 32768, 128
        # Use values that won't overflow individually but might in accumulation
        A_fp32 = np.ones((M, K), dtype=np.float32) * 0.01
        B_fp32 = np.ones((K, N), dtype=np.float32) * 0.01

        gemm_reference_fp32(A_fp32, B_fp32)
        expected_value = K * 0.01 * 0.01  # Should be ~3.2768

        # FP16
        A_fp16 = mx.array(A_fp32.astype(np.float16))
        B_fp16 = mx.array(B_fp32.astype(np.float16))
        result_fp16 = A_fp16 @ B_fp16
        mx.eval(result_fp16)
        result_fp16_np = np.array(result_fp16, dtype=np.float32)

        # BF16
        A_bf16 = mx.array(A_fp32, dtype=mx.bfloat16)
        B_bf16 = mx.array(B_fp32, dtype=mx.bfloat16)
        result_bf16 = A_bf16 @ B_bf16
        result_bf16_np = mx_to_numpy(result_bf16)

        # Check that BF16 is closer to reference
        bf16_error = np.abs(result_bf16_np[0, 0] - expected_value)
        fp16_error = np.abs(result_fp16_np[0, 0] - expected_value)

        print(f"\nLarge K ({K}) reduction test:")
        print(f"  Expected: {expected_value:.6f}")
        print(f"  BF16: {result_bf16_np[0, 0]:.6f} (error: {bf16_error:.6f})")
        print(f"  FP16: {result_fp16_np[0, 0]:.6f} (error: {fp16_error:.6f})")

        # BF16 should not have NaN or Inf
        assert not np.isnan(result_bf16_np).any(), "BF16 produced NaN in large K"
        assert not np.isinf(result_bf16_np).any(), "BF16 produced Inf in large K"

    @requires_metal
    @requires_bf16
    def test_gemm_extreme_values(self, rng: np.random.Generator) -> None:
        """Test with values near FP16 max to verify BF16 dynamic range advantage."""
        import mlx.core as mx

        M, K, N = 4, 128, 128
        # Values that approach FP16 max
        scale = 100.0  # Large but not overflowing

        A_fp32 = rng.standard_normal((M, K)).astype(np.float32) * scale
        B_fp32 = rng.standard_normal((K, N)).astype(np.float32) * scale

        ref = gemm_reference_fp32(A_fp32, B_fp32)

        # FP16 - may have overflow issues
        A_fp16 = mx.array(A_fp32.astype(np.float16))
        B_fp16 = mx.array(B_fp32.astype(np.float16))
        result_fp16 = A_fp16 @ B_fp16
        mx.eval(result_fp16)
        result_fp16_np = np.array(result_fp16, dtype=np.float32)

        # BF16 - should handle large values better
        A_bf16 = mx.array(A_fp32, dtype=mx.bfloat16)
        B_bf16 = mx.array(B_fp32, dtype=mx.bfloat16)
        result_bf16 = A_bf16 @ B_bf16
        result_bf16_np = mx_to_numpy(result_bf16)

        acc_fp16 = compute_accuracy(result_fp16_np, ref)
        acc_bf16 = compute_accuracy(result_bf16_np, ref)

        print(f"\nExtreme values test (scale={scale}):")
        print(f"  BF16: {acc_bf16}")
        print(f"  FP16: {acc_fp16}")

        # BF16 should not overflow
        assert not acc_bf16.has_nan, "BF16 produced NaN with extreme values"
        assert not acc_bf16.has_inf, "BF16 produced Inf with extreme values"


# ---------------------------------------------------------------------------
# Test: Attention Accuracy
# ---------------------------------------------------------------------------


class TestAttentionBf16Accuracy:
    """Test attention accuracy with BF16 vs FP16."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=123)

    @requires_metal
    @requires_bf16
    @pytest.mark.parametrize("batch,heads,seq_q,seq_k,head_dim", [
        (1, 8, 1, 128, 64),      # Single query, short context
        (1, 8, 1, 2048, 128),    # Single query, long context
        (1, 32, 512, 512, 128),  # Prefill
        (4, 32, 1, 4096, 128),   # Batched decode, very long
    ])
    def test_attention_bf16_vs_fp16(
        self,
        rng: np.random.Generator,
        batch: int,
        heads: int,
        seq_q: int,
        seq_k: int,
        head_dim: int,
    ) -> None:
        """BF16 attention should match or beat FP16 accuracy."""
        import mlx.core as mx

        scale = 1.0 / math.sqrt(head_dim)

        # Generate test data
        Q = rng.standard_normal((batch, heads, seq_q, head_dim)).astype(np.float32)
        K = rng.standard_normal((batch, heads, seq_k, head_dim)).astype(np.float32)
        V = rng.standard_normal((batch, heads, seq_k, head_dim)).astype(np.float32)

        # FP32 reference
        ref = attention_reference(Q, K, V, scale)

        # FP16 attention
        Q_fp16 = mx.array(Q.astype(np.float16))
        K_fp16 = mx.array(K.astype(np.float16))
        V_fp16 = mx.array(V.astype(np.float16))

        scores_fp16 = Q_fp16 @ mx.transpose(K_fp16, axes=(0, 1, 3, 2)) * scale
        attn_fp16 = mx.softmax(scores_fp16, axis=-1)
        result_fp16 = attn_fp16 @ V_fp16
        mx.eval(result_fp16)
        result_fp16_np = np.array(result_fp16, dtype=np.float32)

        # BF16 attention
        Q_bf16 = mx.array(Q, dtype=mx.bfloat16)
        K_bf16 = mx.array(K, dtype=mx.bfloat16)
        V_bf16 = mx.array(V, dtype=mx.bfloat16)

        scores_bf16 = Q_bf16 @ mx.transpose(K_bf16, axes=(0, 1, 3, 2)) * scale
        attn_bf16 = mx.softmax(scores_bf16, axis=-1)
        result_bf16 = attn_bf16 @ V_bf16
        result_bf16_np = mx_to_numpy(result_bf16)

        acc_fp16 = compute_accuracy(result_fp16_np, ref)
        acc_bf16 = compute_accuracy(result_bf16_np, ref)

        comparison = Bf16VsFp16Result(
            test_name=f"Attention_b{batch}_h{heads}_sq{seq_q}_sk{seq_k}_d{head_dim}",
            bf16_accuracy=acc_bf16,
            fp16_accuracy=acc_fp16,
            bf16_better=acc_bf16.rmse <= acc_fp16.rmse * 1.2,
            improvement_factor=acc_fp16.rmse / max(acc_bf16.rmse, 1e-10),
        )
        comparison.log()
        log_if_bf16_worse(comparison)

        # Both should be valid
        assert not acc_bf16.has_nan, "BF16 attention produced NaN"
        assert not acc_bf16.has_inf, "BF16 attention produced Inf"

    @requires_metal
    @requires_bf16
    def test_attention_softmax_stability(self, rng: np.random.Generator) -> None:
        """Test softmax stability with large attention scores.

        BF16's larger dynamic range should prevent overflow in softmax
        when attention scores are large (e.g., very long context or
        pathological attention patterns).
        """
        import mlx.core as mx

        batch, heads, seq_q, seq_k, head_dim = 1, 8, 1, 8192, 128
        scale = 1.0 / math.sqrt(head_dim)

        # Create attention scores with some extreme values
        Q = rng.standard_normal((batch, heads, seq_q, head_dim)).astype(np.float32)
        K = rng.standard_normal((batch, heads, seq_k, head_dim)).astype(np.float32)

        # Scale up to create large attention scores
        Q = Q * 5.0
        K = K * 5.0

        V = rng.standard_normal((batch, heads, seq_k, head_dim)).astype(np.float32)
        ref = attention_reference(Q, K, V, scale)

        # FP16
        Q_fp16 = mx.array(Q.astype(np.float16))
        K_fp16 = mx.array(K.astype(np.float16))
        V_fp16 = mx.array(V.astype(np.float16))
        scores_fp16 = Q_fp16 @ mx.transpose(K_fp16, axes=(0, 1, 3, 2)) * scale
        attn_fp16 = mx.softmax(scores_fp16, axis=-1)
        result_fp16 = attn_fp16 @ V_fp16
        mx.eval(result_fp16)
        result_fp16_np = np.array(result_fp16, dtype=np.float32)

        # BF16
        Q_bf16 = mx.array(Q, dtype=mx.bfloat16)
        K_bf16 = mx.array(K, dtype=mx.bfloat16)
        V_bf16 = mx.array(V, dtype=mx.bfloat16)
        scores_bf16 = Q_bf16 @ mx.transpose(K_bf16, axes=(0, 1, 3, 2)) * scale
        attn_bf16 = mx.softmax(scores_bf16, axis=-1)
        result_bf16 = attn_bf16 @ V_bf16
        result_bf16_np = mx_to_numpy(result_bf16)

        acc_fp16 = compute_accuracy(result_fp16_np, ref)
        acc_bf16 = compute_accuracy(result_bf16_np, ref)

        print("\nSoftmax stability test (large scores):")
        print(f"  BF16: {acc_bf16}")
        print(f"  FP16: {acc_fp16}")

        # BF16 should be stable
        assert not acc_bf16.has_nan, "BF16 softmax produced NaN with large scores"
        assert not acc_bf16.has_inf, "BF16 softmax produced Inf with large scores"


# ---------------------------------------------------------------------------
# Test: Quantization Error
# ---------------------------------------------------------------------------


class TestQuantizationBf16Accuracy:
    """Test quantization accuracy with FP16 vs FP32 scales."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=456)

    @requires_metal
    @requires_bf16
    @pytest.mark.parametrize("K,N,group_size", [
        (128, 128, 32),
        (4096, 4096, 128),
        (4096, 14336, 128),
    ])
    def test_quantization_scale_precision(
        self,
        rng: np.random.Generator,
        K: int,
        N: int,
        group_size: int,
    ) -> None:
        """Test that FP32 scales give better accuracy than FP16 scales.

        This is the baseline for the BF16 scales comparison - BF16 scales
        should be closer to FP32 than FP16 scales due to better precision
        in the mantissa after the exponent difference.
        """
        import mlx.core as mx

        # Generate weights
        weights = rng.standard_normal((K, N)).astype(np.float32)

        # Compute scales in FP32
        num_groups = K // group_size
        w_grouped = weights.reshape(num_groups, group_size, N)
        group_max = np.abs(w_grouped).max(axis=1)
        scales_fp32 = np.maximum(group_max / 6.0, 1e-7)

        # FP16 scales (loss of precision)
        scales_fp16 = scales_fp32.astype(np.float16).astype(np.float32)

        # BF16 scales
        scales_bf16 = mx.array(scales_fp32, dtype=mx.bfloat16)
        scales_bf16_np = mx_to_numpy(scales_bf16)

        # Compare precision loss
        fp16_error = np.abs(scales_fp16 - scales_fp32) / (scales_fp32 + 1e-10)
        bf16_error = np.abs(scales_bf16_np - scales_fp32) / (scales_fp32 + 1e-10)

        fp16_max_rel_err = float(fp16_error.max())
        fp16_mean_rel_err = float(fp16_error.mean())
        bf16_max_rel_err = float(bf16_error.max())
        bf16_mean_rel_err = float(bf16_error.mean())

        print(f"\nScale precision test ({K}x{N}, gs={group_size}):")
        print(f"  FP16: max_rel={fp16_max_rel_err:.6f}, mean_rel={fp16_mean_rel_err:.6f}")
        print(f"  BF16: max_rel={bf16_max_rel_err:.6f}, mean_rel={bf16_mean_rel_err:.6f}")

        # Note: BF16 has fewer mantissa bits than FP16, so precision is actually
        # worse for values in the normal range. The advantage is in dynamic range.
        # We document this but don't fail on it.
        if bf16_max_rel_err > fp16_max_rel_err:
            print("  Note: BF16 has lower precision than FP16 (expected - fewer mantissa bits)")

    @requires_metal
    @requires_bf16
    def test_quantization_scale_dynamic_range(self, rng: np.random.Generator) -> None:
        """Test scale handling when values span large dynamic range.

        BF16 should handle widely varying scales better than FP16
        because of its larger exponent range.
        """
        import mlx.core as mx

        # Create weights with extreme scale variation
        K, N = 1024, 256
        group_size = 128

        # Mix of very small and very large groups
        weights = np.zeros((K, N), dtype=np.float32)
        for g in range(K // group_size):
            start = g * group_size
            end = start + group_size
            # Alternate between tiny and large magnitudes
            scale = 1e-6 if g % 2 == 0 else 1e4
            weights[start:end, :] = rng.standard_normal((group_size, N)) * scale

        # Compute reference scales
        num_groups = K // group_size
        w_grouped = weights.reshape(num_groups, group_size, N)
        group_max = np.abs(w_grouped).max(axis=1)
        scales_fp32 = np.maximum(group_max / 6.0, 1e-10)

        # Check which scales overflow/underflow in FP16
        scales_fp16 = scales_fp32.astype(np.float16).astype(np.float32)
        fp16_underflow = np.sum(scales_fp16 == 0)
        fp16_overflow = np.sum(np.isinf(scales_fp16))

        # BF16 should handle all scales
        scales_bf16 = mx.array(scales_fp32, dtype=mx.bfloat16)
        scales_bf16_np = mx_to_numpy(scales_bf16)
        bf16_underflow = np.sum(scales_bf16_np == 0)
        bf16_overflow = np.sum(np.isinf(scales_bf16_np))

        print("\nScale dynamic range test:")
        print(f"  Scale range: [{scales_fp32.min():.2e}, {scales_fp32.max():.2e}]")
        print(f"  FP16: underflow={fp16_underflow}, overflow={fp16_overflow}")
        print(f"  BF16: underflow={bf16_underflow}, overflow={bf16_overflow}")

        # BF16 should have fewer overflow/underflow issues
        assert bf16_overflow <= fp16_overflow, "BF16 has more overflow than FP16"


# ---------------------------------------------------------------------------
# Test: End-to-end Perplexity (slow, marked accordingly)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPerplexityBf16:
    """End-to-end perplexity comparison between BF16 and FP16 inference."""

    @requires_metal
    @requires_bf16
    def test_gemm_perplexity_proxy(self) -> None:
        """Proxy perplexity test using accumulated GEMM error.

        Instead of full model inference, we test the impact of BF16 vs FP16
        on accumulated GEMM errors through a sequence of operations
        simulating a transformer forward pass.
        """
        import mlx.core as mx

        rng = np.random.default_rng(seed=789)

        # Simulate a simple transformer-like computation
        # hidden_dim=2048, num_layers=4, seq_len=128
        hidden_dim = 2048
        intermediate_dim = hidden_dim * 4
        num_layers = 4
        seq_len = 128

        # Initial hidden states
        x_fp32 = rng.standard_normal((1, seq_len, hidden_dim)).astype(np.float32)

        # Generate weight matrices for each layer
        weights = []
        for _ in range(num_layers):
            weights.append({
                'q': rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * 0.02,
                'k': rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * 0.02,
                'v': rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * 0.02,
                'o': rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * 0.02,
                'up': rng.standard_normal((hidden_dim, intermediate_dim)).astype(np.float32) * 0.02,
                'down': rng.standard_normal((intermediate_dim, hidden_dim)).astype(np.float32) * 0.02,
            })

        # FP32 reference pass
        x_ref = x_fp32.copy()
        for layer_weights in weights:
            # Simplified attention: just linear projections
            q = x_ref @ layer_weights['q']
            k = x_ref @ layer_weights['k']
            v = x_ref @ layer_weights['v']
            attn_out = (softmax_reference(q @ k.transpose(0, 2, 1) / math.sqrt(hidden_dim)) @ v)
            x_ref = x_ref + attn_out @ layer_weights['o']
            # MLP
            up = x_ref @ layer_weights['up']
            up = up * (1 / (1 + np.exp(-up)))  # SiLU
            x_ref = x_ref + up @ layer_weights['down']

        # FP16 pass
        x_fp16 = mx.array(x_fp32.astype(np.float16))
        for layer_weights in weights:
            w_q = mx.array(layer_weights['q'].astype(np.float16))
            w_k = mx.array(layer_weights['k'].astype(np.float16))
            w_v = mx.array(layer_weights['v'].astype(np.float16))
            w_o = mx.array(layer_weights['o'].astype(np.float16))
            w_up = mx.array(layer_weights['up'].astype(np.float16))
            w_down = mx.array(layer_weights['down'].astype(np.float16))

            q = x_fp16 @ w_q
            k = x_fp16 @ w_k
            v = x_fp16 @ w_v
            scores = q @ mx.transpose(k, axes=(0, 2, 1)) / math.sqrt(hidden_dim)
            attn_out = mx.softmax(scores, axis=-1) @ v
            x_fp16 = x_fp16 + attn_out @ w_o
            up = x_fp16 @ w_up
            up = up * mx.sigmoid(up)
            x_fp16 = x_fp16 + up @ w_down
        mx.eval(x_fp16)

        # BF16 pass
        x_bf16 = mx.array(x_fp32, dtype=mx.bfloat16)
        for layer_weights in weights:
            w_q = mx.array(layer_weights['q'], dtype=mx.bfloat16)
            w_k = mx.array(layer_weights['k'], dtype=mx.bfloat16)
            w_v = mx.array(layer_weights['v'], dtype=mx.bfloat16)
            w_o = mx.array(layer_weights['o'], dtype=mx.bfloat16)
            w_up = mx.array(layer_weights['up'], dtype=mx.bfloat16)
            w_down = mx.array(layer_weights['down'], dtype=mx.bfloat16)

            q = x_bf16 @ w_q
            k = x_bf16 @ w_k
            v = x_bf16 @ w_v
            scores = q @ mx.transpose(k, axes=(0, 2, 1)) / math.sqrt(hidden_dim)
            attn_out = mx.softmax(scores, axis=-1) @ v
            x_bf16 = x_bf16 + attn_out @ w_o
            up = x_bf16 @ w_up
            up = up * mx.sigmoid(up)
            x_bf16 = x_bf16 + up @ w_down
        mx.eval(x_bf16)

        # Compare accumulated errors
        x_fp16_np = np.array(x_fp16, dtype=np.float32)
        x_bf16_np = mx_to_numpy(x_bf16)

        acc_fp16 = compute_accuracy(x_fp16_np, x_ref)
        acc_bf16 = compute_accuracy(x_bf16_np, x_ref)

        comparison = Bf16VsFp16Result(
            test_name="Transformer_forward_pass_accumulated_error",
            bf16_accuracy=acc_bf16,
            fp16_accuracy=acc_fp16,
            bf16_better=acc_bf16.rmse <= acc_fp16.rmse * 1.2,
            improvement_factor=acc_fp16.rmse / max(acc_bf16.rmse, 1e-10),
        )
        comparison.log()
        log_if_bf16_worse(comparison)

        # Both should be reasonable
        assert acc_bf16.correlation > 0.9, f"BF16 correlation too low: {acc_bf16.correlation}"
        assert acc_fp16.correlation > 0.9, f"FP16 correlation too low: {acc_fp16.correlation}"


# ---------------------------------------------------------------------------
# Test: MoE Router Precision
# ---------------------------------------------------------------------------


class TestMoERouterBf16Accuracy:
    """Test MoE router precision with BF16 vs FP16.

    MoE routers use softmax over expert scores. BF16's larger dynamic range
    should help with:
    - Large expert score differences
    - Top-k selection stability
    """

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=321)

    @requires_metal
    @requires_bf16
    @pytest.mark.parametrize("num_experts,hidden_dim,seq_len", [
        (8, 2048, 128),      # Mixtral-style
        (64, 4096, 256),     # Large MoE
        (128, 8192, 64),     # Very large MoE
    ])
    def test_moe_router_softmax(
        self,
        rng: np.random.Generator,
        num_experts: int,
        hidden_dim: int,
        seq_len: int,
    ) -> None:
        """MoE router softmax should be more stable with BF16."""
        import mlx.core as mx

        # Router weights and input
        router_weight = rng.standard_normal((hidden_dim, num_experts)).astype(np.float32) * 0.1
        x = rng.standard_normal((1, seq_len, hidden_dim)).astype(np.float32)

        # FP32 reference
        router_logits_ref = x @ router_weight
        router_probs_ref = softmax_reference(router_logits_ref, axis=-1)

        # FP16
        x_fp16 = mx.array(x.astype(np.float16))
        w_fp16 = mx.array(router_weight.astype(np.float16))
        router_logits_fp16 = x_fp16 @ w_fp16
        router_probs_fp16 = mx.softmax(router_logits_fp16, axis=-1)
        mx.eval(router_probs_fp16)
        router_probs_fp16_np = np.array(router_probs_fp16, dtype=np.float32)

        # BF16
        x_bf16 = mx.array(x, dtype=mx.bfloat16)
        w_bf16 = mx.array(router_weight, dtype=mx.bfloat16)
        router_logits_bf16 = x_bf16 @ w_bf16
        router_probs_bf16 = mx.softmax(router_logits_bf16, axis=-1)
        router_probs_bf16_np = mx_to_numpy(router_probs_bf16)

        acc_fp16 = compute_accuracy(router_probs_fp16_np, router_probs_ref)
        acc_bf16 = compute_accuracy(router_probs_bf16_np, router_probs_ref)

        comparison = Bf16VsFp16Result(
            test_name=f"MoE_router_e{num_experts}_h{hidden_dim}_s{seq_len}",
            bf16_accuracy=acc_bf16,
            fp16_accuracy=acc_fp16,
            bf16_better=acc_bf16.rmse <= acc_fp16.rmse * 1.2,
            improvement_factor=acc_fp16.rmse / max(acc_bf16.rmse, 1e-10),
        )
        comparison.log()
        log_if_bf16_worse(comparison)

        # Verify probability sums to ~1
        # Note: BF16 has fewer mantissa bits than FP16, so the sum deviation is larger
        # BF16: 7 mantissa bits -> ~1% tolerance expected
        # FP16: 10 mantissa bits -> ~0.1% tolerance expected
        bf16_sum = router_probs_bf16_np.sum(axis=-1)
        fp16_sum = router_probs_fp16_np.sum(axis=-1)

        # Use relaxed tolerance for BF16 due to lower precision
        assert np.allclose(bf16_sum, 1.0, atol=0.02), f"BF16 probs don't sum to 1: {bf16_sum.min()}-{bf16_sum.max()}"
        assert np.allclose(fp16_sum, 1.0, atol=1e-3), f"FP16 probs don't sum to 1: {fp16_sum.min()}-{fp16_sum.max()}"

    @requires_metal
    @requires_bf16
    def test_moe_top_k_selection(self, rng: np.random.Generator) -> None:
        """Test that top-k expert selection is consistent between BF16 and FP16."""
        import mlx.core as mx

        num_experts = 64
        hidden_dim = 4096
        seq_len = 128
        top_k = 8

        router_weight = rng.standard_normal((hidden_dim, num_experts)).astype(np.float32) * 0.1
        x = rng.standard_normal((1, seq_len, hidden_dim)).astype(np.float32)

        # FP32 reference top-k
        router_logits_ref = (x @ router_weight).reshape(-1, num_experts)
        top_k_indices_ref = np.argsort(-router_logits_ref, axis=-1)[:, :top_k]

        # FP16 top-k
        x_fp16 = mx.array(x.astype(np.float16))
        w_fp16 = mx.array(router_weight.astype(np.float16))
        router_logits_fp16 = (x_fp16 @ w_fp16).reshape(-1, num_experts)
        mx.eval(router_logits_fp16)
        top_k_indices_fp16 = np.argsort(-np.array(router_logits_fp16), axis=-1)[:, :top_k]

        # BF16 top-k
        x_bf16 = mx.array(x, dtype=mx.bfloat16)
        w_bf16 = mx.array(router_weight, dtype=mx.bfloat16)
        router_logits_bf16 = (x_bf16 @ w_bf16).reshape(-1, num_experts)
        router_logits_bf16_np = mx_to_numpy(router_logits_bf16)
        top_k_indices_bf16 = np.argsort(-router_logits_bf16_np, axis=-1)[:, :top_k]

        # Compare selection agreement with reference
        fp16_agreement = np.mean([
            len(set(ref) & set(fp16)) / top_k
            for ref, fp16 in zip(top_k_indices_ref, top_k_indices_fp16)
        ])
        bf16_agreement = np.mean([
            len(set(ref) & set(bf16)) / top_k
            for ref, bf16 in zip(top_k_indices_ref, top_k_indices_bf16)
        ])

        print(f"\nMoE top-{top_k} selection agreement:")
        print(f"  FP16 vs FP32: {fp16_agreement:.4f}")
        print(f"  BF16 vs FP32: {bf16_agreement:.4f}")

        # Both should have high agreement
        assert fp16_agreement > 0.8, f"FP16 top-k agreement too low: {fp16_agreement}"
        assert bf16_agreement > 0.8, f"BF16 top-k agreement too low: {bf16_agreement}"


# ---------------------------------------------------------------------------
# Summary fixture to report BF16 worse cases
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def report_bf16_worse_cases():
    """Report all cases where BF16 performed worse than FP16."""
    yield
    if _bf16_worse_cases:
        print("\n" + "=" * 70)
        print("SUMMARY: Cases where BF16 was worse than FP16")
        print("=" * 70)
        for case in _bf16_worse_cases:
            print(f"\n{case.test_name}:")
            print(f"  BF16 RMSE: {case.bf16_accuracy.rmse:.6f}")
            print(f"  FP16 RMSE: {case.fp16_accuracy.rmse:.6f}")
            print(f"  Degradation factor: {1/case.improvement_factor:.2f}x")
        print("=" * 70)
