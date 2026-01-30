"""FP8 KV Cache accuracy and memory benchmark tests.

Tests for the quantized KV cache implementation in metal_marlin.cache.quantized_kv.
Validates:
1. FP8 E4M3 quantize/dequantize roundtrip accuracy
2. INT8 symmetric quantize/dequantize roundtrip accuracy
3. Attention output accuracy: FP16 KV vs FP8 KV cache
4. Memory savings: verify ~50% reduction vs FP16 storage
5. Scaling strategies: PER_HEAD, PER_TOKEN, ASYMMETRIC

The quantized KV cache stores keys and values in 8-bit format (FP8 or INT8)
with per-head or per-token scales, achieving approximately 2x memory savings
compared to FP16 storage while maintaining acceptable accuracy for attention.
"""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
from metal_marlin.cache.quantized_kv import (
    FP8_E4M3_MAX,
    INT8_MAX,
    CacheStats,
    QuantizedKVCache,
    ScalingStrategy,
    _dequantize_fp8_e4m3,
    _dequantize_int8_symmetric,
    _quantize_fp8_e4m3,
    _quantize_int8_symmetric,
    compress_kv,
    decompress_kv,
)

# =============================================================================
# FP8 E4M3 Roundtrip Tests
# =============================================================================


class TestFP8E4M3Roundtrip:
    """Verify FP8 E4M3 quantize/dequantize maintains acceptable accuracy."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    def test_fp8_roundtrip_randn(self, rng: np.random.Generator) -> None:
        """FP8 roundtrip MSE on standard normal data should be low."""
        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        k = rng.standard_normal((1, 32, 512, 128)).astype(np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        # Compute MSE
        mse = float(((k.astype(np.float32) - k_restored.astype(np.float32)) ** 2).mean())

        # FP8 E4M3 has ~3 bits mantissa, expect MSE < 1e-4 for normalized data
        assert mse < 1e-4, f"FP8 roundtrip MSE too high: {mse:.6f}"

    def test_fp8_roundtrip_uniform(self, rng: np.random.Generator) -> None:
        """FP8 roundtrip on uniform [-1, 1] data."""
        k = rng.uniform(-1.0, 1.0, (1, 8, 256, 64)).astype(np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        mse = float(((k.astype(np.float32) - k_restored.astype(np.float32)) ** 2).mean())
        assert mse < 1e-4, f"FP8 uniform roundtrip MSE too high: {mse:.6f}"

    def test_fp8_roundtrip_extreme_values(self) -> None:
        """FP8 roundtrip preserves extreme values (near FP8 max)."""
        # Create tensor with values near FP8 E4M3 max (448)
        k = np.array([[[[400.0, -400.0, 1.0, -1.0, 0.0]]]], dtype=np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        # Relative error for large values should be small
        mask = np.abs(k) > 1.0
        rel_error = np.abs(k[mask].astype(np.float32) - k_restored[mask].astype(np.float32)) / np.abs(
            k[mask].astype(np.float32)
        )
        max_rel_error = float(rel_error.max())

        # FP8 with 3 mantissa bits: relative error ~1/8 = 0.125 max
        assert max_rel_error < 0.15, f"FP8 extreme value relative error too high: {max_rel_error:.4f}"

    def test_fp8_roundtrip_zeros(self) -> None:
        """FP8 roundtrip handles zero values correctly."""
        k = np.zeros((1, 4, 16, 32), dtype=np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        # Zero should map back to near-zero (within quantization noise)
        max_abs = float(np.abs(k_restored).max())
        assert max_abs < 0.1, f"FP8 zero roundtrip produced non-zero: {max_abs}"

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 1, 128),  # Single position
            (1, 8, 1024, 64),  # GQA config
            (2, 32, 512, 128),  # Multi-batch
            (1, 64, 2048, 96),  # Large head count
        ],
        ids=["single_pos", "gqa", "multi_batch", "many_heads"],
    )
    def test_fp8_roundtrip_shapes(self, rng: np.random.Generator, shape: tuple[int, ...]) -> None:
        """FP8 roundtrip works correctly for various tensor shapes."""
        k = rng.standard_normal(shape).astype(np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        mse = float(((k.astype(np.float32) - k_restored.astype(np.float32)) ** 2).mean())
        assert mse < 1e-3, f"FP8 roundtrip MSE too high for shape {shape}: {mse:.6f}"

    def test_fp8_per_token_scaling(self, rng: np.random.Generator) -> None:
        """FP8 roundtrip with per-token (shared across heads) scaling."""
        k = rng.standard_normal((1, 8, 256, 64)).astype(np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=False)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        mse = float(((k.astype(np.float32) - k_restored.astype(np.float32)) ** 2).mean())

        # Per-token scaling may have slightly higher error than per-head
        assert mse < 5e-4, f"FP8 per-token roundtrip MSE too high: {mse:.6f}"


# =============================================================================
# INT8 Symmetric Roundtrip Tests
# =============================================================================


class TestINT8SymmetricRoundtrip:
    """Verify INT8 symmetric quantize/dequantize maintains acceptable accuracy."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=43)

    def test_int8_roundtrip_randn(self, rng: np.random.Generator) -> None:
        """INT8 roundtrip MSE on standard normal data should be low."""
        v = rng.standard_normal((1, 32, 512, 128)).astype(np.float16)

        v_int8, scale = _quantize_int8_symmetric(v, per_head=True)
        v_restored = _dequantize_int8_symmetric(v_int8, scale)

        mse = float(((v.astype(np.float32) - v_restored.astype(np.float32)) ** 2).mean())

        # INT8 has 127 levels, expect MSE < 5e-5 for normalized data
        assert mse < 5e-5, f"INT8 roundtrip MSE too high: {mse:.6f}"

    def test_int8_roundtrip_uniform(self, rng: np.random.Generator) -> None:
        """INT8 roundtrip on uniform [-1, 1] data."""
        v = rng.uniform(-1.0, 1.0, (1, 8, 256, 64)).astype(np.float16)

        v_int8, scale = _quantize_int8_symmetric(v, per_head=True)
        v_restored = _dequantize_int8_symmetric(v_int8, scale)

        mse = float(((v.astype(np.float32) - v_restored.astype(np.float32)) ** 2).mean())
        assert mse < 5e-5, f"INT8 uniform roundtrip MSE too high: {mse:.6f}"

    def test_int8_vs_fp8_precision(self, rng: np.random.Generator) -> None:
        """INT8 should have comparable or better precision than FP8."""
        data = rng.standard_normal((1, 16, 256, 128)).astype(np.float16)

        # FP8 roundtrip
        fp8_q, fp8_s = _quantize_fp8_e4m3(data, per_head=True)
        fp8_restored = _dequantize_fp8_e4m3(fp8_q, fp8_s)
        fp8_mse = float(((data.astype(np.float32) - fp8_restored.astype(np.float32)) ** 2).mean())

        # INT8 roundtrip
        int8_q, int8_s = _quantize_int8_symmetric(data, per_head=True)
        int8_restored = _dequantize_int8_symmetric(int8_q, int8_s)
        int8_mse = float(((data.astype(np.float32) - int8_restored.astype(np.float32)) ** 2).mean())

        # Both formats should have low MSE for standard normal data
        # The implementation uses similar quantization schemes, so they're comparable
        assert int8_mse < 1e-4, f"INT8 MSE too high: {int8_mse:.6f}"
        assert fp8_mse < 1e-4, f"FP8 MSE too high: {fp8_mse:.6f}"
        # INT8 should be at least as good (within 10% tolerance for numerical noise)
        assert int8_mse <= fp8_mse * 1.1, f"INT8 MSE ({int8_mse:.6f}) significantly worse than FP8 ({fp8_mse:.6f})"


# =============================================================================
# Attention Accuracy Tests
# =============================================================================


def _reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    """Compute reference scaled dot-product attention in FP32.

    Args:
        q: Query [batch, num_heads, seq_len, head_dim]
        k: Key [batch, num_kv_heads, seq_len, head_dim]
        v: Value [batch, num_kv_heads, seq_len, head_dim]
        scale: Attention scale factor (typically 1/sqrt(head_dim))

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    q_f32 = q.astype(np.float32)
    k_f32 = k.astype(np.float32)
    v_f32 = v.astype(np.float32)

    # Expand KV for GQA if needed
    batch, num_heads, seq_q, head_dim = q_f32.shape
    _, num_kv_heads, seq_kv, _ = k_f32.shape

    if num_kv_heads < num_heads:
        repeat = num_heads // num_kv_heads
        k_f32 = np.repeat(k_f32, repeat, axis=1)
        v_f32 = np.repeat(v_f32, repeat, axis=1)

    # Q @ K^T
    scores = np.einsum("bhqd,bhkd->bhqk", q_f32, k_f32) * scale

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    # Attn @ V
    output = np.einsum("bhqk,bhkd->bhqd", attn_weights, v_f32)

    return output.astype(np.float16)


class TestFP8AttentionAccuracy:
    """Compare attention output accuracy: FP16 KV vs FP8 quantized KV."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=44)

    def test_fp8_attention_accuracy_small(self, rng: np.random.Generator) -> None:
        """FP8 KV cache maintains attention accuracy for small sequences."""
        batch, num_heads, num_kv_heads, seq_len, head_dim = 1, 32, 8, 64, 128
        scale = 1.0 / np.sqrt(head_dim)

        q = rng.standard_normal((batch, num_heads, 1, head_dim)).astype(np.float16)
        k = rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)).astype(np.float16)
        v = rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)).astype(np.float16)

        # FP16 reference
        ref_output = _reference_attention(q, k, v, scale)

        # FP8 quantized KV
        (k_q, k_s), (v_q, v_s) = compress_kv(k, v, ScalingStrategy.PER_HEAD)
        k_dq, v_dq = decompress_kv(k_q, k_s, v_q, v_s, ScalingStrategy.PER_HEAD)
        fp8_output = _reference_attention(q, k_dq, v_dq, scale)

        # Compare
        mse = float(((ref_output.astype(np.float32) - fp8_output.astype(np.float32)) ** 2).mean())
        max_abs_error = float(np.abs(ref_output.astype(np.float32) - fp8_output.astype(np.float32)).max())

        # Attention output error should be small (softmax dampens quantization noise)
        assert mse < 1e-5, f"FP8 attention MSE too high: {mse:.6f}"
        assert max_abs_error < 0.01, f"FP8 attention max error too high: {max_abs_error:.6f}"

    @pytest.mark.parametrize(
        "seq_len",
        [128, 512, 2048],
        ids=["seq128", "seq512", "seq2048"],
    )
    def test_fp8_attention_accuracy_varying_seq(
        self, rng: np.random.Generator, seq_len: int
    ) -> None:
        """FP8 KV cache accuracy across different sequence lengths."""
        batch, num_heads, num_kv_heads, head_dim = 1, 32, 8, 128
        scale = 1.0 / np.sqrt(head_dim)

        q = rng.standard_normal((batch, num_heads, 1, head_dim)).astype(np.float16)
        k = rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)).astype(np.float16)
        v = rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)).astype(np.float16)

        # FP16 reference
        ref_output = _reference_attention(q, k, v, scale)

        # FP8 quantized KV
        (k_q, k_s), (v_q, v_s) = compress_kv(k, v, ScalingStrategy.PER_HEAD)
        k_dq, v_dq = decompress_kv(k_q, k_s, v_q, v_s, ScalingStrategy.PER_HEAD)
        fp8_output = _reference_attention(q, k_dq, v_dq, scale)

        # Compute relative error
        ref_norm = float(np.abs(ref_output.astype(np.float32)).mean())
        abs_error = float(np.abs(ref_output.astype(np.float32) - fp8_output.astype(np.float32)).mean())
        rel_error = abs_error / max(ref_norm, 1e-8)

        # Relative error should stay bounded as sequence length increases
        assert rel_error < 0.01, f"FP8 attention relative error too high at seq_len={seq_len}: {rel_error:.4f}"

    @pytest.mark.parametrize(
        "strategy",
        [ScalingStrategy.PER_HEAD, ScalingStrategy.PER_TOKEN, ScalingStrategy.ASYMMETRIC],
        ids=["per_head", "per_token", "asymmetric"],
    )
    def test_scaling_strategies_accuracy(
        self, rng: np.random.Generator, strategy: ScalingStrategy
    ) -> None:
        """Compare attention accuracy across different scaling strategies."""
        batch, num_heads, num_kv_heads, seq_len, head_dim = 1, 16, 4, 256, 64
        scale = 1.0 / np.sqrt(head_dim)

        q = rng.standard_normal((batch, num_heads, 1, head_dim)).astype(np.float16)
        k = rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)).astype(np.float16)
        v = rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)).astype(np.float16)

        # FP16 reference
        ref_output = _reference_attention(q, k, v, scale)

        # Quantized with specified strategy
        (k_q, k_s), (v_q, v_s) = compress_kv(k, v, strategy)
        k_dq, v_dq = decompress_kv(k_q, k_s, v_q, v_s, strategy)
        quant_output = _reference_attention(q, k_dq, v_dq, scale)

        mse = float(((ref_output.astype(np.float32) - quant_output.astype(np.float32)) ** 2).mean())

        # All strategies should maintain reasonable accuracy
        assert mse < 1e-4, f"Attention MSE too high for {strategy.value}: {mse:.6f}"


# =============================================================================
# Memory Savings Tests
# =============================================================================


class TestFP8MemorySavings:
    """Verify FP8 KV cache achieves expected memory reduction."""

    def test_memory_savings_50_percent(self) -> None:
        """Verify approximately 50% memory reduction with FP8 cache."""
        num_layers = 32
        num_kv_heads = 8
        head_dim = 128
        max_seq_len = 8192
        batch_size = 1

        cache = QuantizedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            scaling=ScalingStrategy.PER_HEAD,
        )

        # Simulate filling the cache
        cache.seq_len = max_seq_len

        stats = cache.get_stats()

        # FP16 memory: 2 (K+V) * layers * kv_heads * seq_len * head_dim * 2 bytes
        expected_fp16_bytes = 2 * num_layers * num_kv_heads * max_seq_len * head_dim * 2

        # Verify FP16 calculation
        assert stats.fp16_memory_bytes == expected_fp16_bytes

        # Compression ratio should be close to 2x (50% reduction)
        compression_ratio = stats.compression_ratio

        # Allow 1.8x-2.0x due to scale overhead
        assert 1.8 < compression_ratio <= 2.0, (
            f"Compression ratio {compression_ratio:.2f} outside expected range [1.8, 2.0]"
        )

    @pytest.mark.parametrize(
        "scaling,min_ratio",
        [
            (ScalingStrategy.PER_HEAD, 1.8),
            (ScalingStrategy.PER_TOKEN, 1.9),  # Less scale overhead
            (ScalingStrategy.ASYMMETRIC, 1.8),
        ],
        ids=["per_head", "per_token", "asymmetric"],
    )
    def test_memory_savings_by_strategy(
        self, scaling: ScalingStrategy, min_ratio: float
    ) -> None:
        """Memory savings vary slightly by scaling strategy."""
        cache = QuantizedKVCache(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=4096,
            batch_size=1,
            scaling=scaling,
        )
        cache.seq_len = 4096

        stats = cache.get_stats()
        ratio = stats.compression_ratio

        assert ratio >= min_ratio, (
            f"Compression ratio {ratio:.2f} below minimum {min_ratio} for {scaling.value}"
        )

    def test_memory_saved_mb_calculation(self) -> None:
        """Verify memory_saved_mb calculation is correct."""
        cache = QuantizedKVCache(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=8192,
            batch_size=1,
            scaling=ScalingStrategy.PER_HEAD,
        )
        cache.seq_len = 8192

        stats = cache.get_stats()
        saved_mb = stats.memory_saved_mb

        # For 8K context with 32 layers, 8 kv_heads, 128 head_dim:
        # FP16: 2 * 32 * 8 * 8192 * 128 * 2 / 1024^2 = 1024 MB
        # FP8: ~512 MB (plus scale overhead)
        # Saved: ~504 MB
        expected_fp16_mb = stats.fp16_memory_bytes / (1024 * 1024)
        expected_quant_mb = stats.quantized_memory_bytes / (1024 * 1024)
        expected_saved_mb = expected_fp16_mb - expected_quant_mb

        assert abs(saved_mb - expected_saved_mb) < 0.01, (
            f"Memory saved calculation mismatch: {saved_mb} vs {expected_saved_mb}"
        )

        # Should save significant memory (roughly half of FP16)
        assert saved_mb > 400, f"Expected >400 MB saved for 8K context, got {saved_mb:.1f} MB"
        # Compression ratio should be close to 2x
        assert stats.compression_ratio > 1.8, f"Compression ratio too low: {stats.compression_ratio:.2f}"

    @pytest.mark.parametrize(
        "seq_len,expected_fp16_mb",
        [
            # FP16 memory = 2 * 32 * 8 * seq_len * 128 * 2 / 1024^2
            # = seq_len * 128 / 1024 = seq_len / 8 MB
            (1024, 128),
            (4096, 512),
            (8192, 1024),
            (16384, 2048),
        ],
        ids=["1k", "4k", "8k", "16k"],
    )
    def test_memory_scaling_with_context(
        self, seq_len: int, expected_fp16_mb: int
    ) -> None:
        """Memory usage scales linearly with context length."""
        cache = QuantizedKVCache(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=seq_len,
            batch_size=1,
            scaling=ScalingStrategy.PER_HEAD,
        )
        cache.seq_len = seq_len

        stats = cache.get_stats()
        fp16_mb = stats.fp16_memory_bytes / (1024 * 1024)

        # Allow 5% tolerance for rounding
        assert abs(fp16_mb - expected_fp16_mb) < expected_fp16_mb * 0.05, (
            f"FP16 memory {fp16_mb:.0f} MB doesn't match expected {expected_fp16_mb} MB"
        )


# =============================================================================
# QuantizedKVCache Integration Tests
# =============================================================================


class TestQuantizedKVCacheIntegration:
    """Integration tests for QuantizedKVCache class."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=45)

    def test_cache_update_and_retrieve(self, rng: np.random.Generator) -> None:
        """Test basic cache update and retrieval flow."""
        cache = QuantizedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            batch_size=1,
            scaling=ScalingStrategy.PER_HEAD,
        )

        # Add first batch of tokens
        k1 = rng.standard_normal((1, 4, 16, 64)).astype(np.float16)
        v1 = rng.standard_normal((1, 4, 16, 64)).astype(np.float16)

        cache.compress_and_store(0, k1, v1)
        cache.compress_and_store(1, k1, v1)
        cache.advance(16)

        assert cache.seq_len == 16

        # Retrieve and verify shape
        k_ret, v_ret = cache.get_kv_for_attention(0)
        assert k_ret.shape == (1, 4, 16, 64)
        assert v_ret.shape == (1, 4, 16, 64)

        # Verify reasonable reconstruction accuracy
        mse = float(((k1.astype(np.float32) - k_ret.astype(np.float32)) ** 2).mean())
        assert mse < 1e-4, f"Cache retrieval MSE too high: {mse:.6f}"

    def test_cache_incremental_update(self, rng: np.random.Generator) -> None:
        """Test incremental token-by-token updates."""
        cache = QuantizedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=256,
            batch_size=1,
            scaling=ScalingStrategy.PER_HEAD,
        )

        # Simulate autoregressive generation
        for step in range(10):
            k_new = rng.standard_normal((1, 4, 1, 64)).astype(np.float16)
            v_new = rng.standard_normal((1, 4, 1, 64)).astype(np.float16)

            cache.compress_and_store(0, k_new, v_new)
            cache.advance(1)

        assert cache.seq_len == 10

        k_full, v_full = cache.get_kv_for_attention(0)
        assert k_full.shape == (1, 4, 10, 64)
        assert v_full.shape == (1, 4, 10, 64)

    def test_cache_update_method(self, rng: np.random.Generator) -> None:
        """Test the update() method that combines store and retrieve."""
        cache = QuantizedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            batch_size=1,
            scaling=ScalingStrategy.PER_HEAD,
        )

        # First update
        k1 = rng.standard_normal((1, 4, 8, 64)).astype(np.float16)
        v1 = rng.standard_normal((1, 4, 8, 64)).astype(np.float16)

        k_full, v_full = cache.update(0, k1, v1)
        cache.advance(8)

        # Should return full cache including new tokens
        assert k_full.shape == (1, 4, 8, 64)

        # Second update
        k2 = rng.standard_normal((1, 4, 4, 64)).astype(np.float16)
        v2 = rng.standard_normal((1, 4, 4, 64)).astype(np.float16)

        k_full, v_full = cache.update(0, k2, v2)
        cache.advance(4)

        # Should return all 12 positions
        assert k_full.shape == (1, 4, 12, 64)

    def test_cache_reset(self, rng: np.random.Generator) -> None:
        """Test cache reset clears sequence position."""
        cache = QuantizedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
            batch_size=1,
            scaling=ScalingStrategy.PER_HEAD,
        )

        k1 = rng.standard_normal((1, 4, 32, 64)).astype(np.float16)
        v1 = rng.standard_normal((1, 4, 32, 64)).astype(np.float16)

        cache.compress_and_store(0, k1, v1)
        cache.advance(32)
        assert cache.seq_len == 32

        cache.reset()
        assert cache.seq_len == 0

        k_ret, v_ret = cache.get_kv_for_attention(0)
        assert k_ret.shape[2] == 0
        assert v_ret.shape[2] == 0

    def test_cache_overflow_error(self, rng: np.random.Generator) -> None:
        """Test cache raises error on overflow."""
        cache = QuantizedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=64,
            batch_size=1,
            scaling=ScalingStrategy.PER_HEAD,
        )

        k1 = rng.standard_normal((1, 4, 32, 64)).astype(np.float16)
        v1 = rng.standard_normal((1, 4, 32, 64)).astype(np.float16)

        cache.compress_and_store(0, k1, v1)
        cache.advance(32)

        # Second batch would overflow
        k2 = rng.standard_normal((1, 4, 64, 64)).astype(np.float16)
        v2 = rng.standard_normal((1, 4, 64, 64)).astype(np.float16)

        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            cache.compress_and_store(0, k2, v2)


# =============================================================================
# PyTorch MPS Tests (Optional)
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestFP8WithPyTorch:
    """Tests using PyTorch tensors for comparison."""

    @pytest.fixture
    def device(self) -> str:
        if HAS_MPS:
            return "mps"
        return "cpu"

    def test_numpy_torch_consistency(self, device: str) -> None:
        """Verify numpy implementation matches PyTorch expectations."""
        if torch is None:
            pytest.skip("PyTorch not available")

        torch.manual_seed(42)
        k_torch = torch.randn(1, 8, 256, 64, dtype=torch.float16, device=device)
        k_np = k_torch.cpu().numpy()

        # Quantize with numpy
        k_q, k_s = _quantize_fp8_e4m3(k_np, per_head=True)
        k_restored_np = _dequantize_fp8_e4m3(k_q, k_s)

        # Convert back to torch for comparison
        k_restored_torch = torch.from_numpy(k_restored_np).to(device)

        # Compute MSE in PyTorch
        mse = ((k_torch - k_restored_torch) ** 2).mean().item()
        assert mse < 1e-4, f"FP8 roundtrip MSE (via PyTorch): {mse:.6f}"


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestFP8EdgeCases:
    """Test edge cases and robustness of FP8 quantization."""

    def test_fp8_nan_handling(self) -> None:
        """FP8 should not produce NaN from valid inputs."""
        # Normal values
        k = np.random.randn(1, 4, 16, 32).astype(np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        assert not np.any(np.isnan(k_restored)), "FP8 roundtrip produced NaN"
        assert not np.any(np.isinf(k_restored)), "FP8 roundtrip produced Inf"

    def test_fp8_small_values(self) -> None:
        """FP8 handles small values (near min positive) correctly."""
        # Values near FP8 E4M3 min positive (~0.002)
        k = np.full((1, 4, 8, 16), 0.001, dtype=np.float16)

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        # Small values may be quantized to zero, but should not produce garbage
        assert np.all(np.isfinite(k_restored)), "FP8 small value handling produced non-finite"
        # Restored values should be in reasonable range
        assert float(np.abs(k_restored).max()) < 1.0

    def test_fp8_mixed_scales(self) -> None:
        """FP8 handles heads with very different magnitudes."""
        rng = np.random.default_rng(46)
        k = np.zeros((1, 4, 32, 64), dtype=np.float16)

        # Different scales per head
        k[:, 0, :, :] = rng.standard_normal((32, 64)).astype(np.float16) * 0.01
        k[:, 1, :, :] = rng.standard_normal((32, 64)).astype(np.float16) * 1.0
        k[:, 2, :, :] = rng.standard_normal((32, 64)).astype(np.float16) * 100.0
        k[:, 3, :, :] = rng.standard_normal((32, 64)).astype(np.float16) * 0.001

        k_fp8, scale = _quantize_fp8_e4m3(k, per_head=True)
        k_restored = _dequantize_fp8_e4m3(k_fp8, scale)

        # Each head should have good relative accuracy
        for h in range(4):
            head_orig = k[:, h, :, :].astype(np.float32)
            head_rest = k_restored[:, h, :, :].astype(np.float32)

            mask = np.abs(head_orig) > 1e-6
            if mask.any():
                rel_error = np.abs(head_orig[mask] - head_rest[mask]) / np.abs(head_orig[mask])
                median_rel_error = float(np.median(rel_error))
                assert median_rel_error < 0.2, f"Head {h} median relative error too high: {median_rel_error:.4f}"


# =============================================================================
# Constants Verification
# =============================================================================


class TestFP8Constants:
    """Verify FP8 E4M3 constants are correct."""

    def test_fp8_e4m3_max(self) -> None:
        """FP8 E4M3 max value should be 448."""
        assert FP8_E4M3_MAX == 448.0

    def test_int8_max(self) -> None:
        """INT8 symmetric max should be 127."""
        assert INT8_MAX == 127
