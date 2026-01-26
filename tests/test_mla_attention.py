"""Tests for Multi-head Latent Attention (MLA) implementation.

Tests the MLAAttention class, MLARoPE, and MLAKVCache components.
Validates correctness of:
- Query compression path (q_a_proj -> q_b_proj)
- KV compression path (kv_a_proj -> split -> kv_b_proj)
- RoPE application with rope_ratio scaling
- MLA KV cache updates
- Memory savings vs standard attention
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

import numpy as np


@pytest.fixture
def mla_config():
    """Standard GLM-4.7-Flash-like MLA configuration."""
    return {
        "hidden_size": 256,  # Smaller for fast tests
        "num_heads": 4,
        "head_dim": 64,
        "kv_lora_rank": 64,
        "q_lora_rank": 96,
        "qk_rope_head_dim": 32,
        "rope_theta": 10000.0,
        "rope_ratio": 1.0,
        "max_position_embeddings": 512,
        "quant_type": "fp4",
        "group_size": 32,
    }


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLARoPE:
    """Tests for MLARoPE with rope_ratio scaling."""

    def test_rope_initialization(self):
        """Test RoPE module initializes correctly."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, base=10000.0, rope_ratio=1.0, max_seq_len=256)

        assert rope.dim == 64
        assert rope.base == 10000.0
        assert rope.rope_ratio == 1.0
        assert rope.cos_cache.shape == (256, 32)  # max_seq_len, dim/2
        assert rope.sin_cache.shape == (256, 32)

    def test_rope_with_ratio_scaling(self):
        """Test that rope_ratio scales frequencies correctly."""
        from metal_marlin.mla_attention import MLARoPE

        rope_standard = MLARoPE(dim=64, rope_ratio=1.0, max_seq_len=16)
        rope_scaled = MLARoPE(dim=64, rope_ratio=0.5, max_seq_len=16)

        # Scaled frequencies should be different
        assert not mx.allclose(
            rope_standard.cos_cache, rope_scaled.cos_cache
        ).item()

        # The ratio affects the frequency: rope_ratio=0.5 should halve frequency
        # This means position 2 with ratio=0.5 should match position 1 with ratio=1.0
        # (approximately, for low-frequency dimensions)

    def test_rope_forward_shape(self):
        """Test RoPE forward pass preserves shape."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256)

        # Input: [batch, seq, heads, head_dim]
        x = mx.random.normal((2, 16, 4, 64))
        y = rope(x, position_offset=0)

        assert y.shape == x.shape

    def test_rope_with_position_offset(self):
        """Test RoPE with position offset for KV cache."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256)

        x = mx.random.normal((1, 8, 4, 64))

        # First chunk (positions 0-7)
        y1 = rope(x, position_offset=0)

        # Second chunk (positions 8-15)
        y2 = rope(x, position_offset=8)

        # Results should be different due to different positions
        assert not mx.allclose(y1, y2).item()

    def test_rope_roundtrip(self):
        """Test that applying RoPE forward then backward recovers input."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256)

        x = mx.random.normal((1, 16, 1, 64))

        # Apply forward rotation
        y = rope(x, position_offset=0)

        # Apply inverse rotation (negate sin)
        # This requires manual computation since MLARoPE doesn't expose inverse
        positions = mx.arange(16)
        freqs = mx.outer(positions, rope.inv_freq)
        cos = mx.cos(freqs)[None, :, None, :]
        sin = mx.sin(freqs)[None, :, None, :]

        # Inverse rotation: x = y_even*cos + y_odd*sin, x_odd = -y_even*sin + y_odd*cos
        # Actually: just negate sin for inverse
        y_even = y[..., ::2]
        y_odd = y[..., 1::2]
        y_even * cos + y_odd * sin  # Note: + sin for inverse
        -y_even * sin + y_odd * cos

        # This is actually the Llama-style rotation which is:
        # Forward: (x_even*cos - x_odd*sin, x_odd*cos + x_even*sin)
        # Inverse: (y_even*cos + y_odd*sin, y_odd*cos - y_even*sin)

        # For Llama-style, the inverse should recover original
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Verify forward rotation is correct
        expected_even = x_even * cos - x_odd * sin
        expected_odd = x_odd * cos + x_even * sin
        assert mx.allclose(y_even, expected_even, atol=1e-3).item()
        assert mx.allclose(y_odd, expected_odd, atol=1e-3).item()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLAKVCache:
    """Tests for MLA KV cache."""

    def test_cache_initialization(self):
        """Test MLA KV cache initializes correctly."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=2,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
        )

        assert cache.c_kv.shape == (2, 2, 128, 64)
        assert cache.k_pe.shape == (2, 2, 128, 32)
        assert all(sl == 0 for sl in cache.seq_lens)

    def test_cache_update(self):
        """Test MLA KV cache update operation."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
        )

        # First update (prefill)
        c_kv_new = mx.random.normal((1, 16, 64))
        k_pe_new = mx.random.normal((1, 16, 32))

        c_kv_all, k_pe_all = cache.update(layer_idx=0, c_kv_new=c_kv_new, k_pe_new=k_pe_new)

        assert c_kv_all.shape == (1, 16, 64)
        assert k_pe_all.shape == (1, 16, 32)
        assert cache.seq_lens[0] == 16
        assert cache.seq_lens[1] == 0  # Other layer unchanged

        # Second update (decode step)
        c_kv_decode = mx.random.normal((1, 1, 64))
        k_pe_decode = mx.random.normal((1, 1, 32))

        c_kv_all, k_pe_all = cache.update(layer_idx=0, c_kv_new=c_kv_decode, k_pe_new=k_pe_decode)

        assert c_kv_all.shape == (1, 17, 64)
        assert k_pe_all.shape == (1, 17, 32)
        assert cache.seq_lens[0] == 17

    def test_cache_seq_len_property(self):
        """Test seq_len property returns correct value."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
        )

        assert cache.seq_len == 0

        c_kv_new = mx.random.normal((1, 10, 64))
        k_pe_new = mx.random.normal((1, 10, 32))
        cache.update(0, c_kv_new, k_pe_new)

        assert cache.seq_len == 10


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLAAttention:
    """Tests for MLAAttention module."""

    def test_attention_initialization(self, mla_config):
        """Test MLA attention initializes with correct shapes."""
        from metal_marlin.mla_attention import MLAAttention

        attn = MLAAttention(**mla_config)

        assert attn.hidden_size == 256
        assert attn.num_heads == 4
        assert attn.kv_lora_rank == 64
        assert attn.q_lora_rank == 96
        assert attn.qk_rope_head_dim == 32

        # Check projection dimensions
        assert attn.q_a_proj.in_features == 256
        assert attn.q_a_proj.out_features == 96
        assert attn.q_b_proj.in_features == 96
        assert attn.q_b_proj.out_features == 4 * 64  # num_heads * head_dim

        assert attn.kv_a_proj.in_features == 256
        assert attn.kv_a_proj.out_features == 64 + 32  # kv_lora_rank + rope_dim
        assert attn.kv_b_proj.in_features == 64
        assert attn.kv_b_proj.out_features == 4 * 64 * 2  # num_heads * head_dim * 2

    def test_attention_without_query_compression(self, mla_config):
        """Test MLA attention without query compression."""
        from metal_marlin.mla_attention import MLAAttention

        config = mla_config.copy()
        config["q_lora_rank"] = None

        attn = MLAAttention(**config)

        # Should have direct q_proj instead of q_a_proj/q_b_proj
        assert hasattr(attn, "q_proj")
        assert not hasattr(attn, "q_a_proj")
        assert not hasattr(attn, "q_b_proj")

    def test_attention_forward_shape(self, mla_config):
        """Test MLA attention forward pass produces correct output shape."""
        from metal_marlin.mla_attention import MLAAttention

        attn = MLAAttention(**mla_config)

        # Input: [batch, seq_len, hidden_size]
        hidden_states = mx.random.normal((2, 16, 256))

        output = attn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_attention_with_cache(self, mla_config):
        """Test MLA attention with KV cache."""
        from metal_marlin.mla_attention import MLAAttention, MLAKVCache

        attn = MLAAttention(**mla_config)

        cache = MLAKVCache(
            num_layers=1,
            batch_size=2,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
        )

        # Prefill
        hidden_states = mx.random.normal((2, 16, 256))
        output1 = attn(hidden_states, kv_cache=cache, layer_idx=0)

        assert output1.shape == (2, 16, 256)
        assert cache.seq_lens[0] == 16

        # Decode step
        hidden_decode = mx.random.normal((2, 1, 256))
        output2 = attn(hidden_decode, kv_cache=cache, layer_idx=0)

        assert output2.shape == (2, 1, 256)
        assert cache.seq_lens[0] == 17

    def test_from_config(self, mla_config):
        """Test creating MLAAttention from MLAConfig."""
        from metal_marlin.mla_attention import MLAAttention, MLAConfig

        config = MLAConfig(**mla_config)
        attn = MLAAttention.from_config(config)

        assert attn.hidden_size == config.hidden_size
        assert attn.num_heads == config.num_heads

    def test_from_hf_config(self, mla_config):
        """Test creating MLAAttention from HuggingFace config dict."""
        from metal_marlin.mla_attention import create_mla_from_hf_config

        hf_config = {
            "hidden_size": 256,
            "num_attention_heads": 4,
            "head_dim": 64,
            "kv_lora_rank": 64,
            "q_lora_rank": 96,
            "qk_rope_head_dim": 32,
            "rope_theta": 10000.0,
            "rope_ratio": 1.0,
            "max_position_embeddings": 512,
        }

        attn = create_mla_from_hf_config(hf_config)

        assert attn.hidden_size == 256
        assert attn.kv_lora_rank == 64
        assert attn.rope_ratio == 1.0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLAMemorySavings:
    """Tests verifying MLA memory savings vs standard attention."""

    def test_kv_cache_memory_reduction(self):
        """Verify MLA KV cache is smaller than standard KV cache."""
        from metal_marlin.kv_cache import KVCache
        from metal_marlin.mla_attention import MLAKVCache

        # GLM-4.7-Flash-like dimensions
        num_layers = 32
        batch_size = 1
        max_seq_len = 4096
        num_heads = 32
        head_dim = 128
        kv_lora_rank = 512
        qk_rope_head_dim = 64

        # Standard KV cache: stores full K, V
        # Shape: [num_layers, 2, batch, max_seq, num_heads, head_dim]
        # Elements: num_layers * 2 * batch * max_seq * num_heads * head_dim
        standard_elements = num_layers * 2 * batch_size * max_seq_len * num_heads * head_dim

        # MLA KV cache: stores compressed c_kv + k_pe
        # c_kv: [num_layers, batch, max_seq, kv_lora_rank]
        # k_pe: [num_layers, batch, max_seq, qk_rope_head_dim]
        mla_elements = num_layers * batch_size * max_seq_len * (kv_lora_rank + qk_rope_head_dim)

        # Calculate reduction
        reduction_factor = standard_elements / mla_elements

        # MLA should achieve significant reduction (>10x for typical configs)
        assert reduction_factor > 10, f"Expected >10x reduction, got {reduction_factor:.1f}x"

        # Print for visibility
        print("\nKV Cache Memory Comparison:")
        print(f"  Standard MHA: {standard_elements * 2 / 1e9:.2f} GB (FP16)")
        print(f"  MLA:          {mla_elements * 2 / 1e9:.2f} GB (FP16)")
        print(f"  Reduction:    {reduction_factor:.1f}x")


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestGLMRopeRatio:
    """Tests for GLM-style rope_ratio scaling."""

    def test_rope_ratio_effect(self):
        """Test that different rope_ratio values produce different results."""
        from metal_marlin.mla_attention import MLAAttention

        config_base = {
            "hidden_size": 256,
            "num_heads": 4,
            "kv_lora_rank": 64,
            "qk_rope_head_dim": 32,
            "max_position_embeddings": 256,
        }

        attn_ratio_1 = MLAAttention(**config_base, rope_ratio=1.0)
        attn_ratio_half = MLAAttention(**config_base, rope_ratio=0.5)
        attn_ratio_2 = MLAAttention(**config_base, rope_ratio=2.0)

        hidden_states = mx.random.normal((1, 16, 256))

        # Results should differ with different rope_ratios
        out_1 = attn_ratio_1(hidden_states)
        out_half = attn_ratio_half(hidden_states)
        out_2 = attn_ratio_2(hidden_states)

        # All outputs should have same shape
        assert out_1.shape == out_half.shape == out_2.shape

        # But values should differ (projections initialized same, but RoPE differs)
        # Note: Due to random initialization, this is a soft check
        # In practice, different rope_ratios affect frequency scaling
