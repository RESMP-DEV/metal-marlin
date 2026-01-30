"""Tests for Multi-head Latent Attention (MLA) implementation.

Tests the MLARoPE, MLAKVCache, and memory savings calculations.
Uses PyTorch MPS for tensor operations.

Validates correctness of:
- RoPE initialization and rope_ratio scaling
- RoPE forward pass shape preservation
- MLA KV cache updates and memory properties
- Memory savings vs standard attention

NOTE: MLAAttention class tests are skipped when MarlinLinear is unavailable
(requires Metal kernel support for quantized projections).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from metal_marlin._compat import HAS_MPS, HAS_TORCH

# Skip entire module if PyTorch unavailable
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")


def _get_device() -> str:
    """Get appropriate device for tests."""
    return "mps" if HAS_MPS else "cpu"


def _check_marlin_linear_available() -> bool:
    """Check if MarlinLinear (Metal kernels) is available and usable."""
    try:
        from metal_marlin.layers import MarlinLinear

        # Check if it's the stub class
        if MarlinLinear.__doc__ and "Stub" in MarlinLinear.__doc__:
            return False
        # Try instantiation with the dimension form to verify it actually works
        # Use dimensions compatible with FP4 packing (out_features % 8 == 0)
        # and quantization groups (in_features % group_size == 0)
        _layer = MarlinLinear(128, 64, bias=False, quant_type="fp4", group_size=32)
        # If we get here, MarlinLinear is functional
        return True
    except (ImportError, RuntimeError, TypeError):
        # ImportError: module not found
        # RuntimeError: stub class or Metal unavailable
        # TypeError: signature mismatch (e.g., unexpected keyword argument)
        return False
    except Exception:
        # Any other error means something is wrong
        return False


# Marker for tests requiring MarlinLinear/Metal kernels
requires_marlin = pytest.mark.skipif(
    not _check_marlin_linear_available(), reason="Requires MarlinLinear (Metal kernel support)"
)


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


@pytest.fixture
def device() -> str:
    """Return appropriate device."""
    return _get_device()


# ==============================================================================
# MLARoPE Tests (PyTorch-based implementation)
# ==============================================================================


class TestMLARoPE:
    """Tests for MLARoPE with rope_ratio scaling.

    These tests use the RoPE implementation directly without needing
    MarlinLinear or Metal kernel support.
    """

    def test_rope_initialization(self):
        """Test RoPE module initializes correctly."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, base=10000.0, rope_ratio=1.0, max_seq_len=256)

        assert rope.dim == 64
        assert rope.base == 10000.0
        assert rope.rope_ratio == 1.0
        # Check cos/sin caches are registered buffers with correct shape
        assert rope.cos_cache.shape == (256, 32)  # max_seq_len, dim/2
        assert rope.sin_cache.shape == (256, 32)
        assert rope.inv_freq.shape == (32,)  # dim/2

    def test_rope_with_ratio_scaling(self):
        """Test that rope_ratio scales frequencies correctly."""
        from metal_marlin.mla_attention import MLARoPE

        rope_standard = MLARoPE(dim=64, rope_ratio=1.0, max_seq_len=16)
        rope_scaled = MLARoPE(dim=64, rope_ratio=0.5, max_seq_len=16)

        # Scaled frequencies should be different
        assert not torch.allclose(rope_standard.cos_cache, rope_scaled.cos_cache)

        # The ratio affects the frequency: rope_ratio=0.5 should halve inv_freq
        torch.testing.assert_close(
            rope_scaled.inv_freq, rope_standard.inv_freq * 0.5, rtol=1e-5, atol=1e-6
        )

    def test_rope_forward_shape(self, device):
        """Test RoPE forward pass preserves shape."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256)
        rope = rope.to(device)

        # Input: [batch, seq, heads, head_dim]
        x = torch.randn(2, 16, 4, 64, device=device, dtype=torch.float16)
        y = rope(x, position_offset=0)

        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_rope_with_position_offset(self, device):
        """Test RoPE with position offset for KV cache."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256)
        rope = rope.to(device)

        x = torch.randn(1, 8, 4, 64, device=device, dtype=torch.float16)

        # First chunk (positions 0-7)
        y1 = rope(x, position_offset=0)

        # Second chunk (positions 8-15)
        y2 = rope(x, position_offset=8)

        # Results should be different due to different positions
        assert not torch.allclose(y1, y2)

    def test_rope_deterministic(self, device):
        """Test that RoPE output is deterministic."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256)
        rope = rope.to(device)

        x = torch.randn(1, 16, 1, 64, device=device, dtype=torch.float16)

        # Apply forward rotation twice with same offset - should be identical
        y1 = rope(x, position_offset=0)
        y2 = rope(x, position_offset=0)

        # Same input, same position -> same output (reproducibility)
        torch.testing.assert_close(y1, y2, atol=1e-6, rtol=1e-5)

    def test_rope_position_sensitivity(self, device):
        """Test that different positions produce different outputs."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256)
        rope = rope.to(device)

        # Single token at different positions
        x_single = torch.randn(1, 1, 1, 64, device=device, dtype=torch.float16)

        y_pos0 = rope(x_single, position_offset=0)
        y_pos10 = rope(x_single, position_offset=10)

        # Position 10 should differ from position 0
        assert not torch.allclose(y_pos0, y_pos10, atol=1e-3)

    def test_rope_cache_extension(self, device):
        """Test that RoPE cache extends automatically for longer sequences."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=16)
        rope = rope.to(device)

        # Request position beyond initial max_seq_len
        # Input shape: [..., seq_len, dim] where seq_len is the second-to-last dim
        # Using shape [batch, heads, seq_len, dim] for 4D input
        x = torch.randn(1, 1, 8, 64, device=device, dtype=torch.float16)
        y = rope(x, position_offset=20)  # Positions 20-27

        # Should have extended cache
        assert rope.max_seq_len >= 28
        assert rope.cos_cache.shape[0] >= 28
        assert y.shape == x.shape


# ==============================================================================
# MLAKVCache Tests (PyTorch-based implementation)
# ==============================================================================


class TestMLAKVCache:
    """Tests for MLA KV cache."""

    def test_cache_initialization(self, device):
        """Test MLA KV cache initializes correctly."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=2,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            device=device,
        )

        # Check shapes
        assert cache.c_kv.shape == (2, 2, 128, 64)
        assert cache.k_pe.shape == (2, 2, 128, 32)
        assert all(sl == 0 for sl in cache.seq_lens)
        assert cache.c_kv.device.type == device.replace("mps", "mps")
        assert cache.k_pe.device.type == device.replace("mps", "mps")

    def test_cache_update(self, device):
        """Test MLA KV cache update operation."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            device=device,
        )

        # First update (prefill)
        c_kv_new = torch.randn(1, 16, 64, device=device, dtype=torch.float16)
        k_pe_new = torch.randn(1, 16, 32, device=device, dtype=torch.float16)

        c_kv_all, k_pe_all = cache.update(layer_idx=0, c_kv_new=c_kv_new, k_pe_new=k_pe_new)

        assert c_kv_all.shape == (1, 16, 64)
        assert k_pe_all.shape == (1, 16, 32)
        assert cache.seq_lens[0] == 16
        assert cache.seq_lens[1] == 0  # Other layer unchanged

        # Second update (decode step)
        c_kv_decode = torch.randn(1, 1, 64, device=device, dtype=torch.float16)
        k_pe_decode = torch.randn(1, 1, 32, device=device, dtype=torch.float16)

        c_kv_all, k_pe_all = cache.update(layer_idx=0, c_kv_new=c_kv_decode, k_pe_new=k_pe_decode)

        assert c_kv_all.shape == (1, 17, 64)
        assert k_pe_all.shape == (1, 17, 32)
        assert cache.seq_lens[0] == 17

    def test_cache_seq_len_property(self, device):
        """Test seq_len property returns correct value."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            device=device,
        )

        assert cache.seq_len == 0

        c_kv_new = torch.randn(1, 10, 64, device=device, dtype=torch.float16)
        k_pe_new = torch.randn(1, 10, 32, device=device, dtype=torch.float16)
        cache.update(0, c_kv_new, k_pe_new)

        assert cache.seq_len == 10

    def test_cache_multi_layer(self, device):
        """Test cache updates work correctly across multiple layers."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=4,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            device=device,
        )

        # Update different layers with different lengths
        for layer_idx in range(4):
            seq_len = 10 + layer_idx * 2  # 10, 12, 14, 16
            c_kv_new = torch.randn(1, seq_len, 64, device=device, dtype=torch.float16)
            k_pe_new = torch.randn(1, seq_len, 32, device=device, dtype=torch.float16)
            cache.update(layer_idx, c_kv_new, k_pe_new)

        # Verify each layer has correct sequence length
        assert cache.seq_lens == [10, 12, 14, 16]

    def test_cache_reset(self, device):
        """Test cache reset functionality."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            device=device,
        )

        # Add some data
        c_kv_new = torch.randn(1, 16, 64, device=device, dtype=torch.float16)
        k_pe_new = torch.randn(1, 16, 32, device=device, dtype=torch.float16)
        cache.update(0, c_kv_new, k_pe_new)
        assert cache.seq_len == 16

        # Reset
        cache.reset()
        assert cache.seq_len == 0
        assert all(sl == 0 for sl in cache.seq_lens)

    def test_cache_memory_usage(self, device):
        """Test memory usage calculation."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=4,
            batch_size=2,
            max_seq_len=512,
            kv_lora_rank=256,
            qk_rope_head_dim=64,
            device=device,
        )

        # Initially zero
        assert cache.memory_usage_mb() == 0.0

        # Add some data to layer 0
        c_kv_new = torch.randn(2, 100, 256, device=device, dtype=torch.float16)
        k_pe_new = torch.randn(2, 100, 64, device=device, dtype=torch.float16)
        cache.update(0, c_kv_new, k_pe_new)

        # Check memory usage is non-zero
        mem_mb = cache.memory_usage_mb()
        assert mem_mb > 0.0

        # Expected: batch=2, seq=100, (kv_lora_rank + rope_dim) = 320, layers=4
        # But memory_usage_mb uses max(seq_lens) * num_layers
        # For one layer with 100 tokens: 2 * 100 * 320 * 4 * 2 / 1024 / 1024
        # Actually formula is: batch * max_seq * (rank + rope) * layers * 2 / 1024^2
        expected_mb = 2 * 100 * (256 + 64) * 4 * 2 / 1024 / 1024
        assert abs(mem_mb - expected_mb) < 0.01


# ==============================================================================
# MLAAttention Tests (Requires MarlinLinear/Metal kernels)
# ==============================================================================


@requires_marlin
class TestMLAAttention:
    """Tests for MLAAttention module.

    These tests require MarlinLinear (Metal kernel support) to be available.
    """

    def test_attention_initialization(self, mla_config):
        """Test MLA attention initializes with correct shapes."""
        from metal_marlin.mla_attention import MLAAttention

        attn = MLAAttention(**mla_config)

        assert attn.hidden_size == 256
        assert attn.num_heads == 4
        assert attn.kv_lora_rank == 64
        assert attn.q_lora_rank == 96
        assert attn.qk_rope_head_dim == 32

    def test_attention_without_query_compression(self, mla_config):
        """Test MLA attention without query compression."""
        from metal_marlin.mla_attention import MLAAttention

        config = mla_config.copy()
        config["q_lora_rank"] = None

        attn = MLAAttention(**config)

        # Should have direct q_proj instead of q_a_proj/q_b_proj
        assert attn.q_proj is not None
        assert attn.q_a_proj is None
        assert attn.q_b_proj is None

    def test_attention_forward_shape(self, mla_config, device):
        """Test MLA attention forward pass produces correct output shape."""
        from metal_marlin.mla_attention import MLAAttention

        attn = MLAAttention(**mla_config).to(device)

        # Input: [batch, seq_len, hidden_size]
        hidden_states = torch.randn(2, 16, 256, device=device, dtype=torch.float16)

        output = attn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_attention_with_cache(self, mla_config, device):
        """Test MLA attention with KV cache."""
        from metal_marlin.mla_attention import MLAAttention, MLAKVCache

        attn = MLAAttention(**mla_config).to(device)

        cache = MLAKVCache(
            num_layers=1,
            batch_size=2,
            max_seq_len=128,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            device=device,
        )

        # Prefill
        hidden_states = torch.randn(2, 16, 256, device=device, dtype=torch.float16)
        output1 = attn(hidden_states, kv_cache=cache, layer_idx=0)

        assert output1.shape == (2, 16, 256)
        assert cache.seq_lens[0] == 16

        # Decode step
        hidden_decode = torch.randn(2, 1, 256, device=device, dtype=torch.float16)
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
            "kv_lora_rank": 128,  # Must be divisible by group_size
            "q_lora_rank": 128,  # Must be divisible by group_size
            "qk_rope_head_dim": 32,
            "rope_theta": 10000.0,
            "rope_ratio": 1.0,
            "max_position_embeddings": 512,
        }

        attn = create_mla_from_hf_config(hf_config)

        assert attn.hidden_size == 256
        assert attn.kv_lora_rank == 128
        assert attn.rope_ratio == 1.0


# ==============================================================================
# Memory Savings Tests (Pure calculations - no Metal kernels needed)
# ==============================================================================


class TestMLAMemorySavings:
    """Tests verifying MLA memory savings vs standard attention."""

    def test_kv_cache_memory_reduction(self):
        """Verify MLA KV cache is smaller than standard KV cache."""
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

    def test_memory_savings_various_configs(self):
        """Test memory savings across different model configurations."""
        configs = [
            # (kv_lora_rank, qk_rope_head_dim, num_heads, head_dim, expected_min_reduction)
            (512, 64, 32, 128, 10),  # GLM-4.7-Flash
            (256, 64, 32, 128, 20),  # More aggressive compression
            (768, 64, 64, 128, 12),  # Larger model, moderate compression
        ]

        for kv_lora_rank, qk_rope_head_dim, num_heads, head_dim, min_reduction in configs:
            # Standard: num_heads * head_dim * 2 (K and V)
            standard_per_token = num_heads * head_dim * 2
            # MLA: kv_lora_rank + qk_rope_head_dim
            mla_per_token = kv_lora_rank + qk_rope_head_dim

            reduction = standard_per_token / mla_per_token

            assert reduction >= min_reduction, (
                f"Config (kv_rank={kv_lora_rank}, rope={qk_rope_head_dim}, "
                f"heads={num_heads}, dim={head_dim}): "
                f"Expected >={min_reduction}x, got {reduction:.1f}x"
            )


# ==============================================================================
# GLM RoPE Ratio Tests
# ==============================================================================


class TestGLMRopeRatio:
    """Tests for GLM-style rope_ratio scaling."""

    def test_rope_ratio_effect(self, device):
        """Test that different rope_ratio values produce different results."""
        from metal_marlin.mla_attention import MLARoPE

        rope_1 = MLARoPE(dim=64, rope_ratio=1.0, max_seq_len=256).to(device)
        rope_half = MLARoPE(dim=64, rope_ratio=0.5, max_seq_len=256).to(device)
        rope_2 = MLARoPE(dim=64, rope_ratio=2.0, max_seq_len=256).to(device)

        x = torch.randn(1, 16, 4, 64, device=device, dtype=torch.float16)

        # Results should differ with different rope_ratios
        out_1 = rope_1(x)
        out_half = rope_half(x)
        out_2 = rope_2(x)

        # All outputs should have same shape
        assert out_1.shape == out_half.shape == out_2.shape

        # But values should differ due to different frequency scaling
        assert not torch.allclose(out_1, out_half)
        assert not torch.allclose(out_1, out_2)
        assert not torch.allclose(out_half, out_2)

    def test_rope_ratio_inverse_frequency_scaling(self):
        """Test that rope_ratio correctly scales inverse frequencies."""
        from metal_marlin.mla_attention import MLARoPE

        rope_1 = MLARoPE(dim=64, rope_ratio=1.0, max_seq_len=16)
        rope_2 = MLARoPE(dim=64, rope_ratio=2.0, max_seq_len=16)

        # With rope_ratio=2.0, inverse frequencies should be 2x larger
        torch.testing.assert_close(rope_2.inv_freq, rope_1.inv_freq * 2, rtol=1e-5, atol=1e-6)

    def test_rope_ratio_frequency_halving(self):
        """Test that rope_ratio=0.5 halves the frequencies."""
        from metal_marlin.mla_attention import MLARoPE

        rope_1 = MLARoPE(dim=64, rope_ratio=1.0, max_seq_len=16)
        rope_half = MLARoPE(dim=64, rope_ratio=0.5, max_seq_len=16)

        # With rope_ratio=0.5, inverse frequencies should be half
        torch.testing.assert_close(rope_half.inv_freq, rope_1.inv_freq * 0.5, rtol=1e-5, atol=1e-6)


# ==============================================================================
# PyTorch MPS Integration Tests
# ==============================================================================


@pytest.mark.skipif(not HAS_MPS, reason="Requires PyTorch MPS")
class TestPyTorchMPSIntegration:
    """Tests for PyTorch MPS tensor operations with MLA components."""

    def test_rope_on_mps(self):
        """Test RoPE works correctly on MPS device."""
        from metal_marlin.mla_attention import MLARoPE

        rope = MLARoPE(dim=64, max_seq_len=256).to("mps")

        x = torch.randn(1, 16, 4, 64, device="mps", dtype=torch.float16)
        y = rope(x, position_offset=0)

        assert y.device.type == "mps"
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_kv_cache_on_mps(self):
        """Test MLA KV cache works correctly on MPS device."""
        from metal_marlin.mla_attention import MLAKVCache

        cache = MLAKVCache(
            num_layers=2,
            batch_size=4,
            max_seq_len=512,
            kv_lora_rank=256,
            qk_rope_head_dim=64,
            device="mps",
        )

        # Verify tensors are on MPS
        assert cache.c_kv.device.type == "mps"
        assert cache.k_pe.device.type == "mps"

        # Verify contiguous memory layout
        assert cache.c_kv.is_contiguous()
        assert cache.k_pe.is_contiguous()

        # Test update works
        c_kv_new = torch.randn(4, 16, 256, device="mps", dtype=torch.float16)
        k_pe_new = torch.randn(4, 16, 64, device="mps", dtype=torch.float16)

        c_kv_full, k_pe_full = cache.update(0, c_kv_new, k_pe_new)

        assert c_kv_full.device.type == "mps"
        assert k_pe_full.device.type == "mps"

    def test_rope_mps_cpu_consistency(self):
        """Test RoPE produces consistent results on MPS vs CPU."""
        from metal_marlin.mla_attention import MLARoPE

        rope_cpu = MLARoPE(dim=64, max_seq_len=256)
        rope_mps = MLARoPE(dim=64, max_seq_len=256).to("mps")

        x_cpu = torch.randn(1, 16, 2, 64, dtype=torch.float16)
        x_mps = x_cpu.to("mps")

        y_cpu = rope_cpu(x_cpu, position_offset=0)
        y_mps = rope_mps(x_mps, position_offset=0)

        # Results should be very close
        torch.testing.assert_close(y_cpu, y_mps.cpu(), atol=1e-3, rtol=1e-3)


# ==============================================================================
# MLAConfig Tests
# ==============================================================================


class TestMLAConfig:
    """Tests for MLAConfig dataclass."""

    def test_config_defaults(self):
        """Test MLAConfig has sensible defaults."""
        from metal_marlin.mla_attention import MLAConfig

        config = MLAConfig(hidden_size=256, num_heads=4)

        assert config.hidden_size == 256
        assert config.num_heads == 4
        assert config.kv_lora_rank == 512
        assert config.q_lora_rank == 1536
        assert config.qk_rope_head_dim == 64
        assert config.rope_theta == 10000.0
        assert config.rope_ratio == 1.0
        assert config.quant_type == "fp4"

    def test_config_custom_values(self):
        """Test MLAConfig with custom values."""
        from metal_marlin.mla_attention import MLAConfig

        config = MLAConfig(
            hidden_size=4096,
            num_heads=32,
            head_dim=128,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            rope_ratio=0.5,
            max_position_embeddings=8192,
        )

        assert config.hidden_size == 4096
        assert config.head_dim == 128
        assert config.rope_ratio == 0.5
        assert config.max_position_embeddings == 8192

    def test_config_no_query_compression(self):
        """Test MLAConfig with query compression disabled."""
        from metal_marlin.mla_attention import MLAConfig

        config = MLAConfig(
            hidden_size=256,
            num_heads=4,
            q_lora_rank=None,
        )

        assert config.q_lora_rank is None
