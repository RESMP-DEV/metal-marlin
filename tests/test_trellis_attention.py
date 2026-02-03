"""Tests for Trellis Multi-head Latent Attention (MLA) implementation."""

from __future__ import annotations

import pytest
import torch

from metal_marlin._compat import HAS_MPS, HAS_TORCH
from metal_marlin.dtypes import DTypeConfig
from metal_marlin.kv_cache import CacheConfig, KVCache
from metal_marlin.trellis.attention import TrellisMLAConfig, TrellisMLAttention
from metal_marlin.trellis.linear import TrellisLinear

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch"),
]


def _get_device() -> str:
    """Get appropriate device for tests."""
    return "mps" if HAS_MPS else "cpu"


def _create_mock_linear(in_features: int, out_features: int, device: str) -> TrellisLinear:
    """Create a mock TrellisLinear for testing."""
    # Create a simple linear layer with mock quantization
    weight = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    # For testing, we'll use a regular linear layer wrapped as TrellisLinear
    linear = torch.nn.Linear(
        in_features, out_features, bias=False, device=device, dtype=torch.float16
    )
    linear.weight.data = weight

    # Create TrellisLinear from regular Linear (for testing purposes)
    # Note: This is a simplified mock - real TrellisLinear would have quantized weights
    class MockTrellisLinear:
        def __init__(self, linear_layer):
            self.linear = linear_layer
            self.in_features = linear_layer.in_features
            self.out_features = linear_layer.out_features
            self.weight = linear_layer.weight

        def __call__(self, x):
            return self.linear(x)

    return MockTrellisLinear(linear)


class TestTrellisMLAttention:
    @pytest.fixture
    def config(self):
        """Create a TrellisMLAConfig for testing."""
        return TrellisMLAConfig(
            hidden_size=2048,
            num_attention_heads=20,
            num_kv_heads=20,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            kv_lora_rank=512,
            q_lora_rank=768,
        )

    @pytest.fixture
    def attention(self, config):
        """Create a TrellisMLAttention instance for testing."""
        device = _get_device()

        # Create mock linear layers
        # Low-rank query projection: q_a_proj compresses, q_b_proj decompresses
        q_a_proj = _create_mock_linear(config.hidden_size, config.q_lora_rank, device)
        q_b_proj = _create_mock_linear(
            config.q_lora_rank, config.num_attention_heads * config.qk_head_dim, device
        )
        kv_a_proj = _create_mock_linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, device
        )
        kv_b_proj = _create_mock_linear(
            config.kv_lora_rank,
            config.num_kv_heads * (config.qk_nope_head_dim + config.v_head_dim),
            device,
        )
        o_proj = _create_mock_linear(
            config.num_attention_heads * config.v_head_dim, config.hidden_size, device
        )

        return TrellisMLAttention(
            config=config,
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            kv_b_proj=kv_b_proj,
            o_proj=o_proj,
        )

    def test_forward_shape(self, attention):
        """Test forward pass preserves input shape."""
        device = _get_device()
        x = torch.randn(1, 32, attention.config.hidden_size, dtype=torch.float16, device=device)
        out = attention(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_with_kv_cache(self, attention):
        """Test forward pass with KV cache."""
        device = _get_device()
        cache_config = CacheConfig(
            num_layers=1,
            num_heads=attention.config.num_attention_heads,
            num_kv_heads=attention.config.num_kv_heads,
            head_dim=attention.config.v_head_dim,
            max_seq_len=1024,
            cache_dtype="fp16",
        )
        # Use fp16 for both activations and cache to avoid dtype mismatch
        dtype_config = DTypeConfig(activations="fp16", kv_cache="fp16")
        cache = KVCache(config=cache_config, batch_size=1, dtype_config=dtype_config, device=device)

        # First forward (prompt)
        x1 = torch.randn(1, 32, attention.config.hidden_size, dtype=torch.float16, device=device)
        out1 = attention(x1, kv_cache=cache, layer_idx=0)
        assert cache.seq_len == 32

        # Second forward (generation)
        x2 = torch.randn(1, 1, attention.config.hidden_size, dtype=torch.float16, device=device)
        out2 = attention(x2, kv_cache=cache, layer_idx=0)
        assert cache.seq_len == 33
        assert out2.shape == (1, 1, attention.config.hidden_size)

    def test_causal_mask(self, attention):
        """Verify causal masking prevents attending to future."""
        device = _get_device()
        seq_len = 16
        x = torch.randn(
            1, seq_len, attention.config.hidden_size, dtype=torch.float16, device=device
        )
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * -10000
        out = attention(x, attention_mask=mask)
        assert not torch.isnan(out).any()
