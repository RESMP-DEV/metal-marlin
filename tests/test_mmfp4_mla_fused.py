"""Tests for MMFP4MLA fused decode.

Tests the MMFP4 MLA (Multi-head Latent Attention) layer with fused decode kernel,
using GLM-style dimensions.
"""

import pytest
import torch

from metal_marlin._compat import HAS_MPS, torch as _torch
from metal_marlin.kv_cache import MLAKVCache
from metal_marlin.layers.mmfp4_mla import MMFP4MLA


@pytest.mark.skipif(not HAS_MPS, reason="MPS required")
class TestMMFP4MLAFused:
    """Test MMFP4MLA layer with fused decode."""

    @pytest.fixture
    def glm_mla_layer(self):
        """Create MMFP4MLA layer with GLM dimensions."""
        # GLM-4 style dimensions (scaled down for tests)
        layer = MMFP4MLA(
            hidden_size=2048,
            num_heads=16,
            num_kv_heads=16,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            group_size=128,
            rope_theta=10000.0,
            rope_ratio=1.0,
            use_fused_qkv=False,
            use_paged_attention=False,
            use_fused_decode=True,
        ).to("mps")
        layer.eval()
        return layer

    @pytest.fixture
    def mla_kv_cache(self, glm_mla_layer):
        """Create MLAKVCache for testing."""
        return MLAKVCache(
            num_layers=1,
            batch_size=1,
            max_seq_len=128,
            kv_lora_rank=glm_mla_layer.kv_lora_rank,
            qk_rope_head_dim=glm_mla_layer.qk_rope_head_dim,
            device="mps",
            quantize_mode="fp4",
        )

    def test_fused_decode_no_nan(self, glm_mla_layer, mla_kv_cache):
        """Verify fused decode doesn't produce NaN/Inf."""
        # Create input: batch=1, seq=1
        x = torch.randn(1, 1, 2048, dtype=torch.float16, device="mps")
        position_ids = torch.tensor([[0]], dtype=torch.long, device="mps")

        # Run forward
        output = glm_mla_layer(x, position_ids, kv_cache=mla_kv_cache)

        # Assert no NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_fused_decode_output_shape(self, glm_mla_layer, mla_kv_cache):
        """Verify output shape is correct for decode."""
        # Create input: batch=1, seq=1
        x = torch.randn(1, 1, 2048, dtype=torch.float16, device="mps")
        position_ids = torch.tensor([[0]], dtype=torch.long, device="mps")

        # Run forward
        output = glm_mla_layer(x, position_ids, kv_cache=mla_kv_cache)

        # Assert output shape: [batch, seq, hidden_size]
        assert output.shape == (1, 1, 2048), f"Expected shape (1, 1, 2048), got {output.shape}"

    def test_fused_decode_sequence_accumulation(self, glm_mla_layer, mla_kv_cache):
        """Verify fused decode handles sequence accumulation correctly."""
        # Simulate multiple decode steps
        for step in range(5):
            x = torch.randn(1, 1, 2048, dtype=torch.float16, device="mps")
            position_ids = torch.tensor([[step]], dtype=torch.long, device="mps")

            output = glm_mla_layer(x, position_ids, kv_cache=mla_kv_cache)

            # Assert no NaN after each step
            assert not torch.isnan(output).any(), f"Output contains NaN at step {step}"
            assert output.shape == (1, 1, 2048), f"Wrong shape at step {step}"

    def test_fused_decode_with_larger_position_ids(self, glm_mla_layer, mla_kv_cache):
        """Verify fused decode with non-zero position ids."""
        x = torch.randn(1, 1, 2048, dtype=torch.float16, device="mps")
        position_ids = torch.tensor([[10]], dtype=torch.long, device="mps")

        output = glm_mla_layer(x, position_ids, kv_cache=mla_kv_cache)

        # Assert no NaN and correct shape
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert output.shape == (1, 1, 2048)

    def test_fused_decode_dtype_preservation(self, glm_mla_layer, mla_kv_cache):
        """Verify output dtype matches input dtype."""
        x = torch.randn(1, 1, 2048, dtype=torch.float16, device="mps")
        position_ids = torch.tensor([[0]], dtype=torch.long, device="mps")

        output = glm_mla_layer(x, position_ids, kv_cache=mla_kv_cache)

        # Assert dtype is preserved
        assert output.dtype == torch.float16, f"Expected float16, got {output.dtype}"

    def test_mla_layer_creation(self):
        """Test that MMFP4MLA layer can be created with GLM dimensions."""
        layer = MMFP4MLA(
            hidden_size=2048,
            num_heads=16,
            num_kv_heads=16,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ).to("mps")

        # Verify layer attributes
        assert layer.hidden_size == 2048
        assert layer.num_heads == 16
        assert layer.num_kv_heads == 16
        assert layer.q_lora_rank == 1536
        assert layer.kv_lora_rank == 512
        assert layer.qk_nope_head_dim == 128
        assert layer.qk_rope_head_dim == 64
        assert layer.v_head_dim == 128

    def test_fused_decode_different_batch_sizes(self):
        """Test that batch=1 is required for fused decode path."""
        layer = MMFP4MLA(
            hidden_size=512,  # Smaller for faster test
            num_heads=4,
            num_kv_heads=4,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        ).to("mps")
        layer.eval()

        # Only batch=1 is supported for fused decode
        # Test with batch=1
        x = torch.randn(1, 1, 512, dtype=torch.float16, device="mps")
        position_ids = torch.tensor([[0]], dtype=torch.long, device="mps")

        cache = MLAKVCache(
            num_layers=1,
            batch_size=1,
            max_seq_len=64,
            kv_lora_rank=128,
            qk_rope_head_dim=32,
            device="mps",
            quantize_mode="fp4",
        )

        output = layer(x, position_ids, kv_cache=cache)

        assert not torch.isnan(output).any()
        assert output.shape == (1, 1, 512)
