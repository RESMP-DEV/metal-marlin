"""Tests for YaRN RoPE implementation."""

from __future__ import annotations

import math

import pytest
import torch

from metal_marlin.rope import (
    YaRNConfig,
    YaRNRoPE,
    compute_yarn_cos_sin_cache,
    compute_yarn_inv_freq,
    create_rope_from_config,
    get_yarn_mscale,
)


class TestYaRNConfig:
    """Tests for YaRNConfig dataclass."""

    def test_basic_config(self):
        """Test basic config creation."""
        config = YaRNConfig(
            original_max_position=4096,
            scale_factor=4.0,
        )
        assert config.scale_factor == 4.0
        assert config.beta_fast == 32.0
        assert config.beta_slow == 1.0
        assert config.mscale == 1.0
        assert config.rope_type == "yarn"

    def test_invalid_scale_factor(self):
        """Test that scale_factor < 1 raises error."""
        with pytest.raises(ValueError, match="scale_factor must be >= 1.0"):
            YaRNConfig(original_max_position=4096, scale_factor=0.5)

    def test_invalid_beta_range(self):
        """Test that beta_fast <= beta_slow raises error."""
        with pytest.raises(ValueError, match="beta_fast .* must be > beta_slow"):
            YaRNConfig(
                original_max_position=4096,
                scale_factor=4.0,
                beta_fast=1.0,
                beta_slow=32.0,
            )

    def test_from_hf_config_yarn(self):
        """Test parsing HF config with YaRN scaling."""
        hf_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "max_position_embeddings": 16384,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 0.707,
            },
        }
        config = YaRNConfig.from_hf_config(hf_config)
        assert config is not None
        assert config.scale_factor == 4.0
        assert config.original_max_position == 4096
        assert config.beta_fast == 32.0
        assert config.mscale_all_dim == 0.707
        assert config.rope_type == "yarn"

    def test_from_hf_config_no_scaling(self):
        """Test parsing HF config without rope_scaling."""
        hf_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "max_position_embeddings": 4096,
        }
        config = YaRNConfig.from_hf_config(hf_config)
        assert config is None

    def test_from_hf_config_linear_scaling(self):
        """Test that linear scaling returns None (not YaRN)."""
        hf_config = {
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0,
            },
        }
        config = YaRNConfig.from_hf_config(hf_config)
        assert config is None

    def test_from_hf_config_legacy_fields(self):
        """Test parsing legacy top-level rope_scaling fields."""
        hf_config = {
            "rope_scaling_type": "yarn",
            "rope_scaling_factor": 4.0,
            "rope_original_max_position": 4096,
            "rope_beta_fast": 32.0,
            "rope_beta_slow": 1.0,
            "rope_mscale": 1.0,
        }
        config = YaRNConfig.from_hf_config(hf_config)
        assert config is not None
        assert config.scale_factor == 4.0


class TestGetYarnMscale:
    """Tests for mscale computation."""

    def test_mscale_no_scaling(self):
        """Test mscale with scale_factor=1 returns 1."""
        assert get_yarn_mscale(1.0) == 1.0
        assert get_yarn_mscale(0.5) == 1.0  # Below 1 also returns 1

    def test_mscale_basic(self):
        """Test basic mscale computation."""
        # mscale = 0.1 * log(4) + 1 ≈ 1.139
        mscale = get_yarn_mscale(4.0)
        expected = 0.1 * math.log(4.0) + 1.0
        assert abs(mscale - expected) < 1e-6

    def test_mscale_with_all_dim(self):
        """Test mscale with mscale_all_dim."""
        # mscale = 0.1 * 0.707 * log(4) + 1 ≈ 1.098
        mscale = get_yarn_mscale(4.0, mscale_all_dim=0.707)
        expected = 0.1 * 0.707 * math.log(4.0) + 1.0
        assert abs(mscale - expected) < 1e-6

    def test_mscale_increases_with_scale(self):
        """Test that mscale increases with larger scale factors."""
        mscale_2x = get_yarn_mscale(2.0)
        mscale_4x = get_yarn_mscale(4.0)
        mscale_8x = get_yarn_mscale(8.0)
        assert mscale_2x < mscale_4x < mscale_8x


class TestComputeYarnInvFreq:
    """Tests for inv_freq computation."""

    def test_output_shape(self):
        """Test output shape is [dim // 2]."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        inv_freq = compute_yarn_inv_freq(128, 10000.0, config, device="cpu")
        assert inv_freq.shape == (64,)

    def test_inv_freq_range(self):
        """Test inv_freq values are in expected range."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        inv_freq = compute_yarn_inv_freq(128, 10000.0, config, device="cpu")
        # Inverse frequencies should be positive and <= 1
        assert (inv_freq > 0).all()
        assert (inv_freq <= 1).all()

    def test_scale_factor_1_matches_standard(self):
        """Test that scale_factor=1 produces standard RoPE inv_freq."""
        config = YaRNConfig(original_max_position=4096, scale_factor=1.0)
        inv_freq = compute_yarn_inv_freq(128, 10000.0, config, device="cpu")

        # Standard RoPE inv_freq
        dim = 128
        base = 10000.0
        standard_inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        torch.testing.assert_close(inv_freq, standard_inv_freq)

    def test_scaled_inv_freq_modified(self):
        """Test that scaled inv_freq differs from standard."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        scaled_inv_freq = compute_yarn_inv_freq(128, 10000.0, config, device="cpu")

        config_1x = YaRNConfig(original_max_position=4096, scale_factor=1.0)
        standard_inv_freq = compute_yarn_inv_freq(128, 10000.0, config_1x, device="cpu")

        # Low-frequency components should be scaled down
        assert not torch.allclose(scaled_inv_freq, standard_inv_freq)


class TestComputeYarnCosSinCache:
    """Tests for cos/sin cache computation."""

    def test_cache_shapes(self):
        """Test cache shapes match expected."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        cos, sin = compute_yarn_cos_sin_cache(128, 1024, 10000.0, config, device="cpu")
        assert cos.shape == (1024, 64)
        assert sin.shape == (1024, 64)

    def test_cos_sin_bounded(self):
        """Test cos/sin values are properly bounded."""
        config = YaRNConfig(original_max_position=4096, scale_factor=1.0)
        cos, sin = compute_yarn_cos_sin_cache(128, 1024, 10000.0, config, device="cpu")
        # Without mscale, values should be in [-1, 1]
        assert cos.abs().max() <= 1.0
        assert sin.abs().max() <= 1.0

    def test_mscale_applied(self):
        """Test that mscale is applied to cache values."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        cos, sin = compute_yarn_cos_sin_cache(128, 1024, 10000.0, config, device="cpu")
        mscale = get_yarn_mscale(4.0)
        # Values can exceed 1 when mscale > 1
        assert cos.abs().max() <= mscale
        assert sin.abs().max() <= mscale


class TestYaRNRoPE:
    """Tests for YaRNRoPE class."""

    def test_basic_creation(self):
        """Test basic YaRNRoPE creation."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        rope = YaRNRoPE(dim=128, max_seq_len=8192, config=config, device="cpu")
        assert rope.dim == 128
        assert rope.max_seq_len == 8192

    def test_apply_shape(self):
        """Test apply preserves input shape."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        rope = YaRNRoPE(dim=128, max_seq_len=8192, config=config, device="cpu")

        x = torch.randn(2, 8, 64, 128)  # [batch, heads, seq, dim]
        out = rope.apply(x)
        assert out.shape == x.shape

    def test_apply_dtype_preserved(self):
        """Test apply preserves input dtype."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        rope = YaRNRoPE(dim=128, max_seq_len=8192, config=config, device="cpu")

        x = torch.randn(2, 8, 64, 128, dtype=torch.float16)
        out = rope.apply(x)
        assert out.dtype == torch.float16

    def test_apply_with_offset(self):
        """Test apply with position offset."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        rope = YaRNRoPE(dim=128, max_seq_len=8192, config=config, device="cpu")

        x = torch.randn(2, 8, 10, 128)
        out0 = rope.apply(x, offset=0)
        out100 = rope.apply(x, offset=100)
        # Different offsets should produce different results
        assert not torch.allclose(out0, out100)

    def test_extend_cache(self):
        """Test cache extension."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        rope = YaRNRoPE(dim=128, max_seq_len=1024, config=config, device="cpu")
        assert rope.cos_cache.shape[0] == 1024

        rope.extend_cache(2048)
        assert rope.cos_cache.shape[0] == 2048

    def test_no_config_uses_standard_rope(self):
        """Test that None config produces standard RoPE."""
        rope = YaRNRoPE(dim=128, max_seq_len=4096, config=None, device="cpu")
        assert rope.config.scale_factor == 1.0


class TestCreateRopeFromConfig:
    """Tests for create_rope_from_config factory."""

    def test_with_yarn_config(self):
        """Test factory with YaRN config."""
        hf_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "max_position_embeddings": 16384,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
            },
        }
        rope = create_rope_from_config(hf_config, device="cpu")
        assert rope is not None
        assert rope.config.scale_factor == 4.0

    def test_without_scaling_returns_none(self):
        """Test factory returns None without rope_scaling."""
        hf_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "max_position_embeddings": 4096,
        }
        rope = create_rope_from_config(hf_config, device="cpu")
        assert rope is None


class TestRoPECorrectness:
    """Tests for mathematical correctness of RoPE."""

    def test_rotation_properties(self):
        """Test that RoPE rotation preserves vector norms."""
        config = YaRNConfig(original_max_position=4096, scale_factor=1.0)
        rope = YaRNRoPE(dim=128, max_seq_len=4096, config=config, device="cpu")

        x = torch.randn(1, 1, 10, 128)
        x_rotated = rope.apply(x)

        # L2 norm should be approximately preserved (within mscale factor)
        x_norms = x.norm(dim=-1)
        rotated_norms = x_rotated.norm(dim=-1)
        torch.testing.assert_close(x_norms, rotated_norms, rtol=1e-4, atol=1e-4)

    def test_position_sensitivity(self):
        """Test that different positions get different rotations."""
        config = YaRNConfig(original_max_position=4096, scale_factor=4.0)
        rope = YaRNRoPE(dim=128, max_seq_len=8192, config=config, device="cpu")

        # Same vector at different positions
        v = torch.randn(1, 1, 1, 128)
        v_expanded = v.expand(1, 1, 5, 128).clone()

        rotated = rope.apply(v_expanded)

        # Each position should have a unique rotation
        for i in range(5):
            for j in range(i + 1, 5):
                assert not torch.allclose(rotated[:, :, i], rotated[:, :, j])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
