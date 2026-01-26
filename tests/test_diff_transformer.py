"""Tests for Differential Transformer attention implementation.

Tests the differential attention mechanism:
    Output = softmax(Q1 @ K1^T / sqrt(d)) @ V - lambda * softmax(Q2 @ K2^T / sqrt(d)) @ V

Validates:
- Config parsing from HuggingFace format
- DifferentialAttention core computation
- DifferentialMarlinAttention with quantized projections
- Lambda parameter handling (learnable/fixed, per-head/shared)
- GQA support
- Causal masking
- Numerical accuracy against reference implementation
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from metal_marlin._compat import HAS_TORCH, torch

if HAS_TORCH and torch is not None:
    from metal_marlin.architectures import (
        DifferentialAttention,
        DifferentialAttentionConfig,
        DifferentialMarlinAttention,
        create_causal_mask,
        parse_diff_transformer_config,
    )
else:
    pytest.skip("PyTorch required for differential attention tests", allow_module_level=True)


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Tests for DifferentialAttentionConfig parsing."""

    def test_parse_basic_config(self):
        """Parse a basic differential transformer config."""
        config_dict = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "diff_attn_lambda_init": 0.8,
            "diff_attn_lambda_learnable": True,
            "rope_theta": 10000.0,
        }

        config = parse_diff_transformer_config(config_dict)

        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128  # 4096 / 32
        assert config.lambda_init == 0.8
        assert config.lambda_learnable is True
        assert config.rope_theta == 10000.0

    def test_parse_config_defaults(self):
        """Verify default values are applied."""
        config_dict = {
            "hidden_size": 2048,
            "num_attention_heads": 16,
        }

        config = parse_diff_transformer_config(config_dict)

        assert config.num_key_value_heads == 16  # defaults to num_heads
        assert config.head_dim == 128
        assert config.lambda_init == 0.8  # default
        assert config.lambda_learnable is True  # default
        assert config.lambda_per_head is False  # default
        assert config.sublayer_norm is False  # default

    def test_parse_config_with_all_options(self):
        """Parse config with all differential attention options."""
        config_dict = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "diff_attn_lambda_init": 0.5,
            "diff_attn_lambda_learnable": False,
            "diff_attn_lambda_per_head": True,
            "diff_attn_sublayer_norm": True,
            "rope_theta": 50000.0,
            "max_position_embeddings": 8192,
            "attention_bias": True,
        }

        config = parse_diff_transformer_config(config_dict)

        assert config.num_key_value_heads == 4
        assert config.lambda_init == 0.5
        assert config.lambda_learnable is False
        assert config.lambda_per_head is True
        assert config.sublayer_norm is True
        assert config.rope_theta == 50000.0
        assert config.max_position_embeddings == 8192
        assert config.use_bias is True

    def test_config_validation_gqa(self):
        """Verify GQA validation (num_heads divisible by num_kv_heads)."""
        # Valid GQA
        config = DifferentialAttentionConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,  # 32 / 8 = 4 (valid)
        )
        assert config.num_key_value_heads == 8

        # Invalid GQA should raise
        with pytest.raises(ValueError, match="divisible"):
            DifferentialAttentionConfig(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=7,  # 32 / 7 is not integer
            )


# ---------------------------------------------------------------------------
# DifferentialAttention core tests
# ---------------------------------------------------------------------------


class TestDifferentialAttentionCore:
    """Tests for DifferentialAttention core computation."""

    @pytest.fixture
    def basic_config(self):
        """Basic test configuration."""
        return {
            "batch_size": 2,
            "num_heads": 4,
            "num_kv_heads": 4,
            "seq_len": 8,
            "head_dim": 64,
        }

    def reference_diff_attention(
        self,
        q1: np.ndarray,
        k1: np.ndarray,
        v: np.ndarray,
        q2: np.ndarray,
        k2: np.ndarray,
        lambda_val: float,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reference NumPy implementation of differential attention."""
        batch, num_heads, seq_q, head_dim = q1.shape
        _seq_k = k1.shape[2]  # Unused but documents expected shape
        scale = head_dim**-0.5

        # Compute attention scores for both paths
        scores1 = (q1 @ k1.transpose(0, 1, 3, 2)) * scale
        scores2 = (q2 @ k2.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores1 = scores1 + mask
            scores2 = scores2 + mask

        # Stable softmax
        def softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        weights1 = softmax(scores1)
        weights2 = softmax(scores2)

        # Compute outputs
        attn1 = weights1 @ v
        attn2 = weights2 @ v

        # Differential output
        return attn1 - lambda_val * attn2

    def test_basic_differential_attention(self, basic_config):
        """Test basic differential attention computation."""
        batch = basic_config["batch_size"]
        heads = basic_config["num_heads"]
        seq = basic_config["seq_len"]
        dim = basic_config["head_dim"]

        # Create random inputs
        np.random.seed(42)
        q1 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k1 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        q2 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k2 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        lambda_val = 0.8

        # Reference computation
        ref_output = self.reference_diff_attention(q1, k1, v, q2, k2, lambda_val)

        # Metal Marlin computation (disable fused kernel for testing without Metal shader)
        attn = DifferentialAttention(
            num_heads=heads,
            head_dim=dim,
            lambda_init=lambda_val,
            lambda_learnable=False,
            use_fused_kernel=False,
        )

        output = attn(
            torch.tensor(q1),
            torch.tensor(k1),
            torch.tensor(v),
            torch.tensor(q2),
            torch.tensor(k2),
        )
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        # Verify shape
        assert output_np.shape == (batch, heads, seq, dim)

        # Verify accuracy
        np.testing.assert_allclose(output_np, ref_output, rtol=1e-3, atol=1e-4)

    def test_lambda_zero(self, basic_config):
        """Lambda=0 should equal standard attention on path 1."""
        batch = basic_config["batch_size"]
        heads = basic_config["num_heads"]
        seq = basic_config["seq_len"]
        dim = basic_config["head_dim"]

        np.random.seed(42)
        q1 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k1 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        q2 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k2 = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        # Reference: standard attention (lambda=0 means path 2 is ignored)
        scale = dim**-0.5
        scores = (q1 @ k1.transpose(0, 1, 3, 2)) * scale

        def softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        weights = softmax(scores)
        ref_output = weights @ v

        # Differential attention with lambda=0 (disable fused kernel for testing)
        attn = DifferentialAttention(
            num_heads=heads,
            head_dim=dim,
            lambda_init=0.0,
            lambda_learnable=False,
            use_fused_kernel=False,
        )

        output = attn(
            torch.tensor(q1),
            torch.tensor(k1),
            torch.tensor(v),
            torch.tensor(q2),
            torch.tensor(k2),
        )
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        np.testing.assert_allclose(output_np, ref_output, rtol=1e-3, atol=1e-4)

    def test_lambda_one_identical_paths(self, basic_config):
        """Lambda=1 with identical Q1/K1 and Q2/K2 should give near-zero output."""
        batch = basic_config["batch_size"]
        heads = basic_config["num_heads"]
        seq = basic_config["seq_len"]
        dim = basic_config["head_dim"]

        np.random.seed(42)
        q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        # When Q1=Q2 and K1=K2 with lambda=1:
        # output = softmax(Q@K^T)@V - 1.0 * softmax(Q@K^T)@V = 0
        attn = DifferentialAttention(
            num_heads=heads,
            head_dim=dim,
            lambda_init=1.0,
            lambda_learnable=False,
            use_fused_kernel=False,
        )

        output = attn(
            torch.tensor(q),
            torch.tensor(k),
            torch.tensor(v),
            torch.tensor(q),
            torch.tensor(k),  # Same Q and K
        )
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        # Output should be near zero
        assert np.abs(output_np).max() < 1e-5

    def test_per_head_lambda(self, basic_config):
        """Test per-head lambda values."""
        batch = basic_config["batch_size"]
        heads = basic_config["num_heads"]
        seq = basic_config["seq_len"]
        dim = basic_config["head_dim"]

        np.random.seed(42)
        q1 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k1 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        q2 = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k2 = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        # Different lambda per head
        attn = DifferentialAttention(
            num_heads=heads,
            head_dim=dim,
            lambda_init=0.5,
            lambda_learnable=False,
            lambda_per_head=True,
            use_fused_kernel=False,
        )

        output = attn(
            torch.tensor(q1),
            torch.tensor(k1),
            torch.tensor(v),
            torch.tensor(q2),
            torch.tensor(k2),
        )
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        # Just verify it runs without error
        assert output_np.shape == (batch, heads, seq, dim)

    def test_gqa_differential_attention(self):
        """Test differential attention with GQA (fewer KV heads)."""
        batch = 2
        num_q_heads = 8
        num_kv_heads = 2
        seq = 8
        dim = 64

        np.random.seed(42)
        q1 = np.random.randn(batch, num_q_heads, seq, dim).astype(np.float32)
        k1 = np.random.randn(batch, num_kv_heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, num_kv_heads, seq, dim).astype(np.float32)
        q2 = np.random.randn(batch, num_q_heads, seq, dim).astype(np.float32)
        k2 = np.random.randn(batch, num_kv_heads, seq, dim).astype(np.float32)

        # Expand KV for reference
        repeat_factor = num_q_heads // num_kv_heads
        k1_expanded = np.repeat(k1, repeat_factor, axis=1)
        k2_expanded = np.repeat(k2, repeat_factor, axis=1)
        v_expanded = np.repeat(v, repeat_factor, axis=1)

        ref_output = self.reference_diff_attention(
            q1, k1_expanded, v_expanded, q2, k2_expanded, 0.8
        )

        # Metal Marlin (handles GQA internally)
        attn = DifferentialAttention(
            num_heads=num_q_heads,
            head_dim=dim,
            lambda_init=0.8,
            lambda_learnable=False,
            use_fused_kernel=False,
        )

        output = attn(
            torch.tensor(q1),
            torch.tensor(k1),
            torch.tensor(v),
            torch.tensor(q2),
            torch.tensor(k2),
        )
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        np.testing.assert_allclose(output_np, ref_output, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# DifferentialMarlinAttention tests
# ---------------------------------------------------------------------------


class TestDifferentialMarlinAttention:
    """Tests for the full attention layer with quantized projections."""

    def test_creation_from_config(self):
        """Create layer from config."""
        config = DifferentialAttentionConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            lambda_init=0.8,
            lambda_learnable=True,
        )

        attn = DifferentialMarlinAttention.from_config(config)

        assert attn.hidden_size == 512
        assert attn.num_heads == 8
        assert attn.num_kv_heads == 2
        assert attn.head_dim == 64

    def test_forward_shape(self):
        """Test output shape from forward pass."""
        batch = 2
        seq_len = 16
        hidden_size = 256
        num_heads = 4

        attn = DifferentialMarlinAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            quant_type="fp4",
            group_size=32,
            use_fused_kernel=False,
        )

        x = torch.randn(batch, seq_len, hidden_size)
        output = attn(x)
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        assert output_np.shape == (batch, seq_len, hidden_size)

    def test_with_causal_mask(self):
        """Test with causal attention mask."""
        batch = 2
        seq_len = 8
        hidden_size = 256
        num_heads = 4

        attn = DifferentialMarlinAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            lambda_init=0.8,
            quant_type="fp4",
            group_size=32,
            use_fused_kernel=False,
        )

        x = torch.randn(batch, seq_len, hidden_size)
        mask = create_causal_mask(seq_len, device=x.device)

        output = attn(x, attention_mask=mask)
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        assert output_np.shape == (batch, seq_len, hidden_size)

    def test_gqa_layer(self):
        """Test full layer with GQA."""
        batch = 2
        seq_len = 16
        hidden_size = 512
        num_heads = 8
        num_kv_heads = 2

        attn = DifferentialMarlinAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            quant_type="fp4",
            group_size=32,
            use_fused_kernel=False,
        )

        x = torch.randn(batch, seq_len, hidden_size)
        output = attn(x)
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        assert output_np.shape == (batch, seq_len, hidden_size)


# ---------------------------------------------------------------------------
# Causal mask tests
# ---------------------------------------------------------------------------


class TestCausalMask:
    """Tests for causal mask creation."""

    def test_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        seq_len = 8
        mask = create_causal_mask(seq_len, device=torch.device("cpu"))

        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.asarray(mask)

        assert mask_np.shape == (1, 1, seq_len, seq_len)

    def test_causal_mask_values(self):
        """Test causal mask has correct values (upper triangle is -inf)."""
        seq_len = 4
        mask = create_causal_mask(seq_len, device=torch.device("cpu"))

        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze()
        else:
            mask_np = np.asarray(mask).squeeze()

        # Lower triangle and diagonal should be 0
        for i in range(seq_len):
            for j in range(i + 1):
                assert mask_np[i, j] == 0.0

        # Upper triangle should be -inf
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask_np[i, j] == float("-inf")

    def test_single_token_mask(self):
        """Single token should return None (no masking needed)."""
        mask = create_causal_mask(1)
        assert mask is None


# ---------------------------------------------------------------------------
# Lambda parameter tests
# ---------------------------------------------------------------------------


class TestLambdaParameter:
    """Tests for lambda parameter handling."""

    def test_learnable_lambda(self):
        """Test that learnable lambda is stored as log and exp'd."""
        attn = DifferentialAttention(
            num_heads=4,
            head_dim=64,
            lambda_init=0.8,
            lambda_learnable=True,
            use_fused_kernel=False,
        )

        lambda_val = attn.get_lambda()
        if isinstance(lambda_val, torch.Tensor):
            lambda_np = lambda_val.detach().cpu().numpy()
        else:
            lambda_np = np.asarray(lambda_val)

        # Should be close to 0.8
        np.testing.assert_allclose(lambda_np, 0.8, rtol=1e-5)

    def test_fixed_lambda(self):
        """Test fixed lambda is stored directly."""
        attn = DifferentialAttention(
            num_heads=4,
            head_dim=64,
            lambda_init=0.5,
            lambda_learnable=False,
            use_fused_kernel=False,
        )

        lambda_val = attn.get_lambda()
        if isinstance(lambda_val, torch.Tensor):
            lambda_np = lambda_val.detach().cpu().numpy()
        else:
            lambda_np = np.asarray(lambda_val)

        np.testing.assert_allclose(lambda_np, 0.5, rtol=1e-5)

    def test_per_head_lambda_shape(self):
        """Test per-head lambda has correct shape."""
        num_heads = 8
        attn = DifferentialAttention(
            num_heads=num_heads,
            head_dim=64,
            lambda_init=0.8,
            lambda_learnable=True,
            lambda_per_head=True,
            use_fused_kernel=False,
        )

        lambda_val = attn.get_lambda()
        if isinstance(lambda_val, torch.Tensor):
            shape = tuple(lambda_val.shape)
        else:
            shape = np.asarray(lambda_val).shape

        assert shape == (num_heads,)

    def test_shared_lambda_shape(self):
        """Test shared lambda has correct shape."""
        attn = DifferentialAttention(
            num_heads=8,
            head_dim=64,
            lambda_init=0.8,
            lambda_learnable=True,
            lambda_per_head=False,
            use_fused_kernel=False,
        )

        lambda_val = attn.get_lambda()
        if isinstance(lambda_val, torch.Tensor):
            shape = tuple(lambda_val.shape)
        else:
            shape = np.asarray(lambda_val).shape

        assert shape == (1,)


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for numerical stability of differential attention."""

    def test_large_attention_scores(self):
        """Test stability with large attention scores."""
        batch = 2
        heads = 4
        seq = 8
        dim = 64

        # Create inputs that will produce large attention scores
        np.random.seed(42)
        q1 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 10.0
        k1 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 10.0
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        q2 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 10.0
        k2 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 10.0

        attn = DifferentialAttention(
            num_heads=heads,
            head_dim=dim,
            lambda_init=0.8,
            lambda_learnable=False,
            use_fused_kernel=False,
        )

        output = attn(
            torch.tensor(q1),
            torch.tensor(k1),
            torch.tensor(v),
            torch.tensor(q2),
            torch.tensor(k2),
        )
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        # Output should not contain NaN or Inf
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_small_inputs(self):
        """Test stability with very small inputs."""
        batch = 2
        heads = 4
        seq = 8
        dim = 64

        np.random.seed(42)
        q1 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 1e-6
        k1 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 1e-6
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 1e-6
        q2 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 1e-6
        k2 = np.random.randn(batch, heads, seq, dim).astype(np.float32) * 1e-6

        attn = DifferentialAttention(
            num_heads=heads,
            head_dim=dim,
            lambda_init=0.8,
            lambda_learnable=False,
            use_fused_kernel=False,
        )

        output = attn(
            torch.tensor(q1),
            torch.tensor(k1),
            torch.tensor(v),
            torch.tensor(q2),
            torch.tensor(k2),
        )
        output_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )

        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
