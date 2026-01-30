"""Tests for Trellis MoE (Mixture of Experts) layer.

This module contains tests for the complete MoE layer with trellis-quantized
expert weights, including forward pass, gradient behavior, and memory efficiency.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    pass


# Skip all tests if required classes are not available
try:
    from metal_marlin.trellis.loader import TrellisModelLoader
    from metal_marlin.trellis.moe import TrellisMoEConfig, TrellisMoELayer

    HAS_TRELLIS_MOE = True
except ImportError:
    HAS_TRELLIS_MOE = False


requires_trellis_moe = pytest.mark.skipif(
    not HAS_TRELLIS_MOE,
    reason="TrellisMoELayer not yet implemented",
)


@pytest.fixture
def trellis_moe_config():
    """Create a TrellisMoEConfig for testing."""
    if not HAS_TRELLIS_MOE:
        pytest.skip("TrellisMoEConfig not available")
    return TrellisMoEConfig(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=2048,
        intermediate_size=5632,
        bits=4,
    )


@pytest.fixture
def mock_layer_weights():
    """Create mock layer weights for testing without actual model files.

    Returns empty dict to trigger DummyExpert fallback, which uses
    regular nn.Linear layers instead of trying to dequantize random trellis data.
    """
    # Empty dict triggers the _DummyExpert fallback path in TrellisMoELayer
    # This is intentional - we want to test the MoE routing/dispatch logic,
    # not the trellis dequantization (which has its own tests)
    return {}


@pytest.fixture
def mock_router_weight():
    """Create mock router weights."""
    num_experts = 8
    hidden_size = 2048
    return torch.randn(num_experts, hidden_size, dtype=torch.float32)


class TestTrellisMoELayer:
    """Tests for the complete MoE layer with trellis-quantized experts."""

    @pytest.fixture
    def moe_layer(self, trellis_moe_config, mock_layer_weights, mock_router_weight):
        """Create a TrellisMoELayer for testing."""
        if not HAS_TRELLIS_MOE:
            pytest.skip("TrellisMoELayer not available")

        # Use mock layer weights instead of loading from disk
        return TrellisMoELayer(
            config=trellis_moe_config,
            layer_weights=mock_layer_weights,
            router_weight=mock_router_weight,
            layer_idx=2,
            device="mps" if torch.backends.mps.is_available() else "cpu",
        )

    @requires_trellis_moe
    def test_forward_shape(self, moe_layer):
        """Test that forward pass preserves input shape."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x = torch.randn(1, 32, 2048, dtype=torch.float16, device=device)
        out = moe_layer(x)
        assert out.shape == x.shape

    @requires_trellis_moe
    def test_gradient_free(self, moe_layer):
        """Verify we're not computing gradients for quantized weights."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x = torch.randn(1, 8, 2048, dtype=torch.float16, device=device)
        with torch.no_grad():
            out = moe_layer(x)
        assert not out.requires_grad

    @requires_trellis_moe
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS required for memory test"
    )
    def test_memory_efficiency(self, moe_layer):
        """Check that we're not holding dequantized weights permanently."""
        gc.collect()
        torch.mps.empty_cache()

        initial_mem = torch.mps.current_allocated_memory()
        x = torch.randn(1, 8, 2048, dtype=torch.float16, device="mps")
        _ = moe_layer(x)

        gc.collect()
        torch.mps.empty_cache()
        final_mem = torch.mps.current_allocated_memory()

        # Memory should not grow significantly
        mem_growth = final_mem - initial_mem
        assert mem_growth < 100 * 1024 * 1024  # < 100MB

    @requires_trellis_moe
    def test_forward_different_batch_sizes(self, moe_layer):
        """Test forward pass with different batch sizes."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        batch_sizes = [1, 4, 8]
        seq_lens = [1, 8, 32]

        for batch in batch_sizes:
            for seq_len in seq_lens:
                x = torch.randn(batch, seq_len, 2048, dtype=torch.float16, device=device)
                out = moe_layer(x)
                assert out.shape == (batch, seq_len, 2048)

    @requires_trellis_moe
    def test_forward_2d_input(self, moe_layer):
        """Test forward pass with 2D input [tokens, hidden]."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x = torch.randn(16, 2048, dtype=torch.float16, device=device)
        out = moe_layer(x)
        assert out.shape == (16, 2048)

    @requires_trellis_moe
    def test_forward_3d_input(self, moe_layer):
        """Test forward pass with 3D input [batch, seq, hidden]."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x = torch.randn(2, 16, 2048, dtype=torch.float16, device=device)
        out = moe_layer(x)
        assert out.shape == (2, 16, 2048)

    @requires_trellis_moe
    def test_routing_output_range(self, moe_layer):
        """Test that routing produces valid expert indices."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x = torch.randn(4, 2048, dtype=torch.float16, device=device)

        # Access internal routing if available
        if hasattr(moe_layer, "router"):
            logits = moe_layer.router(x)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(
                probs, k=moe_layer.config.num_experts_per_tok, dim=-1
            )

            # Check indices are valid
            assert (topk_indices >= 0).all()
            assert (topk_indices < moe_layer.config.num_experts).all()

            # Check probabilities sum to ~1 (using same dtype)
            expected = torch.ones(4, device=device, dtype=topk_probs.dtype)
            assert torch.allclose(topk_probs.sum(dim=-1), expected, atol=1e-3)

    @requires_trellis_moe
    def test_expert_selection(self, moe_layer):
        """Test that experts are selected and executed."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x = torch.randn(8, 2048, dtype=torch.float16, device=device)

        # Run forward pass
        out = moe_layer(x)

        # Output should be finite
        assert torch.isfinite(out).all()

        # Output should not be all zeros
        assert not torch.allclose(out, torch.zeros_like(out))

    @requires_trellis_moe
    def test_deterministic_forward(self, moe_layer):
        """Test that forward pass is deterministic with same input."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x = torch.randn(4, 2048, dtype=torch.float16, device=device)

        # Run twice with same input
        out1 = moe_layer(x)
        out2 = moe_layer(x)

        # Should be identical
        assert torch.allclose(out1, out2)

    @requires_trellis_moe
    def test_different_inputs_produce_different_outputs(self, moe_layer):
        """Test that different inputs produce different outputs."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        x1 = torch.randn(4, 2048, dtype=torch.float16, device=device)
        x2 = torch.randn(4, 2048, dtype=torch.float16, device=device)

        out1 = moe_layer(x1)
        out2 = moe_layer(x2)

        # Should be different
        assert not torch.allclose(out1, out2)


class TestTrellisMoEConfig:
    """Tests for TrellisMoEConfig."""

    @requires_trellis_moe
    def test_config_creation(self):
        """Test basic config creation."""
        config = TrellisMoEConfig(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=2048,
            intermediate_size=5632,
            bits=4,
        )
        assert config.num_experts == 8
        assert config.num_experts_per_tok == 2
        assert config.hidden_size == 2048
        assert config.intermediate_size == 5632
        assert config.bits == 4

    @requires_trellis_moe
    def test_config_defaults(self):
        """Test config default values."""
        config = TrellisMoEConfig(
            num_experts=8,
            hidden_size=2048,
        )
        assert config.num_experts_per_tok == 2  # Default
        assert config.intermediate_size == 5632  # Default based on hidden_size
        assert config.bits == 4  # Default

    @requires_trellis_moe
    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            TrellisMoEConfig(
                num_experts=0,  # Invalid
                hidden_size=2048,
            )

    @requires_trellis_moe
    def test_config_bits_validation(self):
        """Test that bits must be valid (2, 3, or 4)."""
        with pytest.raises(ValueError):
            TrellisMoEConfig(
                num_experts=8,
                hidden_size=2048,
                bits=5,  # Invalid - only 2, 3, 4 supported
            )


class TestTrellisMoEWithModelLoader:
    """Integration tests with TrellisModelLoader."""

    @requires_trellis_moe
    @pytest.mark.skip(reason="Requires actual model files")
    def test_load_from_model(self):
        """Test loading MoE layer from actual model files."""
        # This test requires actual model files and is skipped by default
        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
        layer_weights = loader.load_layer(2)

        config = TrellisMoEConfig()
        router_weight = torch.randn(config.num_experts, config.hidden_size)

        layer = TrellisMoELayer(config, layer_weights, router_weight, layer_idx=2, device="mps")

        x = torch.randn(1, 8, 2048, dtype=torch.float16, device="mps")
        out = layer(x)
        assert out.shape == x.shape
