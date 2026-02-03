"""Tests for decode hot path buffer reuse."""

import pytest
import torch


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
class TestDecodeBufferReuse:
    """Tests for buffer reuse during decode (batch=1)."""

    @pytest.fixture
    def moe_layer(self):
        from metal_marlin.trellis.testing import create_mock_moe_mlp
        return create_mock_moe_mlp(
            hidden_dim=256,
            intermediate_dim=512,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device="mps",
        )

    def test_buffer_pool_reused_across_calls(self, moe_layer):
        """Test that buffer pool is reused across multiple forward calls."""
        x = torch.randn(1, 256, dtype=torch.float16, device="mps")

        # First forward - creates buffer pool
        with torch.no_grad():
            _ = moe_layer(x)

        pool_after_first = moe_layer._buffer_pool
        assert pool_after_first is not None

        # Second forward - should reuse same pool
        with torch.no_grad():
            _ = moe_layer(x)

        assert moe_layer._buffer_pool is pool_after_first, \
            "Buffer pool should be reused across calls"

    def test_weight_buffers_not_recreated(self, moe_layer):
        """Test that weight buffers are not recreated on each forward."""
        x = torch.randn(1, 256, dtype=torch.float16, device="mps")

        # Get initial buffers (created in __init__ for eager mode)
        initial_buffers = moe_layer._cached_weight_buffers
        assert initial_buffers is not None

        # Run multiple forwards
        for _ in range(5):
            with torch.no_grad():
                _ = moe_layer(x)

        # Buffers should be the exact same object
        assert moe_layer._cached_weight_buffers is initial_buffers, \
            "Weight buffers should not be recreated during decode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
