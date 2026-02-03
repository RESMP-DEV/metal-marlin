"""Tests for memory optimization features."""

import gc
import os
import resource

import psutil
import pytest
import torch


def get_rss_mb() -> float:
    """Get current RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_maxrss_mb() -> float:
    """Get current process max RSS (Resident Set Size) in MB."""
    # resource.getrusage returns maxrss in bytes on Linux, KB on macOS
    import platform

    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        # macOS returns bytes
        return usage.ru_maxrss / (1024 * 1024)
    else:
        # Linux returns KB
        return usage.ru_maxrss / 1024


class TestTrellisMoEMLPEagerBuffers:
    """Tests for TrellisMoEMLP eager buffer creation."""

    @pytest.fixture
    def mock_moe_layer(self):
        """Create a small mock MoE layer for testing."""
        from metal_marlin.trellis.testing import create_mock_moe_mlp

        return create_mock_moe_mlp(
            hidden_dim=256,
            intermediate_dim=512,
            num_experts=4,
            num_experts_per_tok=2,
            bits=3,
            device="mps",
        )

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_eager_buffers_creates_metal_buffers(self, mock_moe_layer):
        """Test that eager buffer creation populates Metal buffers in __init__."""
        # With eager_buffers=True (default), buffers are created during __init__
        assert mock_moe_layer._cached_weight_buffers is not None

        # Verify it's the correct type with expected fields
        buffers = mock_moe_layer._cached_weight_buffers
        assert hasattr(buffers, 'gate_weights')
        assert hasattr(buffers, 'up_weights')
        assert hasattr(buffers, 'down_weights')

        # Calling _get_cached_buffers should return the same cached instance
        same_buffers = mock_moe_layer._get_cached_buffers()
        assert same_buffers is buffers  # Same object, not recreated

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_eager_buffers_frees_stacked_tensors(self, mock_moe_layer):
        """Test that eager mode skips stacked tensor creation for memory efficiency.

        In eager_buffers=True mode, the layer creates Metal buffers directly
        from CPU tensors without creating intermediate stacked MPS tensors.
        This saves ~2x memory during model loading.
        """
        # Eager mode should NOT create stacked tensors
        assert not hasattr(mock_moe_layer, "gate_weights_stacked"), \
            "Stacked tensors should not exist in eager mode"
        assert not hasattr(mock_moe_layer, "up_weights_stacked"), \
            "Stacked tensors should not exist in eager mode"

        # But Metal buffers should exist
        assert mock_moe_layer._cached_weight_buffers is not None
        assert mock_moe_layer._cached_weight_buffers.gate_weights is not None

    @pytest.fixture
    def lazy_moe_layer(self):
        """Create MoE layer with lazy buffer creation (deprecated path)."""
        import warnings

        from metal_marlin.trellis.testing import create_mock_moe_mlp

        # Suppress the deprecation warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return create_mock_moe_mlp(
                hidden_dim=256,
                intermediate_dim=512,
                num_experts=4,
                num_experts_per_tok=2,
                bits=3,
                device="mps",
                eager_buffers=False,  # Test legacy path
            )

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_lazy_buffers_creates_stacked_tensors(self, lazy_moe_layer):
        """Test that lazy mode (eager_buffers=False) creates stacked tensors.

        This is the deprecated path but should still work for compatibility.
        """
        # Lazy mode should create stacked tensors
        assert hasattr(lazy_moe_layer, "gate_weights_stacked")
        assert lazy_moe_layer.gate_weights_stacked is not None

        # But cached buffers should be None until first forward
        assert lazy_moe_layer._cached_weight_buffers is None

        # After calling _get_cached_buffers, buffers should be created
        lazy_moe_layer._get_cached_buffers()
        assert lazy_moe_layer._cached_weight_buffers is not None


class TestMemoryBaseline:
    """Baseline memory tests (without model loading)."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_cpu_to_metal_buffer_no_mps_copy(self):
        """Test that CPU→Metal doesn't create MPS copy."""
        import Metal

        from metal_marlin.metal_dispatch import mps_tensor_to_metal_buffer

        device = Metal.MTLCreateSystemDefaultDevice()

        gc.collect()
        torch.mps.empty_cache()
        mps_before = torch.mps.current_allocated_memory()

        # Create CPU tensor, move to MPS, then convert to Metal buffer
        t_cpu = torch.randn(1000, 1000, dtype=torch.float32)
        t_mps = t_cpu.to("mps")
        torch.mps.synchronize()

        mps_after_move = torch.mps.current_allocated_memory()

        # Convert MPS tensor to Metal buffer - this should NOT create another MPS copy
        buf = mps_tensor_to_metal_buffer(t_mps, device)
        torch.mps.synchronize()

        mps_after_buf = torch.mps.current_allocated_memory()

        # The buffer conversion should not significantly increase MPS allocation
        # Allow small overhead but not a full 4MB copy
        delta = mps_after_buf - mps_after_move
        assert delta < 1024 * 1024, (  # < 1MB increase
            f"MPS allocation increased by {delta / (1024*1024):.2f}MB during buffer creation"
        )


@pytest.mark.skipif(
    not os.path.exists("models/GLM-4.7-Flash-Trellis-3bpw"),
    reason="Model not available"
)
class TestRealModelMemory:
    """Integration tests with real model."""

    def test_model_load_memory_bounded(self):
        """Test that model loading stays within memory bounds."""
        from metal_marlin.trellis.model import TrellisForCausalLM

        gc.collect()
        torch.mps.empty_cache()

        before_rss = get_rss_mb()

        # Load with memory optimization
        model = TrellisForCausalLM.from_pretrained(
            "models/GLM-4.7-Flash-Trellis-3bpw",
            device="mps",
            optimize_memory=True,
        )

        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        gc.collect()

        after_rss = get_rss_mb()
        delta_gb = (after_rss - before_rss) / 1024

        # GLM-4.7-Flash is ~16 GB on disk (47 layers, 64 experts each)
        # Memory includes: weights + Metal buffers + activation overhead
        # Expect ~18-20 GB total after optimization
        assert delta_gb < 20, f"Memory increased by {delta_gb:.2f} GB, expected < 20 GB"

        # Verify model works - use [batch, seq_len, hidden_size]
        hidden_size = 2048  # From model config
        x = torch.randn(1, 8, hidden_size, dtype=torch.float16, device="mps")
        with torch.no_grad():
            # Just run first layer as smoke test
            layer = model.model.layers[2]  # First MoE layer
            out = layer(x)
            assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
            assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
