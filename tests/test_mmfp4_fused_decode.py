"""Tests for fixed MMFP4 fused MoE decode kernel."""
import pytest
import torch

from metal_marlin.layers.mmfp4_expert import MMFP4Expert


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
class TestMMFP4FusedDecode:
    """Test the tiled fused MoE kernel for GLM-4.7 dimensions."""

    @pytest.fixture
    def glm47_expert(self):
        """Create MMFP4Expert with GLM-4.7 dimensions."""
        return MMFP4Expert(
            hidden_size=2048,
            moe_intermediate_size=1536,
            group_size=128,
            use_fused=True,  # Test the fixed fused path
        ).to("mps")

    def test_fused_decode_no_nan(self, glm47_expert):
        """Verify fused kernel doesn't produce NaN/Inf."""
        x = torch.randn(1, 2048, dtype=torch.float16, device="mps")
        output = glm47_expert(x)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert output.shape == (1, 2048)

    def test_fused_vs_standard_parity(self, glm47_expert):
        """Verify fused kernel matches standard 3-kernel path."""
        x = torch.randn(1, 2048, dtype=torch.float16, device="mps")
        
        # Fused path
        glm47_expert.use_fused = True
        fused_out = glm47_expert(x).clone()
        
        # Standard path
        glm47_expert.use_fused = False
        standard_out = glm47_expert(x)
        
        # Allow small numerical differences
        torch.testing.assert_close(fused_out, standard_out, rtol=1e-2, atol=1e-2)

    def test_intermediate_size_1536(self, glm47_expert):
        """Verify kernel handles full intermediate_size=1536."""
        assert glm47_expert.intermediate_size == 1536
        x = torch.randn(1, 2048, dtype=torch.float16, device="mps")
        output = glm47_expert(x)
        assert output.shape == (1, 2048)

    def test_fused_batch_8(self, glm47_expert):
        """Verify fused kernel with batch size 8."""
        x = torch.randn(8, 2048, dtype=torch.float16, device="mps")
        
        # Fused path
        glm47_expert.use_fused = True
        fused_out = glm47_expert(x).clone()
        
        # Standard path
        glm47_expert.use_fused = False
        standard_out = glm47_expert(x)
        
        torch.testing.assert_close(fused_out, standard_out, rtol=1e-2, atol=1e-2)
