"""Validate trellis dequantization quality against reference."""

import numpy as np
import pytest
import torch


class TestDequantQuality:
    """Test that dequantization produces reasonable values."""

    def test_output_range(self):
        """Verify dequantized weights are in reasonable range."""
        from metal_marlin.trellis_linear import TrellisLinear
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
        weights = loader.load_layer(0)

        for name, weight in list(weights.items())[:3]:
            linear = TrellisLinear.from_trellis_weight(weight, device="mps")
            dequant = linear.dequantize()

            # Check range is reasonable (not all zeros, not exploding)
            assert dequant.abs().max() < 10, f"{name} has extreme values"
            assert dequant.abs().mean() > 1e-5, f"{name} is near zero"

            # Check no NaN/Inf
            assert not torch.isnan(dequant).any(), f"{name} has NaN"
            assert not torch.isinf(dequant).any(), f"{name} has Inf"

    def test_deterministic(self):
        """Verify dequantization is deterministic."""
        from metal_marlin.trellis_linear import TrellisLinear
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
        weights = loader.load_layer(0)
        weight = list(weights.values())[0]

        linear1 = TrellisLinear.from_trellis_weight(weight, device="mps")
        linear2 = TrellisLinear.from_trellis_weight(weight, device="mps")

        d1 = linear1.dequantize(use_cache=False)
        d2 = linear2.dequantize(use_cache=False)

        assert torch.allclose(d1, d2), "Dequant is not deterministic"

    def test_forward_consistency(self):
        """Verify forward pass produces consistent results."""
        from metal_marlin.trellis_linear import TrellisLinear
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
        weights = loader.load_layer(0)
        weight = list(weights.values())[0]

        linear = TrellisLinear.from_trellis_weight(weight, device="mps")
        x = torch.randn(1, linear.in_features, dtype=torch.float16, device="mps")

        y1 = linear(x)
        y2 = linear(x)

        assert torch.allclose(y1, y2, rtol=1e-3, atol=1e-3), "Forward not consistent"


@pytest.mark.slow
class TestModelQuality:
    """Higher-level quality tests (slower)."""

    def test_layer_output_distribution(self):
        """Check layer outputs have reasonable distribution."""
        from metal_marlin.trellis_config import TrellisModelConfig
        from metal_marlin.trellis_layer import TrellisDecoderLayer
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")
        config = TrellisModelConfig()

        layer = TrellisDecoderLayer.from_loader(
            loader, config, layer_idx=0, device="mps"
        )

        x = torch.randn(1, 32, 2048, dtype=torch.float16, device="mps")
        y = layer(x)

        # Check output statistics
        assert y.mean().abs() < 5, "Mean too large"
        assert 0.1 < y.std() < 10, f"Std out of range: {y.std()}"
