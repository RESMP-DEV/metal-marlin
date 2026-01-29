"""End-to-end tests for trellis inference pipeline."""

from pathlib import Path

import pytest
import torch

MODEL_PATH = "models/GLM-4.7-Flash-EXL3-3bpw"


@pytest.fixture(scope="module")
def model_available():
    """Check if test model is available."""
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model not found: {MODEL_PATH}")
    return True


class TestTrellisE2E:
    """End-to-end tests for trellis pipeline."""

    def test_load_single_layer(self, model_available):
        """Test loading a single layer."""
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader(MODEL_PATH)
        assert loader.get_num_layers() == 47

        weights = loader.load_layer(0)
        assert len(weights) > 0

    def test_trellis_linear_forward(self, model_available):
        """Test TrellisLinear forward pass."""
        from metal_marlin.trellis_linear import TrellisLinear
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader(MODEL_PATH)
        weights = loader.load_layer(0)

        # Find a weight
        weight_name = list(weights.keys())[0]
        linear = TrellisLinear.from_trellis_weight(weights[weight_name], device="mps")

        x = torch.randn(1, linear.in_features, dtype=torch.float16, device="mps")
        y = linear(x)

        assert y.shape == (1, linear.out_features)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_dense_mlp_forward(self, model_available):
        """Test dense MLP (layer 0) forward."""
        from metal_marlin.trellis_layer import TrellisDenseMLP
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader(MODEL_PATH)
        mlp = TrellisDenseMLP.from_loader(loader, layer_idx=0, device="mps")

        x = torch.randn(1, 8, 2048, dtype=torch.float16, device="mps")
        y = mlp(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    @pytest.mark.slow
    def test_moe_layer_forward(self, model_available):
        """Test MoE layer (layer 2+) forward."""
        from metal_marlin.trellis_loader import TrellisModelLoader
        from metal_marlin.trellis_moe import TrellisMoEConfig, TrellisMoELayer

        loader = TrellisModelLoader(MODEL_PATH)

        # Use random router weights for testing
        config = TrellisMoEConfig()
        router_weight = torch.randn(config.num_experts, config.hidden_size)

        weights = loader.load_layer(2)
        moe = TrellisMoELayer(
            config, weights, router_weight, layer_idx=2, device="mps"
        )

        x = torch.randn(1, 4, 2048, dtype=torch.float16, device="mps")
        y = moe(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    @pytest.mark.slow
    def test_attention_forward(self, model_available):
        """Test attention forward."""
        from metal_marlin.trellis_attention import TrellisMLAttention

        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader(MODEL_PATH)
        attn = TrellisMLAttention.from_loader(loader, layer_idx=0, device="mps")

        x = torch.randn(1, 16, 2048, dtype=torch.float16, device="mps")
        y = attn(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()


@pytest.mark.slow
class TestTrellisE2EGeneration:
    """End-to-end generation tests (slow, require full model)."""

    def test_full_layer_stack(self, model_available):
        """Test forward through multiple layers."""
        from metal_marlin.trellis_config import TrellisModelConfig
        from metal_marlin.trellis_layer import TrellisDecoderLayer
        from metal_marlin.trellis_loader import TrellisModelLoader

        loader = TrellisModelLoader(MODEL_PATH)
        config = TrellisModelConfig()

        x = torch.randn(1, 4, 2048, dtype=torch.float16, device="mps")

        # Test first 5 layers
        for layer_idx in range(5):
            router_weights = {layer_idx: torch.randn(64, 2048)} if layer_idx >= 2 else {}
            layer = TrellisDecoderLayer.from_loader(
                loader, config, layer_idx, router_weights, device="mps"
            )
            x = layer(x)
            loader.clear_layer_cache(layer_idx)

            assert not torch.isnan(x).any(), f"NaN at layer {layer_idx}"

        assert x.shape == (1, 4, 2048)
