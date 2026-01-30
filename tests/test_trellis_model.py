import pytest
import torch

from metal_marlin.trellis_config import TrellisModelConfig

# Skip model tests if classes are not available
try:
    from metal_marlin.trellis_lm import TrellisForCausalLM
    from metal_marlin.trellis_model import TrellisModel

    HAS_TRELLIS_MODEL = True
except ImportError:
    HAS_TRELLIS_MODEL = False

# Skip decoder layer tests if not available
try:
    from metal_marlin.trellis_layer import TrellisDecoderLayer

    HAS_DECODER_LAYER = True
except ImportError:
    HAS_DECODER_LAYER = False

requires_trellis_model = pytest.mark.skipif(
    not HAS_TRELLIS_MODEL,
    reason="TrellisModel/TrellisForCausalLM not yet implemented",
)

requires_decoder_layer = pytest.mark.skipif(
    not HAS_DECODER_LAYER,
    reason="TrellisDecoderLayer not yet implemented",
)


class TestTrellisModel:
    @pytest.fixture
    def config(self):
        return TrellisModelConfig()

    @pytest.fixture
    def glm_config(self):
        """GLM-4.7-Flash style config: MLA, 64 experts, layer 0 dense."""
        return TrellisModelConfig(
            hidden_size=2048,
            num_hidden_layers=47,
            num_experts=64,
            num_experts_per_tok=8,
            first_moe_layer=1,
            kv_lora_rank=512,
            q_lora_rank=768,
        )

    @pytest.fixture
    def qwen_moe_config(self):
        """Qwen3-30B-A3B style config: GQA, 128 experts, all MoE."""
        return TrellisModelConfig(
            hidden_size=2048,
            num_hidden_layers=48,
            num_experts=128,
            num_experts_per_tok=8,
            first_moe_layer=0,  # All layers are MoE
            kv_lora_rank=None,  # Standard GQA, no MLA
        )

    def test_config_dense_model(self, config):
        """Default config is a dense model (no MoE)."""
        assert config.num_experts == 1
        assert not config.is_moe_layer(0)
        assert not config.is_moe_layer(10)
        assert not config.is_moe_layer(31)

    def test_config_glm_moe_layers(self, glm_config):
        """GLM-4.7-Flash: layer 0 is dense, layers 1-46 are MoE."""
        assert not glm_config.is_moe_layer(0)
        assert glm_config.is_moe_layer(1)
        assert glm_config.is_moe_layer(46)
        assert glm_config.is_mla_model()

    def test_config_qwen_moe_layers(self, qwen_moe_config):
        """Qwen3-30B-A3B: all layers are MoE."""
        assert qwen_moe_config.is_moe_layer(0)
        assert qwen_moe_config.is_moe_layer(47)
        assert not qwen_moe_config.is_mla_model()

    @requires_trellis_model
    def test_model_creation(self, config):
        model = TrellisModel(config)
        assert len(model.layers) == 0  # Layers added by from_pretrained

    @requires_decoder_layer
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_single_layer_forward(self):
        """Test forward through a single loaded layer."""
        from metal_marlin.trellis_loader import TrellisModelLoader

        config = TrellisModelConfig()
        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")

        # Load just layer 0 (dense, simpler)
        layer = TrellisDecoderLayer.from_loader(loader, config, layer_idx=0, device="mps")

        x = torch.randn(1, 16, 2048, dtype=torch.float16, device="mps")
        out = layer(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestTrellisModelMoE:
    @requires_decoder_layer
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_moe_layer_forward(self):
        """Test forward through an MoE layer."""
        from metal_marlin.trellis_layer import TrellisDecoderLayer
        from metal_marlin.trellis_loader import TrellisModelLoader

        config = TrellisModelConfig()
        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")

        # Layer 2 is first MoE layer
        # Need router weights - use random for testing
        router_weights = {2: torch.randn(config.num_experts, config.hidden_size)}

        layer = TrellisDecoderLayer.from_loader(
            loader, config, layer_idx=2, router_weights=router_weights, device="mps"
        )

        x = torch.randn(1, 8, 2048, dtype=torch.float16, device="mps")
        out = layer(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    @requires_decoder_layer
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_multiple_layers(self):
        """Test forward through multiple layers."""
        from metal_marlin.trellis_layer import TrellisDecoderLayer
        from metal_marlin.trellis_loader import TrellisModelLoader

        config = TrellisModelConfig()
        loader = TrellisModelLoader("models/GLM-4.7-Flash-EXL3-3bpw")

        # Test layers 0, 1, 2 (dense, dense, first MoE)
        x = torch.randn(1, 4, 2048, dtype=torch.float16, device="mps")

        for layer_idx in [0, 1, 2]:
            router_weights = {layer_idx: torch.randn(64, 2048)} if layer_idx >= 2 else {}
            layer = TrellisDecoderLayer.from_loader(
                loader, config, layer_idx, router_weights, device="mps"
            )
            x = layer(x)
            if hasattr(loader, "clear_layer_cache"):
                loader.clear_layer_cache(layer_idx)

        assert x.shape == (1, 4, 2048)
