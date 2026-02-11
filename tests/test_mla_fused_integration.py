"""Test MLA fused attention integration in TrellisMLAttention."""
import pytest
import torch
from metal_marlin.trellis.attention import TrellisMLAConfig


@pytest.fixture
def glm47_config():
    """GLM-4.7 MLA config (matches metal_optimization_guide.md)."""
    return TrellisMLAConfig(
        hidden_size=2048,
        num_attention_heads=20,
        num_kv_heads=20,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        kv_lora_rank=512,
        q_lora_rank=768,
    )


class TestMLAFusedIntegration:
    def test_config_dimensions(self, glm47_config):
        """Config properties match expected GLM-4.7 dimensions."""
        assert glm47_config.qk_head_dim == 256  # 192 + 64
        assert glm47_config.kv_head_dim == 448  # 192 + 256

    def test_fused_attention_flag(self, glm47_config):
        """Fused attention is enabled by default."""
        assert glm47_config.use_fused_attention is True

    def test_config_rope_theta(self, glm47_config):
        """GLM-4.7 uses 1M rope_theta."""
        assert glm47_config.rope_theta == 1_000_000.0
