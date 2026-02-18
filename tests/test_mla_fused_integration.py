'''Test MLA fused attention integration in TrellisMLAttention.'''
import pytest
import torch

from metal_marlin.trellis.attention import TrellisMLAttention
from metal_marlin.trellis_config import TrellisConfig


@pytest.fixture
def glm47_config():
    return TrellisConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        q_lora_rank=512,
        kv_lora_rank=512,
        head_dim=128,
    )


class TestMLAFusedIntegration:
    def test_fused_decode_enabled(self, glm47_config):
        attn = TrellisMLAttention(glm47_config, layer_idx=0, use_fused_decode=True)
        assert attn.use_fused_decode
        
    def test_fused_matches_unfused(self, glm47_config):
        '''Fused path produces same output as unfused within tolerance.'''
        attn_fused = TrellisMLAttention(glm47_config, layer_idx=0, use_fused_decode=True)
        attn_unfused = TrellisMLAttention(glm47_config, layer_idx=0, use_fused_decode=False)
        
        # Copy weights
        attn_unfused.load_state_dict(attn_fused.state_dict())
        
        x = torch.randn(1, 1, 4096, dtype=torch.float16, device="mps")
        
        out_fused = attn_fused(x)
        out_unfused = attn_unfused(x)
        
        assert torch.allclose(out_fused, out_unfused, atol=1e-3)
