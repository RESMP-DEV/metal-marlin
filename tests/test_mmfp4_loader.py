import json
import torch
from pathlib import Path
import pytest
from safetensors.torch import save_file
from metal_marlin.mmfp4_loader import MMFP4ModelLoader

def test_mmfp4_loader_mock(tmp_path):
    # 1. Create mock safetensors shards
    shard1_path = tmp_path / "model-00001-of-00002.safetensors"
    shard2_path = tmp_path / "model-00002-of-00002.safetensors"
    
    tensors1 = {
        "model.layers.0.self_attn.q_proj.qweight": torch.zeros((128, 64), dtype=torch.int32),
        "model.layers.0.self_attn.q_proj.scales": torch.ones((1, 512), dtype=torch.float16),
    }
    tensors2 = {
        "model.layers.1.self_attn.q_proj.qweight": torch.zeros((128, 64), dtype=torch.int32),
        "model.layers.1.self_attn.q_proj.scales": torch.ones((1, 512), dtype=torch.float16),
    }
    
    save_file(tensors1, str(shard1_path))
    save_file(tensors2, str(shard2_path))
    
    # 2. Create index file
    index = {
        "weight_map": {
            "model.layers.0.self_attn.q_proj.qweight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.scales": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.qweight": "model-00002-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.scales": "model-00002-of-00002.safetensors",
        }
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)
        
    # 3. Test loader
    loader = MMFP4ModelLoader(tmp_path)
    
    # Test layer indices
    assert set(loader._layer_to_tensors.keys()) == {0, 1}
    
    # Test load_layer
    layer0 = loader.load_layer(0, device="cpu")
    assert "model.layers.0.self_attn.q_proj.qweight" in layer0
    assert "model.layers.0.self_attn.q_proj.scales" in layer0
    
    # Test get_quantized_weight
    qw, s = loader.get_quantized_weight("model.layers.0.self_attn.q_proj")
    assert qw.shape == (128, 64)
    assert s.shape == (1, 512)
    
    # Test iterator
    layers = list(loader)
    assert len(layers) == 2
    assert layers[0][0] == 0
    assert layers[1][0] == 1
    
    loader.__exit__(None, None, None)
