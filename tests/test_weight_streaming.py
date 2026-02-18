import json
import pytest
import torch
import tempfile
import os
from pathlib import Path
from safetensors.torch import save_file

from metal_marlin.mmfp4_loader import MMFP4ModelLoader
from metal_marlin.memory.mmfp4_memory import WeightStreamer, WeightStreamConfig

@pytest.fixture
def dummy_model_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create dummy tensors
        tensors = {
            "layers.0.weight": torch.randn(10, 10),
            "layers.1.weight": torch.randn(20, 20)
        }
        
        # Save to safetensors
        save_file(tensors, str(tmp_path / "model.safetensors"))
        
        # Create index file
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "layers.0.weight": "model.safetensors",
                "layers.1.weight": "model.safetensors"
            }
        }
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)
            
        yield tmp_path

def test_loader_stream_weight(dummy_model_dir):
    loader = MMFP4ModelLoader(dummy_model_dir)
    
    # Test streaming a weight
    tensor_name = "layers.0.weight"
    tensor = loader.stream_weight(tensor_name)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (10, 10)
    
    # Verify values match expected loading
    expected = loader.load_tensor(tensor_name)
    assert torch.allclose(tensor, expected)

def test_weight_streamer_mmap_load(dummy_model_dir):
    config = WeightStreamConfig(enable_mmap=True)
    streamer = WeightStreamer(dummy_model_dir, config=config)
    
    # Manually get offset/size from loader to mimic streaming usage
    loader = MMFP4ModelLoader(dummy_model_dir)
    meta = loader.get_tensor_metadata("layers.1.weight")
    assert meta is not None
    
    # Test mmap loading via streamer
    tensor = streamer.stream_weight(
        "layers.1.weight",
        dummy_model_dir / "model.safetensors",
        meta.offset,
        meta.size_bytes,
        dtype=torch.float32,
        shape=meta.shape
    )
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (20, 20)
    
    # Verify values
    expected = loader.load_tensor("layers.1.weight")
    assert torch.allclose(tensor.cpu(), expected.cpu())
    
    streamer.shutdown()

if __name__ == "__main__":
    pytest.main([__file__])
