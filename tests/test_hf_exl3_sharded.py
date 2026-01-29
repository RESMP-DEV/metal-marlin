"""Test sharded model conversion for EXL3 loader."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from metal_marlin.hf_exl3_loader import (
    _parse_layer_index,
    convert_sharded_to_layerwise,
    detect_exl3_format,
)


def create_mock_sharded_model(model_dir: Path, num_shards: int = 2, num_layers: int = 4):
    """Create a mock sharded EXL3 model for testing.
    
    Args:
        model_dir: Directory to create the model in
        num_shards: Number of shards to create
        num_layers: Number of transformer layers
    """
    # Create config
    config = {
        "quantization": {"format": "exl3"},
        "num_hidden_layers": num_layers,
        "hidden_size": 128,
        "intermediate_size": 256,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create tokenizer files
    with open(model_dir / "tokenizer.json", "w") as f:
        json.dump({"version": "1.0"}, f)
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump({"model_type": "bpe"}, f)

    # Distribute layers across shards
    layers_per_shard = num_layers // num_shards

    for shard_idx in range(num_shards):
        tensors = {}

        # Add some base weights to first shard
        if shard_idx == 0:
            tensors["model.embed_tokens.weight"] = torch.randn(1000, 128)
            tensors["model.norm.weight"] = torch.randn(128)
            tensors["lm_head.weight"] = torch.randn(1000, 128)

        # Add layer weights
        start_layer = shard_idx * layers_per_shard
        end_layer = start_layer + layers_per_shard

        for layer_idx in range(start_layer, end_layer):
            prefix = f"model.layers.{layer_idx}"
            tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(128, 128)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(128, 128)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(128, 128)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(128, 128)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(256, 128)
            tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(256, 128)
            tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(128, 256)
            tensors[f"{prefix}.input_layernorm.weight"] = torch.randn(128)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(128)

        # Save shard
        shard_file = model_dir / f"model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
        save_file(tensors, shard_file)


class TestParseLayerIndex:
    """Test the _parse_layer_index helper function."""

    def test_parses_standard_layer_names(self):
        """Test parsing standard transformer layer names."""
        assert _parse_layer_index("model.layers.42.mlp.gate_proj.weight") == 42
        assert _parse_layer_index("model.layers.0.self_attn.q_proj.weight") == 0
        assert _parse_layer_index("model.layers.123.input_layernorm.weight") == 123

    def test_returns_none_for_non_layer_tensors(self):
        """Test that non-layer tensors return None."""
        assert _parse_layer_index("model.embed_tokens.weight") is None
        assert _parse_layer_index("model.norm.weight") is None
        assert _parse_layer_index("lm_head.weight") is None
        assert _parse_layer_index("some_other_tensor") is None

    def test_handles_edge_cases(self):
        """Test edge cases in layer parsing."""
        # Large layer indices
        assert _parse_layer_index("model.layers.999.mlp.down_proj.weight") == 999
        # Single digit
        assert _parse_layer_index("model.layers.9.output.weight") == 9


class TestConvertShardedToLayerwise:
    """Test the convert_sharded_to_layerwise function."""

    def test_converts_sharded_model_correctly(self, tmp_path: Path):
        """Test full conversion of sharded model to layerwise format."""
        # Create mock model
        model_dir = tmp_path / "mock_model"
        model_dir.mkdir()
        create_mock_sharded_model(model_dir, num_shards=2, num_layers=4)

        # Detect format
        metadata = detect_exl3_format(model_dir)
        assert metadata["format"] == "sharded"
        assert metadata["num_shards"] == 2

        # Convert
        output_path = tmp_path / "converted"
        result = convert_sharded_to_layerwise(model_dir, metadata, output_path)

        # Verify output path
        assert result == output_path
        assert output_path.exists()

        # Verify config was copied
        assert (output_path / "config.json").exists()
        with open(output_path / "config.json") as f:
            config = json.load(f)
            assert config["num_hidden_layers"] == 4

        # Verify tokenizer files were copied
        assert (output_path / "tokenizer.json").exists()
        assert (output_path / "tokenizer_config.json").exists()

        # Verify layer directories created
        for i in range(4):
            layer_dir = output_path / f"layer_{i:04d}"
            assert layer_dir.exists(), f"Layer directory {layer_dir} not found"
            assert (layer_dir / "tensor_0001.safetensors").exists()
            assert (layer_dir / "index.json").exists()

            # Verify index content
            with open(layer_dir / "index.json") as f:
                index = json.load(f)
                assert index["layer_idx"] == i
                assert index["total_tensors"] == 9  # 4 attn + 3 mlp + 2 norms
                assert len(index["tensors"]) == 9

        # Verify base weights
        assert (output_path / "base_weights.safetensors").exists()
        assert (output_path / "base_index.json").exists()
        with open(output_path / "base_index.json") as f:
            base_index = json.load(f)
            assert base_index["type"] == "base_weights"
            assert base_index["total_tensors"] == 3  # embed, norm, lm_head

    def test_uses_default_output_path(self, tmp_path: Path):
        """Test that default output path is model_path/layerwise."""
        model_dir = tmp_path / "mock_model"
        model_dir.mkdir()
        create_mock_sharded_model(model_dir, num_shards=1, num_layers=2)

        metadata = detect_exl3_format(model_dir)
        result = convert_sharded_to_layerwise(model_dir, metadata)

        assert result == model_dir / "layerwise"
        assert result.exists()

    def test_single_shard_conversion(self, tmp_path: Path):
        """Test conversion with a single shard."""
        model_dir = tmp_path / "mock_model"
        model_dir.mkdir()
        create_mock_sharded_model(model_dir, num_shards=1, num_layers=4)

        metadata = detect_exl3_format(model_dir)
        # Single file that's not layerwise should be detected as single, not sharded
        # We need to force sharded detection by naming it appropriately

        # Rename to look like sharded
        shard_file = list(model_dir.glob("*.safetensors"))[0]
        shard_file.rename(model_dir / "model-00001-of-00001.safetensors")

        metadata = detect_exl3_format(model_dir)
        assert metadata["format"] == "sharded"

        output_path = tmp_path / "converted"
        convert_sharded_to_layerwise(model_dir, metadata, output_path)

        # Verify all 4 layers created
        for i in range(4):
            assert (output_path / f"layer_{i:04d}").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
