"""Tests for trellis_loader format detection."""

import json
from pathlib import Path

import pytest
import torch


class TestTrellisFormatDetection:
    """Tests for _detect_trellis_format method."""

    def test_detect_metal_marlin_format(self):
        """Should detect metal_marlin format with __indices keys."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        # Create a mock loader (we don't need a real path for this test)
        loader = object.__new__(TrellisModelLoader)

        # Metal marlin format keys
        tensor_dict = {
            "model__layers__0__mlp__down_proj__weight__indices": torch.tensor([]),
            "model__layers__0__mlp__down_proj__weight__scales": torch.tensor([]),
            "model__layers__0__mlp__down_proj__weight__su": torch.tensor([]),
            "model__layers__0__mlp__down_proj__weight__sv": torch.tensor([]),
        }

        result = loader._detect_trellis_format(tensor_dict)
        assert result == "metal_marlin"

    def test_detect_unknown_format(self):
        """Should return 'unknown' for unrecognized format."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        # Unknown format keys
        tensor_dict = {
            "some_random_key": torch.tensor([]),
            "another_key": torch.tensor([]),
        }

        result = loader._detect_trellis_format(tensor_dict)
        assert result == "unknown"


class TestTrellisBaseWeightName:
    """Tests for _get_base_weight_name method."""

    def test_metal_marlin_base_name(self):
        """Should extract base name from metal_marlin format."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        assert loader._get_base_weight_name("weight__indices") == "weight"
        assert loader._get_base_weight_name("weight__scales") == "weight"
        assert loader._get_base_weight_name("weight__su") == "weight"
        assert loader._get_base_weight_name("weight__sv") == "weight"

    def test_exllamav3_base_name(self):
        """Should extract base name from exllamav3 format."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        assert loader._get_base_weight_name("weight.indices") == "weight"
        assert loader._get_base_weight_name("weight.scales") == "weight"
        assert loader._get_base_weight_name("weight.scale") == "weight"
        assert loader._get_base_weight_name("weight.zero_point") == "weight"
        assert loader._get_base_weight_name("weight.su") == "weight"
        assert loader._get_base_weight_name("weight.sv") == "weight"
        assert loader._get_base_weight_name("weight.row_scale") == "weight"
        assert loader._get_base_weight_name("weight.col_scale") == "weight"
        assert loader._get_base_weight_name("weight.qweight") == "weight"


class TestTrellisComponentName:
    """Tests for _get_component_name method."""

    def test_metal_marlin_component(self):
        """Should extract component from metal_marlin format."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        assert loader._get_component_name("weight__indices") == "indices"
        assert loader._get_component_name("weight__scales") == "scales"
        assert loader._get_component_name("weight__su") == "su"
        assert loader._get_component_name("weight__sv") == "sv"

    def test_exllamav3_component(self):
        """Should extract component from exllamav3 format."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        assert loader._get_component_name("weight.indices") == "indices"
        assert loader._get_component_name("weight.qweight") == "indices"
        assert loader._get_component_name("weight.scales") == "scales"
        assert loader._get_component_name("weight.scale") == "scales"
        assert loader._get_component_name("weight.su") == "su"
        assert loader._get_component_name("weight.row_scale") == "su"
        assert loader._get_component_name("weight.sv") == "sv"
        assert loader._get_component_name("weight.col_scale") == "sv"


class TestTrellisInferShape:
    """Tests for _infer_shape method."""

    def test_infer_shape_from_su_sv(self):
        """Should infer shape from su/sv components."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        components = {
            "su": torch.randn(1024),
            "sv": torch.randn(4096),
        }

        shape = loader._infer_shape(components)
        assert shape == (1024, 4096)

    def test_infer_shape_from_row_col_scale(self):
        """Should infer shape from row_scale/col_scale components."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        components = {
            "row_scale": torch.randn(512),
            "col_scale": torch.randn(2048),
        }

        shape = loader._infer_shape(components)
        assert shape == (512, 2048)

    def test_infer_shape_missing_components(self):
        """Should return (0, 0) for missing components."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        components = {}

        shape = loader._infer_shape(components)
        assert shape == (0, 0)


def test_load_layer_from_hf_shards(tmp_path: Path) -> None:
    """Test loading a layer from HF-style sharded safetensors."""
    import numpy as np
    from metal_marlin.trellis.loader import TrellisModelLoader
    from safetensors.numpy import save_file

    # Create mock quantized tensor (4-bit, 64x64 weight)
    bits = 4
    K, N = 64, 64
    tiles_k, tiles_n = 4, 4
    packed_bytes = 128  # 4-bit: 256 indices * 4 / 8 = 128 bytes

    # Pack with header byte
    indices_data = np.zeros(1 + tiles_k * tiles_n *
                            packed_bytes, dtype=np.uint8)
    indices_data[0] = bits  # Header

    tensors = {
        "model__layers__0__mlp__gate_proj__weight__indices": indices_data,
        "model__layers__0__mlp__gate_proj__weight__scales": np.ones((1, K), dtype=np.float32),
        "model__layers__0__mlp__gate_proj__weight__su": np.ones(N, dtype=np.float32),
        "model__layers__0__mlp__gate_proj__weight__sv": np.ones(K, dtype=np.float32),
    }

    # Save shard
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard_path))

    # Create index
    weight_map = {
        name: "model-00001-of-00001.safetensors" for name in tensors.keys()}
    index = {"metadata": {}, "weight_map": weight_map}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    # Create quantization index
    quant_index = {
        "layers": [{
            "name": "model.layers.0.mlp.gate_proj.weight",
            "bits": 4,
            "shape": [K, N],
            "mse": 0.001,
        }]
    }
    (tmp_path / "quantization_index.json").write_text(json.dumps(quant_index))

    # Load and verify
    loader = TrellisModelLoader(tmp_path)
    assert loader._format == "v3_hf_shards"

    weights = loader.load_layer(0)
    assert len(weights) == 1

    weight = list(weights.values())[0]
    assert weight.bits == 4
    assert weight.original_shape == (K, N)


def test_detect_hf_shards_format(tmp_path: Path) -> None:
    """Test format detection for HF-style shards."""
    from metal_marlin.trellis.loader import detect_trellis_format

    # Create mock HF-style structure
    index = {
        "metadata": {"format": "trellis_v3"},
        "weight_map": {
            "model__layers__0__mlp__gate_proj__weight__indices": "model-00001-of-00001.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert detect_trellis_format(tmp_path) == "v3_hf_shards"


def test_detect_v2_layers_format(tmp_path: Path) -> None:
    """Test format detection for v2 layer directories."""
    from metal_marlin.trellis.loader import detect_trellis_format

    # Create mock v2 structure
    (tmp_path / "layer_0000").mkdir()
    (tmp_path / "layer_0000" / "index.json").write_text('{"tensors": []}')

    assert detect_trellis_format(tmp_path) == "v2_layers"


def test_detect_hf_shards_format(tmp_path: Path) -> None:
    """Test format detection for HF-style shards."""
    from metal_marlin.trellis.loader import detect_trellis_format

    # Create mock HF-style structure
    index = {
        "metadata": {"format": "trellis_v3"},
        "weight_map": {
            "model__layers__0__mlp__gate_proj__weight__indices": "model-00001-of-00001.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert detect_trellis_format(tmp_path) == "v3_hf_shards"


def test_detect_v2_layers_format(tmp_path: Path) -> None:
    """Test format detection for v2 layer directories."""
    from metal_marlin.trellis.loader import detect_trellis_format

    # Create mock v2 structure
    (tmp_path / "layer_0000").mkdir()
    (tmp_path / "layer_0000" / "index.json").write_text('{"tensors": []}')

    assert detect_trellis_format(tmp_path) == "v2_layers"
