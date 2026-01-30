"""Tests for trellis_loader format detection."""

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

    def test_detect_exllamav3_gptq_format(self):
        """Should detect exllamav3_gptq format with .qweight keys."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        # ExllamaV3 GPTQ format keys
        tensor_dict = {
            "model.layers.0.mlp.down_proj.weight.qweight": torch.tensor([]),
            "model.layers.0.mlp.down_proj.weight.scale": torch.tensor([]),
        }

        result = loader._detect_trellis_format(tensor_dict)
        assert result == "exllamav3_gptq"

    def test_detect_exllamav3_exl2_format(self):
        """Should detect exllamav3_exl2 format with .scale and .zero_point."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        # ExllamaV3 EXL2 format keys
        tensor_dict = {
            "model.layers.0.mlp.down_proj.weight.scale": torch.tensor([]),
            "model.layers.0.mlp.down_proj.weight.zero_point": torch.tensor([]),
        }

        result = loader._detect_trellis_format(tensor_dict)
        assert result == "exllamav3_exl2"

    def test_detect_exllamav3_trellis_format(self):
        """Should detect exllamav3_trellis format with 'trellis' in key."""
        from metal_marlin.trellis.loader import TrellisModelLoader

        loader = object.__new__(TrellisModelLoader)

        # ExllamaV3 trellis format keys
        tensor_dict = {
            "model.layers.0.mlp.down_proj.trellis_indices": torch.tensor([]),
            "model.layers.0.mlp.down_proj.scales": torch.tensor([]),
        }

        result = loader._detect_trellis_format(tensor_dict)
        assert result == "exllamav3_trellis"

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


class TestConvertExllamaV3Trellis:
    """Tests for _convert_exllamav3_trellis method."""

    def test_convert_simple_exllamav3(self):
        """Should convert simple exllamav3 format to TrellisWeight."""
        from metal_marlin.trellis.loader import TrellisModelLoader, TrellisWeight

        loader = object.__new__(TrellisModelLoader)

        # Simple exllamav3 format tensors
        tensor_dict = {
            "down_proj.indices": torch.randint(0, 16, (10, 10, 256), dtype=torch.int16),
            "down_proj.scales": torch.randn(10, 4096, dtype=torch.float32),
            "down_proj.su": torch.randn(1024, dtype=torch.float32),
            "down_proj.sv": torch.randn(4096, dtype=torch.float32),
        }

        weights = loader._convert_exllamav3_trellis(tensor_dict, 0)

        assert "down_proj" in weights
        weight = weights["down_proj"]
        assert isinstance(weight, TrellisWeight)
        assert weight.indices.shape == (10, 10, 256)
        assert weight.K == 1024
        assert weight.N == 4096

    def test_convert_alternative_naming(self):
        """Should handle alternative exllamav3 naming (scale, row_scale, etc)."""
        from metal_marlin.trellis.loader import TrellisModelLoader, TrellisWeight

        loader = object.__new__(TrellisModelLoader)

        # Alternative naming
        tensor_dict = {
            "up_proj.qweight": torch.randint(0, 16, (5, 8, 256), dtype=torch.int16),
            "up_proj.scale": torch.randn(5, 2048, dtype=torch.float32),
            "up_proj.row_scale": torch.randn(512, dtype=torch.float32),
            "up_proj.col_scale": torch.randn(2048, dtype=torch.float32),
        }

        weights = loader._convert_exllamav3_trellis(tensor_dict, 0)

        assert "up_proj" in weights
        weight = weights["up_proj"]
        assert isinstance(weight, TrellisWeight)
        assert weight.K == 512
        assert weight.N == 2048
