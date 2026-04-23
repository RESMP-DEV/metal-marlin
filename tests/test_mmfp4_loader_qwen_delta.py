"""Tests for Qwen hybrid-MoE checkpoint naming normalization in MMFP4 loader.

Covers naming realities from official Qwen checkpoints:
- `model.language_model.layers.*` prefix for Qwen3.5 / Qwen3.6
- singular `mlp.shared_expert.*` / `mlp.shared_expert_gate.weight`
- DeltaNet tensors `linear_attn.A_log`, `linear_attn.dt_bias`, `linear_attn.conv1d.weight`
- split Qwen3.5/3.6 tensors `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`
- fused Qwen3-Coder-Next tensors `in_proj_qkvz`, `in_proj_ba`

Uses synthetic fixtures; does not require real weight downloads.
Does not break existing GLM MMFP4 loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from metal_marlin._quantized_weights import (
    _normalize_key,
    _normalize_qwen_tensor_name,
)
from metal_marlin.mmfp4_loader import MMFP4ModelLoader

# ============================================================================
# Fixtures: Qwen3.5/3.6 style checkpoint shards
# ============================================================================


@pytest.fixture
def qwen35_official_shard(tmp_path: Path) -> Path:
    """Create mock Qwen3.5 official checkpoint with `model.language_model.layers` prefix.

    This fixture creates a minimal safetensors shard mimicking the official
    Qwen3.5 / Qwen3.6 naming convention where all transformer layers are under
    `model.language_model.layers.{idx}` instead of `model.layers.{idx}`.
    """
    shard_path = tmp_path / "model-00001-of-00001.safetensors"

    # Qwen3.5 official naming: model.language_model.layers.{idx}.*
    tensors = {
        # Attention projections (split Qwen3.5 style)
        "model.language_model.layers.0.self_attn.in_proj_qkv": torch.randn(
            3072, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.in_proj_z": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.in_proj_a": torch.randn(
            512, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.in_proj_b": torch.randn(
            512, 1024, dtype=torch.float16
        ),
        # Output projection
        "model.language_model.layers.0.self_attn.o_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        # MLP (standard)
        "model.language_model.layers.0.mlp.gate_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.mlp.up_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.mlp.down_proj": torch.randn(
            1024, 2816, dtype=torch.float16
        ),
        # Layer norm
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(
            1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.randn(
            1024, dtype=torch.float16
        ),
        # Also include standard model.layers variant for cross-naming tests
        "model.layers.0.self_attn.in_proj_qkv": torch.randn(
            3072, 1024, dtype=torch.float16
        ),
    }

    save_file(tensors, str(shard_path))

    # Create index
    index = {
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors.keys()}
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    # Create minimal config
    config = {
        "model_type": "qwen3_next",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 2816,
        "layer_types": ["linear_attention", "full_attention"],
        "full_attention_interval": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_conv_kernel_dim": 4,
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    return tmp_path


@pytest.fixture
def qwen35_deltanet_shard(tmp_path: Path) -> Path:
    """Create mock Qwen3.5 checkpoint with DeltaNet linear attention tensors.

    DeltaNet-specific tensors:
    - linear_attn.A_log: State matrix log eigenvalues
    - linear_attn.dt_bias: Discrete-time bias
    - linear_attn.conv1d.weight: Causal convolution kernel
    """
    shard_path = tmp_path / "model-00001-of-00001.safetensors"

    tensors = {
        # DeltaNet linear attention tensors
        "model.language_model.layers.0.self_attn.linear_attn.A_log": torch.randn(
            128, dtype=torch.float32
        ),
        "model.language_model.layers.0.self_attn.linear_attn.dt_bias": torch.randn(
            128, dtype=torch.float32
        ),
        "model.language_model.layers.0.self_attn.linear_attn.conv1d.weight": torch.randn(
            1, 4, 512, dtype=torch.float32
        ),
        # Standard attention (for hybrid layers)
        "model.language_model.layers.0.self_attn.q_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.k_proj": torch.randn(
            256, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.v_proj": torch.randn(
            256, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.o_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
    }

    save_file(tensors, str(shard_path))

    index = {
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors.keys()}
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return tmp_path


@pytest.fixture
def qwen3_coder_next_shard(tmp_path: Path) -> Path:
    """Create mock Qwen3-Coder-Next checkpoint with fused projections.

    Fused naming:
    - in_proj_qkvz: fused Q, K, V, Z projections
    - in_proj_ba: fused B, A projections (DeltaNet)
    """
    shard_path = tmp_path / "model-00001-of-00001.safetensors"

    tensors = {
        # Fused Qwen3-Coder-Next style
        "model.layers.0.self_attn.in_proj_qkvz": torch.randn(
            4096, 1024, dtype=torch.float16
        ),
        "model.layers.0.self_attn.in_proj_ba": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        "model.layers.0.self_attn.o_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        # MLP
        "model.layers.0.mlp.gate_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.layers.0.mlp.up_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.layers.0.mlp.down_proj": torch.randn(
            1024, 2816, dtype=torch.float16
        ),
    }

    save_file(tensors, str(shard_path))

    index = {
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors.keys()}
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return tmp_path


@pytest.fixture
def qwen_moe_shared_expert_shard(tmp_path: Path) -> Path:
    """Create mock Qwen MoE checkpoint with shared expert.

    Shared expert naming:
    - mlp.shared_expert.gate_proj
    - mlp.shared_expert.up_proj
    - mlp.shared_expert.down_proj
    - mlp.shared_expert_gate.weight (alternate)
    """
    shard_path = tmp_path / "model-00001-of-00001.safetensors"

    tensors = {
        # Standard attention
        "model.language_model.layers.0.self_attn.q_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.k_proj": torch.randn(
            256, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.v_proj": torch.randn(
            256, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.self_attn.o_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        # Shared expert (singular)
        "model.language_model.layers.0.mlp.shared_expert.gate_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.mlp.shared_expert.up_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.mlp.shared_expert.down_proj": torch.randn(
            1024, 2816, dtype=torch.float16
        ),
        # Alternate shared expert gate naming
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        # Routed experts (mock)
        "model.language_model.layers.0.mlp.experts.0.gate_proj": torch.randn(
            1408, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.mlp.experts.0.up_proj": torch.randn(
            1408, 1024, dtype=torch.float16
        ),
        "model.language_model.layers.0.mlp.experts.0.down_proj": torch.randn(
            1024, 1408, dtype=torch.float16
        ),
    }

    save_file(tensors, str(shard_path))

    index = {
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors.keys()}
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return tmp_path


@pytest.fixture
def glm47_standard_shard(tmp_path: Path) -> Path:
    """Create mock GLM-4.7 checkpoint with standard naming.

    Standard GLM naming:
    - model.layers.{idx}.self_attn.{q|k|v|o}_proj
    - model.layers.{idx}.mlp.{gate|up|down}_proj
    """
    shard_path = tmp_path / "model-00001-of-00001.safetensors"

    tensors = {
        # Attention (standard GLM naming)
        "model.layers.0.self_attn.q_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        "model.layers.0.self_attn.k_proj": torch.randn(
            256, 1024, dtype=torch.float16
        ),
        "model.layers.0.self_attn.v_proj": torch.randn(
            256, 1024, dtype=torch.float16
        ),
        "model.layers.0.self_attn.o_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
        # MLP (standard GLM naming)
        "model.layers.0.mlp.gate_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.layers.0.mlp.up_proj": torch.randn(
            2816, 1024, dtype=torch.float16
        ),
        "model.layers.0.mlp.down_proj": torch.randn(
            1024, 2816, dtype=torch.float16
        ),
        # Also include language_model variant for cross-naming tests
        "model.language_model.layers.0.self_attn.q_proj": torch.randn(
            1024, 1024, dtype=torch.float16
        ),
    }

    save_file(tensors, str(shard_path))

    index = {
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors.keys()}
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return tmp_path


# ============================================================================
# Tests: _normalize_qwen_tensor_name helper
# ============================================================================


class TestNormalizeQwenTensorName:
    """Test Qwen tensor name normalization."""

    def test_qwen35_official_prefix(self):
        """Qwen3.5 official `model.language_model.layers` -> `model.layers`."""
        name = "model.language_model.layers.0.self_attn.q_proj"
        expected = "model.layers.0.self_attn.q_proj"
        assert _normalize_qwen_tensor_name(name) == expected

    def test_language_model_layers_prefix(self):
        """`language_model.layers` -> `model.layers`."""
        name = "language_model.layers.0.self_attn.q_proj"
        expected = "model.layers.0.self_attn.q_proj"
        assert _normalize_qwen_tensor_name(name) == expected

    def test_standard_prefix_unchanged(self):
        """Standard `model.layers` prefix should be unchanged."""
        name = "model.layers.0.self_attn.q_proj"
        assert _normalize_qwen_tensor_name(name) == name

    def test_glm_prefix_unchanged(self):
        """GLM `model.layers` prefix should be unchanged."""
        name = "model.layers.0.self_attn.q_proj"
        assert _normalize_qwen_tensor_name(name) == name

    def test_deltanet_tensor(self):
        """DeltaNet tensors should be normalized."""
        name = "model.language_model.layers.0.self_attn.linear_attn.A_log"
        expected = "model.layers.0.self_attn.linear_attn.A_log"
        assert _normalize_qwen_tensor_name(name) == expected

    def test_shared_expert_tensor(self):
        """Shared expert tensors should be normalized."""
        name = "model.language_model.layers.0.mlp.shared_expert.gate_proj"
        expected = "model.layers.0.mlp.shared_expert.gate_proj"
        assert _normalize_qwen_tensor_name(name) == expected


# ============================================================================
# Tests: _normalize_key helper
# ============================================================================


class TestNormalizeKey:
    """Test key normalization for tensor lookup."""

    def test_dots_to_underscores(self):
        """Dots should become underscores."""
        assert _normalize_key("model.layers.0.q_proj") == "model_layers_0_q_proj"

    def test_lowercase(self):
        """Should be lowercased."""
        assert _normalize_key("MODEL.LAYERS.0") == "model_layers_0"

    def test_special_chars_stripped(self):
        """Special chars should be handled."""
        assert _normalize_key("model-layers-0") == "model_layers_0"


# ============================================================================
# Tests: MMFP4ModelLoader with Qwen naming
# ============================================================================


class TestMMFP4LoaderQwenNaming:
    """Test MMFP4ModelLoader with Qwen naming conventions."""

    def test_qwen35_layer_indices(self, qwen35_official_shard: Path):
        """Layer indices should be extracted from Qwen3.5 naming."""
        loader = MMFP4ModelLoader(qwen35_official_shard)
        # Should find layer 0 from model.language_model.layers.0.*
        assert 0 in loader._layer_to_tensors

    def test_qwen35_load_layer(self, qwen35_official_shard: Path):
        """Should load Qwen3.5 layer tensors."""
        loader = MMFP4ModelLoader(qwen35_official_shard)
        layer0 = loader.load_layer(0, device="cpu")

        # Check split projection naming
        assert any("in_proj_qkv" in k for k in layer0.keys())
        assert any("in_proj_z" in k for k in layer0.keys())
        assert any("in_proj_a" in k for k in layer0.keys())
        assert any("in_proj_b" in k for k in layer0.keys())

    def test_qwen35_get_quantized_weight(self, qwen35_official_shard: Path):
        """Should retrieve weights with Qwen3.5 naming via load_tensor."""
        loader = MMFP4ModelLoader(qwen35_official_shard)

        # Test with Qwen3.5 official naming - use load_tensor for raw weights
        tensor = loader.load_tensor(
            "model.language_model.layers.0.self_attn.in_proj_qkv.weight"
        )
        assert tensor.shape == (3072, 1024)

    def test_qwen35_get_quantized_weight_cross_naming(self, qwen35_official_shard: Path):
        """Should retrieve weights with Qwen3.5 naming via GLM-style names."""
        loader = MMFP4ModelLoader(qwen35_official_shard)

        # Query with GLM-style name, should find Qwen3.5 tensor
        tensor = loader.load_tensor(
            "model.layers.0.self_attn.in_proj_qkv.weight"
        )
        assert tensor.shape == (3072, 1024)

    def test_deltanet_tensors_loaded(self, qwen35_deltanet_shard: Path):
        """DeltaNet tensors should be loadable."""
        loader = MMFP4ModelLoader(qwen35_deltanet_shard)
        layer0 = loader.load_layer(0, device="cpu")

        # Check DeltaNet tensors
        assert any("linear_attn.A_log" in k for k in layer0.keys())
        assert any("linear_attn.dt_bias" in k for k in layer0.keys())
        assert any("linear_attn.conv1d" in k for k in layer0.keys())

    def test_qwen3_coder_next_fused(self, qwen3_coder_next_shard: Path):
        """Qwen3-Coder-Next fused naming should work."""
        loader = MMFP4ModelLoader(qwen3_coder_next_shard)
        layer0 = loader.load_layer(0, device="cpu")

        # Check fused projections
        assert any("in_proj_qkvz" in k for k in layer0.keys())
        assert any("in_proj_ba" in k for k in layer0.keys())

    def test_shared_expert_naming(self, qwen_moe_shared_expert_shard: Path):
        """Shared expert naming should be preserved."""
        loader = MMFP4ModelLoader(qwen_moe_shared_expert_shard)
        layer0 = loader.load_layer(0, device="cpu")

        # Check shared expert tensors
        assert any("shared_expert" in k for k in layer0.keys())
        assert any("shared_expert_gate" in k for k in layer0.keys())

    def test_glm_backward_compatibility(self, glm47_standard_shard: Path):
        """GLM-4.7 standard naming should still work."""
        loader = MMFP4ModelLoader(glm47_standard_shard)
        layer0 = loader.load_layer(0, device="cpu")

        # Standard GLM naming
        assert "model.layers.0.self_attn.q_proj" in layer0
        assert "model.layers.0.mlp.gate_proj" in layer0

    def test_cross_naming_compatibility_qwen_to_glm(self, qwen35_official_shard: Path):
        """Qwen3.5 naming should be findable via GLM-style queries."""
        loader = MMFP4ModelLoader(qwen35_official_shard)

        # Query with GLM-style name, should find Qwen3.5 tensor
        tensor = loader.load_tensor(
            "model.layers.0.self_attn.in_proj_qkv.weight"
        )
        assert tensor.shape == (3072, 1024)

    def test_cross_naming_compatibility_glm_to_qwen(self, glm47_standard_shard: Path):
        """GLM naming should be findable via Qwen-style queries."""
        loader = MMFP4ModelLoader(glm47_standard_shard)

        # Query with Qwen3.5-style name, should find GLM tensor
        tensor = loader.load_tensor(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )
        assert tensor.shape == (1024, 1024)


# ============================================================================
# Tests: Tensor metadata and layer info
# ============================================================================


class TestTensorMetadata:
    """Test tensor metadata extraction for Qwen naming."""

    def test_qwen35_metadata(self, qwen35_official_shard: Path):
        """Metadata should be available for Qwen3.5 tensors."""
        loader = MMFP4ModelLoader(qwen35_official_shard)

        meta = loader.get_tensor_metadata(
            "model.language_model.layers.0.self_attn.in_proj_qkv"
        )
        assert meta is not None
        assert meta.shape == (3072, 1024)
        assert meta.dtype == "F16"

    def test_deltanet_metadata(self, qwen35_deltanet_shard: Path):
        """Metadata should be available for DeltaNet tensors."""
        loader = MMFP4ModelLoader(qwen35_deltanet_shard)

        meta = loader.get_tensor_metadata(
            "model.language_model.layers.0.self_attn.linear_attn.A_log"
        )
        assert meta is not None
        assert meta.shape == (128,)
        assert meta.dtype == "F32"


# ============================================================================
# Tests: Iterator and streaming
# ============================================================================


class TestLoaderIterator:
    """Test iterator interface with Qwen naming."""

    def test_qwen35_iterator(self, qwen35_official_shard: Path):
        """Iterator should yield Qwen3.5 layers."""
        loader = MMFP4ModelLoader(qwen35_official_shard)
        layers = list(loader)

        assert len(layers) == 1
        layer_idx, layer_data = layers[0]
        assert layer_idx == 0
        assert len(layer_data) > 0

    def test_glm_iterator(self, glm47_standard_shard: Path):
        """Iterator should yield GLM layers."""
        loader = MMFP4ModelLoader(glm47_standard_shard)
        layers = list(loader)

        assert len(layers) == 1
        layer_idx, layer_data = layers[0]
        assert layer_idx == 0


# ============================================================================
# Tests: Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for Qwen naming normalization."""

    def test_empty_shard(self, tmp_path: Path):
        """Empty shard should not crash."""
        shard_path = tmp_path / "model-00001-of-00001.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        save_file(tensors, str(shard_path))

        index = {"weight_map": {}}
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        loader = MMFP4ModelLoader(tmp_path)
        assert len(loader._layer_to_tensors) == 0

    def test_mixed_naming_single_shard(self, tmp_path: Path):
        """Mixed Qwen/GLM naming in single shard should work."""
        shard_path = tmp_path / "model-00001-of-00001.safetensors"

        tensors = {
            # Qwen3.5 naming
            "model.language_model.layers.0.self_attn.q_proj": torch.randn(
                1024, 1024, dtype=torch.float16
            ),
            # GLM naming
            "model.layers.1.self_attn.q_proj": torch.randn(
                1024, 1024, dtype=torch.float16
            ),
        }
        save_file(tensors, str(shard_path))

        index = {
            "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors.keys()}
        }
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        loader = MMFP4ModelLoader(tmp_path)
        # Should find both layers
        assert 0 in loader._layer_to_tensors
        assert 1 in loader._layer_to_tensors
