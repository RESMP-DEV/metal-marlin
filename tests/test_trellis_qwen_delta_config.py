"""Tests for Qwen hybrid DeltaNet-family config in TrellisModelConfig.

Covers layer_types, full_attention_interval, DeltaNet linear_* dimensions,
shared expert size, and nested text_config handling for Qwen3.5/3.6 hybrids.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from metal_marlin.trellis.config import TrellisModelConfig

# ============================================================================
# Fixtures: Qwen3.5/3.6 DeltaNet-style configs
# ============================================================================


@pytest.fixture
def qwen35_deltanet_config_dict() -> dict:
    """Qwen3.5-30B-A3B style config with DeltaNet hybrid layers.
    
    This mirrors the structure of a real Qwen3.5 config.json with:
    - layer_types alternating linear/full attention
    - full_attention_interval
    - DeltaNet linear_* dimensions
    - Hybrid MoE with shared experts
    """
    return {
        "model_type": "qwen3_next",
        "vocab_size": 151936,
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        # MoE config
        "num_local_experts": 128,
        "num_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1408,
        "shared_expert_intermediate_size": 5632,
        # DeltaNet hybrid config
        "layer_types": ["linear_attention", "full_attention"] * 24,
        "full_attention_interval": 7,
        # DeltaNet linear attention dimensions
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
    }


@pytest.fixture
def qwen35_multimodal_config_dict() -> dict:
    """Qwen3.5-VL-MoE style multimodal config with nested text_config.
    
    The vision encoder wraps the text LLM in a text_config sub-dict.
    """
    return {
        "model_type": "qwen3_vl_moe",
        "vision_config": {
            "model_type": "qwen3_vl_vision",
            "hidden_size": 1152,
            "num_hidden_layers": 27,
        },
        "text_config": {
            "model_type": "qwen3_next",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            # MoE
            "num_local_experts": 128,
            "num_shared_experts": 1,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            # DeltaNet
            "layer_types": ["full_attention", "linear_attention"] * 24,
            "full_attention_interval": 8,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
        },
    }


@pytest.fixture
def qwen3_dense_config_dict() -> dict:
    """Standard Qwen3 dense model (non-DeltaNet, non-MoE).
    
    Should NOT be detected as hybrid DeltaNet.
    """
    return {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        # Dense model - no MoE, no DeltaNet
        "layer_types": ["full_attention"] * 24,
    }


@pytest.fixture
def glm47_config_dict() -> dict:
    """GLM-4.7-Flash config with MLA attention.
    
    Should work correctly without DeltaNet fields.
    """
    return {
        "model_type": "glm4_moe",
        "vocab_size": 154880,
        "hidden_size": 2048,
        "num_hidden_layers": 47,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "intermediate_size": 14336,
        "max_position_embeddings": 151552,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        # MLA attention
        "kv_lora_rank": 512,
        "q_lora_rank": 768,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        # MoE
        "num_local_experts": 64,
        "num_shared_experts": 1,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1536,
        "shared_expert_intermediate_size": 10240,
        "first_moe_layer": 1,
    }


# ============================================================================
# Tests: from_pretrained with config.json
# ============================================================================


class TestFromPretrainedConfigJson:
    """Test loading Qwen DeltaNet configs from config.json files."""

    def test_qwen35_deltanet_loads_all_fields(self, qwen35_deltanet_config_dict):
        """Qwen3.5 DeltaNet config should load all hybrid fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(qwen35_deltanet_config_dict, f)

            config = TrellisModelConfig.from_pretrained(tmpdir)

            # Core architecture
            assert config.hidden_size == 2048
            assert config.num_hidden_layers == 48
            assert config.vocab_size == 151936

            # MoE config
            assert config.num_experts == 128
            assert config.num_shared_experts == 1
            assert config.num_experts_per_tok == 8
            assert config.moe_intermediate_size == 1408
            assert config.shared_expert_intermediate_size == 5632

            # DeltaNet hybrid config
            assert config.layer_types is not None
            assert len(config.layer_types) == 48
            assert config.layer_types.count("linear_attention") == 24
            assert config.layer_types.count("full_attention") == 24
            assert config.full_attention_interval == 7

            # DeltaNet linear attention dimensions
            assert config.linear_key_head_dim == 128
            assert config.linear_value_head_dim == 128
            assert config.linear_num_key_heads == 4
            assert config.linear_num_value_heads == 4
            assert config.linear_conv_kernel_dim == 4

    def test_qwen35_multimodal_loads_text_config(self, qwen35_multimodal_config_dict):
        """Multimodal Qwen3.5-VL-MoE should extract fields from text_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(qwen35_multimodal_config_dict, f)

            config = TrellisModelConfig.from_pretrained(tmpdir)

            # Should get text_config fields
            assert config.hidden_size == 2048
            assert config.num_hidden_layers == 48
            assert config.vocab_size == 151936

            # MoE from text_config
            assert config.num_experts == 128
            assert config.num_shared_experts == 1
            assert config.shared_expert_intermediate_size == 5632

            # DeltaNet from text_config
            assert config.layer_types is not None
            assert len(config.layer_types) == 48
            assert config.full_attention_interval == 8

            # DeltaNet dimensions from text_config
            assert config.linear_key_head_dim == 128
            assert config.linear_num_key_heads == 4

    def test_qwen3_dense_no_deltanet_fields(self, qwen3_dense_config_dict):
        """Dense Qwen3 should not have DeltaNet fields populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(qwen3_dense_config_dict, f)

            config = TrellisModelConfig.from_pretrained(tmpdir)

            # layer_types present but no linear_attention
            assert config.layer_types is not None
            assert all(t == "full_attention" for t in config.layer_types)

            # No DeltaNet dimensions
            assert config.linear_key_head_dim is None
            assert config.linear_num_key_heads is None

            # Not MoE
            assert config.num_experts == 1

    def test_glm47_mla_no_deltanet(self, glm47_config_dict):
        """GLM-4.7 MLA should work without DeltaNet fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(glm47_config_dict, f)

            config = TrellisModelConfig.from_pretrained(tmpdir)

            # MLA fields
            assert config.kv_lora_rank == 512
            assert config.q_lora_rank == 768
            assert config.qk_nope_head_dim == 192

            # MoE
            assert config.num_experts == 64
            assert config.num_shared_experts == 1
            assert config.first_moe_layer == 1

            # No DeltaNet
            assert config.layer_types is None
            assert config.linear_key_head_dim is None


# ============================================================================
# Tests: _from_dict method
# ============================================================================


class TestFromDict:
    """Test _from_dict method with various config structures."""

    def test_deltanet_fields_direct_mapping(self, qwen35_deltanet_config_dict):
        """DeltaNet fields should map directly when present."""
        config = TrellisModelConfig._from_dict(qwen35_deltanet_config_dict)

        assert config.layer_types == qwen35_deltanet_config_dict["layer_types"]
        assert config.full_attention_interval == 7
        assert config.linear_key_head_dim == 128
        assert config.linear_num_key_heads == 4

    def test_nested_text_config_merging(self, qwen35_multimodal_config_dict):
        """Nested text_config should be merged into top-level."""
        config = TrellisModelConfig._from_dict(qwen35_multimodal_config_dict)

        # Should get text_config values
        assert config.hidden_size == 2048
        assert config.layer_types is not None
        assert config.full_attention_interval == 8

    def test_full_attention_interval_from_nested_dict(self):
        """full_attention_interval in nested text_config should be found."""
        data = {
            "model_type": "qwen3_vl_moe",
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 2048,
                "full_attention_interval": 6,
            },
        }
        config = TrellisModelConfig._from_dict(data)
        assert config.full_attention_interval == 6

    def test_layer_types_must_be_list(self):
        """Invalid layer_types (non-list) should be rejected."""
        data = {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "layer_types": "full_attention",  # Invalid: string instead of list
        }
        config = TrellisModelConfig._from_dict(data)
        assert config.layer_types is None

    def test_shared_expert_intermediate_size(self):
        """shared_expert_intermediate_size should be loaded."""
        data = {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_shared_experts": 1,
            "shared_expert_intermediate_size": 5632,
        }
        config = TrellisModelConfig._from_dict(data)
        assert config.shared_expert_intermediate_size == 5632
        assert config.num_shared_experts == 1


# ============================================================================
# Tests: _from_hf_config method (mock HF config objects)
# ============================================================================


class _MockHFConfig:
    """Mock HuggingFace PreTrainedConfig for testing."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return dict(self.__dict__)


class _MockMultimodalHFConfig:
    """Mock multimodal HF config wrapping text_config."""

    def __init__(self, text_config, **kwargs):
        self.text_config = text_config
        self.__dict__.update(kwargs)


class TestFromHFConfig:
    """Test _from_hf_config with mock HF config objects."""

    def test_qwen35_deltanet_attributes(self, qwen35_deltanet_config_dict):
        """HF config with DeltaNet attributes should be extracted."""
        hf_cfg = _MockHFConfig(**qwen35_deltanet_config_dict)
        config = TrellisModelConfig._from_hf_config(hf_cfg)

        assert config.layer_types is not None
        assert config.full_attention_interval == 7
        assert config.linear_key_head_dim == 128
        assert config.linear_num_key_heads == 4

    def test_full_attention_interval_via_to_dict(self):
        """full_attention_interval in to_dict() should be found."""
        hf_cfg = _MockHFConfig(
            vocab_size=151936,
            hidden_size=2048,
        )
        # Override to_dict to include full_attention_interval
        original_to_dict = hf_cfg.to_dict

        def custom_to_dict():
            d = original_to_dict()
            d["full_attention_interval"] = 5
            return d

        hf_cfg.to_dict = custom_to_dict  # type: ignore[method-assign]
        config = TrellisModelConfig._from_hf_config(hf_cfg)
        assert config.full_attention_interval == 5

    def test_multimodal_nested_text_config(self, qwen35_deltanet_config_dict):
        """Multimodal HF config should extract from text_config."""
        text_cfg = _MockHFConfig(**qwen35_deltanet_config_dict)
        mm_cfg = _MockMultimodalHFConfig(
            text_config=text_cfg,
            model_type="qwen3_vl_moe",
        )
        config = TrellisModelConfig._from_hf_config(mm_cfg)

        assert config.hidden_size == 2048
        assert config.layer_types is not None
        assert config.full_attention_interval == 7

    def test_multimodal_text_config_without_vocab_size_fallback(self):
        """If text_config lacks vocab_size, use parent config."""
        text_cfg = _MockHFConfig(
            hidden_size=2048,
            layer_types=["linear_attention"],
        )
        mm_cfg = _MockMultimodalHFConfig(
            text_config=text_cfg,
            vocab_size=151936,
            hidden_size=4096,  # Different from text_config
        )
        config = TrellisModelConfig._from_hf_config(mm_cfg)

        # Should fall back to parent since text_config lacks vocab_size
        assert config.vocab_size == 151936
        assert config.hidden_size == 4096


# ============================================================================
# Tests: Helper methods
# ============================================================================


class TestHelperMethods:
    """Test is_moe_layer, is_mla_model, and other helpers."""

    def test_is_moe_layer_deltanet_hybrid(self, qwen35_deltanet_config_dict):
        """is_moe_layer should work for DeltaNet hybrid MoE."""
        config = TrellisModelConfig._from_dict(qwen35_deltanet_config_dict)

        assert config.num_experts == 128
        assert config.first_moe_layer == 0  # All layers are MoE

        assert config.is_moe_layer(0) is True
        assert config.is_moe_layer(10) is True
        assert config.is_moe_layer(47) is True

    def test_is_moe_layer_first_moe_layer_offset(self):
        """is_moe_layer should respect first_moe_layer offset."""
        config = TrellisModelConfig(
            num_experts=64,
            first_moe_layer=1,
        )

        assert config.is_moe_layer(0) is False  # Dense layer
        assert config.is_moe_layer(1) is True   # First MoE layer
        assert config.is_moe_layer(10) is True

    def test_is_mla_model_glm(self, glm47_config_dict):
        """is_mla_model should return True for GLM MLA config."""
        config = TrellisModelConfig._from_dict(glm47_config_dict)
        assert config.is_mla_model() is True

    def test_is_mla_model_qwen_deltanet(self, qwen35_deltanet_config_dict):
        """is_mla_model should return False for Qwen DeltaNet (GQA)."""
        config = TrellisModelConfig._from_dict(qwen35_deltanet_config_dict)
        assert config.is_mla_model() is False

    def test_should_skip_layer(self):
        """should_skip_layer should check skip_layers list."""
        config = TrellisModelConfig(skip_layers=[2, 5, 8])

        assert config.should_skip_layer(2) is True
        assert config.should_skip_layer(5) is True
        assert config.should_skip_layer(3) is False
        assert config.should_skip_layer(10) is False

    def test_prune_layers(self):
        """prune_layers should create new config with skip_layers."""
        base = TrellisModelConfig(hidden_size=2048)
        pruned = base.prune_layers([1, 3, 5])

        assert pruned.skip_layers == [1, 3, 5]
        assert base.skip_layers is None  # Original unchanged


# ============================================================================
# Tests: Backward compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Ensure existing GLM/Qwen3/DeepSeek configs still work."""

    def test_glm_defaults_preserved(self):
        """GLM config should preserve MLA defaults."""
        config = TrellisModelConfig(
            kv_lora_rank=512,
            q_lora_rank=768,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
        )

        assert config.kv_lora_rank == 512
        assert config.q_lora_rank == 768
        assert config.qk_nope_head_dim == 192

    def test_qwen3_dense_defaults_preserved(self):
        """Dense Qwen3 config should work without DeltaNet fields."""
        config = TrellisModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_kv_heads=4,
        )

        # DeltaNet fields should be None by default
        assert config.layer_types is None
        assert config.full_attention_interval is None
        assert config.linear_key_head_dim is None

    def test_deepseek_mla_compatibility(self):
        """DeepSeek MLA config should work."""
        config = TrellisModelConfig(
            kv_lora_rank=512,
            q_lora_rank=768,
            num_experts=256,
            first_moe_layer=0,
        )

        assert config.is_mla_model() is True
        assert config.num_experts == 256


# ============================================================================
# Tests: Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_layer_types_list(self):
        """Empty layer_types list should be preserved."""
        config = TrellisModelConfig._from_dict({
            "vocab_size": 151936,
            "hidden_size": 2048,
            "layer_types": [],
        })
        assert config.layer_types == []

    def test_full_attention_interval_string_coerced(self):
        """String full_attention_interval should be coerced to int."""
        config = TrellisModelConfig._from_dict({
            "vocab_size": 151936,
            "hidden_size": 2048,
            "full_attention_interval": "7",
        })
        assert config.full_attention_interval == 7

    def test_full_attention_interval_invalid_string(self):
        """Invalid string full_attention_interval should be ignored."""
        config = TrellisModelConfig._from_dict({
            "vocab_size": 151936,
            "hidden_size": 2048,
            "full_attention_interval": "not_a_number",
        })
        assert config.full_attention_interval is None

    def test_partial_deltanet_fields(self):
        """Partial DeltaNet fields should be loaded (no validation error)."""
        config = TrellisModelConfig._from_dict({
            "vocab_size": 151936,
            "hidden_size": 2048,
            "linear_key_head_dim": 128,
            # Missing other linear_* fields
        })
        assert config.linear_key_head_dim == 128
        assert config.linear_num_key_heads is None

    def test_num_shared_experts_field_mapping(self):
        """num_shared_experts should map correctly."""
        config = TrellisModelConfig._from_dict({
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_shared_experts": 2,
        })
        assert config.num_shared_experts == 2
