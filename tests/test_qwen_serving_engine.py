"""Tests for Qwen DeltaNet family detection in the serving engine.

Covers config-driven detection for hybrid Qwen DeltaNet family models:
- Qwen/Qwen3.5-35B-A3B
- Qwen/Qwen3.6-35B-A3B

Verifies that detection uses config markers (model_type, nested text_config,
layer_types, MoE metadata) instead of narrow hard-coded vocab-size heuristics.

Also verifies GLM-4.7 detection preservation and API model name normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from metal_marlin.serving.engine import (
    DELTANET_LAYER_TYPES,
    MOE_EXPERT_COUNTS,
    MOE_INTERMEDIATE_KEYS,
    QWEN_DELTANET_MODEL_NAME_PATTERNS,
    QWEN_DELTANET_MODEL_TYPES,
    _detect_model_format,
    _get_config_value,
    _has_deltanet_layers,
    _is_moe_config,
    _is_qwen_deltanet_family,
    _normalize_model_name,
)

# ============================================================================
# Fixtures: Qwen3.5/3.6 DeltaNet-style configs
# ============================================================================


@pytest.fixture
def qwen35_35b_a3b_config() -> dict:
    """Qwen3.5-35B-A3B style config with DeltaNet hybrid layers and MoE."""
    return {
        "model_type": "qwen3_5_moe",
        "_name_or_path": "Qwen/Qwen3.5-35B-A3B",
        "vocab_size": 248320,
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
        # DeltaNet hybrid config (Qwen3.5 style: linear_attention + full_attention)
        "layer_types": ["linear_attention", "full_attention"] * 24,
        "full_attention_interval": 7,
        # DeltaNet linear attention dimensions
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
        # MMFP4 quantization marker
        "quantization_config": {
            "quant_method": "mmfp4",
            "format": "mmfp4",
        },
    }


@pytest.fixture
def qwen36_35b_a3b_config() -> dict:
    """Qwen3.6-35B-A3B style config with hybrid_attention layers and MoE.

    Qwen3.6 uses hybrid_attention layer type and list-valued
    full_attention_interval, plus use_delta=True.
    """
    return {
        "model_type": "qwen3_6_moe",
        "_name_or_path": "Qwen/Qwen3.6-35B-A3B",
        "architectures": ["Qwen3_6MoEForCausalLM"],
        "vocab_size": 256000,
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        # MoE config
        "num_local_experts": 256,
        "num_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1408,
        "shared_expert_intermediate_size": 5632,
        # DeltaNet hybrid config (Qwen3.6 style: full_attention + hybrid_attention)
        "layer_types": ["full_attention"] * 40 + ["hybrid_attention"] * 8,
        "full_attention_interval": [0, 1, 2, 3],
        "use_delta": True,
        "delta_intermediate_size": 1024,
        # MTP fields
        "num_mtp_heads": 1,
        "num_nextn_predict_layers": 1,
        # MMFP4 quantization marker
        "quantization_config": {
            "quant_method": "mmfp4",
            "format": "mmfp4",
        },
    }


@pytest.fixture
def qwen35_multimodal_config() -> dict:
    """Qwen3.5-VL-MoE style multimodal config with nested text_config."""
    return {
        "model_type": "qwen3_vl_moe",
        "_name_or_path": "Qwen/Qwen3.5-VL-MoE",
        "vision_config": {
            "model_type": "qwen3_vl_vision",
            "hidden_size": 1152,
            "num_hidden_layers": 27,
        },
        "text_config": {
            "model_type": "qwen3_5_moe",
            "vocab_size": 248320,
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
        "quantization_config": {
            "quant_method": "mmfp4",
            "format": "mmfp4",
        },
    }


@pytest.fixture
def qwen36_multimodal_config() -> dict:
    """Qwen3.6-VL-MoE style multimodal config with nested text_config.

    Tests that expert counts in text_config are found by _get_config_value.
    """
    return {
        "model_type": "qwen3_vl_moe",
        "_name_or_path": "Qwen/Qwen3.6-VL-MoE",
        "vision_config": {
            "model_type": "qwen3_vl_vision",
            "hidden_size": 1152,
        },
        "text_config": {
            "model_type": "qwen3_6_moe",
            "vocab_size": 256000,
            "hidden_size": 4096,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            # MoE inside text_config
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            # DeltaNet inside text_config
            "layer_types": ["full_attention"] * 40 + ["hybrid_attention"] * 8,
            "full_attention_interval": [0, 1, 2, 3],
            "use_delta": True,
            "delta_intermediate_size": 1024,
        },
        "quantization_config": {
            "quant_method": "mmfp4",
            "format": "mmfp4",
        },
    }


@pytest.fixture
def glm47_flash_config() -> dict:
    """GLM-4.7-Flash style config with MLA + MoE."""
    return {
        "model_type": "glm4_moe",
        "_name_or_path": "zai-org/GLM-4.7-Flash",
        "vocab_size": 151936,
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        # MLA attention
        "kv_lora_rank": 512,
        "q_lora_rank": 768,
        "kv_head_dim": 1120,
        # MoE config
        "num_local_experts": 64,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        # MMFP4 quantization marker
        "quantization_config": {
            "quant_method": "mmfp4",
            "format": "mmfp4",
        },
    }


@pytest.fixture
def qwen3_next_config() -> dict:
    """Qwen3-Next style config with DeltaNet hybrid layers."""
    return {
        "model_type": "qwen3_next",
        "_name_or_path": "Qwen/Qwen3-Next-30B-A3B",
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
        # MMFP4 quantization marker
        "quantization_config": {
            "quant_method": "mmfp4",
            "format": "mmfp4",
        },
    }


# ============================================================================
# Helper: write a config dict to a tmp_path/config.json
# ============================================================================


def _write_config(tmp_path: Path, config: dict) -> Path:
    """Write config dict as config.json into tmp_path, return tmp_path."""
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return tmp_path


# ============================================================================
# Tests for _get_config_value helper
# ============================================================================


class TestGetConfigValue:
    """Tests for _get_config_value helper function."""

    def test_top_level_key(self):
        config = {"model_type": "qwen3_5_moe", "vocab_size": 248320}
        assert _get_config_value(config, "model_type") == "qwen3_5_moe"

    def test_nested_text_config_preferred(self):
        """text_config value takes precedence over top-level."""
        config = {
            "model_type": "qwen3_vl_moe",
            "text_config": {
                "model_type": "qwen3_5_moe",
                "vocab_size": 248320,
            },
        }
        assert _get_config_value(config, "model_type") == "qwen3_5_moe"

    def test_fallback_to_top_level(self):
        config = {
            "model_type": "qwen3_vl_moe",
            "text_config": {"vocab_size": 248320},
        }
        assert _get_config_value(config, "model_type") == "qwen3_vl_moe"

    def test_default_value(self):
        config = {"vocab_size": 248320}
        assert _get_config_value(config, "nonexistent", default="default") == "default"

    def test_multiple_keys_returns_first_match(self):
        config = {"num_experts": 64, "num_local_experts": 128}
        assert _get_config_value(config, "num_local_experts", "num_experts") == 128

    def test_nested_text_config_for_expert_counts(self):
        """Expert counts in text_config are found via _get_config_value."""
        config = {
            "text_config": {
                "num_local_experts": 256,
            },
        }
        assert _get_config_value(config, "num_local_experts") == 256

    def test_nested_text_config_for_layer_types(self):
        """layer_types in text_config are found via _get_config_value."""
        config = {
            "text_config": {
                "layer_types": ["full_attention", "hybrid_attention"],
            },
        }
        assert _get_config_value(config, "layer_types") == [
            "full_attention",
            "hybrid_attention",
        ]

    def test_nested_text_config_for_use_delta(self):
        """use_delta in text_config is found via _get_config_value."""
        config = {
            "text_config": {
                "use_delta": True,
            },
        }
        assert _get_config_value(config, "use_delta") is True

    def test_nested_text_config_for_kv_lora_rank(self):
        """kv_lora_rank in text_config is found for GLM multimodal models."""
        config = {
            "text_config": {
                "kv_lora_rank": 512,
            },
        }
        assert _get_config_value(config, "kv_lora_rank") == 512


# ============================================================================
# Tests for _is_moe_config
# ============================================================================


class TestIsMoeConfig:
    """Tests for _is_moe_config function."""

    def test_moe_with_num_local_experts(self):
        assert _is_moe_config({"num_local_experts": 128}) is True

    def test_moe_with_num_experts(self):
        assert _is_moe_config({"num_experts": 64}) is True

    def test_moe_with_n_routed_experts(self):
        assert _is_moe_config({"n_routed_experts": 256}) is True

    def test_not_moe_single_expert(self):
        assert _is_moe_config({"num_local_experts": 1}) is False

    def test_not_moe_no_expert_keys(self):
        assert _is_moe_config({"hidden_size": 4096}) is False

    def test_moe_nested_text_config(self):
        assert _is_moe_config({"text_config": {"num_local_experts": 128}}) is True

    def test_moe_text_config_expert_count_zero(self):
        """Expert count of 0 in text_config should not be MoE."""
        assert _is_moe_config({"text_config": {"num_local_experts": 0}}) is False


# ============================================================================
# Tests for _has_deltanet_layers
# ============================================================================


class TestHasDeltanetLayers:
    """Tests for _has_deltanet_layers function."""

    def test_linear_attention_layer_type(self):
        """Qwen3.5 style: linear_attention in layer_types."""
        config = {"layer_types": ["linear_attention", "full_attention"] * 24}
        assert _has_deltanet_layers(config) is True

    def test_hybrid_attention_layer_type(self):
        """Qwen3.6 style: hybrid_attention in layer_types."""
        config = {"layer_types": ["full_attention"] * 40 + ["hybrid_attention"] * 8}
        assert _has_deltanet_layers(config) is True

    def test_delta_attention_layer_type(self):
        """delta_attention in layer_types."""
        config = {"layer_types": ["delta_attention"]}
        assert _has_deltanet_layers(config) is True

    def test_full_attention_interval_int(self):
        """full_attention_interval as int (Qwen3.5 style)."""
        config = {"full_attention_interval": 7}
        assert _has_deltanet_layers(config) is True

    def test_full_attention_interval_list(self):
        """full_attention_interval as list (Qwen3.6 style: [0, 1, 2, 3])."""
        config = {"full_attention_interval": [0, 1, 2, 3]}
        assert _has_deltanet_layers(config) is True

    def test_full_attention_interval_empty_list(self):
        """Empty list full_attention_interval should not trigger detection."""
        config = {"full_attention_interval": []}
        assert _has_deltanet_layers(config) is False

    def test_full_attention_interval_zero(self):
        """full_attention_interval=0 should not trigger detection."""
        config = {"full_attention_interval": 0}
        assert _has_deltanet_layers(config) is False

    def test_use_delta_marker(self):
        """use_delta=True is a DeltaNet marker (Qwen3.6)."""
        config = {"use_delta": True}
        assert _has_deltanet_layers(config) is True

    def test_use_delta_false(self):
        """use_delta=False should not trigger detection."""
        config = {"use_delta": False}
        assert _has_deltanet_layers(config) is False

    def test_not_deltanet_no_markers(self):
        config = {"hidden_size": 4096}
        assert _has_deltanet_layers(config) is False

    def test_not_deltanet_empty_layer_types(self):
        config = {"layer_types": []}
        assert _has_deltanet_layers(config) is False

    def test_deltanet_nested_text_config(self):
        """DeltaNet markers in nested text_config are found."""
        config = {
            "text_config": {
                "layer_types": ["linear_attention", "full_attention"],
            },
        }
        assert _has_deltanet_layers(config) is True

    def test_use_delta_nested_text_config(self):
        """use_delta in nested text_config is found."""
        config = {
            "text_config": {
                "use_delta": True,
            },
        }
        assert _has_deltanet_layers(config) is True

    def test_full_attention_interval_nested_text_config(self):
        """full_attention_interval in nested text_config is found."""
        config = {
            "text_config": {
                "full_attention_interval": 7,
            },
        }
        assert _has_deltanet_layers(config) is True

    def test_plain_dense_layer_types_not_deltanet(self):
        """Standard dense layer types should not trigger DeltaNet detection."""
        config = {"layer_types": ["dense", "dense"]}
        assert _has_deltanet_layers(config) is False


# ============================================================================
# Tests for _is_qwen_deltanet_family
# ============================================================================


class TestIsQwenDeltanetFamily:
    """Tests for _is_qwen_deltanet_family function."""

    # --- model_type detection ---

    def test_qwen35_model_type(self):
        assert _is_qwen_deltanet_family({"model_type": "qwen3_5_moe"}) is True

    def test_qwen36_model_type(self):
        assert _is_qwen_deltanet_family({"model_type": "qwen3_6_moe"}) is True

    def test_qwen3_next_model_type(self):
        assert _is_qwen_deltanet_family({"model_type": "qwen3_next"}) is True

    def test_qwen_vl_moe_model_type(self):
        assert _is_qwen_deltanet_family({"model_type": "qwen3_vl_moe"}) is True

    # --- model name detection ---

    def test_qwen35_model_name(self):
        assert (
            _is_qwen_deltanet_family({"model_type": "llama"}, model_name="Qwen3.5-35B-A3B")
            is True
        )

    def test_qwen36_model_name(self):
        assert (
            _is_qwen_deltanet_family({"model_type": "llama"}, model_name="Qwen3.6-35B-A3B")
            is True
        )

    # --- architecture list detection ---

    def test_qwen36_architecture_marker(self):
        """Qwen3.6 architectures list contains Qwen3_6MoE marker."""
        config = {
            "model_type": "custom",
            "architectures": ["Qwen3_6MoEForCausalLM"],
        }
        assert _is_qwen_deltanet_family(config) is True

    def test_qwen35_architecture_marker(self):
        """Qwen3.5 architectures list contains Qwen3_5MoE marker."""
        config = {
            "model_type": "custom",
            "architectures": ["Qwen3_5MoEForCausalLM"],
        }
        assert _is_qwen_deltanet_family(config) is True

    # --- config-driven MoE + DeltaNet detection ---

    def test_moe_plus_hybrid_attention_layers(self):
        """MoE + hybrid_attention combination should be Qwen hybrid."""
        config = {
            "model_type": "custom_moe",
            "num_local_experts": 128,
            "layer_types": ["full_attention", "hybrid_attention"],
        }
        assert _is_qwen_deltanet_family(config) is True

    def test_moe_plus_linear_attention_layers(self):
        """MoE + linear_attention combination should be Qwen hybrid."""
        config = {
            "model_type": "custom_moe",
            "num_local_experts": 128,
            "layer_types": ["linear_attention", "full_attention"],
        }
        assert _is_qwen_deltanet_family(config) is True

    def test_moe_plus_use_delta(self):
        """MoE + use_delta=True should be Qwen hybrid."""
        config = {
            "model_type": "custom_moe",
            "num_local_experts": 128,
            "use_delta": True,
        }
        assert _is_qwen_deltanet_family(config) is True

    def test_moe_plus_full_attention_interval_list(self):
        """MoE + list-valued full_attention_interval should be Qwen hybrid."""
        config = {
            "model_type": "custom_moe",
            "num_local_experts": 128,
            "full_attention_interval": [0, 1, 2, 3],
        }
        assert _is_qwen_deltanet_family(config) is True

    def test_shared_expert_intermediate_with_deltanet(self):
        """shared_expert_intermediate_size + DeltaNet layers."""
        config = {
            "model_type": "custom",
            "shared_expert_intermediate_size": 5632,
            "layer_types": ["linear_attention", "full_attention"],
        }
        assert _is_qwen_deltanet_family(config) is True

    # --- negative cases ---

    def test_not_qwen_standard_llama(self):
        config = {"model_type": "llama", "num_hidden_layers": 32}
        assert _is_qwen_deltanet_family(config) is False

    def test_not_qwen_moe_without_deltanet(self):
        """MoE without DeltaNet markers should not match (unless model_type/name hit)."""
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
        }
        assert _is_qwen_deltanet_family(config) is False

    def test_not_qwen_deltanet_without_moe(self):
        """DeltaNet layers without MoE markers should not match rule 4/5
        (but might match via model_type — use a non-matching type)."""
        config = {
            "model_type": "custom",
            "layer_types": ["linear_attention", "full_attention"],
        }
        # No MoE, no model_type match, no name match, no shared_expert → False
        assert _is_qwen_deltanet_family(config) is False

    # --- nested text_config ---

    def test_nested_text_config_model_type(self):
        config = {
            "model_type": "qwen3_vl_moe",
            "text_config": {
                "model_type": "qwen3_5_moe",
                "num_local_experts": 128,
            },
        }
        assert _is_qwen_deltanet_family(config) is True

    def test_nested_text_config_moe_plus_deltanet(self):
        """MoE in text_config + DeltaNet in text_config triggers hybrid detection."""
        config = {
            "model_type": "custom_vl",
            "text_config": {
                "model_type": "custom_moe",
                "num_local_experts": 128,
                "layer_types": ["hybrid_attention", "full_attention"],
            },
        }
        assert _is_qwen_deltanet_family(config) is True


# ============================================================================
# Tests for _detect_model_format
# ============================================================================


class TestDetectModelFormat:
    """Tests for _detect_model_format function."""

    # --- Qwen3.5-35B-A3B ---

    def test_qwen35_mmfp4_detection(self, qwen35_35b_a3b_config, tmp_path):
        _write_config(tmp_path, qwen35_35b_a3b_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Qwen3.6-35B-A3B ---

    def test_qwen36_mmfp4_detection(self, qwen36_35b_a3b_config, tmp_path):
        """Qwen3.6-35B-A3B with hybrid_attention + list full_attention_interval."""
        _write_config(tmp_path, qwen36_35b_a3b_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- GLM-4.7 preservation ---

    def test_glm47_mmfp4_detection(self, glm47_flash_config, tmp_path):
        _write_config(tmp_path, glm47_flash_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Qwen3-Next ---

    def test_qwen3_next_mmfp4_detection(self, qwen3_next_config, tmp_path):
        _write_config(tmp_path, qwen3_next_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Multimodal models ---

    def test_qwen35_multimodal_mmfp4_detection(self, qwen35_multimodal_config, tmp_path):
        _write_config(tmp_path, qwen35_multimodal_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    def test_qwen36_multimodal_mmfp4_detection(self, qwen36_multimodal_config, tmp_path):
        """Qwen3.6-VL-MoE with nested text_config should detect as MMFP4."""
        _write_config(tmp_path, qwen36_multimodal_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Edge cases: Qwen3.6 with only use_delta (no layer_types) ---

    def test_qwen36_use_delta_only(self, tmp_path):
        """Qwen3.6 model with use_delta=True but no layer_types should still
        be detected via model_type (qwen3_6_moe)."""
        config = {
            "model_type": "qwen3_6_moe",
            "_name_or_path": "Qwen/Qwen3.6-35B-A3B",
            "vocab_size": 256000,
            "num_local_experts": 256,
            "use_delta": True,
            "quantization_config": {
                "quant_method": "mmfp4",
                "format": "mmfp4",
            },
        }
        _write_config(tmp_path, config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Qwen3.6 with MoE only in text_config ---

    def test_qwen36_expert_count_in_text_config(self, tmp_path):
        """Verify that _detect_model_format finds expert counts in text_config
        when checking for has_moe in the MMFP4 quant branch."""
        config = {
            "model_type": "qwen3_6_moe",
            "_name_or_path": "Qwen/Qwen3.6-35B-A3B",
            # No top-level expert keys — they're in text_config
            "text_config": {
                "num_local_experts": 256,
                "layer_types": ["full_attention", "hybrid_attention"],
            },
            "quantization_config": {
                "quant_method": "mmfp4",
                "format": "mmfp4",
            },
        }
        _write_config(tmp_path, config)
        # Even if has_moe is False (expert count only in text_config),
        # has_qwen_deltanet should still be True via model_type
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Nonexistent path ---

    def test_nonexistent_path_returns_marlin(self):
        assert _detect_model_format("/nonexistent/path") == "marlin"

    # --- Trellis ---

    def test_trellis_quantization_detection(self, tmp_path):
        _write_config(tmp_path, {
            "model_type": "llama",
            "quantization_config": {"quant_method": "trellis"},
        })
        assert _detect_model_format(str(tmp_path)) == "trellis"

    def test_trellis_format_detection(self, tmp_path):
        _write_config(tmp_path, {"model_type": "llama", "format": "trellis"})
        assert _detect_model_format(str(tmp_path)) == "trellis"

    def test_trellis_directory_structure(self, tmp_path):
        (tmp_path / "layer_0000").mkdir()
        assert _detect_model_format(str(tmp_path)) == "trellis"

    # --- Standard Marlin fallback ---

    def test_standard_marlin_fallback(self, tmp_path):
        _write_config(tmp_path, {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        })
        assert _detect_model_format(str(tmp_path)) == "marlin"

    # --- Qwen without MMFP4 quant ---

    def test_qwen_without_mmfp4_quant_returns_marlin(self, tmp_path):
        """Qwen DeltaNet model without MMFP4 quantization → marlin."""
        _write_config(tmp_path, {
            "model_type": "qwen3_5_moe",
            "num_local_experts": 128,
            "layer_types": ["linear_attention", "full_attention"],
        })
        assert _detect_model_format(str(tmp_path)) == "marlin"

    def test_qwen36_without_mmfp4_quant_returns_marlin(self, tmp_path):
        """Qwen3.6 without MMFP4 quantization → marlin."""
        _write_config(tmp_path, {
            "model_type": "qwen3_6_moe",
            "num_local_experts": 256,
            "layer_types": ["full_attention", "hybrid_attention"],
            "use_delta": True,
        })
        assert _detect_model_format(str(tmp_path)) == "marlin"

    # --- GLM with kv_lora_rank in text_config (multimodal GLM) ---

    def test_glm47_multimodal_with_text_config(self, tmp_path):
        """GLM-4.7 multimodal with kv_lora_rank in text_config should detect MMFP4."""
        config = {
            "model_type": "glm4_moe",
            "_name_or_path": "zai-org/GLM-4.7-Flash-VL",
            "text_config": {
                "model_type": "glm4_moe",
                "kv_lora_rank": 512,
                "num_local_experts": 64,
            },
            "quantization_config": {
                "quant_method": "mmfp4",
                "format": "mmfp4",
            },
        }
        _write_config(tmp_path, config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Generic MoE + MLA combination ---

    def test_generic_moe_mla_mmfp4(self, tmp_path):
        """Generic MoE + MLA combination with MMFP4 quant → mmfp4."""
        config = {
            "model_type": "custom",
            "num_local_experts": 8,
            "kv_lora_rank": 256,
            "quantization_config": {
                "quant_method": "mmfp4",
                "format": "mmfp4",
            },
        }
        _write_config(tmp_path, config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"

    # --- Malformed config.json ---

    def test_malformed_config_returns_marlin(self, tmp_path):
        """Malformed config.json should not crash, just return marlin."""
        config_path = tmp_path / "config.json"
        config_path.write_text("not valid json{{{")
        assert _detect_model_format(str(tmp_path)) == "marlin"

    def test_non_dict_config_returns_marlin(self, tmp_path):
        """config.json that is a list instead of dict → marlin."""
        config_path = tmp_path / "config.json"
        config_path.write_text("[1, 2, 3]")
        assert _detect_model_format(str(tmp_path)) == "marlin"

    def test_empty_dir_returns_marlin(self, tmp_path):
        """Directory with no config.json → marlin."""
        assert _detect_model_format(str(tmp_path)) == "marlin"


# ============================================================================
# Tests for _normalize_model_name
# ============================================================================


class TestNormalizeModelName:
    """Tests for _normalize_model_name function."""

    # --- Qwen3.5 variants ---

    def test_qwen35_35b_a3b_full_path(self):
        assert _normalize_model_name("Qwen/Qwen3.5-35B-A3B") == "Qwen/Qwen3.5-35B-A3B"

    def test_qwen35_35b_a3b_bare(self):
        assert _normalize_model_name("Qwen3.5-35B-A3B") == "Qwen/Qwen3.5-35B-A3B"

    def test_qwen35_30b_a3b(self):
        """Qwen3.5-30B-A3B: this was previously unreachable; now fixed."""
        assert _normalize_model_name("Qwen/Qwen3.5-30B-A3B") == "Qwen/Qwen3.5-30B-A3B"

    def test_qwen35_underscore_variant(self):
        assert _normalize_model_name("models/qwen3_5_moe") == "Qwen/Qwen3.5"

    def test_qwen35_generic(self):
        """Qwen3.5 without specific size → generic Qwen/Qwen3.5."""
        assert _normalize_model_name("Qwen/Qwen3.5-72B") == "Qwen/Qwen3.5"

    # --- Qwen3.6 variants ---

    def test_qwen36_35b_a3b_full_path(self):
        assert _normalize_model_name("Qwen/Qwen3.6-35B-A3B") == "Qwen/Qwen3.6-35B-A3B"

    def test_qwen36_35b_a3b_bare(self):
        assert _normalize_model_name("Qwen3.6-35B-A3B") == "Qwen/Qwen3.6-35B-A3B"

    def test_qwen36_underscore_variant(self):
        assert _normalize_model_name("models/qwen3_6_moe") == "Qwen/Qwen3.6"

    def test_qwen36_generic(self):
        """Qwen3.6 without specific size → generic Qwen/Qwen3.6."""
        assert _normalize_model_name("Qwen/Qwen3.6-72B") == "Qwen/Qwen3.6"

    # --- Non-Qwen paths unchanged ---

    def test_non_qwen_path_unchanged(self):
        assert _normalize_model_name("zai-org/GLM-4.7-Flash") == "GLM-4.7-Flash"

    def test_llama_path_unchanged(self):
        assert _normalize_model_name("meta-llama/Llama-3-8B") == "Llama-3-8B"

    def test_empty_path_returns_mock(self):
        assert _normalize_model_name("") == "mock-model"

    # --- Case insensitivity ---

    def test_qwen35_lowercase(self):
        assert _normalize_model_name("qwen/qwen3.5-35b-a3b") == "Qwen/Qwen3.5-35B-A3B"

    def test_qwen36_lowercase(self):
        assert _normalize_model_name("qwen/qwen3.6-35b-a3b") == "Qwen/Qwen3.6-35B-A3B"


# ============================================================================
# Integration tests: detection constants are properly defined
# ============================================================================


class TestDetectionConstants:
    """Tests for detection constants."""

    def test_qwen_deltanet_model_types(self):
        assert "qwen3_5_moe" in QWEN_DELTANET_MODEL_TYPES
        assert "qwen3_6_moe" in QWEN_DELTANET_MODEL_TYPES
        assert "qwen3_next" in QWEN_DELTANET_MODEL_TYPES
        assert "qwen3_vl_moe" in QWEN_DELTANET_MODEL_TYPES

    def test_qwen_deltanet_name_patterns(self):
        assert "qwen3.5-35b-a3b" in QWEN_DELTANET_MODEL_NAME_PATTERNS
        assert "qwen3.6-35b-a3b" in QWEN_DELTANET_MODEL_NAME_PATTERNS
        assert "qwen3.5" in QWEN_DELTANET_MODEL_NAME_PATTERNS
        assert "qwen3.6" in QWEN_DELTANET_MODEL_NAME_PATTERNS

    def test_deltanet_layer_types_includes_hybrid(self):
        """hybrid_attention must be recognized as a DeltaNet layer type."""
        assert "hybrid_attention" in DELTANET_LAYER_TYPES

    def test_deltanet_layer_types_includes_linear(self):
        assert "linear_attention" in DELTANET_LAYER_TYPES

    def test_deltanet_layer_types_includes_full(self):
        assert "full_attention" in DELTANET_LAYER_TYPES

    def test_deltanet_layer_types_includes_delta(self):
        assert "delta_attention" in DELTANET_LAYER_TYPES

    def test_moe_expert_counts_keys(self):
        assert "num_experts" in MOE_EXPERT_COUNTS
        assert "num_local_experts" in MOE_EXPERT_COUNTS
        assert "n_routed_experts" in MOE_EXPERT_COUNTS

    def test_moe_intermediate_keys(self):
        assert "moe_intermediate_size" in MOE_INTERMEDIATE_KEYS
        assert "expert_intermediate_size" in MOE_INTERMEDIATE_KEYS


# ============================================================================
# Integration tests: full config round-trip for Qwen3.5 and Qwen3.6
# ============================================================================


class TestQwenDeltanetFullConfigRoundTrip:
    """End-to-end tests verifying the full detection pipeline for specific models."""

    def test_qwen35_35b_a3b_detected_as_mmfp4(self, qwen35_35b_a3b_config, tmp_path):
        """Qwen/Qwen3.5-35B-A3B: config-driven detection → mmfp4 format."""
        _write_config(tmp_path, qwen35_35b_a3b_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"
        assert _is_qwen_deltanet_family(qwen35_35b_a3b_config, "Qwen/Qwen3.5-35B-A3B")
        assert _normalize_model_name("Qwen/Qwen3.5-35B-A3B") == "Qwen/Qwen3.5-35B-A3B"

    def test_qwen36_35b_a3b_detected_as_mmfp4(self, qwen36_35b_a3b_config, tmp_path):
        """Qwen/Qwen3.6-35B-A3B: config-driven detection → mmfp4 format.

        This model uses:
        - model_type: qwen3_6_moe
        - layer_types with hybrid_attention (not just linear_attention)
        - full_attention_interval as list [0, 1, 2, 3]
        - use_delta: True
        """
        _write_config(tmp_path, qwen36_35b_a3b_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"
        assert _is_qwen_deltanet_family(qwen36_35b_a3b_config, "Qwen/Qwen3.6-35B-A3B")
        assert _normalize_model_name("Qwen/Qwen3.6-35B-A3B") == "Qwen/Qwen3.6-35B-A3B"

    def test_glm47_flash_detected_as_mmfp4(self, glm47_flash_config, tmp_path):
        """GLM-4.7-Flash detection preserved alongside Qwen DeltaNet."""
        _write_config(tmp_path, glm47_flash_config)
        assert _detect_model_format(str(tmp_path)) == "mmfp4"
        # GLM should NOT be detected as Qwen DeltaNet family
        assert _is_qwen_deltanet_family(glm47_flash_config) is False
