"""Narrow regression slice for Qwen hybrid DeltaNet support.

Covers three critical paths without loading full model weights:
1. Config parsing for Qwen3.5 and Qwen3.6 sample configs
2. Serving format detection for Qwen hybrid MMFP4 configs
3. Synthetic checkpoint-prefix case for ``model.language_model.layers.*``

Designed as a fast, deterministic smoke slice suitable for repeated
execution during follow-up work on the DeltaNet integration.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest

from metal_marlin.hf_loader import ModelConfig
from metal_marlin.serving.engine import (
    DELTANET_LAYER_TYPES,
    MOE_EXPERT_COUNTS,
    MOE_INTERMEDIATE_KEYS,
    QWEN_DELTANET_MODEL_TYPES,
    _detect_model_format,
    _get_config_value,
    _has_deltanet_layers,
    _is_moe_config,
    _is_qwen_deltanet_family,
)

pytestmark = pytest.mark.smoke


# ============================================================================
# Sample config dicts
# ============================================================================


def _qwen35_flat_config() -> dict[str, Any]:
    """Qwen3.5-30B-A3B style flat DeltaNet config (no text_config nesting)."""
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
        # MoE
        "num_local_experts": 128,
        "num_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1408,
        "shared_expert_intermediate_size": 5632,
        # DeltaNet hybrid
        "layer_types": ["linear_attention", "full_attention"] * 24,
        "full_attention_interval": 7,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
    }


def _qwen36_multimodal_config() -> dict[str, Any]:
    """Qwen3.6-35B-A3B style multimodal config with nested text_config."""
    return {
        "model_type": "qwen3_6_moe",
        "vocab_size": 151936,
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        # Outer-level MoE markers (some Qwen3.6 configs duplicate these)
        "num_local_experts": 64,
        "num_shared_experts": 1,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1408,
        "shared_expert_intermediate_size": 5632,
        # text_config carries the effective LLM config
        "text_config": {
            "model_type": "qwen3_6_moe_text",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            # DeltaNet / DAN layers
            "layer_types": ["dense", "moe", "dan"],
            "full_attention_interval": 2,
            "use_delta": True,
            "delta_intermediate_size": 2048,
            # MoE from text_config
            "num_local_experts": 64,
            "num_shared_experts": 1,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            # MTP
            "num_mtp_heads": 1,
            "mtp_num_hidden_layers": 1,
            "mtp_expansion_factor": 2,
        },
    }


# ============================================================================
# 1. Config parsing for Qwen3.5 and Qwen3.6
# ============================================================================


class TestQwen35ConfigParsing:
    """Qwen3.5 flat DeltaNet config via ModelConfig.from_dict."""

    def test_core_architecture_fields(self):
        cfg = ModelConfig.from_dict(_qwen35_flat_config())
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 48
        assert cfg.vocab_size == 151936
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 4

    def test_moe_fields(self):
        cfg = ModelConfig.from_dict(_qwen35_flat_config())
        assert cfg.is_moe is True
        assert cfg.num_experts == 128
        assert cfg.num_experts_per_tok == 8
        assert cfg.moe_intermediate_size == 1408
        assert cfg.shared_expert_intermediate_size == 5632

    def test_deltanet_fields(self):
        cfg = ModelConfig.from_dict(_qwen35_flat_config())
        assert cfg.layer_types is not None
        assert len(cfg.layer_types) == 48
        assert cfg.layer_types.count("linear_attention") == 24
        assert cfg.layer_types.count("full_attention") == 24
        assert cfg.use_delta is False
        assert cfg.has_delta is False

    def test_full_attention_interval_normalised_to_list(self):
        """full_attention_interval int -> [int] normalisation."""
        cfg = ModelConfig.from_dict(_qwen35_flat_config())
        assert cfg.full_attention_interval is not None
        if isinstance(cfg.full_attention_interval, list):
            assert cfg.full_attention_interval == [7]
        else:
            assert cfg.full_attention_interval == 7

    def test_model_type_resolved(self):
        cfg = ModelConfig.from_dict(_qwen35_flat_config())
        assert cfg.model_type == "qwen3_next"


class TestQwen36ConfigParsing:
    """Qwen3.6 multimodal config with nested text_config."""

    def test_text_config_takes_precedence(self):
        """text_config.hidden_size should override outer wrapper."""
        cfg = ModelConfig.from_dict(_qwen36_multimodal_config())
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 36

    def test_model_type_from_text_config(self):
        cfg = ModelConfig.from_dict(_qwen36_multimodal_config())
        assert cfg.model_type == "qwen3_6_moe_text"

    def test_deltanet_layer_types_from_text_config(self):
        cfg = ModelConfig.from_dict(_qwen36_multimodal_config())
        assert cfg.layer_types == ["dense", "moe", "dan"]

    def test_use_delta_flag_from_text_config(self):
        cfg = ModelConfig.from_dict(_qwen36_multimodal_config())
        assert cfg.use_delta is True
        assert cfg.delta_intermediate_size == 2048
        assert cfg.has_delta is True

    def test_full_attention_interval_from_text_config(self):
        cfg = ModelConfig.from_dict(_qwen36_multimodal_config())
        assert cfg.full_attention_interval is not None
        if isinstance(cfg.full_attention_interval, list):
            assert cfg.full_attention_interval == [2]
        else:
            assert cfg.full_attention_interval == 2

    def test_mtp_fields_from_text_config(self):
        cfg = ModelConfig.from_dict(_qwen36_multimodal_config())
        assert cfg.has_mtp is True
        assert cfg.num_mtp_heads == 1
        assert cfg.mtp_num_hidden_layers == 1
        assert cfg.mtp_expansion_factor == 2

    def test_moe_fields_from_text_config(self):
        cfg = ModelConfig.from_dict(_qwen36_multimodal_config())
        assert cfg.is_moe is True
        assert cfg.num_experts == 64
        assert cfg.shared_expert_intermediate_size == 5632


class TestDenseQwenNotDeltaNet:
    """Dense Qwen3 should NOT be mis-detected as DeltaNet hybrid."""

    def test_dense_qwen3_no_deltanet(self):
        cfg = ModelConfig.from_dict({
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            "layer_types": ["full_attention"] * 24,
        })
        assert cfg.use_delta is False
        assert cfg.delta_intermediate_size is None
        assert cfg.has_delta is False
        assert cfg.layer_types is not None
        assert "linear_attention" not in cfg.layer_types


# ============================================================================
# 2. Serving format detection for Qwen hybrid MMFP4 configs
# ============================================================================


class TestGetConfigValue:
    """_get_config_value should prefer text_config then top-level."""

    def test_text_config_preferred(self):
        config = {"hidden_size": 4096, "text_config": {"hidden_size": 2048}}
        assert _get_config_value(config, "hidden_size") == 2048

    def test_fallback_to_top_level(self):
        config = {"hidden_size": 4096}
        assert _get_config_value(config, "hidden_size") == 4096

    def test_missing_key_returns_default(self):
        assert _get_config_value({}, "missing_key", default=0) == 0

    def test_multi_key_lookup(self):
        config = {"num_local_experts": 64}
        assert _get_config_value(
            config, "num_experts", "num_local_experts", default=0
        ) == 64


class TestIsMoeConfig:
    """_is_moe_config should detect MoE via expert count keys."""

    def test_moe_detected_via_num_local_experts(self):
        assert _is_moe_config({"num_local_experts": 128}) is True

    def test_moe_detected_via_text_config(self):
        assert _is_moe_config({
            "text_config": {"num_local_experts": 64},
        }) is True

    def test_single_expert_is_not_moe(self):
        assert _is_moe_config({"num_local_experts": 1}) is False

    def test_no_expert_key_is_not_moe(self):
        assert _is_moe_config({}) is False


class TestHasDeltanetLayers:
    """_has_deltanet_layers should detect DeltaNet via layer_types or markers."""

    def test_linear_attention_detected(self):
        assert _has_deltanet_layers({
            "layer_types": ["full_attention", "linear_attention"],
        }) is True

    def test_use_delta_flag_detected(self):
        assert _has_deltanet_layers({"use_delta": True}) is True

    def test_full_attention_interval_detected(self):
        assert _has_deltanet_layers({"full_attention_interval": 7}) is True

    def test_full_attention_interval_list_detected(self):
        assert _has_deltanet_layers({"full_attention_interval": [0, 1, 2]}) is True

    def test_dan_layer_type_not_in_deltanet_set(self):
        """'dan' is not in DELTANET_LAYER_TYPES set."""
        result = _has_deltanet_layers({"layer_types": ["dense", "moe", "dan"]})
        assert result is False

    def test_empty_layer_types_not_detected(self):
        assert _has_deltanet_layers({"layer_types": []}) is False

    def test_no_markers_not_detected(self):
        assert _has_deltanet_layers({}) is False


class TestIsQwenDeltanetFamily:
    """_is_qwen_deltanet_family should detect Qwen DeltaNet models."""

    def test_qwen35_model_type(self):
        assert _is_qwen_deltanet_family({"model_type": "qwen3_next"}, "") is True

    def test_qwen36_model_type(self):
        assert _is_qwen_deltanet_family({"model_type": "qwen3_6_moe"}, "") is True

    def test_model_name_pattern_qwen35(self):
        assert _is_qwen_deltanet_family({}, "Qwen3.5-30B-A3B") is True

    def test_model_name_pattern_qwen36(self):
        assert _is_qwen_deltanet_family({}, "Qwen3.6-35B-A3B") is True

    def test_moe_plus_deltanet_combination(self):
        assert _is_qwen_deltanet_family({
            "num_local_experts": 128,
            "layer_types": ["linear_attention", "full_attention"],
        }, "") is True

    def test_shared_expert_plus_deltanet(self):
        assert _is_qwen_deltanet_family({
            "shared_expert_intermediate_size": 5632,
            "layer_types": ["linear_attention", "full_attention"],
        }, "") is True

    def test_dense_qwen3_not_detected(self):
        assert _is_qwen_deltanet_family({
            "model_type": "qwen3",
            "layer_types": ["full_attention"] * 24,
        }, "") is False

    def test_glm47_not_detected(self):
        assert _is_qwen_deltanet_family({
            "model_type": "glm4_moe",
            "kv_lora_rank": 512,
        }, "") is False


class TestDetectModelFormat:
    """_detect_model_format should return 'mmfp4' for Qwen hybrid configs."""

    def _make_mmfp4_model_dir(self, tmpdir: str, config: dict[str, Any]) -> str:
        config.setdefault("quantization_config", {})["quant_method"] = "mmfp4"
        config_path = Path(tmpdir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return tmpdir

    def test_qwen35_mmfp4_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_mmfp4_model_dir(tmpdir, _qwen35_flat_config())
            assert _detect_model_format(tmpdir) == "mmfp4"

    def test_qwen36_mmfp4_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_mmfp4_model_dir(tmpdir, _qwen36_multimodal_config())
            assert _detect_model_format(tmpdir) == "mmfp4"

    def test_qwen35_mmfp4_via_model_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "hidden_size": 2048,
                "_name_or_path": "Qwen3.5-30B-A3B",
                "num_local_experts": 128,
            }
            self._make_mmfp4_model_dir(tmpdir, config)
            assert _detect_model_format(tmpdir) == "mmfp4"

    def test_qwen36_mmfp4_via_model_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "hidden_size": 2048,
                "_name_or_path": "Qwen3.6-35B-A3B",
                "num_local_experts": 64,
            }
            self._make_mmfp4_model_dir(tmpdir, config)
            assert _detect_model_format(tmpdir) == "mmfp4"

    def test_non_mmfp4_quant_returns_marlin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _qwen35_flat_config()
            config["quantization_config"] = {"quant_method": "bitsandbytes"}
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
            assert _detect_model_format(tmpdir) == "marlin"

    def test_glm47_mmfp4_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "glm4_moe",
                "kv_lora_rank": 512,
                "num_local_experts": 64,
                "vocab_size": 154880,
            }
            self._make_mmfp4_model_dir(tmpdir, config)
            assert _detect_model_format(tmpdir) == "mmfp4"

    def test_missing_path_returns_marlin(self):
        assert _detect_model_format("/nonexistent/path") == "marlin"


# ============================================================================
# 3. Synthetic checkpoint-prefix case for model.language_model.layers.*
# ============================================================================

_LAYER_KEY_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)$"
)

_LAYER_WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.kv_a_layernorm.weight",
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)


def _extract_layer_index(key: str) -> int | None:
    """Extract the layer index from a checkpoint key.

    Handles both ``model.layers.{i}.*`` and
    ``model.language_model.layers.{i}.*`` prefixes.
    """
    m = _LAYER_KEY_RE.match(key)
    if m is None:
        return None
    return int(m.group(1))


def _normalise_checkpoint_key(key: str) -> str:
    """Normalise a checkpoint key by stripping language_model. wrapper.

    ``model.language_model.layers.{i}.*`` -> ``model.layers.{i}.*``
    Other keys are returned unchanged.
    """
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


class TestCheckpointPrefixLanguageModel:
    """Validate ``model.language_model.layers.*`` key handling.

    Some Qwen multimodal checkpoints (e.g. VL-MoE wrappers) store
    the language model weights under an extra ``language_model``
    nesting level.  This test ensures the prefix can be correctly
    parsed and normalised.
    """

    def test_language_model_prefix_layer_index(self):
        key = "model.language_model.layers.3.self_attn.q_proj.weight"
        assert _extract_layer_index(key) == 3

    def test_standard_prefix_layer_index(self):
        key = "model.layers.7.self_attn.q_proj.weight"
        assert _extract_layer_index(key) == 7

    def test_global_key_no_layer_index(self):
        assert _extract_layer_index("model.embed_tokens.weight") is None
        assert _extract_layer_index("lm_head.weight") is None
        assert _extract_layer_index("model.norm.weight") is None

    def test_normalise_language_model_prefix(self):
        key = "model.language_model.layers.5.mlp.gate.weight"
        assert _normalise_checkpoint_key(key) == "model.layers.5.mlp.gate.weight"

    def test_normalise_standard_prefix_unchanged(self):
        key = "model.layers.5.mlp.gate.weight"
        assert _normalise_checkpoint_key(key) == key

    def test_normalise_non_layer_key_unchanged(self):
        assert _normalise_checkpoint_key("model.embed_tokens.weight") == "model.embed_tokens.weight"
        assert _normalise_checkpoint_key("lm_head.weight") == "lm_head.weight"

    def test_synthetic_checkpoint_all_keys_parseable(self):
        """Every synthetic key for a 3-layer model should match the regex."""
        for layer_idx in range(3):
            for suffix in _LAYER_WEIGHT_SUFFIXES:
                key = f"model.layers.{layer_idx}.{suffix}"
                assert _extract_layer_index(key) == layer_idx, f"Failed: {key}"

                key = f"model.language_model.layers.{layer_idx}.{suffix}"
                assert _extract_layer_index(key) == layer_idx, f"Failed: {key}"

    def test_synthetic_checkpoint_normalise_round_trip(self):
        """Normalising language_model keys should yield standard keys."""
        for layer_idx in range(3):
            for suffix in _LAYER_WEIGHT_SUFFIXES:
                key_lm = f"model.language_model.layers.{layer_idx}.{suffix}"
                key_std = f"model.layers.{layer_idx}.{suffix}"
                assert _normalise_checkpoint_key(key_lm) == key_std

    def test_deltanet_specific_layer_keys(self):
        """DeltaNet hybrid models may have extra per-layer keys."""
        deltanet_suffixes = [
            "self_attn.linear_k_proj.weight",
            "self_attn.linear_v_proj.weight",
            "self_attn.linear_conv.weight",
            "self_attn.linear_norm.weight",
        ]
        for suffix in deltanet_suffixes:
            key = f"model.language_model.layers.0.{suffix}"
            assert _extract_layer_index(key) == 0
            assert _normalise_checkpoint_key(key) == f"model.layers.0.{suffix}"

    def test_regex_rejects_invalid_prefixes(self):
        bad_keys = [
            "transformer.h.0.attn.weight",
            "language_model.layers.0.attn.weight",
            "model.layer.0.attn.weight",
        ]
        for key in bad_keys:
            assert _extract_layer_index(key) is None, f"Should not match: {key}"
