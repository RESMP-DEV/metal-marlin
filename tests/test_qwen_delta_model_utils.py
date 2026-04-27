"""Tests for Qwen hybrid DeltaNet-family config helpers in model_utils.

Covers get_effective_text_config, get_layer_types, get_full_attention_interval,
get_deltanet_metadata, and is_qwen_hybrid_deltanet with mock config objects
that mirror the structure of real Qwen3.5 / Qwen3.6 HuggingFace configs.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from metal_marlin.model_utils import (
    get_deltanet_metadata,
    get_effective_text_config,
    get_full_attention_interval,
    get_layer_types,
    is_qwen_hybrid_deltanet,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight config stubs — no HF dependency required
# ---------------------------------------------------------------------------


class _StubConfig:
    """Minimal config stub that supports ``to_dict`` like PreTrainedConfig."""

    def __init__(self, **kwargs: Any) -> None:
        logger.debug("initializing %s", type(self).__name__)
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        logger.debug("to_dict called")
        return dict(self.__dict__)


class _StubMultimodalConfig:
    """Multimodal wrapper that nests a ``text_config`` sub-object."""

    def __init__(self, text_config: Any, **kwargs: Any) -> None:
        logger.debug("initializing %s with text_config=%s", type(self).__name__, text_config)
        self.text_config = text_config
        self.__dict__.update(kwargs)


# ---------------------------------------------------------------------------
# get_effective_text_config
# ---------------------------------------------------------------------------


class TestGetEffectiveTextConfig:
    """Verify normalisation of top-level multimodal config vs nested text_config."""

    def test_dense_config_returned_as_is(self) -> None:
        logger.info("running test_dense_config_returned_as_is")
        cfg = _StubConfig(vocab_size=151936, hidden_size=2048)
        assert get_effective_text_config(cfg) is cfg

    def test_multimodal_returns_text_config(self) -> None:
        logger.info("running test_multimodal_returns_text_config")
        text_cfg = _StubConfig(vocab_size=151936, hidden_size=2048)
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg)
        assert get_effective_text_config(mm_cfg) is text_cfg

    def test_multimodal_without_vocab_size_falls_back(self) -> None:
        """If text_config lacks vocab_size, return the wrapper itself."""
        logger.info("running test_multimodal_without_vocab_size_falls_back")
        text_cfg = _StubConfig(hidden_size=2048)  # no vocab_size
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg)
        assert get_effective_text_config(mm_cfg) is mm_cfg

    def test_text_config_none(self) -> None:
        """text_config=None should not raise."""
        logger.info("running test_text_config_none")
        cfg = _StubConfig(text_config=None, vocab_size=100)
        assert get_effective_text_config(cfg) is cfg

    def test_plain_object_without_text_config(self) -> None:
        """An object with no text_config attribute is returned unchanged."""

        logger.info("running test_plain_object_without_text_config")
        class Simple:
            vocab_size = 100

        assert get_effective_text_config(Simple()) is not None


# ---------------------------------------------------------------------------
# get_layer_types
# ---------------------------------------------------------------------------


class TestGetLayerTypes:
    """Verify layer_types retrieval from flat and multimodal configs."""

    def test_dense_config_no_layer_types(self) -> None:
        logger.info("running test_dense_config_no_layer_types")
        cfg = _StubConfig(vocab_size=151936)
        assert get_layer_types(cfg) is None

    def test_flat_deltanet_config(self) -> None:
        logger.info("running test_flat_deltanet_config")
        layer_types = ["linear_attention", "full_attention", "linear_attention"]
        cfg = _StubConfig(layer_types=layer_types)
        assert get_layer_types(cfg) == layer_types

    def test_multimodal_deltanet_config(self) -> None:
        """layer_types inside text_config should be surfaced."""
        logger.info("running test_multimodal_deltanet_config")
        layer_types = ["full_attention", "linear_attention"]
        text_cfg = _StubConfig(vocab_size=151936, layer_types=layer_types)
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg)
        assert get_layer_types(mm_cfg) == layer_types

    def test_top_level_layer_types_preferred(self) -> None:
        """When both levels have layer_types, top-level wins."""
        logger.info("running test_top_level_layer_types_preferred")
        top = ["full_attention"]
        nested = ["linear_attention"]
        text_cfg = _StubConfig(vocab_size=151936, layer_types=nested)
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg, layer_types=top)
        assert get_layer_types(mm_cfg) == top

    def test_layer_types_not_a_list(self) -> None:
        """Non-list layer_types should be ignored (return None)."""
        logger.info("running test_layer_types_not_a_list")
        cfg = _StubConfig(layer_types="full_attention")
        assert get_layer_types(cfg) is None


# ---------------------------------------------------------------------------
# get_full_attention_interval
# ---------------------------------------------------------------------------


class TestGetFullAttentionInterval:
    """Verify full_attention_interval retrieval from flat and multimodal configs."""

    def test_direct_attribute(self) -> None:
        logger.info("running test_direct_attribute")
        cfg = _StubConfig(full_attention_interval=4)
        assert get_full_attention_interval(cfg) == 4

    def test_via_to_dict(self) -> None:
        """Attribute reachable only via to_dict fallback."""

        logger.info("running test_via_to_dict")
        class DictOnly:
            def to_dict(self) -> dict[str, Any]:
                logger.debug("to_dict called")
                return {"full_attention_interval": 8}

        assert get_full_attention_interval(DictOnly()) == 8

    def test_via_dunder_dict(self) -> None:
        """Object without to_dict but with __dict__."""

        logger.info("running test_via_dunder_dict")
        class NoToDict:
            pass

        obj = NoToDict()
        obj.full_attention_interval = 3
        assert get_full_attention_interval(obj) == 3

    def test_missing_returns_none(self) -> None:
        logger.info("running test_missing_returns_none")
        cfg = _StubConfig(vocab_size=100)
        assert get_full_attention_interval(cfg) is None

    def test_none_value_returns_none(self) -> None:
        logger.info("running test_none_value_returns_none")
        cfg = _StubConfig(full_attention_interval=None)
        assert get_full_attention_interval(cfg) is None

    def test_string_value_coerced(self) -> None:
        """String that can be int()-ed should still work."""
        logger.info("running test_string_value_coerced")
        cfg = _StubConfig(full_attention_interval="4")
        assert get_full_attention_interval(cfg) == 4

    def test_uncoercible_string_returns_none(self) -> None:
        logger.info("running test_uncoercible_string_returns_none")
        cfg = _StubConfig(full_attention_interval="not_a_number")
        assert get_full_attention_interval(cfg) is None

    def test_multimodal_reads_text_config(self) -> None:
        """For multimodal configs, full_attention_interval in text_config
        should be found even if absent at top level."""
        logger.info("running test_multimodal_reads_text_config")
        text_cfg = _StubConfig(vocab_size=151936, full_attention_interval=7)
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg)
        assert get_full_attention_interval(mm_cfg) == 7

    def test_multimodal_top_level_preferred(self) -> None:
        """Top-level full_attention_interval takes precedence over text_config."""
        logger.info("running test_multimodal_top_level_preferred")
        text_cfg = _StubConfig(vocab_size=151936, full_attention_interval=7)
        mm_cfg = _StubMultimodalConfig(
            text_config=text_cfg, full_attention_interval=2
        )
        assert get_full_attention_interval(mm_cfg) == 2


# ---------------------------------------------------------------------------
# get_deltanet_metadata
# ---------------------------------------------------------------------------

# A complete set of DeltaNet fields matching official Qwen3.5 / Qwen3.6 configs.
_COMPLETE_DELTANET_FIELDS = dict(
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
)


class TestGetDeltanetMetadata:
    """Verify DeltaNet metadata extraction from flat and multimodal configs."""

    def test_wrong_model_type_returns_none(self) -> None:
        logger.info("running test_wrong_model_type_returns_none")
        cfg = _StubConfig(model_type="qwen3", layer_types=["linear_attention"])
        assert get_deltanet_metadata(cfg) is None

    def test_no_model_type_returns_none(self) -> None:
        logger.info("running test_no_model_type_returns_none")
        cfg = _StubConfig(layer_types=["linear_attention"])
        assert get_deltanet_metadata(cfg) is None

    def test_flat_qwen3_next_config(self) -> None:
        logger.info("running test_flat_qwen3_next_config")
        cfg = _StubConfig(model_type="qwen3_next", **_COMPLETE_DELTANET_FIELDS)
        result = get_deltanet_metadata(cfg)
        assert result is not None
        assert result == _COMPLETE_DELTANET_FIELDS

    def test_flat_qwen3_vl_moe_text_config(self) -> None:
        logger.info("running test_flat_qwen3_vl_moe_text_config")
        cfg = _StubConfig(
            model_type="qwen3_vl_moe_text", **_COMPLETE_DELTANET_FIELDS
        )
        result = get_deltanet_metadata(cfg)
        assert result is not None
        assert result["linear_key_head_dim"] == 128
        assert result["linear_conv_kernel_dim"] == 4

    def test_multimodal_reads_text_config_fields(self) -> None:
        """When the top-level config is a multimodal wrapper,
        DeltaNet fields should be read from the nested text_config."""
        logger.info("running test_multimodal_reads_text_config_fields")
        text_cfg = _StubConfig(
            model_type="qwen3_next",
            vocab_size=151936,
            **_COMPLETE_DELTANET_FIELDS,
        )
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg, model_type="qwen3_vl_moe")
        result = get_deltanet_metadata(mm_cfg)
        assert result is not None
        assert result == _COMPLETE_DELTANET_FIELDS

    def test_incomplete_fields_returns_none(self) -> None:
        """Missing any of the five fields → None."""
        logger.info("running test_incomplete_fields_returns_none")
        cfg = _StubConfig(
            model_type="qwen3_next",
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            # missing linear_num_key_heads, linear_num_value_heads, linear_conv_kernel_dim
        )
        assert get_deltanet_metadata(cfg) is None

    def test_dense_model_type_ignored_even_with_fields(self) -> None:
        logger.info("running test_dense_model_type_ignored_even_with_fields")
        cfg = _StubConfig(
            model_type="qwen3",
            **_COMPLETE_DELTANET_FIELDS,
        )
        assert get_deltanet_metadata(cfg) is None

    def test_multimodal_incomplete_text_config_returns_none(self) -> None:
        """Multimodal wrapper where text_config has wrong model_type."""
        logger.info("running test_multimodal_incomplete_text_config_returns_none")
        text_cfg = _StubConfig(
            model_type="qwen3",
            vocab_size=151936,
            **_COMPLETE_DELTANET_FIELDS,
        )
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg, model_type="qwen2_5_vl")
        assert get_deltanet_metadata(mm_cfg) is None


# ---------------------------------------------------------------------------
# is_qwen_hybrid_deltanet
# ---------------------------------------------------------------------------


class TestIsQwenHybridDeltanet:
    """Verify hybrid detection logic and backward compatibility."""

    def test_deltanet_config(self) -> None:
        logger.info("running test_deltanet_config")
        cfg = _StubConfig(
            layer_types=["linear_attention", "full_attention", "linear_attention"]
        )
        assert is_qwen_hybrid_deltanet(cfg) is True

    def test_dense_full_attention_only(self) -> None:
        logger.info("running test_dense_full_attention_only")
        cfg = _StubConfig(layer_types=["full_attention", "full_attention"])
        assert is_qwen_hybrid_deltanet(cfg) is False

    def test_dense_sliding_attention(self) -> None:
        logger.info("running test_dense_sliding_attention")
        cfg = _StubConfig(
            layer_types=["sliding_attention", "full_attention", "sliding_attention"]
        )
        assert is_qwen_hybrid_deltanet(cfg) is False

    def test_missing_layer_types(self) -> None:
        logger.info("running test_missing_layer_types")
        cfg = _StubConfig(vocab_size=100)
        assert is_qwen_hybrid_deltanet(cfg) is False

    def test_multimodal_deltanet(self) -> None:
        """Detection works through multimodal text_config."""
        logger.info("running test_multimodal_deltanet")
        text_cfg = _StubConfig(
            vocab_size=151936,
            layer_types=["full_attention", "linear_attention"],
        )
        mm_cfg = _StubMultimodalConfig(text_config=text_cfg)
        assert is_qwen_hybrid_deltanet(mm_cfg) is True

    def test_empty_layer_types_list(self) -> None:
        logger.info("running test_empty_layer_types_list")
        cfg = _StubConfig(layer_types=[])
        assert is_qwen_hybrid_deltanet(cfg) is False


# ---------------------------------------------------------------------------
# Cross-function integration: full Qwen3.5-style config
# ---------------------------------------------------------------------------


class TestIntegrationQwen35Style:
    """End-to-end check with a config that mirrors Qwen3.5-35B-A3B structure."""

    @pytest.fixture()
    def qwen35_text_config(self) -> _StubConfig:
        logger.debug("qwen35_text_config called")
        return _StubConfig(
            model_type="qwen3_next",
            vocab_size=151936,
            hidden_size=2048,
            layer_types=["linear_attention", "full_attention"] * 28,
            full_attention_interval=7,
            **_COMPLETE_DELTANET_FIELDS,
        )

    def test_all_helpers_agree_flat(self, qwen35_text_config: _StubConfig) -> None:
        """Flat (text-only) config: all helpers should work consistently."""
        logger.info("running test_all_helpers_agree_flat")
        assert get_effective_text_config(qwen35_text_config) is qwen35_text_config
        assert get_layer_types(qwen35_text_config) is not None
        assert "linear_attention" in get_layer_types(qwen35_text_config)  # type: ignore[arg-type]
        assert get_full_attention_interval(qwen35_text_config) == 7
        assert get_deltanet_metadata(qwen35_text_config) is not None
        assert is_qwen_hybrid_deltanet(qwen35_text_config) is True

    def test_all_helpers_agree_multimodal(
        self, qwen35_text_config: _StubConfig
    ) -> None:
        """Multimodal wrapper: all helpers should dig into text_config."""
        logger.info("running test_all_helpers_agree_multimodal")
        mm_cfg = _StubMultimodalConfig(
            text_config=qwen35_text_config, model_type="qwen3_vl_moe"
        )
        assert get_effective_text_config(mm_cfg) is qwen35_text_config
        assert get_layer_types(mm_cfg) is not None
        assert get_full_attention_interval(mm_cfg) == 7
        assert get_deltanet_metadata(mm_cfg) is not None
        assert is_qwen_hybrid_deltanet(mm_cfg) is True


class TestIntegrationQwen3Dense:
    """Backward-compatibility: dense Qwen3 config should not be detected as hybrid."""

    @pytest.fixture()
    def qwen3_config(self) -> _StubConfig:
        logger.debug("qwen3_config called")
        return _StubConfig(
            model_type="qwen3",
            vocab_size=151936,
            hidden_size=2048,
            layer_types=["full_attention", "sliding_attention"] * 14,
        )

    def test_dense_not_hybrid(self, qwen3_config: _StubConfig) -> None:
        logger.info("running test_dense_not_hybrid")
        assert is_qwen_hybrid_deltanet(qwen3_config) is False

    def test_dense_no_deltanet_metadata(self, qwen3_config: _StubConfig) -> None:
        logger.info("running test_dense_no_deltanet_metadata")
        assert get_deltanet_metadata(qwen3_config) is None

    def test_dense_no_full_attention_interval(
        self, qwen3_config: _StubConfig
    ) -> None:
        logger.info("running test_dense_no_full_attention_interval")
        assert get_full_attention_interval(qwen3_config) is None

    def test_dense_layer_types_found(self, qwen3_config: _StubConfig) -> None:
        logger.info("running test_dense_layer_types_found")
        assert get_layer_types(qwen3_config) is not None
        assert "linear_attention" not in get_layer_types(qwen3_config)  # type: ignore[arg-type]
