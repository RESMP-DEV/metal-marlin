from __future__ import annotations
import logging

import importlib
from types import SimpleNamespace

import pytest

from metal_marlin.inference.mmfp4_pipeline import (
    DEFAULT_MLA_KV_QUANT_MODE,
    MMFP4Pipeline,
)
from metal_marlin.kv_cache import MLAKVCache


logger = logging.getLogger(__name__)

KV_CACHE_MODULE = importlib.import_module("metal_marlin.kv_cache")


class _DummyAttention:
    def __init__(self) -> None:
        logger.debug("initializing %s", type(self).__name__)
        self.use_paged_attention = False
        self.use_fused_decode = False
        self.prefer_glm4_fused_kernel = False


class _DummyLayer:
    def __init__(self) -> None:
        logger.debug("initializing %s", type(self).__name__)
        self.self_attn = _DummyAttention()


class _DummyModel:
    def __init__(self, device: str = "cpu", num_layers: int = 2) -> None:
        logger.debug("initializing %s with device=%s, num_layers=%s", type(self).__name__, device, num_layers)
        self.device = device
        self.config = SimpleNamespace()
        self.model = SimpleNamespace(
            layers=[_DummyLayer() for _ in range(num_layers)],
        )

    def eval(self) -> _DummyModel:
        logger.debug("eval called")
        return self


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _build_pipeline(device: str = "cpu") -> MMFP4Pipeline:
    logger.info("_build_pipeline starting")
    model = _DummyModel(device=device)
    tokenizer = _DummyTokenizer()
    return MMFP4Pipeline(model, tokenizer, use_paged_attention=False)


def test_mla_glm4_kernel_preference_uses_runtime_library_fallback(
    monkeypatch,
) -> None:
    logger.info("running test_mla_glm4_kernel_preference_uses_runtime_library_fallback")
    class _RuntimeLibrary:
        def get_function(self, function_name: str) -> object:
            logger.debug("get_function called with function_name=%s", function_name)
            if function_name != "mla_fused_attention_decode_glm4":
                raise KeyError(function_name)
            return object()

    monkeypatch.setattr("metal_marlin.metal_dispatch.get_kernel", lambda _: None)
    monkeypatch.setattr(
        "metal_marlin.mla_fused._get_metal_library",
        lambda: _RuntimeLibrary(),
    )

    pipeline = _build_pipeline(device="mps")

    assert pipeline._glm4_mla_decode_kernel_available is True
    for layer in pipeline.model.model.layers:
        attn = layer.self_attn
        assert attn.use_fused_decode is True
        assert attn.prefer_glm4_fused_kernel is True


def test_mla_glm4_kernel_preference_falls_back_when_unavailable(
    monkeypatch,
) -> None:
    logger.info("running test_mla_glm4_kernel_preference_falls_back_when_unavailable")
    class _NoKernelLibrary:
        def get_function(self, function_name: str) -> object:
            logger.debug("get_function called with function_name=%s", function_name)
            raise KeyError(function_name)

    monkeypatch.setattr("metal_marlin.metal_dispatch.get_kernel", lambda _: None)
    monkeypatch.setattr(
        "metal_marlin.mla_fused._get_metal_library",
        lambda: _NoKernelLibrary(),
    )

    pipeline = _build_pipeline(device="mps")

    assert pipeline._glm4_mla_decode_kernel_available is False
    for layer in pipeline.model.model.layers:
        attn = layer.self_attn
        assert attn.use_fused_decode is True
        assert attn.prefer_glm4_fused_kernel is False


def test_mla_kv_cache_defaults_to_int8_for_non_paged_pipeline(monkeypatch) -> None:
    logger.info("running test_mla_kv_cache_defaults_to_int8_for_non_paged_pipeline")
    monkeypatch.setattr(KV_CACHE_MODULE, "require_mps", lambda *_args, **_kwargs: None)

    pipeline = _build_pipeline(device="cpu")
    pipeline.model.config.kv_lora_rank = 128
    pipeline.model.config.qk_rope_head_dim = 64

    cache = pipeline._create_kv_cache(batch_size=1, max_seq_len=128)

    assert isinstance(cache, MLAKVCache)
    assert cache.quantize_mode == DEFAULT_MLA_KV_QUANT_MODE
