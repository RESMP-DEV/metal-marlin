"""Import regression tests for Trellis public API exports."""

from __future__ import annotations

import importlib

import pytest

from metal_marlin._compat import HAS_TORCH, torch

STABLE_TRELLIS_SYMBOLS = (
    "TrellisForCausalLM",
    "CausalLMOutput",
    "TrellisModel",
    "TrellisDecoderLayer",
    "TrellisMoEMLP",
    "TrellisLinear",
    "TrellisModelLoader",
    "TrellisWeight",
    "TrellisModelConfig",
    "GenerationConfig",
    "TrellisGenerator",
)

LM_LOCAL_PUBLIC_SYMBOLS = (
    "TrellisForCausalLM",
    "CausalLMOutput",
)

LM_MODEL_REEXPORT_SYMBOLS = (
    "TrellisModel",
    "TrellisDecoderLayer",
    "TrellisMoEMLP",
)

_OPTIONAL_TRELLIS_DEPS = {"torch", "transformers", "numpy", "safetensors"}


def _has_usable_torch() -> bool:
    if not HAS_TORCH or torch is None:
        return False
    required_attrs = ("float16", "bfloat16", "float32", "Tensor", "nn")
    return all(hasattr(torch, attr) for attr in required_attrs)


def _require_trellis_runtime() -> None:
    if not _has_usable_torch():
        pytest.skip("Requires a functional PyTorch runtime for Trellis imports")


def _import_trellis_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing_name = (exc.name or "").split(".")[0]
        if missing_name in _OPTIONAL_TRELLIS_DEPS:
            pytest.skip(f"Missing runtime dependency for Trellis imports: {missing_name}")
        raise


def test_stable_public_symbols_importable_from_trellis() -> None:
    """Stable symbols should be importable from ``metal_marlin.trellis``."""
    _require_trellis_runtime()
    trellis = _import_trellis_module("metal_marlin.trellis")

    from metal_marlin.trellis import (
        CausalLMOutput,
        GenerationConfig,
        TrellisDecoderLayer,
        TrellisForCausalLM,
        TrellisGenerator,
        TrellisLinear,
        TrellisModel,
        TrellisModelConfig,
        TrellisModelLoader,
        TrellisMoEMLP,
        TrellisWeight,
    )

    imported_symbols = {
        "TrellisForCausalLM": TrellisForCausalLM,
        "CausalLMOutput": CausalLMOutput,
        "TrellisModel": TrellisModel,
        "TrellisDecoderLayer": TrellisDecoderLayer,
        "TrellisMoEMLP": TrellisMoEMLP,
        "TrellisLinear": TrellisLinear,
        "TrellisModelLoader": TrellisModelLoader,
        "TrellisWeight": TrellisWeight,
        "TrellisModelConfig": TrellisModelConfig,
        "GenerationConfig": GenerationConfig,
        "TrellisGenerator": TrellisGenerator,
    }

    for symbol_name in STABLE_TRELLIS_SYMBOLS:
        assert symbol_name in trellis.__all__
        assert hasattr(trellis, symbol_name)
        assert imported_symbols[symbol_name] is getattr(trellis, symbol_name)


def test_trellis_lm_reexports_are_backward_compatible() -> None:
    """``metal_marlin.trellis.lm`` should re-export stable Trellis model symbols."""
    _require_trellis_runtime()
    trellis_lm = _import_trellis_module("metal_marlin.trellis.lm")
    trellis_model = _import_trellis_module("metal_marlin.trellis.model")

    for symbol_name in LM_LOCAL_PUBLIC_SYMBOLS:
        assert symbol_name in trellis_lm.__all__
        assert hasattr(trellis_lm, symbol_name)

    for symbol_name in LM_MODEL_REEXPORT_SYMBOLS:
        assert symbol_name in trellis_lm.__all__
        assert hasattr(trellis_lm, symbol_name)
        assert getattr(trellis_lm, symbol_name) is getattr(trellis_model, symbol_name)
