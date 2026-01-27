"""
Legacy model implementations.

DEPRECATED: These classes are maintained for backwards compatibility only.
New code should use Transformers + replace_linear_layers().

These will be removed in Metal Marlin v3.0.
"""

from __future__ import annotations

import warnings


def __import_llama():
    from . import llama as legacy_llama

    return legacy_llama


def __import_qwen3():
    from . import qwen3 as legacy_qwen3

    return legacy_qwen3


def __import_attention():
    from . import attention as legacy_attention

    return legacy_attention


def __import_mlp():
    from . import mlp as legacy_mlp

    return legacy_mlp


def __import_glm4():
    from . import glm4 as legacy_glm4

    return legacy_glm4


_LEGACY_EXPORTS = {
    # Llama
    "QuantizedLlama": lambda: __import_llama().QuantizedLlama,
    "QuantizedLlamaAttention": lambda: __import_llama().QuantizedLlamaAttention,
    "QuantizedLlamaLayer": lambda: __import_llama().QuantizedLlamaLayer,
    "QuantizedLlamaMLP": lambda: __import_llama().QuantizedLlamaMLP,
    # Qwen3
    "QuantizedQwen3Attention": lambda: __import_qwen3().QuantizedQwen3Attention,
    "QuantizedQwen3Layer": lambda: __import_qwen3().QuantizedQwen3Layer,
    "QuantizedQwen3MLP": lambda: __import_qwen3().QuantizedQwen3MLP,
    "QuantizedQwen3MoE": lambda: __import_qwen3().QuantizedQwen3MoE,
    # Metal attention/MLP
    "MetalAttention": lambda: __import_attention().MetalAttention,
    "MetalMLAAttention": lambda: __import_attention().MetalMLAAttention,
    "MetalMLP": lambda: __import_mlp().MetalMLP,
    # GLM-4.7
    "MetalGLM47Model": lambda: __import_glm4().MetalGLM47Model,
}

__all__ = sorted(_LEGACY_EXPORTS.keys())


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        warnings.warn(
            f"metal_marlin.legacy.{name} is deprecated. "
            "Use Transformers + replace_linear_layers() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = _LEGACY_EXPORTS[name]()
        globals()[name] = value
        return value
    raise AttributeError(f"module 'metal_marlin.legacy' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
