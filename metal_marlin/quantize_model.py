"""Model-level quantization API for Metal Marlin (PyTorch backend).

Provides a single-call interface to quantize all nn.Linear layers in a
PyTorch model to Marlin FP4 (E2M1) format. Replaces Linear layers
with MarlinLinear layers backed by quantized weight storage.

This module is the PyTorch-only implementation, independent of MLX.

Usage:
    from metal_marlin.quantize_model import quantize_model

    model = load_my_model()
    quantize_model(model, group_size=128)
    # All nn.Linear layers (except lm_head, embed_tokens, etc.) are now
    # MarlinLinear with FP4 weights.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch.nn as torch_nn

from .layers import MarlinLinear

# Constants
FP4_PER_U32 = 8  # 8 FP4 values packed per uint32


# ---------------------------------------------------------------------------
# Model Quantization
# ---------------------------------------------------------------------------


def _quantize_torch(
    model: Any,
    quant_type: str,
    group_size: int,
    skip_layers: set[str],
    layer_filter: Callable[[str, Any], bool] | None,
) -> Any:
    """Quantize PyTorch model in-place."""

    def _should_quantize(name: str, module: Any) -> bool:
        if not isinstance(module, torch_nn.Linear):
            return False
        if any(skip in name for skip in skip_layers):
            return False
        if layer_filter is not None and not layer_filter(name, module):
            return False
        in_features = module.in_features
        out_features = module.out_features
        # Check divisibility requirements
        if in_features % group_size != 0:
            return False
        if out_features % FP4_PER_U32 != 0:
            return False
        return True

    # Collect modules to replace (can't modify during iteration)
    replacements: list[tuple[Any, str, torch_nn.Linear]] = []

    for name, module in model.named_modules():
        if _should_quantize(name, module):
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                child_name = name
            replacements.append((parent, child_name, module))

    # Apply replacements
    for parent, child_name, linear in replacements:
        quantized = MarlinLinear.from_linear(linear, quant_type=quant_type, group_size=group_size)
        setattr(parent, child_name, quantized)

    return model


def quantize_model(
    model: torch_nn.Module,
    quant_type: Literal["fp4"] = "fp4",
    group_size: int = 128,
    skip_layers: set[str] | None = None,
    layer_filter: Callable[[str, Any], bool] | None = None,
) -> torch_nn.Module:
    """Quantize all nn.Linear layers in a PyTorch model to Marlin FP4 format.

    Traverses the model's modules and replaces qualifying nn.Linear
    layers with MarlinLinear instances. The model is modified in-place.

    Args:
        model: PyTorch model to quantize.
        quant_type: Quantization format. Currently only "fp4" (E2M1) is
            supported.
        group_size: Number of elements per quantization group along the
            input dimension. Must divide the layer's in_features. Default: 128.
        skip_layers: Set of name fragments to skip. A layer is skipped if
            any fragment appears anywhere in its dotted path. Defaults to
            {"lm_head", "embed_tokens", "wte", "wpe"}.
        layer_filter: Optional predicate (path, module) -> bool. When
            provided, a layer is only quantized if this returns True.
            Applied after skip_layers filtering.

    Returns:
        The same model instance with qualifying layers replaced in-place.
    """
    if quant_type != "fp4":
        raise NotImplementedError(
            f"Only quant_type='fp4' is currently supported, got {quant_type!r}"
        )

    if not isinstance(model, torch_nn.Module):
        raise TypeError(f"Model type {type(model)} not supported. Requires PyTorch nn.Module.")

    skip_layers = skip_layers or {"lm_head", "embed_tokens", "wte", "wpe"}
    return _quantize_torch(model, quant_type, group_size, skip_layers, layer_filter)


def estimate_model_size(model: torch_nn.Module, group_size: int = 128) -> dict[str, float]:
    """Estimate model memory footprint comparing FP16 vs FP4 quantization.

    Walks the model's parameters and estimates memory usage. For layers
    that are already MarlinLinear, counts packed weight + scales. For
    nn.Linear layers, estimates what quantized size would be.

    Args:
        model: PyTorch model to analyze (quantized or unquantized).
        group_size: Group size for estimation of unquantized layers.

    Returns:
        Dict with keys:
            fp16_bytes: Bytes for non-quantized parameters (FP16).
            quantized_bytes: Bytes for quantized weight storage.
            total_bytes: Sum of fp16_bytes + quantized_bytes.
            fp16_mb, quantized_mb, total_mb: Same in MiB.
            num_quantized_layers: Count of MarlinLinear layers.
            num_unquantized_layers: Count of remaining nn.Linear layers.
    """
    if not isinstance(model, torch_nn.Module):
        raise TypeError(f"Model type {type(model)} not supported. Requires PyTorch nn.Module.")

    fp16_bytes = 0
    quantized_bytes = 0
    num_quantized = 0
    num_unquantized = 0

    for name, module in model.named_modules():
        if isinstance(module, MarlinLinear):
            num_quantized += 1
            K = module.in_features
            N = module.out_features
            # FP4: 4 bits per element = K * N / 2 bytes
            weight_bytes = (K * N) // 2
            num_groups = K // module.group_size
            # Scales: FP16 = 2 bytes each
            scale_bytes = num_groups * N * 2
            bias_bytes = N * 2 if module.bias is not None else 0
            quantized_bytes += weight_bytes + scale_bytes + bias_bytes
        elif isinstance(module, torch_nn.Linear):
            num_unquantized += 1
            in_features = module.in_features
            out_features = module.out_features
            # FP16: 2 bytes per element
            fp16_bytes += out_features * in_features * 2
            if module.bias is not None:
                fp16_bytes += out_features * 2

    total = fp16_bytes + quantized_bytes
    return {
        "fp16_bytes": fp16_bytes,
        "quantized_bytes": quantized_bytes,
        "total_bytes": total,
        "fp16_mb": fp16_bytes / (1024 * 1024),
        "quantized_mb": quantized_bytes / (1024 * 1024),
        "total_mb": total / (1024 * 1024),
        "num_quantized_layers": num_quantized,
        "num_unquantized_layers": num_unquantized,
    }
