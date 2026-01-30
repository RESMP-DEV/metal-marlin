"""Layer replacement utilities for Parakeet ASR models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..inference_metal import MetalQuantizedLinear
from .quant_policy import ParakeetQuantPolicy


def _should_quantize(layer_name: str, policy: ParakeetQuantPolicy) -> bool:
    """Check if a layer should be quantized based on policy."""
    layer_name_lower = layer_name.lower()

    # Attention QKV projections
    if (
        "qkv" in layer_name_lower
        or "query" in layer_name_lower
        or "key" in layer_name_lower
        or "value" in layer_name_lower
    ):
        return policy.quantize_attention_qkv

    # Attention output projection
    if (
        "attention.out" in layer_name_lower
        or "attn.out" in layer_name_lower
        or "proj.out" in layer_name_lower
    ):
        return policy.quantize_attention_out

    # FFN layers
    if "ffn" in layer_name_lower or "feed_forward" in layer_name_lower:
        return policy.quantize_ffn

    # Joint network
    if "joint" in layer_name_lower:
        return policy.quantize_joint

    # Predictor LSTM
    if "lstm" in layer_name_lower:
        return policy.quantize_predictor_lstm

    # Conv layers (keep FP16)
    if "conv" in layer_name_lower:
        return not policy.keep_fp16_conv

    # Embedding layers (keep FP16)
    if "embed" in layer_name_lower:
        return not policy.keep_fp16_embedding

    # LayerNorm layers (keep FP16)
    if "layernorm" in layer_name_lower or "ln" in layer_name_lower:
        return not policy.keep_fp16_layernorm

    # Default: quantize if it's Linear and not explicitly excluded
    return True


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a module by name (handles nested paths)."""
    if "." not in name:
        setattr(model, name, new_module)
        return

    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def replace_parakeet_linear_layers(
    model: nn.Module,
    policy: ParakeetQuantPolicy,
    calibration_data: torch.Tensor | None = None,
) -> nn.Module:
    """
    Replace nn.Linear layers with MetalQuantizedLinear based on policy.

    - FFN linear1, linear2 -> quantized
    - Attention qkv, out -> quantized
    - Joint projections -> quantized
    - Conv layers -> keep FP16
    - LSTM layers -> optional quantization
    - Embedding layers -> keep FP16
    - LayerNorm layers -> keep FP16
    """
    replaced_count = 0
    skipped_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if _should_quantize(name, policy):
            try:
                # Create quantized layer
                quantized = MetalQuantizedLinear.from_linear(
                    module, bits=policy.bits, group_size=policy.group_size
                )

                # Replace the layer
                _set_module(model, name, quantized)
                replaced_count += 1

            except Exception as e:
                print(f"Warning: Failed to quantize layer {name}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"Replaced {replaced_count} Linear layers with MetalQuantizedLinear")
    print(f"Skipped {skipped_count} Linear layers (kept as nn.Linear)")

    return model
