"""Replace Linear layers with Metal-accelerated quantized versions."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..ops.metal_linear import MetalQuantizedLinear


def replace_linear_with_metal(
    model: nn.Module,
    quant_type: str = "int8",
    group_size: int = 128,
    skip_patterns: list[str] | None = None,
    calibration_data: list[torch.Tensor] | None = None,
) -> nn.Module:
    """Replace nn.Linear layers with MetalQuantizedLinear.

    Args:
        model: Model to modify (in-place)
        quant_type: Quantization type ("int8" or "fp4")
        group_size: Quantization group size
        skip_patterns: Layer name patterns to skip (e.g., ["output", "head"])
        calibration_data: Optional data for calibration

    Returns:
        Modified model with Metal quantized layers
    """
    skip_patterns = skip_patterns or []

    # Collect Linear layers to replace
    layers_to_replace: list[tuple[str, nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check skip patterns
            if any(pat in name for pat in skip_patterns):
                print(f"Skipping {name} (matches skip pattern)")
                continue

            # Find parent module
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                child_name = name

            layers_to_replace.append((name, parent, child_name, module))

    # Replace layers
    replaced_count = 0
    for name, parent, child_name, linear in layers_to_replace:
        try:
            metal_linear = MetalQuantizedLinear.from_linear(
                linear,
                quant_type=quant_type,
                group_size=group_size,
            )
            setattr(parent, child_name, metal_linear)
            replaced_count += 1
        except Exception as e:
            print(f"Warning: Could not replace {name}: {e}")

    print(f"Replaced {replaced_count} Linear layers with MetalQuantizedLinear ({quant_type})")
    return model


def replace_parakeet_encoder_layers(
    model: nn.Module,
    quant_type: str = "int8",
    group_size: int = 128,
) -> nn.Module:
    """Replace Linear layers in Parakeet encoder only.

    Keeps decoder in FP16 for accuracy, quantizes encoder for speed.

    Args:
        model: ParakeetTDT model
        quant_type: Quantization type
        group_size: Quantization group size

    Returns:
        Model with quantized encoder
    """
    if not hasattr(model, "encoder"):
        raise ValueError("Model must have 'encoder' attribute")

    encoder = model.encoder
    if not isinstance(encoder, nn.Module):
        raise ValueError(f"model.encoder must be nn.Module, got {type(encoder)}")

    # Only replace encoder layers
    replace_linear_with_metal(
        encoder,
        quant_type=quant_type,
        group_size=group_size,
        skip_patterns=["output_proj"],  # Keep output projection FP16
    )

    return model
