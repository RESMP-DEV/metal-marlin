"""Utilities for replacing nn.Linear layers with MetalQuantizedLinear."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .inference_metal import MetalQuantizedLinear
from .mr_gptq import MRGPTQQuantizer, QuantizationFormat

_SUPPORTED_FORMATS = {"fp4", "int4", "nf4"}


def find_linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Find all nn.Linear layers in a model.

    Args:
        model: PyTorch model to scan.

    Returns:
        Mapping of layer name -> nn.Linear module.
    """
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def get_parent_module(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a dotted module path to its parent module and attribute name.

    Args:
        model: Root model.
        name: Dotted module path (e.g., "model.layers.0.mlp").

    Returns:
        Tuple of (parent_module, attribute_name).
    """
    if "." not in name:
        return model, name

    parent_path, attr = name.rsplit(".", 1)
    parent: nn.Module = model
    for part in parent_path.split("."):
        parent = getattr(parent, part)
    return parent, attr


def quantize_linear_layer(
    linear: nn.Linear,
    bits: int,
    group_size: int,
    format: str,
) -> MetalQuantizedLinear:
    """Quantize an nn.Linear layer into MetalQuantizedLinear.

    Uses MR-GPTQ (Hessian-aware) when calibration data is available on the layer;
    otherwise falls back to RTN (round-to-nearest) quantization.

    Notes:
        MetalQuantizedLinear currently dispatches FP4 kernels for 4-bit weights.
        Using "int4" or "nf4" formats requires compatible kernels at inference.

    Args:
        linear: Source nn.Linear module.
        bits: Quantization bit width (2, 4, or 8).
        group_size: Quantization group size along input dimension.
        format: Quantization format ("fp4", "int4", "nf4") for 4-bit quantization.

    Returns:
        Quantized MetalQuantizedLinear layer.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(linear)}")
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits: {bits}. Use 2, 4, or 8.")
    fmt = format.lower()
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Use one of {_SUPPORTED_FORMATS}.")

    if bits != 4:
        if fmt != "fp4":
            raise ValueError("Non-4-bit quantization requires format='fp4'.")
        return MetalQuantizedLinear.from_float(linear, bits=bits, group_size=group_size)

    hessian = _extract_hessian(linear)
    hessian_np = _coerce_hessian(hessian) if hessian is not None else None
    use_hadamard = _extract_use_hadamard(linear)

    quantizer = MRGPTQQuantizer(
        bits=4,
        format=QuantizationFormat(fmt),
        group_size=group_size,
        use_hadamard=use_hadamard,
    )

    weight = linear.weight.detach().float().cpu().numpy()
    packed, scales, _meta = quantizer.quantize_layer(
        weight,
        hessian=hessian_np,
        layer_name=getattr(linear, "_metal_marlin_layer_name", ""),
        use_hadamard=use_hadamard,
    )

    out_features, in_features = linear.weight.shape
    has_bias = linear.bias is not None
    layer = MetalQuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        bits=4,
        group_size=group_size,
        bias=has_bias,
    )

    # MR-GPTQ packs as [out, in//8]; MetalQuantizedLinear expects [K//8, N].
    packed_t = torch.from_numpy(packed).to(torch.uint32).T
    scales_t = torch.from_numpy(scales).to(torch.float16).T

    if layer._needs_output_slice:
        pad_cols = layer._padded_out_features - out_features
        packed_t = torch.nn.functional.pad(packed_t, (0, pad_cols, 0, 0))
        scales_t = torch.nn.functional.pad(scales_t, (0, pad_cols, 0, 0))

    layer.weight_packed.copy_(packed_t.to("mps"))
    layer.scales.copy_(scales_t.to("mps"))
    if has_bias:
        layer.bias.copy_(linear.bias.detach().half().to("mps"))

    return layer


def replace_linear_layers(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    format: str = "fp4",
    skip_patterns: list[str] | None = None,
    layer_config: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Replace nn.Linear layers with MetalQuantizedLinear in-place.

    Args:
        model: Any nn.Module (typically from AutoModelForCausalLM).
        bits: Default bit width (4 for FP4/INT4).
        group_size: Default quantization group size.
        format: Quantization format ("fp4", "int4", "nf4") for 4-bit quantization.
        skip_patterns: Layer name patterns to skip (keep as nn.Linear).
        layer_config: Per-layer config overrides, matched by exact layer name.
            Supports keys: "bits", "group_size", "format", "skip", "hessian",
            and "use_hadamard".

    Returns:
        Dict with replacement statistics:
        - replaced_count: int
        - skipped_count: int
        - replaced_layers: list[str]
        - total_params_quantized: int
    """
    skip_patterns = skip_patterns or []
    layer_config = layer_config or {}

    replacements: list[tuple[str, nn.Linear]] = []
    skipped_count = 0

    for name, module in model.named_modules():
        if isinstance(module, MetalQuantizedLinear):
            skipped_count += 1
            continue
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in skip_patterns):
            skipped_count += 1
            continue
        cfg = layer_config.get(name, {})
        if cfg.get("skip", False):
            skipped_count += 1
            continue
        replacements.append((name, module))

    replaced_layers: list[str] = []
    total_params_quantized = 0

    for name, module in replacements:
        cfg = layer_config.get(name, {})
        layer_bits = int(cfg.get("bits", bits))
        layer_group_size = int(cfg.get("group_size", group_size))
        layer_format = str(cfg.get("format", format))

        if "hessian" in cfg:
            setattr(module, "_metal_marlin_hessian", cfg["hessian"])
        if "use_hadamard" in cfg:
            setattr(module, "_metal_marlin_use_hadamard", cfg["use_hadamard"])
        setattr(module, "_metal_marlin_layer_name", name)

        try:
            quantized = quantize_linear_layer(
                module,
                bits=layer_bits,
                group_size=layer_group_size,
                format=layer_format,
            )
        except Exception:
            skipped_count += 1
            _cleanup_layer_overrides(module)
            continue

        parent, attr = get_parent_module(model, name)
        setattr(parent, attr, quantized)

        replaced_layers.append(name)
        total_params_quantized += module.weight.numel()
        if module.bias is not None:
            total_params_quantized += module.bias.numel()

        _cleanup_layer_overrides(module)

    return {
        "replaced_count": len(replaced_layers),
        "skipped_count": skipped_count,
        "replaced_layers": replaced_layers,
        "total_params_quantized": total_params_quantized,
    }


def _extract_hessian(module: nn.Module) -> Any | None:
    for attr in ("_metal_marlin_hessian", "hessian", "calibration_hessian", "hessian_info"):
        if hasattr(module, attr):
            value = getattr(module, attr)
            if value is not None:
                return value
    return None


def _coerce_hessian(value: Any) -> np.ndarray:
    if hasattr(value, "hessian"):
        value = value.hessian
    if isinstance(value, dict) and "hessian" in value:
        value = value["hessian"]
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    raise TypeError(f"Unsupported hessian type: {type(value)}")


def _extract_use_hadamard(module: nn.Module) -> bool:
    value = getattr(module, "_metal_marlin_use_hadamard", None)
    if value is None:
        return False
    return bool(value)


def _cleanup_layer_overrides(module: nn.Module) -> None:
    for attr in ("_metal_marlin_hessian", "_metal_marlin_use_hadamard", "_metal_marlin_layer_name"):
        if hasattr(module, attr):
            delattr(module, attr)
