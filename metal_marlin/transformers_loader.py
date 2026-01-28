"""HuggingFace Transformers loader with Metal Marlin layer replacement.

This module loads a Transformers causal LM, optionally runs a calibration
pass to collect activation ranges, then replaces nn.Linear layers with
MetalQuantizedLinear for MPS-backed quantized inference.

Saved format:
- quantized_weights.safetensors (packed weights + scales + other params)
- config.json (original HF config)
- quantization_config.json (bits, group_size, format, per-layer info)
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._compat import HAS_TORCH, torch
from .mixed_precision import MixedPrecisionConfig, Precision, get_layer_config
from .quantize_fp4 import compute_quantization_error, quantize_fp4

_DEFAULT_SKIP_PATTERNS = [
    "lm_head",
    "embed_tokens",
    "wte",
    "wpe",
]

_DEFAULT_PERCENTILE = 99.9
_DEFAULT_SMOOTH_FACTOR = 0.8
_DEFAULT_TORCH_DTYPE = torch.bfloat16 if torch is not None else None

logger = logging.getLogger(__name__)

_MOE_MODEL_TYPE_CLASSES = {
    "glm4_moe_lite": ["Glm4MoeLiteMoE"],
    "qwen2_moe": ["Qwen2MoESparseMoeBlock"],
    "mixtral": ["MixtralSparseMoeBlock"],
    "deepseek_v2": ["DeepseekV2MoE"],
}

_MOE_CLASS_MODULES = {
    "Glm4MoeLiteMoE": [
        "transformers.models.glm4.modeling_glm4",
    ],
    "Qwen2MoESparseMoeBlock": [
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
    ],
    "MixtralSparseMoeBlock": [
        "transformers.models.mixtral.modeling_mixtral",
    ],
    "DeepseekV2MoE": [
        "transformers.models.deepseek_v2.modeling_deepseek_v2",
    ],
}

_DEFAULT_MOE_INCLUDE_PATTERNS = [
    "experts.",
    "expert.",
    "shared_expert",
    "shared_experts",
    "block_sparse_moe.experts",
    "moe.experts",
]

_DEFAULT_MOE_SKIP_PATTERNS = [
    "router",
    "moe_gate",
    "expert_gate",
    "block_sparse_moe.gate",
]


@dataclass
class LayerPlan:
    name: str
    parent: Any
    child_name: str
    module: Any
    quantize: bool
    bits: int
    group_size: int
    fmt: str
    precision: str | None
    skip_reason: str | None


if torch is not None:
    import torch.nn as nn

    class MetalQuantizedMoE(nn.Module):
        """Wrapper for MoE blocks after expert quantization."""

        def __init__(self, moe_module: nn.Module, name: str | None = None) -> None:
            super().__init__()
            self.moe = moe_module
            self.name = name

        def forward(self, *args: Any, **kwargs: Any):
            return self.moe(*args, **kwargs)


def _iter_named_modules_with_parent(model: Any) -> Iterable[tuple[str, Any, Any, str]]:
    module_map = dict(model.named_modules())
    for name, module in model.named_modules():
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = module_map[parent_name]
        else:
            parent = model
            child_name = name
        yield name, module, parent, child_name


def _resolve_moe_type_names(model: Any) -> list[str]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "") if config is not None else ""
    model_type = str(model_type).lower()
    if model_type in _MOE_MODEL_TYPE_CLASSES:
        return _MOE_MODEL_TYPE_CLASSES[model_type]

    # Fallback: detect MoE from config fields
    num_experts = None
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        if config is not None and hasattr(config, attr):
            num_experts = getattr(config, attr)
            if num_experts:
                break
    if num_experts and num_experts > 1:
        names: list[str] = []
        for class_names in _MOE_MODEL_TYPE_CLASSES.values():
            names.extend(class_names)
        return names
    return []


def _resolve_moe_types(type_names: list[str]) -> tuple[type, ...]:
    resolved: list[type] = []
    try:
        import transformers

        for class_name in type_names:
            cls = getattr(transformers, class_name, None)
            if cls is not None:
                resolved.append(cls)
    except Exception:
        pass

    for class_name in type_names:
        for module_path in _MOE_CLASS_MODULES.get(class_name, []):
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name, None)
                if cls is not None:
                    resolved.append(cls)
            except Exception:
                continue

    # Remove duplicates while preserving order
    seen: set[type] = set()
    unique: list[type] = []
    for cls in resolved:
        if cls not in seen:
            unique.append(cls)
            seen.add(cls)
    return tuple(unique)


def find_moe_layers(model: Any) -> list[tuple[str, Any, Any, str]]:
    """Find MoE layers for supported Transformers model types."""
    type_names = _resolve_moe_type_names(model)
    if not type_names:
        return []
    moe_types = _resolve_moe_types(type_names)

    layers: list[tuple[str, Any, Any, str]] = []
    for name, module, parent, child_name in _iter_named_modules_with_parent(model):
        if moe_types and isinstance(module, moe_types):
            layers.append((name, module, parent, child_name))
            continue
        if module.__class__.__name__ in type_names:
            layers.append((name, module, parent, child_name))
    return layers


def _detect_quantization_targets(config: Any) -> dict[str, list[str]]:
    """Detect MoE-specific module patterns for quantization targets."""
    model_type = str(getattr(config, "model_type", "")).lower()
    include_patterns: list[str] = []
    skip_patterns: list[str] = []

    if model_type in _MOE_MODEL_TYPE_CLASSES:
        include_patterns.extend(_DEFAULT_MOE_INCLUDE_PATTERNS)
        skip_patterns.extend(_DEFAULT_MOE_SKIP_PATTERNS)
    else:
        # Fallback if config indicates MoE even without explicit model_type
        num_experts = None
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            value = getattr(config, attr, None)
            if value:
                num_experts = value
                break
        if num_experts and num_experts > 1:
            include_patterns.extend(_DEFAULT_MOE_INCLUDE_PATTERNS)
            skip_patterns.extend(_DEFAULT_MOE_SKIP_PATTERNS)

    return {
        "include_patterns": include_patterns,
        "skip_patterns": skip_patterns,
    }


def _require_torch(feature: str) -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError(f"PyTorch is required for {feature}. Install with: pip install torch")


def _require_mps(device: str) -> None:
    if device != "mps":
        raise RuntimeError("MetalQuantizedLinear only supports device='mps'.")
    if torch is None or not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS backend is not available on this system.")


def _normalize_format(fmt: str) -> str:
    fmt = (fmt or "fp4").lower()
    if fmt.startswith("marlin_"):
        fmt = fmt[len("marlin_") :]
    if fmt in {"fp4", "fp4_e2m1"}:
        return "fp4"
    if fmt in {"fp8", "fp8_e4m3"}:
        return "fp8"
    if fmt in {"int2"}:
        return "int2"
    return fmt


def _format_bits(fmt: str) -> int | None:
    if fmt == "fp4":
        return 4
    if fmt == "fp8":
        return 8
    if fmt == "int2":
        return 2
    return None


def _precision_to_bits_format(precision: Precision) -> tuple[int | None, str | None]:
    if precision == Precision.FP4_E2M1:
        return 4, "fp4"
    if precision == Precision.FP8_E4M3:
        return 8, "fp8"
    if precision == Precision.INT2:
        return 2, "int2"
    return None, None


def _resolve_skip_patterns(skip_patterns: list[str] | None) -> list[str]:
    patterns = skip_patterns if skip_patterns is not None else _DEFAULT_SKIP_PATTERNS
    return [p.lower() for p in patterns]


def _iter_linear_modules(model: Any) -> Iterable[tuple[str, Any, Any, str]]:
    import torch.nn as nn

    module_map = dict(model.named_modules())
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = module_map[parent_name]
        else:
            parent = model
            child_name = name
        yield name, module, parent, child_name


def _build_layer_plan(
    model: Any,
    *,
    bits: int,
    group_size: int,
    fmt: str,
    skip_patterns: list[str] | None,
    mixed_precision_config: MixedPrecisionConfig | None,
    include_names: set[str] | None = None,
) -> list[LayerPlan]:
    skip_patterns = _resolve_skip_patterns(skip_patterns)
    include_names = set(include_names) if include_names else None
    plans: list[LayerPlan] = []

    fmt = _normalize_format(fmt)
    fmt_bits = _format_bits(fmt)
    if fmt_bits is not None and fmt_bits != bits:
        raise ValueError(f"bits={bits} does not match format '{fmt}' (expected {fmt_bits}).")

    for name, module, parent, child_name in _iter_linear_modules(model):
        if include_names is not None and name not in include_names:
            continue
        name_lower = name.lower()
        if any(pat in name_lower for pat in skip_patterns):
            plans.append(
                LayerPlan(
                    name=name,
                    parent=parent,
                    child_name=child_name,
                    module=module,
                    quantize=False,
                    bits=bits,
                    group_size=group_size,
                    fmt=fmt,
                    precision=None,
                    skip_reason="skip_pattern",
                )
            )
            continue

        layer_bits = bits
        layer_group_size = group_size
        layer_fmt = fmt
        precision_name: str | None = None

        if mixed_precision_config is not None:
            layer_cfg = get_layer_config(name, mixed_precision_config)
            precision_name = layer_cfg.precision.value
            if layer_cfg.precision in (Precision.FP16, Precision.BF16):
                plans.append(
                    LayerPlan(
                        name=name,
                        parent=parent,
                        child_name=child_name,
                        module=module,
                        quantize=False,
                        bits=bits,
                        group_size=layer_group_size,
                        fmt=fmt,
                        precision=precision_name,
                        skip_reason="precision_skip",
                    )
                )
                continue

            mp_bits, mp_fmt = _precision_to_bits_format(layer_cfg.precision)
            if mp_bits is None or mp_fmt is None:
                plans.append(
                    LayerPlan(
                        name=name,
                        parent=parent,
                        child_name=child_name,
                        module=module,
                        quantize=False,
                        bits=bits,
                        group_size=layer_group_size,
                        fmt=fmt,
                        precision=precision_name,
                        skip_reason="unsupported_precision",
                    )
                )
                continue

            layer_bits = mp_bits
            layer_fmt = mp_fmt
            layer_group_size = layer_cfg.group_size or layer_group_size

        plans.append(
            LayerPlan(
                name=name,
                parent=parent,
                child_name=child_name,
                module=module,
                quantize=True,
                bits=layer_bits,
                group_size=layer_group_size,
                fmt=layer_fmt,
                precision=precision_name,
                skip_reason=None,
            )
        )

    return plans


def _collect_activation_ranges(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: str,
    max_samples: int,
    max_length: int,
    target_layers: set[str],
) -> dict[str, tuple[float, float]]:
    import torch.nn as nn

    stats: dict[str, dict[str, float]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            if isinstance(inputs, tuple) and inputs:
                x = inputs[0]
            else:
                x = inputs
            if not isinstance(x, torch.Tensor):
                return
            with torch.no_grad():
                x_flat = x.float().flatten()
                if x_flat.numel() == 0:
                    return
                cur_min = float(x_flat.min().item())
                cur_max = float(x_flat.max().item())
                if name not in stats:
                    stats[name] = {"min": cur_min, "max": cur_max}
                else:
                    stats[name]["min"] = min(stats[name]["min"], cur_min)
                    stats[name]["max"] = max(stats[name]["max"], cur_max)

        return hook

    for name, module in model.named_modules():
        if name not in target_layers:
            continue
        if not isinstance(module, nn.Linear):
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    if not texts:
        return {}

    for text in texts[:max_samples]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    return {name: (s["min"], s["max"]) for name, s in stats.items()}


def replace_linear_layers(
    model: Any,
    *,
    bits: int = 4,
    group_size: int = 128,
    fmt: str = "fp4",
    device: str = "mps",
    skip_patterns: list[str] | None = None,
    mixed_precision_config: MixedPrecisionConfig | None = None,
    calibration_ranges: dict[str, tuple[float, float]] | None = None,
    validate: bool = True,
    include_names: set[str] | None = None,
) -> dict[str, Any]:
    """Replace nn.Linear layers with MetalQuantizedLinear using FP4 packing.

    Returns stats dict with per-layer info and aggregate metrics.
    """
    _require_torch("replace_linear_layers")
    _require_mps(device)

    from .inference_metal import MetalQuantizedLinear

    plans = _build_layer_plan(
        model,
        bits=bits,
        group_size=group_size,
        fmt=fmt,
        skip_patterns=skip_patterns,
        mixed_precision_config=mixed_precision_config,
        include_names=include_names,
    )

    calibration_ranges = calibration_ranges or {}

    stats: dict[str, Any] = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "layers": {},
        "errors": [],
    }

    for plan in plans:
        layer_info: dict[str, Any] = {
            "quantized": False,
            "bits": plan.bits,
            "group_size": plan.group_size,
            "format": plan.fmt,
        }
        if plan.precision is not None:
            layer_info["precision"] = plan.precision

        linear = plan.module
        weight = linear.weight.detach()
        stats["original_bytes"] += weight.numel() * weight.element_size()
        if linear.bias is not None:
            stats["original_bytes"] += linear.bias.numel() * linear.bias.element_size()

        if not plan.quantize:
            layer_info["skip_reason"] = plan.skip_reason
            stats["layers"][plan.name] = layer_info
            stats["skipped_count"] += 1
            continue

        in_features = linear.in_features
        out_features = linear.out_features
        pack_factor = 8 if plan.bits == 4 else 4 if plan.bits == 8 else 16 if plan.bits == 2 else None
        if pack_factor is None:
            layer_info["skip_reason"] = "unsupported_bits"
            stats["layers"][plan.name] = layer_info
            stats["skipped_count"] += 1
            continue
        if in_features % plan.group_size != 0:
            layer_info["skip_reason"] = "in_features_not_divisible_by_group_size"
            stats["layers"][plan.name] = layer_info
            stats["skipped_count"] += 1
            continue
        if in_features % pack_factor != 0:
            layer_info["skip_reason"] = "in_features_not_divisible_by_pack_factor"
            stats["layers"][plan.name] = layer_info
            stats["skipped_count"] += 1
            continue

        quant_layer = MetalQuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            bits=plan.bits,
            group_size=plan.group_size,
            bias=linear.bias is not None,
        )

        weight_f32 = weight.float().cpu()
        if getattr(quant_layer, "_needs_output_slice", False):
            pad_rows = quant_layer._padded_out_features - out_features
            if pad_rows > 0:
                weight_f32 = torch.nn.functional.pad(weight_f32, (0, 0, 0, pad_rows), value=0.0)

        if plan.bits == 4:
            act_range = calibration_ranges.get(plan.name)
            activation_ranges = None
            if act_range is not None:
                activation_ranges = {
                    "input_range": act_range,
                    "percentile": _DEFAULT_PERCENTILE,
                    "smooth_factor": _DEFAULT_SMOOTH_FACTOR,
                }
            packed_np, scales_np = quantize_fp4(
                weight_f32.numpy(),
                group_size=plan.group_size,
                activation_ranges=activation_ranges,
                marlin_layout=True,
            )
            packed = torch.from_numpy(packed_np).to(dtype=torch.uint32)
            scales = torch.from_numpy(scales_np).to(dtype=torch.float16)
        elif plan.bits == 8:
            packed, scales = MetalQuantizedLinear._quantize_fp8(weight_f32, plan.group_size)
        elif plan.bits == 2:
            packed, scales = MetalQuantizedLinear._quantize_int2(weight_f32, plan.group_size)
        else:
            layer_info["skip_reason"] = "unsupported_bits"
            stats["layers"][plan.name] = layer_info
            stats["skipped_count"] += 1
            continue

        quant_layer.weight_packed.copy_(packed.to("mps"))
        quant_layer.scales.copy_(scales.to("mps"))
        if linear.bias is not None and quant_layer.bias is not None:
            quant_layer.bias.copy_(linear.bias.detach().to("mps", dtype=torch.float16))

        if plan.bits == 4:
            stats["quantized_bytes"] += packed_np.nbytes + scales_np.nbytes
        else:
            stats["quantized_bytes"] += packed.numel() * packed.element_size()
            stats["quantized_bytes"] += scales.numel() * scales.element_size()
        if linear.bias is not None:
            stats["quantized_bytes"] += linear.bias.numel() * linear.bias.element_size()

        if validate and plan.bits == 4:
            weight_for_error = weight_f32
            if getattr(quant_layer, "_needs_output_slice", False):
                weight_for_error = weight_f32
            err = compute_quantization_error(
                weight_for_error.numpy(),
                packed_np,
                scales_np,
                group_size=plan.group_size,
                marlin_layout=True,
            )
            err["name"] = plan.name
            stats["errors"].append(err)
            layer_info["error"] = {
                "rmse": err["rmse"],
                "max_error": err["max_error"],
                "mean_relative_error": err["mean_relative_error"],
            }

        setattr(plan.parent, plan.child_name, quant_layer)

        layer_info["quantized"] = True
        layer_info["packed_shape"] = list(quant_layer.weight_packed.shape)
        layer_info["scales_shape"] = list(quant_layer.scales.shape)
        stats["layers"][plan.name] = layer_info
        stats["quantized_count"] += 1

    return stats


def _collect_moe_linear_targets(
    model: Any,
    moe_layers: list[tuple[str, Any, Any, str]],
    include_patterns: list[str],
    skip_patterns: list[str],
) -> set[str]:
    prefixes = [name for name, _module, _parent, _child in moe_layers]
    include_patterns = [p.lower() for p in include_patterns if p]
    skip_patterns = [p.lower() for p in skip_patterns if p]
    require_match = bool(include_patterns)

    targets: set[str] = set()
    for name, _module, _parent, _child in _iter_linear_modules(model):
        in_moe = False
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                in_moe = True
                break
        if not in_moe:
            continue

        name_lower = name.lower()
        if require_match and not any(pat in name_lower for pat in include_patterns):
            continue
        if skip_patterns and any(pat in name_lower for pat in skip_patterns):
            continue
        targets.add(name)

    return targets


def replace_moe_layers(
    model: Any,
    *,
    bits: int = 4,
    group_size: int = 128,
    fmt: str = "fp4",
    device: str = "mps",
    skip_patterns: list[str] | None = None,
    mixed_precision_config: MixedPrecisionConfig | None = None,
    calibration_ranges: dict[str, tuple[float, float]] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Quantize MoE expert layers and wrap MoE blocks."""
    _require_torch("replace_moe_layers")
    _require_mps(device)

    moe_layers = find_moe_layers(model)
    stats: dict[str, Any] = {
        "moe_layer_count": len(moe_layers),
        "moe_layers": [name for name, _module, _parent, _child in moe_layers],
        "moe_target_count": 0,
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "layers": {},
        "errors": [],
    }

    if not moe_layers:
        return stats

    config = getattr(model, "config", None)
    target_cfg = _detect_quantization_targets(config) if config is not None else {}
    include_patterns = target_cfg.get("include_patterns", [])
    moe_skip = target_cfg.get("skip_patterns", [])

    combined_skip: list[str]
    if skip_patterns is None:
        combined_skip = list(_DEFAULT_SKIP_PATTERNS) + list(moe_skip)
    else:
        combined_skip = list(skip_patterns) + list(moe_skip)

    target_names = _collect_moe_linear_targets(
        model,
        moe_layers,
        include_patterns=include_patterns,
        skip_patterns=combined_skip,
    )
    stats["moe_target_count"] = len(target_names)

    if target_names:
        moe_stats = replace_linear_layers(
            model,
            bits=bits,
            group_size=group_size,
            fmt=fmt,
            device=device,
            skip_patterns=combined_skip,
            mixed_precision_config=mixed_precision_config,
            calibration_ranges=calibration_ranges,
            validate=validate,
            include_names=target_names,
        )
        for key in (
            "quantized_count",
            "skipped_count",
            "original_bytes",
            "quantized_bytes",
        ):
            stats[key] = moe_stats.get(key, 0)
        stats["layers"] = moe_stats.get("layers", {})
        stats["errors"] = moe_stats.get("errors", [])
    else:
        logger.warning("No MoE expert linear layers matched for quantization.")

    if torch is not None:
        for name, module, parent, child_name in moe_layers:
            if isinstance(module, MetalQuantizedMoE):
                continue
            setattr(parent, child_name, MetalQuantizedMoE(module, name=name))

    return stats


def save_quantized(model: Any, path: Path | str, config: dict[str, Any]) -> None:
    """Save quantized model weights and metadata."""
    _require_torch("save_quantized")

    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if hasattr(tensor, "detach")
    }

    save_file(state_dict, str(path / "quantized_weights.safetensors"))

    if hasattr(model, "config"):
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

    with open(path / "quantization_config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_quantized(path: Path | str, device: str = "mps") -> Any:
    """Load a previously quantized model saved by save_quantized()."""
    _require_torch("load_quantized")
    _require_mps(device)

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    path = Path(path)
    config_path = path / "config.json"
    qconfig_path = path / "quantization_config.json"
    weights_path = path / "quantized_weights.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(f"quantized_weights.safetensors not found in {path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {path}")

    config = AutoConfig.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)

    layer_info: dict[str, Any] = {}
    if qconfig_path.exists():
        with open(qconfig_path) as f:
            qconfig = json.load(f)
        layer_info = qconfig.get("layers", {})
    else:
        qconfig = {}

    state_dict = load_file(str(weights_path))
    quantized_modules = {
        key[: -len(".weight_packed")] for key in state_dict.keys() if key.endswith(".weight_packed")
    }

    from .inference_metal import MetalQuantizedLinear

    for name, module, parent, child_name in _iter_linear_modules(model):
        layer_cfg = layer_info.get(name, {})
        is_quantized = bool(layer_cfg.get("quantized", False) or name in quantized_modules)
        if not is_quantized:
            continue

        layer_bits = int(layer_cfg.get("bits", qconfig.get("bits", 4)))
        layer_group_size = int(layer_cfg.get("group_size", qconfig.get("group_size", 128)))

        quant_layer = MetalQuantizedLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bits=layer_bits,
            group_size=layer_group_size,
            bias=module.bias is not None,
        )
        setattr(parent, child_name, quant_layer)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_and_quantize(
    model_id: str,
    output_path: Path | str | None = None,
    *,
    bits: int = 4,
    group_size: int = 128,
    format: str = "fp4",
    device: str = "mps",
    torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
    calibration_data: list[str] | None = None,
    calibration_samples: int = 128,
    skip_patterns: list[str] | None = None,
    mixed_precision_config: MixedPrecisionConfig | None = None,
    validate: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Load a HuggingFace model, quantize Linear layers, optionally save.

    Returns:
        (quantized_model, stats_dict)
    """
    _require_torch("load_and_quantize")
    _require_mps(device)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers library required for load_and_quantize. "
            "Install with: pip install transformers"
        ) from e

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    target_cfg = _detect_quantization_targets(getattr(model, "config", None))
    moe_skip = target_cfg.get("skip_patterns", [])
    if skip_patterns is None:
        combined_skip_patterns = list(_DEFAULT_SKIP_PATTERNS) + list(moe_skip)
    else:
        combined_skip_patterns = list(skip_patterns) + list(moe_skip)

    calibration_ranges: dict[str, tuple[float, float]] = {}
    if calibration_data:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        plans = _build_layer_plan(
            model,
            bits=bits,
            group_size=group_size,
            fmt=format,
            skip_patterns=combined_skip_patterns,
            mixed_precision_config=mixed_precision_config,
        )
        target_layers = {plan.name for plan in plans if plan.quantize}
        max_length = getattr(model.config, "max_position_embeddings", 2048)
        calibration_ranges = _collect_activation_ranges(
            model,
            tokenizer,
            calibration_data,
            device=device,
            max_samples=calibration_samples,
            max_length=max_length,
            target_layers=target_layers,
        )

    moe_stats: dict[str, Any] | None = None
    moe_layers = find_moe_layers(model)
    if moe_layers:
        logger.info(f"Found {len(moe_layers)} MoE layers")
        moe_stats = replace_moe_layers(
            model,
            bits=bits,
            group_size=group_size,
            fmt=format,
            device=device,
            skip_patterns=combined_skip_patterns,
            mixed_precision_config=mixed_precision_config,
            calibration_ranges=calibration_ranges,
            validate=validate,
        )

    stats = replace_linear_layers(
        model,
        bits=bits,
        group_size=group_size,
        fmt=format,
        device=device,
        skip_patterns=combined_skip_patterns,
        mixed_precision_config=mixed_precision_config,
        calibration_ranges=calibration_ranges,
        validate=validate,
    )

    if moe_stats:
        stats["moe_layer_count"] = moe_stats.get("moe_layer_count", 0)
        stats["moe_layers"] = moe_stats.get("moe_layers", [])
        stats["moe_target_count"] = moe_stats.get("moe_target_count", 0)
        stats["moe_quantized_count"] = moe_stats.get("quantized_count", 0)
        stats["moe_skipped_count"] = moe_stats.get("skipped_count", 0)
        stats["quantized_count"] += moe_stats.get("quantized_count", 0)
        stats["skipped_count"] += moe_stats.get("skipped_count", 0)
        stats["original_bytes"] += moe_stats.get("original_bytes", 0)
        stats["quantized_bytes"] += moe_stats.get("quantized_bytes", 0)
        stats["layers"].update(moe_stats.get("layers", {}))
        stats["errors"].extend(moe_stats.get("errors", []))

    if stats["quantized_bytes"] > 0:
        stats["compression_ratio"] = stats["original_bytes"] / stats["quantized_bytes"]
    else:
        stats["compression_ratio"] = 1.0

    stats["format"] = _normalize_format(format)
    stats["bits"] = bits
    stats["group_size"] = group_size
    stats["device"] = device
    stats["torch_dtype"] = str(torch_dtype)
    stats["skip_patterns"] = _resolve_skip_patterns(combined_skip_patterns)

    if calibration_data:
        stats["calibration"] = {
            "samples": min(len(calibration_data), calibration_samples),
            "percentile": _DEFAULT_PERCENTILE,
            "smooth_factor": _DEFAULT_SMOOTH_FACTOR,
        }

    if stats["errors"]:
        stats["mean_rmse"] = float(
            sum(e["rmse"] for e in stats["errors"]) / len(stats["errors"])
        )
        stats["max_error"] = float(max(e["max_error"] for e in stats["errors"]))

    if output_path is not None:
        config = {
            "format": stats["format"],
            "bits": stats["bits"],
            "group_size": stats["group_size"],
            "device": stats["device"],
            "torch_dtype": stats["torch_dtype"],
            "skip_patterns": stats["skip_patterns"],
            "quantized_count": stats["quantized_count"],
            "skipped_count": stats["skipped_count"],
            "compression_ratio": stats["compression_ratio"],
            "layers": stats["layers"],
        }
        if "calibration" in stats:
            config["calibration"] = stats["calibration"]
        if "mean_rmse" in stats:
            config["mean_rmse"] = stats["mean_rmse"]
        if "max_error" in stats:
            config["max_error"] = stats["max_error"]
        save_quantized(model, output_path, config)

    return model, stats
