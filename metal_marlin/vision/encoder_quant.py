"""Vision encoder quantization for VLMs (Vision-Language Models).

This module focuses on vision-specific quantization challenges:
1. Vision encoders are more sensitive than LLMs (patch embedding + early layers).
2. Patch embeddings have different statistics from text embeddings.
3. Cross-attention layers bridge vision/text and require higher precision.

Core features:
- Vision layer sensitivity analysis (weights + activations + Hessians)
- Image-based calibration (COCO / ImageNet subsets)
- Mixed precision plans (FP8 vision, FP4 LLM)

Usage:
    from metal_marlin.vision import encoder_quant

    # 1) Build calibration dataset (image-based)
    calib = encoder_quant.build_vision_calibration_dataset(
        source="coco",
        num_images=512,
        image_size=(224, 224),
    )

    # 2) Collect Hessians/activations on vision encoder
    activations, hessians = encoder_quant.collect_vision_calibration_stats(
        model=vlm,
        calibration_data=calib,
        config=vision_config,
    )

    # 3) Analyze vision sensitivity
    report = encoder_quant.analyze_vision_encoder_sensitivity(
        weights=vision_weights,
        activations=activations,
        hessians=hessians,
    )

    # 4) Build mixed precision plan (FP8 vision, FP4 LLM)
    vision_map = encoder_quant.build_vision_precision_map(
        weights=vision_weights,
        config=vision_config,
        report=report,
    )
    precision_map = encoder_quant.build_vlm_precision_map(
        weights=vlm_weights,
        vision_config=vision_config,
        vision_precision_map=vision_map,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import HAS_TORCH, to_numpy, torch
from ..calibration.hooks import CalibrationHooks
from ..mixed_precision import Precision
from ..packing.mixed_format import pack_mixed_format_model
from .encoder_config import VisionEncoderConfig
from .image_calibration import VisionCalibrationDataset
from .layer_sensitivity import SensitivityReport, VisionLayerType, analyze_vision_layer_sensitivity

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class VisionCalibrationConfig:
    """Configuration for vision encoder calibration passes."""

    dataset: str = "coco"
    num_images: int = 512
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 8
    max_batches: int | None = 32
    seed: int = 42
    damping: float = 0.01
    collect_hessians: bool = True
    collect_activations: bool = True


def build_vision_calibration_dataset(
    source: str = "coco",
    num_images: int = 512,
    image_size: tuple[int, int] = (224, 224),
    seed: int = 42,
) -> VisionCalibrationDataset:
    """Build a vision calibration dataset (COCO or ImageNet subset).

    Args:
        source: "coco" or "imagenet".
        num_images: Number of images to sample.
        image_size: Target image size (H, W).
        seed: Random seed for sampling.

    Returns:
        VisionCalibrationDataset.
    """
    source_lower = source.lower()
    if source_lower in {"coco", "coco_subset"}:
        return VisionCalibrationDataset.coco_subset(
            num_images=num_images,
            image_size=image_size,
            seed=seed,
        )
    if source_lower in {"imagenet", "imagenet_subset"}:
        return VisionCalibrationDataset.imagenet_subset(
            num_images=num_images,
            image_size=image_size,
            seed=seed,
        )
    raise ValueError(f"Unsupported calibration dataset source: {source}")


class _ActivationCollector:
    """Collects limited activation samples for layer sensitivity analysis."""

    def __init__(self, max_samples: int = 4096) -> None:
        self.max_samples = max_samples
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._counts: dict[str, int] = {}

    def make_hook(self, name: str) -> Callable:
        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs:
                return
            x = inputs[0]
            x_np = to_numpy(x).astype(np.float32)
            if x_np.ndim > 2:
                x_np = x_np.reshape(-1, x_np.shape[-1])
            elif x_np.ndim == 1:
                x_np = x_np.reshape(1, -1)

            remaining = self.max_samples - self._counts.get(name, 0)
            if remaining <= 0:
                return
            if x_np.shape[0] > remaining:
                x_np = x_np[:remaining]

            self._buffers.setdefault(name, []).append(x_np)
            self._counts[name] = self._counts.get(name, 0) + x_np.shape[0]

        return hook

    def get(self) -> dict[str, np.ndarray]:
        return {
            name: np.concatenate(chunks, axis=0) for name, chunks in self._buffers.items() if chunks
        }


def collect_vision_calibration_stats(
    model: Any,
    calibration_data: VisionCalibrationDataset,
    config: VisionEncoderConfig,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    batch_size: int = 8,
    max_batches: int | None = None,
    collect_hessians: bool = True,
    collect_activations: bool = True,
    damping: float = 0.01,
) -> tuple[dict[str, NDArray[np.floating]], dict[str, NDArray[np.floating]]]:
    """Collect activation samples and Hessians for vision encoder layers.

    Args:
        model: VLM or vision encoder model.
        calibration_data: Image calibration dataset.
        config: Vision encoder config (used to filter layers).
        forward_fn: Optional forward function (model, batch) -> outputs.
        max_batches: Max batches to process (None = all).
        collect_hessians: Whether to compute Hessians from inputs.
        collect_activations: Whether to capture activations for sensitivity.
        damping: Hessian damping factor.

    Returns:
        (activations, hessians) dicts keyed by weight names.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for calibration.")

    layer_filter = _make_vision_layer_filter(config)
    hooks = CalibrationHooks(damping=damping) if collect_hessians else None
    activation_collector = _ActivationCollector() if collect_activations else None

    if collect_hessians and hooks is not None:
        hooks.register_linear_hooks(model, layer_filter=layer_filter)

    activation_handles: list[Any] = []
    if collect_activations and activation_collector is not None:
        activation_handles = _register_activation_hooks(model, activation_collector, layer_filter)

    if forward_fn is None:
        forward_fn = _default_forward_fn

    batch_iter = calibration_data.get_batches(batch_size=batch_size)

    if torch is None:
        raise RuntimeError("PyTorch required for calibration.")

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batch_iter):
            if max_batches is not None and idx >= max_batches:
                break
            batch_input = _convert_batch_for_model(model, batch)
            _ = forward_fn(model, batch_input)

    activations = activation_collector.get() if activation_collector else {}
    hessians = hooks.get_hessians() if hooks else {}

    if hooks is not None:
        hooks.remove_hooks()
    for handle in activation_handles:
        handle.remove()

    return _map_stats_to_weight_names(activations), _map_stats_to_weight_names(hessians)


def analyze_vision_encoder_sensitivity(
    weights: dict[str, NDArray[np.floating]],
    activations: dict[str, NDArray[np.floating]] | None = None,
    hessians: dict[str, NDArray[np.floating]] | None = None,
    fp4_threshold: float = 0.02,
    fp8_threshold: float = 0.005,
) -> SensitivityReport:
    """Run sensitivity analysis for vision encoder layers."""
    return analyze_vision_layer_sensitivity(
        weights=weights,
        activations=activations,
        hessians=hessians,
        fp4_threshold=fp4_threshold,
        fp8_threshold=fp8_threshold,
    )


def build_vision_precision_map(
    weights: dict[str, NDArray[np.floating]],
    config: VisionEncoderConfig,
    report: SensitivityReport | None = None,
) -> dict[str, tuple[Precision, int]]:
    """Build per-layer precision map for vision encoder weights."""
    precision_map: dict[str, tuple[Precision, int]] = {}
    report_map: dict[str, Any] = {}

    if report is not None:
        report_map = {layer.name: layer for layer in report.layers}

    for name, weight in weights.items():
        if weight.ndim < 2:
            continue

        layer = report_map.get(name)
        if layer is not None:
            precision = _precision_from_string(layer.recommended_precision)
            group_size = layer.recommended_group_size
            if layer.layer_type in (
                VisionLayerType.PATCH_EMBED,
                VisionLayerType.POSITION_EMBED,
                VisionLayerType.LAYER_NORM,
                VisionLayerType.CLS_TOKEN,
                VisionLayerType.CROSS_ATTENTION,
            ):
                precision = Precision.BF16
                group_size = 0
        else:
            prec_str, group_size = config.get_layer_precision(name)
            precision = _precision_from_string(prec_str)

        precision_map[name] = (precision, group_size)

    return precision_map


def build_vlm_precision_map(
    weights: dict[str, NDArray[np.floating]],
    vision_config: VisionEncoderConfig,
    vision_precision_map: dict[str, tuple[Precision, int]] | None = None,
    llm_precision: Precision = Precision.FP4_E2M1,
    llm_group_size: int = 128,
    keep_llm_sensitive: bool = True,
) -> dict[str, tuple[Precision, int]]:
    """Build mixed-precision map for full VLM weights.

    Vision encoder weights use FP8/BF16 (from vision_precision_map),
    while the LLM defaults to FP4 (optionally preserving embeddings/norms).
    """
    precision_map: dict[str, tuple[Precision, int]] = {}
    vision_precision_map = vision_precision_map or {}

    for name, weight in weights.items():
        if weight.ndim < 2:
            continue

        if name in vision_precision_map or _is_vision_weight_name(name, vision_config):
            if name in vision_precision_map:
                precision_map[name] = vision_precision_map[name]
            else:
                prec_str, group_size = vision_config.get_layer_precision(name)
                precision_map[name] = (_precision_from_string(prec_str), group_size)
            continue

        if keep_llm_sensitive and _is_llm_sensitive_weight(name):
            precision_map[name] = (Precision.BF16, 0)
            continue

        precision_map[name] = (llm_precision, llm_group_size)

    return precision_map


def pack_vlm_mixed_format(
    weights: dict[str, NDArray[np.floating]],
    precision_map: dict[str, tuple[Precision, int]],
    output_path: str,
) -> Any:
    """Pack a mixed-format VLM using the provided precision map."""
    return pack_mixed_format_model(weights, precision_map, output_path)


def _precision_from_string(precision: str) -> Precision:
    mapping = {
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
        "fp8": Precision.FP8_E4M3,
        "fp4": Precision.FP4_E2M1,
        "int8": Precision.INT8,
        "int4": Precision.INT4,
        "int3": Precision.INT3,
        "int2": Precision.INT2,
        "nf3": Precision.NF3,
        "nf2": Precision.NF2,
    }
    return mapping.get(precision, Precision.FP16)


def _make_vision_layer_filter(
    config: VisionEncoderConfig,
) -> Callable[[str, Any], bool]:
    prefix = config.layer_prefix

    def _filter(name: str, _module: Any) -> bool:
        if prefix and name.startswith(prefix):
            return True
        return "vision" in name.lower() or "visual" in name.lower()

    return _filter


def _register_activation_hooks(
    model: Any,
    collector: _ActivationCollector,
    layer_filter: Callable[[str, Any], bool],
) -> list[Any]:
    if not HAS_TORCH or torch is None:
        return []

    import torch.nn as tnn

    handles: list[Any] = []
    for name, module in model.named_modules():
        if not isinstance(module, (tnn.Linear, tnn.Conv2d)):
            continue
        if not layer_filter(name, module):
            continue
        handle = module.register_forward_hook(collector.make_hook(name))
        handles.append(handle)
    return handles


def _default_forward_fn(model: Any, batch: Any) -> Any:
    candidates = [
        "forward_vision",
        "encode_images",
        "get_vision_features",
        "vision_tower",
        "vision_model",
        "vision_encoder",
        "visual",
    ]

    for attr in candidates:
        if not hasattr(model, attr):
            continue
        obj = getattr(model, attr)
        if callable(obj):
            return obj(batch)
        try:
            return obj(batch)
        except Exception:
            continue

    raise ValueError("Unable to infer vision forward method. Provide forward_fn explicitly.")


def _convert_batch_for_model(model: Any, batch: np.ndarray) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch required for calibration.")
    device = _get_torch_device(model)
    return torch.from_numpy(batch).to(device)


def _get_torch_device(model: Any) -> Any:
    if not HAS_TORCH or torch is None:
        return None
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _map_stats_to_weight_names(
    stats: dict[str, NDArray[np.floating]],
) -> dict[str, NDArray[np.floating]]:
    mapped: dict[str, NDArray[np.floating]] = {}
    for name, value in stats.items():
        if name.endswith(".weight"):
            mapped[name] = value
        else:
            mapped[f"{name}.weight"] = value
    return mapped


def _is_vision_weight_name(name: str, config: VisionEncoderConfig) -> bool:
    if config.layer_prefix and name.startswith(config.layer_prefix):
        return True
    return "vision" in name.lower() or "visual" in name.lower()


def _is_llm_sensitive_weight(name: str) -> bool:
    name_lower = name.lower()
    if "embed" in name_lower:
        return True
    if "norm" in name_lower or "layernorm" in name_lower:
        return True
    if "lm_head" in name_lower or "output" in name_lower:
        return True
    return False
