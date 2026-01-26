"""Vision layer sensitivity analysis for quantization.

Vision encoders have different sensitivity characteristics than LLMs:
1. Patch embeddings are critical - they convert raw pixels to features
2. Early layers capture low-level features (edges, textures) - more sensitive
3. Later layers capture semantic features - more robust to quantization
4. Position encodings (especially 2D) need careful handling

This module provides tools for analyzing per-layer sensitivity to
quantization error, enabling optimal precision allocation.

Methodology:
1. Run calibration images through the vision encoder
2. For each layer, measure reconstruction error at different precisions
3. Rank layers by their sensitivity (error amplification factor)
4. Recommend precision based on sensitivity and memory budget

Usage:
    from metal_marlin.vision import (
        analyze_vision_layer_sensitivity,
        VisionLayerSensitivity,
    )

    # Analyze sensitivity
    sensitivity = analyze_vision_layer_sensitivity(
        model=vision_encoder,
        calibration_data=calib_dataset,
        config=encoder_config,
    )

    # Get recommendations
    for layer in sensitivity.layers:
        print(f"{layer.name}: {layer.sensitivity:.3f} -> {layer.recommended_precision}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VisionLayerType(str, Enum):
    """Types of layers in vision encoders."""

    PATCH_EMBED = "patch_embed"  # Conv2D or Linear patch embedding
    POSITION_EMBED = "position_embed"  # Position embeddings (learned, rope, etc)
    CLS_TOKEN = "cls_token"  # CLS token if present
    ATTENTION_QKV = "attention_qkv"  # Q/K/V projections
    ATTENTION_OUT = "attention_out"  # Output projection
    MLP_FC1 = "mlp_fc1"  # First MLP layer
    MLP_FC2 = "mlp_fc2"  # Second MLP layer
    MLP_GATE = "mlp_gate"  # Gated MLP (if present)
    LAYER_NORM = "layer_norm"  # Layer normalization
    CROSS_ATTENTION = "cross_attention"  # Cross-attention (vision-language)
    OUTPUT_PROJ = "output_proj"  # Final output projection
    OTHER = "other"


@dataclass
class VisionLayerInfo:
    """Information about a single vision encoder layer.

    Attributes:
        name: Full layer name from model state dict.
        layer_type: Classified layer type.
        layer_idx: Layer index within encoder (0-indexed).
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        num_params: Number of parameters.
        shape: Weight tensor shape.
    """

    name: str
    layer_type: VisionLayerType
    layer_idx: int = -1
    in_features: int = 0
    out_features: int = 0
    num_params: int = 0
    shape: tuple[int, ...] = ()

    @classmethod
    def from_weight(cls, name: str, weight: NDArray[np.floating]) -> VisionLayerInfo:
        """Create LayerInfo from weight tensor.

        Args:
            name: Layer name.
            weight: Weight tensor.

        Returns:
            VisionLayerInfo with detected type and dimensions.
        """
        layer_type = cls._classify_layer(name)
        shape = weight.shape

        # Determine in/out features based on layer type and shape
        if len(shape) == 2:
            out_features, in_features = shape
        elif len(shape) == 4:  # Conv2D: [out, in, kH, kW]
            out_features, in_features = shape[0], shape[1] * shape[2] * shape[3]
        else:
            in_features = out_features = shape[-1] if shape else 0

        return cls(
            name=name,
            layer_type=layer_type,
            layer_idx=cls._extract_layer_idx(name),
            in_features=in_features,
            out_features=out_features,
            num_params=weight.size,
            shape=tuple(shape),
        )

    @staticmethod
    def _classify_layer(name: str) -> VisionLayerType:
        """Classify layer type from name."""
        name_lower = name.lower()

        # Position embeddings
        if "position" in name_lower or "pos_embed" in name_lower or "rope" in name_lower:
            return VisionLayerType.POSITION_EMBED

        # Patch embedding
        if "patch" in name_lower and "embed" in name_lower:
            return VisionLayerType.PATCH_EMBED

        # CLS token
        if "cls" in name_lower and "token" in name_lower:
            return VisionLayerType.CLS_TOKEN

        # Layer norm
        if "norm" in name_lower or "layernorm" in name_lower:
            return VisionLayerType.LAYER_NORM

        # Cross attention (vision-language)
        if "cross" in name_lower:
            return VisionLayerType.CROSS_ATTENTION

        # Attention layers
        if any(
            p in name_lower for p in ["q_proj", "k_proj", "v_proj", "qkv", "query", "key", "value"]
        ):
            return VisionLayerType.ATTENTION_QKV
        if "o_proj" in name_lower or "out_proj" in name_lower:
            return VisionLayerType.ATTENTION_OUT

        # MLP layers
        if any(p in name_lower for p in ["gate_proj", "gate", "w1"]):
            return VisionLayerType.MLP_GATE
        if any(p in name_lower for p in ["fc1", "up_proj", "c_fc", "w3"]):
            return VisionLayerType.MLP_FC1
        if any(p in name_lower for p in ["fc2", "down_proj", "c_proj", "w2"]):
            return VisionLayerType.MLP_FC2

        # Generic MLP
        if "mlp" in name_lower or "fc" in name_lower:
            return VisionLayerType.MLP_FC1

        return VisionLayerType.OTHER

    @staticmethod
    def _extract_layer_idx(name: str) -> int:
        """Extract layer index from name like 'layers.5.mlp'."""
        import re

        match = re.search(r"layers?\.(\d+)", name.lower())
        if match:
            return int(match.group(1))

        match = re.search(r"block\.(\d+)", name.lower())
        if match:
            return int(match.group(1))

        return -1


@dataclass
class VisionLayerSensitivity:
    """Sensitivity analysis result for a vision encoder layer.

    Attributes:
        layer_info: Layer information.
        sensitivity_score: Overall sensitivity (higher = more sensitive).
        fp4_error: Relative error at FP4 precision.
        fp8_error: Relative error at FP8 precision.
        int4_error: Relative error at INT4 precision.
        recommended_precision: Recommended quantization precision.
        recommended_group_size: Recommended group size.
        error_amplification: How much error is amplified through the layer.
        activation_range: (min, max) of activation values.
        outlier_ratio: Fraction of values > 3 std from mean.
    """

    layer_info: VisionLayerInfo
    sensitivity_score: float = 0.0
    fp4_error: float = 0.0
    fp8_error: float = 0.0
    int4_error: float = 0.0
    recommended_precision: str = "fp4"
    recommended_group_size: int = 128
    error_amplification: float = 1.0
    activation_range: tuple[float, float] = (0.0, 0.0)
    outlier_ratio: float = 0.0

    @property
    def name(self) -> str:
        return self.layer_info.name

    @property
    def layer_type(self) -> VisionLayerType:
        return self.layer_info.layer_type


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report for a vision encoder.

    Attributes:
        layers: Per-layer sensitivity results.
        total_params: Total model parameters.
        quantizable_params: Parameters eligible for quantization.
        recommended_avg_bits: Recommended average bits per weight.
        memory_fp16_mb: Memory at FP16 precision.
        memory_quantized_mb: Memory with recommended quantization.
        metadata: Additional analysis metadata.
    """

    layers: list[VisionLayerSensitivity] = field(default_factory=list)
    total_params: int = 0
    quantizable_params: int = 0
    recommended_avg_bits: float = 0.0
    memory_fp16_mb: float = 0.0
    memory_quantized_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def get_by_type(self, layer_type: VisionLayerType) -> list[VisionLayerSensitivity]:
        """Get all layers of a specific type."""
        return [l for l in self.layers if l.layer_type == layer_type]

    def get_precision_summary(self) -> dict[str, int]:
        """Get count of layers by recommended precision."""
        summary: dict[str, int] = {}
        for layer in self.layers:
            prec = layer.recommended_precision
            summary[prec] = summary.get(prec, 0) + 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_params": self.total_params,
            "quantizable_params": self.quantizable_params,
            "recommended_avg_bits": self.recommended_avg_bits,
            "memory_fp16_mb": self.memory_fp16_mb,
            "memory_quantized_mb": self.memory_quantized_mb,
            "precision_summary": self.get_precision_summary(),
            "layers": [
                {
                    "name": l.name,
                    "type": l.layer_type.value,
                    "sensitivity": l.sensitivity_score,
                    "fp4_error": l.fp4_error,
                    "fp8_error": l.fp8_error,
                    "recommended_precision": l.recommended_precision,
                    "recommended_group_size": l.recommended_group_size,
                }
                for l in self.layers
            ],
            "metadata": self.metadata,
        }


def analyze_vision_layer_sensitivity(
    weights: dict[str, NDArray[np.floating]],
    activations: dict[str, NDArray[np.floating]] | None = None,
    hessians: dict[str, NDArray[np.floating]] | None = None,
    fp4_threshold: float = 0.02,  # Max acceptable error for FP4
    fp8_threshold: float = 0.005,  # Max acceptable error for FP8
) -> SensitivityReport:
    """Analyze quantization sensitivity for vision encoder layers.

    Uses multiple metrics to determine sensitivity:
    1. Weight distribution analysis (outliers, range, kurtosis)
    2. Simulated quantization error at different precisions
    3. Hessian-based importance (if available)
    4. Activation statistics (if available)

    Args:
        weights: Dict of layer_name -> weight tensor.
        activations: Optional dict of layer_name -> activation samples.
        hessians: Optional dict of layer_name -> Hessian matrix.
        fp4_threshold: Error threshold for FP4 recommendation.
        fp8_threshold: Error threshold for FP8 recommendation.

    Returns:
        SensitivityReport with per-layer analysis.
    """
    layers: list[VisionLayerSensitivity] = []
    total_params = 0
    quantizable_params = 0

    for name, weight in weights.items():
        # Skip non-weight tensors
        if weight.ndim < 2 and weight.size < 100:
            continue

        # Create layer info
        layer_info = VisionLayerInfo.from_weight(name, weight)
        total_params += layer_info.num_params

        # Analyze weight distribution
        w = weight.flatten().astype(np.float32)

        # Compute statistics
        w_mean = np.mean(w)
        w_std = np.std(w)
        _w_min, _w_max = np.min(w), np.max(w)  # Reserved for future use

        # Outlier ratio (values > 3 std)
        outlier_mask = np.abs(w - w_mean) > 3 * w_std
        outlier_ratio = np.mean(outlier_mask)

        # Compute kurtosis (measures tail heaviness)
        kurtosis = np.mean(((w - w_mean) / (w_std + 1e-8)) ** 4) - 3

        # Simulate quantization errors
        fp4_error = _simulate_fp4_error(weight)
        fp8_error = _simulate_fp8_error(weight)
        int4_error = _simulate_int4_error(weight)

        # Use Hessian if available for weighted error
        if hessians and name in hessians:
            H = hessians[name]
            # Weight error by Hessian importance
            diag_importance = np.diag(H).mean()
            fp4_error *= 1 + diag_importance
            fp8_error *= 1 + diag_importance
            int4_error *= 1 + diag_importance

        # Use activation statistics if available
        activation_range = (0.0, 0.0)
        error_amplification = 1.0
        if activations and name in activations:
            act = activations[name]
            activation_range = (float(np.min(act)), float(np.max(act)))
            # Estimate error amplification from activation magnitude
            act_std = np.std(act)
            error_amplification = max(1.0, act_std / (w_std + 1e-8))

        # Compute overall sensitivity score
        # Higher score = more sensitive = needs higher precision
        sensitivity_score = (
            fp4_error * 0.4
            + outlier_ratio * 10.0
            + max(0, kurtosis) * 0.1
            + error_amplification * 0.2
        )

        # Adjust sensitivity by layer type
        sensitivity_score = _adjust_sensitivity_by_type(sensitivity_score, layer_info.layer_type)

        # Determine recommended precision
        if layer_info.layer_type in (
            VisionLayerType.PATCH_EMBED,
            VisionLayerType.POSITION_EMBED,
            VisionLayerType.CLS_TOKEN,
            VisionLayerType.LAYER_NORM,
        ):
            # Critical layers - keep high precision
            recommended_precision = "bf16"
            recommended_group_size = 0
        elif fp8_error > fp8_threshold or sensitivity_score > 0.5:
            # Sensitive layers - use FP8
            recommended_precision = "fp8"
            recommended_group_size = 64
            quantizable_params += layer_info.num_params
        elif fp4_error > fp4_threshold or sensitivity_score > 0.3:
            # Moderate sensitivity - FP4 with small groups
            recommended_precision = "fp4"
            recommended_group_size = 64
            quantizable_params += layer_info.num_params
        else:
            # Robust layers - FP4 with large groups
            recommended_precision = "fp4"
            recommended_group_size = 128
            quantizable_params += layer_info.num_params

        layers.append(
            VisionLayerSensitivity(
                layer_info=layer_info,
                sensitivity_score=sensitivity_score,
                fp4_error=fp4_error,
                fp8_error=fp8_error,
                int4_error=int4_error,
                recommended_precision=recommended_precision,
                recommended_group_size=recommended_group_size,
                error_amplification=error_amplification,
                activation_range=activation_range,
                outlier_ratio=outlier_ratio,
            )
        )

    # Sort by layer index and name
    layers.sort(key=lambda x: (x.layer_info.layer_idx, x.name))

    # Calculate memory estimates
    memory_fp16_mb = total_params * 2 / (1024 * 1024)  # 2 bytes per param
    memory_quantized_mb = _estimate_quantized_memory(layers)

    # Calculate average bits
    total_bits = sum(
        l.layer_info.num_params * _precision_to_bits(l.recommended_precision) for l in layers
    )
    avg_bits = total_bits / total_params if total_params > 0 else 16

    return SensitivityReport(
        layers=layers,
        total_params=total_params,
        quantizable_params=quantizable_params,
        recommended_avg_bits=avg_bits,
        memory_fp16_mb=memory_fp16_mb,
        memory_quantized_mb=memory_quantized_mb,
        metadata={
            "fp4_threshold": fp4_threshold,
            "fp8_threshold": fp8_threshold,
            "used_hessians": hessians is not None,
            "used_activations": activations is not None,
        },
    )


def _simulate_fp4_error(weight: NDArray[np.floating], group_size: int = 128) -> float:
    """Simulate FP4 quantization error."""
    w = weight.flatten().astype(np.float32)

    # FP4 E2M1 codebook (positive values, symmetric)
    codebook = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    full_codebook = np.concatenate([-codebook[::-1], codebook])

    # Per-group quantization
    if len(w) < group_size:
        groups = [w]
    else:
        pad_len = (group_size - len(w) % group_size) % group_size
        w_padded = np.concatenate([w, np.zeros(pad_len, dtype=np.float32)])
        groups = w_padded.reshape(-1, group_size)

    total_error = 0.0
    total_weight = 0.0

    for group in groups:
        scale = np.max(np.abs(group)) / 6.0 + 1e-8
        normalized = group / scale
        quantized = full_codebook[np.argmin(np.abs(normalized[:, None] - full_codebook), axis=1)]
        dequantized = quantized * scale

        error = np.mean((group - dequantized) ** 2)
        total_error += error * len(group)
        total_weight += np.mean(group**2) * len(group)

    return np.sqrt(total_error / (total_weight + 1e-8))


def _simulate_fp8_error(weight: NDArray[np.floating]) -> float:
    """Simulate FP8 E4M3 quantization error."""
    w = weight.flatten().astype(np.float32)

    # FP8 E4M3 range: ±448 with 3 mantissa bits
    scale = np.max(np.abs(w)) / 448.0 + 1e-8
    normalized = w / scale

    # Simulate 8-level mantissa quantization (2^3 = 8 values per exponent)
    quantized = np.clip(np.round(normalized * 127) / 127 * 448, -448, 448)
    dequantized = quantized * scale

    mse = np.mean((w - dequantized) ** 2)
    variance = np.mean(w**2)

    return np.sqrt(mse / (variance + 1e-8))


def _simulate_int4_error(weight: NDArray[np.floating], group_size: int = 128) -> float:
    """Simulate INT4 symmetric quantization error."""
    w = weight.flatten().astype(np.float32)

    if len(w) < group_size:
        groups = [w]
    else:
        pad_len = (group_size - len(w) % group_size) % group_size
        w_padded = np.concatenate([w, np.zeros(pad_len, dtype=np.float32)])
        groups = w_padded.reshape(-1, group_size)

    total_error = 0.0
    total_weight = 0.0

    for group in groups:
        scale = np.max(np.abs(group)) / 7.0 + 1e-8
        quantized = np.clip(np.round(group / scale), -8, 7)
        dequantized = quantized * scale

        error = np.mean((group - dequantized) ** 2)
        total_error += error * len(group)
        total_weight += np.mean(group**2) * len(group)

    return np.sqrt(total_error / (total_weight + 1e-8))


def _adjust_sensitivity_by_type(score: float, layer_type: VisionLayerType) -> float:
    """Adjust sensitivity score based on layer type."""
    multipliers = {
        VisionLayerType.PATCH_EMBED: 2.0,  # Critical
        VisionLayerType.POSITION_EMBED: 2.0,  # Critical
        VisionLayerType.CLS_TOKEN: 1.5,
        VisionLayerType.ATTENTION_QKV: 1.3,  # Position-sensitive
        VisionLayerType.ATTENTION_OUT: 1.1,
        VisionLayerType.CROSS_ATTENTION: 1.5,  # Vision-language bridge
        VisionLayerType.MLP_FC1: 1.0,
        VisionLayerType.MLP_FC2: 1.0,
        VisionLayerType.MLP_GATE: 1.0,
        VisionLayerType.LAYER_NORM: 2.0,  # Always keep high
        VisionLayerType.OUTPUT_PROJ: 1.2,
        VisionLayerType.OTHER: 1.0,
    }
    return score * multipliers.get(layer_type, 1.0)


def _precision_to_bits(precision: str) -> int:
    """Convert precision string to bits."""
    return {
        "bf16": 16,
        "fp16": 16,
        "fp8": 8,
        "fp4": 4,
        "int8": 8,
        "int4": 4,
        "int3": 3,
        "int2": 2,
    }.get(precision, 16)


def _estimate_quantized_memory(layers: list[VisionLayerSensitivity]) -> float:
    """Estimate memory usage with recommended quantization."""
    total_bytes = 0

    for layer in layers:
        bits = _precision_to_bits(layer.recommended_precision)
        params = layer.layer_info.num_params

        # Weight memory
        weight_bytes = params * bits / 8

        # Scale memory (for quantized layers)
        if layer.recommended_group_size > 0:
            num_scales = params / layer.recommended_group_size
            scale_bytes = num_scales * 2  # FP16 scales
            weight_bytes += scale_bytes

        total_bytes += weight_bytes

    return total_bytes / (1024 * 1024)


def save_sensitivity_report(report: SensitivityReport, path: str | Path) -> None:
    """Save sensitivity report to JSON file."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


def load_sensitivity_report(path: str | Path) -> dict[str, Any]:
    """Load sensitivity report from JSON file.

    Returns raw dict since VisionLayerInfo needs weight tensors.
    """
    import json

    with open(path) as f:
        return json.load(f)
