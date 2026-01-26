"""Per-layer sensitivity analysis for quantization format selection.

This module implements sensitivity analysis inspired by ExllamaV3 to determine
optimal quantization settings per layer. Different layers have vastly different
sensitivity to quantization error, and using the wrong precision can cause
catastrophic quality degradation.

Sensitivity metrics:
- Hessian trace: tr(H) measures activation magnitude (higher = more important)
- Hessian condition number: max(eig)/min(eig) measures numerical sensitivity
- Weight variance: σ²(W) measures weight distribution spread
- Outlier ratio: fraction of weights with |W| > 3σ (outlier prevalence)

Decision rules based on empirical observations:
1. Router layers (MoE): Always FP16 - critical for expert selection
2. High condition number (>1000): FP8 or FP16 - numerically sensitive
3. High outlier ratio (>0.01): Apply Hadamard first, then re-evaluate
4. Low variance layers (<0.001): Can use aggressive FP4

Reference:
    ExllamaV3 sensitivity analysis: github.com/turboderp/exllamav3

Example:
    >>> from metal_marlin.calibration.sensitivity import (
    ...     analyze_layer_sensitivity,
    ...     compute_model_sensitivity_profile,
    ... )
    >>> from metal_marlin.calibration import BartowskiCalibration
    >>>
    >>> # Analyze a single layer
    >>> weight = np.random.randn(4096, 4096).astype(np.float32)
    >>> hessian = np.eye(4096, dtype=np.float32)  # Placeholder
    >>> sensitivity = analyze_layer_sensitivity(weight, hessian)
    >>> print(f"Recommended: {sensitivity.recommended_bits}-bit {sensitivity.recommended_format}")
    >>>
    >>> # Full model analysis
    >>> calibration = BartowskiCalibration.v3()
    >>> profile = compute_model_sensitivity_profile("path/to/model", calibration)
    >>> for name, sens in profile.items():
    ...     print(f"{name}: {sens.recommended_format}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .bartowski import BartowskiCalibration


@dataclass
class LayerSensitivity:
    """Sensitivity analysis result for a single layer.

    Attributes:
        name: Layer name (e.g., "model.layers.0.self_attn.q_proj").
        hessian_trace: tr(H) - sum of Hessian diagonal, measures activation magnitude.
            Higher values indicate more important layers that affect output strongly.
        hessian_condition: max(eig)/min(eig) - Hessian condition number.
            High values (>1000) indicate numerical sensitivity where small
            perturbations cause large output changes.
        weight_variance: σ²(W) - variance of weight values.
            Low variance suggests weights are clustered, enabling aggressive quantization.
        outlier_ratio: Fraction of weights where |W| > 3σ.
            High ratios indicate outliers that will cause quantization error spikes.
        recommended_bits: Optimal bit width (4, 8, or 16).
        recommended_format: Optimal format ("fp4", "fp8", "fp16", "bf16").
        metrics: Additional computed metrics for debugging.
    """

    name: str
    hessian_trace: float
    hessian_condition: float
    weight_variance: float
    outlier_ratio: float
    recommended_bits: int
    recommended_format: str
    metrics: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"LayerSensitivity({self.name!r}, "
            f"bits={self.recommended_bits}, format={self.recommended_format!r}, "
            f"trace={self.hessian_trace:.2e}, cond={self.hessian_condition:.1f}, "
            f"var={self.weight_variance:.4f}, outliers={self.outlier_ratio:.4f})"
        )


# ============================================================================
# Sensitivity thresholds
# ============================================================================

# Condition number thresholds for quantization decisions
CONDITION_CRITICAL = 10000.0  # Above this, use FP16
CONDITION_SENSITIVE = 1000.0  # Above this, use FP8
CONDITION_NORMAL = 100.0  # Below this, FP4 is safe

# Outlier ratio thresholds
OUTLIER_HIGH = 0.01  # >1% outliers suggests Hadamard needed
OUTLIER_EXTREME = 0.05  # >5% outliers indicates problematic distribution

# Weight variance thresholds
VARIANCE_LOW = 0.001  # Very tight distribution, aggressive quant ok
VARIANCE_HIGH = 1.0  # Wide distribution, may need larger groups

# Hessian trace thresholds (relative to mean)
TRACE_CRITICAL_MULTIPLIER = 10.0  # 10x mean trace = critical layer


# ============================================================================
# Layer type detection
# ============================================================================


def _is_router_layer(name: str) -> bool:
    """Detect if layer is an MoE router (always FP16)."""
    name_lower = name.lower()
    router_patterns = [
        "router",
        "gate.weight",
        "moe_gate",
        "expert_gate",
        "block_sparse_moe.gate",
    ]
    return any(p in name_lower for p in router_patterns)


def _is_embedding_layer(name: str) -> bool:
    """Detect embedding layers (always FP16/BF16)."""
    name_lower = name.lower()
    embed_patterns = ["embed", "wte", "word_embedding"]
    return any(p in name_lower for p in embed_patterns)


def _is_norm_layer(name: str) -> bool:
    """Detect normalization layers (always FP16/BF16)."""
    name_lower = name.lower()
    norm_patterns = ["norm", "layernorm", "rmsnorm", "ln_"]
    return any(p in name_lower for p in norm_patterns)


def _is_attention_layer(name: str) -> bool:
    """Detect attention projection layers (sensitive to position)."""
    name_lower = name.lower()
    attn_patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "c_attn"]
    return any(p in name_lower for p in attn_patterns)


# ============================================================================
# Core analysis functions
# ============================================================================


def _compute_hessian_stats(
    hessian: NDArray[np.floating[Any]] | None,
) -> tuple[float, float]:
    """Compute Hessian trace and condition number.

    Args:
        hessian: Hessian matrix [in_features, in_features] or None.

    Returns:
        Tuple of (trace, condition_number). Returns (0, 1) if hessian is None.
    """
    if hessian is None:
        return 0.0, 1.0

    H = np.asarray(hessian, dtype=np.float64)

    # Trace: sum of diagonal
    trace = float(np.trace(H))

    # Condition number via eigenvalues
    try:
        # Use eigvalsh for symmetric matrices (faster, more stable)
        eigenvalues = np.linalg.eigvalsh(H)
        # Filter small eigenvalues to avoid division by zero
        eig_positive = eigenvalues[eigenvalues > 1e-10]
        if len(eig_positive) == 0:
            condition = 1.0
        else:
            condition = float(eig_positive.max() / eig_positive.min())
    except np.linalg.LinAlgError:
        # Fallback: use numpy's cond function
        condition = float(np.linalg.cond(H))

    return trace, condition


def _compute_weight_stats(
    weight: NDArray[np.floating[Any]],
) -> tuple[float, float, dict[str, float]]:
    """Compute weight statistics for sensitivity analysis.

    Args:
        weight: Weight matrix [out_features, in_features].

    Returns:
        Tuple of (variance, outlier_ratio, extra_metrics).
    """
    w = np.asarray(weight, dtype=np.float64).ravel()

    # Basic statistics
    mean = float(np.mean(w))
    std = float(np.std(w))
    variance = std * std

    # Outlier ratio: |W| > 3σ
    if std > 1e-10:
        threshold = 3.0 * std
        outliers = np.abs(w - mean) > threshold
        outlier_ratio = float(np.mean(outliers))
    else:
        outlier_ratio = 0.0

    # Additional metrics for diagnostics
    metrics = {
        "mean": mean,
        "std": std,
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "abs_max": float(np.max(np.abs(w))),
        "kurtosis": _compute_kurtosis(w, mean, std),
        "sparsity": float(np.mean(np.abs(w) < 1e-6)),
    }

    return variance, outlier_ratio, metrics


def _compute_kurtosis(x: NDArray[np.float64], mean: float, std: float) -> float:
    """Compute excess kurtosis (normal distribution = 0)."""
    if std < 1e-10:
        return 0.0
    centered = x - mean
    return float(np.mean(centered**4) / (std**4) - 3.0)


def recommend_quantization(
    name: str,
    hessian_trace: float,
    hessian_condition: float,
    weight_variance: float,
    outlier_ratio: float,
    mean_trace: float | None = None,
) -> tuple[int, str]:
    """Determine optimal quantization based on sensitivity metrics.

    Decision rules (in order of priority):
    1. Router/embedding/norm layers: Always FP16/BF16
    2. Extreme condition number (>10000): FP16
    3. High condition number (>1000): FP8
    4. High outlier ratio (>5%): FP8 (or Hadamard + FP4)
    5. Critical trace (>10x mean): FP8
    6. Low variance (<0.001): Aggressive FP4 with large groups
    7. Default: FP4 with standard groups

    Args:
        name: Layer name for type detection.
        hessian_trace: Hessian trace value.
        hessian_condition: Hessian condition number.
        weight_variance: Weight variance.
        outlier_ratio: Fraction of outliers.
        mean_trace: Mean Hessian trace across all layers (for relative comparison).

    Returns:
        Tuple of (bits, format) e.g., (4, "fp4") or (16, "bf16").
    """
    # Rule 1: Critical layer types always stay high precision
    if _is_router_layer(name):
        return 16, "bf16"  # Router is sacred for MoE

    if _is_embedding_layer(name) or _is_norm_layer(name):
        return 16, "bf16"

    # Rule 2: Extreme numerical sensitivity
    if hessian_condition > CONDITION_CRITICAL:
        return 16, "fp16"

    # Rule 3: High numerical sensitivity
    if hessian_condition > CONDITION_SENSITIVE:
        return 8, "fp8"

    # Rule 4: Extreme outliers (Hadamard might help, but be conservative)
    if outlier_ratio > OUTLIER_EXTREME:
        return 8, "fp8"

    # Rule 5: High outliers - FP4 with Hadamard recommended
    # The caller should apply Hadamard and re-analyze
    if outlier_ratio > OUTLIER_HIGH:
        # Still use FP4 but note in metrics that Hadamard is recommended
        return 4, "fp4"

    # Rule 6: Critical layers by trace (relative importance)
    if mean_trace is not None and mean_trace > 0:
        if hessian_trace > TRACE_CRITICAL_MULTIPLIER * mean_trace:
            return 8, "fp8"

    # Rule 7: Attention layers are position-sensitive
    if _is_attention_layer(name):
        # Use FP4 but with smaller groups for attention
        return 4, "fp4"

    # Rule 8: Low variance allows aggressive quantization
    if weight_variance < VARIANCE_LOW:
        return 4, "fp4"  # Can use larger groups

    # Default: standard FP4
    return 4, "fp4"


def analyze_layer_sensitivity(
    weight: NDArray[np.floating[Any]],
    hessian: NDArray[np.floating[Any]] | None,
    activations_sample: NDArray[np.floating[Any]] | None = None,
    name: str = "",
    mean_trace: float | None = None,
) -> LayerSensitivity:
    """Determine optimal quantization for a layer.

    Analyzes weight statistics and Hessian information to determine the
    best quantization format for a given layer.

    Args:
        weight: Weight matrix [out_features, in_features].
        hessian: Hessian matrix [in_features, in_features] or None.
            If None, only weight statistics are used for the decision.
        activations_sample: Optional sample activations for additional analysis.
            Shape [n_samples, in_features].
        name: Layer name for type-based decisions and reporting.
        mean_trace: Mean Hessian trace across model (for relative importance).

    Returns:
        LayerSensitivity with computed metrics and recommendation.

    Example:
        >>> weight = np.random.randn(4096, 4096).astype(np.float32)
        >>> hessian = np.eye(4096) * 0.1
        >>> sens = analyze_layer_sensitivity(weight, hessian, name="layer_0.mlp.gate")
        >>> print(f"Use {sens.recommended_bits}-bit {sens.recommended_format}")
    """
    # Compute Hessian statistics
    hessian_trace, hessian_condition = _compute_hessian_stats(hessian)

    # Compute weight statistics
    weight_variance, outlier_ratio, weight_metrics = _compute_weight_stats(weight)

    # Additional analysis from activations if provided
    act_metrics: dict[str, float] = {}
    if activations_sample is not None:
        acts = np.asarray(activations_sample, dtype=np.float64)
        act_metrics["activation_mean"] = float(np.mean(acts))
        act_metrics["activation_std"] = float(np.std(acts))
        act_metrics["activation_max"] = float(np.max(np.abs(acts)))

    # Determine recommendation
    bits, fmt = recommend_quantization(
        name=name,
        hessian_trace=hessian_trace,
        hessian_condition=hessian_condition,
        weight_variance=weight_variance,
        outlier_ratio=outlier_ratio,
        mean_trace=mean_trace,
    )

    # Build metrics dict
    metrics = {
        **weight_metrics,
        **act_metrics,
        "needs_hadamard": outlier_ratio > OUTLIER_HIGH,
    }

    return LayerSensitivity(
        name=name,
        hessian_trace=hessian_trace,
        hessian_condition=hessian_condition,
        weight_variance=weight_variance,
        outlier_ratio=outlier_ratio,
        recommended_bits=bits,
        recommended_format=fmt,
        metrics=metrics,
    )


# ============================================================================
# Model-level analysis
# ============================================================================


def compute_model_sensitivity_profile(
    model_path: str | Path,
    calibration_data: BartowskiCalibration | None = None,
    layers_to_analyze: list[str] | None = None,
    compute_hessians: bool = True,
    verbose: bool = True,
) -> dict[str, LayerSensitivity]:
    """Generate full model sensitivity analysis.

    Analyzes all weight tensors in a model and recommends optimal quantization
    for each layer based on sensitivity metrics.

    Args:
        model_path: Path to HuggingFace model directory or safetensors file.
        calibration_data: Optional calibration dataset for Hessian computation.
            If None, only weight statistics are used.
        layers_to_analyze: Optional list of layer name patterns to analyze.
            If None, analyzes all linear layers.
        compute_hessians: If True and calibration_data provided, compute Hessians.
            Set to False for faster analysis using only weight statistics.
        verbose: Print progress.

    Returns:
        Dict mapping layer names to LayerSensitivity objects.

    Example:
        >>> from metal_marlin.calibration import BartowskiCalibration
        >>> calibration = BartowskiCalibration.v3(max_samples=128)
        >>> profile = compute_model_sensitivity_profile(
        ...     "path/to/model",
        ...     calibration_data=calibration,
        ... )
        >>> for name, sens in profile.items():
        ...     if sens.recommended_bits < 8:
        ...         print(f"{name}: {sens.recommended_format}")
    """
    from safetensors import safe_open

    model_path = Path(model_path)

    # Find safetensors files
    if model_path.is_file():
        st_files = [model_path]
    else:
        st_files = sorted(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files in {model_path}")

    if verbose:
        print(f"Analyzing model: {model_path}")
        print(f"  Calibration data: {'provided' if calibration_data else 'none'}")
        print(f"  Compute Hessians: {compute_hessians}")
        print()

    # Phase 1: Collect all weight tensors and compute basic stats
    weight_stats: dict[str, tuple[NDArray[np.floating[Any]], float, float, dict[str, float]]] = {}

    for st_file in st_files:
        if verbose:
            print(f"Loading {st_file.name}...")

        with safe_open(str(st_file), framework="numpy") as f:
            for name in f.keys():
                # Skip non-weight tensors
                if "weight" not in name.lower():
                    continue

                tensor = f.get_tensor(name)

                # Skip non-2D tensors
                if tensor.ndim != 2:
                    continue

                # Check layer filter
                if layers_to_analyze is not None:
                    if not any(pat in name for pat in layers_to_analyze):
                        continue

                # Compute weight stats
                variance, outlier_ratio, metrics = _compute_weight_stats(tensor)
                weight_stats[name] = (tensor, variance, outlier_ratio, metrics)

    if verbose:
        print(f"Found {len(weight_stats)} weight tensors")

    # Phase 2: Compute Hessians if requested and calibration provided
    hessians: dict[str, NDArray[np.floating[Any]]] = {}

    if compute_hessians and calibration_data is not None:
        if verbose:
            print("\nComputing Hessians from calibration data...")
            print("  (This requires model inference - using simplified estimation)")

        # For a full implementation, we would:
        # 1. Load the model
        # 2. Run calibration samples through it
        # 3. Hook layer inputs to compute X^T @ X
        #
        # For now, we estimate Hessian stats from weight distributions
        # This is less accurate but much faster

        for name, (weight, _, _, _) in weight_stats.items():
            # Simplified Hessian estimation: assume uniform activation distribution
            # This gives H ≈ I scaled by estimated activation magnitude
            in_features = weight.shape[1]

            # Estimate activation scale from weight magnitude
            # (layers with larger weights often have larger activations)
            weight_scale = np.std(weight)
            estimated_activation_var = weight_scale**2

            # Create diagonal Hessian approximation
            hessians[name] = np.diag(
                np.full(in_features, estimated_activation_var, dtype=np.float64)
            )

        if verbose:
            print(f"  Estimated Hessians for {len(hessians)} layers")

    # Phase 3: Compute sensitivity for each layer
    if verbose:
        print("\nComputing layer sensitivities...")

    # First pass: compute all traces to get mean
    traces: list[float] = []
    for name, (weight, _, _, _) in weight_stats.items():
        hessian = hessians.get(name)
        trace, _ = _compute_hessian_stats(hessian)
        traces.append(trace)

    mean_trace = float(np.mean(traces)) if traces else None

    # Second pass: full analysis with mean trace context
    results: dict[str, LayerSensitivity] = {}

    for name, (weight, variance, outlier_ratio, metrics) in weight_stats.items():
        hessian = hessians.get(name)

        sensitivity = analyze_layer_sensitivity(
            weight=weight,
            hessian=hessian,
            activations_sample=None,
            name=name,
            mean_trace=mean_trace,
        )

        results[name] = sensitivity

        if verbose:
            recommendation = f"{sensitivity.recommended_bits}-bit {sensitivity.recommended_format}"
            flags = []
            if sensitivity.metrics.get("needs_hadamard"):
                flags.append("hadamard")
            if sensitivity.hessian_condition > CONDITION_SENSITIVE:
                flags.append("sensitive")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {name}: {recommendation}{flag_str}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        bit_counts = {4: 0, 8: 0, 16: 0}
        format_counts: dict[str, int] = {}
        hadamard_count = 0

        for sens in results.values():
            bit_counts[sens.recommended_bits] = bit_counts.get(sens.recommended_bits, 0) + 1
            format_counts[sens.recommended_format] = format_counts.get(sens.recommended_format, 0) + 1
            if sens.metrics.get("needs_hadamard"):
                hadamard_count += 1

        total = len(results)
        print(f"Total layers: {total}")
        print()
        print("By bit width:")
        for bits in sorted(bit_counts.keys()):
            count = bit_counts[bits]
            pct = count / total * 100 if total > 0 else 0
            print(f"  {bits:2d}-bit: {count:4d} ({pct:5.1f}%)")
        print()
        print("By format:")
        for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"  {fmt:8s}: {count:4d} ({pct:5.1f}%)")
        print()
        print(f"Layers needing Hadamard: {hadamard_count}")

    return results


# ============================================================================
# Utilities
# ============================================================================


def sensitivity_to_config(
    profile: dict[str, LayerSensitivity],
) -> dict[str, dict[str, Any]]:
    """Convert sensitivity profile to quantization config dict.

    Args:
        profile: Output from compute_model_sensitivity_profile.

    Returns:
        Dict mapping layer names to quantization config dicts.
    """
    config = {}

    for name, sens in profile.items():
        layer_config: dict[str, Any] = {
            "bits": sens.recommended_bits,
            "format": sens.recommended_format,
        }

        # Add group size recommendation based on sensitivity
        if sens.recommended_bits == 4:
            if sens.weight_variance < VARIANCE_LOW:
                layer_config["group_size"] = 256  # Low variance: large groups ok
            elif _is_attention_layer(name):
                layer_config["group_size"] = 64  # Attention: smaller groups
            else:
                layer_config["group_size"] = 128  # Default
        elif sens.recommended_bits == 8:
            layer_config["group_size"] = 128
        else:
            layer_config["group_size"] = -1  # Per-channel

        # Add Hadamard recommendation
        if sens.metrics.get("needs_hadamard"):
            layer_config["use_hadamard"] = True
            layer_config["hadamard_block_size"] = 64

        config[name] = layer_config

    return config


def save_sensitivity_profile(
    profile: dict[str, LayerSensitivity],
    path: str | Path,
) -> None:
    """Save sensitivity profile to JSON file.

    Args:
        profile: Output from compute_model_sensitivity_profile.
        path: Output file path.
    """
    import json

    data = {
        name: {
            "hessian_trace": sens.hessian_trace,
            "hessian_condition": sens.hessian_condition,
            "weight_variance": sens.weight_variance,
            "outlier_ratio": sens.outlier_ratio,
            "recommended_bits": sens.recommended_bits,
            "recommended_format": sens.recommended_format,
            "metrics": sens.metrics,
        }
        for name, sens in profile.items()
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_sensitivity_profile(path: str | Path) -> dict[str, LayerSensitivity]:
    """Load sensitivity profile from JSON file.

    Args:
        path: Input file path.

    Returns:
        Dict mapping layer names to LayerSensitivity objects.
    """
    import json

    with open(path) as f:
        data = json.load(f)

    return {
        name: LayerSensitivity(
            name=name,
            hessian_trace=d["hessian_trace"],
            hessian_condition=d["hessian_condition"],
            weight_variance=d["weight_variance"],
            outlier_ratio=d["outlier_ratio"],
            recommended_bits=d["recommended_bits"],
            recommended_format=d["recommended_format"],
            metrics=d.get("metrics", {}),
        )
        for name, d in data.items()
    }
