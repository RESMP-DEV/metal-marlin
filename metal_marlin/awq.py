"""AWQ: Activation-aware Weight Quantization for Generative LLMs.

This module implements the AWQ algorithm which quantizes weights using
activation statistics to identify and protect the most salient weights.

AWQ is superior to GPTQ in many cases because:
1. Uses activation statistics (first-order) instead of Hessian (second-order)
2. Protects only 1% of salient weights via scaling
3. Provides better accuracy for LLMs, especially on perplexity and benchmarks
4. Faster quantization and inference than GPTQ

Algorithm (from AWQ paper):
1. Collect activation statistics from calibration data
2. For each weight channel:
   a. Compute channel importance based on activation magnitude
   b. Select top 1% salient weights
   c. Apply per-channel scaling to protect salient weights
   d. Quantize remaining weights to 4-bit
3. Store quantized weights, scales, and zero-points

Key innovation: Instead of compensating error like GPTQ, AWQ scales
salient weights to ensure they are represented accurately after quantization.

References:
- AWQ Paper: https://arxiv.org/abs/2306.00978
- AWQ GitHub: https://github.com/mit-han-lab/llm-awq
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._compat import from_numpy, to_numpy

# Constants for AWQ
AWQ_SALIENT_RATIO = 0.01  # Protect top 1% of salient weights
AWQ_DEFAULT_GROUP_SIZE = 128
FP4_PER_U32 = 8  # 8 FP4 values packed per uint32

# AWQ uses INT4 for better hardware compatibility
AWQ_QUANT_BITS = 4
AWQ_MAX_INT4 = 7  # INT4 symmetric range [-8, 7]


@dataclass
class AWQResult:
    """Result of AWQ quantization.

    Attributes:
        Q: Quantized weight matrix (packed uint32)
        scales: Per-group scale factors [num_groups, out_features]
        zeros: Per-group zero points [num_groups, out_features]
        q_scale: Per-channel scaling factors for salient weights [out_features]
        quantization_error: Squared quantization error
        salient_ratio: Fraction of protected salient weights
    """

    Q: np.ndarray
    scales: np.ndarray
    zeros: np.ndarray
    q_scale: np.ndarray
    quantization_error: float
    salient_ratio: float


def compute_activation_stats(
    activations: np.ndarray,
    method: Literal["mean", "max", "rms"] = "rms",
) -> np.ndarray:
    """Compute channel-wise activation statistics.

    Args:
        activations: Activation tensor [batch_size, seq_len, in_features]
        method: Method for computing channel importance
            - "mean": Mean absolute activation
            - "max": Maximum absolute activation
            - "rms": Root mean square activation (default)

    Returns:
        Channel-wise importance scores [in_features]
    """
    if method == "mean":
        importance = np.mean(np.abs(activations), axis=(0, 1))
    elif method == "max":
        importance = np.max(np.abs(activations), axis=(0, 1))
    elif method == "rms":
        importance = np.sqrt(np.mean(activations**2, axis=(0, 1)))
    else:
        raise ValueError(f"Unknown method: {method}")

    return importance


def find_salient_weights(
    weights: np.ndarray,
    activation_stats: np.ndarray,
    salient_ratio: float = AWQ_SALIENT_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify salient weights using activation statistics.

    Salient weights are those that connect to high-magnitude activations.
    These weights are protected via scaling to maintain accuracy.

    Args:
        weights: Weight matrix [in_features, out_features]
        activation_stats: Channel importance [in_features]
        salient_ratio: Fraction of weights to protect

    Returns:
        Tuple of:
            salient_mask: Boolean mask [in_features, out_features]
            per_channel_importance: Importance scores per channel [in_features]
    """
    in_features, out_features = weights.shape

    # Compute per-channel importance
    # Higher activation stats = more important channel
    per_channel_importance = activation_stats

    # Normalize importance to [0, 1]
    if np.max(per_channel_importance) > 0:
        per_channel_importance = per_channel_importance / np.max(per_channel_importance)

    # Broadcast to weight shape
    importance_broadcast = per_channel_importance[:, np.newaxis]  # [in_features, 1]
    importance_per_weight = np.tile(
        importance_broadcast, (1, out_features)
    )  # [in_features, out_features]

    # Also consider weight magnitude
    weight_magnitude = np.abs(weights)
    if np.max(weight_magnitude) > 0:
        weight_magnitude = weight_magnitude / np.max(weight_magnitude)

    # Combined importance: activation stats * weight magnitude
    combined_importance = importance_per_weight * weight_magnitude

    # Find top salient_ratio weights
    num_salient = int(np.prod(weights.shape) * salient_ratio)
    salient_mask = np.zeros_like(weights, dtype=bool)

    if num_salient > 0:
        # Get indices of top-k important weights
        flat_importance = combined_importance.ravel()
        threshold = np.partition(flat_importance, -num_salient)[-num_salient]
        salient_mask = combined_importance >= threshold

    return salient_mask, per_channel_importance


def compute_salient_scaling(
    weights: np.ndarray,
    salient_mask: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Compute per-channel scaling factors for salient weights.

    The key innovation of AWQ: scale salient weights so they are
    represented accurately after quantization, then compensate
    by the inverse scaling during inference.

    Args:
        weights: Original weight matrix [in_features, out_features]
        salient_mask: Boolean mask of salient weights
        scales: Per-group scales [num_groups, out_features]
        group_size: Quantization group size

    Returns:
        Per-channel scaling factors [out_features]
    """
    in_features, out_features = weights.shape
    num_groups = in_features // group_size

    # Expand scales to full weight matrix
    scales_expanded = np.repeat(scales, group_size, axis=0)  # [in_features, out_features]

    # Compute scaling for salient weights
    # Goal: ensure salient weights map to quantization levels with minimal error
    q_scale = np.ones(out_features, dtype=np.float32)

    for out_idx in range(out_features):
        salient_weights = weights[:, out_idx][salient_mask[:, out_idx]]
        salient_scales = scales_expanded[:, out_idx][salient_mask[:, out_idx]]

        if len(salient_weights) > 0 and len(salient_scales) > 0:
            # Normalize salient weights by their scales
            normalized_salient = salient_weights / salient_scales

            # Compute optimal scaling to minimize quantization error
            # We want normalized values to map to integer quantization levels
            if np.max(np.abs(normalized_salient)) > 0:
                max_val = np.max(np.abs(normalized_salient))
                scale_factor = AWQ_MAX_INT4 / max_val
                q_scale[out_idx] = scale_factor

    return q_scale


def pack_awq_weights(
    weights: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = AWQ_DEFAULT_GROUP_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack AWQ quantized weights to Marlin-compatible format.

    AWQ uses symmetric INT4 quantization with the same packing as
    standard INT4 for hardware compatibility.

    Args:
        weights: Quantized weight matrix [in_features, out_features]
        scales: Per-group scales [num_groups, out_features]
        zeros: Per-group zero-points [num_groups, out_features]
        group_size: Quantization group size

    Returns:
        Tuple of:
            packed: Packed uint32 weights [in_features // 8, out_features]
            meta: Metadata dictionary
    """
    in_features, out_features = weights.shape
    assert in_features % FP4_PER_U32 == 0, f"in_features must be divisible by {FP4_PER_U32}"
    assert in_features % group_size == 0, (
        f"in_features must be divisible by group_size={group_size}"
    )

    # Ensure weights are in [0, 15] range for packing (shifted INT4)
    # INT4 symmetric: [-8, 7] -> shift by +8 -> [0, 15]
    weights_shifted = weights.astype(np.int16) + AWQ_MAX_INT4
    weights_shifted = np.clip(weights_shifted, 0, 15).astype(np.uint8)

    # Pack 8 consecutive weights into one uint32
    in_packed = in_features // FP4_PER_U32
    packed = np.zeros((in_packed, out_features), dtype=np.uint32)

    for i in range(FP4_PER_U32):
        packed |= weights_shifted[i::FP4_PER_U32, :].astype(np.uint32) << (i * 4)

    meta = {
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "quant_type": "awq_int4",
    }

    return packed, meta


def awq_quantize(
    weights: np.ndarray,
    activations: np.ndarray,
    group_size: int = AWQ_DEFAULT_GROUP_SIZE,
    salient_ratio: float = AWQ_SALIENT_RATIO,
    activation_method: Literal["mean", "max", "rms"] = "rms",
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> AWQResult:
    """Quantize weights using AWQ algorithm.

    Args:
        weights: Weight matrix [in_features, out_features]
        activations: Calibration activations [batch_size, seq_len, in_features]
        group_size: Quantization group size
        salient_ratio: Fraction of salient weights to protect
        activation_method: Method for computing channel importance
        output_backend: Backend for output arrays

    Returns:
        AWQResult with quantized weights and metadata
    """
    in_features, out_features = weights.shape
    weights_fp32 = to_numpy(weights).astype(np.float32)

    # Pad if necessary
    if in_features % group_size != 0:
        pad_len = group_size - (in_features % group_size)
        weights_fp32 = np.pad(weights_fp32, ((0, pad_len), (0, 0)), mode="constant")
        in_features_padded = in_features + pad_len
    else:
        in_features_padded = in_features

    # Compute activation statistics
    activation_stats = compute_activation_stats(activations, method=activation_method)

    # Extend activation stats for padding
    if activation_stats.shape[0] < in_features_padded:
        activation_stats_padded = np.zeros(in_features_padded, dtype=activation_stats.dtype)
        activation_stats_padded[: activation_stats.shape[0]] = activation_stats
        activation_stats = activation_stats_padded

    # Find salient weights
    salient_mask, _ = find_salient_weights(
        weights_fp32, activation_stats, salient_ratio=salient_ratio
    )

    # Initial per-group quantization (symmetric INT4)
    num_groups = in_features_padded // group_size
    weights_grouped = weights_fp32.reshape(num_groups, group_size, out_features)
    group_max = np.abs(weights_grouped).max(axis=1)  # [num_groups, out_features]

    # Compute scales
    scales = np.where(group_max > 0, group_max / AWQ_MAX_INT4, np.float32(1e-7))

    # Apply salient weight scaling
    q_scale = compute_salient_scaling(weights_fp32, salient_mask, scales, group_size)

    # Scale weights by q_scale before quantization
    weights_scaled = weights_fp32 * q_scale[np.newaxis, :]

    # Quantize scaled weights
    weights_grouped_scaled = weights_scaled.reshape(num_groups, group_size, out_features)
    group_max_scaled = np.abs(weights_grouped_scaled).max(axis=1)
    scales_scaled = np.where(
        group_max_scaled > 0, group_max_scaled / AWQ_MAX_INT4, np.float32(1e-7)
    )

    # Quantize to INT4 symmetric
    scales_expanded = np.repeat(scales_scaled, group_size, axis=0)
    normalized = weights_scaled / scales_expanded
    Q = np.clip(np.round(normalized), -AWQ_MAX_INT4 - 1, AWQ_MAX_INT4).astype(np.int8)

    # Compute zero-points (for symmetric quantization, this is typically 0)
    zeros = np.zeros((num_groups, out_features), dtype=np.float32)

    # Quantization error
    dequantized = Q.astype(np.float32) * scales_expanded / q_scale[np.newaxis, :]
    if in_features != in_features_padded:
        dequantized = dequantized[:in_features, :]
    quantization_error = np.sum((weights_fp32[:in_features, :] - dequantized) ** 2)

    # Pack weights
    packed, meta = pack_awq_weights(Q, scales_scaled, zeros, group_size=group_size)

    # Convert to output backend
    packed_out = from_numpy(packed, backend=output_backend)
    scales_out = from_numpy(scales_scaled.astype(np.float16), backend=output_backend)
    zeros_out = from_numpy(zeros.astype(np.float16), backend=output_backend)
    q_scale_out = from_numpy(q_scale.astype(np.float16), backend=output_backend)

    return AWQResult(
        Q=packed_out,
        scales=scales_out,
        zeros=zeros_out,
        q_scale=q_scale_out,
        quantization_error=quantization_error,
        salient_ratio=salient_ratio,
    )


def awq_dequantize(
    packed: Any,
    scales: Any,
    zeros: Any,
    q_scale: Any,
    meta: dict[str, Any],
    weights_dtype: np.dtype | None = None,
    output_backend: Literal["numpy", "mlx", "torch"] = "numpy",
) -> Any:
    """Dequantize AWQ weights back to float.

    Args:
        packed: Packed uint32 weights
        scales: Per-group scales
        zeros: Per-group zero-points
        q_scale: Per-channel salient scaling factors
        meta: Metadata dictionary from pack_awq_weights
        weights_dtype: Output dtype
        output_backend: Backend for output

    Returns:
        Dequantized float weights
    """
    if weights_dtype is None:
        weights_dtype = np.dtype(np.float16)

    packed_np = to_numpy(packed)
    scales_np = to_numpy(scales).astype(np.float32)
    zeros_np = to_numpy(zeros).astype(np.float32)
    q_scale_np = to_numpy(q_scale).astype(np.float32)

    in_features = meta["in_features"]
    out_features = meta["out_features"]
    group_size = meta["group_size"]

    # Unpack uint32 -> INT4 values
    in_packed = in_features // FP4_PER_U32
    Q = np.empty((in_features, out_features), dtype=np.int8)

    for i in range(FP4_PER_U32):
        Q[i::FP4_PER_U32, :] = ((packed_np >> (i * 4)) & 0xF).astype(np.int8)

    # Shift back from [0, 15] to [-8, 7]
    Q = Q - AWQ_MAX_INT4

    # Dequantize
    num_groups = in_features // group_size
    scales_expanded = np.repeat(scales_np, group_size, axis=0)
    dequantized = Q.astype(np.float32) * scales_expanded / q_scale_np[np.newaxis, :]

    # Remove zero-point effect
    zeros_expanded = np.repeat(zeros_np, group_size, axis=0)
    dequantized += zeros_expanded

    return from_numpy(dequantized.astype(weights_dtype), backend=output_backend)


# =============================================================================
# High-level AWQ model quantization
# =============================================================================


def awq_quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    activations_path: str | Path,
    group_size: int = AWQ_DEFAULT_GROUP_SIZE,
    salient_ratio: float = AWQ_SALIENT_RATIO,
    activation_method: Literal["mean", "max", "rms"] = "rms",
    verbose: bool = True,
) -> dict[str, Any]:
    """Quantize all linear layers in a model using AWQ.

    Args:
        model_path: Path to source model (safetensors or HF directory)
        output_path: Path to save quantized model
        activations_path: Path to calibration activation statistics
        group_size: Quantization group size
        salient_ratio: Fraction of salient weights to protect
        activation_method: Method for computing channel importance
        verbose: Print progress

    Returns:
        Stats dictionary
    """
    from pathlib import Path as PathlibPath

    from safetensors import safe_open
    from safetensors.numpy import save_file

    model_path = PathlibPath(model_path)
    output_path = PathlibPath(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find safetensors files
    if model_path.is_file():
        safetensors_files = [model_path]
    else:
        safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Load activation statistics
    activations_path = PathlibPath(activations_path)
    if not activations_path.exists():
        raise FileNotFoundError(f"Activation statistics not found at {activations_path}")

    # For now, use dummy activations (user should provide real calibration data)
    # TODO: Implement proper activation loading
    if verbose:
        print("  Warning: Using dummy activation statistics")
        print("  For production use, provide real calibration activations")

    stats = {
        "quantized_count": 0,
        "skipped_count": 0,
        "original_bytes": 0,
        "quantized_bytes": 0,
        "quant_type": "awq_int4",
        "group_size": group_size,
        "salient_ratio": salient_ratio,
    }

    skip_patterns = ["embed", "norm", "bias", "lm_head"]

    for sf_path in safetensors_files:
        if verbose:
            print(f"  Processing: {sf_path.name}")

        output_tensors = {}

        with safe_open(str(sf_path), framework="numpy") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)

                should_skip = any(pat in name.lower() for pat in skip_patterns)
                is_weight = "weight" in name.lower() and tensor.ndim == 2

                if not should_skip and is_weight:
                    K, N = tensor.shape
                    if N % FP4_PER_U32 == 0 and K % group_size == 0:
                        # Generate dummy activations (should be replaced with real data)
                        dummy_activations = np.random.randn(100, 128, K).astype(np.float32)

                        result = awq_quantize(
                            tensor,
                            dummy_activations,
                            group_size=group_size,
                            salient_ratio=salient_ratio,
                            activation_method=activation_method,
                        )

                        output_tensors[name] = result.Q
                        output_tensors[f"{name}.scales"] = result.scales
                        output_tensors[f"{name}.zeros"] = result.zeros
                        output_tensors[f"{name}.q_scale"] = result.q_scale

                        stats["quantized_count"] += 1
                        stats["original_bytes"] += tensor.nbytes
                        stats["quantized_bytes"] += (
                            result.Q.nbytes
                            + result.scales.nbytes
                            + result.zeros.nbytes
                            + result.q_scale.nbytes
                        )
                    else:
                        output_tensors[name] = tensor
                        stats["skipped_count"] += 1
                else:
                    output_tensors[name] = tensor
                    stats["skipped_count"] += 1

        out_file = output_path / sf_path.name.replace(".safetensors", ".awq.safetensors")
        save_file(output_tensors, str(out_file))

    if stats["quantized_bytes"] > 0:
        stats["compression_ratio"] = stats["original_bytes"] / stats["quantized_bytes"]
    else:
        stats["compression_ratio"] = 1.0

    if verbose:
        print(f"  Quantized: {stats['quantized_count']} tensors")
        print(f"  Skipped: {stats['skipped_count']} tensors")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")

    return stats
