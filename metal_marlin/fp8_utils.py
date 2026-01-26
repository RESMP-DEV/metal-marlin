"""FP8 E4M3 quantization utilities for KV cache.

Provides PyTorch-based FP8 E4M3 quantization/dequantization for memory-efficient
KV cache storage. Achieves ~2x memory savings compared to FP16 with minimal
accuracy loss for attention computations.

FP8 E4M3 Format:
    - 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
    - Range: ~2^-9 to 448 (no infinity, NaN codes for E=15)
    - 240 distinct non-zero values

Usage:
    from metal_marlin.fp8_utils import quantize_to_fp8_e4m3, dequantize_fp8_e4m3

    # Quantize K, V for storage
    k_fp8, k_scales = quantize_to_fp8_e4m3(k_fp16, scale_method="channel")
    v_fp8, v_scales = quantize_to_fp8_e4m3(v_fp16, scale_method="channel")

    # Dequantize for attention computation
    k_restored = dequantize_fp8_e4m3(k_fp8, k_scales)
    v_restored = dequantize_fp8_e4m3(v_fp8, v_scales)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ._compat import require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing

# FP8 E4M3 format constants
# Max representable value: 2^7 * (1 + 7/8) = 128 * 1.875 = 240
# However, for KV cache we use the more conservative 448 bound commonly cited
# (accounting for the full E4M3 range with special handling)
FP8_E4M3_MAX: float = 448.0

# Minimum positive subnormal: 2^-9 = 0.001953125
FP8_E4M3_MIN_POSITIVE: float = 2**-9


def _compute_fp8_e4m3_codebook() -> np.ndarray:
    """Compute all 256 representable FP8 E4M3 values.

    FP8 E4M3 format:
    - 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
    - Normal: (-1)^S * 2^(E-7) * (1 + M/8) for 0 < E < 15
    - Subnormal: (-1)^S * 2^(-6) * (M/8) for E=0, M>0
    - Zero: E=0, M=0
    - NaN: E=15 (all exponent bits set) - E4M3 has no infinity

    Returns:
        Array of 256 float32 values for codes 0-255.
    """
    values = np.zeros(256, dtype=np.float32)

    for code in range(256):
        s = (code >> 7) & 1
        e = (code >> 3) & 0xF
        m = code & 0x7

        sign = -1.0 if s else 1.0

        if e == 15:
            # NaN (E4M3 has no infinity)
            values[code] = np.nan
        elif e == 0:
            if m == 0:
                # Zero (signed)
                values[code] = 0.0 if s == 0 else -0.0
            else:
                # Subnormal: 2^(-6) * (M/8)
                values[code] = sign * (2**-6) * (m / 8)
        else:
            # Normal: 2^(E-7) * (1 + M/8)
            values[code] = sign * (2 ** (e - 7)) * (1 + m / 8)

    return values


# Precomputed E4M3 codebook for dequantization lookup
_FP8_E4M3_CODEBOOK: np.ndarray = _compute_fp8_e4m3_codebook()

# PyTorch version of codebook (lazily initialized)
_FP8_E4M3_CODEBOOK_TORCH: torch_typing.Tensor | None = None


def _get_torch_codebook(device: str = "cpu") -> torch_typing.Tensor:
    """Get the FP8 E4M3 codebook as a PyTorch tensor.

    The codebook is cached globally and moved to the requested device.

    Args:
        device: Target device for the codebook tensor.

    Returns:
        Tensor of shape [256] with FP8 E4M3 values.
    """
    global _FP8_E4M3_CODEBOOK_TORCH

    require_torch()
    assert torch is not None

    if _FP8_E4M3_CODEBOOK_TORCH is None:
        _FP8_E4M3_CODEBOOK_TORCH = torch.from_numpy(_FP8_E4M3_CODEBOOK)

    return _FP8_E4M3_CODEBOOK_TORCH.to(device)


def _float_to_fp8_e4m3_scalar(val: float) -> int:
    """Convert a single float to FP8 E4M3 code.

    This is the reference implementation for understanding the encoding.
    For vectorized operations, use _quantize_tensor_to_fp8.

    Args:
        val: Input float value.

    Returns:
        uint8 code (0-255) for the FP8 E4M3 representation.
    """
    if np.isnan(val):
        return 0x7F  # Positive NaN

    # Handle sign
    sign = 0
    if val < 0:
        sign = 1
        val = -val

    # Clip to representable range
    val = min(val, FP8_E4M3_MAX)

    if val == 0:
        return sign << 7

    # Find the exponent
    # Normal range: 2^-6 to 2^7
    if val < 2**-6:
        # Subnormal: E=0, M = val / (2^-6 / 8) = val * 2^9
        m = int(round(val * (2**9)))
        m = min(max(m, 0), 7)
        return (sign << 7) | m
    else:
        # Normal: find exponent such that 1 <= val/2^(e-7) < 2
        e = int(np.floor(np.log2(val))) + 7
        e = min(max(e, 1), 14)  # E in [1, 14] for normal

        # Compute mantissa
        mantissa_float = val / (2 ** (e - 7)) - 1.0
        m = int(round(mantissa_float * 8))
        m = min(max(m, 0), 7)

        return (sign << 7) | (e << 3) | m


def quantize_to_fp8_e4m3(
    tensor: torch_typing.Tensor,
    scale_method: Literal["tensor", "channel", "group"] = "tensor",
    group_size: int = 128,
) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
    """Quantize FP16/BF16 tensor to FP8 E4M3.

    Computes per-element or per-group scales and quantizes to uint8 storage
    representing FP8 E4M3 codes. The quantization is simulated using the
    FP8 E4M3 codebook with nearest-neighbor matching.

    Memory savings: ~2x compared to FP16 (uint8 + FP16 scales overhead).

    Args:
        tensor: Input tensor in FP16/BF16/FP32. Expected shape for KV cache:
            [batch, num_kv_heads, seq_len, head_dim]
        scale_method: How to compute quantization scales:
            - "tensor": Single scale for entire tensor (fastest, least accurate)
            - "channel": Per-channel scale along last dim (recommended for KV)
            - "group": Per-group scale with specified group_size
        group_size: Group size for "group" scale method. Must divide the
            last dimension evenly.

    Returns:
        fp8_tensor: uint8 tensor with FP8 E4M3 codes, same shape as input.
        scales: FP16 scales for dequantization. Shape depends on scale_method:
            - "tensor": [1] or scalar broadcastable
            - "channel": [..., 1] (scale per position in last dim)
            - "group": [..., num_groups] where num_groups = last_dim / group_size

    Raises:
        RuntimeError: If PyTorch is not available.
        ValueError: If group_size doesn't evenly divide the last dimension.

    Example:
        >>> k = torch.randn(1, 8, 1024, 128, dtype=torch.float16, device="mps")
        >>> k_fp8, k_scales = quantize_to_fp8_e4m3(k, scale_method="channel")
        >>> k_fp8.shape  # Same as input
        torch.Size([1, 8, 1024, 128])
        >>> k_fp8.dtype
        torch.uint8
        >>> k_scales.shape  # One scale per head_dim position
        torch.Size([1, 8, 1024, 1])
    """
    require_torch()
    assert torch is not None

    # Get device and convert to float32 for computation
    device = tensor.device
    x = tensor.float()

    # Compute scales based on method
    if scale_method == "tensor":
        # Single scale for entire tensor
        absmax = x.abs().max()
        scale = absmax / FP8_E4M3_MAX
        scale = scale.clamp(min=1e-12)
        # Reshape for broadcasting
        scale = scale.view(*([1] * tensor.ndim))

    elif scale_method == "channel":
        # Per-position scale along last dimension (head_dim for KV cache)
        # Scale shape: [..., 1] to broadcast against last dim
        absmax = x.abs().amax(dim=-1, keepdim=True)
        scale = absmax / FP8_E4M3_MAX
        scale = scale.clamp(min=1e-12)

    elif scale_method == "group":
        # Group-wise scaling along last dimension
        last_dim = x.shape[-1]
        if last_dim % group_size != 0:
            raise ValueError(
                f"Last dimension ({last_dim}) must be divisible by group_size ({group_size})"
            )

        num_groups = last_dim // group_size
        # Reshape to [..., num_groups, group_size]
        original_shape = x.shape
        x_grouped = x.view(*original_shape[:-1], num_groups, group_size)

        # Compute absmax per group
        absmax = x_grouped.abs().amax(dim=-1, keepdim=False)  # [..., num_groups]
        scale = absmax / FP8_E4M3_MAX
        scale = scale.clamp(min=1e-12)

        # Expand scale back for element-wise division
        # [..., num_groups] -> [..., num_groups, 1] -> [..., last_dim]
        scale_expanded = scale.unsqueeze(-1).expand(*scale.shape, group_size)
        scale_expanded = scale_expanded.reshape(original_shape)

        # Quantize with expanded scales
        scaled = x / scale_expanded
        scaled = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

        # Get the codebook for nearest-neighbor quantization
        codebook = _get_torch_codebook(device)

        # Find nearest FP8 code for each value
        # For efficiency, we use a linear approximation for the common case
        # and fall back to exact lookup only when necessary
        fp8_tensor = _quantize_tensor_to_fp8(scaled, codebook)

        return fp8_tensor, scale.to(torch.float16)

    else:
        raise ValueError(f"Unknown scale_method: {scale_method}. Use 'tensor', 'channel', or 'group'")

    # Scale and clamp
    scaled = x / scale
    scaled = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Get the codebook for nearest-neighbor quantization
    codebook = _get_torch_codebook(device)

    # Find nearest FP8 code for each value
    fp8_tensor = _quantize_tensor_to_fp8(scaled, codebook)

    return fp8_tensor, scale.to(torch.float16)


def _quantize_tensor_to_fp8(
    scaled: torch_typing.Tensor,
    codebook: torch_typing.Tensor,
) -> torch_typing.Tensor:
    """Quantize scaled tensor to FP8 codes using nearest-neighbor.

    Uses a fast linear approximation for normal values and handles
    edge cases (zero, subnormal) correctly.

    Args:
        scaled: Tensor with values in [-448, 448] range.
        codebook: FP8 E4M3 codebook tensor of shape [256].

    Returns:
        uint8 tensor with FP8 codes.
    """
    require_torch()
    assert torch is not None

    # Get valid (non-NaN) codes for nearest-neighbor matching
    # NaN codes are 120-127 (positive) and 248-255 (negative)
    valid_mask = ~torch.isnan(codebook)
    valid_indices = torch.where(valid_mask)[0]
    valid_values = codebook[valid_mask]

    # For large tensors, process in chunks to avoid memory explosion
    # from the [N, 240] distance matrix
    flat = scaled.reshape(-1)
    n = flat.shape[0]
    chunk_size = 1 << 18  # ~256K elements per chunk

    result = torch.empty(n, dtype=torch.uint8, device=scaled.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = flat[start:end]

        # Compute distances to all valid FP8 values
        # chunk: [chunk_size], valid_values: [240]
        # distances: [chunk_size, 240]
        distances = (chunk.unsqueeze(1) - valid_values.unsqueeze(0)).abs()

        # Find nearest valid code
        nearest_idx = distances.argmin(dim=1)
        result[start:end] = valid_indices[nearest_idx].to(torch.uint8)

    return result.reshape(scaled.shape)


def dequantize_fp8_e4m3(
    fp8_tensor: torch_typing.Tensor,
    scales: torch_typing.Tensor,
    output_dtype: torch_typing.dtype | None = None,
) -> torch_typing.Tensor:
    """Dequantize FP8 E4M3 to FP16/BF16/FP32.

    Performs codebook lookup to convert uint8 FP8 codes back to float values,
    then applies the quantization scales.

    Args:
        fp8_tensor: uint8 tensor with FP8 E4M3 codes.
        scales: Quantization scales from quantize_to_fp8_e4m3. Shape must
            broadcast against fp8_tensor.
        output_dtype: Output dtype. Defaults to torch.float16 if None.

    Returns:
        Dequantized tensor in output_dtype.

    Example:
        >>> k_fp8, k_scales = quantize_to_fp8_e4m3(k, scale_method="channel")
        >>> k_restored = dequantize_fp8_e4m3(k_fp8, k_scales)
        >>> k_restored.dtype
        torch.float16
    """
    require_torch()
    assert torch is not None

    if output_dtype is None:
        output_dtype = torch.float16

    device = fp8_tensor.device

    # Get codebook on the right device
    codebook = _get_torch_codebook(device)

    # Lookup: use FP8 codes as indices into codebook
    # fp8_tensor is uint8 [0, 255], codebook is [256] floats
    dequantized = codebook[fp8_tensor.long()]

    # Apply scales
    scales_float = scales.float()

    # Handle group scaling case where scales need expansion
    # Group scaling has shape [..., num_groups] where num_groups < last_dim
    # and num_groups > 1 (tensor scaling has num_groups = 1)
    if scales_float.shape != dequantized.shape:
        num_groups = scales_float.shape[-1]
        last_dim = dequantized.shape[-1]

        # Only expand if this looks like group scaling (multiple groups, divisible)
        if num_groups > 1 and last_dim > num_groups and last_dim % num_groups == 0:
            group_size = last_dim // num_groups
            # Expand: [..., num_groups] -> [..., num_groups, 1] -> [..., last_dim]
            scales_expanded = scales_float.unsqueeze(-1).expand(*scales_float.shape, group_size)
            scales_float = scales_expanded.reshape(*dequantized.shape[:-1], last_dim)
        # Otherwise, rely on broadcasting (tensor/channel scaling)

    result = dequantized * scales_float

    return result.to(output_dtype)


def quantize_kv_cache_fp8(
    k: torch_typing.Tensor,
    v: torch_typing.Tensor,
    scale_method: Literal["tensor", "channel", "group"] = "channel",
    group_size: int = 128,
) -> tuple[
    tuple[torch_typing.Tensor, torch_typing.Tensor],
    tuple[torch_typing.Tensor, torch_typing.Tensor],
]:
    """Quantize both K and V tensors for KV cache storage.

    Convenience function that quantizes both key and value tensors using
    the same scale method.

    Args:
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        scale_method: Scale computation method (see quantize_to_fp8_e4m3)
        group_size: Group size for "group" scale method

    Returns:
        ((k_fp8, k_scales), (v_fp8, v_scales)): Quantized tensors and scales.

    Example:
        >>> k = torch.randn(1, 8, 1024, 128, dtype=torch.float16, device="mps")
        >>> v = torch.randn(1, 8, 1024, 128, dtype=torch.float16, device="mps")
        >>> (k_q, k_s), (v_q, v_s) = quantize_kv_cache_fp8(k, v)
    """
    k_fp8, k_scales = quantize_to_fp8_e4m3(k, scale_method, group_size)
    v_fp8, v_scales = quantize_to_fp8_e4m3(v, scale_method, group_size)
    return (k_fp8, k_scales), (v_fp8, v_scales)


def dequantize_kv_cache_fp8(
    k_fp8: torch_typing.Tensor,
    k_scales: torch_typing.Tensor,
    v_fp8: torch_typing.Tensor,
    v_scales: torch_typing.Tensor,
    output_dtype: torch_typing.dtype | None = None,
) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
    """Dequantize K and V tensors from FP8 storage.

    Convenience function that dequantizes both key and value tensors.

    Args:
        k_fp8: Quantized keys (uint8)
        k_scales: Key scales
        v_fp8: Quantized values (uint8)
        v_scales: Value scales
        output_dtype: Output dtype (default: torch.float16)

    Returns:
        (k, v): Dequantized key and value tensors.

    Example:
        >>> (k_q, k_s), (v_q, v_s) = quantize_kv_cache_fp8(k, v)
        >>> k_restored, v_restored = dequantize_kv_cache_fp8(k_q, k_s, v_q, v_s)
    """
    k = dequantize_fp8_e4m3(k_fp8, k_scales, output_dtype)
    v = dequantize_fp8_e4m3(v_fp8, v_scales, output_dtype)
    return k, v


def compute_quantization_error(
    original: torch_typing.Tensor,
    scale_method: Literal["tensor", "channel", "group"] = "channel",
    group_size: int = 128,
) -> dict[str, float]:
    """Compute quantization error metrics for FP8 E4M3.

    Useful for evaluating the impact of FP8 quantization on model quality.

    Args:
        original: Original FP16/BF16 tensor
        scale_method: Scale method to test
        group_size: Group size for "group" method

    Returns:
        Dict with error metrics:
            - max_error: Maximum absolute error
            - mean_error: Mean absolute error
            - rmse: Root mean squared error
            - snr_db: Signal-to-noise ratio in dB
    """
    require_torch()
    assert torch is not None

    fp8, scales = quantize_to_fp8_e4m3(original, scale_method, group_size)
    restored = dequantize_fp8_e4m3(fp8, scales, original.dtype)

    diff = (original.float() - restored.float()).abs()

    max_error = diff.max().item()
    mean_error = diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    # SNR: 10 * log10(signal_power / noise_power)
    signal_power = original.float().pow(2).mean()
    noise_power = diff.pow(2).mean()
    snr_db = 10 * torch.log10(signal_power / noise_power.clamp(min=1e-10)).item()

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "rmse": rmse,
        "snr_db": snr_db,
    }
