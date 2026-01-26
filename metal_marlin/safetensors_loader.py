"""
Load and convert weights from safetensors format to Marlin-quantized layout.

Streams weights from .safetensors files, quantizes 2D weight matrices on-the-fly
to FP4 (E2M1) or INT4 format with per-group scales, and can serialize the result
as a .marlin.safetensors file ready for inference.

This module uses numpy by default for weight loading and processing, with optional
conversion to MLX arrays when MLX is available and requested.

Usage:
    from safetensors_loader import convert_model_to_marlin

    # Numpy output (default, no MLX required)
    convert_model_to_marlin(
        "model.safetensors",
        "model.marlin.safetensors",
        quant_type="fp4",
        group_size=128,
    )

    # MLX output (requires MLX)
    for name, packed, scales in load_and_quantize_safetensors(
        "model.safetensors",
        output_backend="mlx",
    ):
        ...  # packed and scales are mx.array
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

# Lazy MLX import for optional conversion
_mlx_core: mx | None = None


def _get_mlx() -> mx:
    """Lazily import and cache MLX core module."""
    global _mlx_core
    if _mlx_core is None:
        try:
            import mlx.core as mx

            _mlx_core = mx
        except ImportError as e:
            raise ImportError(
                "MLX is required for output_backend='mlx'. "
                "Install with: pip install mlx"
            ) from e
    return _mlx_core


def _has_mlx() -> bool:
    """Check if MLX is available without importing it."""
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


# Type alias for output backend selection
OutputBackend = Literal["numpy", "mlx"]


def _to_output_backend(
    arr: np.ndarray, backend: OutputBackend
) -> np.ndarray | mx.array:
    """Convert numpy array to requested backend format.

    Args:
        arr: Numpy array to convert
        backend: Target backend ("numpy" or "mlx")

    Returns:
        Array in requested format
    """
    if backend == "numpy":
        return arr
    mx = _get_mlx()
    return mx.array(arr)


def _quantize_tensor_fp4_numpy(
    weight: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a 2D weight tensor to FP4 (E2M1) with per-group scales.

    Uses the same algorithm as metal_marlin.pack_fp4_weights but operates
    on numpy arrays directly.

    Args:
        weight: float16/float32 tensor [K, N]
        group_size: elements per quantization group

    Returns:
        (packed_weights [K, N//8] as uint32, scales [K//group_size, N] as float16)
    """
    # Import pack_fp4_weights lazily - it may use MLX internally
    # but we convert back to numpy at the end
    from metal_marlin import pack_fp4_weights

    # Convert to MLX for the packing operation (pack_fp4_weights requires it)
    mx = _get_mlx()
    weight_mx = mx.array(weight)

    # pack_fp4_weights expects [out_features, in_features] (row-major, PyTorch convention)
    # and transposes internally. Since we already have [K, N], pass transposed.
    packed, scales = pack_fp4_weights(weight_mx.T, group_size=group_size)

    # Convert back to numpy
    return np.array(packed), np.array(scales)


def _quantize_tensor_int4_numpy(
    weight: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize a 2D weight tensor to asymmetric INT4 with per-group scales and zeros.

    Each group of `group_size` elements along K shares one scale and zero_point.
    Values are quantized to [0, 15] and packed 8 per uint32.

    This implementation is pure numpy - no MLX dependency.

    Args:
        weight: float16/float32 tensor [K, N]
        group_size: elements per quantization group

    Returns:
        (packed_weights [K, N//8] as uint32,
         scales [K//group_size, N] as float16,
         zeros [K//group_size, N] as float16)
    """
    w = weight.astype(np.float32)
    K, N = w.shape

    if N % 8 != 0:
        raise ValueError(f"N ({N}) must be divisible by 8 for INT4 packing")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    n_groups = K // group_size
    w_grouped = w.reshape(n_groups, group_size, N)

    # Per-group min/max for asymmetric quantization
    g_min = w_grouped.min(axis=1)  # [n_groups, N]
    g_max = w_grouped.max(axis=1)  # [n_groups, N]

    # Scale and zero_point: map [min, max] -> [0, 15]
    scales_np = (g_max - g_min) / 15.0
    scales_np = np.maximum(scales_np, 1e-7)
    zeros_np = g_min  # zero_point in float domain

    # Quantize each element to [0, 15]
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    zeros_expanded = np.repeat(zeros_np, group_size, axis=0)  # [K, N]

    w_quant = np.clip(
        np.round((w - zeros_expanded) / scales_expanded), 0, 15
    ).astype(np.uint8)

    # Pack 8 values along N into uint32
    packed_np = np.zeros((K, N // 8), dtype=np.uint32)
    for i in range(8):
        packed_np |= w_quant[:, i::8].astype(np.uint32) << (i * 4)

    return packed_np, scales_np.astype(np.float16), zeros_np.astype(np.float16)


def load_and_quantize_safetensors(
    path: str | Path,
    quant_type: str = "fp4",
    group_size: int = 128,
    output_backend: OutputBackend = "numpy",
) -> Generator[tuple[str, np.ndarray | mx.array, np.ndarray | mx.array | None], None, None]:
    """
    Stream weights from a safetensors file, quantizing 2D weight matrices on-the-fly.

    Non-weight tensors (embeddings, biases, norms) are yielded unchanged.

    Args:
        path: Path to .safetensors file.
        quant_type: "fp4" for E2M1 FP4 or "int4" for asymmetric unsigned INT4.
        group_size: Number of elements per quantization group.
        output_backend: "numpy" (default) or "mlx" for output array type.
            MLX backend requires MLX to be installed.

    Yields:
        (name, packed_weights_or_tensor, scales_or_None) tuples.
        For quantized layers, scales is a float16 array.
        For non-quantized layers, scales is None and the tensor is passed through.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)  # Returns numpy array directly

            # Only quantize 2D weight matrices
            if "weight" in name and tensor.ndim == 2:
                if quant_type == "fp4":
                    packed, scales = _quantize_tensor_fp4_numpy(tensor, group_size)
                elif quant_type == "int4":
                    packed, scales, _zeros = _quantize_tensor_int4_numpy(tensor, group_size)
                else:
                    raise ValueError(f"Unknown quant_type: {quant_type!r}. Use 'fp4' or 'int4'.")

                yield (
                    name,
                    _to_output_backend(packed, output_backend),
                    _to_output_backend(scales, output_backend),
                )
            else:
                yield name, _to_output_backend(tensor, output_backend), None


def convert_model_to_marlin(
    input_path: str | Path,
    output_path: str | Path,
    quant_type: str = "fp4",
    group_size: int = 128,
) -> None:
    """
    Convert a safetensors model to Marlin-quantized format.

    Reads all tensors from input_path, quantizes weight matrices, and saves
    the result as a new safetensors file with `.packed` and `.scales` suffixes
    for quantized layers.

    This function uses numpy throughout - no MLX dependency required.

    Args:
        input_path: Source .safetensors file.
        output_path: Destination path (typically .marlin.safetensors).
        quant_type: "fp4" or "int4".
        group_size: Quantization group size.
    """
    from safetensors.numpy import save_file

    output_tensors: dict[str, np.ndarray] = {}

    # Use numpy backend for serialization (safetensors.numpy expects np.ndarray)
    for name, packed, scales in load_and_quantize_safetensors(
        input_path, quant_type, group_size, output_backend="numpy"
    ):
        if scales is not None:
            output_tensors[f"{name}.packed"] = packed  # type: ignore[assignment]
            output_tensors[f"{name}.scales"] = scales  # type: ignore[assignment]
        else:
            output_tensors[name] = packed  # type: ignore[assignment]

    save_file(output_tensors, str(output_path))
    print(f"Saved Marlin model to {output_path}")


# Convenience function to check MLX availability
def has_mlx_backend() -> bool:
    """Check if MLX backend is available for loading.

    Returns:
        True if MLX can be imported, False otherwise
    """
    return _has_mlx()


# Backward compatibility aliases for MLX-based quantization (used by converters/)
def _quantize_tensor_fp4(
    weight: mx.array, group_size: int = 128
) -> tuple[mx.array, mx.array]:
    """
    Quantize a 2D MLX weight tensor to FP4 (E2M1) with per-group scales.

    This is a backward-compatible wrapper that accepts and returns MLX arrays.
    For new code, prefer using numpy-based functions with output_backend parameter.

    Args:
        weight: MLX float16 tensor [K, N]
        group_size: elements per quantization group

    Returns:
        (packed_weights [K, N//8] as uint32, scales [K//group_size, N] as float16)
    """
    mx = _get_mlx()
    # Convert MLX to numpy, quantize, convert back
    weight_np = np.array(weight)
    packed_np, scales_np = _quantize_tensor_fp4_numpy(weight_np, group_size)
    return mx.array(packed_np), mx.array(scales_np)


def _quantize_tensor_int4(
    weight: mx.array, group_size: int = 128
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Quantize a 2D MLX weight tensor to asymmetric INT4 with per-group scales and zeros.

    This is a backward-compatible wrapper that accepts and returns MLX arrays.
    For new code, prefer using numpy-based functions with output_backend parameter.

    Args:
        weight: MLX float16 tensor [K, N]
        group_size: elements per quantization group

    Returns:
        (packed_weights [K, N//8] as uint32,
         scales [K//group_size, N] as float16,
         zeros [K//group_size, N] as float16)
    """
    mx = _get_mlx()
    # Convert MLX to numpy, quantize, convert back
    weight_np = np.array(weight)
    packed_np, scales_np, zeros_np = _quantize_tensor_int4_numpy(weight_np, group_size)
    return mx.array(packed_np), mx.array(scales_np), mx.array(zeros_np)
