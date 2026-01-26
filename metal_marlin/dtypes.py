"""Centralized dtype configuration for Metal Marlin.

This module provides a unified dtype configuration system that allows consistent
dtype handling across all components: quantization, kernels, inference, attention,
and KV cache.

MLX is optional - numpy dtype operations work without it. MLX dtypes return None
when MLX is not installed.

Usage:
    from metal_marlin.dtypes import (
        DTypeConfig,
        get_mlx_dtype,
        get_numpy_dtype,
        numpy_dtype_for_mlx,
        mlx_dtype_to_str,
        HAS_MLX,
    )

    # Create config with custom dtypes
    config = DTypeConfig(
        weights="bf16",
        activations="bf16",
        kv_cache="fp8",
    )

    # Get MLX dtype for activations (None if MLX unavailable)
    act_dtype = config.mlx_activations  # mx.bfloat16 or None

    # Get numpy dtype for scales (always works)
    scale_dtype = get_numpy_dtype(config.scales)  # np.float16 or np.float32

    # Convert MLX dtype to numpy dtype for storage
    # (useful when you have an MLX array and need numpy storage format)
    if HAS_MLX:
        np_dtype = numpy_dtype_for_mlx(mx.bfloat16)  # np.float16 (numpy storage)

    # Convert MLX dtype to string name
    if HAS_MLX:
        dtype_str = mlx_dtype_to_str(mx.bfloat16)  # "bf16"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._compat import HAS_MLX, mx

# Type aliases for dtype string literals
WeightDType = Literal["fp16", "bf16"]
ActivationDType = Literal["fp16", "bf16"]
AccumulationDType = Literal["fp32"]  # Always FP32 for accuracy
ScaleDType = Literal["fp16", "fp32"]
KVCacheDType = Literal["fp16", "bf16", "fp8"]


@dataclass
class DTypeConfig:
    """Centralized dtype configuration for Metal Marlin operations.

    This configuration controls the precision used throughout the inference
    pipeline. Default values are chosen for a balance of performance and
    accuracy on Apple Silicon.

    Attributes:
        weights: Storage dtype for dequantized weights (fp16 or bf16).
            Note: Quantized weights are always stored as uint32 with packed
            nibbles; this controls the dtype after dequantization.
        activations: Dtype for input/output activations (fp16 or bf16).
        accumulation: Dtype for GEMM accumulation. Always FP32 for accuracy.
        scales: Storage dtype for quantization scales (fp16 or fp32).
            FP32 scales provide slightly better accuracy but use 2x memory.
        kv_cache: Storage dtype for KV cache entries (fp16, bf16, or fp8).
            FP8 provides 2x memory savings for long context at some accuracy cost.
    """

    weights: WeightDType = "bf16"
    activations: ActivationDType = "bf16"
    accumulation: AccumulationDType = "fp32"
    scales: ScaleDType = "fp16"
    kv_cache: KVCacheDType = "bf16"

    @property
    def mlx_weights(self) -> Any | None:
        """Get MLX dtype for weights. Returns None if MLX unavailable."""
        if not HAS_MLX:
            return None
        return _STR_TO_MLX[self.weights]

    @property
    def mlx_activations(self) -> Any | None:
        """Get MLX dtype for activations. Returns None if MLX unavailable."""
        if not HAS_MLX:
            return None
        return _STR_TO_MLX[self.activations]

    @property
    def mlx_accumulation(self) -> Any | None:
        """Get MLX dtype for accumulation. Returns None if MLX unavailable."""
        if not HAS_MLX:
            return None
        return _STR_TO_MLX[self.accumulation]

    @property
    def mlx_scales(self) -> Any | None:
        """Get MLX dtype for scales. Returns None if MLX unavailable."""
        if not HAS_MLX:
            return None
        return _STR_TO_MLX[self.scales]

    @property
    def mlx_kv_cache(self) -> Any | None:
        """Get MLX dtype for KV cache. Returns None if MLX unavailable."""
        if not HAS_MLX:
            return None
        return _STR_TO_MLX[self.kv_cache]

    @property
    def numpy_weights(self) -> np.dtype:
        """Get numpy dtype for weights."""
        return _STR_TO_NUMPY[self.weights]

    @property
    def numpy_activations(self) -> np.dtype:
        """Get numpy dtype for activations."""
        return _STR_TO_NUMPY[self.activations]

    @property
    def numpy_accumulation(self) -> np.dtype:
        """Get numpy dtype for accumulation."""
        return _STR_TO_NUMPY[self.accumulation]

    @property
    def numpy_scales(self) -> np.dtype:
        """Get numpy dtype for scales."""
        return _STR_TO_NUMPY[self.scales]

    @property
    def metal_weights(self) -> str:
        """Get Metal shader type name for weights."""
        return _STR_TO_METAL[self.weights]

    @property
    def metal_activations(self) -> str:
        """Get Metal shader type name for activations."""
        return _STR_TO_METAL[self.activations]

    @property
    def metal_accumulation(self) -> str:
        """Get Metal shader type name for accumulation."""
        return _STR_TO_METAL[self.accumulation]

    @property
    def metal_scales(self) -> str:
        """Get Metal shader type name for scales."""
        return _STR_TO_METAL[self.scales]


# Mapping from string dtype names to MLX dtypes
# Populated only when MLX is available
_STR_TO_MLX: dict[str, Any] = {}
if HAS_MLX and mx is not None:
    _STR_TO_MLX = {
        "fp16": mx.float16,
        "bf16": mx.bfloat16,
        "fp32": mx.float32,
        "fp8": mx.float16,  # MLX doesn't have native fp8, use fp16 for storage
    }

# Mapping from string dtype names to numpy dtypes
_STR_TO_NUMPY: dict[str, np.dtype] = {
    "fp16": np.float16,
    "bf16": np.float16,  # numpy doesn't have native bf16, use fp16
    "fp32": np.float32,
    "fp8": np.float16,  # use fp16 as storage format
}

# Mapping from string dtype names to Metal shader type names
_STR_TO_METAL: dict[str, str] = {
    "fp16": "half",
    "bf16": "bfloat",
    "fp32": "float",
    "fp8": "half",  # Metal doesn't have native fp8
}

# Mapping from MLX dtype to numpy dtype
# Note: numpy doesn't have native bf16, so we use fp16 for storage
# The actual bf16<->fp32 conversion happens in MLX
# Populated only when MLX is available
_MLX_TO_NUMPY: dict[Any, np.dtype] = {}
_MLX_TO_STR: dict[Any, str] = {}

if HAS_MLX and mx is not None:
    _MLX_TO_NUMPY = {
        mx.float16: np.float16,
        mx.bfloat16: np.float16,  # numpy storage; MLX handles actual bf16
        mx.float32: np.float32,
    }
    _MLX_TO_STR = {
        mx.float16: "fp16",
        mx.bfloat16: "bf16",
        mx.float32: "fp32",
    }


def get_mlx_dtype(dtype_str: str) -> Any:
    """Convert dtype string to MLX dtype.

    Args:
        dtype_str: One of "fp16", "bf16", "fp32", "fp8"

    Returns:
        Corresponding MLX dtype

    Raises:
        ImportError: If MLX is not available
        ValueError: If dtype_str is not recognized
    """
    if not HAS_MLX:
        raise ImportError(
            "MLX is not available. Install with: pip install mlx"
        )
    if dtype_str not in _STR_TO_MLX:
        raise ValueError(
            f"Unknown dtype: {dtype_str!r}. "
            f"Valid options: {list(_STR_TO_MLX.keys())}"
        )
    return _STR_TO_MLX[dtype_str]


def get_numpy_dtype(dtype_str: str) -> np.dtype:
    """Convert dtype string to numpy dtype.

    Args:
        dtype_str: One of "fp16", "bf16", "fp32", "fp8"

    Returns:
        Corresponding numpy dtype

    Raises:
        ValueError: If dtype_str is not recognized
    """
    if dtype_str not in _STR_TO_NUMPY:
        raise ValueError(
            f"Unknown dtype: {dtype_str!r}. "
            f"Valid options: {list(_STR_TO_NUMPY.keys())}"
        )
    return _STR_TO_NUMPY[dtype_str]


def get_metal_type(dtype_str: str) -> str:
    """Convert dtype string to Metal shader type name.

    Args:
        dtype_str: One of "fp16", "bf16", "fp32", "fp8"

    Returns:
        Corresponding Metal type name (half, bfloat, float)

    Raises:
        ValueError: If dtype_str is not recognized
    """
    if dtype_str not in _STR_TO_METAL:
        raise ValueError(
            f"Unknown dtype: {dtype_str!r}. "
            f"Valid options: {list(_STR_TO_METAL.keys())}"
        )
    return _STR_TO_METAL[dtype_str]


def numpy_dtype_for_mlx(mlx_dtype: Any) -> np.dtype:
    """Convert MLX dtype to corresponding numpy dtype for storage.

    This is useful when you have an MLX array and need to store it in numpy
    format. Note that bf16 maps to fp16 for numpy storage since numpy doesn't
    have native bfloat16 support; the actual bf16 semantics are preserved in MLX.

    Args:
        mlx_dtype: MLX dtype (mx.float16, mx.bfloat16, mx.float32)

    Returns:
        Corresponding numpy dtype for storage

    Raises:
        ImportError: If MLX is not available
        ValueError: If mlx_dtype is not recognized

    Examples:
        >>> if HAS_MLX:
        ...     numpy_dtype_for_mlx(mx.float16)  # numpy.float16
        ...     numpy_dtype_for_mlx(mx.bfloat16)  # numpy.float16 (numpy storage)
        ...     numpy_dtype_for_mlx(mx.float32)  # numpy.float32
    """
    if not HAS_MLX:
        raise ImportError(
            "MLX is not available. Install with: pip install mlx"
        )
    if mlx_dtype not in _MLX_TO_NUMPY:
        raise ValueError(
            f"Unknown MLX dtype: {mlx_dtype!r}. "
            f"Valid options: {list(_MLX_TO_NUMPY.keys())}"
        )
    return _MLX_TO_NUMPY[mlx_dtype]


def mlx_dtype_to_str(mlx_dtype: Any) -> str:
    """Convert MLX dtype to string name.

    Args:
        mlx_dtype: MLX dtype (mx.float16, mx.bfloat16, mx.float32)

    Returns:
        String name ("fp16", "bf16", "fp32")

    Raises:
        ImportError: If MLX is not available
        ValueError: If mlx_dtype is not recognized
    """
    if not HAS_MLX:
        raise ImportError(
            "MLX is not available. Install with: pip install mlx"
        )
    if mlx_dtype not in _MLX_TO_STR:
        raise ValueError(
            f"Unknown MLX dtype: {mlx_dtype!r}. "
            f"Valid options: {list(_MLX_TO_STR.keys())}"
        )
    return _MLX_TO_STR[mlx_dtype]


# Global default configuration
# Can be overridden per-module or per-function call
_default_config: DTypeConfig | None = None


def get_default_config() -> DTypeConfig:
    """Get the global default dtype configuration.

    Returns a cached default configuration. The configuration uses:
    - bf16 for weights and activations (better dynamic range than fp16)
    - fp32 for accumulation (required for accuracy)
    - fp16 for scales (memory efficient, sufficient precision)
    - bf16 for KV cache (good balance)

    Returns:
        Default DTypeConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = DTypeConfig()
    return _default_config


def set_default_config(config: DTypeConfig) -> None:
    """Set the global default dtype configuration.

    Args:
        config: New default configuration
    """
    global _default_config
    _default_config = config


def reset_default_config() -> None:
    """Reset the global default configuration to factory defaults."""
    global _default_config
    _default_config = None


# Preset configurations for common use cases
def fp16_config() -> DTypeConfig:
    """Create FP16-everywhere configuration for maximum compatibility.

    Use this when:
    - Running on older hardware without BF16 support
    - Maximum compatibility is needed
    - Memory is not a primary concern
    """
    return DTypeConfig(
        weights="fp16",
        activations="fp16",
        accumulation="fp32",
        scales="fp16",
        kv_cache="fp16",
    )


def bf16_config() -> DTypeConfig:
    """Create BF16-everywhere configuration for better dynamic range.

    Use this when:
    - Running on Apple Silicon M1 Pro/Max or newer
    - Training or fine-tuning (better gradient flow)
    - Working with models that have large weight magnitudes
    """
    return DTypeConfig(
        weights="bf16",
        activations="bf16",
        accumulation="fp32",
        scales="fp16",
        kv_cache="bf16",
    )


def memory_efficient_config() -> DTypeConfig:
    """Create memory-efficient configuration for long context.

    Use this when:
    - Running long context inference (>8K tokens)
    - Memory is constrained
    - Slight accuracy loss is acceptable
    """
    return DTypeConfig(
        weights="fp16",
        activations="fp16",
        accumulation="fp32",
        scales="fp16",
        kv_cache="fp8",
    )


def high_precision_config() -> DTypeConfig:
    """Create high-precision configuration for accuracy-critical tasks.

    Use this when:
    - Maximum accuracy is required
    - Memory is not a concern
    - Running evaluation or benchmarks
    """
    return DTypeConfig(
        weights="bf16",
        activations="bf16",
        accumulation="fp32",
        scales="fp32",
        kv_cache="bf16",
    )
