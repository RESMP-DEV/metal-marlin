"""Unified quantization utilities for Metal Marlin.

This module provides shared quantization/dequantization implementations
used by both KVCacheTorch and MLAKVCache to avoid code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._compat import require_torch, torch

if TYPE_CHECKING:
    import torch as torch_typing


# FP8 E4M3 format constant
FP8_E4M3_MAX: float = 448.0


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to PyTorch dtype."""
    require_torch("dtype conversion")
    
    # Use a module-level cache to avoid recreating the mapping
    if not hasattr(_get_torch_dtype, "_cache"):
        _get_torch_dtype._cache = {
            "fp16": getattr(torch, "float16", torch.float32),
            "bf16": getattr(torch, "bfloat16", torch.float32),
            "fp32": getattr(torch, "float32", None),
            "fp8": getattr(torch, "float16", torch.float32),
            # Aliases
            "float16": getattr(torch, "float16", torch.float32),
            "bfloat16": getattr(torch, "bfloat16", torch.float32),
            "float32": getattr(torch, "float32", None),
        }
    
    cache = _get_torch_dtype._cache
    if dtype_str not in cache:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    
    dtype = cache[dtype_str]
    if dtype is None:
        raise RuntimeError(f"Torch dtype {dtype_str} not available in this environment")
    
    return dtype  # type: ignore[return-value]


def vectorized_pack(quantized_reshaped: torch_typing.Tensor) -> torch_typing.Tensor:
    """Vectorized pack of 8 FP4 values (last dim) into uint32.
    
    Args:
        quantized_reshaped: [..., 8] (uint8) values in 0-15
        
    Returns:
        Packed uint32 tensor [...]
    """
    require_torch("vectorized_pack")
    device = quantized_reshaped.device
    
    # Powers of 16 for shifting: 16^0, 16^1, ..., 16^7
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=device, dtype=torch.int32)
    
    # Using sum instead of bitwise_or reduction
    packed = (quantized_reshaped.to(torch.int32) << shifts).sum(dim=-1)
    
    return packed.to(torch.int32)


def vectorized_unpack(packed: torch_typing.Tensor, scale: torch_typing.Tensor) -> torch_typing.Tensor:
    """Vectorized unpack of FP4 uint32 tensor.
    
    Args:
        packed: [..., dim] (int32) packed FP4 values
        scale: [..., 1] (float16) or [..., dim] if per-channel
        
    Returns:
        Unpacked float16 tensor [..., dim * 8]
    """
    require_torch("vectorized_unpack")
    device = packed.device
    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)  # [8]
    
    # Broadcasting: packed[..., 1] >> shifts[8] -> [..., 8]
    # packed.unsqueeze(-1) is [..., dim, 1]
    # result is [..., dim, 8]
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0x0F).to(torch.float16) - 8.0
    
    # Flatten last two dimensions: [..., dim, 8] -> [..., dim*8]
    output_shape = packed.shape[:-1] + (packed.shape[-1] * 8,)
    unpacked = unpacked.view(output_shape)
    
    return unpacked * scale.float() / 2.0


def quantize_fp4(
    tensor: torch_typing.Tensor,
    scale_dtype: torch.dtype | str | None = None,
) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
    """Quantize tensor to FP4 packed format (E2M1).
    
    FP4 E2M1 format: 2-bit exponent, 1-bit mantissa, 1-bit sign.
    Packs 8 FP4 values into 1 uint32 (4 bytes).
    
    Args:
        tensor: Input tensor to quantize [..., dim]
        scale_dtype: Target dtype for scales (default: float16)
        
    Returns:
        packed_uint32: Packed uint32 tensor [..., dim // 8]
        scales_fp16: Scale factors [..., 1]
    """
    require_torch("quantize_fp4")
    
    # Compute scale based on max absolute value
    abs_max = tensor.abs().amax(dim=-1, keepdim=True)
    abs_max = torch.clamp(abs_max, min=1e-8)
    scale = abs_max / 6.0
    
    # Quantize to 4-bit indices (0-15)
    scaled = tensor / scale
    scaled = torch.clamp(scaled, -6.0, 6.0)
    quantized = torch.round(scaled * 2.0).to(torch.int8)
    quantized = torch.clamp(quantized + 8, 0, 15).to(torch.uint8)
    
    # Reshape to pack 8 nibbles per uint32
    *batch_dims, dim = tensor.shape
    reshaped = quantized.view(*batch_dims, dim // 8, 8)
    
    packed = vectorized_pack(reshaped)
    
    if scale_dtype is not None:
        if isinstance(scale_dtype, str):
            scale_dtype = _get_torch_dtype(scale_dtype)
        scale = scale.to(scale_dtype)
    else:
        scale = scale.to(torch.float16)
    
    return packed, scale


def dequantize_fp4(
    packed: torch_typing.Tensor,
    scale: torch_typing.Tensor,
) -> torch_typing.Tensor:
    """Dequantize FP4 packed tensor to float.
    
    Args:
        packed: Packed uint32 tensor [..., dim // 8]
        scale: Scale factors [..., 1]
        
    Returns:
        Dequantized float16 tensor [..., dim]
    """
    require_torch("dequantize_fp4")
    return vectorized_unpack(packed, scale)


def quantize_fp8(
    tensor: torch_typing.Tensor,
    scale_method: str = "tensor",
    scale_dtype: torch.dtype | str | None = None,
) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
    """Quantize tensor to FP8 E4M3 format.
    
    Args:
        tensor: Input tensor to quantize
        scale_method: "tensor" for per-tensor scaling or "channel" for per-channel
        scale_dtype: Target dtype for scales (default: float16)
        
    Returns:
        quantized: uint8 tensor with FP8 values
        scale: Scale factors
    """
    require_torch("quantize_fp8")
    
    if scale_method == "channel":
        abs_val = tensor.abs()
        row_max = abs_val.amax(dim=-1, keepdim=True)
        scale = torch.clamp(abs_val, max=row_max) / FP8_E4M3_MAX
        scale = torch.clamp(scale, min=1e-12)
    else:  # tensor
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / FP8_E4M3_MAX
    
    scaled = tensor / scale
    scaled = torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = torch.round(scaled / FP8_E4M3_MAX * 127.0) + 128.0
    quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
    
    if scale_dtype is not None:
        if isinstance(scale_dtype, str):
            scale_dtype = _get_torch_dtype(scale_dtype)
        scale = scale.to(scale_dtype)
    elif scale.dtype != torch.float16:
        scale = scale.to(torch.float16)
    
    return quantized, scale


def dequantize_fp8(
    quantized: torch_typing.Tensor,
    scale: torch_typing.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch_typing.Tensor:
    """Dequantize FP8 tensor to float.
    
    Args:
        quantized: uint8 tensor with FP8 values
        scale: Scale factors
        output_dtype: Target output dtype
        
    Returns:
        Dequantized tensor
    """
    require_torch("dequantize_fp8")
    signed = quantized.float() - 128.0
    return (signed / 127.0 * FP8_E4M3_MAX * scale.float()).to(output_dtype)


def quantize_int8(
    tensor: torch_typing.Tensor,
    scale_method: str = "tensor",
    scale_dtype: torch.dtype | str | None = None,
) -> tuple[torch_typing.Tensor, torch_typing.Tensor]:
    """Quantize tensor to INT8 format with symmetric quantization.
    
    Args:
        tensor: Input tensor to quantize
        scale_method: "tensor" for per-tensor scaling or "channel" for per-channel
        scale_dtype: Target dtype for scales (default: float16)
        
    Returns:
        quantized: int8 tensor
        scale: Scale factors
    """
    require_torch("quantize_int8")
    
    INT8_MAX = 127.0
    
    if scale_method == "channel":
        dim = -1
    else:
        dim = (-1,)
    
    abs_max = tensor.abs().amax(dim=dim, keepdim=True)
    abs_max = torch.clamp(abs_max, min=1e-8)
    scale = abs_max / INT8_MAX
    
    scaled = tensor / scale
    scaled = torch.clamp(scaled, -INT8_MAX, INT8_MAX)
    quantized = torch.round(scaled).to(torch.int8)
    
    if scale_dtype is not None:
        if isinstance(scale_dtype, str):
            scale_dtype = _get_torch_dtype(scale_dtype)
        scale = scale.to(scale_dtype)
    elif scale.dtype != torch.float16:
        scale = scale.to(torch.float16)
    
    return quantized, scale


def dequantize_int8(
    quantized: torch_typing.Tensor,
    scale: torch_typing.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch_typing.Tensor:
    """Dequantize INT8 tensor to float.
    
    Args:
        quantized: int8 tensor
        scale: Scale factors
        output_dtype: Target output dtype
        
    Returns:
        Dequantized tensor
    """
    require_torch("dequantize_int8")
    return (quantized.float() * scale.float()).to(output_dtype)


__all__ = [
    "FP8_E4M3_MAX",
    "quantize_fp4",
    "dequantize_fp4",
    "quantize_fp8",
    "dequantize_fp8",
    "quantize_int8",
    "dequantize_int8",
    "vectorized_pack",
    "vectorized_unpack",
    "_get_torch_dtype",
]
