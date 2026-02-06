"""Common quantization utilities for KV caches."""

from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0
INT8_MAX = 127.0
FP4_MAX = 6.0


def quantize_fp8(
    tensor: torch.Tensor,
    scale_method: str = "tensor",
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 E4M3 format (simulated with uint8)."""
    if scale_method == "channel":
        # Per-channel (last dim) scaling
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
    else:
        # Per-tensor scaling
        abs_max = tensor.abs().amax()
    
    scale = (abs_max / FP8_E4M3_MAX).clamp(min=1e-12)
    scaled = (tensor / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    
    # Map [-448, 448] to [0, 255] centered at 128
    quantized = torch.round(scaled / FP8_E4M3_MAX * 127.0) + 128.0
    quantized = quantized.clamp(0, 255).to(torch.uint8)
    
    return quantized, scale.to(scale_dtype)


def dequantize_fp8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize simulated FP8 tensor to float."""
    signed = quantized.to(torch.float32) - 128.0
    return (signed / 127.0 * FP8_E4M3_MAX * scale.to(torch.float32)).to(output_dtype)


def quantize_int8(
    tensor: torch.Tensor,
    scale_method: str = "tensor",
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 format with symmetric quantization."""
    if scale_method == "channel":
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
    else:
        abs_max = tensor.abs().amax()
        
    scale = (abs_max / INT8_MAX).clamp(min=1e-12)
    scaled = (tensor / scale).clamp(-INT8_MAX, INT8_MAX)
    quantized = torch.round(scaled).to(torch.int8)
    
    return quantized, scale.to(scale_dtype)


def dequantize_int8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize INT8 tensor to float."""
    return (quantized.to(torch.float32) * scale.to(torch.float32)).to(output_dtype)


def quantize_fp4(
    tensor: torch.Tensor,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP4 packed format."""
    abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / FP4_MAX
    scaled = (tensor / scale).clamp(-FP4_MAX, FP4_MAX)
    
    # Map to 4-bit unsigned [0, 15]
    quantized = torch.round(scaled * 2.0).to(torch.int8)
    quantized = (quantized + 8).clamp(0, 15).to(torch.uint8)
    
    batch, heads, seq, dim = tensor.shape
    reshaped = quantized.view(batch, heads, seq, dim // 8, 8)
    
    packed = torch.zeros(
        (batch, heads, seq, dim // 8),
        dtype=torch.int32,
        device=tensor.device,
    )
    for i in range(8):
        packed = packed | (reshaped[..., i].to(torch.int32) << (i * 4))
        
    return packed, scale.to(scale_dtype)


def dequantize_fp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize FP4 packed tensor to float."""
    batch, heads, seq, packed_dim = packed.shape
    dim = packed_dim * 8
    
    unpacked_list = []
    for i in range(8):
        nibble = (packed >> (i * 4)) & 0xF
        signed = nibble.to(torch.float32) - 8.0
        unpacked_list.append(signed)
        
    unpacked = torch.stack(unpacked_list, dim=-1)
    unpacked = unpacked.view(batch, heads, seq, dim)
    return (unpacked * scale.to(torch.float32) / 2.0).to(output_dtype)
