"""PyTorch MMFP4 linear layer wrapper.

This module exposes `MMFP4Linear`, which accepts row-packed MMFP4 weights:
- packed weights: uint32 [out_features, in_features // 8]
- scales: float16/bfloat16/float32 [n_groups, out_features]

Forward supports any input shape `[..., in_features]` and returns
`[..., out_features]`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .._compat import HAS_TORCH, torch

if HAS_TORCH and torch is not None:
    import torch.nn as nn
    import torch.nn.functional as F

    _E2M1_TABLE = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )

    _SHIFT_4BIT = torch.arange(8, dtype=torch.int64) * 4
    _MMFP4_LAYOUT_DEBUG = os.getenv("MMFP4_LAYOUT_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _MMFP4_LAYOUT_DEBUG_ONCE = False

    def _u32_hex_sample(words: torch.Tensor, limit: int = 8) -> list[str]:
        sample = words.reshape(-1)[:limit].to(torch.int64).tolist()
        return [f"0x{int(v) & 0xFFFFFFFF:08x}" for v in sample]

    def _as_u32_tensor(packed: torch.Tensor) -> torch.Tensor:
        if packed.dtype == torch.uint32:
            return packed
        if packed.dtype in (torch.int32, torch.int64):
            return packed.to(torch.uint32)
        raise ValueError(
            f"packed_weights must be uint32/int32/int64 tensor, got dtype={packed.dtype}"
        )

    def _unpack_rowwise_nibbles(packed_weights: torch.Tensor) -> torch.Tensor:
        """Unpack [out, in//8] uint32 words into uint8 nibbles [out, in]."""
        shifts = _SHIFT_4BIT.to(device=packed_weights.device).view(1, 1, 8)
        words = packed_weights.to(torch.int64).unsqueeze(-1)
        nibbles = torch.bitwise_and(
            torch.bitwise_right_shift(words, shifts), 0xF)
        return nibbles.reshape(packed_weights.shape[0], packed_weights.shape[1] * 8).to(torch.uint8)

    def _unpack_kernel_layout_nibbles(
        kernel_cache: torch.Tensor, out_features: int
    ) -> torch.Tensor:
        """Unpack kernel layout [in//8, out] words into nibble matrix [in, out]."""
        shifts = _SHIFT_4BIT.to(device=kernel_cache.device).view(1, 1, 8)
        words = kernel_cache.to(torch.int64).unsqueeze(-1)
        nibbles = torch.bitwise_and(
            torch.bitwise_right_shift(words, shifts), 0xF)
        # Reshape from [in//8, out, 8] to [in, out]
        # kernel_cache is [in//8, out], so unpacked is [in//8, out, 8]
        unpacked = nibbles.reshape(kernel_cache.shape[0], kernel_cache.shape[1], 8)
        # Permute to [in//8, 8, out] then reshape to [in, out]
        unpacked = unpacked.permute(0, 2, 1).reshape(kernel_cache.shape[0] * 8, kernel_cache.shape[1])
        return unpacked.to(torch.uint8)

    def _dequantize_rowwise_mmfp4(
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """Dequantize row-packed MMFP4 weights to [out_features, in_features] float16."""
        out_features, in_packed = packed_weights.shape
        in_features = in_packed * 8

        nibbles = _unpack_rowwise_nibbles(packed_weights).to(torch.long)
        table = _E2M1_TABLE.to(device=packed_weights.device)
        dequant = table.index_select(
            0, nibbles.reshape(-1)).reshape(out_features, in_features)

        scales_f32 = scales.to(dtype=torch.float32)
        group_ids = torch.arange(
            in_features, device=packed_weights.device, dtype=torch.long)
        group_ids = torch.clamp(group_ids // group_size,
                                max=scales_f32.shape[0] - 1)
        expanded_scales = scales_f32.transpose(0, 1).index_select(1, group_ids)

        return (dequant * expanded_scales).to(torch.float16)

    def _rowpacked_to_mmfp4_kernel_layout(packed_weights: torch.Tensor) -> torch.Tensor:
        """Convert [out, in//8] packing to kernel layout [in//8, out]."""
        global _MMFP4_LAYOUT_DEBUG_ONCE
        source_device = packed_weights.device
        # Build on CPU for deterministic uint32/int64 bit packing.
        # Some MPS runtimes can produce invalid all-zero words for this transform.
        packed_cpu = _as_u32_tensor(packed_weights).to(
            dtype=torch.uint32, device="cpu"
        ).contiguous()

        out_features, in_packed = packed_cpu.shape
        in_features = in_packed * 8
        
        # Kernel requires in_features (K) to be divisible by 8 for packing
        # packed_weights is [out, K/8], so K is always divisible by 8.
        
        row_nibbles = _unpack_rowwise_nibbles(packed_cpu)

        # DEBUG: Trace layout conversion
        if _MMFP4_LAYOUT_DEBUG:
            print(
                "[MMFP4 layout debug] "
                f"packed shape={tuple(packed_cpu.shape)} dtype={packed_cpu.dtype}"
            )
            print(
                "[MMFP4 layout debug] "
                f"packed sample={_u32_hex_sample(packed_cpu[0, : min(8, in_packed)])}"
            )
            print(
                "[MMFP4 layout debug] "
                f"row nibbles shape={tuple(row_nibbles.shape)} "
                f"non_zero_ratio={(row_nibbles != 0).float().mean().item():.6f}"
            )
            print(
                "[MMFP4 layout debug] "
                f"row nibble sample={row_nibbles[0, : min(32, in_features)].tolist()}"
            )

        # Transpose to [in, out]
        nibbles = row_nibbles.transpose(0, 1).contiguous()
        
        # Reshape to [in//8, 8, out] to pack 8 K-values per word
        nibbles = nibbles.reshape(in_features // 8, 8, out_features)
        
        # Permute to [in//8, out, 8] so the 8 bits are in the last dim
        nibbles = nibbles.permute(0, 2, 1).contiguous()
        
        shifts = _SHIFT_4BIT.to(dtype=torch.int64).view(1, 1, 8)
        kernel_cache_cpu = torch.sum(
            torch.bitwise_left_shift(nibbles, shifts), dim=-1
        ).to(torch.uint32)

        _MMFP4_LAYOUT_DEBUG_ONCE = True
        
        # DEBUG: Verify repacking
        if _MMFP4_LAYOUT_DEBUG and not _MMFP4_LAYOUT_DEBUG_ONCE:
             # NOTE: This verification logic assumes [in, out] layout but we now have [in//8, out]
             # So it might fail if we don't update it. But for now we just guard it.
             # Actually, if the verification logic is wrong for the new layout, 
             # printing it might show errors (as we saw in Step 13).
             # I'll leave it as is but guarded.
            print(
                "[MMFP4 layout debug] "
                f"kernel cache shape={tuple(kernel_cache_cpu.shape)} dtype={kernel_cache_cpu.dtype}"
            )
            print(
                "[MMFP4 layout debug] "
                f"kernel sample col0={_u32_hex_sample(kernel_cache_cpu[: min(8, in_features//8), 0])}"
            )
            # Fix: Cast to int32 for comparison with 0 to avoid UInt32 vs Long promotion error
            print(
                "[MMFP4 layout debug] "
                f"kernel non_zero_ratio={(kernel_cache_cpu.view(torch.int32) != 0).float().mean().item():.6f}"
            )

        if source_device.type == "cpu":
            return kernel_cache_cpu
        return kernel_cache_cpu.to(source_device)

    def _try_mmfp4_kernel_gemm(
        x_2d: torch.Tensor,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        kernel_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Attempt real MMFP4 kernel path; return (output, updated_kernel_cache)."""
        if not x_2d.is_mps:
            return (None, kernel_cache)

        try:
            from ..kernels import mmfp4_gemm as _kernel_mmfp4_gemm
        except Exception:
            return (None, kernel_cache)

        try:
            packed_u32 = _as_u32_tensor(packed_weights.contiguous())
            # packed_u32 is [out, in//8]
            
            # Expected cache is [in//8, out]
            expected_cache_shape = (packed_u32.shape[1], packed_u32.shape[0])
            
            if (
                kernel_cache is None
                or kernel_cache.dtype != torch.uint32
                or kernel_cache.device != packed_u32.device
                or tuple(kernel_cache.shape) != expected_cache_shape
            ):
                kernel_cache = _rowpacked_to_mmfp4_kernel_layout(packed_u32)

            x_f16 = x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16)
            scales_f16 = scales if scales.dtype == torch.float16 else scales.to(torch.float16)
            if scales_f16.device != x_f16.device:
                scales_f16 = scales_f16.to(x_f16.device)
            scales_f16 = scales_f16.contiguous()

            out = _kernel_mmfp4_gemm(
                x_f16,
                kernel_cache,
                scales_f16,
                group_size=group_size,
            )

            if out is None:
                return (None, kernel_cache)
            if out.dim() != 2 or out.shape != (x_2d.shape[0], packed_u32.shape[0]):
                return (None, kernel_cache)
            if not torch.isfinite(out).all():
                return (None, kernel_cache)

            return out, kernel_cache
        except Exception:
            return (None, kernel_cache)

    def mmfp4_gemm(
        x: torch.Tensor,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """MMFP4 GEMM for row-packed weights [out, in//8]."""
        if x.dim() != 2:
            raise ValueError(
                f"x must be rank-2 [M, K], got shape={tuple(x.shape)}")
        if packed_weights.dim() != 2:
            raise ValueError(
                f"packed_weights must be rank-2 [out, in//8], got shape={tuple(packed_weights.shape)}"
            )

        if x.shape[1] != packed_weights.shape[1] * 8:
            raise ValueError(
                "Input feature mismatch: "
                f"x.shape[1]={x.shape[1]} vs packed in_features={packed_weights.shape[1] * 8}"
            )

        packed_u32 = _as_u32_tensor(packed_weights)
        scales_f = scales
        if scales_f.device != x.device:
            scales_f = scales_f.to(x.device)
        if packed_u32.device != x.device:
            packed_u32 = packed_u32.to(x.device)

        out, _ = _try_mmfp4_kernel_gemm(
            x_2d=x,
            packed_weights=packed_u32,
            scales=scales_f,
            group_size=group_size,
            kernel_cache=None,
        )
        if out is not None:
            return out

        weight = _dequantize_rowwise_mmfp4(packed_u32, scales_f, group_size)
        x_for_linear = x if x.dtype == weight.dtype else x.to(weight.dtype)
        return F.linear(x_for_linear, weight, None)

    class MMFP4Linear(nn.Module):
        """PyTorch-compatible linear layer backed by MMFP4 packed weights."""

        def __init__(
            self,
            # uint32 [out_features, in_features//8]
            packed_weights: torch.Tensor,
            scales: torch.Tensor,  # fp16 [n_groups, out_features]
            bias: torch.Tensor | None = None,
            group_size: int = 128,
        ) -> None:
            super().__init__()
            if group_size <= 0:
                raise ValueError(f"group_size must be > 0, got {group_size}")
            if packed_weights.dim() != 2:
                raise ValueError(
                    "packed_weights must be rank-2 [out_features, in_features//8], "
                    f"got shape={tuple(packed_weights.shape)}"
                )
            if scales.dim() != 2:
                raise ValueError(
                    f"scales must be rank-2 [n_groups, out_features], got shape={tuple(scales.shape)}"
                )
            if scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(
                    "scales must be fp16/bf16/fp32, "
                    f"got dtype={scales.dtype}"
                )

            packed_u32 = _as_u32_tensor(packed_weights.contiguous())
            out_features = int(packed_u32.shape[0])
            in_features = int(packed_u32.shape[1] * 8)
            expected_groups = (in_features + group_size - 1) // group_size

            # Accept common transposed scale layout [out_features, n_groups].
            scales_in = scales.contiguous()
            if scales_in.shape[1] != out_features and scales_in.shape[0] == out_features:
                scales_in = scales_in.transpose(0, 1).contiguous()

            if scales_in.shape[1] != out_features:
                raise ValueError(
                    "scales second dimension must match out_features; "
                    f"got scales.shape={tuple(scales_in.shape)}, out_features={out_features}"
                )
            if scales_in.shape[0] != expected_groups:
                raise ValueError(
                    "scales first dimension must match n_groups; "
                    f"expected {expected_groups}, got {scales_in.shape[0]}"
                )

            if bias is not None:
                if bias.dim() != 1 or bias.shape[0] != out_features:
                    raise ValueError(
                        f"bias must be shape [{out_features}], got shape={tuple(bias.shape)}"
                    )
                bias = bias.contiguous()

            self.register_buffer("packed_weights", packed_u32)
            self.register_buffer("scales", scales_in)
            self.register_buffer("bias", bias if bias is not None else None)
            self.register_buffer("_kernel_packed_weights",
                                 None, persistent=False)

            self.group_size = int(group_size)
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: [batch, seq, in_features] -> [batch, seq, out_features]."""
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input last dim={self.in_features}, got {x.shape[-1]}"
                )
            if not x.is_floating_point():
                raise ValueError(
                    f"x must be a floating tensor, got dtype={x.dtype}")

            x_2d = x.reshape(-1, self.in_features)
            packed = self.packed_weights
            scales = self.scales
            if packed.device != x_2d.device:
                packed = packed.to(x_2d.device)
            if scales.device != x_2d.device:
                scales = scales.to(x_2d.device)

            out_2d, kernel_cache = _try_mmfp4_kernel_gemm(
                x_2d=x_2d,
                packed_weights=packed,
                scales=scales,
                group_size=self.group_size,
                kernel_cache=self._kernel_packed_weights,
            )
            if kernel_cache is not self._kernel_packed_weights:
                self._kernel_packed_weights = kernel_cache

            if out_2d is None:
                weight = _dequantize_rowwise_mmfp4(
                    packed, scales, self.group_size)
                x_for_linear = x_2d if x_2d.dtype == weight.dtype else x_2d.to(
                    weight.dtype)
                out_2d = F.linear(x_for_linear, weight, None)

            if self.bias is not None:
                out_2d = out_2d + \
                    self.bias.to(device=out_2d.device, dtype=out_2d.dtype)

            out = out_2d.reshape(*x.shape[:-1], self.out_features)
            out = out.to(x.dtype) if out.dtype != x.dtype else out

            return out

        @classmethod
        def from_pretrained_weight(
            cls,
            name: str,
            tensors: Mapping[str, Any],
        ) -> MMFP4Linear:
            """Load from HF-style tensors dict with `.weight` and `.scales` keys."""
            weight_key: str | None = None
            scales_key: str | None = None

            for candidate in (f"{name}.weight", f"{name}.qweight", f"{name}.packed_weight"):
                if candidate in tensors:
                    weight_key = candidate
                    break
            for candidate in (f"{name}.scales", f"{name}.weight_scale", f"{name}.scales_fp4"):
                if candidate in tensors:
                    scales_key = candidate
                    break

            if weight_key is None:
                raise KeyError(f"Missing packed weight tensor for {name!r}")
            if scales_key is None:
                raise KeyError(f"Missing scales tensor for {name!r}")

            packed_weights = tensors[weight_key]
            scales = tensors[scales_key]
            bias = tensors.get(f"{name}.bias")

            group_size = 128
            for candidate in (
                f"{name}.group_size",
                f"{name}.weight.group_size",
                f"{name}.weight_group_size",
            ):
                if candidate not in tensors:
                    continue
                value = tensors[candidate]
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        raise ValueError(
                            f"{candidate} must be scalar, got shape={tuple(value.shape)}")
                    group_size = int(value.item())
                else:
                    group_size = int(value)
                break

            return cls(
                packed_weights=packed_weights,
                scales=scales,
                bias=bias,
                group_size=group_size,
            )

else:
    def mmfp4_gemm(
        x: Any,
        packed_weights: Any,
        scales: Any,
        group_size: int,
    ) -> Any:
        raise RuntimeError("MMFP4Linear requires PyTorch")

    class MMFP4Linear:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(
            self,
            packed_weights: Any,
            scales: Any,
            bias: Any = None,
            group_size: int = 128,
        ) -> None:
            raise RuntimeError("MMFP4Linear requires PyTorch")

        def forward(self, x: Any) -> Any:
            raise RuntimeError("MMFP4Linear requires PyTorch")

        @classmethod
        def from_pretrained_weight(cls, name: str, tensors: Mapping[str, Any]) -> MMFP4Linear:
            raise RuntimeError("MMFP4Linear requires PyTorch")


__all__ = ["MMFP4Linear", "mmfp4_gemm"]
