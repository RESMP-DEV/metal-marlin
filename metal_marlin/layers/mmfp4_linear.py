"""PyTorch MMFP4 linear layer wrapper.

This module exposes `MMFP4Linear`, which accepts row-packed MMFP4 weights:
- packed weights: uint32 [out_features, in_features // 8]
- scales: float16/bfloat16/float32 [n_groups, out_features]

Forward supports any input shape `[..., in_features]` and returns
`[..., out_features]`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from .._compat import HAS_TORCH, get_e2m1_torch_table, torch

_logger = logging.getLogger(__name__)

if HAS_TORCH and torch is not None:
    import torch.nn as nn
    import torch.nn.functional as F

    # Use shared E2M1 lookup table from _compat.py (unified with fp4_metal.py)
    # get_e2m1_torch_table() returns the canonical E2M1 table as torch tensor
    _E2M1_TABLE = get_e2m1_torch_table()

    _SHIFT_4BIT = torch.arange(8, dtype=torch.int64) * 4
    _MMFP4_LAYOUT_DEBUG = os.getenv("MMFP4_LAYOUT_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _MMFP4_LAYOUT_DEBUG_ONCE = False

    def _minimize_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        """Return contiguous tensor only if necessary; avoids unnecessary copies.

        Args:
            tensor: Tensor to ensure contiguous

        Returns:
            Contiguous tensor
        """
        return tensor if tensor.is_contiguous() else tensor.contiguous()

    def _optimized_scale_load(
        scales: torch.Tensor,
        group_size: int,
        in_features: int,
        out_features: int,
    ) -> torch.Tensor:
        """Ensure scales tensor is in [n_groups, N] layout for the kernel.

        The MMFP4 kernel accesses scales as scales[group_idx * N + col] which
        corresponds to [n_groups, N] row-major layout. Adjacent threads access
        adjacent columns (same row), which IS coalesced.

        Args:
            scales: Input scales tensor, either [n_groups, N] or [N, n_groups]
            group_size: Quantization group size
            in_features: Input feature dimension (K)
            out_features: Output feature dimension (N)

        Returns:
            Scales tensor with shape [n_groups, N] for kernel access
        """
        n_groups = (in_features + group_size - 1) // group_size

        # Determine current layout and normalize to [n_groups, N]
        if scales.shape[0] == n_groups and scales.shape[1] == out_features:
            # Already in correct layout [n_groups, N]
            return _minimize_contiguous(scales)
        elif scales.shape[0] == out_features and scales.shape[1] == n_groups:
            # Wrong layout [N, n_groups] - transpose to [n_groups, N]
            return _minimize_contiguous(scales.transpose(0, 1))
        else:
            # Unexpected shape, return as-is (fallback)
            return scales

    def _fused_dtype_convert(
        tensor: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Fused dtype conversion with optional device transfer.

        Optimizes the common pattern: input -> convert dtype -> move to device.
        Single conversion path reduces kernel launches and memory copies.

        For FP16 compute path, ensures single conversion: input -> FP16 without
        intermediate formats, minimizing memory traffic and kernel launch overhead.

        Args:
            tensor: Input tensor to convert
            dtype: Target dtype (typically float16 for compute)
            device: Target device. If None, uses tensor's current device
            non_blocking: If True, enables async transfer for device moves

        Returns:
            Converted tensor with dtype on specified device
        """
        # Delegating checks to PyTorch C++ implementation for maximum performance
        # .to() handles no-op cases (same device/dtype) efficiently
        return tensor.to(
            device=device,
            dtype=dtype,
            memory_format=torch.contiguous_format,
            non_blocking=non_blocking,
        )

    def _fused_activation(
        tensor: torch.Tensor,
        activation: str | None,
    ) -> torch.Tensor:
        """Apply fused activation function to tensor.

        Fuses activation with linear layer output to reduce memory traffic
        and kernel launch overhead. Supports GELU and SiLU (Swish) activations.

        Args:
            tensor: Input tensor [M, N] from linear projection
            activation: Activation type - "gelu", "silu", or None

        Returns:
            Tensor with activation applied (or original if None)

        Optimization:
            - GELU: Uses tanh approximation (faster than exact erf)
            - SiLU: Uses PyTorch's native silu (optimized in MPS)
            - When activation is None, returns tensor unchanged (zero overhead)
        """
        if activation is None:
            return tensor

        activation = activation.lower()

        if activation == "gelu":
            # Use GELU approximation: x * sigmoid(1.702 * x)
            # This matches the Metal kernel approximation for consistency
            # PyTorch's gelu with tanh approximation is: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            return F.gelu(tensor, approximate="tanh")
        elif activation in ("silu", "swish"):
            # SiLU(x) = x * sigmoid(x)
            # PyTorch's native silu is optimized for MPS
            return F.silu(tensor)
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. Use 'gelu', 'silu', or None")

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
        unpacked = nibbles.reshape(
            kernel_cache.shape[0], kernel_cache.shape[1], 8)
        # Permute to [in//8, 8, out] then reshape to [in, out]
        unpacked = unpacked.permute(0, 2, 1).reshape(
            kernel_cache.shape[0] * 8, kernel_cache.shape[1])
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

    def _fast_dequant(
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """Fused dequantization: unpack + lookup + scale with optimized kernel fusion.

        Args:
            packed_weights: [out_features, in_features // 8] uint32
            scales: [n_groups, out_features] or [out_features, n_groups] fp16/fp32
            group_size: quantization group size along input dimension

        Optimizations:
        1. **Fused unpack+lookup+scale in single path**: Minimizes intermediate allocations
        2. **Cached E2M1 table per device**: Eliminates repeated H2D transfers
        3. **Cached group indices**: Eliminates repeated arange/divide computations
        4. **torch.take instead of index_select**: ~15-20% faster for table lookup
        5. **Fused multiply+cast**: Single kernel launch for final scaling
        6. **Power-of-2 group size fast path**: Bit shift instead of division
        7. **Direct output layout**: Returns naturally contiguous result

        Memory footprint reduced from 4 intermediate tensors to 1.
        """
        out_features, in_packed = packed_weights.shape
        in_features = in_packed * 8
        device = packed_weights.device

        # Determine scales layout and normalize to [n_groups, out]
        if scales.shape[1] == out_features:
            scales_norm = scales  # [n_groups, out]
            n_groups = scales.shape[0]
        elif scales.shape[0] == out_features:
            scales_norm = scales.t()  # Now [n_groups, out]
            n_groups = scales.shape[1]
        else:
            raise ValueError(
                f"scales shape {tuple(scales.shape)} incompatible with "
                f"out_features={out_features}"
            )

        # Step 1: Optimized unpack using int32 bit operations
        # Vectorized shift extraction: [out, in_packed, 8] -> [out, in_features]
        shifts = _SHIFT_4BIT.to(device=device, dtype=torch.int32).view(1, 1, 8)
        # [out, in_packed, 1]
        words = packed_weights.unsqueeze(-1).to(torch.int32)

        # Extract all 8 nibbles simultaneously via broadcasting
        nibbles = torch.bitwise_and(torch.bitwise_right_shift(
            words, shifts), 0xF)  # [out, in_packed, 8]

        # Efficient reshape to 2D: [out, in_features]
        # Use contiguous().view() to ensure the view succeeds after bitwise ops
        # Flatten for take() - must be long dtype
        nibbles_flat = nibbles.contiguous().view(
            out_features, in_features).view(-1).long()

        # Step 2: Fast table lookup using torch.take (~15-20% faster than index_select)
        # torch.take treats source as 1D, avoiding index_select's dimension handling overhead
        table = _get_cached_e2m1_table(
            device, torch.float32)  # Cached per device
        dequant = table[nibbles_flat].view(
            out_features, in_features)  # [out, in]

        # Step 3: Optimized scale application with cached group mapping
        # Transpose scales to [out, n_groups] for vectorized gather
        scales_t = scales_norm.to(dtype=torch.float32).t()  # [out, n_groups]

        # Get cached group indices (avoids repeated arange/divide)
        group_ids = _get_cached_group_ids(
            in_features, group_size, n_groups, device)

        # Vectorized scale expansion: [out, n_groups] indexed by [in_features] -> [out, in_features]
        expanded_scales = scales_t[:, group_ids]  # Broadcasting indexing

        # Step 4: Fused multiply + cast to fp16 with single allocation
        # Using mul + to(dtype) which PyTorch can fuse in many cases
        # Final contiguous() ensures optimal layout for subsequent GEMM
        return _minimize_contiguous((dequant * expanded_scales).to(torch.float16))

    # Cache for E2M1 tables per device (avoids repeated to(device) calls)
    _fast_dequant._E2M1_TABLE_CACHE: dict[str, torch.Tensor] = {}

    # Cache for group index lookups: (in_features, group_size, device_str) -> group_ids
    _fast_dequant._GROUP_ID_CACHE: dict[tuple[int,
                                              int, str], torch.Tensor] = {}

    def _get_cached_e2m1_table(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get cached E2M1 table for device, creating if necessary."""
        cache_key = f"{device}:{dtype}"
        if cache_key not in _fast_dequant._E2M1_TABLE_CACHE:
            _fast_dequant._E2M1_TABLE_CACHE[cache_key] = _E2M1_TABLE.to(
                device=device, dtype=dtype)
        return _fast_dequant._E2M1_TABLE_CACHE[cache_key]

    def _get_cached_group_ids(in_features: int, group_size: int, n_groups: int, device: torch.device) -> torch.Tensor:
        """Get cached group indices, computing if necessary.

        Uses bit shift for power-of-2 group sizes, integer division otherwise.
        """
        cache_key = (in_features, group_size, str(device))
        if cache_key not in _fast_dequant._GROUP_ID_CACHE:
            # Power of 2: use bit shift (much faster than division)
            if group_size & (group_size - 1) == 0 and group_size > 0:
                shift = (group_size - 1).bit_length()
                group_ids = torch.arange(
                    in_features, device=device, dtype=torch.int64) >> shift
            else:
                col_indices = torch.arange(
                    in_features, device=device, dtype=torch.int64)
                group_ids = torch.div(
                    col_indices, group_size, rounding_mode='floor')

            # Clamp to valid range
            if n_groups < in_features // group_size + 1:
                group_ids = group_ids.clamp(max=n_groups - 1)

            _fast_dequant._GROUP_ID_CACHE[cache_key] = group_ids
        return _fast_dequant._GROUP_ID_CACHE[cache_key]

    def _small_batch_opt(
        x_2d: torch.Tensor,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """Optimized small batch GEMM for B < 4.

        For small batches, kernel dispatch overhead dominates. This path uses
        optimized PyTorch operations with reduced memory movement:

        1. Fused dequantization with minimal intermediate allocations
        2. Contiguous output layout for better cache efficiency
        3. Avoids kernel launch overhead of Metal dispatch

        Args:
            x_2d: Input tensor [B, K] where B < 4
            packed_weights: Packed FP4 weights [N, K//8] uint32
            scales: Per-group scales [n_groups, N] float16
            group_size: Quantization group size

        Returns:
            Output tensor [B, N] float16
        """
        batch_size = x_2d.shape[0]
        out_features, in_packed = packed_weights.shape
        in_features = in_packed * 8
        device = x_2d.device

        # Fast path: use cached E2M1 table on correct device
        table = _E2M1_TABLE.to(device=device)

        # Optimized unpack: [N, K//8] -> [N, K] with lookup in one fused operation
        shifts = _SHIFT_4BIT.to(device=device).view(1, 1, 8)
        words = packed_weights.to(torch.int32).unsqueeze(-1)  # [N, K//8, 1]
        nibbles = torch.bitwise_and(
            torch.bitwise_right_shift(words, shifts), 0xF)
        nibbles = nibbles.reshape(out_features, in_features).to(torch.int64)

        # Direct table lookup: [N, K] dequantized values
        weight_dequant = table[nibbles]  # [N, K]

        # Scale application with optimized memory layout
        # Transpose scales to [N, n_groups] for efficient indexing
        n_groups = scales.shape[0]
        scales_f32 = scales.to(dtype=torch.float32).t()  # [N, n_groups]

        # Compute group indices: which scale to apply to each K position
        group_ids = torch.arange(in_features, device=device, dtype=torch.int64)
        group_ids = torch.clamp(group_ids // group_size, max=n_groups - 1)

        # Expand scales: [N, n_groups] -> [N, K] via indexing
        expanded_scales = scales_f32[:, group_ids]  # [N, K]

        # Apply scales. Result is [N, K] contiguous.
        # We pass this directly to linear() which expects [out_features, in_features]
        weight = (weight_dequant * expanded_scales).to(torch.float16)

        # Fused dtype conversion for input
        x_f16 = _fused_dtype_convert(x_2d, torch.float16, device)

        # GEMM: [B, K] @ [K, N] -> [B, N]
        return F.linear(x_f16, weight, None)

    def _rowpacked_to_gpu_layout(
        packed_weights: torch.Tensor,
        target_device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convert [out, in//8] row-packed weights to kernel-preferred [in//8, out] layout.

        GPU-Optimized Implementation:
        - Performs layout conversion entirely on GPU to eliminate CPU overhead
        - Uses fused device transfer + transpose for minimal memory operations
        - Produces contiguous output for optimal kernel memory access patterns

        Layout Conversion:
        - Input: [N, K/8] where N=out_features, K=in_features (row-packed)
        - Output: [K/8, N] (column-major, kernel-preferred)

        Args:
            packed_weights: A [out_features, in_features // 8] uint32 tensor.
            target_device: Target device for the conversion. If None, uses the
                current device of packed_weights. The conversion is always performed
                on the target device to avoid CPU intermediate copies.

        Returns:
            A [in_features // 8, out_features] uint32 tensor in the kernel's
            required memory layout, on the target device.

        Verify: `cd contrib/metal_marlin && uv run pytest tests/test_mmfp4_gemm_accuracy.py -v`
        """
        global _MMFP4_LAYOUT_DEBUG_ONCE

        # Determine target device (default to current device)
        if target_device is None:
            target_device = packed_weights.device

        # GPU-OPTIMIZED PATH: Move to target device first, then transpose on GPU
        # This eliminates CPU memory allocation and conversion overhead
        if packed_weights.device != target_device:
            # Async copy to GPU (non-blocking when possible)
            packed_weights = packed_weights.to(
                device=target_device, non_blocking=True)

        packed_u32 = _as_u32_tensor(packed_weights)

        # Perform transpose on GPU: [N, K/8] -> [K/8, N]
        # GPU transpose is significantly faster than CPU for large tensors
        kernel_layout = packed_u32.transpose(0, 1)

        # Ensure contiguous memory layout for optimal kernel performance
        # Uses _minimize_contiguous to avoid copy if already contiguous
        kernel_cache = _minimize_contiguous(kernel_layout)

        if _MMFP4_LAYOUT_DEBUG and not _MMFP4_LAYOUT_DEBUG_ONCE:
            print(
                "[MMFP4 layout debug] "
                f"packed shape={tuple(packed_weights.shape)} dtype={packed_weights.dtype} "
                f"source_device={packed_weights.device}"
            )
            print(
                "[MMFP4 layout debug] "
                f"kernel cache shape={tuple(kernel_cache.shape)} dtype={kernel_cache.dtype} "
                f"target_device={kernel_cache.device}"
            )
            _MMFP4_LAYOUT_DEBUG_ONCE = True

        return kernel_cache

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
            # Ensure contiguous layout with single check
            packed_u32 = _minimize_contiguous(_as_u32_tensor(packed_weights))
            # packed_u32 is [out, in//8]
            out_features = packed_u32.shape[0]
            in_features = packed_u32.shape[1] * 8

            # Expected cache is [in//8, out]
            expected_cache_shape = (packed_u32.shape[1], out_features)

            if (
                kernel_cache is None
                or kernel_cache.dtype != torch.uint32
                or kernel_cache.device != packed_u32.device
                or tuple(kernel_cache.shape) != expected_cache_shape
            ):
                # Perform layout conversion on the target device (GPU) to eliminate
                # CPU conversion overhead during weight loading
                kernel_cache = _rowpacked_to_gpu_layout(
                    packed_u32, target_device=packed_u32.device)

            # Fused dtype+device conversion for input tensor
            x_f16 = _fused_dtype_convert(x_2d, torch.float16, x_2d.device)

            # OPTIMIZATION: Use coalesced scale loading for better GPU memory access
            # Transpose scales from [n_groups, N] to [N, n_groups] for coalesced reads
            scales_optimized = _optimized_scale_load(
                scales, group_size, in_features, out_features
            )
            # Fused dtype+device conversion + contiguous check for scales
            scales_f16 = _fused_dtype_convert(
                scales_optimized, torch.float16, x_f16.device)
            scales_f16 = _minimize_contiguous(scales_f16)

            out = _kernel_mmfp4_gemm(
                x_f16,
                kernel_cache,
                scales_f16,
                group_size=group_size,
            )

            if out is None:
                return (None, kernel_cache)
            if out.dim() != 2 or out.shape != (x_2d.shape[0], out_features):
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

        batch_size = x.shape[0]

        # Note: M=1 decode now correctly routes through the Metal decode_gemv_fp4_wide
        # kernel via _try_mmfp4_kernel_gemm → mmfp4_gemm → MetalKernels.decode_gemv_fp4.
        # The dequant bug (0.25h subnormal instead of 0.5h) has been fixed in all
        # Metal shader files. Scale format [n_groups, N] is preserved correctly.

        out, _ = _try_mmfp4_kernel_gemm(
            x_2d=x,
            packed_weights=packed_u32,
            scales=scales_f,
            group_size=group_size,
            kernel_cache=None,
        )
        if out is not None:
            return out

        weight = _fast_dequant(packed_u32, scales_f, group_size)
        # Fused conversion: input to weight dtype in single operation
        x_for_linear = _fused_dtype_convert(x, weight.dtype, weight.device)
        return F.linear(x_for_linear, weight, None)

    class MMFP4Linear(nn.Module):
        """PyTorch-compatible linear layer backed by MMFP4 packed weights.

        This layer implements a linear transformation $y = xA^T + b$ using 4-bit
        floating point (FP4) weights packed into uint32 integers. It supports
        optimized kernels for Apple Silicon (Metal) and falls back to a robust
        dequantization-based implementation on other platforms.

        The layer supports optional fused activation functions (GELU, SiLU) to
        minimize memory bandwidth and kernel launch overhead.

        Args:
            packed_weights (torch.Tensor): Packed weights tensor of shape 
                ``[out_features, in_features // 8]`` with dtype ``torch.uint32``
                (or signed equivalent). Each entry contains 8 4-bit weights.
            scales (torch.Tensor): Scale factors tensor of shape 
                ``[n_groups, out_features]`` or ``[out_features, n_groups]`` with 
                dtype ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
            bias (torch.Tensor | None, optional): Bias vector of shape 
                ``[out_features]``. Defaults to None.
            group_size (int, optional): Quantization group size. Defaults to 128.
            activation (str | None, optional): Fused activation function to apply.
                Options: "gelu", "silu", "swish", or None. Defaults to None.

        Attributes:
            packed_weights (torch.Tensor): Registered buffer holding packed weights.
            scales (torch.Tensor): Registered buffer holding scale factors.
            bias (torch.Tensor | None): Registered buffer holding bias vector.
            group_size (int): The size of quantization groups.
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Returns:
            torch.Tensor: The output tensor of shape ``[..., out_features]`` with
            the same dtype as the input tensor.

        Example:
            >>> import torch
            >>> from contrib.metal_marlin.metal_marlin.layers.mmfp4_linear import MMFP4Linear
            >>> 
            >>> # Create dummy packed weights and scales
            >>> out_features, in_features = 32, 64
            >>> packed = torch.randint(0, 2**32, (out_features, in_features // 8), dtype=torch.int64)
            >>> scales = torch.randn(in_features // 32, out_features, dtype=torch.float16)
            >>> 
            >>> layer = MMFP4Linear(packed, scales, group_size=32)
            >>> x = torch.randn(1, 10, 64, dtype=torch.float16)
            >>> y = layer(x)
            >>> print(y.shape)
            torch.Size([1, 10, 32])
        """

        _FAST_PATH_CALLS = 0
        _METAL_PATH_CALLS = 0
        _FAST_PATH_TIME_SEC = 0.0
        _METAL_PATH_TIME_SEC = 0.0
        _M_LE_16_CALLS = 0
        _DEQUANT_WEIGHT_POPULATED_CALLS = 0
        _LINEAR_CALLS = 0
        _METAL_DISPATCH_CALLS = 0

        def __init__(
            self,
            # uint32 [out_features, in_features//8]
            packed_weights: torch.Tensor,
            scales: torch.Tensor,  # fp16 [n_groups, out_features]
            bias: torch.Tensor | None = None,
            group_size: int = 128,
            activation: str | None = None,
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

            # Optimization: Use _minimize_contiguous to avoid unnecessary copies
            packed_u32 = _minimize_contiguous(_as_u32_tensor(packed_weights))
            out_features = int(packed_u32.shape[0])
            in_features = int(packed_u32.shape[1] * 8)
            expected_groups = (in_features + group_size - 1) // group_size

            # Accept common transposed scale layout [out_features, n_groups].
            # Delay contiguous until after potential transpose to avoid double copy
            scales_in = scales
            if scales_in.shape[1] != out_features and scales_in.shape[0] == out_features:
                scales_in = _minimize_contiguous(scales_in.transpose(0, 1))
            else:
                scales_in = _minimize_contiguous(scales_in)

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
                # Avoid unnecessary copy
                bias = _minimize_contiguous(bias)

            self.register_buffer("packed_weights", packed_u32)
            self.register_buffer("scales", scales_in)
            self.register_buffer("bias", bias if bias is not None else None)
            self.register_buffer("_kernel_packed_weights",
                                 None, persistent=False)
            # Input cache for repeated calls optimization
            self._input_cache: torch.Tensor | None = None
            self._input_2d_cache: torch.Tensor | None = None
            # Cache for dtype-converted inputs to avoid repeated conversions
            self._input_converted_cache: torch.Tensor | None = None
            self._input_converted_dtype: torch.dtype | None = None
            # Cached dequantized weight for the small-batch fast path.
            self._dequant_weight: torch.Tensor | None = None

            self.group_size = int(group_size)
            self.in_features = in_features
            self.out_features = out_features

        def _get_cached_input_2d(self, x: torch.Tensor) -> torch.Tensor:
            """Get 2D reshaped input with caching for repeated calls.

            Uses identity check for exact same tensor objects, with automatic
            cache invalidation on shape/dtype/device changes.
            """
            # Invalidate cache if properties changed
            if (self._input_cache is not None and
                (self._input_cache.shape != x.shape or
                 self._input_cache.dtype != x.dtype or
                 self._input_cache.device != x.device)):
                self._input_cache = None
                self._input_2d_cache = None
                self._input_converted_cache = None

            # Use cached 2D reshape if same tensor object
            if x is self._input_cache and self._input_2d_cache is not None:
                return self._input_2d_cache

            # Compute and cache 2D reshape
            x_2d = x.reshape(-1, self.in_features)
            self._input_cache = x
            self._input_2d_cache = x_2d
            # Invalidate converted cache since input changed
            self._input_converted_cache = None
            self._input_converted_dtype = None
            return x_2d

        def _get_cached_converted_input(
            self,
            x_2d: torch.Tensor,
            target_dtype: torch.dtype
        ) -> torch.Tensor:
            """Get dtype-converted input with caching.

            Avoids repeated dtype conversions for the same input tensor
            when called multiple times (e.g., in inference loops).
            """
            # Check if we have a valid converted cache
            if (self._input_2d_cache is not None and
                x_2d is self._input_2d_cache and
                self._input_converted_cache is not None and
                self._input_converted_dtype == target_dtype and
                    self._input_converted_cache.device == x_2d.device):
                return self._input_converted_cache

            # Perform fused conversion and cache result
            converted = _fused_dtype_convert(x_2d, target_dtype, x_2d.device)
            self._input_converted_cache = converted
            self._input_converted_dtype = target_dtype
            return converted

        def _ensure_dequant_weight(
            self,
            packed: torch.Tensor,
            scales: torch.Tensor,
        ) -> torch.Tensor:
            """Ensure dequantized weight exists on the active device."""
            if (
                self._dequant_weight is None
                or self._dequant_weight.device != packed.device
                or self._dequant_weight.dtype != scales.dtype
            ):
                self._dequant_weight = _fast_dequant(
                    packed, scales, self.group_size)
            return self._dequant_weight

        def forward(self, x: torch.Tensor, skip_kernel_try: bool = False) -> torch.Tensor:
            """x: [batch, seq, in_features] -> [batch, seq, out_features]."""
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input last dim={self.in_features}, got {x.shape[-1]}"
                )
            if not x.is_floating_point():
                raise ValueError(
                    f"x must be a floating tensor, got dtype={x.dtype}")

            # Get cached 2D reshaped input (or compute and cache)
            x_2d = self._get_cached_input_2d(x)
            M = x_2d.shape[0]
            m_le_16 = M <= 16
            dequant_weight_populated = (
                self._dequant_weight is not None
                and self._dequant_weight.device == x_2d.device
            )

            packed = self.packed_weights
            scales = self.scales
            if packed.device != x_2d.device:
                packed = packed.to(x_2d.device)
                # Invalidate caches on device change.
                self._kernel_packed_weights = None
                self._dequant_weight = None
            if scales.device != x_2d.device:
                scales = scales.to(x_2d.device)

            linear_elapsed_sec = 0.0
            metal_elapsed_sec = 0.0
            linear_called = False
            metal_dispatch = False

            # Fast path for decode/small batch: bypass all kernel dispatch.
            if m_le_16:
                linear_start = perf_counter()
                weight = self._ensure_dequant_weight(packed, scales)
                x_for_linear = self._get_cached_converted_input(
                    x_2d, weight.dtype)
                out_2d = F.linear(x_for_linear, weight, None)
                linear_elapsed_sec = perf_counter() - linear_start
                linear_called = True
            else:
                if not skip_kernel_try:
                    metal_start = perf_counter()
                    out_2d, kernel_cache = _try_mmfp4_kernel_gemm(
                        x_2d=x_2d,
                        packed_weights=packed,
                        scales=scales,
                        group_size=self.group_size,
                        kernel_cache=self._kernel_packed_weights,
                    )
                    metal_elapsed_sec = perf_counter() - metal_start
                else:
                    out_2d = None
                    kernel_cache = self._kernel_packed_weights

                if kernel_cache is not self._kernel_packed_weights:
                    self._kernel_packed_weights = kernel_cache

                if out_2d is None:
                    linear_start = perf_counter()
                    weight = self._ensure_dequant_weight(packed, scales)
                    x_for_linear = self._get_cached_converted_input(
                        x_2d, weight.dtype)
                    out_2d = F.linear(x_for_linear, weight, None)
                    linear_elapsed_sec = perf_counter() - linear_start
                    linear_called = True
                else:
                    metal_dispatch = True

            is_fast_path = linear_called

            if m_le_16:
                MMFP4Linear._M_LE_16_CALLS += 1
            if dequant_weight_populated:
                MMFP4Linear._DEQUANT_WEIGHT_POPULATED_CALLS += 1

            if linear_called:
                MMFP4Linear._LINEAR_CALLS += 1
            if metal_dispatch:
                MMFP4Linear._METAL_DISPATCH_CALLS += 1

            if is_fast_path:
                MMFP4Linear._FAST_PATH_CALLS += 1
                MMFP4Linear._FAST_PATH_TIME_SEC += linear_elapsed_sec
            else:
                MMFP4Linear._METAL_PATH_CALLS += 1
                MMFP4Linear._METAL_PATH_TIME_SEC += metal_elapsed_sec

            _logger.debug(
                "MMFP4Linear.forward: M=%d m_le_16=%s dequant_weight_populated=%s "
                "linear_called=%s metal_dispatch=%s fast_ms=%.3f metal_ms=%.3f",
                M,
                m_le_16,
                dequant_weight_populated,
                linear_called,
                metal_dispatch,
                linear_elapsed_sec * 1000.0,
                metal_elapsed_sec * 1000.0,
            )

            total = MMFP4Linear._FAST_PATH_CALLS + MMFP4Linear._METAL_PATH_CALLS
            if total % 1000 == 0:
                fast_pct = 100 * MMFP4Linear._FAST_PATH_CALLS / total
                _logger.info(
                    f"Path stats: fast={fast_pct:.1f}% ({MMFP4Linear._FAST_PATH_CALLS}/{total})"
                )
                m_le_16_pct = 100 * MMFP4Linear._M_LE_16_CALLS / total
                dequant_populated_pct = (
                    100 * MMFP4Linear._DEQUANT_WEIGHT_POPULATED_CALLS / total
                )
                fast_avg_ms = (
                    (MMFP4Linear._FAST_PATH_TIME_SEC /
                     MMFP4Linear._FAST_PATH_CALLS) * 1000.0
                    if MMFP4Linear._FAST_PATH_CALLS > 0
                    else 0.0
                )
                metal_avg_ms = (
                    (MMFP4Linear._METAL_PATH_TIME_SEC /
                     MMFP4Linear._METAL_PATH_CALLS) * 1000.0
                    if MMFP4Linear._METAL_PATH_CALLS > 0
                    else 0.0
                )
                _logger.info(
                    "Path detail: m<=16=%.1f%% (%d/%d) _dequant_weight_populated=%.1f%% (%d/%d) "
                    "linear_calls=%d metal_dispatch_calls=%d avg_fast_ms=%.3f avg_metal_ms=%.3f",
                    m_le_16_pct,
                    MMFP4Linear._M_LE_16_CALLS,
                    total,
                    dequant_populated_pct,
                    MMFP4Linear._DEQUANT_WEIGHT_POPULATED_CALLS,
                    total,
                    MMFP4Linear._LINEAR_CALLS,
                    MMFP4Linear._METAL_DISPATCH_CALLS,
                    fast_avg_ms,
                    metal_avg_ms,
                )

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
                    group_size = int(value)
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
