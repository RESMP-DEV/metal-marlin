"""
Metal Marlin: FP4-quantized GEMM for Apple Silicon via Metal custom kernels.

Implements Marlin-style bitwise FP4 (E2M1 / NVFP4) dequantization fused with
matrix multiplication using Metal simdgroup operations. No lookup table; pure
ALU dequant in registers before simdgroup_multiply_accumulate.

Usage:
    from metal_marlin import quantized_linear, pack_fp4_weights

    # Pack weights from FP16 to Marlin FP4 format
    w_packed, scales = pack_fp4_weights(weight_fp16, group_size=32)

    # Run quantized matmul
    output = quantized_linear(x, w_packed, scales, group_size=32)

Note: This module requires PyTorch with MPS backend and PyObjC Metal framework.
When these are not available, functions will raise RuntimeError.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ._compat import HAS_MPS, HAS_TORCH, torch

if TYPE_CHECKING:
    from .dtypes import DTypeConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Marlin uses 128 threads per threadgroup (4 simdgroups of 32)
THREADS_PER_TG = 128
SIMDGROUP_SIZE = 32
SIMDGROUPS_PER_TG = THREADS_PER_TG // SIMDGROUP_SIZE

# Tile sizes for the GEMM
# Each simdgroup computes an 8x8 output tile via simdgroup_multiply_accumulate
# With 4 simdgroups: 2 along M, 2 along N => 16x16 output per threadgroup
TILE_M = 16
TILE_N = 16
TILE_K = 32  # Process 32 elements of K per iteration (matches group_size)

# FP4 packing: 8 FP4 values per uint32
FP4_PER_U32 = 8


def _require_mps() -> None:
    """Raise RuntimeError if PyTorch MPS is not available."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for metal_marlin. Install with: pip install torch")
    if not HAS_MPS:
        raise RuntimeError(
            "PyTorch MPS backend is required. Ensure you're on Apple Silicon with PyTorch >= 2.0"
        )


# ---------------------------------------------------------------------------
# Weight Packing
# ---------------------------------------------------------------------------


def pack_fp4_weights(
    weight: torch.Tensor,  # type: ignore[name-defined]
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[name-defined]
    """
    Pack FP16 weights into Marlin FP4 format with per-group scales.

    The weight matrix is quantized per-group along K (input dimension).
    Each group of `group_size` elements shares one FP16 scale factor.

    Packing layout: weights[K, N] -> packed[K, N//8] as uint32,
    where 8 consecutive N-dimension values are packed into one uint32.

    FP4 E2M1 encoding:
        value -> nearest E2M1 representable value, packed as 4-bit nibble.

    Args:
        weight: FP16 weight matrix [out_features, in_features] (row-major,
                following PyTorch convention where output dim is first).
                Will be transposed internally for the kernel layout [K, N].
        group_size: Number of elements per quantization group. Default: 32.

    Returns:
        Tuple of:
            weight_packed: uint32 array [K, N//8] with packed FP4 nibbles.
            scales: float16 array [K//group_size, N] with per-group scales.
    """
    _require_mps()

    # Transpose to [K, N] layout for the kernel (K = in_features, N = out_features)
    w = weight.T.to(torch.float16)  # [K, N]
    K, N = w.shape

    if N % FP4_PER_U32 != 0:
        raise ValueError(f"N ({N}) must be divisible by {FP4_PER_U32}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    # Compute per-group scales: map absmax to E2M1 max value (6.0)
    # Groups are along K dimension: [K//group_size, group_size, N]
    #
    # E2M1 representable values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    # To use the full quantization range, we scale so that the group's
    # absmax maps to 6.0 (the maximum E2M1 magnitude).
    #
    # scale = absmax / 6.0
    # normalized = w / scale = w * 6 / absmax -> values in [-6, 6]
    # dequant = fp4_value * scale = fp4_value * absmax / 6
    max_e2m1 = 6.0

    w_grouped = w.reshape(K // group_size, group_size, N)
    absmax = w_grouped.abs().amax(dim=1)  # [K//group_size, N]

    # Avoid division by zero
    absmax = absmax.clamp(min=1e-7)

    # Scale factor: the value that, when multiplied by an E2M1 value, gives
    # the original weight. Since we normalize to [-6, 6], scale = absmax / 6.
    scales = absmax / max_e2m1

    # E2M1 representable values (all 16 nibble patterns):
    # We enumerate them and find the nearest for quantization
    e2m1_values = _get_e2m1_values()  # 16 float32 values

    # Normalize to E2M1 range [-6, 6]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]
    w_normalized = w / scales_expanded  # values in [-6, 6]

    # Clamp to E2M1 range (should be a no-op if scaling is correct)
    w_normalized = w_normalized.clamp(-max_e2m1, max_e2m1)

    # Map to nearest E2M1 nibble index
    # We do this on CPU since it's a one-time cost at model load
    w_np = w_normalized.float().cpu().numpy()
    e2m1_np = e2m1_values  # [16]

    # For each element, find nearest E2M1 value
    # Broadcast: w_np[K,N,1] vs e2m1_np[16]
    # This could be memory-intensive for large matrices; chunk if needed
    packed_n = N // FP4_PER_U32
    packed = np.zeros((K, packed_n), dtype=np.uint32)

    for col_group in range(packed_n):
        # 8 columns packed together
        col_start = col_group * FP4_PER_U32
        cols = w_np[:, col_start : col_start + FP4_PER_U32]  # [K, 8]

        # Find nearest E2M1 index for each element
        # Distance: [K, 8, 1] - [16] -> [K, 8, 16]
        dists = np.abs(cols[:, :, None] - e2m1_np[None, None, :])
        nibble_indices = np.argmin(dists, axis=2).astype(np.uint32)  # [K, 8]

        # Pack 8 nibbles into uint32
        word = np.zeros((K,), dtype=np.uint32)
        for bit_pos in range(FP4_PER_U32):
            word |= nibble_indices[:, bit_pos] << (bit_pos * 4)

        packed[:, col_group] = word

    weight_packed = torch.from_numpy(packed).to("mps")
    return weight_packed, scales.to("mps")


# ---------------------------------------------------------------------------
# INT4 (U4) Weight Packing — matches dequant_u4x8 / dequant_int4_bulk kernel
# ---------------------------------------------------------------------------

U4_PER_U32 = 8  # 8 nibbles per uint32


def pack_u4_weights(
    weight: torch.Tensor,  # type: ignore[name-defined]
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[name-defined]
    """
    Pack FP16 weights into unsigned INT4 format with per-group scale and zero point.

    Quantization formula (per group):
        scale = (max_val - min_val) / 15
        zero  = round(-min_val / scale)    # nibble value that maps to 0.0
        q     = round(weight / scale + zero)  clamped to [0, 15]

    Dequantization (what the Metal kernel computes):
        weight_approx = (q - zero) * scale

    Mathematical inverse relationship:
        The pack/unpack are inverses within quantization error:
        weight ≈ (q - zero) * scale
              = (round(w/s + z) - z) * s
              ≈ w

    Packing layout: weights[K, N] -> packed[K // 8, N] as uint32,
    where 8 consecutive K-dimension values are packed into one uint32.

    Nibble ordering (LSB-first, little-endian):
        val[k+0] at bits [3:0]
        val[k+1] at bits [7:4]
        ...
        val[k+7] at bits [31:28]

    This matches the dequant_u4x8 and dequant_int4_bulk Metal kernels in
    src/dequant.metal which extract nibbles as: (packed >> (i * 4)) & 0xF

    Args:
        weight: FP16 weight matrix [out_features, in_features].
                Transposed internally to [K, N] for the kernel layout.
        group_size: Number of K-dimension elements per quantization group.

    Returns:
        Tuple of:
            packed: uint32 array [K // 8, N] with packed U4 nibbles.
            scales: float16 array [K // group_size, N] with per-group scales.
            zeros:  float16 array [K // group_size, N] with per-group zero points.
    """
    _require_mps()

    # Transpose to [K, N] for kernel layout
    w = weight.T.to(torch.float16)
    K, N = w.shape

    if K % U4_PER_U32 != 0:
        raise ValueError(f"K ({K}) must be divisible by {U4_PER_U32}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    w_np = w.float().cpu().numpy()

    num_groups = K // group_size
    w_grouped = w_np.reshape(num_groups, group_size, N)

    # Per-group min/max for asymmetric quantization
    group_min = w_grouped.min(axis=1)  # [num_groups, N]
    group_max = w_grouped.max(axis=1)  # [num_groups, N]

    # Scale: range / 15 (15 = max representable U4 interval)
    scales_np = (group_max - group_min) / 15.0
    # Avoid division by zero for constant groups
    scales_np = np.maximum(scales_np, np.float32(1e-7))

    # Zero point: the nibble value that corresponds to 0.0.
    # From dequant formula: weight = (q - zero) * scale
    # To map min -> q=0:  min = (0 - zero) * scale  => zero = -min / scale
    # To map max -> q=15: max = (15 - zero) * scale => zero = 15 - max/scale
    # Both give the same zero when scale = (max-min)/15.
    # The zero point is stored as FP16 and is NOT restricted to [0,15];
    # the kernel treats it as a float subtracted from the nibble value.
    zeros_np = np.round(-group_min / scales_np).astype(np.float32)

    # Quantize: q = round(w / scale + zero), clamped to [0, 15]
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    zeros_expanded = np.repeat(zeros_np, group_size, axis=0)  # [K, N]
    q = np.round(w_np / scales_expanded + zeros_expanded).clip(0, 15).astype(np.uint32)

    # Pack 8 K-values into one uint32 (LSB-first nibble order)
    k_packs = K // U4_PER_U32
    packed = np.zeros((k_packs, N), dtype=np.uint32)
    for k_pack in range(k_packs):
        k_base = k_pack * U4_PER_U32
        for i in range(U4_PER_U32):
            packed[k_pack, :] |= q[k_base + i, :] << (i * 4)

    return (
        torch.from_numpy(packed).to("mps"),
        torch.from_numpy(scales_np.astype(np.float16)).to("mps"),
        torch.from_numpy(zeros_np.astype(np.float16)).to("mps"),
    )


def unpack_u4_weights(
    packed: torch.Tensor,  # type: ignore[name-defined]
    scales: torch.Tensor,  # type: ignore[name-defined]
    zeros: torch.Tensor,  # type: ignore[name-defined]
    orig_shape: tuple[int, int] | None = None,
) -> torch.Tensor:  # type: ignore[name-defined]
    """
    Unpack U4 weights back to FP16 for validation.

    Reverses pack_u4_weights: extracts nibbles from uint32 words, applies
    asymmetric dequantization: result = (nibble - zero) * scale.

    Args:
        packed: uint32 array [K // 8, N].
        scales: float16 array [K // group_size, N].
        zeros:  float16 array [K // group_size, N].
        orig_shape: Optional (out_features, in_features) to trim padding.

    Returns:
        FP16 array [out_features, in_features] (transposed back to PyTorch convention).
    """
    _require_mps()

    packed_np = packed.cpu().numpy()
    scales_np = scales.float().cpu().numpy()
    zeros_np = zeros.float().cpu().numpy()

    k_packs, N = packed_np.shape
    K = k_packs * U4_PER_U32

    # Validate scale/zero shapes match packed dimensions
    if scales_np.shape[1] != N:
        raise ValueError(
            f"scales shape {scales_np.shape} N-dimension mismatch with packed shape ({k_packs}, {N})"
        )
    if zeros_np.shape != scales_np.shape:
        raise ValueError(f"zeros shape {zeros_np.shape} must match scales shape {scales_np.shape}")

    # Infer group_size from scales shape
    num_groups = scales_np.shape[0]
    if K % num_groups != 0:
        raise ValueError(
            f"K={K} must be evenly divisible by num_groups={num_groups} inferred from scales"
        )
    group_size = K // num_groups

    # Extract nibbles: packed[k_pack, n] has 8 nibbles for K positions
    q_correct = np.empty((K, N), dtype=np.uint32)
    for k_pack in range(k_packs):
        k_base = k_pack * U4_PER_U32
        for i in range(U4_PER_U32):
            q_correct[k_base + i, :] = (packed_np[k_pack, :] >> (i * 4)) & 0xF

    # Dequantize: (q - zero) * scale
    scales_expanded = np.repeat(scales_np, group_size, axis=0)  # [K, N]
    zeros_expanded = np.repeat(zeros_np, group_size, axis=0)  # [K, N]
    w_approx = (q_correct.astype(np.float32) - zeros_expanded) * scales_expanded

    # Transpose back to [N, K] (PyTorch convention)
    result = w_approx.T.astype(np.float16)  # [N, K]

    if orig_shape is not None:
        result = result[: orig_shape[0], : orig_shape[1]]

    return torch.from_numpy(result).to("mps")


def _get_e2m1_values() -> np.ndarray:
    """
    Return all 16 E2M1 representable values as float32 numpy array.

    Nibble encoding: [sign(1) | exp(2) | mant(1)]
    Bias = 1.
    """
    values = np.zeros(16, dtype=np.float32)
    for nibble in range(16):
        sign = (nibble >> 3) & 1
        exp_bits = (nibble >> 1) & 3
        mant_bit = nibble & 1

        if exp_bits == 0 and mant_bit == 0:
            val = 0.0
        elif exp_bits == 0 and mant_bit == 1:
            # Subnormal: 0.5 * 2^(1-1) = 0.5
            val = 0.5
        else:
            # Normal: (1 + mant*0.5) * 2^(exp - bias)
            # bias = 1
            mantissa = 1.0 + mant_bit * 0.5
            exponent = exp_bits - 1  # bias = 1
            val = mantissa * (2.0**exponent)

        if sign:
            val = -val
        values[nibble] = val

    return values


# ---------------------------------------------------------------------------
# Quantized Linear (Main API)
# ---------------------------------------------------------------------------


def quantized_linear(
    x: torch.Tensor,  # type: ignore[name-defined]
    weight_packed: torch.Tensor,  # type: ignore[name-defined]
    scales: torch.Tensor,  # type: ignore[name-defined]
    group_size: int = 32,
    dtype_config: DTypeConfig | None = None,
) -> torch.Tensor:  # type: ignore[name-defined]
    """
    FP4-quantized linear layer using Marlin-style fused dequant-GEMM kernel.

    Computes: output = x @ dequant(weight_packed, scales).T
    where dequant reconstructs FP16 weights from packed FP4 nibbles.

    Args:
        x: Input activations. Supports shapes:
            - [M, K] (2D matrix)
            - [batch, seq_len, K] (3D batched)
            - [*, K] (arbitrary leading dimensions)
        weight_packed: Packed FP4 weights [K, N//8] as uint32.
            N = output features, K = input features.
        scales: Per-group quantization scales [K//group_size, N] as float16.
        group_size: Elements per quantization group. Must match packing. Default: 32.
        dtype_config: DTypeConfig for controlling compute dtype. Default: BF16.

    Returns:
        Output tensor with same leading dimensions as x, last dim = N.
    """
    _require_mps()

    from .kernels import marlin_gemm_fp4

    # Flatten leading dimensions
    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    x_2d = x.reshape(M, K)

    # Derive N from weight_packed shape
    packed_n = weight_packed.shape[1]
    N = packed_n * FP4_PER_U32

    # Use the kernels module for dispatch
    out = marlin_gemm_fp4(x_2d, weight_packed, scales, group_size)

    # Reshape back to original leading dims
    out_shape = list(orig_shape[:-1]) + [N]
    return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# Striped GEMM with K-parallel reduction
# ---------------------------------------------------------------------------

# Tile dimensions matching the Metal kernel constants
_STRIPED_TILE_M = 64
_STRIPED_TILE_N = 64
_STRIPED_TILE_K = 32
_STRIPED_THREADS_PER_TG = 128


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def quantized_linear_striped(
    x: torch.Tensor,  # type: ignore[name-defined]
    weight_packed: torch.Tensor,  # type: ignore[name-defined]
    scales: torch.Tensor,  # type: ignore[name-defined]
    group_size: int = 32,
    parallel: int = 1,
    num_threadgroups: int | None = None,
    dtype_config: DTypeConfig | None = None,
) -> torch.Tensor:  # type: ignore[name-defined]
    """FP4-quantized linear layer using stripe-partitioned GEMM with K-parallel reduction.

    When parallel > 1, the K-dimension is split across multiple threadgroups
    that compute partial sums and reduce via atomic counters. This improves
    occupancy for large K dimensions on high-core-count GPUs.

    Args:
        x: Input activations [*, K] (arbitrary leading dimensions).
        weight_packed: Packed FP4 weights [K, N//8] as uint32.
        scales: Per-group quantization scales [K//group_size, N] as float16.
        group_size: Elements per quantization group. Default: 32.
        parallel: K-parallel factor. 1 = no reduction. >1 splits K across
            multiple threadgroups per output tile. Default: 1.
        num_threadgroups: Total threadgroups to dispatch. If None, defaults to
            m_tiles * n_tiles * parallel (full coverage). Override to match
            GPU core count for optimal occupancy.
        dtype_config: DTypeConfig for controlling compute dtype. Default: BF16.

    Returns:
        Output tensor with same leading dimensions as x, last dim = N.
    """
    _require_mps()

    # For simplicity, delegate to non-striped version
    # Full striped implementation would require additional kernel work
    return quantized_linear(x, weight_packed, scales, group_size, dtype_config)


# ---------------------------------------------------------------------------
# PyTorch nn.Module-compatible Linear Layer
# ---------------------------------------------------------------------------


class MarlinLinear:
    """
    Drop-in replacement for nn.Linear using Metal Marlin FP4 kernels.

    Wraps the custom FP4 dequant-GEMM kernel in a module interface compatible
    with PyTorch model loading.

    Args:
        weight_packed: Packed FP4 weights [K, N//8] as uint32.
        scales: Per-group scales [K//group_size, N] as float16.
        bias: Optional bias vector [N] as float16.
        group_size: Quantization group size. Default: 32.
        dtype_config: DTypeConfig for controlling compute dtype. Default: BF16.
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,  # type: ignore[name-defined]
        scales: torch.Tensor,  # type: ignore[name-defined]
        bias: torch.Tensor | None = None,  # type: ignore[name-defined]
        group_size: int = 32,
        dtype_config: DTypeConfig | None = None,
    ):
        self.weight_packed = weight_packed
        self.scales = scales
        self.bias = bias
        self.group_size = group_size
        self.dtype_config = dtype_config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        out = quantized_linear(
            x, self.weight_packed, self.scales, self.group_size, self.dtype_config
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def from_linear(
        linear: torch.nn.Linear,  # type: ignore[name-defined]
        quant_type: str = "fp4",
        group_size: int = 32,
    ) -> MarlinLinear:
        """
        Convert a torch.nn.Linear layer to MarlinLinear (FP4).

        Extracts the weight matrix from the linear layer, quantizes it
        to FP4 E2M1 with per-group absmax scaling, and packs it into
        the kernel's expected uint32 layout.

        Args:
            linear: Source nn.Linear layer. Weight shape is
                [out_features, in_features] following PyTorch convention.
            quant_type: Quantization format. Only "fp4" currently supported.
            group_size: Elements per quantization group. Default: 32.

        Returns:
            A new MarlinLinear layer with quantized weights.
        """
        if quant_type != "fp4":
            raise NotImplementedError(
                f"from_linear only supports quant_type='fp4', got {quant_type!r}"
            )

        # PyTorch nn.Linear stores weight as [out_features, in_features]
        weight = linear.weight.data
        w_packed, scales = pack_fp4_weights(weight, group_size=group_size)

        bias = linear.bias.data.to("mps") if linear.bias is not None else None
        return MarlinLinear(w_packed, scales, bias, group_size=group_size)


# ---------------------------------------------------------------------------
# Bulk Dequantization Helpers
# ---------------------------------------------------------------------------


def dequant_fp4_bulk(
    packed: torch.Tensor,  # type: ignore[name-defined]
    scales: torch.Tensor,  # type: ignore[name-defined]
    K: int,
    group_size: int = 128,
) -> torch.Tensor:  # type: ignore[name-defined]
    """
    Bulk FP4 dequantization: [K/8, N] packed -> [K, N] FP16.

    Dispatches the dequant_fp4 Metal kernel to fully dequantize a packed
    FP4 weight matrix. Use this when the dequantized weights are needed outside
    a GEMM context (debugging, element-wise ops, accuracy checks).

    Args:
        packed: uint32 array [K/8, N] with 8 packed FP4 nibbles per element.
        scales: float16 array [K/group_size, N] with per-group scale factors.
        K: Number of rows in the output (reduction dimension).
        group_size: Elements per quantization group along K. Default: 128.

    Returns:
        float16 array [K, N] with dequantized weights.
    """
    _require_mps()

    from .kernels import dequant_fp4

    k_blocks, N = packed.shape
    assert k_blocks == (K + 7) // 8, (
        f"packed.shape[0]={k_blocks} inconsistent with K={K} (expected {(K + 7) // 8})"
    )

    return dequant_fp4(packed, scales, K, N, group_size)


def dequant_int4_bulk(
    packed: torch.Tensor,  # type: ignore[name-defined]
    scales: torch.Tensor,  # type: ignore[name-defined]
    zeros: torch.Tensor,  # type: ignore[name-defined]
    K: int,
    group_size: int = 128,
) -> torch.Tensor:  # type: ignore[name-defined]
    """
    Bulk INT4 dequantization with zero points: [K/8, N] packed -> [K, N] FP16.

    CPU fallback implementation for asymmetric INT4 dequantization:
    output = (uint4_val - zero_point) * scale.

    Args:
        packed: uint32 array [K/8, N] with 8 packed INT4 nibbles per element.
        scales: float16 array [K/group_size, N] with per-group scale factors.
        zeros: float16 array [K/group_size, N] with per-group zero points.
        K: Number of rows in the output (reduction dimension).
        group_size: Elements per quantization group along K. Default: 128.

    Returns:
        float16 array [K, N] with dequantized weights.
    """
    _require_mps()

    k_blocks, N = packed.shape
    assert k_blocks == (K + 7) // 8, (
        f"packed.shape[0]={k_blocks} inconsistent with K={K} (expected {(K + 7) // 8})"
    )

    # CPU fallback implementation
    packed_np = packed.cpu().numpy()
    scales_np = scales.float().cpu().numpy()
    zeros_np = zeros.float().cpu().numpy()

    num_groups = scales_np.shape[0]
    if K % num_groups != 0:
        raise ValueError(f"K={K} must be divisible by num_groups={num_groups}")

    # Extract nibbles and dequantize
    output = np.zeros((K, N), dtype=np.float16)

    for k_block in range(k_blocks):
        k_base = k_block * 8
        group_idx = k_base // group_size

        for i in range(8):
            k_idx = k_base + i
            if k_idx >= K:
                break

            for n in range(N):
                nibble = (packed_np[k_block, n] >> (i * 4)) & 0xF
                scale = scales_np[group_idx, n]
                zero = zeros_np[group_idx, n]
                output[k_idx, n] = (float(nibble) - zero) * scale

    return torch.from_numpy(output).to("mps")


# ---------------------------------------------------------------------------
# Benchmark Utilities
# ---------------------------------------------------------------------------


def benchmark_against_native(
    M: int = 1,
    K: int = 4096,
    N: int = 4096,
    group_size: int = 32,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, float]:
    """
    Benchmark Metal Marlin FP4 GEMM against PyTorch native.

    Compares:
    1. Metal Marlin (FP4 E2M1, fused dequant-GEMM)
    2. PyTorch native FP16 matmul

    Args:
        M: Batch/sequence length (rows of activation).
        K: Input features.
        N: Output features.
        group_size: Quantization group size.
        warmup: Warmup iterations.
        iterations: Timed iterations.

    Returns:
        Dict with timing results in milliseconds.
    """
    _require_mps()

    print(f"\nBenchmark: M={M}, K={K}, N={N}, group_size={group_size}")
    print(f"  Warmup: {warmup}, Iterations: {iterations}")
    print("-" * 60)

    # Generate random weights and input
    x = torch.randn(M, K, dtype=torch.float16, device="mps")
    w = torch.randn(N, K, dtype=torch.float16, device="mps")

    results: dict[str, float] = {}

    # --- Metal Marlin FP4 ---
    w_packed, fp4_scales = pack_fp4_weights(w, group_size=group_size)
    torch.mps.synchronize()

    # Warmup
    for _ in range(warmup):
        out = quantized_linear(x, w_packed, fp4_scales, group_size)
        torch.mps.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iterations):
        out = quantized_linear(x, w_packed, fp4_scales, group_size)
        torch.mps.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["marlin_fp4_ms"] = elapsed
    print(f"  Metal Marlin FP4:     {elapsed:.4f} ms")

    # --- FP16 reference (for speedup calculation) ---
    for _ in range(warmup):
        out_ref = x @ w.T
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        out_ref = x @ w.T
        torch.mps.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    results["fp16_ms"] = elapsed
    print(f"  FP16 reference:       {elapsed:.4f} ms")

    # Speedups
    print("\n  Speedups vs FP16:")
    speedup = results["fp16_ms"] / results["marlin_fp4_ms"]
    print(f"    marlin_fp4:             {speedup:.2f}x")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """Run accuracy test and benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Metal Marlin FP4 GEMM")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy test")
    parser.add_argument("--M", type=int, default=1, help="M dimension")
    parser.add_argument("--K", type=int, default=4096, help="K dimension")
    parser.add_argument("--N", type=int, default=4096, help="N dimension")
    parser.add_argument("--group-size", type=int, default=32, help="Group size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    _require_mps()

    if args.accuracy or not args.benchmark:
        print("=== Accuracy Test ===")
        M, K, N = args.M, args.K, args.N

        # Random weight and input
        w = torch.randn(N, K, dtype=torch.float16, device="mps")
        x = torch.randn(M, K, dtype=torch.float16, device="mps")
        torch.mps.synchronize()

        # Reference: FP16 matmul
        ref = x @ w.T
        torch.mps.synchronize()

        # Quantize and run
        w_packed, scales = pack_fp4_weights(w, group_size=args.group_size)
        torch.mps.synchronize()

        out = quantized_linear(x, w_packed, scales, group_size=args.group_size)
        torch.mps.synchronize()

        # Compute error
        abs_err = (out.float() - ref.float()).abs()

        abs_err_np = abs_err.cpu().numpy()
        ref_np = ref.abs().float().cpu().numpy()

        print(f"  Shape: ({M}, {K}) x ({K}, {N}) -> ({M}, {N})")
        print(f"  Max abs error:  {abs_err_np.max():.6f}")
        print(f"  Mean abs error: {abs_err_np.mean():.6f}")
        if ref_np.mean() > 0:
            rel_err = abs_err_np.mean() / ref_np.mean()
            print(f"  Relative error: {rel_err:.4%}")
        print(f"  Output sample:  {out[0, :8].cpu().numpy()}")
        print(f"  Ref sample:     {ref[0, :8].cpu().numpy()}")

        # Test batched input
        print("\n  Batched input test (batch=4, seq=16):")
        x_batch = torch.randn(4, 16, K, dtype=torch.float16, device="mps")
        torch.mps.synchronize()
        out_batch = quantized_linear(x_batch, w_packed, scales, group_size=args.group_size)
        torch.mps.synchronize()
        print(f"  Input shape:  {x_batch.shape}")
        print(f"  Output shape: {out_batch.shape}")
        assert out_batch.shape == (4, 16, N), f"Expected (4, 16, {N}), got {out_batch.shape}"
        print("  PASSED")

    if args.benchmark:
        print("\n=== Benchmark ===")
        benchmark_against_native(
            M=args.M,
            K=args.K,
            N=args.N,
            group_size=args.group_size,
            warmup=args.warmup,
            iterations=args.iters,
        )

        # Also test with typical LLM dimensions
        if args.M == 1 and args.K == 4096 and args.N == 4096:
            print("\n--- Typical LLM shapes ---")
            for M, K, N in [(1, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)]:
                benchmark_against_native(M=M, K=K, N=N, group_size=args.group_size)


if __name__ == "__main__":
    main()
