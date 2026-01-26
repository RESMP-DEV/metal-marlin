"""High-level nn.Module wrappers for Metal Marlin quantized GEMM kernels.

Provides MarlinLinear, a drop-in replacement for mlx.nn.Linear and
mlx.nn.QuantizedLinear in inference mode. Weights are stored in packed
FP4 (E2M1) or INT4 format with per-group scales, and the forward pass
dispatches to the fused dequant-GEMM Metal kernel when MLX is available,
or falls back to a numpy-based CPU implementation otherwise.

Usage:
    from metal_marlin.layers import MarlinLinear

    # Convert an existing linear layer (requires MLX)
    layer = MarlinLinear.from_linear(existing_linear, group_size=32)
    output = layer(x)

    # Or wrap pre-packed weights directly
    layer = MarlinLinear(
        in_features=4096, out_features=4096,
        weight_packed=w_packed, scales=scales,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._compat import HAS_MLX, from_numpy, mx, nn, require_mlx, to_numpy
from .dtypes import DTypeConfig, get_default_config

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn

# E2M1 codebook: the 16 representable FP4 values.
# Used for CPU fallback dequantization.
_E2M1_VALUES: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# FP4 packing: 8 FP4 values per uint32
FP4_PER_U32 = 8


def _dequant_fp4_numpy(
    packed: np.ndarray,
    scales: np.ndarray,
    K: int,
    N: int,
    group_size: int,
) -> np.ndarray:
    """CPU fallback: dequantize packed FP4 weights to float32.

    Args:
        packed: uint32 array [K, N//8] with 8 packed FP4 nibbles per element.
        scales: float array [K//group_size, N] with per-group scale factors.
        K: Number of rows (input features).
        N: Number of columns (output features).
        group_size: Elements per quantization group along K.

    Returns:
        float32 array [K, N] with dequantized weights.
    """
    packed_n = N // FP4_PER_U32

    # Extract nibble indices from packed uint32 words
    # packed layout: packed[k, g] has nibbles for cols [g*8, g*8+8)
    indices = np.empty((K, N), dtype=np.uint8)
    for g in range(packed_n):
        col_start = g * FP4_PER_U32
        for i in range(FP4_PER_U32):
            indices[:, col_start + i] = (
                (packed[:, g] >> (i * 4)) & 0xF
            ).astype(np.uint8)

    # Dequantize via E2M1 codebook
    values = _E2M1_VALUES[indices].astype(np.float32)

    # Apply per-group scales
    scales_f32 = scales.astype(np.float32)
    scales_expanded = np.repeat(scales_f32, group_size, axis=0)  # [K, N]
    values *= scales_expanded

    return values


def _forward_numpy(
    x: np.ndarray,
    weight_packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """CPU fallback: dequant + matmul via numpy.

    This is slower than the Metal kernel but works without MLX.

    Args:
        x: Input activations [*, K].
        weight_packed: Packed FP4 weights [K, N//8] as uint32.
        scales: Per-group scales [K//group_size, N].
        group_size: Elements per quantization group.
        bias: Optional bias [N].

    Returns:
        Output [*, N] as float32.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    M = int(np.prod(orig_shape[:-1]))

    x_2d = x.reshape(M, K).astype(np.float32)

    packed_n = weight_packed.shape[1]
    N = packed_n * FP4_PER_U32

    # Dequantize weights
    weights_dequant = _dequant_fp4_numpy(weight_packed, scales, K, N, group_size)

    # Matrix multiply: x @ W.T -> [M, K] @ [K, N] -> [M, N]
    # Note: our weights are [K, N], so no transpose needed
    out = x_2d @ weights_dequant

    if bias is not None:
        out = out + bias.astype(np.float32)

    # Reshape back
    out_shape = list(orig_shape[:-1]) + [N]
    return out.reshape(out_shape)


class MarlinLinear:
    """Quantized linear layer using Metal Marlin fused dequant-GEMM kernels.

    Stores weights in packed FP4 (E2M1) format with per-group FP16 scales.
    When MLX is available, the forward pass dispatches to a Metal compute
    kernel that dequantizes in registers and accumulates via simdgroup
    multiply-accumulate. When MLX is not available, falls back to a
    numpy-based CPU implementation.

    Supports FP4 quantization now; INT4 support is planned pending kernel
    implementation.

    This is a drop-in replacement for nn.Linear in inference (no backward pass
    through quantized weights).

    Note:
        When MLX is available, this class inherits from mlx.nn.Module.
        When MLX is not available, it's a standalone class.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        bias: Whether this layer has a bias term.
        quant_type: Quantization format. Currently only "fp4" is supported.
        group_size: Number of elements per quantization group along K.
            Must divide in_features evenly. Default: 32.
        weight_packed: Pre-packed weight tensor [K, N//8] as uint32.
            If None, initialized to zeros (use from_linear or load weights).
        scales: Pre-computed per-group scales [K//group_size, N].
            If None, initialized to ones. Dtype determined by dtype_config.
        zeros: Per-group zero points for asymmetric INT4 [K//group_size, N].
            Only used when quant_type="int4". Currently unused.
        dtype_config: Dtype configuration for scales, activations, and bias.
            If None, uses the global default configuration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: Literal["fp4", "int4", "int4_sym"] = "fp4",
        group_size: int = 32,
        weight_packed: Any | None = None,
        scales: Any | None = None,
        zeros: Any | None = None,
        dtype_config: DTypeConfig | None = None,
    ):
        # Initialize nn.Module if MLX is available
        if HAS_MLX and nn is not None:
            nn.Module.__init__(self)

        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.group_size = group_size
        self.dtype_config = dtype_config if dtype_config is not None else get_default_config()

        if in_features % FP4_PER_U32 != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"pack factor ({FP4_PER_U32})"
            )
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"group_size ({group_size})"
            )

        # Packed weights: [K, N // pack_factor] as uint32
        # The kernel expects weights packed along N: 8 FP4 values per uint32
        pack_factor = FP4_PER_U32
        if out_features % pack_factor != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"pack factor ({pack_factor})"
            )

        num_groups = in_features // group_size

        if HAS_MLX and mx is not None:
            # MLX path: store as mx.array
            if weight_packed is not None:
                self.weight = weight_packed
            else:
                self.weight = mx.zeros(
                    (in_features, out_features // pack_factor), dtype=mx.uint32
                )

            if scales is not None:
                self.scales = scales
            else:
                self.scales = mx.ones(
                    (num_groups, out_features), dtype=self.dtype_config.mlx_scales
                )

            if quant_type == "int4" and zeros is not None:
                self.zeros = zeros
            else:
                self.zeros = None

            if bias:
                self.bias = mx.zeros(
                    (out_features,), dtype=self.dtype_config.mlx_activations
                )
            else:
                self.bias = None
        else:
            # Numpy fallback path
            if weight_packed is not None:
                self.weight = to_numpy(weight_packed)
            else:
                self.weight = np.zeros(
                    (in_features, out_features // pack_factor), dtype=np.uint32
                )

            if scales is not None:
                self.scales = to_numpy(scales)
            else:
                self.scales = np.ones(
                    (num_groups, out_features), dtype=np.float16
                )

            if quant_type == "int4" and zeros is not None:
                self.zeros = to_numpy(zeros)
            else:
                self.zeros = None

            if bias:
                self.bias = np.zeros((out_features,), dtype=np.float32)
            else:
                self.bias = None

    def _forward_mlx(self, x: Any) -> Any:
        """Forward pass using MLX Metal kernel."""
        # Import here to avoid circular import and only when MLX is available
        from .metal_marlin import quantized_linear

        out = quantized_linear(
            x, self.weight, self.scales, self.group_size, self.dtype_config
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_numpy(self, x: Any) -> Any:
        """Forward pass using numpy CPU fallback."""
        x_np = to_numpy(x)
        weight_np = to_numpy(self.weight)
        scales_np = to_numpy(self.scales)
        bias_np = to_numpy(self.bias) if self.bias is not None else None

        out = _forward_numpy(x_np, weight_np, scales_np, self.group_size, bias_np)

        # Return as MLX array if MLX is available and input was MLX
        if HAS_MLX and mx is not None:
            return from_numpy(out, backend="mlx")
        return out

    def __call__(self, x: Any) -> Any:
        """Forward pass: fused FP4 dequant + GEMM.

        Uses Metal kernel when MLX is available, otherwise falls back to
        numpy CPU implementation.

        Args:
            x: Input activations of shape [*, in_features].
                Supports arbitrary leading batch dimensions.

        Returns:
            Output of shape [*, out_features].
        """
        if self.quant_type == "fp4":
            if HAS_MLX:
                return self._forward_mlx(x)
            else:
                return self._forward_numpy(x)
        elif self.quant_type in ("int4", "int4_sym"):
            raise NotImplementedError(
                "INT4 GEMM kernel not yet implemented. "
                "Use quant_type='fp4' or wait for INT4 kernel support."
            )
        else:
            raise ValueError(f"Unknown quant_type: {self.quant_type!r}")

    @classmethod
    def from_linear(
        cls,
        linear: Any,
        quant_type: Literal["fp4"] = "fp4",
        group_size: int = 32,
        dtype_config: DTypeConfig | None = None,
    ) -> MarlinLinear:
        """Quantize an existing nn.Linear layer to FP4 Marlin format.

        Requires MLX to be available.

        Extracts the weight matrix from the linear layer, quantizes it
        to FP4 E2M1 with per-group absmax scaling, and packs it into
        the kernel's expected uint32 layout.

        Args:
            linear: Source nn.Linear layer. Weight shape is
                [out_features, in_features] following MLX/PyTorch convention.
            quant_type: Quantization format. Only "fp4" currently supported.
            group_size: Elements per quantization group. Default: 32.
            dtype_config: Dtype configuration for scales and activations.
                If None, uses the global default configuration.

        Returns:
            A new MarlinLinear layer with quantized weights.
        """
        require_mlx("from_linear conversion")

        if quant_type != "fp4":
            raise NotImplementedError(
                f"from_linear only supports quant_type='fp4', got {quant_type!r}"
            )

        # Import here to avoid issues when MLX not available
        from .metal_marlin import pack_fp4_weights

        # MLX nn.Linear stores weight as [out_features, in_features]
        weight = linear.weight
        out_features, in_features = weight.shape

        # pack_fp4_weights expects [out_features, in_features] and transposes internally
        packed, scales = pack_fp4_weights(weight, group_size=group_size)
        mx.eval(packed, scales)

        has_bias = hasattr(linear, "bias") and linear.bias is not None
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            quant_type="fp4",
            group_size=group_size,
            weight_packed=packed,
            scales=scales,
            dtype_config=dtype_config,
        )

        if has_bias:
            layer.bias = linear.bias

        return layer

    @classmethod
    def from_quantized_linear(
        cls,
        ql: Any,
        group_size: int = 32,
        dtype_config: DTypeConfig | None = None,
    ) -> MarlinLinear:
        """Convert an mlx.nn.QuantizedLinear (affine INT4) to MarlinLinear (FP4).

        Requires MLX to be available.

        Dequantizes the affine-quantized weights back to FP16, then
        re-quantizes them into FP4 E2M1 Marlin format. This introduces
        a small additional quantization error due to the format change.

        Args:
            ql: Source QuantizedLinear layer (affine 4-bit).
            group_size: Marlin quantization group size. Default: 32.
            dtype_config: Dtype configuration for scales and activations.
                If None, uses the global default configuration.

        Returns:
            A new MarlinLinear layer with FP4-packed weights.
        """
        require_mlx("from_quantized_linear conversion")

        # Import here to avoid issues when MLX not available
        from .metal_marlin import pack_fp4_weights

        # Dequantize from MLX affine format back to FP16
        w_fp16 = mx.dequantize(
            ql.weight,
            ql.scales,
            ql.biases,
            ql.group_size,
            ql.bits,
        )
        mx.eval(w_fp16)

        # w_fp16 is [out_features, in_features]
        out_features, in_features = w_fp16.shape

        # Re-quantize to FP4 Marlin format
        packed, scales = pack_fp4_weights(w_fp16, group_size=group_size)
        mx.eval(packed, scales)

        # Check for bias (QuantizedLinear may or may not have one)
        has_bias = hasattr(ql, "bias") and ql.bias is not None
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            quant_type="fp4",
            group_size=group_size,
            weight_packed=packed,
            scales=scales,
            dtype_config=dtype_config,
        )

        if has_bias:
            layer.bias = ql.bias

        return layer

    def extra_repr(self) -> str:
        backend = "MLX" if HAS_MLX else "numpy"
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"quant_type={self.quant_type!r}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None}, "
            f"backend={backend}"
        )


# Make MarlinLinear inherit from nn.Module when MLX is available
# This is done dynamically to avoid import errors when MLX is not installed
if HAS_MLX and nn is not None:
    # Create a new class that properly inherits from nn.Module
    _OriginalMarlinLinear = MarlinLinear

    class MarlinLinear(nn.Module):  # type: ignore[no-redef]
        """Quantized linear layer using Metal Marlin fused dequant-GEMM kernels."""

        __doc__ = _OriginalMarlinLinear.__doc__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            # Use the original class logic
            _OriginalMarlinLinear.__init__(self, *args, **kwargs)

        # Forward all other methods to the original implementation
        __call__ = _OriginalMarlinLinear.__call__
        _forward_mlx = _OriginalMarlinLinear._forward_mlx
        _forward_numpy = _OriginalMarlinLinear._forward_numpy
        from_linear = _OriginalMarlinLinear.from_linear
        from_quantized_linear = _OriginalMarlinLinear.from_quantized_linear
        extra_repr = _OriginalMarlinLinear.extra_repr
