"""High-level nn.Module wrappers for Metal Marlin quantized GEMM kernels.

Provides MarlinLinear, a drop-in replacement for torch.nn.Linear in inference
mode with quantized weights.

When PyTorch is not available, provides a stub MarlinLinear that raises RuntimeError
on instantiation.

Usage:
    from metal_marlin.layers import MarlinLinear

    # From dimensions (initializes with random quantized weights)
    layer = MarlinLinear(in_features=512, out_features=256, bias=True)
    output = layer(x)

    # From pre-packed weights
    layer = MarlinLinear(weight_packed, scales, bias, group_size=32)
    output = layer(x)

    # Convert from torch.nn.Linear
    layer = MarlinLinear.from_linear(existing_linear)
"""

from __future__ import annotations

from typing import Any, overload

from ._compat import HAS_TORCH, torch

if HAS_TORCH and torch is not None:
    import torch.nn as nn

    class MarlinLinear(nn.Module):
        """PyTorch module for quantized linear layers using Metal Marlin kernels.

        This module wraps quantized FP4 weights and provides a forward pass
        that dequantizes and computes the linear transformation.

        Two constructor forms are supported:
        1. From dimensions: MarlinLinear(in_features, out_features, bias=False, ...)
           Creates a layer with random quantized weights.
        2. From pre-packed weights: MarlinLinear(weight_packed, scales, bias, group_size)
           Creates a layer from pre-quantized weights.

        Args (dimension form):
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias. Default: False.
            quant_type: Quantization type. Only "fp4" supported. Default: "fp4".
            group_size: Elements per quantization group. Default: 128.

        Args (pre-packed form):
            weight_packed: Packed FP4 weights [K, N//8] as uint32.
            scales: Per-group scales [K//group_size, N] as float16.
            bias: Optional bias vector [N] as float16.
            group_size: Quantization group size. Default: 32.
        """

        @overload
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            *,
            quant_type: str = "fp4",
            group_size: int = 128,
        ) -> None: ...

        @overload
        def __init__(
            self,
            weight_packed: torch.Tensor,
            scales: torch.Tensor,
            bias: torch.Tensor | None = None,
            group_size: int = 32,
        ) -> None: ...

        def __init__(
            self,
            in_features_or_weight: int | torch.Tensor,
            out_features_or_scales: int | torch.Tensor,
            bias: bool | torch.Tensor | None = False,
            group_size: int = 128,
            *,
            quant_type: str = "fp4",
        ) -> None:
            super().__init__()

            # Determine constructor form based on first argument type
            if isinstance(in_features_or_weight, int):
                # Dimension-based constructor
                self._init_from_dims(
                    in_features=in_features_or_weight,
                    out_features=int(out_features_or_scales),
                    use_bias=bool(bias),
                    quant_type=quant_type,
                    group_size=group_size,
                )
            else:
                # Pre-packed weight constructor
                self._init_from_packed(
                    weight_packed=in_features_or_weight,
                    scales=out_features_or_scales,  # type: ignore[arg-type]
                    bias=bias if not isinstance(bias, bool) else None,
                    group_size=group_size,
                )

        def _init_from_dims(
            self,
            in_features: int,
            out_features: int,
            use_bias: bool,
            quant_type: str,
            group_size: int,
        ) -> None:
            """Initialize from dimensions with random weights."""
            if quant_type != "fp4":
                raise NotImplementedError(f"Only quant_type='fp4' is supported, got {quant_type!r}")

            # Ensure dimensions are compatible with quantization
            FP4_PER_U32 = 8
            if out_features % FP4_PER_U32 != 0:
                # Pad to nearest multiple
                padded_out = ((out_features + FP4_PER_U32 - 1) // FP4_PER_U32) * FP4_PER_U32
            else:
                padded_out = out_features

            if in_features % group_size != 0:
                padded_in = ((in_features + group_size - 1) // group_size) * group_size
            else:
                padded_in = in_features

            # Create random FP16 weights and quantize
            weight = torch.randn(padded_out, padded_in, dtype=torch.float16) * 0.02
            w_packed, scales = self._pack_fp4_weights(weight, group_size)

            # Slice to actual dimensions if padded
            self._padded_in = padded_in
            self._padded_out = padded_out
            self._actual_in = in_features
            self._actual_out = out_features

            self.register_buffer("weight_packed", w_packed)
            self.register_buffer("scales", scales)

            if use_bias:
                bias_tensor = torch.zeros(out_features, dtype=torch.float16)
                self.register_buffer("bias", bias_tensor)
            else:
                self.register_buffer("bias", None)

            self.group_size = group_size
            self.in_features = in_features
            self.out_features = out_features

        def _init_from_packed(
            self,
            weight_packed: torch.Tensor,
            scales: torch.Tensor,
            bias: torch.Tensor | None,
            group_size: int,
        ) -> None:
            """Initialize from pre-packed weights."""
            self.register_buffer("weight_packed", weight_packed)
            self.register_buffer("scales", scales)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.register_buffer("bias", None)
            self.group_size = group_size

            # Derive dimensions from packed weights
            # weight_packed shape: [K, N//8]
            K = weight_packed.shape[0]
            packed_n = weight_packed.shape[1]
            self.in_features = K
            self.out_features = packed_n * 8  # 8 FP4 values per uint32

            self._padded_in = self.in_features
            self._padded_out = self.out_features
            self._actual_in = self.in_features
            self._actual_out = self.out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with fused dequant-GEMM.

            Args:
                x: Input tensor [*, in_features]

            Returns:
                Output tensor [*, out_features]
            """
            # Dequantize weights and perform matmul
            weight_fp16 = self._dequantize_fp4()
            # Cast input to match weight dtype
            x_fp16 = x.to(weight_fp16.dtype)

            # Handle padding if necessary
            if self._actual_in != self._padded_in:
                # Pad input
                pad_size = self._padded_in - self._actual_in
                x_fp16 = torch.nn.functional.pad(x_fp16, (0, pad_size))

            bias = self.bias.to(weight_fp16.dtype) if self.bias is not None else None
            out = torch.nn.functional.linear(x_fp16, weight_fp16, None)

            # Slice output if padded
            if self._actual_out != self._padded_out:
                out = out[..., : self._actual_out]

            # Add bias after slicing - use in-place to avoid MPS validation error
            if bias is not None:
                out.add_(bias)

            # Cast back to input dtype
            return out.to(x.dtype)

        def _dequantize_fp4(self) -> torch.Tensor:
            """Dequantize packed FP4 weights to FP16.

            Returns:
                Dequantized weight tensor [out_features, in_features]
            """
            K = self._padded_in
            N = self._padded_out
            device = self.weight_packed.device
            dtype = torch.float16

            # Unpack FP4 nibbles from uint32
            # weight_packed shape: [K, N//8]
            weight_fp16 = torch.zeros(K, N, dtype=dtype, device=device)

            # E2M1 FP4 dequantization values (all 16 nibble patterns)
            e2m1_values = self._get_e2m1_table().to(device)

            for k in range(K):
                for n_block in range(N // 8):
                    word = self.weight_packed[k, n_block].item()
                    for i in range(8):
                        nibble = (word >> (i * 4)) & 0xF
                        n_idx = n_block * 8 + i
                        # Apply scale
                        scale_k = k // self.group_size
                        scale = self.scales[scale_k, n_idx]
                        weight_fp16[k, n_idx] = e2m1_values[nibble] * scale

            # Transpose to [N, K] for nn.functional.linear
            return weight_fp16.T

        @staticmethod
        def _get_e2m1_table() -> torch.Tensor:
            """Return all 16 E2M1 representable values as float16."""
            values = torch.zeros(16, dtype=torch.float32)
            for nibble in range(16):
                sign = (nibble >> 3) & 1
                exp_bits = (nibble >> 1) & 3
                mant_bit = nibble & 1

                if exp_bits == 0 and mant_bit == 0:
                    val = 0.0
                elif exp_bits == 0 and mant_bit == 1:
                    val = 0.5  # Subnormal
                else:
                    mantissa = 1.0 + mant_bit * 0.5
                    exponent = exp_bits - 1  # bias = 1
                    val = mantissa * (2.0**exponent)

                if sign:
                    val = -val
                values[nibble] = val

            return values.to(torch.float16)

        @staticmethod
        def from_linear(
            linear: nn.Linear,
            quant_type: str = "fp4",
            group_size: int = 32,
        ) -> MarlinLinear:
            """Convert a torch.nn.Linear layer to MarlinLinear (FP4).

            Extracts the weight matrix from the linear layer, quantizes it
            to FP4 E2M1 with per-group absmax scaling, and packs it into
            the kernel's expected uint32 layout.

            Args:
                linear: Source nn.Linear layer.
                quant_type: Quantization format. Only "fp4" currently supported.
                group_size: Elements per quantization group. Default: 32.

            Returns:
                A new MarlinLinear layer with quantized weights.
            """
            if quant_type != "fp4":
                raise NotImplementedError(
                    f"from_linear only supports quant_type='fp4', got {quant_type!r}"
                )

            weight = linear.weight.data  # [out_features, in_features]
            w_packed, scales = MarlinLinear._pack_fp4_weights(weight, group_size)

            bias = linear.bias.data if linear.bias is not None else None
            return MarlinLinear(w_packed, scales, bias, group_size=group_size)

        @staticmethod
        def _pack_fp4_weights(
            weight: torch.Tensor, group_size: int = 32
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Pack FP16 weights into Marlin FP4 format with per-group scales.

            Args:
                weight: FP16 weight matrix [out_features, in_features].
                group_size: Number of elements per quantization group.

            Returns:
                Tuple of (weight_packed, scales)
            """
            # Transpose to [K, N] layout (K = in_features, N = out_features)
            w = weight.T.to(torch.float16)  # [K, N]
            K, N = w.shape

            FP4_PER_U32 = 8
            if N % FP4_PER_U32 != 0:
                raise ValueError(f"N ({N}) must be divisible by {FP4_PER_U32}")
            if K % group_size != 0:
                raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

            # Compute per-group scales
            max_e2m1 = 6.0
            w_grouped = w.reshape(K // group_size, group_size, N)
            absmax = w_grouped.abs().amax(dim=1)  # [K//group_size, N]
            absmax = torch.clamp(absmax, min=1e-7)
            scales = absmax / max_e2m1

            # E2M1 representable values
            e2m1_values = MarlinLinear._get_e2m1_table()

            # Normalize and quantize
            scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]
            w_normalized = w / scales_expanded
            w_normalized = torch.clamp(w_normalized, -max_e2m1, max_e2m1)

            # Find nearest E2M1 nibble index
            w_np = w_normalized.cpu().numpy()
            e2m1_np = e2m1_values.cpu().numpy()

            import numpy as np

            packed_n = N // FP4_PER_U32
            packed = np.zeros((K, packed_n), dtype=np.uint32)

            for col_group in range(packed_n):
                col_start = col_group * FP4_PER_U32
                cols = w_np[:, col_start : col_start + FP4_PER_U32]
                dists = np.abs(cols[:, :, None] - e2m1_np[None, None, :])
                nibble_indices = np.argmin(dists, axis=2).astype(np.uint32)

                word = np.zeros((K,), dtype=np.uint32)
                for bit_pos in range(FP4_PER_U32):
                    word |= nibble_indices[:, bit_pos] << (bit_pos * 4)
                packed[:, col_group] = word

            weight_packed = torch.from_numpy(packed).to(weight.device)
            return weight_packed, scales

else:
    # Stub class when PyTorch is not available
    class MarlinLinear:  # type: ignore[no-redef]
        """Stub for MarlinLinear when PyTorch is not available.

        This class raises RuntimeError on instantiation, allowing the module
        to be imported without PyTorch for type checking and documentation purposes.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "MarlinLinear requires PyTorch. Install PyTorch with: pip install torch"
            )

        def __call__(self, x: Any) -> Any:
            raise RuntimeError("MarlinLinear requires PyTorch")

        def forward(self, x: Any) -> Any:
            raise RuntimeError("MarlinLinear requires PyTorch")

        @staticmethod
        def from_linear(linear: Any, quant_type: str = "fp4", group_size: int = 32) -> MarlinLinear:
            raise RuntimeError("MarlinLinear requires PyTorch")


if HAS_TORCH and torch is not None:
    class MixedPrecisionLinear(nn.Module):
        """Linear layer that handles both BF16 and FP4 quantized inputs.
        
        This layer automatically detects the input precision and applies the
        appropriate computation path. For BF16 inputs, it uses standard
        torch.nn.functional.linear. For FP4 inputs, it dequantizes and
        processes using the FP4 pipeline.
        
        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias. Default: True.
            fp4_group_size: Group size for FP4 quantization. Default: 128.
        """
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            fp4_group_size: int = 128,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.fp4_group_size = fp4_group_size
            
            # BF16 weights (full precision path)
            self.weight_bf16 = nn.Parameter(
                torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02
            )
            
            # FP4 quantized weights (quantized path)
            # Initialize with quantized version of BF16 weights
            with torch.no_grad():
                weight_fp16 = self.weight_bf16.to(torch.float16)
                w_packed, scales = MarlinLinear._pack_fp4_weights(
                    weight_fp16, fp4_group_size
                )
            
            self.register_buffer("weight_fp4_packed", w_packed)
            self.register_buffer("scales", scales)
            
            # Shared bias
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
            else:
                self.register_parameter("bias", None)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with automatic precision detection.
            
            Args:
                x: Input tensor [*, in_features]. Can be BF16, FP16, or FP4-quantized.
            
            Returns:
                Output tensor [*, out_features] in the same dtype as input.
            """
            input_dtype = x.dtype
            
            # Detect if input is FP4-quantized (stored as uint32)
            if x.dtype == torch.uint32:
                return self._forward_fp4(x)
            elif x.dtype == torch.bfloat16:
                return self._forward_bf16(x)
            elif x.dtype == torch.float16:
                # Treat FP16 as standard precision, but use FP4 weights for efficiency
                return self._forward_fp16(x)
            else:
                # Cast to BF16 for unsupported dtypes
                return self._forward_bf16(x.to(torch.bfloat16)).to(input_dtype)
        
        def _forward_bf16(self, x: torch.Tensor) -> torch.Tensor:
            """Standard BF16 forward pass."""
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            out = torch.nn.functional.linear(x, self.weight_bf16, bias)
            return out
        
        def _forward_fp16(self, x: torch.Tensor) -> torch.Tensor:
            """FP16 forward pass using dequantized FP4 weights."""
            # Dequantize FP4 weights to FP16
            weight_fp16 = self._dequantize_fp4()
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            out = torch.nn.functional.linear(x, weight_fp16, bias)
            return out
        
        def _forward_fp4(self, x: torch.Tensor) -> torch.Tensor:
            """FP4 quantized forward pass.
            
            Expects x to be packed uint32 FP4 values. Dequantizes both
            input and weights, computes matmul, and returns result.
            """
            # Dequantize input (assumed to be in same format as weights)
            x_dequant = self._dequantize_fp4_input(x)
            
            # Dequantize weights
            weight_fp16 = self._dequantize_fp4()
            
            # Compute matmul in FP16
            bias = self.bias.to(torch.float16) if self.bias is not None else None
            out = torch.nn.functional.linear(x_dequant, weight_fp16, bias)
            return out
        
        def _dequantize_fp4(self) -> torch.Tensor:
            """Dequantize packed FP4 weights to FP16.
            
            Returns:
                Dequantized weight tensor [out_features, in_features] as FP16.
            """
            K = self.in_features
            N = self.out_features
            device = self.weight_fp4_packed.device
            
            # Ensure dimensions are compatible
            FP4_PER_U32 = 8
            padded_K = ((K + self.fp4_group_size - 1) // self.fp4_group_size) * self.fp4_group_size
            padded_N = ((N + FP4_PER_U32 - 1) // FP4_PER_U32) * FP4_PER_U32
            
            weight_fp16 = torch.zeros(padded_K, padded_N, dtype=torch.float16, device=device)
            
            # E2M1 FP4 dequantization lookup table
            e2m1_values = MarlinLinear._get_e2m1_table().to(device)
            
            # Unpack and dequantize
            for k in range(self.weight_fp4_packed.shape[0]):
                for n_block in range(self.weight_fp4_packed.shape[1]):
                    word = self.weight_fp4_packed[k, n_block].item()
                    for i in range(8):
                        nibble = (word >> (i * 4)) & 0xF
                        n_idx = n_block * 8 + i
                        if n_idx >= padded_N:
                            break
                        scale_k = k // self.fp4_group_size
                        scale = self.scales[scale_k, n_idx]
                        weight_fp16[k, n_idx] = e2m1_values[nibble] * scale
            
            # Slice to actual dimensions and transpose
            weight_fp16 = weight_fp16[:K, :N]
            return weight_fp16.T  # [N, K] for nn.functional.linear
        
        def _dequantize_fp4_input(self, x: torch.Tensor) -> torch.Tensor:
            """Dequantize FP4 packed input tensor.
            
            Args:
                x: Packed FP4 input [batch, in_features//8] as uint32.
            
            Returns:
                Dequantized FP16 tensor [batch, in_features].
            """
            # This is a simplified implementation
            # Real implementation would need input-specific scales
            batch_dims = x.shape[:-1]
            packed_features = x.shape[-1]
            features = packed_features * 8
            
            device = x.device
            e2m1_values = MarlinLinear._get_e2m1_table().to(device)
            
            # Unpack
            x_fp16 = torch.zeros(*batch_dims, features, dtype=torch.float16, device=device)
            
            # Simple unpacking without scales (would need scales in real use)
            x_flat = x.reshape(-1, packed_features)
            out_flat = x_fp16.reshape(-1, features)
            
            for i in range(x_flat.shape[0]):
                for j in range(packed_features):
                    word = x_flat[i, j].item()
                    for k in range(8):
                        nibble = (word >> (k * 4)) & 0xF
                        out_flat[i, j * 8 + k] = e2m1_values[nibble]
            
            return x_fp16
        
        @staticmethod
        def from_linear(
            linear: nn.Linear,
            fp4_group_size: int = 128,
        ) -> MixedPrecisionLinear:
            """Convert a torch.nn.Linear to MixedPrecisionLinear.
            
            Args:
                linear: Source nn.Linear layer.
                fp4_group_size: Group size for FP4 quantization.
            
            Returns:
                MixedPrecisionLinear with copied weights.
            """
            layer = MixedPrecisionLinear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=linear.bias is not None,
                fp4_group_size=fp4_group_size,
            )
            
            # Copy weights
            with torch.no_grad():
                layer.weight_bf16.copy_(linear.weight.data.to(torch.bfloat16))
                if linear.bias is not None:
                    layer.bias.copy_(linear.bias.data)
                
                # Requantize to FP4
                weight_fp16 = linear.weight.data.to(torch.float16)
                w_packed, scales = MarlinLinear._pack_fp4_weights(
                    weight_fp16, fp4_group_size
                )
                layer.weight_fp4_packed.copy_(w_packed)
                layer.scales.copy_(scales)
            
            return layer

else:
    class MixedPrecisionLinear:  # type: ignore[no-redef]
        """Stub for MixedPrecisionLinear when PyTorch is not available."""
        
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "MixedPrecisionLinear requires PyTorch. Install PyTorch with: pip install torch"
            )


__all__ = ["MarlinLinear", "MixedPrecisionLinear"]
