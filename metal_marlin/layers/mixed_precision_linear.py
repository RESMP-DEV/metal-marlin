"""MixedPrecisionLinear: A linear layer that handles both BF16 and FP4 inputs.

This module provides `MixedPrecisionLinear`, a drop-in replacement for 
torch.nn.Linear that supports both BF16 (full precision) and FP4 (quantized) 
input tensors. The layer automatically detects input precision and dispatches
to the appropriate computation path.

For BF16 inputs: Uses standard BF16 GEMM with optional FP4 weights.
For FP4 inputs: Dequantizes inputs and uses FP4 weight kernels.

Usage:
    from metal_marlin.layers import MixedPrecisionLinear
    
    # Create layer
    layer = MixedPrecisionLinear(in_features=512, out_features=256, bias=True)
    
    # BF16 input (standard precision)
    x_bf16 = torch.randn(1, 10, 512, dtype=torch.bfloat16)
    out_bf16 = layer(x_bf16)
    
    # FP4 input (quantized)
    x_fp4 = torch.randint(0, 2**32, (1, 10, 64), dtype=torch.uint32)  # 512//8
    x_scales = torch.randn(10, 4, dtype=torch.float16)  # scales for input
    out_fp4 = layer(x_fp4, input_scales=x_scales)
"""

from __future__ import annotations

from typing import Any

from .._compat import HAS_TORCH, get_e2m1_torch_table, torch

if HAS_TORCH and torch is not None:
    import torch.nn as nn
    import torch.nn.functional as F

    from .mmfp4_linear import MMFP4Linear, _fast_dequant, _minimize_contiguous

    _E2M1_TABLE = get_e2m1_torch_table()

    class MixedPrecisionLinear(nn.Module):
        """Linear layer that handles both BF16 and FP4 inputs.
        
        This layer provides a unified interface for linear transformations
        with automatic precision detection. It maintains both BF16 weights
        for full-precision paths and FP4 quantized weights for efficient
        inference.
        
        The forward pass automatically detects input precision:
        - BF16/FP16 inputs: Use standard GEMM (can optionally use FP4 weights)
        - FP4 inputs (packed uint32): Dequantize and compute
        
        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias. Default: True.
            fp4_group_size: Group size for FP4 quantization. Default: 128.
            use_fp4_weights: If True, use FP4 weights even for BF16 inputs.
                            Default: False (use BF16 weights for BF16 inputs).
        
        Attributes:
            in_features: Number of input features.
            out_features: Number of output features.
            fp4_group_size: Size of FP4 quantization groups.
            weight_bf16: BF16 weight parameter [out_features, in_features].
            weight_fp4_packed: Packed FP4 weights buffer [out_features, in_features//8].
            scales: FP4 scale factors buffer [n_groups, out_features].
            bias: Optional bias parameter [out_features].
        
        Example:
            >>> layer = MixedPrecisionLinear(512, 256, bias=True)
            >>> 
            >>> # BF16 inference
            >>> x = torch.randn(2, 10, 512, dtype=torch.bfloat16)
            >>> out = layer(x)
            >>> print(out.shape)
            torch.Size([2, 10, 256])
            >>> 
            >>> # FP4 inference
            >>> x_fp4 = torch.randint(0, 2**32, (2, 10, 64), dtype=torch.int64)
            >>> x_scales = torch.randn(4, 10, dtype=torch.float16)  # 512//128=4 groups
            >>> out_fp4 = layer(x_fp4, input_scales=x_scales)
        """
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            fp4_group_size: int = 128,
            use_fp4_weights: bool = False,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.fp4_group_size = fp4_group_size
            self.use_fp4_weights = use_fp4_weights
            
            # Validate dimensions
            if in_features % 8 != 0:
                raise ValueError(
                    f"in_features ({in_features}) must be divisible by 8 for FP4 packing"
                )
            if in_features % fp4_group_size != 0:
                raise ValueError(
                    f"in_features ({in_features}) must be divisible by fp4_group_size ({fp4_group_size})"
                )
            
            # BF16 weights for full-precision path
            self.weight_bf16 = nn.Parameter(
                torch.randn(out_features, in_features, dtype=torch.bfloat16).mul_(0.02)
            )
            
            # FP4 quantized weights for efficient path
            # Initialize by quantizing BF16 weights
            with torch.no_grad():
                weight_fp16 = self.weight_bf16.to(torch.float16)
                w_packed, scales = self._pack_weights_fp4(weight_fp16, fp4_group_size)
            
            self.register_buffer("weight_fp4_packed", w_packed)
            self.register_buffer("scales", scales)
            
            # Shared bias (stored in FP32 for numerical stability)
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
            else:
                self.register_parameter("bias", None)
            
            # Cache for kernel optimization
            self._kernel_cache: torch.Tensor | None = None
            self._input_cache: torch.Tensor | None = None
        
        def forward(
            self, 
            x: torch.Tensor, 
            input_scales: torch.Tensor | None = None
        ) -> torch.Tensor:
            """Forward pass with automatic precision detection.
            
            Args:
                x: Input tensor. Can be:
                   - BF16/FP16: [batch, *, in_features] floating point
                   - FP4 packed: [batch, *, in_features//8] uint32/int32/int64
                input_scales: Required for FP4 inputs. Scale factors for 
                    dequantizing the input tensor. Shape depends on input
                    quantization groups.
            
            Returns:
                Output tensor [batch, *, out_features] with same dtype as input.
            
            Raises:
                ValueError: If input features don't match or if input_scales
                    are missing for FP4 inputs.
            """
            input_dtype = x.dtype
            
            # Detect input precision and dispatch
            if x.dtype in (torch.uint32, torch.int32, torch.int64):
                # FP4 packed input
                if input_scales is None:
                    raise ValueError(
                        "input_scales required for FP4 packed input. "
                        "Provide scales tensor matching input quantization groups."
                    )
                # Return FP16 output (don't cast back to uint32 input dtype)
                return self._forward_fp4(x, input_scales)
            elif x.dtype == torch.bfloat16:
                return self._forward_bf16(x)
            elif x.dtype == torch.float16:
                return self._forward_fp16(x)
            elif x.dtype == torch.float32:
                # FP32 input - convert to BF16 internally, return as FP32
                return self._forward_bf16(x.to(torch.bfloat16)).to(torch.float32)
            else:
                raise ValueError(
                    f"Unsupported input dtype: {x.dtype}. "
                    "Supported: bfloat16, float16, float32, uint32 (FP4 packed)"
                )
        
        def batch_aware_dispatch(
            self, 
            x: torch.Tensor, 
            lora_u: list[torch.Tensor] | torch.Tensor | None = None,
            lora_v: list[torch.Tensor] | torch.Tensor | None = None,
            lora_indices: torch.Tensor | None = None,
            input_scales: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Dispatch batch-aware LoRA forward pass.
            
            Args:
                x: Input tensor.
                lora_u: LoRA U matrices. List of [in, rank] or packed [num_adapters, in, rank].
                lora_v: LoRA V matrices. List of [rank, out] or packed [num_adapters, rank, out].
                lora_indices: Tensor of shape [batch] mapping samples to adapter indices.
                              -1 indicates no adapter.
                input_scales: Scales for FP4 input.
            """
            # For uint32 inputs, we need to compute base output in float dtype
            # since LoRA operations require floating point arithmetic
            input_dtype = x.dtype
            if input_dtype in (torch.uint32, torch.int32, torch.int64):
                # FP4 packed input - compute base output in float dtype
                if input_scales is None:
                     raise ValueError("input_scales required for FP4 LoRA dispatch")
                # Get base output in FP16 (dequantized)
                out = self._forward_fp4(x, input_scales)
            else:
                # Normal forward for floating point inputs
                out = self.forward(x)
            
            if lora_u is None or lora_v is None or lora_indices is None:
                return out
                
            # Flatten
            batch_shape = x.shape[:-1]
            flat_x = x.reshape(-1, x.shape[-1])
            
            # Dequantize input if needed for LoRA term
            if x.dtype in (torch.uint32, torch.int32, torch.int64):
                # Flatten scales to match flat_x
                # scales shape: [batch, n_groups] -> [total_tokens, n_groups]
                flat_scales = input_scales.reshape(-1, input_scales.shape[-1])
                flat_x_dequant = self._dequantize_fp4_input(flat_x, flat_scales)
            else:
                flat_x_dequant = flat_x
            
            flat_out = out.reshape(-1, self.out_features)
            flat_indices = lora_indices.reshape(-1)
            
            unique_indices = torch.unique(flat_indices)
            
            for idx in unique_indices:
                if idx.item() == -1:
                    continue
                
                idx_val = int(idx.item())
                mask = flat_indices == idx
                
                if isinstance(lora_u, list):
                    u = lora_u[idx_val]
                    v = lora_v[idx_val]
                else:
                    u = lora_u[idx_val]
                    v = lora_v[idx_val]
                
                # Compute LoRA term
                # Ensure input is in correct dtype for LoRA (usually BF16 or FP16)
                group_x = flat_x_dequant[mask].to(u.dtype)
                delta = (group_x @ u) @ v
                
                flat_out[mask] += delta.to(flat_out.dtype)
                
            return flat_out.reshape(*batch_shape, self.out_features)

        def _forward_bf16(self, x: torch.Tensor) -> torch.Tensor:
            """BF16 forward pass using BF16 weights.
            
            Args:
                x: BF16 input tensor [batch, *, in_features].
            
            Returns:
                BF16 output tensor [batch, *, out_features].
            """
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            out = F.linear(x, self.weight_bf16, bias)
            return out
        
        def _forward_fp16(self, x: torch.Tensor) -> torch.Tensor:
            """FP16 forward pass.
            
            If use_fp4_weights is True, uses dequantized FP4 weights.
            Otherwise, uses BF16 weights converted to FP16.
            
            Args:
                x: FP16 input tensor [batch, *, in_features].
            
            Returns:
                FP16 output tensor [batch, *, out_features].
            """
            if self.use_fp4_weights:
                # Use FP4 weights - dequantize on the fly
                weight_fp16 = self._dequantize_fp4_weights()
            else:
                # Use BF16 weights converted to FP16
                weight_fp16 = self.weight_bf16.to(torch.float16)
            
            bias = self.bias.to(torch.float16) if self.bias is not None else None
            out = F.linear(x, weight_fp16, bias)
            return out
        
        def _forward_fp4(
            self, 
            x: torch.Tensor, 
            input_scales: torch.Tensor
        ) -> torch.Tensor:
            """FP4 packed input forward pass.
            
            Dequantizes both input and weights, then computes matmul.
            
            Args:
                x: Packed FP4 input [batch, *, in_features//8] as uint32/int32/int64.
                input_scales: Scale factors for input dequantization.
                    Shape: [batch, n_groups] or [n_groups] for shared scales.
            
            Returns:
                FP16 output tensor [batch, *, out_features].
            """
            # Ensure input is on same device
            if x.device != self.weight_fp4_packed.device:
                x = x.to(self.weight_fp4_packed.device)
            if input_scales.device != self.weight_fp4_packed.device:
                input_scales = input_scales.to(self.weight_fp4_packed.device)
            
            # Dequantize input
            x_dequant = self._dequantize_fp4_input(x, input_scales)
            
            # Dequantize weights
            weight_fp16 = self._dequantize_fp4_weights()
            
            # Compute in FP16
            bias = self.bias.to(torch.float16) if self.bias is not None else None
            out = F.linear(x_dequant, weight_fp16, bias)
            
            return out
        
        def _dequantize_fp4_weights(self) -> torch.Tensor:
            """Dequantize FP4 packed weights to FP16.
            
            Uses optimized _fast_dequant from mmfp4_linear module.
            
            Returns:
                Dequantized weight tensor [out_features, in_features] as FP16.
            """
            return _fast_dequant(
                self.weight_fp4_packed, 
                self.scales, 
                self.fp4_group_size
            )
        
        def _dequantize_fp4_input(
            self, 
            x: torch.Tensor, 
            scales: torch.Tensor
        ) -> torch.Tensor:
            """Dequantize FP4 packed input tensor.
            
            Args:
                x: Packed FP4 input tensor [batch, *, in_features//8] as uint32.
                scales: Scale factors. Can be:
                    - [n_groups] for per-layer scales
                    - [batch, n_groups] for per-batch scales
                    - [batch, *, n_groups] for per-token scales
            
            Returns:
                Dequantized FP16 tensor [batch, *, in_features].
            """
            # Convert to uint32 if needed
            x_u32 = x.to(torch.uint32) if x.dtype != torch.uint32 else x
            
            # Get shapes
            batch_shape = x_u32.shape[:-1]
            packed_features = x_u32.shape[-1]
            features = packed_features * 8
            
            device = x_u32.device
            
            # Flatten batch dimensions for processing
            x_flat = x_u32.reshape(-1, packed_features)  # [B, K//8]
            batch_size = x_flat.shape[0]
            
            # Unpack nibbles: [B, K//8] -> [B, K]
            shifts = torch.arange(8, device=device, dtype=torch.int32).mul_(4).view(1, 1, 8)
            words = x_flat.unsqueeze(-1).to(torch.int32)  # [B, K//8, 1]
            nibbles = torch.bitwise_and(torch.bitwise_right_shift(words, shifts), 0xF)  # [B, K//8, 8]
            nibbles = nibbles.reshape(batch_size, features).to(torch.long)  # [B, K]
            
            # Table lookup
            table = _E2M1_TABLE.to(device=device, dtype=torch.float32)
            dequant = table[nibbles]  # [B, K]
            
            # Apply scales
            # Handle different scale shapes
            n_groups = (features + self.fp4_group_size - 1) // self.fp4_group_size
            
            if scales.dim() == 1 and scales.shape[0] == n_groups:
                # Shared scales [n_groups]
                scales_flat = scales.to(torch.float32).unsqueeze(0).expand(batch_size, -1)  # [B, n_groups]
            elif scales.dim() == 2:
                # Per-batch scales [batch, n_groups] or similar
                scales_flat = scales.to(torch.float32).reshape(batch_size, -1)
                if scales_flat.shape[1] != n_groups:
                    raise ValueError(
                        f"Scale dimension mismatch: expected {n_groups} groups, "
                        f"got {scales_flat.shape[1]}"
                    )
            else:
                raise ValueError(
                    f"Unsupported scales shape: {scales.shape}. "
                    f"Expected [n_groups] or [batch, n_groups]"
                )
            
            # Expand scales to feature dimension
            # [B, n_groups] -> [B, K] via group indexing
            group_ids = torch.arange(features, device=device, dtype=torch.long)
            group_ids = torch.div(group_ids, self.fp4_group_size, rounding_mode='floor')
            group_ids = group_ids.clamp(max=n_groups - 1)
            
            expanded_scales = scales_flat[:, group_ids]  # [B, K]
            
            # Apply scales and convert to FP16
            result = (dequant * expanded_scales).to(torch.float16)
            
            # Reshape back to original batch shape
            return result.reshape(*batch_shape, features)
        
        @staticmethod
        def _pack_weights_fp4(
            weight: torch.Tensor, 
            group_size: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Pack FP16 weights into FP4 format.
            
            Args:
                weight: FP16 weight matrix [out_features, in_features].
                group_size: Number of elements per quantization group.
            
            Returns:
                Tuple of (packed_weights, scales):
                - packed_weights: [out_features, in_features//8] uint32
                - scales: [n_groups, out_features] float16
            """
            out_features, in_features = weight.shape
            
            if in_features % 8 != 0:
                raise ValueError(f"in_features ({in_features}) must be divisible by 8")
            if in_features % group_size != 0:
                raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")
            
            n_groups = in_features // group_size
            
            # Get E2M1 table
            table = _E2M1_TABLE.to(weight.device, torch.float32)
            
            # Compute per-group scales (per-row, per-group)
            weight_f32 = weight.to(torch.float32)
            weight_grouped = weight_f32.reshape(out_features, n_groups, group_size)
            absmax = weight_grouped.abs().amax(dim=2)  # [out_features, n_groups]
            absmax = torch.clamp(absmax, min=1e-7)
            max_e2m1 = 6.0  # Maximum representable value in E2M1
            scales = (absmax / max_e2m1).to(torch.float16)  # [out_features, n_groups]
            
            # Normalize
            scales_expanded = scales.unsqueeze(2).expand(-1, -1, group_size).reshape(out_features, in_features)
            normalized = weight_f32 / scales_expanded.to(torch.float32)
            normalized = torch.clamp(normalized, -max_e2m1, max_e2m1)
            
            # Find nearest E2M1 nibble
            # [out_features, in_features, 16] distances
            dists = normalized.unsqueeze(-1).abs().sub(table.unsqueeze(0).unsqueeze(0)).abs()
            nibbles = dists.argmin(dim=-1).to(torch.uint32)  # [out_features, in_features]
            
            # Pack 8 nibbles into uint32 using int64 for CPU compatibility
            packed_n = in_features // 8
            # Use int64 for intermediate packing to avoid uint32 bitwise op limitations on CPU
            packed_i64 = torch.zeros(out_features, packed_n, dtype=torch.int64, device=weight.device)
            
            for i in range(8):
                packed_i64 |= nibbles[:, i::8].to(torch.int64) << (i * 4)
            
            packed = packed_i64.to(torch.uint32)
            
            # Transpose scales to [n_groups, out_features] for kernel compatibility
            scales = scales.transpose(0, 1).contiguous()
            
            return packed, scales
        
        @classmethod
        def from_linear(
            cls,
            linear: nn.Linear,
            fp4_group_size: int = 128,
            use_fp4_weights: bool = False,
        ) -> MixedPrecisionLinear:
            """Convert a torch.nn.Linear to MixedPrecisionLinear.
            
            Copies weights from the source linear layer and quantizes
            to FP4 for the quantized weight path.
            
            Args:
                linear: Source nn.Linear layer.
                fp4_group_size: Group size for FP4 quantization. Default: 128.
                use_fp4_weights: Use FP4 weights for FP16 inputs. Default: False.
            
            Returns:
                MixedPrecisionLinear with copied and quantized weights.
            
            Example:
                >>> linear = nn.Linear(512, 256)
                >>> mixed_layer = MixedPrecisionLinear.from_linear(linear)
            """
            layer = cls(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=linear.bias is not None,
                fp4_group_size=fp4_group_size,
                use_fp4_weights=use_fp4_weights,
            )
            
            with torch.no_grad():
                # Copy BF16 weights
                layer.weight_bf16.copy_(linear.weight.data.to(torch.bfloat16))
                
                # Copy bias
                if linear.bias is not None:
                    layer.bias.copy_(linear.bias.data)
                
                # Re-quantize to FP4
                weight_fp16 = linear.weight.data.to(torch.float16)
                w_packed, scales = cls._pack_weights_fp4(weight_fp16, fp4_group_size)
                layer.weight_fp4_packed.copy_(w_packed)
                layer.scales.copy_(scales)
            
            return layer
        
        def extra_repr(self) -> str:
            """String representation for printing."""
            return (
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"fp4_group_size={self.fp4_group_size}, "
                f"use_fp4_weights={self.use_fp4_weights}"
            )

else:
    # Stub class when PyTorch is not available
    class MixedPrecisionLinear:  # type: ignore[no-redef]
        """Stub for MixedPrecisionLinear when PyTorch is not available.
        
        This class raises RuntimeError on instantiation, allowing the module
        to be imported without PyTorch for type checking purposes.
        """
        
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "MixedPrecisionLinear requires PyTorch. "
                "Install PyTorch with: pip install torch"
            )
        
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("MixedPrecisionLinear requires PyTorch")
        
        @classmethod
        def from_linear(cls, *args: Any, **kwargs: Any) -> MixedPrecisionLinear:
            raise RuntimeError("MixedPrecisionLinear requires PyTorch")


__all__ = ["MixedPrecisionLinear"]
