"""Metal-accelerated quantized Linear layer.

Replaces nn.Linear with a version that uses our custom Metal GEMM kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gemm_fp4 import GemmFp4
from .gemm_int8 import GemmInt8, pack_int8_weights


class MetalQuantizedLinear(nn.Module):
    """Linear layer using custom Metal GEMM kernels.

    Supports INT8 (W8A16) and FP4 quantization with our handwritten
    Metal shaders instead of generic MPS operations.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias (default: True)
        quant_type: Quantization type ("int8", "fp4", or None for FP16)
        group_size: Quantization group size (default: 128)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str | None = "int8",
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.group_size = group_size

        if quant_type == "int8":
            # Packed INT8 weights: [K//4, N] as uint32
            self.register_buffer(
                "weight_packed", torch.zeros(in_features // 4, out_features, dtype=torch.uint32)
            )
            # Scales: [K//group_size, N]
            n_groups = (in_features + group_size - 1) // group_size
            self.register_buffer("scales", torch.ones(n_groups, out_features, dtype=torch.float16))
            self.register_buffer("zeros", torch.zeros(n_groups, out_features, dtype=torch.float16))
            self._gemm = GemmInt8()

        elif quant_type == "fp4":
            # FP4 packed weights
            self.register_buffer(
                "weight_packed", torch.zeros(in_features // 2, out_features, dtype=torch.uint8)
            )
            self.register_buffer(
                "scales",
                torch.ones(
                    (in_features + group_size - 1) // group_size, out_features, dtype=torch.float16
                ),
            )
            self._gemm = GemmFp4()

        else:
            # FP16 weights (no quantization)
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self._gemm = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quant_type: str = "int8",
        group_size: int = 128,
        scales: torch.Tensor | None = None,
        zeros: torch.Tensor | None = None,
    ) -> MetalQuantizedLinear:
        """Create MetalQuantizedLinear from existing nn.Linear.

        Args:
            linear: Source nn.Linear layer
            quant_type: Quantization type
            group_size: Quantization group size
            scales: Pre-computed scales (optional, will calibrate if None)
            zeros: Pre-computed zeros (optional)

        Returns:
            Quantized MetalQuantizedLinear layer
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            quant_type=quant_type,
            group_size=group_size,
        )

        # Quantize weights
        weight = linear.weight.data.T.contiguous()  # [in, out]

        if quant_type == "int8":
            if scales is None:
                # Simple per-channel quantization
                scales = weight.abs().max(dim=0, keepdim=True)[0] / 127.0
                # Reshape scales to match [K//group_size, N]
                n_groups = (layer.in_features + group_size - 1) // group_size
                scales = scales.expand(n_groups, layer.out_features)

            # Quantize and pack
            # Use scales mean for uniform quantization
            scale_val = scales.mean(dim=0, keepdim=True)
            weight_int8 = torch.clamp(weight / scale_val, -128, 127).to(torch.int8)
            # weight is [in, out], pack_int8_weights expects [K, N] where K is input dim
            # So we can use weight directly, but need to ensure correct orientation
            layer.weight_packed.copy_(pack_int8_weights(weight_int8))
            layer.scales.copy_(scales.half())
            if zeros is not None:
                layer.zeros.copy_(zeros.half())

        elif quant_type == "fp4":
            # FP4 quantization (simplified)
            if scales is None:
                scales = weight.abs().max(dim=0, keepdim=True)[0] / 7.0
            # Pack to FP4 (implementation in gemm_fp4.py)
            from .gemm_fp4 import pack_fp4_weights

            layer.weight_packed.copy_(pack_fp4_weights(weight, scales))
            layer.scales.copy_(scales.half())

        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Metal GEMM kernel.

        Args:
            x: Input tensor [*, in_features]

        Returns:
            Output tensor [*, out_features]
        """
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        if self._gemm is not None:
            # Use our custom Metal kernel
            if self.quant_type == "int8":
                out = self._gemm.forward(
                    x_2d.half(),
                    self.weight_packed,
                    self.scales,
                    self.zeros if hasattr(self, "zeros") else None,
                    self.group_size,
                )
            else:  # fp4
                out = self._gemm.forward(
                    x_2d.half(),
                    self.weight_packed,
                    self.scales,
                    self.group_size,
                )
        else:
            # FP16 fallback
            out = torch.nn.functional.linear(x_2d, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out.view(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant_type={self.quant_type}"
        )
