import torch
import torch.nn as nn

from .metal_dispatch import MetalKernelLibrary
from .quantized_loader import QuantizedTensor


class QuantizedLinear(nn.Module):
    """Linear layer with FP4/INT4 quantized weights using Metal kernels."""

    def __init__(
        self,
        quantized_weight: QuantizedTensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.weight_data = quantized_weight.data
        self.weight_scales = quantized_weight.scales
        self.format = quantized_weight.format
        self.group_size = quantized_weight.group_size
        self.out_features, self.in_features = quantized_weight.original_shape
        self.bias = bias

        # Optional Hadamard for QuaRot-style quantization
        self.needs_hadamard = quantized_weight.needs_hadamard
        self.hadamard = quantized_weight.hadamard_matrix

        # Get Metal kernel library
        self._lib = MetalKernelLibrary.get_instance()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, in_features]
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)

        # Apply Hadamard rotation if needed
        if self.needs_hadamard and self.hadamard is not None:
            x_flat = x_flat @ self.hadamard.to(x.device)

        # Dispatch to fused dequant+GEMM kernel
        out = self._dispatch_quantized_gemm(x_flat)

        if self.bias is not None:
            # Work around PyTorch MPS Metal validation bug where the
            # add_dense_scalar kernel binds a read-only buffer with write access.
            # Using add_ (in-place) avoids allocating output in the kernel.
            out.add_(self.bias)

        return out.view(*batch_shape, self.out_features)

    def _dispatch_quantized_gemm(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to appropriate Metal kernel based on format."""
        if self.format == "fp4":
            return self._lib.fp4_gemm(
                x,
                self.weight_data,
                self.weight_scales,
                self.out_features,
                self.in_features,
                self.group_size,
            )
        if self.format == "int4":
            return self._lib.int4_gemm(
                x,
                self.weight_data,
                self.weight_scales,
                self.out_features,
                self.in_features,
                self.group_size,
            )
        raise ValueError(f"Unsupported quantized format: {self.format}")
