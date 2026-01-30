"""FP4 GEMM operations using Metal Marlin kernels.

Provides high-level interface for FP4 quantized matrix multiplication
using Metal-optimized kernels for Apple Silicon.
"""

from __future__ import annotations

from typing import Any

import torch

from ..metal_dispatch import HAS_METAL, HAS_MPS, dispatch_gemm_fp4, get_default_library


class GemmFp4:
    """FP4 quantized GEMM operation using Metal Marlin kernels.

    This class provides a high-level interface for performing FP4 quantized
    matrix multiplication using Metal-optimized kernels on Apple Silicon.
    """

    def __init__(self, library: Any | None = None) -> None:
        """Initialize the FP4 GEMM operation.

        Args:
            library: Optional MetalKernelLibrary instance. If None, uses default.
        """
        if not HAS_METAL or not HAS_MPS:
            raise ImportError(
                "FP4 GEMM requires Metal and MPS support. "
                "Install PyObjC: pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
            )

        self._library = library or get_default_library()

    def forward(
        self,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        """Perform FP4 quantized GEMM operation.

        Args:
            x: Input tensor of shape (..., in_features)
            packed_weight: FP4 quantized weight matrix
            scales: Scaling factors for dequantization
            group_size: Group size for quantization

        Returns:
            Output tensor with shape (..., out_features)
        """
        # Ensure input is on MPS device
        if not x.is_mps:
            x = x.to("mps")
        if not packed_weight.is_mps:
            packed_weight = packed_weight.to("mps")
        if not scales.is_mps:
            scales = scales.to("mps")

        # Reshape for 2D GEMM if needed
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        M, K = x.shape
        N = packed_weight.shape[-1] * 8  # FP4 packs 8 values per byte

        # Perform the GEMM operation
        output = dispatch_gemm_fp4(self._library, x, packed_weight, scales, M, N, K, group_size)

        # Restore original shape if needed
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], N)

        return output

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Make the instance callable."""
        return self.forward(*args, **kwargs)
