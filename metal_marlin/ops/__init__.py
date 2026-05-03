"""Metal Marlin GEMM operations.

Provides high-level interfaces for quantized GEMM operations using
Metal-optimized kernels for Apple Silicon.
"""
import logging

from .gemm_fp4 import GemmFp4
from .gemm_int8 import GemmInt8, pack_int8_weights
from .metal_linear import MetalQuantizedLinear


logger = logging.getLogger(__name__)

__all__ = ["GemmFp4", "GemmInt8", "pack_int8_weights", "MetalQuantizedLinear"]
