"""Compatibility utilities for ASR module.

Provides feature detection for Metal backends and optional dependencies.
"""

import torch

# Metal backend availability
HAS_MPS = torch.backends.mps.is_available()

# Check for Metal Marlin custom kernels
try:
    from ..ops import kernel_dispatch

    HAS_MARLIN_METAL = hasattr(kernel_dispatch, "dispatch_gemm")
except ImportError:
    HAS_MARLIN_METAL = False

# Check for INT8 GEMM support
try:
    from ..ops.gemm_int8 import GemmInt8

    HAS_INT8_GEMM = True
except ImportError:
    HAS_INT8_GEMM = False

# Check for FP4 GEMM support
try:
    from ..ops.gemm_fp4 import GemmFp4

    HAS_FP4_GEMM = True
except ImportError:
    HAS_FP4_GEMM = False
