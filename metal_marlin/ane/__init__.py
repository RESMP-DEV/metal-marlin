"""ANE (Apple Neural Engine) acceleration for metal_marlin.

This package provides ANE-accelerated implementations of common neural network
operations using coremltools, enabling efficient execution on Apple Silicon.

Key benefit of ANE INT8:
- M4 Max ANE: 38 TOPS INT8 (vs ~27 TFLOPS FP16 on GPU)
- Hardware INT8 execution - no Python dispatch overhead
- Use export_encoder_to_coreml(..., quantize_weights="int8") for best performance
"""

from .conv_ane import ANEConv1d, create_ane_conv1d, is_ane_available, maybe_ane_conv1d
from .export_encoder import ANEEncoder, export_encoder_to_coreml

try:
    import coremltools

    HAS_COREMLTOOLS = True
except ImportError:
    HAS_COREMLTOOLS = False

__all__ = [
    "ANEConv1d",
    "ANEEncoder",
    "HAS_COREMLTOOLS",
    "create_ane_conv1d",
    "export_encoder_to_coreml",
    "is_ane_available",
    "maybe_ane_conv1d",
]
