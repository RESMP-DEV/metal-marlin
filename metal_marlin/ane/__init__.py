"""ANE (Apple Neural Engine) acceleration for metal_marlin.

This package provides ANE-accelerated implementations of common neural network
operations using coremltools, enabling efficient execution on Apple Silicon.
"""

from .conv_ane import ANEConv1d, create_ane_conv1d, is_ane_available, maybe_ane_conv1d

__all__ = [
    "ANEConv1d",
    "create_ane_conv1d",
    "is_ane_available",
    "maybe_ane_conv1d",
]
