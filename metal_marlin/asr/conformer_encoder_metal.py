"""Conformer encoder with Metal Marlin backend for quantized inference.

This module provides the ConformerEncoderMetal class with Metal Marlin integration.
The actual implementation is in quant_int8.py to avoid import conflicts.
"""

# Import the actual implementation
from .quant_int8 import ConformerEncoderMetal

# Re-export for convenience
__all__ = ["ConformerEncoderMetal"]
