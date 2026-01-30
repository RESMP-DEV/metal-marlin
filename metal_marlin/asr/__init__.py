"""ASR (Automatic Speech Recognition) module for MetalMarlin.

Provides components for TDT (Transducer Dynamic Temperature) speech recognition:
- ConformerEncoder: Complete Conformer encoder with subsampling and positional encoding
- ConformerBlock: Individual Conformer block with Macaron-style architecture
- ConvSubsampling: Convolutional subsampling for temporal reduction
- RelativePositionalEncoding: Sinusoidal relative positional encoding
- TDTPredictor: Autoregressive label predictor (LSTM-based language model)
- TDTJoint: Joint network combining encoder and predictor outputs
- TDTConfig: Configuration for TDT models
- ConformerConfig: Configuration for Conformer models
"""

from __future__ import annotations

from .config import TDTConfig
from .conformer_block import ConformerBlock
from .conformer_config import ConformerConfig
from .conformer_encoder import ConformerEncoder
from .conformer_encoder_metal import ConformerEncoderMetal as ConformerEncoderMetalAdvanced
from .parakeet_loader import ParakeetQuantizedLoader, load_quantized_state_dict
from .parakeet_model import ParakeetTDT
from .positional_encoding import RelativePositionalEncoding
from .quant_int8 import (
    ConformerEncoderMetal,
    calibrate_int8_scales,
    pack_linear_to_int8,
    quantize_conformer_to_int8,
)
from .subsampling import ConvSubsampling
from .tdt_joint import TDTJoint
from .tdt_predictor import TDTPredictor

__all__ = [
    "TDTConfig",
    "TDTJoint",
    "TDTPredictor",
    "ConformerConfig",
    "ConformerEncoder",
    "ConformerBlock",
    "ConvSubsampling",
    "RelativePositionalEncoding",
    "ParakeetQuantizedLoader",
    "ParakeetTDT",
    "load_quantized_state_dict",
    "ConformerEncoderMetal",
    "ConformerEncoderMetalAdvanced",
    "calibrate_int8_scales",
    "pack_linear_to_int8",
    "quantize_conformer_to_int8",
]
