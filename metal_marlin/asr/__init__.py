"""ASR (Automatic Speech Recognition) module for MetalMarlin.

Provides components for TDT (Transducer Dynamic Temperature) speech recognition:
- TDTPredictor: Autoregressive label predictor (LSTM-based language model)
- TDTJoint: Joint network combining encoder and predictor outputs
- TDTConfig: Configuration for TDT models
"""

from __future__ import annotations

from .config import TDTConfig
from .tdt_joint import TDTJoint
from .tdt_predictor import TDTPredictor

__all__ = ["TDTConfig", "TDTJoint", "TDTPredictor"]
