"""Configuration dataclasses for TDT (Transducer Dynamic Temperature) ASR models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TDTConfig:
    """Configuration for TDT (Transducer Dynamic Temperature) ASR model.

    TDT is a variant of RNN-T (Recurrent Neural Network Transducer) that uses
    a joint network to combine encoder and predictor outputs for speech recognition.

    Attributes:
        vocab_size: Size of the output vocabulary (including blank token).
        predictor_hidden_size: Hidden dimension for the predictor LSTM.
        predictor_num_layers: Number of LSTM layers in the predictor.
        encoder_hidden_size: Hidden dimension of the acoustic encoder.
        joint_hidden_size: Hidden dimension of the joint network.
        blank_id: Index of the blank token for CTC/RNN-T decoding.
        max_duration: Maximum duration value for duration prediction head.
    """

    vocab_size: int = 1024
    predictor_hidden_size: int = 320
    predictor_num_layers: int = 2
    encoder_hidden_size: int = 512
    joint_hidden_size: int = 512
    blank_id: int = 0
    max_duration: int = 100
