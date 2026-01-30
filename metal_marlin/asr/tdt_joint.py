"""TDT (Transducer Dynamic Temperature) joint network for ASR models.

This module implements the joint network that combines encoder and predictor
outputs in RNN-T/Transducer models for speech recognition. The joint network
produces both vocabulary logits and duration predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TDTConfig


class TDTJoint(nn.Module):
    """TDT joint network combining encoder and predictor outputs.

    The joint network takes outputs from the acoustic encoder and the
    predictor (language model) and combines them to produce:
    1. Vocabulary logits for token prediction
    2. Duration predictions for timing information

    Args:
        config: TDTConfig object containing model hyperparameters
    """

    def __init__(self, config: TDTConfig) -> None:
        super().__init__()

        # Add max_duration to config if not present
        if not hasattr(config, "max_duration"):
            config.max_duration = 100  # Default max duration

        self.encoder_proj = nn.Linear(config.encoder_hidden_size, config.joint_hidden_size)
        self.predictor_proj = nn.Linear(config.predictor_hidden_size, config.joint_hidden_size)
        self.output_linear = nn.Linear(config.joint_hidden_size, config.vocab_size)
        # TDT-specific: duration head
        self.duration_head = nn.Linear(config.joint_hidden_size, config.max_duration + 1)

    def forward(
        self, encoder_out: torch.Tensor, predictor_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the TDT joint network.

        Args:
            encoder_out: Encoder output tensor [B, T, enc_dim]
            predictor_out: Predictor output tensor [B, U, pred_dim]

        Returns:
            Tuple containing:
            - logits: Vocabulary logits [B, T, U, vocab_size]
            - durations: Duration predictions [B, T, U, max_duration+1]
        """
        # Project encoder and predictor outputs
        encoder_proj = self.encoder_proj(encoder_out)  # [B, T, joint_hidden]
        predictor_proj = self.predictor_proj(predictor_out)  # [B, U, joint_hidden]

        # Add broadcasting to create joint representation
        # encoder_proj: [B, T, 1, joint_hidden] + predictor_proj: [B, 1, U, joint_hidden]
        joint = encoder_proj.unsqueeze(2) + predictor_proj.unsqueeze(1)

        # Apply tanh activation
        joint = torch.tanh(joint)

        # Generate logits and duration predictions
        logits = self.output_linear(joint)  # [B, T, U, vocab_size]
        durations = self.duration_head(joint)  # [B, T, U, max_duration+1]

        return logits, durations
