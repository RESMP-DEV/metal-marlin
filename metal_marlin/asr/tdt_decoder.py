"""TDT (Transducer Dynamic Temperature) decoder for ASR models.

This module implements the complete TDT decoder that combines the predictor
and joint network to perform speech recognition with duration prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TDTConfig
from .tdt_joint import TDTJoint
from .tdt_predictor import TDTPredictor


class TDTDecoder(nn.Module):
    """Complete TDT decoder combining predictor and joint network.

    The TDT decoder is responsible for generating transcription hypotheses from
    encoder outputs using an autoregressive predictor and a joint network.
    It combines acoustic and linguistic information to produce both vocabulary
    logits and duration predictions.

    Args:
        config: TDTConfig containing decoder architecture parameters.

    Attributes:
        predictor: TDTPredictor module for generating label predictions.
        joint: TDTJoint module for combining encoder and predictor outputs.
        blank_id: Index of the blank token used in RNN-T decoding.
    """

    def __init__(self, config: TDTConfig):
        super().__init__()
        self.config = config
        self.predictor = TDTPredictor(config)
        self.joint = TDTJoint(config)
        self.blank_id = config.blank_id

    def forward(
        self,
        encoder_out: torch.Tensor,
        targets: torch.Tensor,
        predictor_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the TDT decoder during training.

        Args:
            encoder_out: Encoder output tensor of shape [B, T, encoder_hidden_size]
                where B is batch size and T is sequence length.
            targets: Target token indices of shape [B, U] where U is the
                number of target tokens. These should include the blank token
                at appropriate positions for teacher forcing.
            predictor_state: Optional LSTM state tuple (h, c) from previous step.
                If provided, each tensor has shape [num_layers, B, predictor_hidden_size].

        Returns:
            A tuple containing:
            - logits: Vocabulary logits of shape [B, T, U, vocab_size]
            - durations: Duration predictions of shape [B, T, U, max_duration+1]
            - predictor_state: Updated predictor LSTM state tuple (h, c)
        """
        # Pass targets through predictor (teacher forcing during training)
        predictor_out, predictor_state = self.predictor(targets, predictor_state)

        # Combine with encoder output through joint network
        logits, durations = self.joint(encoder_out, predictor_out)

        return logits, durations, predictor_state

    def infer(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 100,
        predictor_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], tuple[torch.Tensor, torch.Tensor]]:
        """Inference method for generating hypotheses from encoder outputs.

        This method implements the standard RNN-T inference algorithm where
        the decoder generates hypotheses autoregressively, emitting tokens
        one at a time and maintaining internal state.

        Args:
            encoder_out: Encoder output tensor of shape [B, T, encoder_hidden_size].
            max_len: Maximum number of tokens to generate to prevent infinite loops.
            predictor_state: Optional initial predictor LSTM state.

        Returns:
            A tuple containing:
            - logits: Final logits of shape [B, T, final_U, vocab_size]
            - durations: Final duration predictions of shape [B, T, final_U, max_duration+1]
            - tokens: Generated token sequence as list of indices
            - predictor_state: Final predictor LSTM state
        """
        batch_size, time_steps = encoder_out.size(0), encoder_out.size(1)
        device = encoder_out.device

        # Initialize with blank token
        tokens = torch.full((batch_size, 1), self.blank_id, dtype=torch.long, device=device)
        hypothesis_tokens = [self.blank_id]  # Track emitted tokens

        for step in range(max_len):
            # Get predictor output for current tokens
            predictor_out, predictor_state = self.predictor(tokens, predictor_state)

            # Combine with encoder output
            step_logits, step_durations = self.joint(encoder_out, predictor_out)

            # Get last time step and last utterance position for token selection
            last_logits = step_logits[:, -1, -1, :]  # [B, vocab_size]

            # Sample or take argmax (using greedy for simplicity)
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)  # [B, 1]

            # Append to tokens
            tokens = torch.cat([tokens, next_token], dim=1)

            # Track first batch's tokens for output
            hypothesis_tokens.append(next_token[0, 0].item())

            # Stop if blank token was emitted
            if next_token[0, 0].item() == self.blank_id:
                break

        # Final forward pass to get complete outputs
        predictor_out, predictor_state = self.predictor(tokens, predictor_state)
        logits, durations = self.joint(encoder_out, predictor_out)

        return logits, durations, hypothesis_tokens, predictor_state
