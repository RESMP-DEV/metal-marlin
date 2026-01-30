"""TDT (Transducer Dynamic Temperature) predictor module for ASR.

The predictor is an autoregressive language model that generates label predictions
based on previously emitted tokens. In RNN-T/TDT architecture, it works together
with an acoustic encoder and a joint network to perform streaming speech recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TDTConfig


class TDTPredictor(nn.Module):
    """Autoregressive predictor for TDT.

    The predictor is a simple LSTM-based language model that takes previously
    emitted tokens and produces hidden representations to be combined with
    acoustic encoder outputs by the joint network.

    Args:
        config: TDTConfig containing predictor architecture parameters.

    Attributes:
        embedding: Token embedding layer mapping vocab indices to hidden dim.
        lstm: Multi-layer LSTM for processing token sequences.
    """

    def __init__(self, config: TDTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.predictor_hidden_size)
        self.lstm = nn.LSTM(
            config.predictor_hidden_size,
            config.predictor_hidden_size,
            num_layers=config.predictor_num_layers,
            batch_first=True,
        )

    def forward(
        self, tokens: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the predictor.

        Args:
            tokens: Input token indices of shape [B, U] where B is batch size
                and U is the number of previous tokens (label sequence length).
            state: Optional LSTM hidden state tuple (h, c). If provided, each
                tensor has shape [num_layers, B, predictor_hidden_size].
                If None, the LSTM starts with zero states.

        Returns:
            A tuple containing:
            - output: Hidden representations of shape [B, U, predictor_hidden_size]
            - new_state: Updated LSTM state tuple (h, c) for continuation
        """
        # Embed tokens: [B, U] -> [B, U, predictor_hidden_size]
        embedded = self.embedding(tokens)

        # LSTM forward: [B, U, H] -> [B, U, H], plus state
        output, new_state = self.lstm(embedded, state)

        return output, new_state
