"""Conformer Feed-Forward Network module.

Implements the FFN component used in Conformer architecture:
Linear → Swish → Dropout → Linear → Dropout with residual scaling (0.5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_config import ConformerConfig


class ConformerFFN(nn.Module):
    """Conformer Feed-Forward Network module.

    This module implements the FFN component used in the Conformer architecture.
    It consists of two linear layers with a Swish activation in between,
    dropout layers for regularization, and residual scaling.
    """

    def __init__(self, config: ConformerConfig):
        """Initialize the ConformerFFN module.

        Args:
            config: ConformerConfig containing model parameters
        """
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.ffn_intermediate_size)
        self.linear2 = nn.Linear(config.ffn_intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConformerFFN module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization
        residual = x
        x = self.layer_norm(x)

        # Linear → Swish → Dropout → Linear → Dropout
        x = self.linear1(x)
        x = F.silu(x)  # Swish activation (SiLU)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        # Residual connection with scaling
        return residual + 0.5 * x
