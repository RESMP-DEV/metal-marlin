"""Relative positional encoding for Conformer ASR models.

Implements the relative positional encoding used in the original Conformer paper,
which provides position information that can be effectively combined with
relative attention mechanisms.
"""

import math

import torch
import torch.nn as nn


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Conformer ASR models.

    This implementation follows the approach in the original Conformer paper,
    providing relative positional embeddings that can be added to attention
    logits during multi-head self-attention.

    Args:
        config: ConformerConfig containing model parameters including:
            - hidden_size: Model hidden dimension (must be divisible by 2)
            - max_length: Maximum sequence length for positional encoding
    """

    def __init__(self, config, max_length: int = 5000):
        """Initialize RelativePositionalEncoding module.

        Args:
            config: ConformerConfig containing model parameters
            max_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        assert config.hidden_size % 2 == 0, "hidden_size must be even for sinusoidal encoding"

        self.hidden_size = config.hidden_size
        self.max_length = max_length

        # Create sinusoidal positional encoding matrix
        pe = torch.zeros(max_length, config.hidden_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, config.hidden_size, 2).float()
            * (-math.log(10000.0) / config.hidden_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RelativePositionalEncoding module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Positional embedding tensor of shape (batch_size, seq_len, hidden_size)

        Note:
            When tracing for CoreML/ANE, seq_len must be <= max_length.
            The assertion is skipped during tracing for compatibility.
        """
        seq_len = x.size(1)
        # Skip assertion during tracing for CoreML compatibility
        if not torch.jit.is_tracing():
            assert seq_len <= self.max_length, (
                f"Sequence length {seq_len} exceeds max_length {self.max_length}"
            )

        return self.pe[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
