"""Conformer block for ASR models.

Implements a single Conformer block with Macaron-style architecture:
FFN → MHSA → Conv → FFN with residual connections and layer normalization.
"""

import torch
import torch.nn as nn

from .conformer_attention import ConformerAttention
from .conformer_config import ConformerConfig
from .conformer_conv import ConformerConvModule
from .conformer_ffn import ConformerFFN


class ConformerBlock(nn.Module):
    """Conformer block module.

    Implements the Macaron-style Conformer block architecture:
    FFN → MHSA → Conv → FFN with residual connections and layer normalization.

    Args:
        config: ConformerConfig containing model parameters
    """

    def __init__(self, config: ConformerConfig):
        """Initialize ConformerBlock module.

        Args:
            config: ConformerConfig containing model parameters
        """
        super().__init__()
        self.ffn1 = ConformerFFN(config)
        self.mhsa = ConformerAttention(config)
        self.conv = ConformerConvModule(config)
        self.ffn2 = ConformerFFN(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConformerBlock module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            pos_emb: Positional embeddings of shape (batch_size, seq_len, hidden_size)
            mask: Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # First FFN with residual connection (scaled by 0.5)
        x = x + 0.5 * self.ffn1(self.norm(x))

        # Multi-head self-attention
        x = x + self.mhsa(self.norm(x), pos_emb, mask)

        # Convolution module
        x = x + self.conv(self.norm(x))

        # Second FFN with residual connection (scaled by 0.5)
        x = x + 0.5 * self.ffn2(self.norm(x))

        # Final layer normalization
        return self.norm(x)
