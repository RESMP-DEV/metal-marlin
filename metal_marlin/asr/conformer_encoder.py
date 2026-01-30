"""Conformer encoder for ASR models.

Implements the complete Conformer encoder architecture including:
- Convolutional subsampling
- Relative positional encoding
- Stack of Conformer blocks

The Conformer encoder processes mel spectrograms to produce contextual
embeddings suitable for speech recognition tasks.
"""

import torch
import torch.nn as nn

from .conformer_block import ConformerBlock
from .conformer_config import ConformerConfig
from .positional_encoding import RelativePositionalEncoding
from .subsampling import ConvSubsampling


class ConformerEncoder(nn.Module):
    """Conformer encoder for ASR models.

    Implements the complete Conformer encoder architecture that processes
    mel spectrograms through convolutional subsampling, applies relative
    positional encoding, and processes the sequence through multiple
    Conformer blocks.

    Args:
        config: ConformerConfig containing model parameters including:
            - num_layers: Number of Conformer blocks
            - hidden_size: Model hidden dimension
            - num_attention_heads: Number of attention heads
            - ffn_intermediate_size: FFN intermediate size
            - conv_kernel_size: Convolution kernel size
            - dropout: Dropout probability
    """

    def __init__(self, config: ConformerConfig):
        """Initialize ConformerEncoder module.

        Args:
            config: ConformerConfig containing model parameters
        """
        super().__init__()
        self.config = config

        # Convolutional subsampling layer
        self.subsampling = ConvSubsampling(config)

        # Relative positional encoding
        self.pos_enc = RelativePositionalEncoding(config)

        # Stack of Conformer blocks
        self.layers = nn.ModuleList([ConformerBlock(config) for _ in range(config.num_layers)])

    def forward(
        self, mel: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ConformerEncoder module.

        Args:
            mel: Input mel spectrogram tensor of shape (batch_size, seq_len, n_mels)
            lengths: Original sequence lengths before subsampling

        Returns:
            Tuple containing:
            - Output tensor of shape (batch_size, seq_len//4, hidden_size)
            - Updated sequence lengths after subsampling
        """
        # Apply convolutional subsampling
        # Input: mel (B, T, n_mels)
        # Output: x (B, T//4, hidden_size), lengths (B,)
        x, lengths = self.subsampling(mel, lengths)

        # Add relative positional encoding
        pos_emb = self.pos_enc(x)

        # Process through Conformer blocks
        for layer in self.layers:
            x = layer(x, pos_emb, mask=None)

        return x, lengths
