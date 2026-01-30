"""Convolutional subsampling module for Conformer ASR models.

Implements 2x Conv2d layers with stride 2 for 4x temporal reduction,
commonly used in Parakeet and other Conformer-based ASR models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_config import ConformerConfig


class ConvSubsampling(nn.Module):
    """Convolutional subsampling module for Conformer ASR models.

    This module reduces the temporal resolution of input mel spectrograms by a factor of 4
    using two 2D convolutional layers with stride 2, while projecting the feature
    dimension to the model's hidden size.
    """

    def __init__(self, config: ConformerConfig):
        """Initialize ConvSubsampling module.

        Args:
            config: ConformerConfig containing model parameters including:
                - n_mels: Number of mel frequency bands (input channels)
                - hidden_size: Model hidden dimension (output channels)
                - dropout: Dropout probability
        """
        super().__init__()

        # First conv layer: input channels = n_mels, output channels = hidden_size
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Mono audio
            out_channels=config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Second conv layer: maintains hidden_size
        self.conv2 = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ConvSubsampling module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_mels)
            lengths: Original sequence lengths before subsampling

        Returns:
            Tuple containing:
            - Output tensor of shape (batch_size, seq_len//4, hidden_size)
            - New sequence lengths after subsampling
        """
        batch_size, seq_len, n_mels = x.shape

        # Add channel dimension: (B, T, n_mels) -> (B, 1, T, n_mels)
        x = x.unsqueeze(1)

        # Apply convolutional layers with ReLU activation
        x = self.conv1(x)  # (B, hidden_size, T//2, n_mels//2)
        x = F.relu(x)

        x = self.conv2(x)  # (B, hidden_size, T//4, n_mels//4)
        x = F.relu(x)

        # Reshape to (B, T//4, hidden_size * (n_mels//4))
        b, c, t, f = x.shape
        x = x.transpose(1, 2).contiguous()  # (B, T//4, hidden_size, n_mels//4)
        x = x.view(b, t, c * f)  # (B, T//4, hidden_size * (n_mels//4))

        # Project to hidden_size using a linear layer
        if c * f != c:  # If feature dimension needs projection
            projection = nn.Linear(c * f, c, device=x.device)
            x = projection(x)

        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Update sequence lengths (floor division by 4)
        new_lengths = torch.div(lengths, 4, rounding_mode="floor")

        return x, new_lengths
