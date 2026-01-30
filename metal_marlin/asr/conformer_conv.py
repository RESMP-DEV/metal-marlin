"""Conformer convolution module for ASR models.

This module implements the convolutional submodule used in Conformer encoder layers.
The module consists of: LayerNorm → Pointwise Conv → GLU → Depthwise Conv →
BatchNorm → Swish → Pointwise Conv → Dropout.

This module is designed to be offloaded to ANE (Apple Neural Engine) in the
hybrid approach for efficient inference on Apple Silicon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_config import ConformerConfig


class ConformerConvModule(nn.Module):
    """Conformer convolution module.

    Implements the convolutional submodule used in Conformer encoder layers.
    The module follows the architecture: LayerNorm → Pointwise Conv → GLU →
    Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout.

    Args:
        config: ConformerConfig object containing model hyperparameters

    Input shape:
        x: [B, T, C] where B=batch_size, T=sequence_length, C=hidden_size

    Output shape:
        [B, T, C] with the same dimensions as input
    """

    def __init__(self, config: ConformerConfig) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # First pointwise convolution (expands to 2x hidden_size for GLU)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size, config.hidden_size * 2, kernel_size=1, bias=True
        )

        # Depthwise convolution with proper padding
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,  # Same padding
            groups=config.hidden_size,  # Depthwise
            bias=True,
        )

        # Batch normalization for depthwise conv output
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)

        # Second pointwise convolution (projects back to hidden_size)
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size=1, bias=True
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conformer convolution module.

        Args:
            x: Input tensor [B, T, C] where B=batch_size, T=time_steps, C=channels

        Returns:
            Output tensor [B, T, C] with same shape as input
        """
        # Store original shape
        batch_size, seq_len, hidden_size = x.shape

        # Apply LayerNorm
        x = self.layer_norm(x)

        # Transpose for conv1d operations: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # Pointwise convolution + GLU activation
        # GLU: split in half, apply sigmoid to second half, multiply element-wise
        x = self.pointwise_conv1(x)  # [B, 2*C, T]
        x, gate = x.chunk(2, dim=1)  # Split into two [B, C, T] tensors
        x = x * torch.sigmoid(gate)  # GLU activation

        # Depthwise convolution
        x = self.depthwise_conv(x)  # [B, C, T]

        # Batch normalization and Swish activation
        x = self.batch_norm(x)  # [B, C, T]
        x = F.silu(x)  # Swish (SiLU) activation

        # Final pointwise convolution
        x = self.pointwise_conv2(x)  # [B, C, T]

        # Apply dropout
        x = self.dropout(x)

        # Transpose back: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        return x

    def extra_repr(self) -> str:
        """Return string representation for debugging."""
        return (
            f"hidden_size={self.pointwise_conv2.out_channels}, "
            f"kernel_size={self.depthwise_conv.kernel_size[0]}"
        )
