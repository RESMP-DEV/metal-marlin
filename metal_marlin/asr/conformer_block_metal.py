"""Conformer block with Metal backend acceleration.

Implements a single Conformer block using Metal-accelerated components:
- MetalQuantizedLinear FFN layers for improved throughput
- Metal attention kernels for efficient self-attention
- PyTorch convolution (kept as-is since MPS conv is efficient)

This provides maximum performance on Apple Silicon by leveraging Metal
kernels for compute-intensive operations while keeping efficient MPS
operations for convolution.
"""

from typing import Literal

import torch
import torch.nn as nn

from .conformer_attention_metal import ConformerAttentionMetal
from .conformer_config import ConformerConfig
from .conformer_conv import ConformerConvModule
from .conformer_ffn_metal import ConformerFFNMetal


class ConformerBlockMetal(nn.Module):
    """Conformer block with Metal backend acceleration.

    Implements the Macaron-style Conformer block architecture using
    Metal-accelerated components for optimal performance on Apple Silicon:
    - FFN layers use MetalQuantizedLinear for quantized matrix multiplication
    - Attention uses Metal kernels for Q/K/V projections and attention computation
    - Convolution module uses standard PyTorch conv (efficient on MPS)

    Args:
        config: ConformerConfig containing model parameters
        quant_type: Quantization type for Metal-accelerated layers ("fp4" or "int8")
    """

    def __init__(
        self,
        config: ConformerConfig,
        quant_type: Literal["fp4", "int8"] = "fp4",
    ):
        """Initialize ConformerBlockMetal module.

        Args:
            config: ConformerConfig containing model parameters
            quant_type: Quantization type for Metal-accelerated layers
        """
        super().__init__()

        # FFN modules with MetalQuantizedLinear backends
        self.ffn1 = ConformerFFNMetal(config, quant_type)
        self.ffn2 = ConformerFFNMetal(config, quant_type)

        # Multi-head attention with Metal backend
        self.attention = ConformerAttentionMetal(config, quant_type)

        # Convolution module - keep as PyTorch conv (efficient on MPS)
        self.conv = ConformerConvModule(config)

        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of ConformerBlockMetal module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            pos_emb: Positional embeddings of shape (batch_size, seq_len, hidden_size)
            mask: Attention mask of shape (batch_size, seq_len, seq_len) or None

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # First FFN with residual connection (scaled by 0.5)
        # Note: FFN modules handle their own layer norm internally
        x = x + 0.5 * self.ffn1(x)

        # Multi-head self-attention
        # Attention module handles layer norm internally
        x = x + self.attention(x, pos_emb, mask)

        # Convolution module with residual connection
        # Conv module handles layer norm internally
        x = x + self.conv(x)

        # Second FFN with residual connection (scaled by 0.5)
        x = x + 0.5 * self.ffn2(x)

        # Final layer normalization
        return self.final_norm(x)

    def extra_repr(self) -> str:
        """Return string representation for debugging."""
        return "ConformerBlockMetal with Metal-accelerated FFN and attention"
