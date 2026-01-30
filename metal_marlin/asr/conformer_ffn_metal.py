"""Conformer Feed-Forward Network module with Metal Marlin acceleration.

Implements FFN component used in Conformer architecture using Metal Marlin's
custom GEMM kernels for quantized inference.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.gemm_fp4 import GemmFp4
from ..ops.gemm_int8 import GemmInt8
from .conformer_config import ConformerConfig


class ConformerFFNMetal(nn.Module):
    """Conformer Feed-Forward Network with Metal Marlin GEMM backend.

    This module implements FFN component used in Conformer architecture
    using Metal Marlin's custom GEMM kernels for quantized inference.
    It consists of two quantized GEMM layers with a Swish activation
    in between, dropout layers for regularization, and residual scaling.
    """

    def __init__(
        self,
        config: ConformerConfig,
        quant_type: Literal["fp4", "int8"] = "fp4",
    ):
        """Initialize ConformerFFNMetal module.

        Args:
            config: ConformerConfig containing model parameters
            quant_type: Quantization type - "fp4" or "int8"
        """
        super().__init__()
        self.config = config
        self.quant_type = quant_type
        self.hidden_size = config.hidden_size
        self.ffn_intermediate_size = config.ffn_intermediate_size
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Initialize GEMM backend
        self.gemm = GemmFp4() if quant_type == "fp4" else GemmInt8()

        # Placeholder weights - will be set during quantization
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None
        self.group_size = 128

        # For initialization, create regular linear layers as fallback
        self.linear1 = nn.Linear(config.hidden_size, config.ffn_intermediate_size)
        self.linear2 = nn.Linear(config.ffn_intermediate_size, config.hidden_size)

    def set_quantized_weights(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias, group_size=128):
        """Set quantized weights for FFN layers.

        Args:
            fc1_weight: Quantized weights for first linear layer
            fc1_bias: Bias for first linear layer (optional)
            fc2_weight: Quantized weights for second linear layer
            fc2_bias: Bias for second linear layer (optional)
            group_size: Group size used for quantization
        """
        self.fc1_weight = fc1_weight
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConformerFFNMetal module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization
        residual = x
        x = self.layer_norm(x)

        # Use quantized weights if available, otherwise fall back to regular linear layers
        if self.fc1_weight is not None and self.fc2_weight is not None:
            # Quantized forward pass using Metal Marlin GEMM
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, self.hidden_size)

            # fc1: hidden_size -> ffn_intermediate_size
            x = self.gemm.forward(
                x_flat,
                self.fc1_weight,
                self.fc1_weight.scales
                if hasattr(self.fc1_weight, "scales")
                else self.fc1_weight[1],
                self.group_size,
            )

            if self.fc1_bias is not None:
                x = x + self.fc1_bias

            # Swish activation (SiLU)
            x = F.silu(x)

            # Dropout
            x = self.dropout(x)

            # fc2: ffn_intermediate_size -> hidden_size
            x = self.gemm.forward(
                x,
                self.fc2_weight,
                self.fc2_weight.scales
                if hasattr(self.fc2_weight, "scales")
                else self.fc2_weight[1],
                self.group_size,
            )

            if self.fc2_bias is not None:
                x = x + self.fc2_bias

            x = x.view(*batch_shape, self.hidden_size)
        else:
            # Fallback to regular PyTorch linear layers
            x = self.linear1(x)
            x = F.silu(x)  # Swish activation (SiLU)
            x = self.dropout(x)
            x = self.linear2(x)

        x = self.dropout(x)

        # Residual connection with scaling
        return residual + 0.5 * x
