"""Conformer conv module compiled for ANE execution.

This module provides an ANE-accelerated version of the ConformerConvModule
by compiling the entire convolution graph to ANE for efficient inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_config import ConformerConfig

# Check for CoreTools availability for ANE compilation
try:
    import coremltools as ct

    HAS_CORETOOLS = True
except ImportError:
    HAS_CORETOOLS = False
    print("Warning: coremltools not available. ANE compilation will be disabled.")


class ConformerConvModuleANE(nn.Module):
    """
    Conformer conv module compiled for ANE execution.

    Original: pointwise -> GLU -> depthwise -> BN -> Swish -> pointwise
    ANE version: Same ops but compiled to single ANE graph.
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_kernel_size

        # Set device first
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Build the full conv module
        self._module = self._build_module(config)
        # Move internal module to appropriate device
        self._module.to(self._device)

        # Compile to ANE if available
        self._ane_model = None
        if HAS_CORETOOLS:
            self._ane_model = self._compile_to_ane()

    def _build_module(self, config: ConformerConfig) -> nn.Module:
        """Build the complete Conformer convolution module."""

        class ConformerConvGraph(nn.Module):
            """Internal module representing the full conv graph for compilation."""

            def __init__(self, config: ConformerConfig):
                super().__init__()

                # Layer normalization
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
                """Forward pass of the complete conv module."""
                # Store original shape
                batch_size, seq_len, hidden_size = x.shape

                # Apply LayerNorm
                x = self.layer_norm(x)

                # Transpose for conv1d operations: [B, T, C] -> [B, C, T]
                x = x.transpose(1, 2)

                # Pointwise convolution + GLU activation
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

        return ConformerConvGraph(config)

    def _compile_to_ane(self):
        """Compile the module to ANE using CoreTools."""
        if not HAS_CORETOOLS:
            return None

        try:
            # Create example input for tracing
            example_input = torch.randn(1, 100, self.hidden_size)  # [B, T, C]

            # Set module to eval mode for tracing
            self._module.eval()

            # Trace the module
            traced = torch.jit.trace(self._module, example_input)

            # Convert to CoreML model with ANE compute units
            ane_model = ct.convert(
                traced,
                inputs=[ct.TensorType(shape=example_input.shape)],
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )

            return ane_model

        except Exception as e:
            print(f"Warning: Failed to compile to ANE: {e}")
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ANE-compiled Conformer conv module.

        Args:
            x: Input tensor [B, T, C] on MPS device

        Returns:
            Output tensor [B, T, C] on same device as input
        """
        # If ANE model is available and input is appropriate size, use ANE
        if self._ane_model is not None and x.shape[0] == 1:  # ANE works best with batch_size=1
            try:
                # Transfer to CPU, run ANE, transfer back
                x_cpu = x.cpu()

                # Convert input to numpy for CoreML
                x_np = x_cpu.numpy()

                # Run ANE inference
                out_dict = self._ane_model.predict({"input": x_np})
                out_np = out_dict["output"]

                # Convert back to torch tensor and transfer to original device
                return torch.from_numpy(out_np).to(x.device)

            except Exception as e:
                # Fallback to MPS if ANE inference fails
                print(f"Warning: ANE inference failed, falling back to MPS: {e}")

        # Fallback to standard MPS execution
        return self._module(x)

    def extra_repr(self) -> str:
        """Return string representation for debugging."""
        ane_status = "ANE compiled" if self._ane_model else "MPS only"
        return f"hidden_size={self.hidden_size}, kernel_size={self.kernel_size}, mode={ane_status}"


# Factory function for easy instantiation
def create_conformer_conv_ane(config: ConformerConfig) -> ConformerConvModuleANE:
    """Create a ConformerConvModuleANE instance."""
    return ConformerConvModuleANE(config)
