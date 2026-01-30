"""Build hybrid Parakeet model with Metal Marlin quantization and ANE offload.

This module provides utilities to build hybrid Parakeet ASR models that use:
- Metal Marlin for quantized linear layers (4-bit GEMM)
- Apple Neural Engine (ANE) for convolutional layers
- Mixed precision strategy for optimal performance on Apple Silicon
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    from ..ane.conv_ane import HAS_COREMLTOOLS, ANEConv1d
except ImportError:
    # Fallback if ane module not available
    HAS_COREMLTOOLS = False
    ANEConv1d = None

from .conformer_conv import ConformerConvModule
from .quant_policy import ParakeetQuantPolicy
from .replace_layers import _set_module, replace_parakeet_linear_layers

# Type alias for Parakeet TDT models
ParakeetTDT = nn.Module


class ParakeetQuantizedLoader:
    """Loader for Parakeet ASR models with quantization support."""

    def __init__(self, model_path: Path) -> None:
        """Initialize loader with model path.

        Args:
            model_path: Path to the Parakeet model checkpoint
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

    def load(self, device: str = "mps") -> ParakeetTDT:
        """Load Parakeet model from checkpoint.

        Args:
            device: Device to load the model on (default: "mps")

        Returns:
            Loaded ParakeetTDT model
        """
        # For now, we'll assume the model is saved as a standard PyTorch checkpoint
        # In a full implementation, this would handle different model formats
        checkpoint = torch.load(self.model_path, map_location=device)

        # Extract model state and config
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model_state = checkpoint["model"]
            elif "state_dict" in checkpoint:
                model_state = checkpoint["state_dict"]
            else:
                model_state = checkpoint
        else:
            model_state = checkpoint

        # Create a basic ParakeetTDT model structure
        # In a full implementation, this would use the actual Parakeet model class
        model = self._create_parakeet_model()

        # Load weights
        if isinstance(model_state, dict):
            model.load_state_dict(model_state, strict=False)

        model.to(device)
        model.eval()

        return model

    def _create_parakeet_model(self) -> ParakeetTDT:
        """Create a basic ParakeetTDT model structure.

        Returns:
            Basic ParakeetTDT model
        """

        # This is a placeholder implementation
        # In a full implementation, this would create the actual Parakeet TDT model
        class BasicParakeetTDT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Basic structure for demonstration
                self.encoder = nn.Sequential(
                    nn.Linear(80, 256),  # Input features (e.g., MFCC)
                    nn.ReLU(),
                    nn.Linear(256, 256),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(256, 1024),  # Hidden dimension
                    nn.ReLU(),
                    nn.Linear(1024, 1000),  # Vocabulary size
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        return BasicParakeetTDT()


def build_hybrid_parakeet(
    model_path: Path,
    bits: int = 4,
    use_ane_conv: bool = True,
) -> ParakeetTDT:
    """Build hybrid Parakeet model:
    - Linear layers quantized with Metal Marlin
    - Conv layers compiled to ANE

    Args:
        model_path: Path to the Parakeet model checkpoint
        bits: Quantization bits for linear layers (default: 4)
        use_ane_conv: Whether to compile conv layers to ANE (default: True)

    Returns:
        Hybrid ParakeetTDT model with quantized linear and ANE conv layers
    """
    # Load base model
    loader = ParakeetQuantizedLoader(model_path)
    model = loader.load(device="mps")

    # Replace linear with quantized
    policy = ParakeetQuantPolicy(bits=bits)
    model = replace_parakeet_linear_layers(model, policy)

    # Replace conv with ANE
    if use_ane_conv:
        model = replace_conv_with_ane(model)

    return model


class ConformerConvModuleANE(nn.Module):
    """ANE-accelerated ConformerConvModule.

    This class wraps a ConformerConvModule and compiles its Conv1d layers
    to run on Apple Neural Engine for improved performance.
    """

    def __init__(self, original_module: ConformerConvModule) -> None:
        """Initialize ANE version from existing ConformerConvModule.

        Args:
            original_module: Original ConformerConvModule to convert
        """
        super().__init__()

        # Copy non-conv layers as-is
        self.layer_norm = original_module.layer_norm
        self.batch_norm = original_module.batch_norm
        self.dropout = original_module.dropout

        # Wrap conv layers with ANE
        if HAS_COREMLTOOLS:
            self.pointwise_conv1_ane = ANEConv1d(original_module.pointwise_conv1)
            self.depthwise_conv_ane = ANEConv1d(original_module.depthwise_conv)
            self.pointwise_conv2_ane = ANEConv1d(original_module.pointwise_conv2)
            self.use_ane = True
        else:
            # Fallback to original conv layers if coremltools not available
            self.pointwise_conv1 = original_module.pointwise_conv1
            self.depthwise_conv = original_module.depthwise_conv
            self.pointwise_conv2 = original_module.pointwise_conv2
            self.use_ane = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using ANE-accelerated conv layers.

        Args:
            x: Input tensor [B, T, C]

        Returns:
            Output tensor [B, T, C]
        """
        # Store original shape
        batch_size, seq_len, hidden_size = x.shape

        # Apply LayerNorm
        x = self.layer_norm(x)

        # Transpose for conv1d operations: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # Pointwise convolution + GLU activation
        if self.use_ane:
            x = self.pointwise_conv1_ane(x)  # [B, 2*C, T]
        else:
            x = self.pointwise_conv1(x)  # [B, 2*C, T]

        x, gate = x.chunk(2, dim=1)  # Split into two [B, C, T] tensors
        x = x * torch.sigmoid(gate)  # GLU activation

        # Depthwise convolution
        if self.use_ane:
            x = self.depthwise_conv_ane(x)  # [B, C, T]
        else:
            x = self.depthwise_conv(x)  # [B, C, T]

        # Batch normalization and Swish activation
        x = self.batch_norm(x)  # [B, C, T]
        x = torch.nn.functional.silu(x)  # Swish (SiLU) activation

        # Final pointwise convolution
        if self.use_ane:
            x = self.pointwise_conv2_ane(x)  # [B, C, T]
        else:
            x = self.pointwise_conv2(x)  # [B, C, T]

        # Apply dropout
        x = self.dropout(x)

        # Transpose back: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        return x

    @classmethod
    def from_module(cls, module: ConformerConvModule) -> ConformerConvModuleANE:
        """Create ANE version from existing ConformerConvModule.

        Args:
            module: Original ConformerConvModule

        Returns:
            New ConformerConvModuleANE instance
        """
        return cls(module)


def replace_conv_with_ane(model: ParakeetTDT) -> ParakeetTDT:
    """Replace ConformerConvModule with ANE versions.

    Args:
        model: ParakeetTDT model to modify

    Returns:
        Model with ConformerConvModule replaced by ANE versions
    """
    if not HAS_COREMLTOOLS:
        print("Warning: coremltools not available, skipping ANE conv optimization")
        return model

    replaced_count = 0

    for name, module in model.named_modules():
        if isinstance(module, ConformerConvModule):
            ane_conv = ConformerConvModuleANE.from_module(module)
            _set_module(model, name, ane_conv)
            replaced_count += 1

    print(f"Replaced {replaced_count} ConformerConvModule instances with ANE versions")

    return model
