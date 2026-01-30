"""Depthwise Conv1d optimized for ANE.

This module provides ANEDepthwiseConv1d, a specialized implementation
of depthwise separable convolution optimized for Apple Neural Engine.
ANE is particularly efficient for depthwise operations since each
input channel has its own dedicated filter.
"""

from __future__ import annotations

from typing import Any

from .._compat import HAS_TORCH, torch


class ANEDepthwiseConv1d:
    """Depthwise conv1d optimized for ANE.

    Depthwise convolution uses groups=in_channels, meaning each input
    channel has its own filter. This operation is highly efficient on ANE
    due to the parallel nature of the processing.
    """

    def __init__(self, in_channels: int, kernel_size: int, padding: int):
        """Initialize ANE-optimized depthwise convolution.

        Args:
            in_channels: Number of input channels (also equals groups)
            kernel_size: Size of the convolution kernel
            padding: Amount of zero padding applied to both sides

        Note:
            This creates a depthwise conv with groups=in_channels
        """
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch is required for ANEDepthwiseConv1d")

        self.in_channels = in_channels
        self.out_channels = in_channels  # Depthwise keeps same channel count
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = in_channels  # Depthwise convolution

        # Create the underlying depthwise Conv1d
        self._conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as in_channels for depthwise
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,  # Key: groups=in_channels for depthwise
            bias=False,  # Typically no bias in depthwise conv
        )

    @classmethod
    def from_conv(cls, conv: Any) -> ANEDepthwiseConv1d:
        """Create ANEDepthwiseConv1d from existing Conv1d.

        Args:
            conv: PyTorch Conv1d module that must be depthwise (groups=in_channels)

        Returns:
            ANEDepthwiseConv1d instance with copied weights

        Raises:
            AssertionError: If conv is not depthwise (groups != in_channels)
            ValueError: If conv doesn't have required attributes
        """
        if not hasattr(conv, "in_channels") or not hasattr(conv, "groups"):
            raise ValueError("conv must have in_channels and groups attributes")

        assert conv.groups == conv.in_channels, "Must be depthwise conv (groups == in_channels)"

        # Create new depthwise conv
        depthwise_conv = cls(
            in_channels=conv.in_channels,
            kernel_size=conv.kernel_size[0]
            if hasattr(conv.kernel_size, "__iter__")
            else conv.kernel_size,
            padding=conv.padding[0] if hasattr(conv.padding, "__iter__") else conv.padding,
        )

        # Copy weights if available
        if hasattr(conv, "weight") and conv.weight is not None:
            depthwise_conv._conv.weight.data.copy_(conv.weight.data)

        # Copy bias if available
        if hasattr(conv, "bias") and conv.bias is not None:
            if depthwise_conv._conv.bias is not None:
                depthwise_conv._conv.bias.data.copy_(conv.bias.data)

        return depthwise_conv

    def forward(self, x: Any) -> Any:
        """Forward pass through depthwise convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, length)

        Returns:
            Output tensor of shape (batch_size, out_channels, new_length)
        """
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch not available")

        return self._conv(x)

    def __call__(self, x: Any) -> Any:
        """Make instance callable like PyTorch module."""
        return self.forward(x)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying Conv1d."""
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._conv, name)


def create_ane_depthwise_conv1d(
    in_channels: int, kernel_size: int, padding: int = 0
) -> ANEDepthwiseConv1d:
    """Create an ANE-optimized depthwise Conv1d.

    Args:
        in_channels: Number of input/output channels
        kernel_size: Size of convolution kernel
        padding: Amount of zero padding (default: 0)

    Returns:
        ANEDepthwiseConv1d instance
    """
    return ANEDepthwiseConv1d(in_channels, kernel_size, padding)


def maybe_ane_depthwise_conv1d(conv: Any, use_ane: bool = True) -> Any:
    """Conditionally convert Conv1d to ANE-optimized depthwise if applicable.

    Args:
        conv: PyTorch Conv1d module
        use_ane: Whether to use ANE optimization (default: True)

    Returns:
        ANEDepthwiseConv1d if conv is depthwise and use_ane=True, otherwise original conv
    """
    if not use_ane:
        return conv

    if not hasattr(conv, "groups") or not hasattr(conv, "in_channels"):
        return conv

    # Check if it's already a depthwise convolution
    if conv.groups == conv.in_channels:
        try:
            return ANEDepthwiseConv1d.from_conv(conv)
        except Exception:
            # Fallback to original if conversion fails
            pass

    return conv
