"""ANE-accelerated Conv1d implementation using coremltools.

This module provides ANEConv1d, a wrapper that executes Conv1d operations
on Apple Neural Engine while maintaining PyTorch compatibility.

The implementation uses coremltools to compile PyTorch Conv1d layers to
Core ML models that can execute on ANE.
"""

from __future__ import annotations

from typing import Any

from .._compat import HAS_TORCH, torch

# Feature flag for coremltools availability
HAS_COREMLTOOLS: bool = False

# Module reference for coremltools
coremltools: Any = None

# Try importing coremltools
try:
    import coremltools as _coremltools  # type: ignore
    from coremltools.models.neural_network import NeuralNetworkBuilder  # type: ignore

    coremltools = _coremltools
    HAS_COREMLTOOLS = True
except ImportError:
    NeuralNetworkBuilder = None  # type: ignore
    pass


class ANEConv1d:
    """Conv1d that executes on Apple Neural Engine.

    This class wraps a PyTorch Conv1d and compiles it to run on ANE
    using coremltools. The original Conv1d is kept as fallback.
    """

    def __init__(self, conv: Any):
        """Initialize ANE Conv1d wrapper.

        Args:
            conv: PyTorch nn.Conv1d module to wrap

        Raises:
            RuntimeError: If coremltools or PyTorch is not available
            ValueError: If conv is not a Conv1d module
        """
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch is required for ANEConv1d")
        if not HAS_COREMLTOOLS or coremltools is None:
            raise RuntimeError("coremltools is required for ANEConv1d")

        if not hasattr(conv, "in_channels") or not hasattr(conv, "out_channels"):
            raise ValueError("conv must be a Conv1d module")

        self._conv = conv  # Keep for fallback
        self._ane_model = self._compile_to_ane(conv)
        self._input_name = "input"
        self._output_name = "output"

    @staticmethod
    def _compile_to_ane(conv: Any) -> Any:
        """Compile a Conv1d module to ANE execution.

        Args:
            conv: PyTorch nn.Conv1d module

        Returns:
            Compiled Core ML model
        """
        if not HAS_COREMLTOOLS or coremltools is None:
            raise RuntimeError("coremltools not available")

        # Set conv to evaluation mode for tracing
        conv.eval()

        # Create example input for tracing
        example_input = torch.randn(1, conv.in_channels, 1000)

        # Trace the conv operation
        with torch.no_grad():
            traced = torch.jit.trace(conv, example_input)

        # Convert to Core ML with ANE target
        mlmodel = coremltools.convert(
            traced,
            inputs=[
                coremltools.TensorType(
                    name="input", shape=(1, conv.in_channels, coremltools.RangeDim(1, 10000))
                )
            ],
            compute_units=coremltools.ComputeUnit.CPU_AND_NE,  # Force ANE usage
            minimum_deployment_target=coremltools.target.macOS13,
        )

        return mlmodel

    def forward(self, x: Any) -> Any:
        """Forward pass through ANE-accelerated Conv1d.

        Args:
            x: Input tensor of shape (batch_size, in_channels, length)

        Returns:
            Output tensor of shape (batch_size, out_channels, new_length)

        Note:
            Falls back to PyTorch Conv1d if ANE execution fails
        """
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch not available")

        # Ensure input is on CPU for Core ML processing
        if x.is_cuda:
            x_cpu = x.cpu()
        else:
            x_cpu = x

        try:
            # Execute on ANE via Core ML
            input_dict = {self._input_name: x_cpu.numpy()}
            output_dict = self._ane_model.predict(input_dict)
            result = torch.from_numpy(output_dict[self._output_name])

            # Move result back to original device
            if x.is_cuda:
                result = result.to(x.device)

            return result

        except Exception:
            # Fallback to original PyTorch conv if ANE fails
            return self._conv(x)

    def __call__(self, x: Any) -> Any:
        """Make the instance callable like a PyTorch module."""
        return self.forward(x)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped Conv1d module.

        This allows accessing Conv1d properties like weight, bias, etc.
        """
        if name.startswith("_"):
            # Don't delegate private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._conv, name)


def create_ane_conv1d(conv: Any) -> ANEConv1d:
    """Create an ANE-accelerated Conv1d from a PyTorch Conv1d.

    Args:
        conv: PyTorch nn.Conv1d module

    Returns:
        ANEConv1d wrapper instance

    Raises:
        ImportError: If coremltools is not available
        RuntimeError: If PyTorch is not available
        ValueError: If conv is not a valid Conv1d module
    """
    return ANEConv1d(conv)


def is_ane_available() -> bool:
    """Check if ANE acceleration is available.

    Returns:
        True if both coremltools and PyTorch are available
    """
    return HAS_COREMLTOOLS and HAS_TORCH


# Convenience function for conditional ANE usage
def maybe_ane_conv1d(conv: Any, use_ane: bool = True) -> Any:
    """Conditionally wrap Conv1d with ANE acceleration.

    Args:
        conv: PyTorch nn.Conv1d module
        use_ane: Whether to use ANE if available (default: True)

    Returns:
        Either ANEConv1d wrapper or original conv based on availability and use_ane
    """
    if use_ane and is_ane_available():
        try:
            return ANEConv1d(conv)
        except Exception:
            # Fallback to original if ANE setup fails
            pass
    return conv
