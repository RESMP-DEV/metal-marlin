"""
Metal-accelerated FP4 E2M1 quantization.

Provides GPU-accelerated quantization to 4-bit E2M1 format (NVFP4/MXFP4),
used by GLM-4 and other models for efficient inference.

The E2M1 format uses:
  - 1 sign bit
  - 2 exponent bits (bias = 1)
  - 1 mantissa bit

This gives 16 representable values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}

Usage:
    from metal_marlin.fp4_metal import FP4Metal

    fp4 = FP4Metal()
    indices, scales = fp4.quantize(tensor, group_size=128)
    reconstructed = fp4.dequantize(indices, scales, group_size=128)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from metal_marlin._compat import (
    E2M1_VALUES,
    E2M1_POSITIVE,
    HAS_TORCH,
    dequantize_e2m1,
    torch,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import Metal

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None  # type: ignore[assignment]

HAS_MPS = False
if HAS_TORCH and torch is not None:
    try:
        HAS_MPS = torch.backends.mps.is_available()
    except AttributeError:
        HAS_MPS = False

_SHADER_DIR = Path(__file__).parent.parent / "src"

# DEPRECATED: These re-exports are kept for backward compatibility.
# Import directly from metal_marlin._compat instead:
#   from metal_marlin._compat import E2M1_VALUES, E2M1_POSITIVE, dequantize_e2m1
#
# These will be removed in a future version.

# Re-export dequantize_e2m1 from _compat for backward compatibility
# (this is the canonical implementation now)
from metal_marlin._compat import dequantize_e2m1  # noqa: F401


class FP4Metal:
    """Metal-accelerated FP4 E2M1 quantization."""

    def __init__(self, device=None):
        """Initialize FP4 Metal dispatcher.

        Args:
            device: Metal device. If None, uses system default.

        Raises:
            RuntimeError: If Metal is not available.
        """
        # Initialize lookup tables (always needed for CPU fallback)
        self._e2m1_table = E2M1_VALUES
        self._e2m1_positive = E2M1_POSITIVE

        self._use_metal = HAS_METAL and HAS_MPS

        if self._use_metal:
            self._device = device or Metal.MTLCreateSystemDefaultDevice()
            self._command_queue = self._device.newCommandQueue()

            # Load shader
            shader_path = _SHADER_DIR / "fp4_quantize.metal"
            if shader_path.exists():
                self._shader_source = shader_path.read_text()
                self._compile_pipelines()
            else:
                # Fall back to CPU if shader not found
                self._use_metal = False

        # Always set up the E2M1 lookup tables (used by both Metal and CPU paths)
        # These are now imported from metal_marlin._compat (unified)
        self._e2m1_table = E2M1_VALUES
        self._e2m1_positive = E2M1_POSITIVE

    @property
    def available(self) -> bool:
        """Return True if Metal FP4 is available and ready."""
        return self._use_metal

    def _compile_pipelines(self):
        """Compile Metal compute pipelines."""
        options = Metal.MTLCompileOptions.new()
        library, error = self._device.newLibraryWithSource_options_error_(
            self._shader_source, options, None
        )
        if library is None:
            raise RuntimeError(f"Failed to compile FP4 shader: {error}")

        # Get pipeline states for each kernel
        self._quantize_fn = library.newFunctionWithName_("fp4_quantize")
        self._dequantize_fn = library.newFunctionWithName_("fp4_dequantize")
        self._pack_fn = library.newFunctionWithName_("fp4_pack_pair")
        self._scale_fn = library.newFunctionWithName_("fp4_compute_scale")

    def quantize(
        self,
        values: torch.Tensor | NDArray,
        group_size: int = 128,
    ) -> tuple[torch.Tensor | NDArray, torch.Tensor | NDArray]:
        """Quantize values to FP4 E2M1 format.

        Args:
            values: Input tensor to quantize
            group_size: Number of elements per scale group

        Returns:
            Tuple of (indices, scales) where:
            - indices: uint8 tensor of 4-bit indices [0-15]
            - scales: float32 tensor of per-group scales
        """
        # Convert to numpy for processing
        if HAS_TORCH and isinstance(values, torch.Tensor):
            v = values.cpu().numpy()
            return_torch = True
            original_device = values.device
        else:
            v = np.asarray(values, dtype=np.float32)
            return_torch = False
            original_device = None

        flat = v.flatten().astype(np.float32)
        n = len(flat)
        n_groups = (n + group_size - 1) // group_size

        # Pad to multiple of group_size
        padded_n = n_groups * group_size
        if padded_n > n:
            flat = np.pad(flat, (0, padded_n - n), mode="constant")

        # Compute per-group scales
        reshaped = flat.reshape(n_groups, group_size)
        scales = np.max(np.abs(reshaped), axis=1) / 6.0  # 6.0 is max E2M1 magnitude
        scales = np.maximum(scales, 1e-8)  # Avoid division by zero

        # Normalize by scales
        normalized = reshaped / scales[:, np.newaxis]

        # Quantize each value to nearest E2M1
        indices = np.empty(padded_n, dtype=np.uint8)
        for i, val in enumerate(normalized.flatten()):
            sign_bit = 8 if val < 0 else 0
            abs_val = abs(val)
            # Find nearest positive E2M1 value
            pos_idx = np.argmin(np.abs(self._e2m1_positive - abs_val))
            indices[i] = sign_bit | pos_idx

        # Trim to original size
        indices = indices[:n].reshape(v.shape)

        if return_torch:
            indices = torch.from_numpy(indices).to(original_device)
            scales = torch.from_numpy(scales.astype(np.float32)).to(original_device)

        return indices, scales

    def dequantize(
        self,
        indices: torch.Tensor | NDArray,
        scales: torch.Tensor | NDArray,
        group_size: int = 128,
    ) -> torch.Tensor | NDArray:
        """Dequantize FP4 indices back to float values.

        Args:
            indices: uint8 tensor of 4-bit indices [0-15]
            scales: Per-group scale factors
            group_size: Elements per scale group

        Returns:
            Reconstructed float tensor
        """
        if HAS_TORCH and isinstance(indices, torch.Tensor):
            # Use PyTorch path to avoid CPU sync
            if scales.device != indices.device:
                scales = scales.to(indices.device)
            
            # Dequantize without scaling
            values = dequantize_e2m1(indices)
            
            # Apply scales
            # Reshape for broadcasting
            original_shape = indices.shape
            flat_values = values.reshape(-1)
            n = flat_values.shape[0]
            n_groups = scales.shape[0] # Assuming scales is 1D or [n_groups]
            
            # This logic assumes flat packing correspondence like the numpy path
            padded_n = n_groups * group_size
            
            # If we need padding to match groups
            if padded_n > n:
                padding = torch.zeros(padded_n - n, device=values.device, dtype=values.dtype)
                flat_values = torch.cat([flat_values, padding])
            
            reshaped_values = flat_values.reshape(n_groups, group_size)
            reshaped_values = reshaped_values * scales.view(-1, 1)
            
            # Trim and reshape back
            result = reshaped_values.flatten()[:n].reshape(original_shape)
            return result
            
        else:
            # Fallback to Numpy path
            if HAS_TORCH and isinstance(indices, torch.Tensor):
                # This branch shouldn't be reached if indices is Tensor, but kept for safety logic structure
                idx = indices.cpu().numpy()
                sc = scales.cpu().numpy()
                return_torch = True
                original_device = indices.device
                original_shape = indices.shape
            else:
                idx = np.asarray(indices, dtype=np.uint8)
                sc = np.asarray(scales, dtype=np.float32)
                return_torch = False
                original_device = None
                original_shape = idx.shape

            flat_idx = idx.flatten()
            n = len(flat_idx)

            # Lookup E2M1 values
            values = self._e2m1_table[flat_idx]

            # Apply scales
            n_groups = len(sc)
            padded_n = n_groups * group_size
            if padded_n > n:
                values = np.pad(values, (0, padded_n - n), mode="constant")

            values = values.reshape(n_groups, group_size)
            values = values * sc[:, np.newaxis]
            values = values.flatten()[:n].reshape(original_shape)

            if return_torch:
                values = torch.from_numpy(values.astype(np.float32)).to(original_device)

            return values

    def pack_pair(
        self,
        lo: torch.Tensor | NDArray,
        hi: torch.Tensor | NDArray,
    ) -> torch.Tensor | NDArray:
        """Pack two 4-bit values into one byte.

        Args:
            lo: Lower 4 bits (indices 0-15)
            hi: Upper 4 bits (indices 0-15)

        Returns:
            Packed bytes: lo | (hi << 4)
        """
        if HAS_TORCH and isinstance(lo, torch.Tensor):
            lo_np = lo.cpu().numpy().astype(np.uint8)
            hi_np = hi.cpu().numpy().astype(np.uint8)
            packed = (lo_np & 0x0F) | ((hi_np & 0x0F) << 4)
            return torch.from_numpy(packed).to(lo.device)
        else:
            lo_np = np.asarray(lo, dtype=np.uint8)
            hi_np = np.asarray(hi, dtype=np.uint8)
            return (lo_np & 0x0F) | ((hi_np & 0x0F) << 4)


# Convenience functions
def quantize_fp4_metal(values, group_size: int = 128):
    """Quantize to FP4 using Metal (convenience function)."""
    fp4 = FP4Metal()
    return fp4.quantize(values, group_size)


def dequantize_fp4_metal(indices, scales, group_size: int = 128):
    """Dequantize FP4 using Metal (convenience function)."""
    fp4 = FP4Metal()
    return fp4.dequantize(indices, scales, group_size)
