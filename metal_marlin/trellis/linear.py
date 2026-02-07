"""TrellisLinear: nn.Module for EXL3 trellis-quantized linear layers.

Provides drop-in replacement for nn.Linear with on-the-fly dequantization
using Metal GPU acceleration.

Key optimization: Weights stay compressed on GPU as packed uint8 indices.
Dequantization happens on-the-fly in Metal kernels during forward pass,
reducing GPU memory by ~3x compared to full FP16 storage.

Usage:
    from metal_marlin.trellis.loader import TrellisModelLoader
    from metal_marlin.trellis.linear import TrellisLinear

    loader = TrellisModelLoader("model_dir")
    weight = loader.load_weight("layers.0.mlp.gate_proj")
    linear = TrellisLinear.from_trellis_weight(weight)

    # Forward pass with automatic on-the-fly dequantization
    output = linear(input_tensor)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from ..metal_dispatch import HAS_METAL, HAS_MPS, MetalKernelLibrary, cpu_tensor_to_metal_texture
from ..quantization.trellis_codebook import TrellisCodebook
from .dispatch import (
    dispatch_gemm_trellis_auto,
    dispatch_gemm_trellis_decode,
    dispatch_trellis_dequant_packed,
)

if TYPE_CHECKING:
    from .loader import TrellisWeight


class TrellisLinear(nn.Module):
    """Linear layer with EXL3 trellis-quantized weights.

    Keeps weights compressed on GPU in packed format. Dequantization happens
    on-the-fly during forward pass using fused Metal GPU kernels.

    Memory optimization: Weights stored as packed uint8 indices + scales,
    reducing GPU memory by ~3x compared to storing full FP16 weights.
    Metal kernels read packed data directly and dequantize tile-by-tile.

    The forward() method uses fused Metal kernels for efficient dequantization
    and matrix multiplication without intermediate allocations.

    Attributes:
        in_features: Size of each input sample (K).
        out_features: Size of each output sample (N).
        bits: Quantization bit width (2, 3, 4, or 8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize TrellisLinear.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bits: Quantization bit width (2, 3, or 4).
            bias: If set to True, adds a learnable bias. Default: False.
            device: Device to place parameters on.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Tile dimensions for trellis format
        TILE_DIM = 16
        tiles_k = (in_features + TILE_DIM - 1) // TILE_DIM
        tiles_n = (out_features + TILE_DIM - 1) // TILE_DIM

        # Register buffers (not parameters - these aren't trained)
        packed_bytes = {2: 64, 3: 96, 4: 128, 8: 256}[bits]
        self.register_buffer(
            "packed_indices",
            torch.zeros(tiles_k, tiles_n, packed_bytes, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales",
            torch.ones(
                (in_features + 127) // 128, out_features, dtype=torch.float32, device=device
            ),
        )
        self.register_buffer(
            "su",
            torch.ones(in_features, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "sv",
            torch.ones(out_features, dtype=torch.float32, device=device),
        )

        # Pre-compute codebook grid
        codebook = TrellisCodebook(bits=bits)
        grid = torch.from_numpy(codebook.get_grid()).float()
        self.register_buffer("grid", grid)

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16, device=device),
            )
        else:
            self.bias = None

        # Lazy-loaded Metal library (can be shared via set_lib())
        self._lib: MetalKernelLibrary | None = None

        # Metal buffer cache for CPU->GPU transfer
        self._metal_buffer: dict | None = None
        self._grid_texture: Any | None = None

    def set_lib(self, lib: MetalKernelLibrary) -> None:
        """Set a shared Metal library for this linear layer.

        Call this to share a single MetalKernelLibrary across multiple
        TrellisLinear instances, enabling batch dispatch across all of them.

        Args:
            lib: Shared MetalKernelLibrary instance.
        """
        self._lib = lib

    @classmethod
    def from_trellis_weight(
        cls,
        weight: TrellisWeight,
        bias: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> TrellisLinear:
        """Create TrellisLinear from a loaded TrellisWeight.

        Keeps weights compressed on GPU - stores only packed indices.
        Dequantization happens on-the-fly during forward pass.

        Args:
            weight: TrellisWeight from TrellisModelLoader.
            bias: Optional bias tensor.
            device: Device to place parameters on (default: mps if available).

        Returns:
            TrellisLinear module initialized with compressed weight data.
        """
        # TrellisWeight: K = out_features, N = in_features
        # (weight matrix is [out_features, in_features])
        K, N = weight.original_shape
        out_features, in_features = K, N

        # Use MPS by default if available
        if device is None:
            device = "mps" if HAS_MPS else "cpu"

        # Don't call __init__ which creates wrong-sized buffers
        # Instead, directly create module and set buffers
        module = object.__new__(cls)
        nn.Module.__init__(module)

        module.in_features = in_features
        module.out_features = out_features
        module.bits = weight.bits

        # Store ONLY compressed data on GPU - no unpacking
        # This reduces GPU memory by ~3x compared to full FP16 weights
        module.register_buffer("packed_indices", weight.packed_indices.to(device))
        module.register_buffer("scales", weight.scales.to(device))
        module.register_buffer("su", weight.su.to(device))
        module.register_buffer("sv", weight.sv.to(device))

        # Pre-compute codebook grid (small, shared across all layers)
        codebook = TrellisCodebook(bits=weight.bits)
        grid = torch.from_numpy(codebook.get_grid()).float().to(device)
        module.register_buffer("grid", grid)

        if bias is not None:
            module.register_buffer("bias", bias.to(device))
        else:
            module.bias = None

        # Initialize Metal library reference
        module._lib = None
        module._metal_buffer = None
        module._grid_texture = None

        return module

    def _get_lib(self) -> MetalKernelLibrary:
        """Get or create Metal kernel library."""
        if self._lib is None:
            self._lib = MetalKernelLibrary.from_source_dir()
        return self._lib

    def _get_grid_texture(self) -> Any:
        """Get or create Metal texture for codebook grid."""
        if self._grid_texture is None:
            lib = self._get_lib()
            # Convert grid to texture (must be on CPU)
            # Ensure float32 for texture creation if needed, or half if supported
            # cpu_tensor_to_metal_texture supports both.
            grid_cpu = self.grid.cpu().float().contiguous()
            self._grid_texture = cpu_tensor_to_metal_texture(grid_cpu, lib.device)
        return self._grid_texture

    def _create_metal_buffer_from_cpu(self) -> None:
        """Create Metal buffer directly from CPU tensor data.

        Called once to create the buffer. After this, the PyTorch
        buffer tensors can optionally be deleted to save memory.
        """
        if self._metal_buffer is not None:
            return  # Already created

        from ..metal_dispatch import cpu_tensor_to_metal_buffer

        lib = self._get_lib()

        # Create buffers from CPU copies
        self._metal_buffer = {
            "packed_indices": cpu_tensor_to_metal_buffer(
                self.packed_indices.cpu().contiguous(), lib.device
            ),
            "scales": cpu_tensor_to_metal_buffer(self.scales.cpu().half().contiguous(), lib.device),
            "su": cpu_tensor_to_metal_buffer(self.su.cpu().half().contiguous(), lib.device),
            "sv": cpu_tensor_to_metal_buffer(self.sv.cpu().half().contiguous(), lib.device),
        }

        # Ensure texture is created
        self._get_grid_texture()

    def dequantize(self) -> torch.Tensor:
        """Dequantize weights to FP16.

        Returns:
            Dequantized weights [out_features, in_features] float16.
        """

        # K = out_features, N = in_features (weight matrix convention)
        K, N = self.out_features, self.in_features

        # Compute group_size from scales shape (same logic as forward())
        n_groups = self.scales.shape[0]
        group_size = (self.in_features + n_groups - 1) // n_groups

        if HAS_METAL and HAS_MPS and self.packed_indices.is_mps:
            try:
                lib = self._get_lib()

                weights = dispatch_trellis_dequant_packed(
                    lib,
                    self.packed_indices,  # uint8 packed
                    self.scales,
                    self.grid,  # Grid tensor, not texture
                    self.su,
                    self.sv,
                    K,
                    N,
                    self.bits,
                    group_size,
                )
                return weights
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Metal dequantize failed, falling back to CPU: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # CPU fallback - dequantize packed indices
        return self._dequantize_cpu_packed(K, N, group_size)

    def _dequantize_cpu_packed(self, K: int, N: int, group_size: int) -> torch.Tensor:
        """CPU fallback for dequantizing packed trellis weights.

        Args:
            K: Number of output features.
            N: Number of input features.
            group_size: Quantization group size.

        Returns:
            Dequantized weights [K, N] float16.
        """
        import numpy as np

        # Move tensors to CPU
        packed_indices = self.packed_indices.cpu().numpy()
        scales = self.scales.cpu().float().numpy()
        grid = self.grid.cpu().float().numpy()
        su = self.su.cpu().float().numpy()
        sv = self.sv.cpu().float().numpy()

        # Tile dimensions
        TILE_DIM = 16
        tiles_k = (K + TILE_DIM - 1) // TILE_DIM
        tiles_n = (N + TILE_DIM - 1) // TILE_DIM

        # Allocate output
        output = np.zeros((K, N), dtype=np.float32)

        # Determine how to unpack based on bits
        bits = self.bits
        n_levels = len(grid)

        # Unpack packed_indices to get actual codebook indices
        # packed_indices shape: [tiles_k, tiles_n, packed_bytes]
        # Each tile is 16x16 = 256 elements
        for tile_k in range(tiles_k):
            for tile_n in range(tiles_n):
                packed_tile = packed_indices[tile_k, tile_n]

                # Unpack indices from packed bytes
                indices = self._unpack_tile_indices(packed_tile, bits, n_levels)

                # Dequantize each element in the tile
                for local_k in range(TILE_DIM):
                    for local_n in range(TILE_DIM):
                        k = tile_k * TILE_DIM + local_k
                        n = tile_n * TILE_DIM + local_n

                        if k >= K or n >= N:
                            continue

                        local_offset = local_k * TILE_DIM + local_n
                        idx = int(indices[local_offset])
                        idx = max(0, min(idx, n_levels - 1))

                        # Get scale for this position
                        group_idx = n // group_size
                        if group_idx >= scales.shape[0]:
                            group_idx = scales.shape[0] - 1
                        scale = scales[group_idx, k] if k < scales.shape[1] else 1.0

                        # Dequantize
                        dequant_val = grid[idx] * scale
                        dequant_val *= su[n] * sv[k]

                        output[k, n] = dequant_val

        return torch.from_numpy(output).half()

    def _unpack_tile_indices(
        self, packed_tile: np.ndarray, bits: int, n_levels: int
    ) -> np.ndarray:
        """Unpack packed bytes into codebook indices.

        Args:
            packed_tile: Packed bytes for one tile.
            bits: Quantization bit width (2, 3, or 4).
            n_levels: Number of codebook levels.

        Returns:
            Unpacked indices array of length 256 (16x16 tile).
        """
        # Number of elements per tile
        n_elements = 256  # 16x16

        if bits == 4:
            # 4-bit: 2 indices per byte
            indices = np.zeros(n_elements, dtype=np.uint8)
            for i in range(min(len(packed_tile), n_elements // 2)):
                byte = packed_tile[i]
                indices[i * 2] = byte & 0x0F
                indices[i * 2 + 1] = (byte >> 4) & 0x0F
        elif bits == 3:
            # 3-bit: more complex packing (8 indices in 3 bytes)
            indices = np.zeros(n_elements, dtype=np.uint8)
            byte_idx = 0
            for i in range(0, n_elements, 8):
                if byte_idx + 2 >= len(packed_tile):
                    break
                b0 = packed_tile[byte_idx]
                b1 = packed_tile[byte_idx + 1]
                b2 = packed_tile[byte_idx + 2]
                byte_idx += 3

                indices[i + 0] = b0 & 0x07
                indices[i + 1] = (b0 >> 3) & 0x07
                indices[i + 2] = ((b0 >> 6) | ((b1 & 0x01) << 2)) & 0x07
                indices[i + 3] = (b1 >> 1) & 0x07
                indices[i + 4] = (b1 >> 4) & 0x07
                indices[i + 5] = ((b1 >> 7) | ((b2 & 0x03) << 1)) & 0x07
                indices[i + 6] = (b2 >> 2) & 0x07
                indices[i + 7] = (b2 >> 5) & 0x07
        elif bits == 2:
            # 2-bit: 4 indices per byte
            indices = np.zeros(n_elements, dtype=np.uint8)
            for i in range(min(len(packed_tile), n_elements // 4)):
                byte = packed_tile[i]
                indices[i * 4] = byte & 0x03
                indices[i * 4 + 1] = (byte >> 2) & 0x03
                indices[i * 4 + 2] = (byte >> 4) & 0x03
                indices[i * 4 + 3] = (byte >> 6) & 0x03
        elif bits == 8:
            # 8-bit: direct index storage (1 index per byte, no unpacking needed)
            indices = packed_tile[:min(len(packed_tile), n_elements)]
        else:
            # Unknown bit width, return zeros
            indices = np.zeros(n_elements, dtype=np.uint8)

        return indices

    def clear_metal_resources(self) -> None:
        """Release Metal resources to free GPU memory."""
        self._lib = None
        # Metal buffers are managed by PyObjC, should auto-release

    def clear_pytorch_tensors(self) -> None:
        """Clear PyTorch buffer tensors after Metal buffers are created.

        Call this after _create_metal_buffer_from_cpu() to free the
        PyTorch tensor memory. The Metal buffers will be used for forward.

        Warning: After calling this, dequantize() will not work.
        """
        if self._metal_buffer is None:
            raise RuntimeError(
                "Cannot clear tensors before Metal buffers are created. "
                "Call _create_metal_buffer_from_cpu() first."
            )

        # Replace buffers with tiny placeholders to keep nn.Module happy
        # but free the memory
        self.register_buffer("packed_indices", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("su", torch.empty(0, dtype=torch.float16))
        self.register_buffer("sv", torch.empty(0, dtype=torch.float16))
        # Keep grid as it's small

        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.clear_metal_resources()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly decompression.

        Weights remain compressed on GPU. Metal kernels read packed indices
        directly and dequantize on-the-fly to FP16 during GEMM computation.
        No intermediate FP16 weight matrix is materialized.

        Memory savings: ~3x compared to storing full FP16 weights.

        Args:
            x: Input tensor [..., in_features].

        Returns:
            Output tensor [..., out_features].
        """
        batch_shape = x.shape[:-1]
        # Metal kernels expect float16 input - convert if necessary
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        x_flat = x.reshape(-1, self.in_features)
        M = x_flat.shape[0]

        # Compute group_size from scales shape
        # scales shape is [n_groups, out_features], groups are along in_features dim
        n_groups = self.scales.shape[0]
        if n_groups == 0:
            if self.in_features == 0:
                 group_size = 1 # arbitrary, won't iterate
            else:
                 raise ValueError(f"Invalid scales shape {self.scales.shape} for in_features={self.in_features}")
        else:
            group_size = (self.in_features + n_groups - 1) // n_groups

        # Try Metal dispatch with CPU fallback
        if HAS_METAL and HAS_MPS and x.is_mps:
            try:
                # Get or create Metal library
                lib = self._get_lib()

                # Dispatch to fused dequant+GEMM kernels
                # Kernels read packed_indices directly, dequantize tile-by-tile
                # Choose kernel based on M (decode vs prefill)
                if M <= 16:
                    output = dispatch_gemm_trellis_decode(
                        lib,
                        x_flat,
                        self.packed_indices,  # Packed uint8, never unpacked
                        self.scales,
                        self.grid,  # Codebook grid tensor
                        self.su,
                        self.sv,
                        self.in_features,
                        self.out_features,
                        self.bits,
                        group_size,
                    )
                else:
                    output = dispatch_gemm_trellis_auto(
                        lib,
                        x_flat,
                        self.packed_indices,  # Packed uint8, never unpacked
                        self.scales,
                        self.grid,  # Codebook grid tensor
                        self.su,
                        self.sv,
                        self.in_features,
                        self.out_features,
                        self.bits,
                        group_size,
                    )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Metal TrellisLinear dispatch failed, falling back to CPU: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                output = self._forward_cpu_fallback(x_flat, group_size)
        else:
            # No Metal available or input not on MPS
            if HAS_MPS and x.is_mps:
                import warnings

                warnings.warn(
                    "Metal not available for TrellisLinear, using CPU fallback",
                    RuntimeWarning,
                    stacklevel=2,
                )
            output = self._forward_cpu_fallback(x_flat, group_size)

        output = output.view(*batch_shape, self.out_features)

        if self.bias is not None:
            # Work around PyTorch MPS Metal validation bug where the
            # add_dense_scalar kernel binds a read-only buffer with write access.
            # Using add_ (in-place) avoids allocating output in the kernel.
            output.add_(self.bias)

        return output

    def _forward_cpu_fallback(self, x_flat: torch.Tensor, group_size: int) -> torch.Tensor:
        """CPU fallback for forward pass when Metal dispatch fails.

        Dequantizes weights and performs standard matmul on CPU.

        Args:
            x_flat: Flattened input tensor [M, in_features].
            group_size: Quantization group size.

        Returns:
            Output tensor [M, out_features].
        """
        # Move input to CPU if needed
        x_cpu = x_flat.cpu().float()

        # Dequantize weights using CPU path
        weights = self.dequantize().cpu().float()

        # Standard matmul: x @ W^T (weights are [out_features, in_features])
        output = x_cpu @ weights.t()

        # Move back to original device
        if x_flat.is_mps:
            return output.half().to("mps")
        return output.half()

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, bias={self.bias is not None}"
        )


class TrellisModelWrapper(nn.Module):
    """Wrapper that replaces model Linear layers with TrellisLinear.

    Enables inference with EXL3 trellis-quantized models by replacing
    nn.Linear layers with TrellisLinear on-the-fly.
    """

    def __init__(
        self,
        model: nn.Module,
        model_dir: str | Path,
    ) -> None:
        """Initialize TrellisModelWrapper.

        Args:
            model: Base model architecture (unquantized).
            model_dir: Directory containing quantized model weights.
        """
        super().__init__()
        self.model = model
        self.model_dir = Path(model_dir)

        # Lazy-load weights
        self._loader = None

    def _get_loader(self):
        """Get or create TrellisModelLoader."""
        if self._loader is None:
            from .loader import TrellisModelLoader

            self._loader = TrellisModelLoader(self.model_dir)
        return self._loader

    def replace_linear_layers(self) -> int:
        """Replace all Linear layers with TrellisLinear.

        Returns:
            Number of layers replaced.
        """
        loader = self._get_loader()
        count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if we have quantized weights for this layer
                try:
                    weight = loader.load_weight(name + ".weight")
                except (KeyError, FileNotFoundError):
                    continue

                # Get bias if present
                bias = module.bias.data if module.bias is not None else None

                # Create TrellisLinear
                trellis_linear = TrellisLinear.from_trellis_weight(
                    weight,
                    bias=bias,
                    device="mps" if HAS_MPS else "cpu",
                )

                # Replace in parent module
                parent_name, child_name = self._split_name(name)
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                else:
                    parent = self.model
                setattr(parent, child_name, trellis_linear)
                count += 1

        return count

    @staticmethod
    def _split_name(name: str) -> tuple[str, str]:
        """Split module name into parent and child parts."""
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model."""
        return self.model(*args, **kwargs)
