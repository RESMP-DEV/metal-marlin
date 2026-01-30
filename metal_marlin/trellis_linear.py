"""TrellisLinear: nn.Module for EXL3 trellis-quantized linear layers.

Provides drop-in replacement for nn.Linear with on-the-fly dequantization
using Metal GPU acceleration.

Usage:
    from metal_marlin.trellis_loader import TrellisModelLoader
    from metal_marlin.trellis_linear import TrellisLinear

    loader = TrellisModelLoader("model_dir")
    weight = loader.load_weight("layers.0.mlp.gate_proj")
    linear = TrellisLinear.from_trellis_weight(weight)

    # Forward pass with automatic dequantization
    output = linear(input_tensor)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .metal_dispatch import HAS_METAL, HAS_MPS, MetalKernelLibrary
from .quantization.trellis_codebook import TrellisCodebook
from .trellis_dispatch import (
    dequantize_trellis_weight,
    dispatch_trellis_dequant_packed,
)

if TYPE_CHECKING:
    from .trellis_loader import TrellisWeight


class TrellisLinear(nn.Module):
    """Linear layer with EXL3 trellis-quantized weights.

    Stores weights in compressed trellis format and dequantizes on-the-fly
    during forward pass using Metal GPU acceleration.

    Attributes:
        in_features: Size of each input sample (K).
        out_features: Size of each output sample (N).
        bits: Quantization bit width (2, 3, or 4).
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
        packed_bytes = {2: 64, 3: 96, 4: 128}[bits]
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

        # Lazy-loaded Metal library
        self._lib: MetalKernelLibrary | None = None
        self._dequantized_cache: torch.Tensor | None = None

    @classmethod
    def from_trellis_weight(
        cls,
        weight: TrellisWeight,
        bias: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> TrellisLinear:
        """Create TrellisLinear from a loaded TrellisWeight.

        Args:
            weight: TrellisWeight from TrellisModelLoader.
            bias: Optional bias tensor.
            device: Device to place parameters on.

        Returns:
            TrellisLinear module initialized with the weight data.
        """
        # TrellisWeight: K = out_features, N = in_features
        # (weight matrix is [out_features, in_features])
        K, N = weight.original_shape
        out_features, in_features = K, N

        # Don't call __init__ which creates wrong-sized buffers
        # Instead, directly create module and set buffers
        module = object.__new__(cls)
        nn.Module.__init__(module)

        module.in_features = in_features
        module.out_features = out_features
        module.bits = weight.bits

        # Move tensors to device
        def to_dev(t: torch.Tensor) -> torch.Tensor:
            return t.to(device) if device else t

        # Register buffers with actual weight data
        module.register_buffer("packed_indices", to_dev(weight.packed_indices.clone()))
        module.register_buffer("scales", to_dev(weight.scales.clone()))
        module.register_buffer("su", to_dev(weight.su.clone()))
        module.register_buffer("sv", to_dev(weight.sv.clone()))

        # Pre-compute codebook grid
        codebook = TrellisCodebook(bits=weight.bits)
        grid = torch.from_numpy(codebook.get_grid()).float()
        module.register_buffer("grid", to_dev(grid))

        if bias is not None:
            module.register_buffer("bias", to_dev(bias.clone()))
        else:
            module.bias = None

        # Initialize cache
        module._lib = None
        module._dequantized_cache = None

        return module

    def _get_lib(self) -> MetalKernelLibrary:
        """Get or create Metal kernel library."""
        if self._lib is None:
            self._lib = MetalKernelLibrary.from_source_dir()
        return self._lib

    def dequantize(self, use_cache: bool = True) -> torch.Tensor:
        """Dequantize weights to FP16.

        Args:
            use_cache: If True, cache the dequantized weights for reuse.
                      Set to False for memory-constrained scenarios.

        Returns:
            Dequantized weights [out_features, in_features] float16.
        """
        if use_cache and self._dequantized_cache is not None:
            return self._dequantized_cache

        # K = out_features, N = in_features (weight matrix convention)
        K, N = self.out_features, self.in_features

        if HAS_METAL and HAS_MPS and self.packed_indices.is_mps:
            lib = self._get_lib()

            weights = dispatch_trellis_dequant_packed(
                lib,
                self.packed_indices,  # uint8 packed
                self.scales,
                self.grid,
                self.su,
                self.sv,
                K,
                N,
                self.bits,
            )
        else:
            # CPU fallback - unpack then dequant
            from dataclasses import dataclass

            @dataclass
            class _FakeWeight:
                packed_indices: torch.Tensor
                scales: torch.Tensor
                su: torch.Tensor
                sv: torch.Tensor
                bits: int
                original_shape: tuple[int, int]

            fake_weight = _FakeWeight(
                packed_indices=self.packed_indices,
                scales=self.scales,
                su=self.su,
                sv=self.sv,
                bits=self.bits,
                original_shape=(K, N),
            )
            weights = dequantize_trellis_weight(fake_weight, use_metal=False)

        if use_cache:
            self._dequantized_cache = weights

        return weights

    def clear_cache(self) -> None:
        """Clear dequantized weight cache to free memory."""
        self._dequantized_cache = None

    # Class-level setting to control weight caching behavior
    # Set to False for MoE models to prevent excessive memory usage
    enable_cache: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization.

        Args:
            x: Input tensor [..., in_features].

        Returns:
            Output tensor [..., out_features].
        """
        # Dequantize weights (use cache setting from class variable)
        weights = self.dequantize(use_cache=TrellisLinear.enable_cache)

        # Ensure input is on same device and dtype
        if x.device != weights.device:
            x = x.to(weights.device)
        if x.dtype != weights.dtype:
            x = x.to(weights.dtype)

        # Linear: y = x @ W^T
        output = torch.mm(x.view(-1, self.in_features), weights.t())
        output = output.view(*x.shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias

        return output

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
            from .trellis_loader import TrellisModelLoader

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
