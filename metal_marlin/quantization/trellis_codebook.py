"""Trellis codebook for EXL3 quantization.

Implements the EXL3 codebook structure with magic multipliers
for efficient CUDA/Metal kernel lookup.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrellisCodebook:
    """Codebook for trellis quantization.

    EXL3 uses specialized codebooks with magic multipliers
    for efficient CUDA/Metal kernel lookup.

    Attributes:
        bits: Bits per weight (2-8)
        scale: Codebook scale factor from ExllamaV3
        mcg_mult: Multi-component grid magic multiplier
        mul1_mult: Multiplicative codebook magic multiplier
    """
    bits: int  # 2-8 bits per weight
    scale: float = 1.24371088  # codebook_scale from ExllamaV3
    mcg_mult: int = 0xCBAC1FED  # Multi-component grid
    mul1_mult: int = 0x83DCD12D  # Multiplicative codebook

    def __post_init__(self) -> None:
        """Validate bit width."""
        if not 2 <= self.bits <= 8:
            raise ValueError(f"bits must be between 2 and 8, got {self.bits}")

    def get_grid(self) -> NDArray[np.float32]:
        """Return quantization grid for this bit width.

        Returns:
            Array of quantization levels for the codebook.
            Grid is uniform symmetric centered at 0, scaled by codebook_scale.
        """
        n_levels = 2 ** self.bits
        # Uniform symmetric grid centered at 0
        grid = np.linspace(
            -(n_levels - 1) / 2,
            (n_levels - 1) / 2,
            n_levels,
            dtype=np.float32
        )
        # Scale by codebook scale factor
        return grid * np.float32(self.scale)

    def quantize_value(self, val: float, scale: float) -> tuple[int, float]:
        """Quantize single value, return (index, dequantized).

        Args:
            val: Value to quantize
            scale: Scale factor for this tile (applied to normalized value)

        Returns:
            Tuple of (quantized index, dequantized value)
        """
        grid = self.get_grid()
        normalized = val / scale

        # Find nearest grid point
        idx = int(np.argmin(np.abs(grid - normalized)))
        dequantized = float(grid[idx] * scale)

        return idx, dequantized

    def get_n_levels(self) -> int:
        """Return number of quantization levels.

        Returns:
            Number of levels (2^bits)
        """
        return 2 ** self.bits
