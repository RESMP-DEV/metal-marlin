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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from ..metal_dispatch import HAS_METAL, HAS_MPS, MetalKernelLibrary
from ..quantization.trellis_codebook import TrellisCodebook
from .dispatch import (
    dispatch_gemm_trellis_auto,
    dispatch_gemm_trellis_decode,
    dispatch_trellis_dequant_packed,
)

if TYPE_CHECKING:
    from .loader import TrellisWeight


@dataclass
class MixedBPWLayout:
    """Memory layout information for mixed bit-width weight storage.
    
    This class tracks how weight groups with different bit-widths are packed
    contiguously and provides lookup tables for efficient kernel dispatch.
    
    Attributes:
        group_offsets: Tensor of shape [num_groups + 1] containing byte offsets
            into packed_weights for each group. The last element is the total size.
        group_bits: Tensor of shape [num_groups] containing bit-width for each group.
        group_indices: Tensor of shape [num_groups] containing the original group index
            for each contiguous block (for reconstructing the weight matrix).
        packed_size: Total size of packed weights in bytes.
        unique_bits: Sorted list of unique bit-widths present.
        bit_to_groups: Dictionary mapping bit-width to list of group indices.
    """
    group_offsets: torch.Tensor  # [num_groups + 1] int64
    group_bits: torch.Tensor     # [num_groups] int8
    group_indices: torch.Tensor  # [num_groups] int32 - original positions
    packed_size: int
    unique_bits: list[int]
    bit_to_groups: dict[int, list[int]]
    
    def get_groups_for_bitwidth(self, bits: int) -> list[int]:
        """Get the group indices that use a specific bit-width."""
        return self.bit_to_groups.get(bits, [])
    
    @property
    def num_groups(self) -> int:
        """Total number of weight groups."""
        return len(self.group_bits)
    
    def get_bitwidth_distribution(self) -> dict[int, int]:
        """Get count of groups for each bit-width.
        
        Returns:
            Dictionary mapping bit-width to group count.
        """
        return {bits: len(groups) for bits, groups in self.bit_to_groups.items()}
    
    def get_bitwidth_coverage(self) -> dict[int, float]:
        """Get fraction of groups for each bit-width (0.0 to 1.0).
        
        Returns:
            Dictionary mapping bit-width to coverage fraction.
        """
        total = self.num_groups
        if total == 0:
            return {}
        return {bits: len(groups) / total for bits, groups in self.bit_to_groups.items()}


class TrellisLinear(nn.Module):
    """Linear layer with EXL3 trellis-quantized weights.
    Keeps weights compressed on GPU in packed format. Dequantization happens
    on-the-fly during forward pass using fused Metal GPU kernels.
    Attributes:
        in_features: Size of each input sample (K).
        out_features: Size of each output sample (N).
        bits: Quantization bit width (2, 3, 4, or 8).
        bits_per_group: Tensor mapping weight groups to bit-widths for mixed BPW.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize TrellisLinear."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Mixed BPW support
        self._is_mixed_bpw = False
        self._mixed_bpw_layout: MixedBPWLayout | None = None
        self._mixed_bpw_unique_bits: list[int] = []
        self._mixed_bpw_bit_to_groups: dict[int, list[int]] = {}
        
        # Tile dimensions
        TILE_DIM = 16
        tiles_k = (in_features + TILE_DIM - 1) // TILE_DIM
        tiles_n = (out_features + TILE_DIM - 1) // TILE_DIM

        # Register buffers
        packed_bytes = {2: 64, 3: 96, 4: 128, 8: 256}.get(bits, 128)
        self.register_buffer(
            "packed_indices",
            torch.zeros(tiles_k, tiles_n, packed_bytes, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales",
            torch.ones((in_features + 127) // 128, out_features, dtype=torch.float32, device=device),
        )
        self.register_buffer("su", torch.ones(in_features, dtype=torch.float32, device=device))
        self.register_buffer("sv", torch.ones(out_features, dtype=torch.float32, device=device))

        if bits > 0:
            codebook = TrellisCodebook(bits=bits)
            grid = torch.from_numpy(codebook.get_grid()).float()
            self.register_buffer("grid", grid)

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.bias = None

        # Mixed BPW: bits_per_group buffer
        self.register_buffer("bits_per_group", torch.full((tiles_k,), fill_value=bits, dtype=torch.int8, device=device))
        
        self._lib: MetalKernelLibrary | None = None
        self._grid_texture: Any | None = None
        self._mixed_grids: dict[int, torch.Tensor] = {}

    @staticmethod
    def _pack_same_bitwidth_groups_contiguously(
        packed_indices_by_bits: dict[int, torch.Tensor],
        scales_by_bits: dict[int, torch.Tensor],
        bits_per_group: torch.Tensor,
    ) -> tuple[MixedBPWLayout, torch.Tensor, torch.Tensor]:
        """Build optimized memory layout for mixed bit-width weight storage."""
        unique_bits = sorted(packed_indices_by_bits.keys())
        
        sorted_group_indices = torch.argsort(bits_per_group, stable=True)
        group_bits = bits_per_group[sorted_group_indices]
        
        # Flatten packed indices for each bit-width first
        flattened_by_bits = {}
        for bits in unique_bits:
            # Flatten the 3D tensor [num_groups, tiles_n, packed_bytes] to 1D
            flat = packed_indices_by_bits[bits].reshape(-1)
            num_groups_for_bit = scales_by_bits[bits].shape[0]
            bytes_per_group = flat.numel() // num_groups_for_bit
            flattened_by_bits[bits] = (flat, bytes_per_group)
        
        packed_tensors = []
        scale_tensors = []
        
        group_counters = {bits: 0 for bits in unique_bits}

        for original_idx in sorted_group_indices:
            bits = bits_per_group[original_idx.item()].item()
            group_num_in_batch = group_counters[bits]
            scale_tensors.append(scales_by_bits[bits][group_num_in_batch, :].unsqueeze(0))
            
            flat_packed, bytes_per_group = flattened_by_bits[bits]
            start = group_num_in_batch * bytes_per_group
            end = start + bytes_per_group
            packed_tensors.append(flat_packed[start:end])
            
            group_counters[bits] += 1

        packed_indices = torch.cat(packed_tensors)
        scales = torch.cat(scale_tensors, dim=0)
        
        group_offsets = [0]
        bit_to_groups = {bits: [] for bits in unique_bits}
        
        for i, bits_val in enumerate(group_bits):
            bits = bits_val.item()
            flat_packed, bytes_per_group = flattened_by_bits[bits]
            group_offsets.append(group_offsets[-1] + bytes_per_group)
            bit_to_groups[bits].append(i)

        layout = MixedBPWLayout(
            group_offsets=torch.tensor(group_offsets, dtype=torch.int64),
            group_bits=group_bits,
            group_indices=sorted_group_indices.to(torch.int32),
            packed_size=packed_indices.numel(),
            unique_bits=unique_bits,
            bit_to_groups=bit_to_groups,
        )
        
        return layout, packed_indices, scales

    @classmethod
    def from_mixed_bpw_tensors(
        cls,
        in_features: int,
        out_features: int,
        bits_per_group: torch.Tensor,
        packed_indices_dict: dict[int, torch.Tensor],
        scales_dict: dict[int, torch.Tensor],
        su: torch.Tensor,
        sv: torch.Tensor,
        bias: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> TrellisLinear:
        """Create TrellisLinear from mixed bit-width tensors."""
        if device is None:
            device = "mps" if HAS_MPS else "cpu"

        unique_bits = sorted(torch.unique(bits_per_group).tolist())
        max_bits = max(unique_bits) if unique_bits else 0
        
        module = cls(in_features, out_features, bits=max_bits, bias=bias is not None, device=device)
        module._is_mixed_bpw = True
        
        layout, packed_indices, scales = cls._pack_same_bitwidth_groups_contiguously(
            packed_indices_dict, scales_dict, bits_per_group
        )
        
        module.packed_indices = nn.Parameter(packed_indices.to(device), requires_grad=False)
        module.scales = nn.Parameter(scales.to(device), requires_grad=False)
        module.su = nn.Parameter(su.to(device), requires_grad=False)
        module.sv = nn.Parameter(sv.to(device), requires_grad=False)
        if bias is not None:
            module.bias = nn.Parameter(bias.to(device), requires_grad=False)

        module.bits_per_group = nn.Parameter(bits_per_group.to(device=device, dtype=torch.int8), requires_grad=False)
        module._mixed_bpw_layout = layout
        module._mixed_bpw_unique_bits = unique_bits
        module._mixed_bpw_bit_to_groups = layout.bit_to_groups

        for bits in unique_bits:
            if bits == 8:
                continue
            codebook = TrellisCodebook(bits=bits)
            grid = torch.from_numpy(codebook.get_grid()).float().to(device)
            module.register_buffer(f"grid_{bits}", grid)
            module._mixed_grids[bits] = grid
        
        return module

    def forward_mixed_bpw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed bit-width weight storage.
        
        Optimized to batch process groups by bit-width, minimizing kernel launch overhead.
        Uses single kernel launch per bit-width instead of per-group launches.
        """
        if not self._is_mixed_bpw or self._mixed_bpw_layout is None:
            raise RuntimeError("Layer is not configured for mixed bit-width forward pass.")

        batch_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        M = x.shape[0]
        
        output = torch.zeros(M, self.out_features, dtype=x.dtype, device=x.device)
        
        layout = self._mixed_bpw_layout
        lib = self._get_lib()

        n_groups = self.scales.shape[0]
        group_size = (self.in_features + n_groups - 1) // n_groups if n_groups > 0 else self.in_features

        # Process each bit-width, batching groups to minimize kernel launches
        for bits in layout.unique_bits:
            groups_for_bit = layout.bit_to_groups.get(bits, [])
            if not groups_for_bit:
                continue
            
            if bits == 8:
                # For 8-bit, dequantize and process only the 8-bit groups
                # Collect feature ranges for 8-bit groups
                feature_ranges = []
                for contig_group_idx in groups_for_bit:
                    original_group_idx = layout.group_indices[contig_group_idx].item()
                    start_feature = original_group_idx * group_size
                    end_feature = min(start_feature + group_size, self.in_features)
                    if start_feature < end_feature:
                        feature_ranges.append((start_feature, end_feature))
                
                if not feature_ranges:
                    continue
                
                # Concatenate input slices for all 8-bit groups
                x_slices = [x[:, start:end] for start, end in feature_ranges]
                x_concat = torch.cat(x_slices, dim=1)
                
                # Get corresponding weight slices
                if not hasattr(self, '_full_weight_dequant'):
                    self._full_weight_dequant = self.dequant()
                
                weight_slices = [self._full_weight_dequant[:, start:end] for start, end in feature_ranges]
                weight_concat = torch.cat(weight_slices, dim=1)
                
                # Matmul for 8-bit groups
                output += x_concat @ weight_concat.t()
            else:
                # For 2/3/4-bit, batch process all groups of this bit-width
                grid = self._mixed_grids.get(bits)
                if grid is None:
                    continue
                
                # Gather all groups of this bit-width
                packed_list = []
                scales_list = []
                feature_ranges = []
                
                for contig_group_idx in groups_for_bit:
                    original_group_idx = layout.group_indices[contig_group_idx].item()
                    
                    start_feature = original_group_idx * group_size
                    end_feature = min(start_feature + group_size, self.in_features)
                    if start_feature >= end_feature:
                        continue
                    
                    feature_ranges.append((start_feature, end_feature))
                    
                    start_offset = layout.group_offsets[contig_group_idx].item()
                    end_offset = layout.group_offsets[contig_group_idx + 1].item()
                    packed_slice = self.packed_indices.view(-1)[start_offset:end_offset]
                    packed_list.append(packed_slice)
                    scales_list.append(self.scales[contig_group_idx:contig_group_idx + 1])
                
                if not packed_list:
                    continue
                
                # Concatenate packed indices and scales for batched processing
                packed_concat = torch.cat(packed_list)
                scales_concat = torch.cat(scales_list, dim=0)
                
                # Compute total input features for this bit-width
                total_features = sum(end - start for start, end in feature_ranges)
                
                if total_features == 0:
                    continue
                
                # Reshape packed indices for kernel
                # Determine packed bytes per tile
                packed_bytes = {2: 64, 3: 96, 4: 128, 8: 256}.get(bits, 128)
                num_groups_bit = len(packed_list)
                packed_reshaped = packed_concat.view(num_groups_bit, 1, packed_bytes)
                
                # Build input tensor by concatenating slices
                x_slices = [x[:, start:end] for start, end in feature_ranges]
                x_concat = torch.cat(x_slices, dim=1)
                
                # Build su tensor for all features
                su_slices = [self.su[start:end] for start, end in feature_ranges]
                su_concat = torch.cat(su_slices, dim=0)
                
                # Dispatch batched kernel (single launch per bit-width)
                partial_out = dispatch_gemm_trellis_auto(
                    lib,
                    x_concat,  # [M, total_features]
                    packed_reshaped,  # [num_groups, 1, packed_bytes]
                    scales_concat,  # [num_groups, N]
                    grid,  # codebook grid
                    su_concat,  # [total_features]
                    self.sv,  # [N]
                    total_features,  # K for this bit-width
                    self.out_features,  # N
                    bits,
                    group_size,
                )
                
                output += partial_out

        if self.bias is not None:
            output += self.bias

        return output.view(*batch_shape, self.out_features)

    def set_lib(self, lib: MetalKernelLibrary) -> None:
        self._lib = lib

    @classmethod
    def from_trellis_weight(
        cls,
        weight: TrellisWeight,
        bias: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> TrellisLinear:
        """Create TrellisLinear from a loaded TrellisWeight."""
        if device is None:
            device = "mps" if HAS_MPS else "cpu"
        
        module = cls(
            in_features=weight.original_shape[1],
            out_features=weight.original_shape[0],
            bits=weight.bits,
            bias=bias is not None,
            device=device,
        )

        packed_indices = weight.packed_indices
        if packed_indices.shape != module.packed_indices.shape:
            expected = module.packed_indices.shape
            is_swapped_tile_axes = (
                packed_indices.ndim == 3
                and packed_indices.shape[0] == expected[1]
                and packed_indices.shape[1] == expected[0]
                and packed_indices.shape[2] == expected[2]
            )
            if is_swapped_tile_axes:
                packed_indices = packed_indices.permute(1, 0, 2).contiguous()
            else:
                raise ValueError(
                    "Packed indices shape mismatch: "
                    f"got {tuple(weight.packed_indices.shape)}, "
                    f"expected {tuple(expected)}"
                )

        module.packed_indices.copy_(packed_indices)
        module.scales.copy_(weight.scales)
        module.su.copy_(weight.su)
        module.sv.copy_(weight.sv)
        if bias is not None:
            module.bias.copy_(bias)
            
        return module

    def _get_lib(self) -> MetalKernelLibrary:
        if self._lib is None:
            self._lib = MetalKernelLibrary.from_source_dir()
        return self._lib

    def dequantize(self) -> torch.Tensor:
        """Dequantize weights to FP16."""
        K, N = self.out_features, self.in_features
        n_groups = self.scales.shape[0]
        group_size = (self.in_features + n_groups - 1) // n_groups

        if HAS_METAL and HAS_MPS and self.packed_indices.is_mps:
            try:
                return dispatch_trellis_dequant_packed(
                    self._get_lib(), self.packed_indices, self.scales, self.grid,
                    self.su, self.sv, K, N, self.bits, group_size
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Metal dequantize failed, falling back to CPU: {e}", RuntimeWarning)

        return self._dequantize_cpu_packed(K, N, group_size)

    def _dequantize_cpu_packed(self, K: int, N: int, group_size: int) -> torch.Tensor:
        """CPU fallback for dequantizing packed trellis weights."""
        packed_indices = self.packed_indices.cpu().numpy()
        scales = self.scales.cpu().float().numpy()
        grid = self.grid.cpu().float().numpy()
        su = self.su.cpu().float().numpy()
        sv = self.sv.cpu().float().numpy()

        TILE_DIM = 16
        tiles_k, tiles_n = (K + TILE_DIM - 1) // TILE_DIM, (N + TILE_DIM - 1) // TILE_DIM
        output = np.zeros((K, N), dtype=np.float32)

        for tile_k in range(tiles_k):
            for tile_n in range(tiles_n):
                indices = self._unpack_tile_indices(packed_indices[tile_k, tile_n], self.bits, len(grid))
                for local_k in range(TILE_DIM):
                    for local_n in range(TILE_DIM):
                        k, n = tile_k * TILE_DIM + local_k, tile_n * TILE_DIM + local_n
                        if k >= K or n >= N:
                            continue

                        idx = int(indices[local_k * TILE_DIM + local_n])
                        group_idx = min(n // group_size, scales.shape[0] - 1)
                        scale = scales[group_idx, k]

                        output[k, n] = grid[idx] * scale * su[n] * sv[k]

        return torch.from_numpy(output).half()

    def _unpack_tile_indices(self, packed_tile: np.ndarray, bits: int, n_levels: int) -> np.ndarray:
        """Unpack packed bytes into codebook indices."""
        n_elements = 256
        indices = np.zeros(n_elements, dtype=np.uint8)
        if bits == 4:
            for i in range(n_elements // 2):
                byte = packed_tile[i]
                indices[i * 2] = byte & 0x0F
                indices[i * 2 + 1] = (byte >> 4) & 0x0F
        elif bits == 2:
            for i in range(n_elements // 4):
                byte = packed_tile[i]
                indices[i * 4] = byte & 0x03
                indices[i * 4 + 1] = (byte >> 2) & 0x03
                indices[i * 4 + 2] = (byte >> 4) & 0x03
                indices[i * 4 + 3] = (byte >> 6) & 0x03
        elif bits == 3:
            byte_idx = 0
            for i in range(0, n_elements, 8):
                if byte_idx + 2 >= len(packed_tile):
                    break
                b0, b1, b2 = packed_tile[byte_idx:byte_idx+3]
                indices[i] = b0 & 0x07
                indices[i+1] = (b0 >> 3) & 0x07
                indices[i+2] = ((b0 >> 6) | (b1 << 2)) & 0x07
                indices[i+3] = (b1 >> 1) & 0x07
                indices[i+4] = (b1 >> 4) & 0x07
                indices[i+5] = ((b1 >> 7) | (b2 << 1)) & 0x07
                indices[i+6] = (b2 >> 2) & 0x07
                indices[i+7] = (b2 >> 5) & 0x07
                byte_idx += 3
        elif bits == 8:
            indices[:] = packed_tile[:n_elements]
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly decompression."""
        if self._is_mixed_bpw:
            return self.forward_mixed_bpw(x)

        batch_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        M = x.shape[0]

        n_groups = self.scales.shape[0]
        group_size = (self.in_features + n_groups - 1) // n_groups if n_groups > 0 else self.in_features

        if HAS_METAL and HAS_MPS and x.is_mps:
            lib = self._get_lib()
            try:
                kernel_dispatch = dispatch_gemm_trellis_decode if M <= 16 else dispatch_gemm_trellis_auto
                output = kernel_dispatch(
                    lib, x, self.packed_indices, self.scales, self.grid,
                    self.su, self.sv, self.in_features, self.out_features, self.bits, group_size
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Metal TrellisLinear dispatch failed: {e}", RuntimeWarning)
                output = self._forward_cpu_fallback(x, group_size)
        else:
            output = self._forward_cpu_fallback(x, group_size)

        output = output.view(*batch_shape, self.out_features)
        if self.bias is not None:
            output.add_(self.bias)
        return output

    def _forward_cpu_fallback(self, x_flat: torch.Tensor, group_size: int) -> torch.Tensor:
        """CPU fallback for forward pass."""
        weights = self.dequantize().to(x_flat.device)
        return x_flat @ weights.t()

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bits={self.bits if not self._is_mixed_bpw else 'mixed'}, bias={self.bias is not None}")


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
