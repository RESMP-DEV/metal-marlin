"""SparseTrellisLinear: 2:4 structured sparsity with trellis quantization.

Combines 2:4 structured sparsity (50% weight pruning) with 3-bit trellis
quantization for maximum memory efficiency and inference speed on MoE experts.

Memory savings stack multiplicatively:
- 3-bit quantization: 5.3x compression over FP16
- 2:4 sparsity: 2x additional compression (only store non-zeros)
- Total: ~10x memory reduction vs FP16 dense weights

Storage format:
    packed_indices_sparse: [tiles_k_sparse, tiles_n, packed_bytes] uint8
        - tiles_k_sparse = tiles_k // 2 (half the K dimension due to sparsity)
        - Each tile still 16x16 but only non-zero values stored

    sparse_metadata: [k_blocks, N] uint32
        - k_blocks = K // 4 (one 4-bit nibble per 4-element block)
        - 8 blocks (32 K positions) packed per uint32
        - Nibble format: [idx1:2][idx0:2] where idx0, idx1 in [0,3]

    scales_sparse: [n_groups, out_features] half
        - Same as dense (scales apply after dequant + scatter)

    su_sparse: [in_features] half  (full, used for sign flip before multiply)
    sv_sparse: [out_features] half (full, used for sign flip after multiply)

Kernel workflow:
1. For each K-tile (16 K positions = 4 sparsity groups):
   a. Load 4 metadata nibbles from sparse_metadata
   b. Decode 8 position indices (2 per group)
   c. Load 8 sparse trellis-packed weights from packed_indices_sparse
   d. Dequantize via grid lookup + scale
   e. Scatter to dense 16-element K tile positions
2. Continue with standard trellis GEMM compute

This module provides:
- SparseTrellisLinear: nn.Module for sparse+quantized linear layers
- prune_to_24_sparse: Convert dense TrellisLinear to sparse
- pack_sparse_weights: Compress pruned weights to sparse format
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .linear import TrellisLinear

# 2:4 sparsity constants
SPARSE_GROUP_SIZE = 4  # Dense K elements per sparsity group
SPARSE_NNZ_PER_GROUP = 2  # Non-zeros kept per group
SPARSE_COMPRESSION_RATIO = 2  # K_dense / K_sparse

# Metadata packing constants
META_BITS_PER_GROUP = 4  # 2 bits per index × 2 indices
META_GROUPS_PER_U32 = 8  # 32 bits / 4 bits
META_DENSE_K_PER_U32 = META_GROUPS_PER_U32 * SPARSE_GROUP_SIZE  # 32


@dataclass
class SparseWeightFormat:
    """Describes the sparse weight storage format.

    Attributes:
        tiles_k_sparse: Number of K-dimension tiles after sparsity compression.
        tiles_n: Number of N-dimension tiles (unchanged from dense).
        packed_bytes_per_tile: Bytes per tile (96 for 3-bit, etc).
        metadata_words_per_col: Number of uint32 metadata words per output column.
        total_sparse_bytes: Total bytes for sparse packed weights.
        total_metadata_bytes: Total bytes for sparsity metadata.
        compression_vs_dense: Compression ratio vs dense trellis format.
    """

    tiles_k_sparse: int
    tiles_n: int
    packed_bytes_per_tile: int
    metadata_words_per_col: int
    total_sparse_bytes: int
    total_metadata_bytes: int
    compression_vs_dense: float


def compute_sparse_format(
    in_features: int, out_features: int, bits: int
) -> SparseWeightFormat:
    """Compute storage requirements for sparse trellis format.

    Args:
        in_features: Input dimension (K, dense).
        out_features: Output dimension (N).
        bits: Quantization bits (2, 3, or 4).

    Returns:
        SparseWeightFormat describing the storage layout.
    """
    TILE_DIM = 16
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]

    # Dense tile counts
    tiles_k_dense = (in_features + TILE_DIM - 1) // TILE_DIM
    tiles_n = (out_features + TILE_DIM - 1) // TILE_DIM

    # Sparse K dimension: half the dense K (2:4 stores 2 of every 4)
    k_sparse = in_features // SPARSE_COMPRESSION_RATIO
    tiles_k_sparse = (k_sparse + TILE_DIM - 1) // TILE_DIM

    # Metadata: one uint32 covers 32 dense K positions
    k_blocks = in_features // SPARSE_GROUP_SIZE
    metadata_words_per_col = (k_blocks + META_GROUPS_PER_U32 - 1) // META_GROUPS_PER_U32

    total_sparse_bytes = tiles_k_sparse * tiles_n * packed_bytes
    total_metadata_bytes = metadata_words_per_col * out_features * 4  # uint32

    # Dense trellis size for comparison
    total_dense_bytes = tiles_k_dense * tiles_n * packed_bytes

    compression = total_dense_bytes / (total_sparse_bytes + total_metadata_bytes)

    return SparseWeightFormat(
        tiles_k_sparse=tiles_k_sparse,
        tiles_n=tiles_n,
        packed_bytes_per_tile=packed_bytes,
        metadata_words_per_col=metadata_words_per_col,
        total_sparse_bytes=total_sparse_bytes,
        total_metadata_bytes=total_metadata_bytes,
        compression_vs_dense=compression,
    )


def prune_to_24_sparse(
    weight: torch.Tensor, return_mask: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune a weight matrix to 2:4 structured sparsity.

    For every 4 consecutive elements along the K (input) dimension,
    keeps the 2 with largest magnitude and zeros the other 2.

    Args:
        weight: Dense weight tensor [out_features, in_features].
        return_mask: If True, also return the binary sparsity mask.

    Returns:
        pruned_weight: Weight with 2:4 sparsity pattern [out_features, in_features].
        positions: Indices of kept positions [out_features, in_features // 2].
        mask: (optional) Binary mask [out_features, in_features] where 1 = kept.
    """
    out_features, in_features = weight.shape
    assert in_features % SPARSE_GROUP_SIZE == 0, (
        f"in_features ({in_features}) must be divisible by {SPARSE_GROUP_SIZE}"
    )

    # Reshape to [out, num_groups, 4] for per-group processing
    num_groups = in_features // SPARSE_GROUP_SIZE
    w_grouped = weight.view(out_features, num_groups, SPARSE_GROUP_SIZE)

    # Find top-2 positions per group by magnitude
    abs_grouped = w_grouped.abs()
    _, top2_indices = abs_grouped.topk(SPARSE_NNZ_PER_GROUP, dim=2, largest=True)
    # top2_indices: [out, num_groups, 2]

    # Sort indices to maintain order (important for deterministic metadata)
    top2_indices = top2_indices.sort(dim=2).values

    # Create sparse mask
    mask_grouped = torch.zeros_like(w_grouped, dtype=torch.bool)
    mask_grouped.scatter_(2, top2_indices, True)

    # Apply mask
    pruned_grouped = w_grouped * mask_grouped

    # Reshape back
    pruned_weight = pruned_grouped.view(out_features, in_features)

    # Extract kept positions for metadata
    # positions: [out, num_groups * 2] flattened indices within each group
    positions = top2_indices.view(out_features, -1)

    if return_mask:
        mask = mask_grouped.view(out_features, in_features)
        return pruned_weight, positions, mask
    return pruned_weight, positions


def pack_sparse_metadata(
    positions: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    """Pack sparsity position indices into uint32 metadata tensor.

    Args:
        positions: Position indices [out_features, num_groups * 2] where each pair
            is (idx0, idx1) with values in [0, 3].
        out_features: Output dimension (N).
        in_features: Input dimension (K, dense).

    Returns:
        metadata: Packed uint32 tensor [metadata_words_per_col, out_features].
    """
    num_groups = in_features // SPARSE_GROUP_SIZE
    positions = positions.view(out_features, num_groups, 2)

    # Number of uint32 words needed per column
    words_per_col = (num_groups + META_GROUPS_PER_U32 - 1) // META_GROUPS_PER_U32

    # Allocate output
    metadata = torch.zeros(words_per_col, out_features, dtype=torch.int64)

    # Pack positions into nibbles
    # Each group: nibble = (idx1 << 2) | idx0
    for g in range(num_groups):
        word_idx = g // META_GROUPS_PER_U32
        nibble_idx = g % META_GROUPS_PER_U32

        idx0 = positions[:, g, 0].long()  # [out_features]
        idx1 = positions[:, g, 1].long()  # [out_features]
        nibble = (idx1 << 2) | idx0  # 4-bit value

        shift = nibble_idx * META_BITS_PER_GROUP
        metadata[word_idx, :] |= nibble << shift

    return metadata.to(torch.int32)


def compress_sparse_weights(
    weight: torch.Tensor, positions: torch.Tensor, bits: int
) -> torch.Tensor:
    """Compress pruned weights to sparse trellis format.

    Extracts only the non-zero values and packs them in trellis tile format.
    The sparse tile has tiles_k_sparse = tiles_k // 2 since only half the
    K-dimension values are stored.

    Args:
        weight: Pruned weight tensor [out_features, in_features] with 2:4 sparsity.
        positions: Position indices [out_features, num_groups, 2].
        bits: Quantization bits.

    Returns:
        packed_sparse: Compressed weights [tiles_k_sparse, tiles_n, packed_bytes].
    """
    out_features, in_features = weight.shape
    num_groups = in_features // SPARSE_GROUP_SIZE
    positions = positions.view(out_features, num_groups, 2)

    # Extract non-zero values in order
    # sparse_values: [out_features, in_features // 2]
    sparse_values_list = []
    for n in range(out_features):
        col_values = []
        for g in range(num_groups):
            idx0, idx1 = positions[n, g, 0].item(), positions[n, g, 1].item()
            base_k = g * SPARSE_GROUP_SIZE
            col_values.append(weight[n, base_k + idx0])
            col_values.append(weight[n, base_k + idx1])
        sparse_values_list.append(torch.stack(col_values))

    sparse_values = torch.stack(sparse_values_list)  # [out, in // 2]

    # Now pack into trellis tile format
    # This is a placeholder - actual packing requires the trellis quantization
    # codebook lookup which is done by the TrellisWeight class
    return sparse_values


class SparseTrellisLinear(nn.Module):
    """Linear layer with 2:4 structured sparsity and trellis quantization.

    Combines 50% structured pruning with 3-bit quantization for ~10x
    memory reduction over FP16 dense weights.

    Forward pass:
    1. Load sparse metadata and compressed trellis weights
    2. Decode metadata to get position indices
    3. Dequantize sparse values via grid lookup
    4. Scatter to dense positions
    5. Compute GEMM with scattered weights

    Attributes:
        in_features: Input dimension (K, dense).
        out_features: Output dimension (N).
        bits: Quantization bit width.
        sparsity_pattern: "2:4" for structured sparsity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 3,
        bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize SparseTrellisLinear.

        Args:
            in_features: Input dimension (must be divisible by 4).
            out_features: Output dimension.
            bits: Quantization bits (default 3).
            bias: Whether to include bias.
            device: Target device.
        """
        super().__init__()

        assert in_features % SPARSE_GROUP_SIZE == 0, (
            f"in_features must be divisible by {SPARSE_GROUP_SIZE}"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.sparsity_pattern = "2:4"

        # Compute storage format
        fmt = compute_sparse_format(in_features, out_features, bits)
        self._format = fmt

        # Sparse packed indices (half the K-dimension tiles)
        TILE_DIM = 16
        self.register_buffer(
            "packed_indices_sparse",
            torch.zeros(
                fmt.tiles_k_sparse, fmt.tiles_n, fmt.packed_bytes_per_tile,
                dtype=torch.uint8, device=device
            ),
        )

        # Sparsity metadata: [words_per_col, out_features]
        self.register_buffer(
            "sparse_metadata",
            torch.zeros(
                fmt.metadata_words_per_col, out_features,
                dtype=torch.int32, device=device
            ),
        )

        # Scales for dequantization (same as dense)
        n_groups = (in_features + 127) // 128
        self.register_buffer(
            "scales",
            torch.ones(n_groups, out_features, dtype=torch.float16, device=device),
        )

        # Sign flip vectors (full dimension, applied at boundaries)
        self.register_buffer(
            "su", torch.ones(in_features, dtype=torch.float16, device=device)
        )
        self.register_buffer(
            "sv", torch.ones(out_features, dtype=torch.float16, device=device)
        )

        # Codebook grid
        from ..quantization.trellis_codebook import TrellisCodebook
        codebook = TrellisCodebook(bits=bits)
        grid = torch.from_numpy(codebook.get_grid()).half()
        self.register_buffer("grid", grid.to(device) if device else grid)

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16, device=device),
            )
        else:
            self.bias = None

        # Metal library reference (shared)
        self._lib: Any = None

    def set_lib(self, lib: Any) -> None:
        """Set shared Metal library."""
        self._lib = lib

    @classmethod
    def from_trellis_linear(
        cls,
        dense: TrellisLinear,
        device: torch.device | str | None = None,
    ) -> SparseTrellisLinear:
        """Create sparse layer from a dense TrellisLinear.

        Prunes weights to 2:4 sparsity pattern (keeps top-2 by magnitude
        per 4-element block) and repacks into sparse format.

        Args:
            dense: Dense TrellisLinear layer to prune.
            device: Target device (default: same as dense).

        Returns:
            SparseTrellisLinear with pruned weights.
        """
        if device is None:
            device = dense.packed_indices.device

        # First, dequantize to get full FP16 weights for pruning decision
        # This is a one-time cost during model preparation
        from .dispatch import dispatch_trellis_dequant_packed

        # Dequantize to dense
        dense_weight = dispatch_trellis_dequant_packed(
            packed_indices=dense.packed_indices,
            scales=dense.scales.half(),
            su=dense.su.half(),
            sv=dense.sv.half(),
            grid=dense.grid.half(),
            out_features=dense.out_features,
            in_features=dense.in_features,
            bits=dense.bits,
        )

        # Prune to 2:4 sparsity
        pruned_weight, positions = prune_to_24_sparse(dense_weight)

        # Create sparse layer
        sparse = cls(
            in_features=dense.in_features,
            out_features=dense.out_features,
            bits=dense.bits,
            bias=dense.bias is not None,
            device=device,
        )

        # Pack metadata
        metadata = pack_sparse_metadata(positions, dense.out_features, dense.in_features)
        sparse.sparse_metadata.copy_(metadata.to(device))

        # Re-quantize and pack sparse weights
        # For now, store pruned values directly - full requantization would
        # need to go through the TrellisWeight compression pipeline
        sparse_values = compress_sparse_weights(
            pruned_weight, positions.view(dense.out_features, -1, 2), dense.bits
        )
        # TODO: Properly pack sparse_values into trellis tile format
        # This requires calling the EXL3 quantization on the sparse values

        # Copy scales and sign vectors (these remain full-dimension)
        sparse.scales.copy_(dense.scales.half().to(device))
        sparse.su.copy_(dense.su.half().to(device))
        sparse.sv.copy_(dense.sv.half().to(device))
        sparse.grid.copy_(dense.grid.half().to(device))

        if dense.bias is not None:
            sparse.bias.copy_(dense.bias.to(device))

        return sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse trellis GEMM.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features].

        Returns:
            Output tensor [batch, out_features] or [batch, seq, out_features].
        """
        # Reshape if needed
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq, _ = x.shape
            x = x.view(-1, self.in_features)
        else:
            batch = x.shape[0]
            seq = None

        # Dispatch to sparse Metal kernel
        # TODO: Implement dispatch_gemm_trellis_sparse
        # For now, fall back to dense computation

        # This is a placeholder - actual sparse dispatch would use
        # packed_indices_sparse and sparse_metadata
        raise NotImplementedError(
            "Sparse GEMM kernel not yet implemented. "
            "Use from_trellis_linear() to create, then kernel integration needed."
        )

    def extra_repr(self) -> str:
        """Extra representation for print()."""
        fmt = self._format
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, sparsity={self.sparsity_pattern}, "
            f"compression={fmt.compression_vs_dense:.1f}x"
        )


def convert_moe_to_sparse(
    moe_layer: nn.Module,
    calibration_data: torch.Tensor | None = None,
) -> nn.Module:
    """Convert all experts in an MoE layer to 2:4 sparse format.

    Args:
        moe_layer: MoE layer with experts attribute.
        calibration_data: Optional activations for importance-based pruning.

    Returns:
        MoE layer with sparse experts.
    """
    from .linear import TrellisLinear

    if not hasattr(moe_layer, "experts"):
        raise ValueError("MoE layer must have 'experts' attribute")

    # Convert each expert's linear layers
    for expert_idx, expert in enumerate(moe_layer.experts):
        for name, module in expert.named_modules():
            if isinstance(module, TrellisLinear):
                sparse_module = SparseTrellisLinear.from_trellis_linear(module)
                # Replace in parent
                parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
                child_name = name.split(".")[-1]
                if parent_name:
                    parent = expert
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, sparse_module)
                else:
                    setattr(expert, child_name, sparse_module)

    return moe_layer


__all__ = [
    "SparseTrellisLinear",
    "SparseWeightFormat",
    "compute_sparse_format",
    "prune_to_24_sparse",
    "pack_sparse_metadata",
    "compress_sparse_weights",
    "convert_moe_to_sparse",
    "SPARSE_GROUP_SIZE",
    "SPARSE_NNZ_PER_GROUP",
    "SPARSE_COMPRESSION_RATIO",
]
