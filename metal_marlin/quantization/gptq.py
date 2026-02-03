"""GPTQ quantization support for Metal Marlin.

Handles loading and processing of GPTQ-quantized weights (qweight, qzeros, scales)
for Metal dequantization kernels.
"""

from __future__ import annotations

import numpy as np
import torch


def unpack_gptq_zeros(qzeros: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Unpack 4-bit packed zeros to float16.

    Args:
        qzeros: Packed int32 zeros [n_groups, out_features // 8]
        bits: Number of bits (default 4)

    Returns:
        Unpacked zeros [n_groups, out_features] as float16
    """
    qzeros = qzeros.cpu().numpy()
    n_groups, packed_cols = qzeros.shape

    # Each int32 holds 8 * 4-bit values
    # GPTQ packing order for zeros is typically:
    # 0x11223344 -> 4, 4, 3, 3, 2, 2, 1, 1 (nibbles)
    # But usually it's just standard 32-bit packing.
    # AutoGPTQ / OPTQ uses:
    # (qzeros[:, i] >> (bits * j)) & mask

    unpacked = np.zeros((n_groups, packed_cols * 8), dtype=np.int32)
    mask = (1 << bits) - 1

    for i in range(8):
        unpacked[:, i::8] = (qzeros >> (bits * i)) & mask

    # GPTQ zeros are stored as (zero + 1) to avoid -1?
    # Actually standard GPTQ is symmetric-ish: zero = qzero + 1
    # But let's stick to standard unpacking first.
    # Check AutoGPTQ implementation: zeros = qzeros + 1 in some versions.
    # Usually it's just raw values.

    return torch.from_numpy(unpacked.astype(np.float16))


def load_gptq_weights(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    bits: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare GPTQ weights for Metal dequantization.

    Args:
        qweight: Packed int32 weights [in_features // 8, out_features]
        qzeros: Packed int32 zeros [n_groups, out_features // 8]
        scales: Float16 scales [n_groups, out_features]
        g_idx: Group indices (optional, used for act-order)
        perm: Permutation indices (optional, used for act-order)
        bits: Quantization bits (default 4)

    Returns:
        qweight: Packed int32 weights (passed through)
        scales: Float16 scales
        zeros: Float16 unpacked zeros (dequantized)
    """
    # 1. Unpack zeros
    # GPTQ stores zeros packed. We need them unpacked for the kernel
    # (or kernel unpacks them, but unpacking once here is easier for now).
    if qzeros.dtype == torch.int32:
        zeros = unpack_gptq_zeros(qzeros, bits=bits)
    else:
        zeros = qzeros

    # 2. Reshape/permute if necessary
    # Metal kernel expects specific layout.
    # Current assumption:
    # qweight: [K/8, N] (int32) where each int32 has 8 weights along K (column-major in block)
    # scales: [n_groups, N]
    # zeros: [n_groups, N]

    # GPTQ layout is typically [in_features // 8, out_features] for qweight.
    # The packing is usually row-major within the int32?
    # "The layout of the qweight tensor is [in_features // 32 * bits, out_features]"
    # For 4-bit: [in_features // 8, out_features]
    # Element (i, j) of qweight contains weights for 8 input channels.

    # Return as is for now, assuming kernel handles GPTQ packing
    return qweight, scales, zeros.to(scales.device)
