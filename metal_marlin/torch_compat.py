"""PyTorch compatibility for Marlin weights.

Allows loading Marlin-quantized weights in PyTorch for cross-platform inference,
and importing GPTQ-format weights for use with the Metal Marlin kernels.

Format notes:
- Marlin packs 8 x 4-bit nibbles along N into uint32: packed[K, N // 8]
- GPTQ/AutoGPTQ packs 8 x 4-bit values along K into int32: qweight[K // 8, N]
"""

from __future__ import annotations

from typing import Any

import numpy as np


def export_for_pytorch(
    packed_weights: np.ndarray,
    scales: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Export Marlin weights in PyTorch-compatible GPTQ format.

    Converts from Marlin packing (N-dimension, uint32[K, N//8]) to
    GPTQ convention (K-dimension, int32[K//8, N]).

    Output matches GPTQ-for-LLaMa / AutoGPTQ convention:
    - qweight: [K // 8, N] int32 (8 x 4-bit values packed along K)
    - scales: [num_groups, N] fp16
    - g_idx: [K] int32 (group index for each input element)
    - bits: scalar int32 (4)
    - group_size: scalar int32

    Args:
        packed_weights: Marlin-packed uint32 array [K, N // 8].
        scales: float16/float32 scale array [num_groups, N].
        metadata: Dict with 'in_features' (K), 'out_features' (N), 'group_size'.

    Returns:
        Dict of numpy arrays in GPTQ state_dict format.
    """
    K = metadata["in_features"]
    N = metadata["out_features"]
    group_size = metadata["group_size"]

    # Unpack Marlin format: packed[K, N//8] -> nibbles[K, N]
    unpacked = _unpack_marlin(packed_weights, N)

    # Repack in GPTQ format: nibbles[K, N] -> qweight[K//8, N]
    qweight = _pack_gptq(unpacked)

    # Generate group index: maps each K-dimension element to its group
    g_idx = np.arange(K, dtype=np.int32) // group_size

    return {
        "qweight": qweight,
        "scales": scales.astype(np.float16),
        "g_idx": g_idx,
        "bits": np.array([4], dtype=np.int32),
        "group_size": np.array([group_size], dtype=np.int32),
    }


def import_from_pytorch(
    state_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Import weights from PyTorch GPTQ format to Marlin packing.

    Converts GPTQ K-dimension packing to Marlin N-dimension packing.

    Args:
        state_dict: Dict with 'qweight' [K//8, N], 'scales' [G, N],
                    and optionally 'group_size' scalar.

    Returns:
        Tuple of (packed_marlin [K, N//8], scales [G, N], group_size).
    """
    qweight = state_dict["qweight"]
    scales = state_dict["scales"]
    group_size = int(state_dict.get("group_size", np.array([128]))[0])

    # Unpack GPTQ: qweight[K//8, N] -> nibbles[K, N]
    unpacked = _unpack_gptq(qweight)

    # Repack for Marlin: nibbles[K, N] -> packed[K, N//8]
    packed_marlin = _pack_marlin(unpacked)

    return packed_marlin, scales.astype(np.float16), group_size


def _unpack_marlin(packed: np.ndarray, N: int) -> np.ndarray:
    """Unpack Marlin format: 8 nibbles along N per uint32.

    packed[K, N//8] uint32 -> unpacked[K, N] uint8
    """
    K = packed.shape[0]
    packed_u32 = packed.astype(np.uint32)
    unpacked = np.zeros((K, N), dtype=np.uint8)

    for i in range(8):
        unpacked[:, i::8] = ((packed_u32 >> (i * 4)) & 0xF).astype(np.uint8)

    return unpacked


def _pack_marlin(nibbles: np.ndarray) -> np.ndarray:
    """Pack nibbles along N into Marlin uint32 format.

    nibbles[K, N] uint8 -> packed[K, N//8] uint32
    """
    K, N = nibbles.shape
    assert N % 8 == 0, f"N={N} must be divisible by 8"

    packed = np.zeros((K, N // 8), dtype=np.uint32)
    for i in range(8):
        packed |= (nibbles[:, i::8].astype(np.uint32) & 0xF) << (i * 4)

    return packed


def _unpack_gptq(qweight: np.ndarray) -> np.ndarray:
    """Unpack GPTQ format: 8 nibbles along K per int32.

    qweight[K//8, N] int32 -> unpacked[K, N] uint8
    """
    K8, N = qweight.shape
    K = K8 * 8
    q_u32 = qweight.view(np.uint32)
    unpacked = np.zeros((K, N), dtype=np.uint8)

    for i in range(8):
        unpacked[i::8, :] = ((q_u32 >> (i * 4)) & 0xF).astype(np.uint8)

    return unpacked


def _pack_gptq(nibbles: np.ndarray) -> np.ndarray:
    """Pack nibbles along K into GPTQ int32 format.

    nibbles[K, N] uint8 -> qweight[K//8, N] int32
    """
    K, N = nibbles.shape
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(8):
        packed |= (nibbles[i::8, :].astype(np.uint32) & 0xF) << (i * 4)

    # GPTQ convention uses int32 (same bit pattern)
    return packed.view(np.int32)
