"""Bit-packing utilities for Trellis quantized indices.

Trellis quantization produces indices in range [0, 2^bits - 1]. These need to be
packed efficiently for storage:
- 2-bit: 4 indices per byte
- 3-bit: 8 indices per 3 bytes (24 bits)
- 4-bit: 2 indices per byte
- 5-bit: 8 indices per 5 bytes (40 bits)
- 6-bit: 4 indices per 3 bytes (24 bits)
- 8-bit: 1 index per byte (no packing needed)

Storage format:
- Packed indices stored as uint8 array
- Header byte encodes bits_per_index
- Shape stored in metadata (not in packed data)
"""

from __future__ import annotations

import numpy as np


def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack N-bit indices into a uint8 array.

    Args:
        indices: Array of indices in range [0, 2^bits - 1].
                 Shape can be arbitrary but will be flattened internally.
        bits: Bits per index (2-8).

    Returns:
        Packed uint8 array. First byte is the bits value for decoding.

    Raises:
        ValueError: If bits is not in valid range or indices overflow.

    Example:
        >>> idx = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int16)
        >>> packed = pack_indices(idx, bits=2)
        >>> packed.shape  # Much smaller than original
        (3,)  # 1 header byte + 2 data bytes for 8 2-bit indices
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in range [2, 8], got {bits}")

    # Flatten and ensure proper dtype
    flat = indices.flatten().astype(np.uint16)
    n_indices = len(flat)

    # Validate range
    max_val = (1 << bits) - 1
    if flat.max() > max_val:
        raise ValueError(
            f"Index values exceed max for {bits}-bit: max={flat.max()}, limit={max_val}"
        )

    if bits == 8:
        # No packing needed, just add header
        packed = np.empty(1 + n_indices, dtype=np.uint8)
        packed[0] = bits
        packed[1:] = flat.astype(np.uint8)
        return packed

    if bits == 4:
        # 2 indices per byte (common case, optimized)
        n_bytes = (n_indices + 1) // 2
        packed = np.zeros(1 + n_bytes, dtype=np.uint8)
        packed[0] = bits

        # Pack pairs
        for i in range(0, n_indices - 1, 2):
            packed[1 + i // 2] = (flat[i] & 0xF) | ((flat[i + 1] & 0xF) << 4)

        # Handle odd element
        if n_indices % 2:
            packed[1 + n_indices // 2] = flat[-1] & 0xF

        return packed

    if bits == 2:
        # 4 indices per byte
        n_bytes = (n_indices + 3) // 4
        packed = np.zeros(1 + n_bytes, dtype=np.uint8)
        packed[0] = bits

        for i in range(0, n_indices - 3, 4):
            packed[1 + i // 4] = (
                (flat[i] & 0x3)
                | ((flat[i + 1] & 0x3) << 2)
                | ((flat[i + 2] & 0x3) << 4)
                | ((flat[i + 3] & 0x3) << 6)
            )

        # Handle remainder
        remainder = n_indices % 4
        if remainder:
            byte_idx = 1 + n_indices // 4
            val = 0
            for j in range(remainder):
                val |= (flat[n_indices - remainder + j] & 0x3) << (j * 2)
            packed[byte_idx] = val

        return packed

    # General case: bit-level packing (works for 3, 5, 6 bits)
    total_bits = n_indices * bits
    n_bytes = (total_bits + 7) // 8
    packed = np.zeros(1 + n_bytes, dtype=np.uint8)
    packed[0] = bits

    # Pack bit by bit (slower but correct for any bit width)
    bit_pos = 0
    for val in flat:
        for b in range(bits):
            if val & (1 << b):
                byte_idx = 1 + bit_pos // 8
                bit_offset = bit_pos % 8
                packed[byte_idx] |= 1 << bit_offset
            bit_pos += 1

    return packed


def unpack_indices(packed: np.ndarray, n_indices: int) -> np.ndarray:
    """Unpack a uint8 array back into N-bit indices.

    Args:
        packed: Packed uint8 array from pack_indices(). First byte is bits value.
        n_indices: Expected number of indices (needed since packing may pad).

    Returns:
        Array of unpacked indices as int16 (for Metal compatibility).

    Example:
        >>> packed = np.array([2, 0b11100100, 0b00011011], dtype=np.uint8)
        >>> unpack_indices(packed, 8)  # 8 2-bit indices
        array([0, 1, 2, 3, 3, 2, 1, 0], dtype=int16)
    """
    if len(packed) == 0:
        raise ValueError("Empty packed array")

    bits = int(packed[0])
    if bits < 2 or bits > 8:
        raise ValueError(f"Invalid bits value in header: {bits}")

    data = packed[1:]

    if bits == 8:
        # No unpacking needed
        return data[:n_indices].astype(np.int16)

    if bits == 4:
        # 2 indices per byte (optimized)
        unpacked = np.zeros(n_indices, dtype=np.int16)
        for i in range(0, n_indices - 1, 2):
            byte = data[i // 2]
            unpacked[i] = byte & 0xF
            unpacked[i + 1] = (byte >> 4) & 0xF

        if n_indices % 2:
            unpacked[-1] = data[n_indices // 2] & 0xF

        return unpacked

    if bits == 2:
        # 4 indices per byte (optimized)
        unpacked = np.zeros(n_indices, dtype=np.int16)
        for i in range(0, n_indices - 3, 4):
            byte = data[i // 4]
            unpacked[i] = byte & 0x3
            unpacked[i + 1] = (byte >> 2) & 0x3
            unpacked[i + 2] = (byte >> 4) & 0x3
            unpacked[i + 3] = (byte >> 6) & 0x3

        remainder = n_indices % 4
        if remainder:
            byte = data[n_indices // 4]
            base = n_indices - remainder
            for j in range(remainder):
                unpacked[base + j] = (byte >> (j * 2)) & 0x3

        return unpacked

    # General case: bit-level unpacking
    unpacked = np.zeros(n_indices, dtype=np.int16)
    mask = (1 << bits) - 1

    bit_pos = 0
    for i in range(n_indices):
        val = 0
        for b in range(bits):
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            if data[byte_idx] & (1 << bit_offset):
                val |= 1 << b
            bit_pos += 1
        unpacked[i] = val

    return unpacked


def pack_trellis_indices(
    indices: np.ndarray,
    bits: int,
) -> tuple[np.ndarray, dict]:
    """Pack trellis indices with shape preservation metadata.

    Args:
        indices: Trellis indices [tiles_k, tiles_n, 256] int16
        bits: Quantization bits (2-8)

    Returns:
        Tuple of (packed_data, metadata) where metadata contains:
        - "shape": Original shape as list
        - "n_indices": Total number of indices
        - "bits": Bits per index
        - "dtype": "packed_uint8"
    """
    original_shape = list(indices.shape)
    n_indices = int(np.prod(original_shape))

    packed = pack_indices(indices, bits)

    metadata = {
        "shape": original_shape,
        "n_indices": n_indices,
        "bits": bits,
        "dtype": "packed_uint8",
    }

    return packed, metadata


def unpack_trellis_indices(
    packed: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    """Unpack trellis indices using stored metadata.

    Args:
        packed: Packed uint8 array from pack_trellis_indices()
        metadata: Metadata dict with shape, n_indices, bits

    Returns:
        Unpacked indices in original shape as int16
    """
    n_indices = metadata["n_indices"]
    shape = tuple(metadata["shape"])

    unpacked = unpack_indices(packed, n_indices)
    return unpacked.reshape(shape)


def compute_packed_size(n_indices: int, bits: int) -> int:
    """Compute packed size in bytes for given parameters.

    Args:
        n_indices: Number of indices to pack
        bits: Bits per index

    Returns:
        Total bytes (including 1-byte header)
    """
    if bits == 8:
        return 1 + n_indices
    return 1 + (n_indices * bits + 7) // 8


def compute_compression_ratio(shape: tuple, bits: int) -> float:
    """Compute compression ratio vs int16 storage.

    Args:
        shape: Array shape
        bits: Bits per index

    Returns:
        Compression ratio (e.g., 5.33 for 3-bit vs 16-bit)
    """
    n_indices = int(np.prod(shape))
    int16_bytes = n_indices * 2  # int16 = 2 bytes
    packed_bytes = compute_packed_size(n_indices, bits)
    return int16_bytes / packed_bytes


# Vectorized versions for better performance on large arrays
def pack_indices_vectorized(indices: np.ndarray, bits: int) -> np.ndarray:
    """Vectorized version of pack_indices for better performance.

    Uses numpy operations for faster packing on large arrays.
    Supports 2, 3, 4, 5, 6, 8 bit packing.
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in range [2, 8], got {bits}")

    flat = indices.flatten().astype(np.uint32)
    n_indices = len(flat)

    if bits == 8:
        packed = np.empty(1 + n_indices, dtype=np.uint8)
        packed[0] = bits
        packed[1:] = flat.astype(np.uint8)
        return packed

    if bits == 4:
        # Vectorized 4-bit packing: 2 per byte
        n_pairs = (n_indices + 1) // 2
        padded = np.zeros(n_pairs * 2, dtype=np.uint32)
        padded[:n_indices] = flat

        low = padded[0::2] & 0xF
        high = (padded[1::2] & 0xF) << 4
        data = (low | high).astype(np.uint8)

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    if bits == 2:
        # Vectorized 2-bit packing: 4 per byte
        n_quads = (n_indices + 3) // 4
        padded = np.zeros(n_quads * 4, dtype=np.uint32)
        padded[:n_indices] = flat

        b0 = padded[0::4] & 0x3
        b1 = (padded[1::4] & 0x3) << 2
        b2 = (padded[2::4] & 0x3) << 4
        b3 = (padded[3::4] & 0x3) << 6
        data = (b0 | b1 | b2 | b3).astype(np.uint8)

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    if bits == 3:
        # Vectorized 3-bit packing: 8 indices -> 3 bytes (24 bits)
        n_groups = (n_indices + 7) // 8
        padded = np.zeros(n_groups * 8, dtype=np.uint32)
        padded[:n_indices] = flat

        # Reshape to groups of 8
        groups = padded.reshape(-1, 8)

        # Pack 8 3-bit values into 3 bytes (24 bits)
        # Combine into 32-bit integers, then extract bytes
        packed_32 = (
            (groups[:, 0] & 0x7)
            | ((groups[:, 1] & 0x7) << 3)
            | ((groups[:, 2] & 0x7) << 6)
            | ((groups[:, 3] & 0x7) << 9)
            | ((groups[:, 4] & 0x7) << 12)
            | ((groups[:, 5] & 0x7) << 15)
            | ((groups[:, 6] & 0x7) << 18)
            | ((groups[:, 7] & 0x7) << 21)
        )

        # Extract 3 bytes per group
        byte0 = (packed_32 & 0xFF).astype(np.uint8)
        byte1 = ((packed_32 >> 8) & 0xFF).astype(np.uint8)
        byte2 = ((packed_32 >> 16) & 0xFF).astype(np.uint8)

        # Interleave bytes
        data = np.empty(n_groups * 3, dtype=np.uint8)
        data[0::3] = byte0
        data[1::3] = byte1
        data[2::3] = byte2

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    if bits == 6:
        # Vectorized 6-bit packing: 4 indices -> 3 bytes (24 bits)
        n_groups = (n_indices + 3) // 4
        padded = np.zeros(n_groups * 4, dtype=np.uint32)
        padded[:n_indices] = flat

        # Reshape to groups of 4
        groups = padded.reshape(-1, 4)

        # Pack 4 6-bit values into 3 bytes (24 bits)
        packed_24 = (
            (groups[:, 0] & 0x3F)
            | ((groups[:, 1] & 0x3F) << 6)
            | ((groups[:, 2] & 0x3F) << 12)
            | ((groups[:, 3] & 0x3F) << 18)
        )

        # Extract 3 bytes per group
        byte0 = (packed_24 & 0xFF).astype(np.uint8)
        byte1 = ((packed_24 >> 8) & 0xFF).astype(np.uint8)
        byte2 = ((packed_24 >> 16) & 0xFF).astype(np.uint8)

        # Interleave bytes
        data = np.empty(n_groups * 3, dtype=np.uint8)
        data[0::3] = byte0
        data[1::3] = byte1
        data[2::3] = byte2

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    # Fall back to general case for 5-bit
    return pack_indices(indices, bits)


def unpack_indices_vectorized(packed: np.ndarray, n_indices: int) -> np.ndarray:
    """Vectorized version of unpack_indices for better performance."""
    if len(packed) == 0:
        raise ValueError("Empty packed array")

    bits = int(packed[0])
    data = packed[1:]

    if bits == 8:
        return data[:n_indices].astype(np.int16)

    if bits == 4:
        # Vectorized 4-bit unpacking
        expanded = np.zeros(len(data) * 2, dtype=np.int16)
        expanded[0::2] = data & 0xF
        expanded[1::2] = (data >> 4) & 0xF
        return expanded[:n_indices]

    if bits == 2:
        # Vectorized 2-bit unpacking
        expanded = np.zeros(len(data) * 4, dtype=np.int16)
        expanded[0::4] = data & 0x3
        expanded[1::4] = (data >> 2) & 0x3
        expanded[2::4] = (data >> 4) & 0x3
        expanded[3::4] = (data >> 6) & 0x3
        return expanded[:n_indices]

    if bits == 3:
        # Vectorized 3-bit unpacking: 3 bytes -> 8 indices (24 bits)
        n_groups = len(data) // 3
        if n_groups == 0:
            return unpack_indices(packed, n_indices)

        # Extract bytes in groups of 3
        byte0 = data[0::3].astype(np.uint32)
        byte1 = data[1::3].astype(np.uint32)
        byte2 = data[2::3].astype(np.uint32)

        # Combine into 24-bit integers
        packed_24 = byte0 | (byte1 << 8) | (byte2 << 16)

        # Extract 8 3-bit values from each 24-bit integer
        expanded = np.zeros(n_groups * 8, dtype=np.int16)
        expanded[0::8] = packed_24 & 0x7
        expanded[1::8] = (packed_24 >> 3) & 0x7
        expanded[2::8] = (packed_24 >> 6) & 0x7
        expanded[3::8] = (packed_24 >> 9) & 0x7
        expanded[4::8] = (packed_24 >> 12) & 0x7
        expanded[5::8] = (packed_24 >> 15) & 0x7
        expanded[6::8] = (packed_24 >> 18) & 0x7
        expanded[7::8] = (packed_24 >> 21) & 0x7
        return expanded[:n_indices]

    if bits == 6:
        # Vectorized 6-bit unpacking: 3 bytes -> 4 indices (24 bits)
        n_groups = len(data) // 3
        if n_groups == 0:
            return unpack_indices(packed, n_indices)

        # Extract bytes in groups of 3
        byte0 = data[0::3].astype(np.uint32)
        byte1 = data[1::3].astype(np.uint32)
        byte2 = data[2::3].astype(np.uint32)

        # Combine into 24-bit integers
        packed_24 = byte0 | (byte1 << 8) | (byte2 << 16)

        # Extract 4 6-bit values from each 24-bit integer
        expanded = np.zeros(n_groups * 4, dtype=np.int16)
        expanded[0::4] = packed_24 & 0x3F
        expanded[1::4] = (packed_24 >> 6) & 0x3F
        expanded[2::4] = (packed_24 >> 12) & 0x3F
        expanded[3::4] = (packed_24 >> 18) & 0x3F
        return expanded[:n_indices]

    # Fall back to general case for 5-bit
    return unpack_indices(packed, n_indices)
