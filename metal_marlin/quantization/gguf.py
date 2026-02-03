"""GGUF file format parser and Q4_K quantization support.

GGUF (GPT-Generated Unified Format) is the binary format used by llama.cpp
for storing quantized large language models.

This module provides:
1. GGUF file parsing (header, metadata, tensor info)
2. Q4_K weight extraction and unpacking
3. Dequantization kernel interface for Metal
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ============================================================================
# GGUF File Format Constants
# ============================================================================

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_UINT64 = 6
GGUF_TYPE_INT64 = 7
GGUF_TYPE_FLOAT32 = 8
GGUF_TYPE_BOOL = 9
GGUF_TYPE_STRING = 10
GGUF_TYPE_ARRAY = 11
GGUF_TYPE_UINT64_ARRAY = 12

# GGUF tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15

# Q4_K block parameters
Q4_K_BLOCK_SIZE = 32
Q4_K_SUPERBLOCK_SIZE = 256
Q4_K_PACKED_SIZE = 18  # bytes per Q4_K block

# ============================================================================
# GGUF Header
# ============================================================================


class GGUFHeader:
    """GGUF file header information."""

    def __init__(self, magic: bytes, version: int, tensor_count: int, metadata_kv_count: int):
        self.magic = magic
        self.version = version
        self.tensor_count = tensor_count
        self.metadata_kv_count = metadata_kv_count

    @classmethod
    def from_bytes(cls, data: bytes) -> GGUFHeader:
        """Parse GGUF header from bytes."""
        if len(data) < 16:
            raise ValueError("GGUF header too short")

        magic = data[0:4]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic}")

        version = struct.unpack("<I", data[4:8])[0]
        tensor_count = struct.unpack("<Q", data[8:16])[0]
        metadata_kv_count = struct.unpack("<Q", data[16:24])[0]

        return cls(magic, version, tensor_count, metadata_kv_count)


# ============================================================================
# GGUF Tensor Information
# ============================================================================


class GGUFTensorInfo:
    """Information about a tensor in the GGUF file."""

    def __init__(
        self,
        name: str,
        dimensions: list[int],
        ggml_type: int,
        offset: int,
    ):
        self.name = name
        self.dimensions = dimensions
        self.ggml_type = ggml_type
        self.offset = offset

    @property
    def num_elements(self) -> int:
        """Total number of elements in the tensor."""
        result = 1
        for dim in self.dimensions:
            result *= dim
        return result

    @property
    def is_q4_k(self) -> bool:
        """Check if this tensor uses Q4_K quantization."""
        return self.ggml_type == GGML_TYPE_Q4_K


# ============================================================================
# GGUF File Parser
# ============================================================================


class GGUFFile:
    """Parser for GGUF quantized model files."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._file = None
        self.header: GGUFHeader | None = None
        self.tensors: list[GGUFTensorInfo] = []
        self.metadata: dict[str, any] = {}

    def __enter__(self):
        self._file = open(self.filepath, "rb")
        self._parse_header()
        self._parse_metadata()
        self._parse_tensor_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def _parse_header(self):
        """Parse GGUF file header."""
        header_data = self._file.read(24)
        self.header = GGUFHeader.from_bytes(header_data)

    def _parse_metadata(self):
        """Parse GGUF metadata key-value pairs."""
        for _ in range(self.header.metadata_kv_count):
            key_len = struct.unpack("<Q", self._file.read(8))[0]
            key = self._file.read(key_len).decode("utf-8")

            value_type = struct.unpack("<I", self._file.read(4))[0]
            value = self._read_value(value_type)
            self.metadata[key] = value

    def _parse_tensor_info(self):
        """Parse tensor information."""
        for _ in range(self.header.tensor_count):
            name_len = struct.unpack("<Q", self._file.read(8))[0]
            name = self._file.read(name_len).decode("utf-8")

            n_dims = struct.unpack("<I", self._file.read(4))[0]
            dimensions = [struct.unpack("<Q", self._file.read(8))[0] for _ in range(n_dims)]

            ggml_type = struct.unpack("<I", self._file.read(4))[0]
            offset = struct.unpack("<Q", self._file.read(8))[0]

            self.tensors.append(GGUFTensorInfo(name, dimensions, ggml_type, offset))

    def _read_value(self, value_type: int) -> any:
        """Read a GGUF value of the given type."""
        if value_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", self._file.read(1))[0]
        elif value_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", self._file.read(1))[0]
        elif value_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", self._file.read(2))[0]
        elif value_type == GGML_TYPE_INT16:
            return struct.unpack("<h", self._file.read(2))[0]
        elif value_type == GGUF_TYPE_UINT32:
            return struct.unpack("<I", self._file.read(4))[0]
        elif value_type == GGML_TYPE_INT32:
            return struct.unpack("<i", self._file.read(4))[0]
        elif value_type == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", self._file.read(8))[0]
        elif value_type == GGML_TYPE_INT64:
            return struct.unpack("<q", self._file.read(8))[0]
        elif value_type == GGML_TYPE_FLOAT32:
            return struct.unpack("<f", self._file.read(4))[0]
        elif value_type == GGML_TYPE_BOOL:
            return struct.unpack("<?", self._file.read(1))[0]
        elif value_type == GGML_TYPE_STRING:
            str_len = struct.unpack("<Q", self._file.read(8))[0]
            return self._file.read(str_len).decode("utf-8")
        elif value_type == GGML_TYPE_ARRAY:
            value_len = struct.unpack("<Q", self._file.read(8))[0]
            elem_type = struct.unpack("<I", self._file.read(4))[0]
            return [self._read_value(elem_type) for _ in range(value_len)]
        elif value_type == GGML_TYPE_UINT64_ARRAY:
            arr_len = struct.unpack("<Q", self._file.read(8))[0]
            return [struct.unpack("<Q", self._file.read(8))[0] for _ in range(arr_len)]
        else:
            raise ValueError(f"Unknown GGUF value type: {value_type}")

    def get_tensor_data(self, tensor_info: GGUFTensorInfo) -> bytes:
        """Read raw tensor data from file."""
        self._file.seek(tensor_info.offset)
        size = self._get_tensor_size(tensor_info)
        return self._file.read(size)

    def _get_tensor_size(self, tensor_info: GGUFTensorInfo) -> int:
        """Calculate the size of a tensor in bytes."""
        if tensor_info.ggml_type == GGML_TYPE_Q4_K:
            n_elements = tensor_info.num_elements
            n_blocks = (n_elements + Q4_K_BLOCK_SIZE - 1) // Q4_K_BLOCK_SIZE
            return n_blocks * Q4_K_PACKED_SIZE
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_info.ggml_type}")

    def find_tensor(self, name: str) -> GGUFTensorInfo | None:
        """Find a tensor by name."""
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        return None


# ============================================================================
# Q4_K Quantization
# ============================================================================


def decode_q4_k_scale(scale_bits: int) -> np.float16:
    """Decode Q4_K scale from 4-bit encoded value.

    scale = exp2(scale_bits) / 16.0
    """
    return np.float16(np.exp2(scale_bits) / 16.0)


def decode_q4_k_min(min_bits: int) -> np.float16:
    """Decode Q4_K min value from 4-bit encoded value.

    min = -exp2(min_bits)
    """
    return np.float16(-np.exp2(min_bits))


class Q4KWeights:
    """Container for Q4_K quantized weights with metadata."""

    def __init__(
        self,
        packed_data: NDArray[np.uint8],
        scales: NDArray[np.float16],
        mins: NDArray[np.float16],
        num_elements: int,
    ):
        self.packed_data = packed_data
        self.scales = scales
        self.mins = mins
        self.num_elements = num_elements
        self.num_blocks = (num_elements + Q4_K_BLOCK_SIZE - 1) // Q4_K_BLOCK_SIZE


def unpack_q4_k_block(
    block_data: bytes,
) -> tuple[NDArray[np.uint8], NDArray[np.float16], np.float16]:
    """Unpack a single Q4_K block.

    Q4_K block layout (18 bytes):
    - Bytes 0-15: 128 packed 4-bit weights (32 weights)
    - Byte 16: 2 scales (4 bits each)
    - Byte 17: 1 min value (4 bits) + 4 bits reserved

    Returns:
        (weights, scales, min)
        weights: 32 uint8 values (4-bit weights unpacked)
        scales: 2 FP16 scale values
        min: 1 FP16 minimum value
    """
    if len(block_data) != Q4_K_PACKED_SIZE:
        raise ValueError(f"Q4_K block must be {Q4_K_PACKED_SIZE} bytes, got {len(block_data)}")

    # Extract packed 4-bit weights (128 bits = 16 bytes)
    packed_weights = np.frombuffer(block_data[:16], dtype=np.uint8)

    # Unpack 4-bit weights (each byte contains 2 weights)
    weights = np.empty(32, dtype=np.uint8)
    for i in range(16):
        lo = packed_weights[i] & 0x0F
        hi = (packed_weights[i] >> 4) & 0x0F
        weights[i * 2] = lo
        weights[i * 2 + 1] = hi

    # Extract scales (byte 16: 2x4-bit scales)
    scale_byte = block_data[16]
    scale_lo = scale_byte & 0x0F
    scale_hi = (scale_byte >> 4) & 0x0F

    # Convert to FP16 scale values (scales are encoded as 4-bit indices)
    scales_fp16 = np.array(
        [decode_q4_k_scale(scale_lo), decode_q4_k_scale(scale_hi)], dtype=np.float16
    )

    # Extract min value (byte 17: 4-bit min)
    min_byte = block_data[17]
    min_bits = min_byte & 0x0F

    # Min encoding: min = -exp2(index)
    min_fp16 = decode_q4_k_min(min_bits)

    return weights, scales_fp16, min_fp16


def extract_q4_k_weights(tensor_data: bytes, num_elements: int) -> Q4KWeights:
    """Extract Q4_K weights from tensor data.

    Args:
        tensor_data: Raw Q4_K quantized tensor data
        num_elements: Total number of elements in the tensor

    Returns:
        Q4KWeights object with packed data, scales, and mins
    """
    n_blocks = (num_elements + Q4_K_BLOCK_SIZE - 1) // Q4_K_BLOCK_SIZE

    if len(tensor_data) != n_blocks * Q4_K_PACKED_SIZE:
        raise ValueError(
            f"Tensor data size mismatch: expected {n_blocks * Q4_K_PACKED_SIZE} bytes, "
            f"got {len(tensor_data)}"
        )

    # Pre-allocate arrays
    packed_data = np.frombuffer(tensor_data, dtype=np.uint8)
    scales = np.zeros(n_blocks * 2, dtype=np.float16)
    mins = np.zeros(n_blocks, dtype=np.float16)

    # Unpack each block
    for block_idx in range(n_blocks):
        block_start = block_idx * Q4_K_PACKED_SIZE
        block_data = tensor_data[block_start : block_start + Q4_K_PACKED_SIZE]

        _, block_scales, block_min = unpack_q4_k_block(block_data)

        scales[block_idx * 2] = block_scales[0]
        scales[block_idx * 2 + 1] = block_scales[1]
        mins[block_idx] = block_min

    return Q4KWeights(packed_data, scales, mins, num_elements)


def dequantize_q4_k_cpu(weights: Q4KWeights) -> NDArray[np.float16]:
    """Dequantize Q4_K weights on CPU (reference implementation).

    Args:
        weights: Q4KWeights object

    Returns:
        Dequantized FP16 weights
    """
    n_elements = weights.num_elements
    output = np.zeros(n_elements, dtype=np.float16)

    for block_idx in range(weights.num_blocks):
        block_start = block_idx * Q4_K_PACKED_SIZE
        block_data = bytes(weights.packed_data[block_start : block_start + Q4_K_PACKED_SIZE])

        block_weights, _, _ = unpack_q4_k_block(block_data)

        # Use pre-parsed scales and mins
        scale_lo = weights.scales[block_idx * 2]
        scale_hi = weights.scales[block_idx * 2 + 1]
        block_min = weights.mins[block_idx]

        # Dequantize each weight in the block
        output_start = block_idx * Q4_K_BLOCK_SIZE
        for i in range(min(Q4_K_BLOCK_SIZE, n_elements - output_start)):
            # Determine which scale to use (first 16 weights use scale_lo, next 16 use scale_hi)
            scale = scale_lo if i < 16 else scale_hi

            # Q4_K dequantization: weight = (packed_weight - 8) * scale + min
            weight_uint8 = block_weights[i]
            # Convert to signed int to avoid uint8 underflow before converting to float
            weight_signed = np.float16(int(weight_uint8) - 8)
            output[output_start + i] = weight_signed * scale + block_min

    return output


# ============================================================================
# Utility Functions
# ============================================================================


def load_gguf_model(filepath: str | Path) -> GGUFFile:
    """Load a GGUF model file.

    Args:
        filepath: Path to GGUF file

    Returns:
        GGUFFile object
    """
    return GGUFFile(filepath)


def get_q4_k_tensors(gguf_file: GGUFFile) -> list[GGUFTensorInfo]:
    """Get all Q4_K quantized tensors from a GGUF file.

    Args:
        gguf_file: GGUFFile object

    Returns:
        List of GGUFTensorInfo objects for Q4_K tensors
    """
    return [t for t in gguf_file.tensors if t.is_q4_k]


def prepare_q4_k_for_metal(
    weights: Q4KWeights,
) -> tuple[NDArray[np.uint8], NDArray[np.float16], NDArray[np.float16]]:
    """Prepare Q4_K weights for Metal dequantization kernel.

    Converts internal representation to Metal-friendly format:
    - Packed weights remain as uint8
    - Scales as FP16 array
    - Min values as FP16 array

    Args:
        weights: Q4KWeights object

    Returns:
        (packed_data, scales, mins) tuple suitable for Metal buffers
    """
    return weights.packed_data, weights.scales, weights.mins
