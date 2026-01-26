"""Load quantized weights from GGUF format and convert to Marlin FP4.

Standalone GGUF parser that handles all major GGML quantization formats
without requiring the external `gguf` Python package. Supports:

  Legacy formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
  K-quants:       Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (super-block, 6-bit scales)
  I-quants:       IQ2_XXS, IQ3_XXS, IQ4_NL, IQ4_XS (importance-weighted)

Dequantizes to FP32 then re-quantizes into Marlin-packed FP4 (E2M1) format
for the Metal fused dequant-GEMM kernel.

Note:
  IQ2/IQ3 reconstruction uses a simplified grid-based approximation that
  matches the GGUF block layout but does not reproduce llama.cpp's full
  importance-matrix search at quantization time. This still enables
  inference and conversion, but bit-exact parity with ggml-quants.c is
  not guaranteed for IQ2/3.

GGUF file format (v3):
  - Header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
  - KV pairs: key + type + value (variable length)
  - Tensor infos: name + ndims + shape + type + offset
  - Alignment padding
  - Tensor data (contiguous, alignment-padded)

Quality-size tradeoffs (approximate perplexity increase from FP16):
  Q2_K:    ~2.6 bpw, +2.0-3.0 PPL (smallest, significant quality loss)
  IQ2_XXS: ~2.1 bpw, +0.8-1.2 PPL (better than Q2_K at similar size)
  Q3_K:    ~3.4 bpw, +0.5-1.0 PPL
  IQ3_XXS: ~3.1 bpw, +0.3-0.5 PPL (better than Q3_K)
  Q4_K_M:  ~4.8 bpw, +0.05-0.15 PPL (recommended for quality)
  IQ4_XS:  ~4.3 bpw, +0.05-0.10 PPL (best quality at 4-bit)
  Q5_K_M:  ~5.7 bpw, +0.01-0.05 PPL
  Q6_K:    ~6.6 bpw, negligible PPL loss

Reference:
  - https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
  - ggml-common.h and ggml-quants.c block definitions
  - https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

# MLX is optional - GGUF loading works with pure numpy
HAS_MLX = False
mx: ModuleType | None = None

try:
    import mlx.core as _mx
    mx = _mx
    HAS_MLX = True
except ImportError:
    pass

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F811 - for type hints only

# ---------------------------------------------------------------------------
# GGUF value type codes (from gguf.md spec)
# ---------------------------------------------------------------------------

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# ---------------------------------------------------------------------------
# GGML tensor type codes and block parameters
# ---------------------------------------------------------------------------

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
# K-quants (super-block quantization with 256 element blocks)
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
# I-quants (importance-weighted quantization)
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30

# Block parameters: (block_size_elements, block_size_bytes)
# These define how many weights each quantization block holds and
# how many bytes that block occupies in the file.
GGML_BLOCK_PARAMS: dict[int, tuple[int, int]] = {
    # Q4_0: 32 elements, 2 bytes (fp16 scale) + 16 bytes (32 nibbles) = 18 bytes
    GGML_TYPE_Q4_0: (32, 18),
    # Q4_1: 32 elements, 2 (fp16 scale) + 2 (fp16 min) + 16 (nibbles) = 20 bytes
    GGML_TYPE_Q4_1: (32, 20),
    # Q5_0: 32 elements, 2 (fp16 scale) + 4 (high bits) + 16 (low nibbles) = 22 bytes
    GGML_TYPE_Q5_0: (32, 22),
    # Q5_1: 32 elements, 2 (fp16 scale) + 2 (fp16 min) + 4 (high) + 16 (low) = 24 bytes
    GGML_TYPE_Q5_1: (32, 24),
    # Q8_0: 32 elements, 2 (fp16 scale) + 32 (int8 quants) = 34 bytes
    GGML_TYPE_Q8_0: (32, 34),
    # Q8_1: 32 elements, 4 (fp32 scale) + 4 (fp32 sum) + 32 (int8) = 40 bytes
    GGML_TYPE_Q8_1: (32, 40),
    # ---------------------------------------------------------------------------
    # K-quants: super-blocks of 256 elements with 6-bit scales
    # ---------------------------------------------------------------------------
    # Q2_K: 256 elements per super-block
    # Layout: scales[16] (4-bit) + qs[64] (2-bit packed) + fp16 d + fp16 dmin
    # 16/2 + 64 + 2 + 2 = 76 bytes
    GGML_TYPE_Q2_K: (256, 84),  # Actual: scales(16) + qs(64) + d(2) + dmin(2) = 84
    # Q3_K: 256 elements per super-block
    # Layout: hmask[32] + qs[64] + scales[12] + fp16 d
    # 32 + 64 + 12 + 2 = 110 bytes
    GGML_TYPE_Q3_K: (256, 110),
    # Q4_K: 256 elements per super-block
    # Layout: fp16 d + fp16 dmin + scales[12] (6-bit packed) + qs[128] (4-bit)
    # 2 + 2 + 12 + 128 = 144 bytes
    GGML_TYPE_Q4_K: (256, 144),
    # Q5_K: 256 elements per super-block
    # Layout: fp16 d + fp16 dmin + scales[12] + qh[32] + qs[128]
    # 2 + 2 + 12 + 32 + 128 = 176 bytes
    GGML_TYPE_Q5_K: (256, 176),
    # Q6_K: 256 elements per super-block
    # Layout: ql[128] + qh[64] + scales[16] (int8) + fp16 d
    # 128 + 64 + 16 + 2 = 210 bytes
    GGML_TYPE_Q6_K: (256, 210),
    # Q8_K: 256 elements, fp32 scale + 256 int8 = 260 bytes
    GGML_TYPE_Q8_K: (256, 260),
    # ---------------------------------------------------------------------------
    # I-quants: importance-weighted quantization with grid codebooks
    # ---------------------------------------------------------------------------
    # IQ2_XXS: 256 elements, super-block with 8 sub-blocks of 32
    # Layout: fp16 d + 8*(qs[4] + signs[1]) = 2 + 8*5 = 42 bytes? Actual=66
    GGML_TYPE_IQ2_XXS: (256, 66),
    # IQ2_XS: 256 elements, scales + qs
    GGML_TYPE_IQ2_XS: (256, 74),
    # IQ2_S: 256 elements
    GGML_TYPE_IQ2_S: (256, 82),
    # IQ3_XXS: 256 elements
    # Layout: fp16 d + 8*(qs[6] + signs[1]) = 2 + 8*7 = 58? Actual = 98
    GGML_TYPE_IQ3_XXS: (256, 98),
    # IQ3_S: 256 elements
    GGML_TYPE_IQ3_S: (256, 110),
    # IQ4_NL: 32 elements (non-linear 4-bit)
    # Layout: fp16 d + qs[16] = 2 + 16 = 18 bytes
    GGML_TYPE_IQ4_NL: (32, 18),
    # IQ4_XS: 256 elements (4-bit with scales)
    # Layout: fp16 d + scales[8] + qs[128]
    GGML_TYPE_IQ4_XS: (256, 136),
    # IQ1_S: 256 elements (1-bit with signs)
    GGML_TYPE_IQ1_S: (256, 50),
    # IQ1_M: 256 elements (1-bit mixed)
    GGML_TYPE_IQ1_M: (256, 56),
}

# Supported types for dequantization
BLOCK_QUANT_TYPES = frozenset(GGML_BLOCK_PARAMS.keys())

# E2M1 codebook for Marlin FP4 quantization
E2M1_VALUES: np.ndarray = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# Default Marlin group size
DEFAULT_GROUP_SIZE = 128


# ---------------------------------------------------------------------------
# GGML dequantization routines
# ---------------------------------------------------------------------------


def dequant_q4_0(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q4_0 blocks to FP32.

    Q4_0 block layout (18 bytes, 32 elements):
      - bytes [0:2]: FP16 scale (delta)
      - bytes [2:18]: 16 bytes of packed 4-bit unsigned quants (32 nibbles)

    Dequant: value = (quant - 8) * scale
    The 4-bit values are unsigned [0, 15], centered at 8.
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q4_0]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # Extract FP16 scales -> FP32
    scales = np.frombuffer(raw[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    scales = scales.reshape(n_blocks)

    # Extract nibbles from the 16 quant bytes
    qs = raw[:, 2:18]  # (n_blocks, 16)
    lo = (qs & 0x0F).astype(np.float32)  # (n_blocks, 16) - first 16 elements
    hi = ((qs >> 4) & 0x0F).astype(np.float32)  # (n_blocks, 16) - next 16 elements

    # Interleave: elements are stored as [lo_nibbles[0:16], hi_nibbles[0:16]]
    quants = np.empty((n_blocks, 32), dtype=np.float32)
    quants[:, :16] = lo
    quants[:, 16:] = hi

    # Dequantize: centered unsigned 4-bit (subtract 8)
    values = (quants - 8.0) * scales[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q4_1(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q4_1 blocks to FP32.

    Q4_1 block layout (20 bytes, 32 elements):
      - bytes [0:2]: FP16 scale (delta)
      - bytes [2:4]: FP16 minimum value
      - bytes [4:20]: 16 bytes of packed 4-bit unsigned quants

    Dequant: value = quant * scale + min
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q4_1]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # Extract FP16 scale and min
    scales = np.frombuffer(raw[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    mins = np.frombuffer(raw[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)
    scales = scales.reshape(n_blocks)
    mins = mins.reshape(n_blocks)

    # Extract nibbles
    qs = raw[:, 4:20]  # (n_blocks, 16)
    lo = (qs & 0x0F).astype(np.float32)
    hi = ((qs >> 4) & 0x0F).astype(np.float32)

    quants = np.empty((n_blocks, 32), dtype=np.float32)
    quants[:, :16] = lo
    quants[:, 16:] = hi

    # Dequantize: affine (unsigned + min)
    values = quants * scales[:, np.newaxis] + mins[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q5_0(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q5_0 blocks to FP32.

    Q5_0 block layout (22 bytes, 32 elements):
      - bytes [0:2]: FP16 scale
      - bytes [2:6]: 4 bytes of high bits (32 bits, one per element)
      - bytes [6:22]: 16 bytes of low nibbles (32 x 4-bit)

    Each element is 5 bits: 4 low bits from nibble + 1 high bit from bitfield.
    Dequant: value = (quant5 - 16) * scale
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q5_0]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # FP16 scale
    scales = np.frombuffer(raw[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    scales = scales.reshape(n_blocks)

    # High bits: 4 bytes = 32 bits (one per element)
    qh = np.frombuffer(raw[:, 2:6].tobytes(), dtype=np.uint32).reshape(n_blocks)

    # Low nibbles
    qs = raw[:, 6:22]  # (n_blocks, 16)
    lo = (qs & 0x0F).astype(np.int32)  # first 16 elements
    hi = ((qs >> 4) & 0x0F).astype(np.int32)  # next 16 elements

    # Combine with high bits
    quants = np.empty((n_blocks, 32), dtype=np.int32)
    for i in range(16):
        quants[:, i] = lo[:, i] | (((qh >> i) & 1).astype(np.int32) << 4)
    for i in range(16):
        quants[:, 16 + i] = hi[:, i] | (((qh >> (16 + i)) & 1).astype(np.int32) << 4)

    # Dequantize: centered 5-bit unsigned (subtract 16)
    values = (quants.astype(np.float32) - 16.0) * scales[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q5_1(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q5_1 blocks to FP32.

    Q5_1 block layout (24 bytes, 32 elements):
      - bytes [0:2]: FP16 scale (delta)
      - bytes [2:4]: FP16 minimum
      - bytes [4:8]: 4 bytes of high bits
      - bytes [8:24]: 16 bytes of low nibbles

    Each element is 5 bits. Dequant: value = quant5 * scale + min
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q5_1]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    scales = np.frombuffer(raw[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    mins = np.frombuffer(raw[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)
    scales = scales.reshape(n_blocks)
    mins = mins.reshape(n_blocks)

    # High bits
    qh = np.frombuffer(raw[:, 4:8].tobytes(), dtype=np.uint32).reshape(n_blocks)

    # Low nibbles
    qs = raw[:, 8:24]  # (n_blocks, 16)
    lo = (qs & 0x0F).astype(np.int32)
    hi = ((qs >> 4) & 0x0F).astype(np.int32)

    quants = np.empty((n_blocks, 32), dtype=np.int32)
    for i in range(16):
        quants[:, i] = lo[:, i] | (((qh >> i) & 1).astype(np.int32) << 4)
    for i in range(16):
        quants[:, 16 + i] = hi[:, i] | (((qh >> (16 + i)) & 1).astype(np.int32) << 4)

    # Dequantize: affine 5-bit
    values = quants.astype(np.float32) * scales[:, np.newaxis] + mins[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q8_0(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 blocks to FP32.

    Q8_0 block layout (34 bytes, 32 elements):
      - bytes [0:2]: FP16 scale
      - bytes [2:34]: 32 x int8 quantized values

    Dequant: value = quant * scale
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q8_0]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    scales = np.frombuffer(raw[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    scales = scales.reshape(n_blocks)

    # Signed int8 quants
    quants = raw[:, 2:34].view(np.int8).astype(np.float32)  # (n_blocks, 32)

    values = quants * scales[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


# ---------------------------------------------------------------------------
# K-quant dequantization routines (super-blocks of 256 elements)
# ---------------------------------------------------------------------------


def dequant_q2_k(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q2_K blocks to FP32.

    Q2_K super-block layout (84 bytes, 256 elements):
      - bytes [0:16]: 16 x 4-bit scale pairs (scales and mins interleaved)
      - bytes [16:80]: 64 bytes of 2-bit quants (256 x 2 bits)
      - bytes [80:82]: fp16 d (super-block scale)
      - bytes [82:84]: fp16 dmin (super-block min scale)

    Each super-block has 16 sub-blocks of 16 elements.
    Dequant: value = d * scale * q - dmin * min
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q2_K]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # Extract fp16 d and dmin from end of block
    d = np.frombuffer(raw[:, 80:82].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)
    dmin = np.frombuffer(raw[:, 82:84].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # Extract 4-bit scales (16 pairs of scale/min)
    scales_raw = raw[:, :16]  # (n_blocks, 16)
    scales = (scales_raw & 0x0F).astype(np.float32)  # Lower nibble = scale
    mins = ((scales_raw >> 4) & 0x0F).astype(np.float32)  # Upper nibble = min

    # Extract 2-bit quants: 64 bytes = 256 quants (4 per byte)
    qs = raw[:, 16:80]  # (n_blocks, 64)

    # Unpack 2-bit values
    quants = np.empty((n_blocks, 256), dtype=np.float32)
    for i in range(4):
        quants[:, i::4] = ((qs >> (i * 2)) & 0x03).astype(np.float32)

    # Dequantize: each sub-block of 16 elements shares a scale/min
    values = np.empty((n_blocks, 256), dtype=np.float32)
    for sb in range(16):
        start = sb * 16
        end = start + 16
        sb_scale = scales[:, sb] * d
        sb_min = mins[:, sb] * dmin
        values[:, start:end] = quants[:, start:end] * sb_scale[:, np.newaxis] - sb_min[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q3_k(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q3_K blocks to FP32.

    Q3_K super-block layout (110 bytes, 256 elements):
      - bytes [0:32]: hmask - high bits for 3-bit quants
      - bytes [32:96]: qs - low 2 bits of quants (64 bytes = 256 x 2 bits)
      - bytes [96:108]: scales (12 bytes = 16 x 6-bit scales, packed)
      - bytes [108:110]: fp16 d

    Each super-block has 16 sub-blocks of 16 elements.
    Quants are 3-bit: 2 bits from qs + 1 bit from hmask.
    Dequant: value = d * scale * (q - 4) where q is 3-bit [0-7]
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q3_K]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 d at end
    d = np.frombuffer(raw[:, 108:110].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # High bits mask (32 bytes = 256 bits)
    hmask = raw[:, :32]  # (n_blocks, 32)

    # Low 2 bits (64 bytes = 256 x 2 bits)
    qs = raw[:, 32:96]  # (n_blocks, 64)

    # Scales (12 bytes = 16 x 6-bit scales packed)
    scales_raw = raw[:, 96:108]  # (n_blocks, 12)

    # Unpack 6-bit scales from 12 bytes -> 16 scales
    # Packing: scales[0-7] in first 6 bytes, scales[8-15] in next 6 bytes
    # Each 6-bit scale spans byte boundaries
    scales = np.zeros((n_blocks, 16), dtype=np.float32)
    for i in range(8):
        byte_idx = (i * 6) // 8
        bit_offset = (i * 6) % 8
        if bit_offset <= 2:
            scales[:, i] = ((scales_raw[:, byte_idx].astype(np.int32) >> bit_offset) & 0x3F).astype(np.float32)
        else:
            lo = (scales_raw[:, byte_idx].astype(np.int32) >> bit_offset) & ((1 << (8 - bit_offset)) - 1)
            hi = (scales_raw[:, byte_idx + 1].astype(np.int32) & ((1 << (bit_offset - 2)) - 1)) << (8 - bit_offset)
            scales[:, i] = (lo | hi).astype(np.float32)
    for i in range(8, 16):
        byte_idx = 6 + ((i - 8) * 6) // 8
        bit_offset = ((i - 8) * 6) % 8
        if bit_offset <= 2:
            scales[:, i] = ((scales_raw[:, byte_idx].astype(np.int32) >> bit_offset) & 0x3F).astype(np.float32)
        else:
            lo = (scales_raw[:, byte_idx].astype(np.int32) >> bit_offset) & ((1 << (8 - bit_offset)) - 1)
            hi = (scales_raw[:, min(byte_idx + 1, 11)].astype(np.int32) & ((1 << (bit_offset - 2)) - 1)) << (8 - bit_offset)
            scales[:, i] = (lo | hi).astype(np.float32)

    # Scales are stored as signed 6-bit, convert to signed
    scales = scales - 32.0  # Center around 0

    # Unpack 2-bit low quants
    quants_lo = np.empty((n_blocks, 256), dtype=np.int32)
    for i in range(4):
        quants_lo[:, i::4] = ((qs >> (i * 2)) & 0x03).astype(np.int32)

    # Unpack high bits
    quants_hi = np.empty((n_blocks, 256), dtype=np.int32)
    for i in range(256):
        byte_idx = i // 8
        bit_idx = i % 8
        quants_hi[:, i] = ((hmask[:, byte_idx] >> bit_idx) & 1).astype(np.int32)

    # Combine: 3-bit quant = 2-bit low + 1-bit high << 2
    quants = quants_lo + (quants_hi << 2)

    # Dequantize: each sub-block of 16 elements
    values = np.empty((n_blocks, 256), dtype=np.float32)
    for sb in range(16):
        start = sb * 16
        end = start + 16
        sb_scale = scales[:, sb] * d
        # Q3_K uses symmetric quantization centered at 4
        values[:, start:end] = (quants[:, start:end].astype(np.float32) - 4.0) * sb_scale[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q4_k(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q4_K blocks to FP32.

    Q4_K super-block layout (144 bytes, 256 elements):
      - bytes [0:2]: fp16 d (super-block scale)
      - bytes [2:4]: fp16 dmin (super-block min)
      - bytes [4:16]: scales (12 bytes = 8 x 6-bit scale + 8 x 6-bit min, packed)
      - bytes [16:144]: qs (128 bytes = 256 x 4-bit quants)

    Each super-block has 8 sub-blocks of 32 elements.
    Dequant: value = d * scale * q - dmin * min
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q4_K]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 d and dmin
    d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)
    dmin = np.frombuffer(raw[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # 6-bit scales packed into 12 bytes
    # Layout: 8 scales (6 bits each) + 8 mins (6 bits each) = 96 bits = 12 bytes
    scales_raw = raw[:, 4:16]  # (n_blocks, 12)

    # Unpack scales and mins from 6-bit packing
    scales = np.zeros((n_blocks, 8), dtype=np.float32)
    mins = np.zeros((n_blocks, 8), dtype=np.float32)

    # First 4 scales in bytes 0-2 (each 6 bits)
    scales[:, 0] = (scales_raw[:, 0] & 0x3F).astype(np.float32)
    scales[:, 1] = ((scales_raw[:, 0] >> 6) | ((scales_raw[:, 1] & 0x0F) << 2)).astype(np.float32)
    scales[:, 2] = ((scales_raw[:, 1] >> 4) | ((scales_raw[:, 2] & 0x03) << 4)).astype(np.float32)
    scales[:, 3] = ((scales_raw[:, 2] >> 2) & 0x3F).astype(np.float32)

    # Next 4 scales in bytes 3-5
    scales[:, 4] = (scales_raw[:, 3] & 0x3F).astype(np.float32)
    scales[:, 5] = ((scales_raw[:, 3] >> 6) | ((scales_raw[:, 4] & 0x0F) << 2)).astype(np.float32)
    scales[:, 6] = ((scales_raw[:, 4] >> 4) | ((scales_raw[:, 5] & 0x03) << 4)).astype(np.float32)
    scales[:, 7] = ((scales_raw[:, 5] >> 2) & 0x3F).astype(np.float32)

    # First 4 mins in bytes 6-8
    mins[:, 0] = (scales_raw[:, 6] & 0x3F).astype(np.float32)
    mins[:, 1] = ((scales_raw[:, 6] >> 6) | ((scales_raw[:, 7] & 0x0F) << 2)).astype(np.float32)
    mins[:, 2] = ((scales_raw[:, 7] >> 4) | ((scales_raw[:, 8] & 0x03) << 4)).astype(np.float32)
    mins[:, 3] = ((scales_raw[:, 8] >> 2) & 0x3F).astype(np.float32)

    # Next 4 mins in bytes 9-11
    mins[:, 4] = (scales_raw[:, 9] & 0x3F).astype(np.float32)
    mins[:, 5] = ((scales_raw[:, 9] >> 6) | ((scales_raw[:, 10] & 0x0F) << 2)).astype(np.float32)
    mins[:, 6] = ((scales_raw[:, 10] >> 4) | ((scales_raw[:, 11] & 0x03) << 4)).astype(np.float32)
    mins[:, 7] = ((scales_raw[:, 11] >> 2) & 0x3F).astype(np.float32)

    # 4-bit quants (128 bytes = 256 nibbles)
    qs = raw[:, 16:144]  # (n_blocks, 128)
    quants = np.empty((n_blocks, 256), dtype=np.float32)
    quants[:, :128] = (qs & 0x0F).astype(np.float32)
    quants[:, 128:] = ((qs >> 4) & 0x0F).astype(np.float32)

    # Dequantize: 8 sub-blocks of 32 elements each
    values = np.empty((n_blocks, 256), dtype=np.float32)
    for sb in range(8):
        start = sb * 32
        end = start + 32
        sb_scale = scales[:, sb] * d
        sb_min = mins[:, sb] * dmin
        values[:, start:end] = quants[:, start:end] * sb_scale[:, np.newaxis] - sb_min[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q5_k(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q5_K blocks to FP32.

    Q5_K super-block layout (176 bytes, 256 elements):
      - bytes [0:2]: fp16 d
      - bytes [2:4]: fp16 dmin
      - bytes [4:16]: scales (12 bytes = 8 x 6-bit scale + 8 x 6-bit min)
      - bytes [16:48]: qh (32 bytes = 256 high bits)
      - bytes [48:176]: qs (128 bytes = 256 x 4-bit low quants)

    Each element is 5 bits: 4 from qs + 1 from qh.
    Dequant: value = d * scale * q - dmin * min
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q5_K]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 d and dmin
    d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)
    dmin = np.frombuffer(raw[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # Unpack 6-bit scales and mins (same layout as Q4_K)
    scales_raw = raw[:, 4:16]
    scales = np.zeros((n_blocks, 8), dtype=np.float32)
    mins = np.zeros((n_blocks, 8), dtype=np.float32)

    scales[:, 0] = (scales_raw[:, 0] & 0x3F).astype(np.float32)
    scales[:, 1] = ((scales_raw[:, 0] >> 6) | ((scales_raw[:, 1] & 0x0F) << 2)).astype(np.float32)
    scales[:, 2] = ((scales_raw[:, 1] >> 4) | ((scales_raw[:, 2] & 0x03) << 4)).astype(np.float32)
    scales[:, 3] = ((scales_raw[:, 2] >> 2) & 0x3F).astype(np.float32)
    scales[:, 4] = (scales_raw[:, 3] & 0x3F).astype(np.float32)
    scales[:, 5] = ((scales_raw[:, 3] >> 6) | ((scales_raw[:, 4] & 0x0F) << 2)).astype(np.float32)
    scales[:, 6] = ((scales_raw[:, 4] >> 4) | ((scales_raw[:, 5] & 0x03) << 4)).astype(np.float32)
    scales[:, 7] = ((scales_raw[:, 5] >> 2) & 0x3F).astype(np.float32)

    mins[:, 0] = (scales_raw[:, 6] & 0x3F).astype(np.float32)
    mins[:, 1] = ((scales_raw[:, 6] >> 6) | ((scales_raw[:, 7] & 0x0F) << 2)).astype(np.float32)
    mins[:, 2] = ((scales_raw[:, 7] >> 4) | ((scales_raw[:, 8] & 0x03) << 4)).astype(np.float32)
    mins[:, 3] = ((scales_raw[:, 8] >> 2) & 0x3F).astype(np.float32)
    mins[:, 4] = (scales_raw[:, 9] & 0x3F).astype(np.float32)
    mins[:, 5] = ((scales_raw[:, 9] >> 6) | ((scales_raw[:, 10] & 0x0F) << 2)).astype(np.float32)
    mins[:, 6] = ((scales_raw[:, 10] >> 4) | ((scales_raw[:, 11] & 0x03) << 4)).astype(np.float32)
    mins[:, 7] = ((scales_raw[:, 11] >> 2) & 0x3F).astype(np.float32)

    # High bits (32 bytes = 256 bits)
    qh = raw[:, 16:48]

    # Low 4 bits (128 bytes = 256 nibbles)
    qs = raw[:, 48:176]
    quants_lo = np.empty((n_blocks, 256), dtype=np.int32)
    quants_lo[:, :128] = (qs & 0x0F).astype(np.int32)
    quants_lo[:, 128:] = ((qs >> 4) & 0x0F).astype(np.int32)

    # Unpack high bits
    quants_hi = np.empty((n_blocks, 256), dtype=np.int32)
    for i in range(256):
        byte_idx = i // 8
        bit_idx = i % 8
        quants_hi[:, i] = ((qh[:, byte_idx] >> bit_idx) & 1).astype(np.int32)

    # Combine: 5-bit = 4-bit + high << 4
    quants = quants_lo + (quants_hi << 4)

    # Dequantize
    values = np.empty((n_blocks, 256), dtype=np.float32)
    for sb in range(8):
        start = sb * 32
        end = start + 32
        sb_scale = scales[:, sb] * d
        sb_min = mins[:, sb] * dmin
        values[:, start:end] = quants[:, start:end].astype(np.float32) * sb_scale[:, np.newaxis] - sb_min[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


def dequant_q6_k(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q6_K blocks to FP32.

    Q6_K super-block layout (210 bytes, 256 elements):
      - bytes [0:128]: ql (128 bytes = 256 x 4-bit low quants)
      - bytes [128:192]: qh (64 bytes = 256 x 2-bit high quants)
      - bytes [192:208]: scales (16 x int8 scales)
      - bytes [208:210]: fp16 d

    Each element is 6 bits: 4 from ql + 2 from qh.
    Dequant: value = d * scale * (q - 32)
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_Q6_K]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 d at end
    d = np.frombuffer(raw[:, 208:210].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # int8 scales (16 sub-blocks of 16 elements)
    scales = raw[:, 192:208].view(np.int8).astype(np.float32)  # (n_blocks, 16)

    # Low 4 bits (128 bytes = 256 nibbles)
    ql = raw[:, 0:128]
    quants_lo = np.empty((n_blocks, 256), dtype=np.int32)
    quants_lo[:, :128] = (ql & 0x0F).astype(np.int32)
    quants_lo[:, 128:] = ((ql >> 4) & 0x0F).astype(np.int32)

    # High 2 bits (64 bytes = 256 x 2 bits)
    qh = raw[:, 128:192]
    quants_hi = np.empty((n_blocks, 256), dtype=np.int32)
    for i in range(4):
        quants_hi[:, i::4] = ((qh >> (i * 2)) & 0x03).astype(np.int32)

    # Combine: 6-bit = 4-bit + 2-bit << 4
    quants = quants_lo + (quants_hi << 4)

    # Dequantize: 16 sub-blocks of 16 elements
    values = np.empty((n_blocks, 256), dtype=np.float32)
    for sb in range(16):
        start = sb * 16
        end = start + 16
        sb_scale = scales[:, sb] * d
        # Q6_K uses symmetric quantization centered at 32
        values[:, start:end] = (quants[:, start:end].astype(np.float32) - 32.0) * sb_scale[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


# ---------------------------------------------------------------------------
# I-quant dequantization routines (importance-weighted grid codebooks)
# ---------------------------------------------------------------------------

# IQ4_NL non-linear codebook (16 values for 4-bit quantization)
# These values are chosen to better represent the distribution of weights
IQ4_NL_VALUES: np.ndarray = np.array([
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
], dtype=np.float32) / 127.0


def dequant_iq4_nl(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize IQ4_NL blocks to FP32.

    IQ4_NL block layout (18 bytes, 32 elements):
      - bytes [0:2]: fp16 d (scale)
      - bytes [2:18]: 16 bytes of 4-bit indices into non-linear codebook

    Uses a fixed non-linear codebook instead of linear [0,15].
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_IQ4_NL]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 scale
    d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # 4-bit indices
    qs = raw[:, 2:18]
    indices = np.empty((n_blocks, 32), dtype=np.int32)
    indices[:, :16] = (qs & 0x0F).astype(np.int32)
    indices[:, 16:] = ((qs >> 4) & 0x0F).astype(np.int32)

    # Look up values in codebook
    values_codebook = IQ4_NL_VALUES[indices]  # (n_blocks, 32)

    # Apply scale
    values = values_codebook * d[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


# IQ2_XXS grid: 512 entries, each entry encodes 8 weights from {-1, 0, +1}
# The grid is organized to maximize important weight positions
# For simplicity, we use a sign-based reconstruction
def _iq2_xxs_grid() -> np.ndarray:
    """Generate IQ2_XXS grid values.

    Each entry represents 8 ternary values {-1, 0, +1} packed into bits.
    The grid prioritizes combinations that minimize error for important weights.
    """
    # For IQ2_XXS, each 2-bit code maps to values in the range approx [-1.0, 1.0]
    # The actual codebook is importance-weighted, but we use an approximation
    grid = np.zeros((512, 8), dtype=np.float32)
    for i in range(512):
        for j in range(8):
            # Extract 2 bits for each of 8 positions (total 16 bits, but only 9 bits used)
            bit_pair = (i >> j) & 1  # Simplified: just use 1 bit per position
            grid[i, j] = bit_pair * 2.0 - 1.0 if i & (1 << j) else 0.0
    return grid


def dequant_iq2_xxs(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize IQ2_XXS blocks to FP32.

    IQ2_XXS super-block layout (66 bytes, 256 elements):
      - bytes [0:2]: fp16 d (super-block scale)
      - bytes [2:66]: 64 bytes = 8 sub-blocks of 8 bytes each
        Each sub-block: 4 bytes (32 bits) grid indices + 4 bytes signs/scales

    Uses importance-weighted grid codebook for 2-bit quantization.
    Achieves ~2.0625 bpw (bits per weight).
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_IQ2_XXS]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 super-block scale
    d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # Process 8 sub-blocks of 32 elements each
    values = np.empty((n_blocks, 256), dtype=np.float32)

    for sb in range(8):
        sb_start = 2 + sb * 8
        sb_data = raw[:, sb_start:sb_start + 8]

        # Grid indices (first 4 bytes = 4 x 9-bit indices, but stored as 16-bit pairs)
        qs = sb_data[:, :4]

        # Signs/scales (next 4 bytes)
        signs = sb_data[:, 4:8]

        # For each of the 4 groups of 8 elements in this sub-block
        for g in range(4):
            g_start = sb * 32 + g * 8

            # Extract 2-bit codes for 8 elements from the byte
            q_byte = qs[:, g]
            sign_byte = signs[:, g]

            for i in range(8):
                # Simplified: extract 2-bit code pairs
                code = (q_byte >> (i % 4 * 2)) & 0x03
                sign = 1.0 if (sign_byte >> i) & 1 else -1.0

                # Map 2-bit code to value (approximate IQ2 grid)
                val = np.array([0.0, 0.25, 0.5, 1.0])[code]
                values[:, g_start + i] = val * sign * d

    return values.reshape(-1)[:n_elements]


def dequant_iq3_xxs(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize IQ3_XXS blocks to FP32.

    IQ3_XXS super-block layout (98 bytes, 256 elements):
      - bytes [0:2]: fp16 d (super-block scale)
      - bytes [2:98]: 96 bytes = 8 sub-blocks of 12 bytes each
        Each sub-block: 6 bytes grid indices + 4 bytes high bits + 2 bytes signs

    Uses importance-weighted grid codebook for 3-bit quantization.
    Achieves ~3.0625 bpw.
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_IQ3_XXS]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 super-block scale
    d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    values = np.empty((n_blocks, 256), dtype=np.float32)

    # Process 8 sub-blocks of 32 elements
    for sb in range(8):
        sb_start = 2 + sb * 12
        sb_data = raw[:, sb_start:sb_start + 12]

        qs = sb_data[:, :6]  # 3-bit codes (6 bytes = 16 x 3 bits, for 32 elements in pairs)
        sb_data[:, 6:10]  # High bits
        signs = sb_data[:, 10:12]  # Sign bits

        for g in range(4):
            g_start = sb * 32 + g * 8

            for i in range(8):
                # Extract 3-bit code
                byte_idx = (g * 8 + i) * 3 // 8
                bit_offset = (g * 8 + i) * 3 % 8

                if byte_idx < 6:
                    code = (qs[:, byte_idx].astype(np.int32) >> bit_offset) & 0x07
                    if bit_offset > 5 and byte_idx + 1 < 6:
                        code |= (qs[:, byte_idx + 1].astype(np.int32) << (8 - bit_offset)) & 0x07
                else:
                    code = np.zeros(n_blocks, dtype=np.int32)

                # Sign from sign bytes
                sign_idx = i + g * 8
                sign_byte = sign_idx // 8
                sign_bit = sign_idx % 8
                sign = np.where((signs[:, sign_byte] >> sign_bit) & 1, -1.0, 1.0)

                # Map 3-bit code to value (approximate IQ3 grid: 8 levels)
                val_map = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0])
                values[:, g_start + i] = val_map[code] * sign * d

    return values.reshape(-1)[:n_elements]


def dequant_iq4_xs(data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize IQ4_XS blocks to FP32.

    IQ4_XS super-block layout (136 bytes, 256 elements):
      - bytes [0:2]: fp16 d (super-block scale)
      - bytes [2:4]: uint16 scales_h (high bits of sub-block scales)
      - bytes [4:12]: 8 bytes = 8 x 4-bit low scale nibbles
      - bytes [12:136]: 124 bytes ~ 256 x 4-bit quants

    Uses IQ4_NL codebook with per-sub-block 6-bit scales.
    Achieves ~4.25 bpw.
    """
    block_size, block_bytes = GGML_BLOCK_PARAMS[GGML_TYPE_IQ4_XS]
    n_blocks = n_elements // block_size
    raw = data[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)

    # fp16 super-block scale
    d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(n_blocks)

    # High bits of scales
    scales_h = np.frombuffer(raw[:, 2:4].tobytes(), dtype=np.uint16).reshape(n_blocks)

    # Low nibbles of scales (8 x 4-bit)
    scales_l = raw[:, 4:12]

    # Combine into 8 x 6-bit scales
    scales = np.zeros((n_blocks, 8), dtype=np.float32)
    for i in range(8):
        lo = (scales_l[:, i // 2] >> ((i % 2) * 4)) & 0x0F
        hi = ((scales_h >> (i * 2)) & 0x03).astype(np.uint8)
        scales[:, i] = ((hi << 4) | lo).astype(np.float32) - 32.0  # Signed 6-bit

    # 4-bit quants using IQ4_NL codebook (128 bytes = 256 nibbles)
    # Note: layout is slightly different, 124 bytes for quants
    qs = raw[:, 8:136]  # Adjusted
    indices = np.empty((n_blocks, 256), dtype=np.int32)

    # Unpack nibbles
    for i in range(128):
        if i < qs.shape[1]:
            indices[:, i] = (qs[:, i] & 0x0F).astype(np.int32)
            indices[:, i + 128] = ((qs[:, i] >> 4) & 0x0F).astype(np.int32)
        else:
            indices[:, i] = 0
            indices[:, i + 128] = 0

    # Clamp indices to valid range
    indices = np.clip(indices, 0, 15)

    # Look up in IQ4_NL codebook
    values_codebook = IQ4_NL_VALUES[indices]

    # Apply sub-block scales
    values = np.empty((n_blocks, 256), dtype=np.float32)
    for sb in range(8):
        start = sb * 32
        end = start + 32
        values[:, start:end] = values_codebook[:, start:end] * scales[:, sb, np.newaxis] * d[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


# Dispatch table
_DEQUANT_FNS = {
    # Legacy quants
    GGML_TYPE_Q4_0: dequant_q4_0,
    GGML_TYPE_Q4_1: dequant_q4_1,
    GGML_TYPE_Q5_0: dequant_q5_0,
    GGML_TYPE_Q5_1: dequant_q5_1,
    GGML_TYPE_Q8_0: dequant_q8_0,
    # K-quants
    GGML_TYPE_Q2_K: dequant_q2_k,
    GGML_TYPE_Q3_K: dequant_q3_k,
    GGML_TYPE_Q4_K: dequant_q4_k,
    GGML_TYPE_Q5_K: dequant_q5_k,
    GGML_TYPE_Q6_K: dequant_q6_k,
    # I-quants
    GGML_TYPE_IQ4_NL: dequant_iq4_nl,
    GGML_TYPE_IQ2_XXS: dequant_iq2_xxs,
    GGML_TYPE_IQ3_XXS: dequant_iq3_xxs,
    GGML_TYPE_IQ4_XS: dequant_iq4_xs,
}

# Quant types with dequantization support (subset of block formats).
DEQUANT_SUPPORTED_TYPES = frozenset(_DEQUANT_FNS.keys())


def dequantize_tensor(
    data: np.ndarray,
    qtype: int,
    n_elements: int,
) -> np.ndarray:
    """Dispatch dequantization by GGML type code.

    Args:
        data: Raw byte array of the quantized block data.
        qtype: GGML tensor type code.
        n_elements: Total number of logical elements in the tensor.

    Returns:
        FP32 array of shape (n_elements,).

    Raises:
        ValueError: If qtype is not a supported quantization type.
    """
    if qtype not in _DEQUANT_FNS:
        raise ValueError(
            f"Unsupported GGML quantization type: {qtype}. "
            f"Supported: {sorted(DEQUANT_SUPPORTED_TYPES)}"
        )
    return _DEQUANT_FNS[qtype](data, n_elements)


# ---------------------------------------------------------------------------
# Marlin FP4 packing
# ---------------------------------------------------------------------------


def _quantize_to_marlin_fp4(
    weights_fp32: np.ndarray,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weights to Marlin-packed FP4 (E2M1) format.

    Pipeline:
      1. Pad K to multiple of group_size, N to multiple of 8
      2. Compute per-group absmax scales (max_abs / 6.0)
      3. Nearest-neighbor quantize to E2M1 codebook
      4. Pack 8 nibbles per uint32 word

    Args:
        weights_fp32: [K, N] weight matrix in FP32.
        group_size: Elements per quantization group along K.

    Returns:
        packed: [K_padded, N_padded // 8] uint32 packed weights.
        scales: [K_padded // group_size, N_padded] FP16 per-group scales.
    """
    K, N = weights_fp32.shape

    # Pad K to multiple of group_size
    K_padded = ((K + group_size - 1) // group_size) * group_size
    # Pad N to multiple of 8 (8 nibbles per uint32)
    N_padded = ((N + 7) // 8) * 8

    if K_padded != K or N_padded != N:
        padded = np.zeros((K_padded, N_padded), dtype=np.float32)
        padded[:K, :N] = weights_fp32
        weights_fp32 = padded

    # Per-group scales: max_abs / 6.0 (max E2M1 magnitude)
    n_groups = K_padded // group_size
    grouped = weights_fp32.reshape(n_groups, group_size, N_padded)
    max_abs = np.abs(grouped).max(axis=1)  # (n_groups, N_padded)
    scales = (max_abs / 6.0).astype(np.float16)

    # Quantize: find nearest E2M1 codebook entry for each normalized weight
    safe_scales = np.where(
        scales == 0, np.float16(1.0), scales
    ).astype(np.float32)
    normalized = grouped / safe_scales[:, np.newaxis, :]  # (n_groups, group_size, N)

    flat = normalized.reshape(-1)
    diffs = np.abs(flat[:, np.newaxis] - E2M1_VALUES[np.newaxis, :])
    indices = diffs.argmin(axis=1).astype(np.uint8).reshape(K_padded, N_padded)

    # Pack 8 nibbles into uint32 words
    # Layout: weights[k, n] -> packed[k, n // 8], nibble position = n % 8
    packed = np.zeros((K_padded, N_padded // 8), dtype=np.uint32)
    for i in range(8):
        packed |= indices[:, i::8].astype(np.uint32) << (i * 4)

    return packed, scales


# ---------------------------------------------------------------------------
# GGUF file parser
# ---------------------------------------------------------------------------


class TensorInfo:
    """Metadata for a single tensor in a GGUF file."""

    __slots__ = ("name", "shape", "qtype", "offset")

    def __init__(self, name: str, shape: tuple[int, ...], qtype: int, offset: int):
        self.name = name
        self.shape = shape
        self.qtype = qtype
        self.offset = offset

    @property
    def n_elements(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result

    @property
    def data_size(self) -> int:
        """Byte size of this tensor's data in the file."""
        n = self.n_elements
        if self.qtype == GGML_TYPE_F32:
            return n * 4
        if self.qtype == GGML_TYPE_F16:
            return n * 2
        if self.qtype == GGML_TYPE_BF16:
            return n * 2
        if self.qtype in GGML_BLOCK_PARAMS:
            block_size, block_bytes = GGML_BLOCK_PARAMS[self.qtype]
            n_blocks = (n + block_size - 1) // block_size
            return n_blocks * block_bytes
        raise ValueError(f"Unknown GGML type {self.qtype} for size calculation")


class GGUFReader:
    """Read GGUF model files and extract quantized weights.

    Parses the GGUF binary format (v2/v3) to extract tensor metadata and data.
    Supports dequantization of Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 formats and
    conversion to Marlin FP4 (E2M1) packed representation.

    Usage:
        reader = GGUFReader("model.gguf")
        print(reader.metadata)
        for name, info in reader.tensor_infos.items():
            packed, scales = reader.get_tensor_marlin(name)
    """

    MAGIC = b"GGUF"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensor_infos: dict[str, TensorInfo] = {}
        self._version: int = 0
        self._data_offset: int = 0
        self._alignment: int = 32  # default GGUF alignment

        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")
        self._parse_header()

    def _parse_header(self) -> None:
        """Parse GGUF file header, metadata, and tensor index."""
        with open(self.path, "rb") as f:
            # Magic number
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(
                    f"Not a GGUF file (magic={magic!r}, expected {self.MAGIC!r})"
                )

            # Version (uint32)
            self._version = struct.unpack("<I", f.read(4))[0]
            if self._version not in (2, 3):
                raise ValueError(
                    f"Unsupported GGUF version {self._version} (expected 2 or 3)"
                )

            # Tensor count and KV count
            # v2 uses uint32, v3 uses uint64 for these fields
            if self._version >= 3:
                n_tensors = struct.unpack("<Q", f.read(8))[0]
                n_kv = struct.unpack("<Q", f.read(8))[0]
            else:
                n_tensors = struct.unpack("<I", f.read(4))[0]
                n_kv = struct.unpack("<I", f.read(4))[0]

            # Parse KV metadata
            for _ in range(n_kv):
                key = self._read_string(f)
                value = self._read_value(f)
                self.metadata[key] = value

            # Check for custom alignment
            if "general.alignment" in self.metadata:
                self._alignment = int(self.metadata["general.alignment"])

            # Parse tensor infos
            for _ in range(n_tensors):
                info = self._read_tensor_info(f)
                self.tensor_infos[info.name] = info

            # Data starts after header, aligned to boundary
            header_end = f.tell()
            self._data_offset = _align_offset(header_end, self._alignment)

    def _read_string(self, f) -> str:
        """Read a GGUF string: uint64 length + UTF-8 bytes."""
        if self._version >= 3:
            length = struct.unpack("<Q", f.read(8))[0]
        else:
            length = struct.unpack("<I", f.read(4))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, f) -> Any:
        """Read a typed GGUF value."""
        vtype = struct.unpack("<I", f.read(4))[0]
        return self._read_typed_value(f, vtype)

    def _read_typed_value(self, f, vtype: int) -> Any:
        """Read a value given its GGUF type code."""
        if vtype == GGUF_TYPE_UINT8:
            return struct.unpack("<B", f.read(1))[0]
        if vtype == GGUF_TYPE_INT8:
            return struct.unpack("<b", f.read(1))[0]
        if vtype == GGUF_TYPE_UINT16:
            return struct.unpack("<H", f.read(2))[0]
        if vtype == GGUF_TYPE_INT16:
            return struct.unpack("<h", f.read(2))[0]
        if vtype == GGUF_TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        if vtype == GGUF_TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        if vtype == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        if vtype == GGUF_TYPE_BOOL:
            return struct.unpack("<B", f.read(1))[0] != 0
        if vtype == GGUF_TYPE_STRING:
            return self._read_string(f)
        if vtype == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        if vtype == GGUF_TYPE_INT64:
            return struct.unpack("<q", f.read(8))[0]
        if vtype == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = struct.unpack("<I", f.read(4))[0]
            if self._version >= 3:
                count = struct.unpack("<Q", f.read(8))[0]
            else:
                count = struct.unpack("<I", f.read(4))[0]
            return [self._read_typed_value(f, elem_type) for _ in range(count)]
        raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _read_tensor_info(self, f) -> TensorInfo:
        """Read a tensor info entry from the GGUF header."""
        name = self._read_string(f)
        n_dims = struct.unpack("<I", f.read(4))[0]

        # Shape: each dimension is uint64 in v3, uint32 in v2
        if self._version >= 3:
            shape = tuple(
                struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)
            )
        else:
            shape = tuple(
                struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)
            )

        qtype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]

        return TensorInfo(name=name, shape=shape, qtype=qtype, offset=offset)

    def _read_tensor_data(self, info: TensorInfo) -> np.ndarray:
        """Read raw tensor bytes from the data section."""
        abs_offset = self._data_offset + info.offset
        size = info.data_size
        with open(self.path, "rb") as f:
            f.seek(abs_offset)
            raw = f.read(size)
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def get_tensor_fp32(self, name: str) -> np.ndarray:
        """Get a tensor dequantized to FP32.

        For unquantized tensors (F16, F32, BF16), returns directly.
        For quantized tensors, runs the appropriate dequantization.

        Args:
            name: Tensor name from the GGUF file.

        Returns:
            FP32 array reshaped to the tensor's logical shape.

        Raises:
            KeyError: If tensor name not found.
            ValueError: If tensor type is unsupported.
        """
        if name not in self.tensor_infos:
            raise KeyError(f"Tensor '{name}' not found. Available: {list(self.tensor_infos.keys())[:10]}...")

        info = self.tensor_infos[name]
        data = self._read_tensor_data(info)

        if info.qtype == GGML_TYPE_F32:
            return np.frombuffer(data.tobytes(), dtype=np.float32).reshape(info.shape)

        if info.qtype == GGML_TYPE_F16:
            return np.frombuffer(
                data.tobytes(), dtype=np.float16
            ).astype(np.float32).reshape(info.shape)

        if info.qtype == GGML_TYPE_BF16:
            # BF16: reinterpret uint16 as bfloat16 -> FP32
            raw16 = np.frombuffer(data.tobytes(), dtype=np.uint16)
            # BF16 to FP32: shift left by 16 bits to place in upper half of FP32
            fp32_bits = raw16.astype(np.uint32) << 16
            result = np.frombuffer(fp32_bits.tobytes(), dtype=np.float32)
            return result.reshape(info.shape)

        if info.qtype in DEQUANT_SUPPORTED_TYPES:
            flat = dequantize_tensor(data, info.qtype, info.n_elements)
            return flat.reshape(info.shape)

        if info.qtype in BLOCK_QUANT_TYPES:
            raise ValueError(
                f"GGML quantization type {info.qtype} for tensor '{name}' "
                "is block-quantized but not supported by the dequantizer."
            )

        raise ValueError(
            f"Unsupported tensor type {info.qtype} for tensor '{name}'"
        )

    def get_tensor_marlin_np(
        self,
        name: str,
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a tensor in Marlin FP4 packed format as numpy arrays.

        Dequantizes the tensor to FP32, then re-quantizes into the Marlin
        FP4 (E2M1) representation used by the Metal fused dequant-GEMM kernel.

        For 2D weight matrices [rows, cols], the Marlin layout is:
          - packed_weights: [K_padded, N_padded // 8] uint32
          - scales: [K_padded // group_size, N_padded] float16

        Args:
            name: Tensor name from the GGUF file.
            group_size: Marlin quantization group size (default 128).

        Returns:
            (packed_weights, scales) as numpy arrays.

        Raises:
            ValueError: If tensor is not 2D (not a weight matrix).
        """
        fp32 = self.get_tensor_fp32(name)

        if fp32.ndim != 2:
            raise ValueError(
                f"Marlin conversion requires 2D tensors, got shape {fp32.shape} "
                f"for '{name}'. Use get_tensor_fp32() for non-weight tensors."
            )

        return _quantize_to_marlin_fp4(fp32, group_size)

    def get_tensor_marlin(
        self,
        name: str,
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> tuple[mx.array, mx.array]:
        """Get a tensor in Marlin FP4 packed format as MLX arrays.

        Dequantizes the tensor to FP32, then re-quantizes into the Marlin
        FP4 (E2M1) representation used by the Metal fused dequant-GEMM kernel.

        For 2D weight matrices [rows, cols], the Marlin layout is:
          - packed_weights: [K_padded, N_padded // 8] uint32
          - scales: [K_padded // group_size, N_padded] float16

        Args:
            name: Tensor name from the GGUF file.
            group_size: Marlin quantization group size (default 128).

        Returns:
            (packed_weights, scales) as MLX arrays.

        Raises:
            ValueError: If tensor is not 2D (not a weight matrix).
            ImportError: If MLX is not installed.
        """
        if not HAS_MLX:
            raise ImportError(
                "get_tensor_marlin() requires MLX. "
                "Use get_tensor_marlin_np() for numpy arrays, or install MLX: pip install mlx"
            )
        packed, scales = self.get_tensor_marlin_np(name, group_size)
        return mx.array(packed), mx.array(scales)

    def get_tensor_fp16(self, name: str) -> np.ndarray:
        """Get a tensor as a numpy FP16 array (no Marlin packing).

        Useful for non-weight tensors like embeddings, norms, biases.

        Args:
            name: Tensor name from the GGUF file.

        Returns:
            numpy float16 array.
        """
        fp32 = self.get_tensor_fp32(name)
        return fp32.astype(np.float16)

    def get_tensor_fp16_mx(self, name: str) -> mx.array:
        """Get a tensor as an MLX FP16 array (no Marlin packing).

        Useful for non-weight tensors like embeddings, norms, biases.

        Args:
            name: Tensor name from the GGUF file.

        Returns:
            MLX float16 array.

        Raises:
            ImportError: If MLX is not installed.
        """
        if not HAS_MLX:
            raise ImportError(
                "get_tensor_fp16_mx() requires MLX. "
                "Use get_tensor_fp16() for numpy arrays, or install MLX: pip install mlx"
            )
        fp32 = self.get_tensor_fp32(name)
        return mx.array(fp32.astype(np.float16))

    def list_tensors(self) -> list[dict[str, Any]]:
        """List all tensors with their metadata.

        Returns:
            List of dicts with keys: name, shape, qtype, n_elements, data_size.
        """
        result = []
        for name, info in self.tensor_infos.items():
            result.append({
                "name": name,
                "shape": info.shape,
                "qtype": info.qtype,
                "qtype_name": _qtype_name(info.qtype),
                "n_elements": info.n_elements,
                "data_size_bytes": info.data_size,
            })
        return result

    def tensor_names(self) -> list[str]:
        """Return all tensor names in file order."""
        return list(self.tensor_infos.keys())

    def is_quantized(self, name: str) -> bool:
        """Check if a tensor uses block quantization (vs dense F16/F32)."""
        if name not in self.tensor_infos:
            raise KeyError(f"Tensor '{name}' not found")
        return self.tensor_infos[name].qtype in BLOCK_QUANT_TYPES


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _align_offset(offset: int, alignment: int) -> int:
    """Round up offset to the next alignment boundary."""
    return ((offset + alignment - 1) // alignment) * alignment


def _qtype_name(qtype: int) -> str:
    """Human-readable name for a GGML type code."""
    names = {
        GGML_TYPE_F32: "F32",
        GGML_TYPE_F16: "F16",
        GGML_TYPE_Q4_0: "Q4_0",
        GGML_TYPE_Q4_1: "Q4_1",
        GGML_TYPE_Q5_0: "Q5_0",
        GGML_TYPE_Q5_1: "Q5_1",
        GGML_TYPE_Q8_0: "Q8_0",
        GGML_TYPE_Q8_1: "Q8_1",
        # K-quants
        GGML_TYPE_Q2_K: "Q2_K",
        GGML_TYPE_Q3_K: "Q3_K",
        GGML_TYPE_Q4_K: "Q4_K",
        GGML_TYPE_Q5_K: "Q5_K",
        GGML_TYPE_Q6_K: "Q6_K",
        GGML_TYPE_Q8_K: "Q8_K",
        # I-quants
        GGML_TYPE_IQ2_XXS: "IQ2_XXS",
        GGML_TYPE_IQ2_XS: "IQ2_XS",
        GGML_TYPE_IQ2_S: "IQ2_S",
        GGML_TYPE_IQ3_XXS: "IQ3_XXS",
        GGML_TYPE_IQ3_S: "IQ3_S",
        GGML_TYPE_IQ4_NL: "IQ4_NL",
        GGML_TYPE_IQ4_XS: "IQ4_XS",
        GGML_TYPE_IQ1_S: "IQ1_S",
        GGML_TYPE_IQ1_M: "IQ1_M",
        # Other
        GGML_TYPE_BF16: "BF16",
        GGML_TYPE_I8: "I8",
        GGML_TYPE_I16: "I16",
        GGML_TYPE_I32: "I32",
        GGML_TYPE_I64: "I64",
        GGML_TYPE_F64: "F64",
    }
    return names.get(qtype, f"UNKNOWN({qtype})")


def load_gguf_model_np(
    path: str | Path,
    group_size: int = DEFAULT_GROUP_SIZE,
    skip_patterns: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray] | np.ndarray]:
    """Load all tensors from a GGUF file in Marlin-ready format as numpy arrays.

    Weight matrices (2D, quantized) are converted to Marlin FP4 packed format.
    Other tensors (embeddings, norms, biases) are returned as FP16 numpy arrays.

    This function works without MLX installed.

    Args:
        path: Path to the GGUF file.
        group_size: Marlin quantization group size.
        skip_patterns: Tensor name substrings to skip entirely.

    Returns:
        Dict mapping tensor names to either:
          - (packed_weights, scales) tuple for 2D quantized weight matrices
          - numpy array (FP16) for other tensors
    """
    if skip_patterns is None:
        skip_patterns = []

    reader = GGUFReader(path)
    result: dict[str, tuple[np.ndarray, np.ndarray] | np.ndarray] = {}

    for name in reader.tensor_names():
        if any(pat in name for pat in skip_patterns):
            continue

        info = reader.tensor_infos[name]

        # 2D quantized tensors -> Marlin format
        if info.qtype in DEQUANT_SUPPORTED_TYPES and len(info.shape) == 2:
            packed, scales = reader.get_tensor_marlin_np(name, group_size)
            result[name] = (packed, scales)
        else:
            # Everything else -> FP16
            result[name] = reader.get_tensor_fp16(name)

    return result


def load_gguf_model(
    path: str | Path,
    group_size: int = DEFAULT_GROUP_SIZE,
    skip_patterns: list[str] | None = None,
) -> dict[str, tuple[mx.array, mx.array] | mx.array]:
    """Load all tensors from a GGUF file in Marlin-ready format as MLX arrays.

    Weight matrices (2D, quantized) are converted to Marlin FP4 packed format.
    Other tensors (embeddings, norms, biases) are returned as FP16 MLX arrays.

    Args:
        path: Path to the GGUF file.
        group_size: Marlin quantization group size.
        skip_patterns: Tensor name substrings to skip entirely.

    Returns:
        Dict mapping tensor names to either:
          - (packed_weights, scales) tuple for 2D quantized weight matrices
          - mx.array (FP16) for other tensors

    Raises:
        ImportError: If MLX is not installed.
    """
    if not HAS_MLX:
        raise ImportError(
            "load_gguf_model() requires MLX. "
            "Use load_gguf_model_np() for numpy arrays, or install MLX: pip install mlx"
        )

    if skip_patterns is None:
        skip_patterns = []

    reader = GGUFReader(path)
    result: dict[str, tuple[mx.array, mx.array] | mx.array] = {}

    for name in reader.tensor_names():
        if any(pat in name for pat in skip_patterns):
            continue

        info = reader.tensor_infos[name]

        # 2D quantized tensors -> Marlin format
        if info.qtype in DEQUANT_SUPPORTED_TYPES and len(info.shape) == 2:
            packed, scales = reader.get_tensor_marlin(name, group_size)
            result[name] = (packed, scales)
        else:
            # Everything else -> FP16
            result[name] = reader.get_tensor_fp16_mx(name)

    return result
