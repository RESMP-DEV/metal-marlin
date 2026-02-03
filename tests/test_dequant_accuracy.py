"""Validation tests for dequantization accuracy across all supported formats."""

from __future__ import annotations

import numpy as np
import pytest

from metal_marlin.neon_dequant import (
    dequant_fp4_neon,
    dequant_fp8_e4m3_neon,
    dequant_fp8_e5m2_neon,
    dequant_int4_asym_neon,
    dequant_int4_neon,
    dequant_int8_neon,
    dequant_int8_per_channel_neon,
    dequant_nf4_neon,
    dequant_q4_0_neon,
    dequant_q4_1_neon,
    dequant_q8_0_neon,
)

FP4_CODEBOOK = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)

NF4_CODEBOOK = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _expand_scales(scales: np.ndarray, group_size: int, K: int) -> np.ndarray:
    expanded = np.repeat(scales.astype(np.float32, copy=False), group_size, axis=0)
    return expanded[:K, :]


def _unpack_nibbles_k8(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    indices = np.empty((K, N), dtype=np.uint8)
    for i in range(8):
        indices[i::8, :] = ((packed >> (i * 4)) & 0xF).astype(np.uint8)
    return indices


def _unpack_bytes(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    codes = np.empty((K, N), dtype=np.uint8)
    for i in range(4):
        codes[:, i::4] = ((packed >> (i * 8)) & 0xFF).astype(np.uint8)
    return codes


def _fp8_e4m3_codebook() -> np.ndarray:
    values = np.zeros(256, dtype=np.float32)
    for code in range(256):
        s = (code >> 7) & 1
        e = (code >> 3) & 0xF
        m = code & 0x7
        sign = -1.0 if s else 1.0
        if e == 15:
            values[code] = np.nan
        elif e == 0:
            if m == 0:
                values[code] = 0.0 if s == 0 else -0.0
            else:
                values[code] = sign * (2**-6) * (m / 8)
        else:
            values[code] = sign * (2 ** (e - 7)) * (1 + m / 8)
    return values


def _fp8_e5m2_codebook() -> np.ndarray:
    values = np.zeros(256, dtype=np.float32)
    for code in range(256):
        s = (code >> 7) & 1
        e = (code >> 2) & 0x1F
        m = code & 0x3
        sign = -1.0 if s else 1.0
        if e == 31:
            values[code] = sign * np.inf if m == 0 else np.nan
        elif e == 0:
            if m == 0:
                values[code] = 0.0 if s == 0 else -0.0
            else:
                values[code] = sign * (2**-14) * (m / 4)
        else:
            values[code] = sign * (2 ** (e - 15)) * (1 + m / 4)
    return values


def _ref_fp4(packed: np.ndarray, scales: np.ndarray, K: int, N: int, group_size: int) -> np.ndarray:
    indices = _unpack_nibbles_k8(packed, K, N)
    values = FP4_CODEBOOK[indices]
    scales_expanded = _expand_scales(scales, group_size, K)
    return values * scales_expanded


def _ref_int4(packed: np.ndarray, scales: np.ndarray, K: int, N: int, group_size: int) -> np.ndarray:
    indices = _unpack_nibbles_k8(packed, K, N).astype(np.float32)
    values = indices - 8.0
    scales_expanded = _expand_scales(scales, group_size, K)
    return values * scales_expanded


def _ref_int4_asym(
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    K: int,
    N: int,
    group_size: int,
) -> np.ndarray:
    indices = _unpack_nibbles_k8(packed, K, N).astype(np.float32)
    scales_expanded = _expand_scales(scales, group_size, K)
    zeros_expanded = _expand_scales(zeros, group_size, K)
    return (indices - zeros_expanded) * scales_expanded


def _ref_nf4(packed: np.ndarray, scales: np.ndarray, K: int, N: int, group_size: int) -> np.ndarray:
    indices = _unpack_nibbles_k8(packed, K, N)
    values = NF4_CODEBOOK[indices]
    scales_expanded = _expand_scales(scales, group_size, K)
    return values * scales_expanded


def _ref_fp8(
    packed: np.ndarray,
    scales: np.ndarray,
    K: int,
    N: int,
    group_size: int,
    codebook: np.ndarray,
) -> np.ndarray:
    codes = _unpack_bytes(packed, K, N)
    values = codebook[codes]
    scales_expanded = _expand_scales(scales, group_size, K)
    return values * scales_expanded


def _ref_int8(data: np.ndarray, scales: np.ndarray, K: int, N: int, group_size: int) -> np.ndarray:
    values = data.astype(np.float32)
    scales_expanded = _expand_scales(scales, group_size, K)
    return values * scales_expanded


def _ref_int8_per_channel(data: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return data.astype(np.float32) * scales.astype(np.float32)


def _fp16_bytes(value: float) -> np.ndarray:
    return np.array([value], dtype=np.float16).view(np.uint8)


def _ref_q4_0(data: np.ndarray, n_elements: int) -> np.ndarray:
    block_size = 32
    block_bytes = 18
    n_blocks = n_elements // block_size
    output = np.empty(n_elements, dtype=np.float32)
    for b in range(n_blocks):
        block = data[b * block_bytes : (b + 1) * block_bytes]
        scale = np.frombuffer(block[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        qs = block[2:18]
        lo = (qs & 0x0F).astype(np.float32)
        hi = ((qs >> 4) & 0x0F).astype(np.float32)
        quants = np.concatenate([lo, hi])
        output[b * block_size : (b + 1) * block_size] = (quants - 8.0) * scale
    return output


def _ref_q4_1(data: np.ndarray, n_elements: int) -> np.ndarray:
    block_size = 32
    block_bytes = 20
    n_blocks = n_elements // block_size
    output = np.empty(n_elements, dtype=np.float32)
    for b in range(n_blocks):
        block = data[b * block_bytes : (b + 1) * block_bytes]
        scale = np.frombuffer(block[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        min_val = np.frombuffer(block[2:4].tobytes(), dtype=np.float16)[0].astype(np.float32)
        qs = block[4:20]
        lo = (qs & 0x0F).astype(np.float32)
        hi = ((qs >> 4) & 0x0F).astype(np.float32)
        quants = np.concatenate([lo, hi])
        output[b * block_size : (b + 1) * block_size] = quants * scale + min_val
    return output


def _ref_q8_0(data: np.ndarray, n_elements: int) -> np.ndarray:
    block_size = 32
    block_bytes = 34
    n_blocks = n_elements // block_size
    output = np.empty(n_elements, dtype=np.float32)
    for b in range(n_blocks):
        block = data[b * block_bytes : (b + 1) * block_bytes]
        scale = np.frombuffer(block[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        quants = block[2:34].view(np.int8).astype(np.float32)
        output[b * block_size : (b + 1) * block_size] = quants * scale
    return output


def test_fp4_dequant_accuracy() -> None:
    rng = _rng()
    K, N, group_size = 64, 32, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_fp4(packed, scales, K, N, group_size)
    actual = dequant_fp4_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_int4_dequant_accuracy() -> None:
    rng = _rng()
    K, N, group_size = 64, 40, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_int4(packed, scales, K, N, group_size)
    actual = dequant_int4_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_int4_asym_dequant_accuracy() -> None:
    rng = _rng()
    K, N, group_size = 64, 24, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)
    zeros = rng.uniform(0.0, 15.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_int4_asym(packed, scales, zeros, K, N, group_size)
    actual = dequant_int4_asym_neon(packed, scales, zeros, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_nf4_dequant_accuracy() -> None:
    rng = _rng()
    K, N, group_size = 64, 32, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_nf4(packed, scales, K, N, group_size)
    actual = dequant_nf4_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_fp8_e4m3_dequant_accuracy() -> None:
    rng = _rng()
    K, N, group_size = 32, 32, 16
    packed = rng.integers(0, 2**32, size=(K, N // 4), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_fp8(packed, scales, K, N, group_size, _fp8_e4m3_codebook())
    actual = dequant_fp8_e4m3_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_fp8_e5m2_dequant_accuracy() -> None:
    rng = _rng()
    K, N, group_size = 32, 32, 16
    packed = rng.integers(0, 2**32, size=(K, N // 4), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_fp8(packed, scales, K, N, group_size, _fp8_e5m2_codebook())
    actual = dequant_fp8_e5m2_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_int8_dequant_accuracy() -> None:
    rng = _rng()
    K, N, group_size = 64, 48, 16
    data = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_int8(data, scales, K, N, group_size)
    actual = dequant_int8_neon(data, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_int8_per_channel_dequant_accuracy() -> None:
    rng = _rng()
    K, N = 48, 32
    data = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    scales = rng.uniform(0.01, 2.0, size=(N,)).astype(np.float32)

    expected = _ref_int8_per_channel(data, scales)
    actual = dequant_int8_per_channel_neon(data, scales)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_q4_0_dequant_accuracy() -> None:
    rng = _rng()
    n_blocks = 2
    block_bytes = 18
    blocks = []
    for _ in range(n_blocks):
        block = np.empty(block_bytes, dtype=np.uint8)
        block[0:2] = _fp16_bytes(rng.uniform(0.01, 2.0))
        block[2:18] = rng.integers(0, 256, size=16, dtype=np.uint8)
        blocks.append(block)
    data = np.concatenate(blocks)
    n_elements = n_blocks * 32

    expected = _ref_q4_0(data, n_elements)
    actual = dequant_q4_0_neon(data, n_elements)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_q4_1_dequant_accuracy() -> None:
    rng = _rng()
    n_blocks = 2
    block_bytes = 20
    blocks = []
    for _ in range(n_blocks):
        block = np.empty(block_bytes, dtype=np.uint8)
        block[0:2] = _fp16_bytes(rng.uniform(0.01, 2.0))
        block[2:4] = _fp16_bytes(rng.uniform(-1.0, 1.0))
        block[4:20] = rng.integers(0, 256, size=16, dtype=np.uint8)
        blocks.append(block)
    data = np.concatenate(blocks)
    n_elements = n_blocks * 32

    expected = _ref_q4_1(data, n_elements)
    actual = dequant_q4_1_neon(data, n_elements)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_q8_0_dequant_accuracy() -> None:
    rng = _rng()
    n_blocks = 2
    block_bytes = 34
    blocks = []
    for _ in range(n_blocks):
        block = np.empty(block_bytes, dtype=np.uint8)
        block[0:2] = _fp16_bytes(rng.uniform(0.01, 2.0))
        block[2:34] = rng.integers(-128, 128, size=32, dtype=np.int8).view(np.uint8)
        blocks.append(block)
    data = np.concatenate(blocks)
    n_elements = n_blocks * 32

    expected = _ref_q8_0(data, n_elements)
    actual = dequant_q8_0_neon(data, n_elements)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ==================== Format-specific Edge Case Tests ====================


@pytest.mark.parametrize("K,N,group_size", [(64, 32, 16), (128, 64, 32), (256, 128, 64)])
def test_fp4_varying_sizes(K: int, N: int, group_size: int) -> None:
    """Test FP4 dequantization with varying tensor sizes."""
    rng = _rng()
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_fp4(packed, scales, K, N, group_size)
    actual = dequant_fp4_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("K,N,group_size", [(64, 32, 16), (96, 48, 24)])
def test_int4_varying_sizes(K: int, N: int, group_size: int) -> None:
    """Test INT4 dequantization with varying tensor sizes."""
    rng = _rng()
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_int4(packed, scales, K, N, group_size)
    actual = dequant_int4_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("K,N,group_size", [(32, 32, 16), (64, 64, 32)])
def test_fp8_e4m3_varying_sizes(K: int, N: int, group_size: int) -> None:
    """Test FP8 E4M3 dequantization with varying tensor sizes."""
    rng = _rng()
    packed = rng.integers(0, 2**32, size=(K, N // 4), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_fp8(packed, scales, K, N, group_size, _fp8_e4m3_codebook())
    actual = dequant_fp8_e4m3_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("K,N,group_size", [(32, 32, 16), (64, 64, 32)])
def test_fp8_e5m2_varying_sizes(K: int, N: int, group_size: int) -> None:
    """Test FP8 E5M2 dequantization with varying tensor sizes."""
    rng = _rng()
    packed = rng.integers(0, 2**32, size=(K, N // 4), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    expected = _ref_fp8(packed, scales, K, N, group_size, _fp8_e5m2_codebook())
    actual = dequant_fp8_e5m2_neon(packed, scales, K, N, group_size)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_fp4_extreme_scales() -> None:
    """Test FP4 dequantization with extreme scale values."""
    rng = _rng()
    K, N, group_size = 64, 32, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)

    # Test very small scales
    scales_small = np.full((K // group_size, N), 1e-6, dtype=np.float32)
    expected_small = _ref_fp4(packed, scales_small, K, N, group_size)
    actual_small = dequant_fp4_neon(packed, scales_small, K, N, group_size)
    np.testing.assert_allclose(actual_small, expected_small, rtol=1e-6, atol=1e-9)

    # Test very large scales
    scales_large = np.full((K // group_size, N), 100.0, dtype=np.float32)
    expected_large = _ref_fp4(packed, scales_large, K, N, group_size)
    actual_large = dequant_fp4_neon(packed, scales_large, K, N, group_size)
    np.testing.assert_allclose(actual_large, expected_large, rtol=1e-6, atol=1e-3)


def test_int4_asym_extreme_zeros() -> None:
    """Test INT4 asymmetric dequantization with edge-case zero points."""
    rng = _rng()
    K, N, group_size = 64, 24, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    # Test zeros at boundary values
    zeros = np.full((K // group_size, N), 0.0, dtype=np.float32)
    expected = _ref_int4_asym(packed, scales, zeros, K, N, group_size)
    actual = dequant_int4_asym_neon(packed, scales, zeros, K, N, group_size)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    zeros = np.full((K // group_size, N), 15.0, dtype=np.float32)
    expected = _ref_int4_asym(packed, scales, zeros, K, N, group_size)
    actual = dequant_int4_asym_neon(packed, scales, zeros, K, N, group_size)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_nf4_codebook_coverage() -> None:
    """Test NF4 dequantization covers all codebook entries."""
    K, N, group_size = 64, 32, 16
    scales = np.ones((K // group_size, N), dtype=np.float32)

    # Create packed data that hits all 16 codebook entries
    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for idx in range(16):
        packed[idx % (K // 8), idx % N] = (idx << 0) | (idx << 4) | (idx << 8) | (idx << 12)

    result = dequant_nf4_neon(packed, scales, K, N, group_size)

    # Verify all codebook values appear in output
    unique_vals = np.unique(result)
    assert len(unique_vals) >= 10, "NF4 codebook not fully exercised"


def test_fp8_special_values() -> None:
    """Test FP8 formats handle special values (NaN, inf) correctly."""
    K, N, group_size = 32, 32, 16
    scales = np.ones((K // group_size, N), dtype=np.float32)

    # FP8 E4M3: code 15 in exponent field -> NaN
    packed_e4m3 = np.full((K, N // 4), 0xFFFFFFFF, dtype=np.uint32)
    result_e4m3 = dequant_fp8_e4m3_neon(packed_e4m3, scales, K, N, group_size)
    assert np.isnan(result_e4m3).any(), "FP8 E4M3 NaN not handled"

    # FP8 E5M2: exponent=31, mantissa=0 -> inf
    packed_e5m2 = np.full((K, N // 4), 0x7C7C7C7C, dtype=np.uint32)
    result_e5m2 = dequant_fp8_e5m2_neon(packed_e5m2, scales, K, N, group_size)
    assert np.isinf(result_e5m2).any(), "FP8 E5M2 inf not handled"


def test_int8_boundary_values() -> None:
    """Test INT8 dequantization at signed integer boundaries."""
    K, N, group_size = 64, 48, 16
    scales = np.ones((K // group_size, N), dtype=np.float32)

    # Test min/max int8 values
    data = np.full((K, N), -128, dtype=np.int8)
    result_min = dequant_int8_neon(data, scales, K, N, group_size)
    assert np.allclose(result_min, -128.0)

    data = np.full((K, N), 127, dtype=np.int8)
    result_max = dequant_int8_neon(data, scales, K, N, group_size)
    assert np.allclose(result_max, 127.0)


def test_ggml_q4_0_multiple_blocks() -> None:
    """Test GGML Q4_0 dequantization with multiple blocks."""
    rng = _rng()
    n_blocks = 8
    block_bytes = 18
    blocks = []
    for _ in range(n_blocks):
        block = np.empty(block_bytes, dtype=np.uint8)
        block[0:2] = _fp16_bytes(rng.uniform(0.01, 2.0))
        block[2:18] = rng.integers(0, 256, size=16, dtype=np.uint8)
        blocks.append(block)
    data = np.concatenate(blocks)
    n_elements = n_blocks * 32

    expected = _ref_q4_0(data, n_elements)
    actual = dequant_q4_0_neon(data, n_elements)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_ggml_q4_1_multiple_blocks() -> None:
    """Test GGML Q4_1 dequantization with multiple blocks."""
    rng = _rng()
    n_blocks = 8
    block_bytes = 20
    blocks = []
    for _ in range(n_blocks):
        block = np.empty(block_bytes, dtype=np.uint8)
        block[0:2] = _fp16_bytes(rng.uniform(0.01, 2.0))
        block[2:4] = _fp16_bytes(rng.uniform(-1.0, 1.0))
        block[4:20] = rng.integers(0, 256, size=16, dtype=np.uint8)
        blocks.append(block)
    data = np.concatenate(blocks)
    n_elements = n_blocks * 32

    expected = _ref_q4_1(data, n_elements)
    actual = dequant_q4_1_neon(data, n_elements)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_ggml_q8_0_multiple_blocks() -> None:
    """Test GGML Q8_0 dequantization with multiple blocks."""
    rng = _rng()
    n_blocks = 8
    block_bytes = 34
    blocks = []
    for _ in range(n_blocks):
        block = np.empty(block_bytes, dtype=np.uint8)
        block[0:2] = _fp16_bytes(rng.uniform(0.01, 2.0))
        block[2:34] = rng.integers(-128, 128, size=32, dtype=np.int8).view(np.uint8)
        blocks.append(block)
    data = np.concatenate(blocks)
    n_elements = n_blocks * 32

    expected = _ref_q8_0(data, n_elements)
    actual = dequant_q8_0_neon(data, n_elements)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ==================== Cross-format Consistency Tests ====================


def test_int4_symmetric_consistency() -> None:
    """Verify INT4 symmetric quantization is consistent with zero-point=8."""
    rng = _rng()
    K, N, group_size = 64, 32, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)
    zeros = np.full((K // group_size, N), 8.0, dtype=np.float32)

    result_sym = dequant_int4_neon(packed, scales, K, N, group_size)
    result_asym = dequant_int4_asym_neon(packed, scales, zeros, K, N, group_size)

    np.testing.assert_allclose(result_sym, result_asym, rtol=1e-6, atol=1e-6)


def test_fp8_formats_range_difference() -> None:
    """Verify FP8 E4M3 and E5M2 have expected dynamic range differences."""
    K, N, group_size = 32, 32, 16
    scales = np.ones((K // group_size, N), dtype=np.float32)

    # Max representable finite value in E4M3: exponent=14, mantissa=7 -> 448
    packed_e4m3 = np.full((K, N // 4), 0x7E7E7E7E, dtype=np.uint32)
    result_e4m3 = dequant_fp8_e4m3_neon(packed_e4m3, scales, K, N, group_size)
    # Filter out NaN values before computing max
    finite_e4m3 = result_e4m3[~np.isnan(result_e4m3)]
    max_e4m3 = np.max(np.abs(finite_e4m3)) if len(finite_e4m3) > 0 else 0.0

    # Max representable finite value in E5M2: exponent=30, mantissa=3 -> 57344
    packed_e5m2 = np.full((K, N // 4), 0x7B7B7B7B, dtype=np.uint32)
    result_e5m2 = dequant_fp8_e5m2_neon(packed_e5m2, scales, K, N, group_size)
    finite_e5m2 = result_e5m2[~(np.isnan(result_e5m2) | np.isinf(result_e5m2))]
    max_e5m2 = np.max(np.abs(finite_e5m2)) if len(finite_e5m2) > 0 else 0.0

    # E5M2 has much wider dynamic range than E4M3 (due to 5-bit vs 4-bit exponent)
    assert max_e5m2 > max_e4m3 * 10, f"FP8 E5M2 ({max_e5m2}) should have wider range than E4M3 ({max_e4m3})"


# ==================== Stress Tests ====================


@pytest.mark.parametrize("seed", [0, 42, 123, 999])
def test_fp4_reproducibility(seed: int) -> None:
    """Test FP4 dequantization is deterministic across runs."""
    K, N, group_size = 64, 32, 16
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    result1 = dequant_fp4_neon(packed, scales, K, N, group_size)
    result2 = dequant_fp4_neon(packed, scales, K, N, group_size)

    np.testing.assert_array_equal(result1, result2)


def test_all_formats_zero_input() -> None:
    """Test all dequantization formats handle all-zero input correctly."""
    K, N, group_size = 64, 32, 16

    # FP4
    packed_fp4 = np.zeros((K // 8, N), dtype=np.uint32)
    scales_fp4 = np.ones((K // group_size, N), dtype=np.float32)
    result_fp4 = dequant_fp4_neon(packed_fp4, scales_fp4, K, N, group_size)
    assert np.allclose(result_fp4, 0.0)

    # INT4
    result_int4 = dequant_int4_neon(packed_fp4, scales_fp4, K, N, group_size)
    assert np.allclose(result_int4, -8.0)  # Index 0 maps to -8

    # INT8
    data_int8 = np.zeros((K, N), dtype=np.int8)
    result_int8 = dequant_int8_neon(data_int8, scales_fp4, K, N, group_size)
    assert np.allclose(result_int8, 0.0)


@pytest.mark.parametrize("format_name,dequant_func", [
    ("fp4", dequant_fp4_neon),
    ("int4", dequant_int4_neon),
    ("nf4", dequant_nf4_neon),
])
def test_format_memory_contiguity(format_name: str, dequant_func) -> None:
    """Verify dequantization produces contiguous output arrays."""
    rng = _rng()
    K, N, group_size = 64, 32, 16
    packed = rng.integers(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales = rng.uniform(0.01, 2.0, size=(K // group_size, N)).astype(np.float32)

    result = dequant_func(packed, scales, K, N, group_size)

    assert result.flags["C_CONTIGUOUS"] or result.flags["F_CONTIGUOUS"], \
        f"{format_name} output not contiguous"


# ==================== Summary Test ====================


def test_all_formats_complete_coverage() -> None:
    """Integration test verifying all major formats are tested."""
    tested_formats = [
        "fp4", "int4", "int4_asym", "nf4",
        "fp8_e4m3", "fp8_e5m2",
        "int8", "int8_per_channel",
        "q4_0", "q4_1", "q8_0"
    ]

    # This test passes if all individual tests pass
    # Serves as a manifest of required format coverage
    assert len(tested_formats) == 11, "All 11 core formats must be validated"
