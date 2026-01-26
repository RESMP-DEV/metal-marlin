"""Comprehensive unit tests for all dequantization functions.

Validates FP4 (E2M1/NVFP4) and INT4 (U4/S4) dequantization at three levels:
  1. Pure-Python bitwise construction (reference correctness)
  2. Magic number trick simulation (INT4 algorithm verification)
  3. Metal kernel execution via PyObjC (GPU correctness)

Usage:
    cd metal_marlin
    uv run pytest tests/test_dequant.py -v
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

# ============================================================================
# FP4 E2M1 reference values (ground truth)
# ============================================================================

FP4_E2M1_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float16)

# Expected FP16 bit patterns for each FP4 code (for bit-exact verification)
FP4_E2M1_BITS: dict[int, int] = {
    0x0: 0x0000,  # +0.0
    0x1: 0x3800,  # +0.5
    0x2: 0x3C00,  # +1.0
    0x3: 0x3E00,  # +1.5
    0x4: 0x4000,  # +2.0
    0x5: 0x4200,  # +3.0
    0x6: 0x4400,  # +4.0
    0x7: 0x4600,  # +6.0
    0x8: 0x8000,  # -0.0
    0x9: 0xB800,  # -0.5
    0xA: 0xBC00,  # -1.0
    0xB: 0xBE00,  # -1.5
    0xC: 0xC000,  # -2.0
    0xD: 0xC200,  # -3.0
    0xE: 0xC400,  # -4.0
    0xF: 0xC600,  # -6.0
}


# ============================================================================
# Reference implementations
# ============================================================================


def fp16_from_bits(bits: int) -> np.float16:
    """Construct an FP16 value from its raw bit pattern."""
    return np.frombuffer(struct.pack('<H', bits & 0xFFFF), dtype=np.float16)[0]


def dequant_fp4_scalar(nibble: int) -> np.float16:
    """Bitwise FP4 E2M1 -> FP16 dequantization (no LUT).

    FP4 E2M1 layout: [S:1][E:2][M:1]
      Normal (E>0):    (-1)^S * 2^(E-1) * (1 + M*0.5)
      Subnormal (E=0, M=1): (-1)^S * 0.5
      Zero (E=0, M=0): (-1)^S * 0.0

    FP16 reconstruction:
      sign_16 = S << 15
      exp_16  = (E + 14) << 10  [for E>0; maps bias-1 to bias-15]
      mant_16 = M << 9          [1 mantissa bit -> position 9 of 10-bit field]
    """
    assert 0 <= nibble <= 15
    S = (nibble >> 3) & 1
    E = (nibble >> 1) & 3
    M = nibble & 1

    if E == 0:
        if M == 0:
            fp16_bits = S << 15
        else:
            # Subnormal 0.5: exp=14 in FP16 (2^(14-15) = 0.5), mant=0
            fp16_bits = (S << 15) | (14 << 10)
    else:
        fp16_exp = E + 14
        fp16_mant = M << 9
        fp16_bits = (S << 15) | (fp16_exp << 10) | fp16_mant

    return fp16_from_bits(fp16_bits)


def dequant_fp4_x8(packed_u32: int, scale: float = 1.0) -> list[np.float16]:
    """Dequantize 8 FP4 values from a packed uint32."""
    results = []
    for i in range(8):
        nibble = (packed_u32 >> (i * 4)) & 0xF
        val = dequant_fp4_scalar(nibble)
        results.append(np.float16(float(val) * np.float16(scale)))
    return results


def pack_fp4_values(values: list[int]) -> int:
    """Pack up to 8 FP4 nibble codes into a uint32."""
    assert len(values) <= 8
    packed = 0
    for i, v in enumerate(values):
        packed |= (v & 0xF) << (i * 4)
    return packed


def magic_dequant_u4_scalar(val_4bit: int) -> float:
    """Magic bias trick for unsigned INT4: OR with 0x6400, subtract 1024.0."""
    assert 0 <= val_4bit <= 15
    biased_bits = (val_4bit & 0x000F) | 0x6400
    biased_fp16 = fp16_from_bits(biased_bits)
    bias = fp16_from_bits(0x6400)  # 1024.0
    return float(np.float16(biased_fp16) - np.float16(bias))


def magic_dequant_s4_scalar(val_4bit: int) -> float:
    """Magic bias trick for signed INT4: OR with 0x6400, subtract 1032.0."""
    assert 0 <= val_4bit <= 15
    biased_bits = (val_4bit & 0x000F) | 0x6400
    biased_fp16 = fp16_from_bits(biased_bits)
    bias = np.float16(1032.0)  # 1024.0 + 8.0 (offset for signed)
    return float(np.float16(biased_fp16) - bias)


def dequant_u4x8(packed_u32: int, scale: float, zero_point: float) -> list[float]:
    """Reference unsigned INT4 x8 dequant with scale and zero_point."""
    results = []
    for i in range(8):
        val = (packed_u32 >> (i * 4)) & 0xF
        raw = magic_dequant_u4_scalar(val)
        dequantized = float((np.float16(raw) - np.float16(zero_point)) * np.float16(scale))
        results.append(dequantized)
    return results


def dequant_s4x8(packed_u32: int, scale: float, zero_point: float) -> list[float]:
    """Reference signed INT4 x8 dequant with scale and zero_point."""
    results = []
    for i in range(8):
        val = (packed_u32 >> (i * 4)) & 0xF
        raw = magic_dequant_s4_scalar(val)
        dequantized = float((np.float16(raw) - np.float16(zero_point)) * np.float16(scale))
        results.append(dequantized)
    return results


def ref_dequant_u4_bulk(packed_u32: np.ndarray, scales: np.ndarray,
                         zeros: np.ndarray, group_size: int) -> np.ndarray:
    """Reference bulk unsigned INT4 dequantization."""
    n_packed = len(packed_u32)
    output = np.zeros(n_packed * 8, dtype=np.float16)
    for i, packed in enumerate(packed_u32):
        base_idx = i * 8
        group_idx = base_idx // group_size
        scale = scales[group_idx]
        zero_point = zeros[group_idx]
        for nibble in range(8):
            val = (int(packed) >> (nibble * 4)) & 0xF
            output[base_idx + nibble] = np.float16(
                (np.float16(val) - zero_point) * scale
            )
    return output


def ref_dequant_s4_bulk(packed_u32: np.ndarray, scales: np.ndarray,
                         zeros: np.ndarray, group_size: int) -> np.ndarray:
    """Reference bulk signed INT4 dequantization (offset binary)."""
    n_packed = len(packed_u32)
    output = np.zeros(n_packed * 8, dtype=np.float16)
    for i, packed in enumerate(packed_u32):
        base_idx = i * 8
        group_idx = base_idx // group_size
        scale = scales[group_idx]
        zero_point = zeros[group_idx]
        for nibble in range(8):
            val = (int(packed) >> (nibble * 4)) & 0xF
            signed_val = np.float16(val) - np.float16(8.0) - zero_point
            output[base_idx + nibble] = np.float16(signed_val * scale)
    return output


def ref_dequant_fp4_bulk(packed_u32: np.ndarray, scales: np.ndarray,
                          group_size: int) -> np.ndarray:
    """Reference bulk FP4 E2M1 dequantization."""
    n_packed = len(packed_u32)
    output = np.zeros(n_packed * 8, dtype=np.float16)
    for i, packed in enumerate(packed_u32):
        base_idx = i * 8
        group_idx = base_idx // group_size
        scale = scales[group_idx]
        for nibble_pos in range(8):
            nibble = (int(packed) >> (nibble_pos * 4)) & 0xF
            val = dequant_fp4_scalar(nibble)
            output[base_idx + nibble_pos] = np.float16(float(val) * float(scale))
    return output


# ============================================================================
# Metal kernel helpers
# ============================================================================


def _check_metal_available() -> bool:
    try:
        import Metal  # noqa: F401
        return True
    except ImportError:
        return False


def _read_metal_buffer(buf, nbytes: int) -> bytes:
    """Read bytes from a Metal buffer (handles PyObjC variants)."""
    import ctypes
    contents = buf.contents()
    if isinstance(contents, int):
        arr = (ctypes.c_char * nbytes).from_address(contents)
        return bytes(arr)
    elif isinstance(contents, memoryview):
        return bytes(contents[:nbytes])
    elif hasattr(contents, '__getitem__'):
        return b''.join(contents[i] for i in range(nbytes))
    else:
        ptr = ctypes.cast(contents, ctypes.POINTER(ctypes.c_char * nbytes))
        return bytes(ptr.contents)


def _compile_dequant_shader():
    """Compile the dequant.metal shader and return (device, library)."""
    from pathlib import Path

    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device is not None, "No Metal device found"

    shader_path = Path(__file__).parent.parent / "src" / "dequant.metal"
    source = shader_path.read_text()
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    assert err is None, f"Metal compile error: {err}"
    return device, library


def run_metal_fp4_all_codes() -> np.ndarray:
    """Run test_fp4_all_codes kernel, returns 16 FP16 values."""
    import Metal
    device, library = _compile_dequant_shader()

    func = library.newFunctionWithName_("test_fp4_all_codes")
    assert func is not None
    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    assert err is None

    output_size = 16 * 2
    buf_output = device.newBufferWithLength_options_(
        output_size, Metal.MTLResourceStorageModeShared)

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 0)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(16, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, output_size)
    return np.frombuffer(raw, dtype=np.float16).copy()


def run_metal_fp4_packed_scaled(packed: np.uint32, scale: np.float16) -> np.ndarray:
    """Run test_fp4_packed_scaled kernel: 8 FP4 values with scale."""
    import Metal
    device, library = _compile_dequant_shader()

    func = library.newFunctionWithName_("test_fp4_packed_scaled")
    assert func is not None
    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    assert err is None

    packed_bytes = np.array([packed], dtype=np.uint32).tobytes()
    scale_bytes = np.array([scale], dtype=np.float16).tobytes()

    buf_packed = device.newBufferWithBytes_length_options_(
        packed_bytes, 4, Metal.MTLResourceStorageModeShared)
    buf_scale = device.newBufferWithBytes_length_options_(
        scale_bytes, 2, Metal.MTLResourceStorageModeShared)
    buf_output = device.newBufferWithLength_options_(
        16, Metal.MTLResourceStorageModeShared)

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 2)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, 16)
    return np.frombuffer(raw, dtype=np.float16).copy()


def run_metal_int4_dequant(packed_weights: np.ndarray, scales: np.ndarray,
                            zeros: np.ndarray, group_size: int,
                            is_signed: bool) -> np.ndarray:
    """Run dequant_int4_kernel on Metal hardware."""
    import Metal
    device, library = _compile_dequant_shader()

    func = library.newFunctionWithName_("dequant_int4_kernel")
    assert func is not None
    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    assert err is None

    num_elements = len(packed_weights) * 8
    num_packed = len(packed_weights)

    buf_packed = device.newBufferWithBytes_length_options_(
        packed_weights.tobytes(), packed_weights.nbytes,
        Metal.MTLResourceStorageModeShared)
    buf_scales = device.newBufferWithBytes_length_options_(
        scales.tobytes(), scales.nbytes,
        Metal.MTLResourceStorageModeShared)
    buf_zeros = device.newBufferWithBytes_length_options_(
        zeros.tobytes(), zeros.nbytes,
        Metal.MTLResourceStorageModeShared)

    output_size = num_elements * 2
    buf_output = device.newBufferWithLength_options_(
        output_size, Metal.MTLResourceStorageModeShared)

    buf_num_elements = device.newBufferWithBytes_length_options_(
        np.array([num_elements], dtype=np.uint32).tobytes(), 4,
        Metal.MTLResourceStorageModeShared)
    buf_group_size = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(), 4,
        Metal.MTLResourceStorageModeShared)
    buf_is_signed = device.newBufferWithBytes_length_options_(
        np.array([1 if is_signed else 0], dtype=np.uint32).tobytes(), 4,
        Metal.MTLResourceStorageModeShared)

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_scales, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_zeros, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 3)
    encoder.setBuffer_offset_atIndex_(buf_num_elements, 0, 4)
    encoder.setBuffer_offset_atIndex_(buf_group_size, 0, 5)
    encoder.setBuffer_offset_atIndex_(buf_is_signed, 0, 6)

    threads_per_group = min(256, num_packed)
    num_groups = (num_packed + threads_per_group - 1) // threads_per_group
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(num_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_group, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, output_size)
    return np.frombuffer(raw, dtype=np.float16).copy()


# ============================================================================
# Test: FP4 E2M1 exact values
# ============================================================================


class TestFP4ExactValues:
    """FP4 E2M1 has exactly 16 representable values. Verify all of them."""

    @pytest.mark.parametrize("code,expected_val", [
        (0b0000, 0.0),
        (0b0001, 0.5),
        (0b0010, 1.0),
        (0b0011, 1.5),
        (0b0100, 2.0),
        (0b0101, 3.0),
        (0b0110, 4.0),
        (0b0111, 6.0),
        (0b1000, -0.0),
        (0b1001, -0.5),
        (0b1010, -1.0),
        (0b1011, -1.5),
        (0b1100, -2.0),
        (0b1101, -3.0),
        (0b1110, -4.0),
        (0b1111, -6.0),
    ])
    def test_fp4_code_produces_expected_value(self, code: int, expected_val: float):
        """Each FP4 code maps to exactly one value via bitwise construction."""
        result = dequant_fp4_scalar(code)
        if expected_val == 0.0:
            # Distinguish +0.0 from -0.0 by bit pattern
            result_bits = np.array([result]).view(np.uint16)[0]
            expected_bits = FP4_E2M1_BITS[code]
            assert result_bits == expected_bits, (
                f"Code {code:04b}: got bits 0x{result_bits:04X}, "
                f"expected 0x{expected_bits:04X}"
            )
        else:
            assert float(result) == pytest.approx(expected_val, abs=1e-5), (
                f"Code {code:04b}: got {result}, expected {expected_val}"
            )

    @pytest.mark.parametrize("code", range(16))
    def test_fp4_bit_pattern_exact(self, code: int):
        """Verify the exact FP16 bit pattern for each FP4 code."""
        result = dequant_fp4_scalar(code)
        result_bits = np.array([result]).view(np.uint16)[0]
        expected_bits = FP4_E2M1_BITS[code]
        assert result_bits == expected_bits, (
            f"Code 0x{code:X}: result 0x{result_bits:04X} != expected 0x{expected_bits:04X}"
        )

    @pytest.mark.parametrize("code", range(16))
    def test_bitwise_matches_lut(self, code: int):
        """Bitwise construction produces same bits as LUT lookup."""
        bitwise = dequant_fp4_scalar(code)
        lut_val = FP4_E2M1_LUT[code]
        bitwise_bits = np.array([bitwise]).view(np.uint16)[0]
        lut_bits = np.array([lut_val]).view(np.uint16)[0]
        assert bitwise_bits == lut_bits

    def test_positive_negative_symmetry(self):
        """Negative codes are exact negations of positive codes (except zero)."""
        for pos_code in range(1, 8):
            neg_code = pos_code + 8
            pos_val = float(dequant_fp4_scalar(pos_code))
            neg_val = float(dequant_fp4_scalar(neg_code))
            assert neg_val == -pos_val, (
                f"Asymmetry: code {pos_code}={pos_val}, code {neg_code}={neg_val}"
            )

    def test_monotonic_positive(self):
        """Positive FP4 values are monotonically increasing (codes 1-7)."""
        prev = 0.0
        for code in range(1, 8):
            val = float(dequant_fp4_scalar(code))
            assert val > prev, f"Non-monotonic at code {code}: {val} <= {prev}"
            prev = val

    def test_value_range(self):
        """All values lie in [-6.0, 6.0]."""
        for code in range(16):
            val = float(dequant_fp4_scalar(code))
            assert -6.0 <= val <= 6.0, f"Code {code} out of range: {val}"


# ============================================================================
# Test: FP4 with scale factors
# ============================================================================


class TestFP4WithScale:
    """Test FP4 dequantization combined with per-group scale factors."""

    @pytest.mark.parametrize("scale", [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    def test_unit_value_scaled(self, scale: float):
        """Code 0b0010 (value 1.0) scaled by various factors."""
        packed = pack_fp4_values([0b0010] * 8)
        results = dequant_fp4_x8(packed, scale=scale)
        for i in range(8):
            expected = np.float16(1.0 * np.float16(scale))
            assert results[i] == expected, (
                f"Scale {scale}, index {i}: got {results[i]}, expected {expected}"
            )

    def test_zero_scale_produces_zeros(self):
        """Zero scale should produce all zeros regardless of input."""
        packed = 0xFFFFFFFF  # All codes = 0xF (value -6.0)
        results = dequant_fp4_x8(packed, scale=0.0)
        for i, r in enumerate(results):
            assert float(r) == 0.0, f"Index {i}: got {r} with zero scale"

    def test_large_scale_within_fp16(self):
        """Scale producing values near FP16 max (65504)."""
        # Code 7 = 6.0, scale = 5000.0 -> 30000.0 (within FP16 range)
        packed = pack_fp4_values([0b0111] * 8)
        scale = np.float16(5000.0)
        results = dequant_fp4_x8(packed, scale=float(scale))
        expected = np.float16(6.0 * scale)
        for r in results:
            assert abs(float(r) - float(expected)) < 200.0  # FP16 precision at this magnitude

    def test_small_scale_subnormal_fp16(self):
        """Very small scale producing near-zero FP16 values."""
        packed = pack_fp4_values([0b0001] * 8)  # value 0.5
        scale = np.float16(1.0 / 512.0)  # ~0.00195
        results = dequant_fp4_x8(packed, scale=float(scale))
        for r in results:
            # 0.5 * (1/512) = ~0.000977 which is representable in FP16
            assert float(r) >= 0.0
            assert float(r) < 0.01

    @pytest.mark.parametrize("seed", range(20))
    def test_random_packed_scaled(self, seed: int):
        """Random FP4 codes with random scale match element-wise computation."""
        rng = np.random.default_rng(seed + 1000)
        nibbles = rng.integers(0, 16, size=8, dtype=np.uint8)
        packed = pack_fp4_values(nibbles.tolist())
        scale = float(np.float16(rng.uniform(0.01, 8.0)))

        results = dequant_fp4_x8(packed, scale=scale)
        for i in range(8):
            expected = np.float16(float(FP4_E2M1_LUT[nibbles[i]]) * np.float16(scale))
            r_bits = np.array([results[i]]).view(np.uint16)[0]
            e_bits = np.array([expected]).view(np.uint16)[0]
            assert r_bits == e_bits, (
                f"Seed {seed}, nibble {i} (code 0x{nibbles[i]:X}): "
                f"got 0x{r_bits:04X}, expected 0x{e_bits:04X}"
            )


# ============================================================================
# Test: FP4 batch (bulk) dequantization
# ============================================================================


class TestFP4Batch:
    """Test batch FP4 dequantization across multiple groups."""

    def test_single_group(self):
        """Single group of 8 elements."""
        packed = np.array([pack_fp4_values([0, 1, 2, 3, 4, 5, 6, 7])], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        result = ref_dequant_fp4_bulk(packed, scales, group_size=128)
        expected = FP4_E2M1_LUT[:8]
        for i in range(8):
            r_bits = result.view(np.uint16)[i]
            e_bits = expected.view(np.uint16)[i]
            assert r_bits == e_bits

    def test_multiple_groups_different_scales(self):
        """Two groups with different scales."""
        packed = np.array([
            pack_fp4_values([2, 2, 2, 2, 2, 2, 2, 2]),  # All 1.0
            pack_fp4_values([2, 2, 2, 2, 2, 2, 2, 2]),  # All 1.0
        ], dtype=np.uint32)
        scales = np.array([1.0, 3.0], dtype=np.float16)
        result = ref_dequant_fp4_bulk(packed, scales, group_size=8)

        # Group 0: 1.0 * 1.0 = 1.0
        for i in range(8):
            assert float(result[i]) == pytest.approx(1.0)
        # Group 1: 1.0 * 3.0 = 3.0
        for i in range(8, 16):
            assert float(result[i]) == pytest.approx(3.0)

    def test_large_batch(self):
        """128 elements (16 uint32 words) across 4 groups."""
        rng = np.random.default_rng(42)
        num_packed = 16
        packed = np.array(
            [pack_fp4_values(rng.integers(0, 16, size=8).tolist()) for _ in range(num_packed)],
            dtype=np.uint32,
        )
        scales = np.float16(rng.uniform(0.1, 4.0, size=4))
        group_size = 32  # 32 elements per group

        result = ref_dequant_fp4_bulk(packed, scales, group_size)
        assert result.shape == (128,)
        assert result.dtype == np.float16

        # Cross-validate: each element matches scalar dequant
        for i in range(num_packed):
            for j in range(8):
                idx = i * 8 + j
                group_idx = idx // group_size
                nibble = (int(packed[i]) >> (j * 4)) & 0xF
                expected = np.float16(
                    float(dequant_fp4_scalar(nibble)) * float(scales[group_idx])
                )
                assert result[idx] == expected, f"Mismatch at idx {idx}"


# ============================================================================
# Test: INT4 unsigned (U4) dequantization
# ============================================================================


class TestU4Dequant:
    """Test unsigned INT4 dequantization using magic bias trick."""

    @pytest.mark.parametrize("val", range(16))
    def test_u4_identity(self, val: int):
        """Each U4 value [0,15] dequants to itself via magic trick."""
        result = magic_dequant_u4_scalar(val)
        assert result == pytest.approx(float(val), abs=1e-3), (
            f"U4 val={val}: got {result}"
        )

    def test_u4x8_sequential(self):
        """Pack values 0-7 and verify identity dequant."""
        packed = pack_fp4_values(list(range(8)))
        results = dequant_u4x8(packed, scale=1.0, zero_point=0.0)
        for i in range(8):
            assert abs(results[i] - float(i)) < 1e-3

    def test_u4x8_max_values(self):
        """All 15s dequant to 15.0."""
        packed = 0xFFFFFFFF
        results = dequant_u4x8(packed, scale=1.0, zero_point=0.0)
        for r in results:
            assert abs(r - 15.0) < 1e-3

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 0.125])
    def test_u4x8_with_scale(self, scale: float):
        """Scale factor applied correctly."""
        packed = pack_fp4_values(list(range(8)))
        results = dequant_u4x8(packed, scale=scale, zero_point=0.0)
        for i in range(8):
            expected = float(np.float16(i) * np.float16(scale))
            assert abs(results[i] - expected) < 0.05

    @pytest.mark.parametrize("zero_point", [0.0, 4.0, 8.0, 15.0])
    def test_u4x8_with_zero_point(self, zero_point: float):
        """Zero point subtracted correctly (asymmetric quantization)."""
        packed = pack_fp4_values(list(range(8)))
        results = dequant_u4x8(packed, scale=1.0, zero_point=zero_point)
        for i in range(8):
            expected = float((np.float16(i) - np.float16(zero_point)) * np.float16(1.0))
            assert abs(results[i] - expected) < 0.05

    def test_u4_centered_at_8(self):
        """Common use case: zero_point=8 maps [0,15] to [-8,7]."""
        packed = pack_fp4_values(list(range(8)))
        results = dequant_u4x8(packed, scale=1.0, zero_point=8.0)
        for i in range(8):
            expected = float(i - 8)
            assert abs(results[i] - expected) < 0.05


# ============================================================================
# Test: INT4 signed (S4) dequantization
# ============================================================================


class TestS4Dequant:
    """Test signed INT4 dequantization (offset binary: stored = actual + 8)."""

    @pytest.mark.parametrize("stored_val,expected_signed", [
        (0, -8.0),
        (1, -7.0),
        (7, -1.0),
        (8, 0.0),
        (9, 1.0),
        (15, 7.0),
    ])
    def test_s4_individual_values(self, stored_val: int, expected_signed: float):
        """Verify offset binary mapping: stored - 8 = signed value."""
        result = magic_dequant_s4_scalar(stored_val)
        assert abs(result - expected_signed) < 0.05, (
            f"Stored {stored_val}: got {result}, expected {expected_signed}"
        )

    def test_s4_full_range(self):
        """All 16 stored values map to [-8, 7]."""
        for stored in range(16):
            result = magic_dequant_s4_scalar(stored)
            expected = float(stored - 8)
            assert abs(result - expected) < 0.05

    def test_s4x8_basic(self):
        """S4 x8 dequant with scale=1, zero=0."""
        # Store values 0-7 (represent -8 to -1)
        packed = pack_fp4_values(list(range(8)))
        results = dequant_s4x8(packed, scale=1.0, zero_point=0.0)
        for i in range(8):
            expected = float(i - 8)
            assert abs(results[i] - expected) < 0.05

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    def test_s4x8_with_scale(self, scale: float):
        """S4 dequant with various scale factors."""
        packed = pack_fp4_values([8, 9, 10, 11, 12, 13, 14, 15])  # actual 0-7
        results = dequant_s4x8(packed, scale=scale, zero_point=0.0)
        for i in range(8):
            expected = float(np.float16(i) * np.float16(scale))
            assert abs(results[i] - expected) < 0.1


# ============================================================================
# Test: INT4 bulk dequantization
# ============================================================================


class TestINT4Bulk:
    """Test bulk INT4 dequantization across multiple groups."""

    def test_u4_single_group(self):
        """U4 bulk: single group, scale=1, zero=0."""
        packed = np.array([0x76543210], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        zeros = np.array([0.0], dtype=np.float16)
        result = ref_dequant_u4_bulk(packed, scales, zeros, group_size=128)
        expected = np.arange(8, dtype=np.float16)
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_u4_asymmetric(self):
        """U4 bulk with asymmetric quantization."""
        packed = np.array([0x76543210], dtype=np.uint32)
        scales = np.array([0.5], dtype=np.float16)
        zeros = np.array([4.0], dtype=np.float16)
        result = ref_dequant_u4_bulk(packed, scales, zeros, group_size=128)
        expected = np.float16([(i - 4.0) * 0.5 for i in range(8)])
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_s4_full_range(self):
        """S4 bulk: two words covering all 16 stored values."""
        packed = np.array([0x76543210, 0xFEDCBA98], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        zeros = np.array([0.0], dtype=np.float16)
        result = ref_dequant_s4_bulk(packed, scales, zeros, group_size=128)
        expected = np.float16([i - 8.0 for i in range(16)])
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_multiple_groups(self):
        """U4 bulk with multiple quantization groups and different parameters."""
        packed = np.array([0x76543210, 0xFEDCBA98], dtype=np.uint32)
        scales = np.array([1.0, 2.0], dtype=np.float16)
        zeros = np.array([0.0, 5.0], dtype=np.float16)
        result = ref_dequant_u4_bulk(packed, scales, zeros, group_size=8)
        # Group 0: (val - 0) * 1
        expected_g0 = np.float16([float(i) for i in range(8)])
        # Group 1: (val - 5) * 2
        expected_g1 = np.float16([(i - 5.0) * 2.0 for i in [8, 9, 10, 11, 12, 13, 14, 15]])
        expected = np.concatenate([expected_g0, expected_g1])
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    @pytest.mark.parametrize("seed", range(10))
    def test_u4_random_large(self, seed: int):
        """Large random U4 buffer correctness."""
        rng = np.random.default_rng(seed + 300)
        group_size = 128
        num_groups = 8
        num_packed = (group_size * num_groups) // 8

        packed = rng.integers(0, 2**32, size=num_packed, dtype=np.uint32)
        scales = np.float16(rng.uniform(0.01, 2.0, size=num_groups))
        zeros = np.float16(rng.uniform(0, 15, size=num_groups))

        result = ref_dequant_u4_bulk(packed, scales, zeros, group_size)
        assert result.shape == (group_size * num_groups,)
        assert result.dtype == np.float16
        assert not np.any(np.isnan(result))


# ============================================================================
# Test: Magic number trick vs reference cross-validation
# ============================================================================


class TestMagicVsReference:
    """Cross-validate the magic number trick against direct computation."""

    @pytest.mark.parametrize("seed", range(20))
    def test_u4_magic_matches_direct(self, seed: int):
        """Magic trick U4 values match simple subtraction."""
        rng = np.random.default_rng(seed + 400)
        values = rng.integers(0, 16, size=8, dtype=np.uint8)
        packed = pack_fp4_values(values.tolist())

        scale = float(np.float16(rng.uniform(0.01, 2.0)))
        zero_point = float(np.float16(rng.uniform(0, 15)))

        magic_results = dequant_u4x8(packed, scale, zero_point)

        packed_arr = np.array([packed], dtype=np.uint32)
        scales_arr = np.array([scale], dtype=np.float16)
        zeros_arr = np.array([zero_point], dtype=np.float16)
        ref_results = ref_dequant_u4_bulk(packed_arr, scales_arr, zeros_arr, group_size=128)

        for i in range(8):
            assert abs(magic_results[i] - float(ref_results[i])) < 0.1, (
                f"Mismatch at nibble {i}: magic={magic_results[i]}, ref={ref_results[i]}"
            )

    def test_magic_bias_is_1024(self):
        """Verify that 0x6400 in FP16 is exactly 1024.0."""
        bias = fp16_from_bits(0x6400)
        assert float(bias) == 1024.0

    def test_magic_trick_produces_integers(self):
        """Magic trick for U4: OR with 0x6400 and subtract 1024 gives integers."""
        for val in range(16):
            result = magic_dequant_u4_scalar(val)
            assert result == int(result), f"U4 {val} -> non-integer {result}"


# ============================================================================
# Test: Edge cases and numerical boundaries
# ============================================================================


class TestEdgeCases:
    """Test numerical edge cases for dequantization."""

    def test_fp4_zero_scale_all_codes(self):
        """Zero scale produces exactly 0.0 for all FP4 codes."""
        for code in range(16):
            packed = pack_fp4_values([code] * 8)
            results = dequant_fp4_x8(packed, scale=0.0)
            for i, r in enumerate(results):
                assert float(r) == 0.0, (
                    f"Code {code}, idx {i}: non-zero {r} with scale=0"
                )

    def test_u4_zero_scale(self):
        """Zero scale for U4 produces all zeros."""
        packed = 0xFFFFFFFF
        results = dequant_u4x8(packed, scale=0.0, zero_point=0.0)
        for r in results:
            assert abs(r) < 1e-5

    def test_fp4_max_representable(self):
        """Largest FP4 value (6.0) with largest safe scale."""
        # 6.0 * 10000.0 = 60000.0, which is within FP16 range (max ~65504)
        packed = pack_fp4_values([0b0111] * 8)
        results = dequant_fp4_x8(packed, scale=10000.0)
        for r in results:
            # FP16 can represent up to 65504
            val = float(r)
            assert 50000.0 < val < 65504.0

    def test_fp4_negative_max_representable(self):
        """Largest negative FP4 value with large scale."""
        packed = pack_fp4_values([0b1111] * 8)  # -6.0
        results = dequant_fp4_x8(packed, scale=10000.0)
        for r in results:
            val = float(r)
            assert -65504.0 < val < -50000.0

    def test_fp4_subnormal_scaled(self):
        """FP4 subnormal (0.5) with very small scale stays finite."""
        packed = pack_fp4_values([0b0001] * 8)  # 0.5
        results = dequant_fp4_x8(packed, scale=1.0 / 1024.0)
        for r in results:
            val = float(r)
            assert val >= 0.0
            assert np.isfinite(val)

    def test_u4_all_same_value(self):
        """All 8 nibbles set to the same value."""
        for target in range(16):
            packed = 0
            for i in range(8):
                packed |= target << (i * 4)
            results = dequant_u4x8(packed, scale=1.0, zero_point=0.0)
            for r in results:
                assert abs(r - float(target)) < 0.05

    def test_packing_unpacking_roundtrip(self):
        """Pack and unpack nibbles preserves values."""
        for trial in range(100):
            rng = np.random.default_rng(trial)
            nibbles = rng.integers(0, 16, size=8, dtype=np.uint8)
            packed = pack_fp4_values(nibbles.tolist())
            for i in range(8):
                extracted = (packed >> (i * 4)) & 0xF
                assert extracted == nibbles[i]

    def test_alternating_zero_max(self):
        """Alternating 0 and max values."""
        packed = pack_fp4_values([0, 7, 0, 7, 0, 7, 0, 7])
        results = dequant_fp4_x8(packed, scale=1.0)
        expected = [0.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0, 6.0]
        for i, (r, e) in enumerate(zip(results, expected)):
            assert abs(float(r) - e) < 1e-3, f"Index {i}: {r} != {e}"


# ============================================================================
# Test: FP16 precision boundaries
# ============================================================================


class TestFP16Precision:
    """Verify dequant behavior at FP16 precision limits."""

    def test_smallest_positive_subnormal(self):
        """FP4 code 0x1 (0.5) is the smallest non-zero positive value."""
        result = dequant_fp4_scalar(0x1)
        assert float(result) == 0.5
        # Verify it's a normal FP16 (not subnormal)
        bits = np.array([result]).view(np.uint16)[0]
        exp_field = (bits >> 10) & 0x1F
        assert exp_field > 0, "Should be a normal FP16 number"

    def test_fp4_values_are_fp16_exact(self):
        """All 16 FP4 values are exactly representable in FP16."""
        for code in range(16):
            val = dequant_fp4_scalar(code)
            # Round-trip through float32 and back
            val_f32 = np.float32(val)
            val_back = np.float16(val_f32)
            assert np.array([val]).view(np.uint16)[0] == np.array([val_back]).view(np.uint16)[0]

    def test_u4_values_are_fp16_exact(self):
        """All 16 U4 integer values [0,15] are exactly representable in FP16."""
        for val in range(16):
            result = magic_dequant_u4_scalar(val)
            assert result == float(int(result))

    def test_scale_multiplication_precision(self):
        """Multiplication by scale preserves FP16 precision within 1 ULP."""
        code = 0b0011  # 1.5
        val = dequant_fp4_scalar(code)
        scale = np.float16(2.0)
        result = np.float16(float(val) * float(scale))
        # 1.5 * 2.0 = 3.0, which is exact in FP16
        assert float(result) == 3.0

    def test_near_overflow(self):
        """Scaled FP4 value near FP16 overflow stays finite."""
        # 6.0 * 10920 = 65520 < 65504 ... hmm, let's use a value that fits
        # FP16 max is 65504. 6.0 * 10900 = 65400 (safe)
        val = np.float16(6.0) * np.float16(10900.0)
        assert np.isfinite(val)


# ============================================================================
# Test: Metal kernels (skipped if PyObjC unavailable)
# ============================================================================


@pytest.mark.skipif(not _check_metal_available(),
                    reason="Metal API (PyObjC) not available")
class TestMetalFP4Kernels:
    """Test Metal FP4 dequant kernels against reference implementation."""

    def test_all_16_codes(self):
        """Metal kernel produces exact LUT values for all 16 FP4 codes."""
        metal_out = run_metal_fp4_all_codes()
        assert len(metal_out) == 16

        for code in range(16):
            metal_bits = metal_out.view(np.uint16)[code]
            expected_bits = FP4_E2M1_BITS[code]
            assert metal_bits == expected_bits, (
                f"Code 0x{code:X}: Metal 0x{metal_bits:04X}, expected 0x{expected_bits:04X}"
            )

    @pytest.mark.parametrize("scale", [0.125, 0.5, 1.0, 2.0, 4.0])
    def test_codes_0_to_7_various_scales(self, scale: float):
        """Positive FP4 codes with various scales."""
        packed = np.uint32(0x76543210)
        metal_out = run_metal_fp4_packed_scaled(packed, np.float16(scale))
        for i in range(8):
            expected = np.float16(float(FP4_E2M1_LUT[i]) * np.float16(scale))
            assert metal_out[i] == expected, (
                f"Code {i}, scale {scale}: Metal={metal_out[i]}, expected={expected}"
            )

    def test_negative_codes(self):
        """Negative FP4 codes (8-F) with scale=1."""
        packed = np.uint32(0xFEDCBA98)
        metal_out = run_metal_fp4_packed_scaled(packed, np.float16(1.0))
        expected = FP4_E2M1_LUT[8:]
        for i in range(8):
            m_bits = metal_out.view(np.uint16)[i]
            e_bits = expected.view(np.uint16)[i]
            assert m_bits == e_bits

    @pytest.mark.parametrize("seed", range(20))
    def test_random_packed_values(self, seed: int):
        """Random FP4 packed values match reference."""
        rng = np.random.default_rng(seed + 700)
        nibbles = rng.integers(0, 16, size=8, dtype=np.uint8)
        packed_val = pack_fp4_values(nibbles.tolist())
        scale = np.float16(rng.uniform(0.01, 8.0))

        metal_out = run_metal_fp4_packed_scaled(np.uint32(packed_val), scale)
        for i in range(8):
            expected = np.float16(float(FP4_E2M1_LUT[nibbles[i]]) * float(scale))
            m_bits = metal_out.view(np.uint16)[i]
            e_bits = np.array([expected]).view(np.uint16)[0]
            assert m_bits == e_bits, (
                f"Seed {seed}, nibble {i}: Metal 0x{m_bits:04X} != ref 0x{e_bits:04X}"
            )


@pytest.mark.skipif(not _check_metal_available(),
                    reason="Metal API (PyObjC) not available")
class TestMetalINT4Kernels:
    """Test Metal INT4 dequant kernels against reference implementation."""

    def test_u4_identity(self):
        """U4 with scale=1, zero=0 produces raw integer values."""
        packed = np.array([0x76543210], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        zeros = np.array([0.0], dtype=np.float16)

        metal_out = run_metal_int4_dequant(packed, scales, zeros,
                                            group_size=128, is_signed=False)
        ref_out = ref_dequant_u4_bulk(packed, scales, zeros, group_size=128)
        np.testing.assert_array_almost_equal(metal_out, ref_out, decimal=2)

    def test_s4_identity(self):
        """S4 with scale=1, zero=0 maps stored [0-7] to [-8, -1]."""
        packed = np.array([0x76543210], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        zeros = np.array([0.0], dtype=np.float16)

        metal_out = run_metal_int4_dequant(packed, scales, zeros,
                                            group_size=128, is_signed=True)
        ref_out = ref_dequant_s4_bulk(packed, scales, zeros, group_size=128)
        np.testing.assert_array_almost_equal(metal_out, ref_out, decimal=2)

    def test_u4_asymmetric(self):
        """U4 with non-zero zero_point (asymmetric quantization)."""
        packed = np.array([0x76543210], dtype=np.uint32)
        scales = np.array([0.5], dtype=np.float16)
        zeros = np.array([4.0], dtype=np.float16)

        metal_out = run_metal_int4_dequant(packed, scales, zeros,
                                            group_size=128, is_signed=False)
        ref_out = ref_dequant_u4_bulk(packed, scales, zeros, group_size=128)
        np.testing.assert_allclose(metal_out, ref_out, rtol=1e-2, atol=0.1)

    @pytest.mark.parametrize("seed", range(10))
    def test_u4_random_multigroup(self, seed: int):
        """Random U4 buffer with multiple groups."""
        rng = np.random.default_rng(42 + seed)
        group_size = 128
        num_groups = 4
        num_packed = (group_size * num_groups) // 8

        packed = rng.integers(0, 2**32, size=num_packed, dtype=np.uint32)
        scales = np.float16(rng.uniform(0.01, 2.0, size=num_groups))
        zeros = np.float16(rng.uniform(0, 15, size=num_groups))

        metal_out = run_metal_int4_dequant(packed, scales, zeros,
                                            group_size=group_size, is_signed=False)
        ref_out = ref_dequant_u4_bulk(packed, scales, zeros, group_size=group_size)
        np.testing.assert_allclose(metal_out, ref_out, rtol=1e-2, atol=0.1)

    @pytest.mark.parametrize("seed", range(10))
    def test_s4_random_multigroup(self, seed: int):
        """Random S4 buffer with multiple groups."""
        rng = np.random.default_rng(100 + seed)
        group_size = 128
        num_groups = 4
        num_packed = (group_size * num_groups) // 8

        packed = rng.integers(0, 2**32, size=num_packed, dtype=np.uint32)
        scales = np.float16(rng.uniform(0.01, 2.0, size=num_groups))
        zeros = np.float16(rng.uniform(-4, 4, size=num_groups))

        metal_out = run_metal_int4_dequant(packed, scales, zeros,
                                            group_size=group_size, is_signed=True)
        ref_out = ref_dequant_s4_bulk(packed, scales, zeros, group_size=group_size)
        np.testing.assert_allclose(metal_out, ref_out, rtol=1e-2, atol=0.1)

    @pytest.mark.parametrize("group_size", [8, 32, 64, 128, 256])
    def test_u4_various_group_sizes(self, group_size: int):
        """U4 correctness across different quantization group sizes."""
        rng = np.random.default_rng(999)
        num_elements = 512
        num_packed = num_elements // 8
        num_groups = num_elements // group_size

        packed = rng.integers(0, 2**32, size=num_packed, dtype=np.uint32)
        scales = np.float16(rng.uniform(0.1, 2.0, size=num_groups))
        zeros = np.float16(rng.uniform(0, 8, size=num_groups))

        metal_out = run_metal_int4_dequant(packed, scales, zeros,
                                            group_size=group_size, is_signed=False)
        ref_out = ref_dequant_u4_bulk(packed, scales, zeros, group_size=group_size)
        np.testing.assert_allclose(metal_out, ref_out, rtol=1e-2, atol=0.1)


# ============================================================================
# Test: E2M1 encoding properties
# ============================================================================


class TestE2M1Properties:
    """Test mathematical properties of the FP4 E2M1 encoding."""

    def test_exactly_16_representable_values(self):
        """FP4 has exactly 16 distinct bit patterns."""
        values = set()
        for code in range(16):
            val = float(dequant_fp4_scalar(code))
            if val == 0.0:
                # Distinguish +0 and -0 by sign bit
                bits = np.array([dequant_fp4_scalar(code)]).view(np.uint16)[0]
                values.add(('zero', bits >> 15))
            else:
                values.add(val)
        # 14 distinct non-zero values + positive zero + negative zero = 16
        assert len(values) == 16

    def test_exponent_bias_is_1(self):
        """E2M1 uses bias=1, so stored E=1 means actual exponent 0."""
        # E=1, M=0: 2^(1-1) * 1.0 = 1.0
        assert float(dequant_fp4_scalar(0b0010)) == 1.0
        # E=2, M=0: 2^(2-1) * 1.0 = 2.0
        assert float(dequant_fp4_scalar(0b0100)) == 2.0
        # E=3, M=0: 2^(3-1) * 1.0 = 4.0
        assert float(dequant_fp4_scalar(0b0110)) == 4.0

    def test_mantissa_adds_half(self):
        """Setting M=1 multiplies by 1.5 (adds 0.5 to implicit 1.0)."""
        # E=1: 1.0 vs 1.5
        assert float(dequant_fp4_scalar(0b0010)) == 1.0
        assert float(dequant_fp4_scalar(0b0011)) == 1.5
        # E=2: 2.0 vs 3.0
        assert float(dequant_fp4_scalar(0b0100)) == 2.0
        assert float(dequant_fp4_scalar(0b0101)) == 3.0
        # E=3: 4.0 vs 6.0
        assert float(dequant_fp4_scalar(0b0110)) == 4.0
        assert float(dequant_fp4_scalar(0b0111)) == 6.0

    def test_subnormal_is_half(self):
        """E=0, M=1 gives the subnormal value 0.5."""
        assert float(dequant_fp4_scalar(0b0001)) == 0.5
        assert float(dequant_fp4_scalar(0b1001)) == -0.5

    def test_spacing_doubles_per_binade(self):
        """Value spacing doubles with each exponent increment."""
        # Binade E=1: spacing between 1.0 and 1.5 is 0.5
        # Binade E=2: spacing between 2.0 and 3.0 is 1.0
        # Binade E=3: spacing between 4.0 and 6.0 is 2.0
        s1 = float(dequant_fp4_scalar(0b0011)) - float(dequant_fp4_scalar(0b0010))  # 0.5
        s2 = float(dequant_fp4_scalar(0b0101)) - float(dequant_fp4_scalar(0b0100))  # 1.0
        s3 = float(dequant_fp4_scalar(0b0111)) - float(dequant_fp4_scalar(0b0110))  # 2.0
        assert s1 == pytest.approx(0.5)
        assert s2 == pytest.approx(1.0)
        assert s3 == pytest.approx(2.0)
        assert s2 / s1 == pytest.approx(2.0)
        assert s3 / s2 == pytest.approx(2.0)


# ============================================================================
# Test: Packing correctness
# ============================================================================


class TestPacking:
    """Test nibble packing/unpacking consistency."""

    def test_pack_identity(self):
        """Pack and extract recovers original nibbles."""
        nibbles = [0, 1, 2, 3, 4, 5, 6, 7]
        packed = pack_fp4_values(nibbles)
        assert packed == 0x76543210

    def test_pack_all_same(self):
        """All-same nibbles pack correctly."""
        for val in range(16):
            packed = pack_fp4_values([val] * 8)
            for i in range(8):
                assert (packed >> (i * 4)) & 0xF == val

    def test_pack_max_value(self):
        """All 0xF nibbles produce 0xFFFFFFFF."""
        packed = pack_fp4_values([0xF] * 8)
        assert packed == 0xFFFFFFFF

    def test_pack_zero(self):
        """All zeros produce 0."""
        packed = pack_fp4_values([0] * 8)
        assert packed == 0

    def test_pack_fewer_than_8(self):
        """Packing fewer than 8 values fills lower nibbles only."""
        packed = pack_fp4_values([5, 10])
        assert (packed >> 0) & 0xF == 5
        assert (packed >> 4) & 0xF == 10
        # Upper nibbles should be zero
        for i in range(2, 8):
            assert (packed >> (i * 4)) & 0xF == 0

    @pytest.mark.parametrize("seed", range(50))
    def test_pack_roundtrip_random(self, seed: int):
        """Random nibbles survive pack/unpack."""
        rng = np.random.default_rng(seed + 2000)
        nibbles = rng.integers(0, 16, size=8).tolist()
        packed = pack_fp4_values(nibbles)
        for i in range(8):
            assert (packed >> (i * 4)) & 0xF == nibbles[i]


# ============================================================================
# Test: Group boundary handling
# ============================================================================


class TestGroupBoundaries:
    """Verify correct scale/zero assignment at group boundaries."""

    def test_first_element_of_each_group(self):
        """First element of each group uses that group's scale."""
        group_size = 8
        num_groups = 4
        num_packed = (group_size * num_groups) // 8

        # All values = 2 (code for 1.0 in FP4)
        packed = np.array([pack_fp4_values([2] * 8)] * num_packed, dtype=np.uint32)
        scales = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)

        result = ref_dequant_fp4_bulk(packed, scales, group_size)
        for g in range(num_groups):
            start = g * group_size
            expected = float(np.float16(1.0 * scales[g]))
            assert float(result[start]) == pytest.approx(expected, abs=0.01), (
                f"Group {g} first element: {result[start]} != {expected}"
            )

    def test_last_element_of_each_group(self):
        """Last element of each group still uses that group's scale."""
        group_size = 8
        num_groups = 4
        num_packed = (group_size * num_groups) // 8

        packed = np.array([pack_fp4_values([2] * 8)] * num_packed, dtype=np.uint32)
        scales = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)

        result = ref_dequant_fp4_bulk(packed, scales, group_size)
        for g in range(num_groups):
            end = (g + 1) * group_size - 1
            expected = float(np.float16(1.0 * scales[g]))
            assert float(result[end]) == pytest.approx(expected, abs=0.01)

    def test_group_transition(self):
        """Elements at group boundary use correct neighboring scales."""
        group_size = 8
        packed = np.array([
            pack_fp4_values([2] * 8),  # indices 0-7 -> group 0 (scale 1.0)
            pack_fp4_values([2] * 8),  # indices 8-15 -> group 1 (scale 10.0)
        ], dtype=np.uint32)
        scales = np.array([1.0, 10.0], dtype=np.float16)

        result = ref_dequant_fp4_bulk(packed, scales, group_size)
        # Last in group 0: 1.0 * 1.0 = 1.0
        assert float(result[7]) == pytest.approx(1.0, abs=0.01)
        # First in group 1: 1.0 * 10.0 = 10.0
        assert float(result[8]) == pytest.approx(10.0, abs=0.1)
