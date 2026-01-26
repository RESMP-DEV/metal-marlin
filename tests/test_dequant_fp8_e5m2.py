"""Numerical validation of FP8 E5M2 → FP16 dequantization.

Tests the bitwise E5M2 dequant implementation against a reference (numpy)
implementation. Validates all 256 codes including normals, subnormals,
zeros, infinities, and NaNs.

FP8 E5M2 format: [1 sign][5 exponent (bias=15)][2 mantissa]
Same exponent range and bias as FP16, making conversion a pure bit extension.

Usage:
    cd metal_marlin
    uv run pytest tests/test_dequant_fp8_e5m2.py -v
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

# ============================================================================
# Reference implementation (pure Python, matching Metal bit-ops exactly)
# ============================================================================


def fp16_bits_to_float(bits: int) -> np.float16:
    """Convert a 16-bit integer to FP16 float value."""
    packed = struct.pack('<H', bits & 0xFFFF)
    return np.frombuffer(packed, dtype=np.float16)[0]


def float_to_fp16_bits(val: np.float16) -> int:
    """Convert FP16 float to its 16-bit integer representation."""
    return int(np.array([val]).view(np.uint16)[0])


def ref_dequant_fp8_e5m2(code: int) -> np.float16:
    """Reference FP8 E5M2 → FP16 dequantization via bitwise field extension.

    This is the Python equivalent of the Metal dequant_fp8_e5m2 function.
    Since E5M2 and FP16 share the same exponent width (5) and bias (15),
    conversion is a direct bit-field rearrangement.

    FP8 E5M2: [S:1][E:5][M:2]
      Normal (0 < E < 31): (-1)^S * 2^(E-15) * (1 + M/4)
      Subnormal (E=0, M>0): (-1)^S * 2^(-14) * (M/4)
      Zero (E=0, M=0): +/- 0.0
      Infinity (E=31, M=0): +/- Inf
      NaN (E=31, M!=0): NaN

    FP16 equivalent: place S at bit 15, E at bits [14:10], M<<8 at bits [9:0].
    """
    assert 0 <= code <= 255
    S = (code >> 7) & 1
    E = (code >> 2) & 0x1F
    M = code & 0x3

    # Direct field placement: same exponent, mantissa left-aligned
    mant16 = M << 8
    fp16_bits = (S << 15) | (E << 10) | mant16
    return fp16_bits_to_float(fp16_bits)


def ref_dequant_fp8_e5m2_value(code: int) -> float:
    """Compute the mathematical value of an FP8 E5M2 code.

    This computes the value from the format definition (not bit-casting),
    serving as an independent cross-check.
    """
    S = (code >> 7) & 1
    E = (code >> 2) & 0x1F
    M = code & 0x3
    sign = (-1.0) ** S

    if E == 0 and M == 0:
        return 0.0 * sign  # +/- 0.0
    elif E == 0:
        # Subnormal: 2^(1-bias) * (M / 2^mantissa_bits) = 2^(-14) * M/4
        return sign * (2.0**-14) * (M / 4.0)
    elif E == 31 and M == 0:
        return sign * float('inf')
    elif E == 31:
        return float('nan')
    else:
        # Normal: 2^(E-15) * (1 + M/4)
        return sign * (2.0**(E - 15)) * (1.0 + M / 4.0)


def ref_dequant_fp8_e5m2_x4(packed_u32: int, scale: float = 1.0) -> list[np.float16]:
    """Dequantize 4 FP8 E5M2 values from a packed uint32 with scale."""
    results = []
    for i in range(4):
        byte_val = (packed_u32 >> (i * 8)) & 0xFF
        val = ref_dequant_fp8_e5m2(byte_val)
        results.append(np.float16(float(val) * scale))
    return results


# ============================================================================
# NVIDIA FP8 E5M2 reference table
# ============================================================================
# Generate the complete reference table for all 256 codes.
# E5M2 representable values (positive, non-special):
#   Subnormals (E=0): M/4 * 2^(-14) for M in {1,2,3}
#     = {2^(-16), 2^(-15), 3*2^(-16)}
#   Normals: 2^(E-15) * (1 + M/4) for E in [1,30], M in [0,3]
#   Infinity: E=31, M=0
#   NaN: E=31, M in {1,2,3}

def build_fp8_e5m2_reference_table() -> np.ndarray:
    """Build the complete 256-entry FP8 E5M2 → FP16 reference table.

    Returns an array of fp16 bit patterns (uint16) for each code [0..255].
    This uses the mathematical definition, not bit-casting, as ground truth.
    """
    table = np.zeros(256, dtype=np.uint16)
    for code in range(256):
        S = (code >> 7) & 1
        E = (code >> 2) & 0x1F
        M = code & 0x3

        if E == 31 and M != 0:
            # NaN: preserve sign, set quiet NaN in FP16
            # Our Metal shader does: sign | exp(31) | mant(M<<8)
            # which gives a signaling NaN with payload.
            # For comparison, we match the Metal behavior exactly.
            table[code] = (S << 15) | (0x1F << 10) | (M << 8)
        elif E == 31 and M == 0:
            # Infinity
            table[code] = (S << 15) | (0x1F << 10)
        elif E == 0 and M == 0:
            # Zero
            table[code] = S << 15
        elif E == 0:
            # Subnormal: same representation in FP16 (shared bias)
            # FP8 subnormal = 2^(-14) * M/4 = M * 2^(-16)
            # FP16 subnormal = 2^(-14) * frac, where frac = mantissa/1024
            # M * 2^(-16) = 2^(-14) * M/4 = 2^(-14) * (M*256)/1024
            # So fp16_mant = M * 256 = M << 8, exp = 0
            table[code] = (S << 15) | (M << 8)
        else:
            # Normal: same exp, mantissa left-aligned
            table[code] = (S << 15) | (E << 10) | (M << 8)

    return table


FP8_E5M2_REFERENCE = build_fp8_e5m2_reference_table()


# ============================================================================
# Tests: Bitwise construction correctness (pure Python)
# ============================================================================


class TestFP8E5M2BitwiseConstruction:
    """Verify bitwise FP8 E5M2 → FP16 construction matches mathematical values."""

    def test_all_256_codes_bit_exact(self):
        """Every FP8 E5M2 code [0x00..0xFF] matches the reference table."""
        for code in range(256):
            result = ref_dequant_fp8_e5m2(code)
            result_bits = float_to_fp16_bits(result)
            expected_bits = int(FP8_E5M2_REFERENCE[code])
            assert result_bits == expected_bits, (
                f"Code 0x{code:02X}: got 0x{result_bits:04X}, "
                f"expected 0x{expected_bits:04X}"
            )

    def test_positive_zero(self):
        """Code 0x00 is +0.0."""
        val = ref_dequant_fp8_e5m2(0x00)
        assert val == np.float16(0.0)
        assert float_to_fp16_bits(val) == 0x0000

    def test_negative_zero(self):
        """Code 0x80 is -0.0."""
        val = ref_dequant_fp8_e5m2(0x80)
        assert float_to_fp16_bits(val) == 0x8000

    def test_positive_one(self):
        """Code for +1.0: S=0, E=15, M=0 → byte = 0b0_01111_00 = 0x3C."""
        val = ref_dequant_fp8_e5m2(0x3C)
        assert val == np.float16(1.0), f"Got {val}"

    def test_negative_one(self):
        """Code for -1.0: S=1, E=15, M=0 → byte = 0b1_01111_00 = 0xBC."""
        val = ref_dequant_fp8_e5m2(0xBC)
        assert val == np.float16(-1.0), f"Got {val}"

    def test_positive_infinity(self):
        """+Inf: S=0, E=31, M=0 → byte = 0b0_11111_00 = 0x7C."""
        val = ref_dequant_fp8_e5m2(0x7C)
        assert np.isinf(val) and val > 0

    def test_negative_infinity(self):
        """-Inf: S=1, E=31, M=0 → byte = 0b1_11111_00 = 0xFC."""
        val = ref_dequant_fp8_e5m2(0xFC)
        assert np.isinf(val) and val < 0

    def test_nan_codes(self):
        """E=31, M!=0 are NaN."""
        # Positive NaN codes: 0x7D, 0x7E, 0x7F
        for code in [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]:
            val = ref_dequant_fp8_e5m2(code)
            assert np.isnan(val), f"Code 0x{code:02X} should be NaN, got {val}"

    def test_subnormals_positive(self):
        """Positive subnormals: E=0, M in {1,2,3}."""
        # M=1: 2^(-16) = 1.52587890625e-05
        # M=2: 2^(-15) = 3.0517578125e-05
        # M=3: 3*2^(-16) = 4.57763671875e-05
        expected_values = [2.0**-16, 2.0**-15, 3.0 * 2.0**-16]
        for m, exp_val in enumerate(expected_values, start=1):
            code = m  # S=0, E=0, M=m
            val = float(ref_dequant_fp8_e5m2(code))
            assert abs(val - exp_val) / exp_val < 1e-3, (
                f"Subnormal M={m}: got {val}, expected {exp_val}"
            )

    def test_subnormals_negative(self):
        """Negative subnormals: S=1, E=0, M in {1,2,3}."""
        expected_values = [-(2.0**-16), -(2.0**-15), -(3.0 * 2.0**-16)]
        for m, exp_val in enumerate(expected_values, start=1):
            code = 0x80 | m  # S=1, E=0, M=m
            val = float(ref_dequant_fp8_e5m2(code))
            assert abs(val - exp_val) / abs(exp_val) < 1e-3, (
                f"Negative subnormal M={m}: got {val}, expected {exp_val}"
            )

    def test_max_normal_positive(self):
        """Largest positive normal: S=0, E=30, M=3 → 2^15 * (1+3/4) = 57344."""
        code = (30 << 2) | 3  # 0b0_11110_11 = 0x7B
        val = ref_dequant_fp8_e5m2(code)
        expected = np.float16(57344.0)
        assert val == expected, f"Got {val}, expected {expected}"

    def test_min_normal_positive(self):
        """Smallest positive normal: S=0, E=1, M=0 → 2^(-14)."""
        code = 1 << 2  # 0b0_00001_00 = 0x04
        val = ref_dequant_fp8_e5m2(code)
        expected = np.float16(2.0**-14)
        assert val == expected, f"Got {val}, expected {expected}"

    def test_mathematical_values_match(self):
        """Cross-check: bitwise construction matches mathematical computation."""
        for code in range(256):
            bitwise_val = ref_dequant_fp8_e5m2(code)
            math_val = ref_dequant_fp8_e5m2_value(code)

            if np.isnan(bitwise_val):
                assert np.isnan(math_val) or math_val != math_val
                continue

            # Compare as float (handles +-0, inf correctly)
            bitwise_f = float(bitwise_val)
            if np.isinf(bitwise_val):
                assert np.isinf(math_val) and np.sign(bitwise_f) == np.sign(math_val)
            elif bitwise_f == 0.0:
                assert math_val == 0.0
            else:
                assert abs(bitwise_f - math_val) / abs(math_val) < 1e-3, (
                    f"Code 0x{code:02X}: bitwise={bitwise_f}, math={math_val}"
                )


class TestFP8E5M2PackedDequant:
    """Test packed (x4) dequantization."""

    def test_x4_identity(self):
        """Pack 4 known values and dequant with scale=1."""
        # Codes: +0.0 (0x00), +1.0 (0x3C), -1.0 (0xBC), +Inf (0x7C)
        packed = 0x00 | (0x3C << 8) | (0xBC << 16) | (0x7C << 24)
        results = ref_dequant_fp8_e5m2_x4(packed, scale=1.0)

        assert results[0] == np.float16(0.0)
        assert results[1] == np.float16(1.0)
        assert results[2] == np.float16(-1.0)
        assert np.isinf(results[3]) and results[3] > 0

    def test_x4_with_scale(self):
        """Pack normal values and apply scale=0.5."""
        # +1.0 (0x3C), +2.0 (0x40), +0.5 (0x38), +1.5 (0x3E)
        packed = 0x3C | (0x40 << 8) | (0x38 << 16) | (0x3E << 24)
        results = ref_dequant_fp8_e5m2_x4(packed, scale=0.5)

        expected = [0.5, 1.0, 0.25, 0.75]
        for i, exp in enumerate(expected):
            assert abs(float(results[i]) - exp) < 1e-2, (
                f"Byte {i}: got {results[i]}, expected {exp}"
            )

    @pytest.mark.parametrize("seed", range(20))
    def test_random_packed(self, seed):
        """Random packed E5M2 values match element-wise dequant."""
        rng = np.random.default_rng(seed + 1000)
        bytes_val = rng.integers(0, 256, size=4, dtype=np.uint8)
        packed = int(bytes_val[0]) | (int(bytes_val[1]) << 8) | \
                 (int(bytes_val[2]) << 16) | (int(bytes_val[3]) << 24)

        scale = float(np.float16(rng.uniform(0.01, 4.0)))
        results = ref_dequant_fp8_e5m2_x4(packed, scale=scale)

        for i in range(4):
            expected = ref_dequant_fp8_e5m2(int(bytes_val[i]))
            expected_scaled = np.float16(float(expected) * scale)

            r_bits = float_to_fp16_bits(results[i])
            e_bits = float_to_fp16_bits(expected_scaled)

            # NaN comparison: both should be NaN
            if np.isnan(results[i]):
                assert np.isnan(expected_scaled), (
                    f"Byte {i} (0x{bytes_val[i]:02X}): result is NaN but "
                    f"expected {expected_scaled}"
                )
            else:
                assert r_bits == e_bits, (
                    f"Byte {i} (0x{bytes_val[i]:02X}): got 0x{r_bits:04X} "
                    f"({results[i]}), expected 0x{e_bits:04X} ({expected_scaled})"
                )


# ============================================================================
# Metal kernel execution
# ============================================================================


def _check_metal_available() -> bool:
    """Check if Metal API is available via PyObjC."""
    try:
        import Metal  # noqa: F401
        return True
    except ImportError:
        return False


def _read_metal_buffer(buf, nbytes: int) -> bytes:
    """Read bytes from a Metal buffer."""
    import ctypes

    contents = buf.contents()
    if isinstance(contents, int):
        arr = (ctypes.c_char * nbytes).from_address(contents)
        return bytes(arr)
    elif isinstance(contents, memoryview):
        return bytes(contents[:nbytes])
    elif hasattr(contents, '__getitem__'):
        raw = b''.join(contents[i] for i in range(nbytes))
        return raw
    else:
        ptr = ctypes.cast(contents, ctypes.POINTER(ctypes.c_char * nbytes))
        return bytes(ptr.contents)


def run_metal_fp8_e5m2_all_codes() -> np.ndarray:
    """Run the test_fp8_e5m2_all_codes Metal kernel, returns 256 half values."""
    from pathlib import Path

    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device is not None, "No Metal device found"

    shader_path = Path(__file__).parent.parent / "src" / "dequant_fp8.metal"
    source = shader_path.read_text()
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    assert err is None, f"Metal compile error: {err}"

    func = library.newFunctionWithName_("test_fp8_e5m2_all_codes")
    assert func is not None, "Kernel 'test_fp8_e5m2_all_codes' not found"

    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    assert err is None, f"Pipeline error: {err}"

    # Output buffer: 256 half values = 512 bytes
    output_size = 256 * 2
    buf_output = device.newBufferWithLength_options_(
        output_size, Metal.MTLResourceStorageModeShared)

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 0)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, output_size)
    return np.frombuffer(raw, dtype=np.float16).copy()


def run_metal_fp8_e5m2_packed_scaled(packed: np.uint32,
                                      scale: np.float16) -> np.ndarray:
    """Run test_fp8_e5m2_packed_scaled kernel: dequant 4 E5M2 values."""
    from pathlib import Path

    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device is not None

    shader_path = Path(__file__).parent.parent / "src" / "dequant_fp8.metal"
    source = shader_path.read_text()
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    assert err is None, f"Metal compile error: {err}"

    func = library.newFunctionWithName_("test_fp8_e5m2_packed_scaled")
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
        8, Metal.MTLResourceStorageModeShared)  # 4 halfs = 8 bytes

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 2)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(1, 1, 1))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, 8)
    return np.frombuffer(raw, dtype=np.float16).copy()


# ============================================================================
# Metal kernel tests
# ============================================================================


@pytest.mark.skipif(not _check_metal_available(),
                    reason="Metal API (PyObjC) not available")
class TestMetalFP8E5M2Kernel:
    """Run Metal FP8 E5M2 dequant kernels and verify against reference."""

    def test_all_256_codes_match_reference(self):
        """Metal test_fp8_e5m2_all_codes produces bit-exact reference values."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        assert len(metal_out) == 256

        metal_bits = metal_out.view(np.uint16)
        mismatches = []
        for code in range(256):
            expected_bits = int(FP8_E5M2_REFERENCE[code])
            got_bits = int(metal_bits[code])
            if got_bits != expected_bits:
                mismatches.append(
                    f"  Code 0x{code:02X}: Metal=0x{got_bits:04X} "
                    f"({metal_out[code]}), expected=0x{expected_bits:04X} "
                    f"({fp16_bits_to_float(expected_bits)})"
                )

        assert not mismatches, (
            f"{len(mismatches)} mismatches out of 256 codes:\n"
            + "\n".join(mismatches[:20])
        )

    def test_zero_codes(self):
        """Verify +0 and -0."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        bits = metal_out.view(np.uint16)
        assert bits[0x00] == 0x0000, f"+0: got 0x{bits[0x00]:04X}"
        assert bits[0x80] == 0x8000, f"-0: got 0x{bits[0x80]:04X}"

    def test_infinity_codes(self):
        """Verify +Inf and -Inf."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        assert np.isinf(metal_out[0x7C]) and metal_out[0x7C] > 0
        assert np.isinf(metal_out[0xFC]) and metal_out[0xFC] < 0

    def test_nan_codes(self):
        """Verify NaN codes (E=31, M!=0)."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        for code in [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]:
            assert np.isnan(metal_out[code]), (
                f"Code 0x{code:02X} should be NaN, got {metal_out[code]}"
            )

    def test_one_positive(self):
        """+1.0 at code 0x3C."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        assert metal_out[0x3C] == np.float16(1.0)

    def test_one_negative(self):
        """-1.0 at code 0xBC."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        assert metal_out[0xBC] == np.float16(-1.0)

    def test_packed_scale_1(self):
        """Packed codes with scale=1.0."""
        # +0.0, +1.0, -1.0, +2.0
        packed = np.uint32(0x00 | (0x3C << 8) | (0xBC << 16) | (0x40 << 24))
        metal_out = run_metal_fp8_e5m2_packed_scaled(packed, np.float16(1.0))

        assert metal_out[0] == np.float16(0.0)
        assert metal_out[1] == np.float16(1.0)
        assert metal_out[2] == np.float16(-1.0)
        assert metal_out[3] == np.float16(2.0)

    def test_packed_scale_half(self):
        """Packed codes with scale=0.5."""
        # +2.0 (0x40), +4.0 (0x44), +0.5 (0x38), +1.0 (0x3C)
        packed = np.uint32(0x40 | (0x44 << 8) | (0x38 << 16) | (0x3C << 24))
        metal_out = run_metal_fp8_e5m2_packed_scaled(packed, np.float16(0.5))

        expected = [1.0, 2.0, 0.25, 0.5]
        for i, exp in enumerate(expected):
            assert abs(float(metal_out[i]) - exp) < 1e-2, (
                f"Byte {i}: got {metal_out[i]}, expected {exp}"
            )

    @pytest.mark.parametrize("seed", range(20))
    def test_random_packed_e5m2(self, seed):
        """Random packed E5M2 values match element-wise reference."""
        rng = np.random.default_rng(seed + 2000)

        # Avoid NaN codes for clean numerical comparison
        valid_codes = [c for c in range(256)
                       if not ((c >> 2) & 0x1F) == 31 or (c & 0x3) == 0]
        bytes_val = rng.choice(valid_codes, size=4).astype(np.uint8)
        packed_val = (int(bytes_val[0]) | (int(bytes_val[1]) << 8) |
                      (int(bytes_val[2]) << 16) | (int(bytes_val[3]) << 24))

        scale = np.float16(rng.uniform(0.01, 4.0))
        metal_out = run_metal_fp8_e5m2_packed_scaled(np.uint32(packed_val), scale)
        ref_out = ref_dequant_fp8_e5m2_x4(packed_val, scale=float(scale))

        for i in range(4):
            m_bits = int(metal_out.view(np.uint16)[i])
            e_bits = float_to_fp16_bits(ref_out[i])

            if np.isnan(metal_out[i]):
                assert np.isnan(ref_out[i])
            elif np.isinf(metal_out[i]):
                assert np.isinf(ref_out[i])
            else:
                assert m_bits == e_bits, (
                    f"Seed {seed}, byte {i} (code 0x{bytes_val[i]:02X}): "
                    f"Metal=0x{m_bits:04X} ({metal_out[i]}), "
                    f"ref=0x{e_bits:04X} ({ref_out[i]})"
                )


# ============================================================================
# Cross-validation: E5M2 vs NVIDIA reference values
# ============================================================================


class TestFP8E5M2NvidiaReference:
    """Cross-validate against known NVIDIA FP8 E5M2 values.

    These are spot-checked values from the NVIDIA FP8 specification
    (IEEE 754-like E5M2 variant used in H100/H200 Transformer Engine).
    """

    # Key reference values from NVIDIA's FP8 spec:
    # Format: (fp8_code, expected_fp16_value)
    NVIDIA_REFERENCE_VALUES = [
        # Zeros
        (0x00, 0.0),
        (0x80, -0.0),
        # Powers of 2
        (0x3C, 1.0),       # E=15, M=0
        (0x40, 2.0),       # E=16, M=0
        (0x44, 4.0),       # E=17, M=0
        (0x48, 8.0),       # E=18, M=0
        (0x38, 0.5),       # E=14, M=0
        (0x34, 0.25),      # E=13, M=0
        (0x30, 0.125),     # E=12, M=0
        # Non-power-of-2 normals
        (0x3D, 1.25),      # E=15, M=1: 2^0 * (1 + 1/4)
        (0x3E, 1.5),       # E=15, M=2: 2^0 * (1 + 2/4)
        (0x3F, 1.75),      # E=15, M=3: 2^0 * (1 + 3/4)
        (0x41, 2.5),       # E=16, M=1: 2^1 * (1 + 1/4)
        (0x42, 3.0),       # E=16, M=2: 2^1 * (1 + 2/4)
        (0x43, 3.5),       # E=16, M=3: 2^1 * (1 + 3/4)
        # Max normal
        (0x7B, 57344.0),   # E=30, M=3: 2^15 * 1.75
        # Min normal
        (0x04, 2.0**-14),  # E=1, M=0: 2^(-14)
        # Subnormals
        (0x01, 2.0**-16),  # E=0, M=1: 2^(-14) * 1/4
        (0x02, 2.0**-15),  # E=0, M=2: 2^(-14) * 2/4
        (0x03, 3.0 * 2.0**-16),  # E=0, M=3: 2^(-14) * 3/4
        # Negative normals
        (0xBC, -1.0),
        (0xC0, -2.0),
        (0xBF, -1.75),
    ]

    @pytest.mark.parametrize("code,expected", NVIDIA_REFERENCE_VALUES,
                             ids=[f"0x{c:02X}" for c, _ in NVIDIA_REFERENCE_VALUES])
    def test_nvidia_reference_value(self, code: int, expected: float):
        """Each reference value matches the dequantized output."""
        result = ref_dequant_fp8_e5m2(code)
        result_f = float(result)

        if expected == 0.0:
            assert result_f == 0.0
            # Check sign for -0
            if code & 0x80:
                assert float_to_fp16_bits(result) == 0x8000
        else:
            assert abs(result_f - expected) / abs(expected) < 1e-3, (
                f"Code 0x{code:02X}: got {result_f}, expected {expected}"
            )

    def test_symmetry(self):
        """Positive and negative codes produce values with opposite signs."""
        for code in range(128):
            pos_val = ref_dequant_fp8_e5m2(code)
            neg_val = ref_dequant_fp8_e5m2(code | 0x80)

            if np.isnan(pos_val):
                assert np.isnan(neg_val)
                continue

            pos_bits = float_to_fp16_bits(pos_val)
            neg_bits = float_to_fp16_bits(neg_val)

            # Negative should be positive with sign bit flipped
            assert neg_bits == (pos_bits | 0x8000), (
                f"Code 0x{code:02X}: pos=0x{pos_bits:04X}, "
                f"neg=0x{neg_bits:04X}, expected 0x{pos_bits | 0x8000:04X}"
            )

    def test_monotonicity_positive_normals(self):
        """Positive normal codes are monotonically increasing."""
        prev = -float('inf')
        for code in range(4, 0x7C):  # E=1,M=0 to E=30,M=3
            val = float(ref_dequant_fp8_e5m2(code))
            assert val > prev, (
                f"Non-monotonic at code 0x{code:02X}: {val} <= {prev}"
            )
            prev = val

    def test_subnormals_less_than_min_normal(self):
        """All subnormals are smaller than the minimum normal."""
        min_normal = float(ref_dequant_fp8_e5m2(0x04))  # E=1, M=0
        for m in range(1, 4):
            sub_val = float(ref_dequant_fp8_e5m2(m))
            assert sub_val < min_normal, (
                f"Subnormal M={m} ({sub_val}) >= min_normal ({min_normal})"
            )
