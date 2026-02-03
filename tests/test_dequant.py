"""Comprehensive unit tests for all dequantization functions (Metal).

Validates dequantization at multiple levels:
  1. Pure-Python bitwise construction (reference correctness)
  2. Magic number trick simulation (INT4 algorithm verification)
  3. Metal kernel execution via PyObjC (GPU correctness)

Covered formats:
  - FP4 E2M1 (NVIDIA FP4)
  - INT4 unsigned (U4) and signed (S4)
  - FP8 E5M2
  - Sub-4-bit: INT2, INT3, NF2, NF3

Usage:
    cd metal_marlin
    uv run pytest tests/test_dequant.py -v
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Add metal_marlin package to path for sub4bit imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None  # type: ignore[assignment]


# ============================================================================
# FP4 E2M1 reference values (ground truth)
# ============================================================================

FP4_E2M1_LUT = np.array(
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
    dtype=np.float16,
)

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
    return np.frombuffer(struct.pack("<H", bits & 0xFFFF), dtype=np.float16)[0]


def float_to_fp16_bits(val: np.float16) -> int:
    """Convert FP16 float to its 16-bit integer representation."""
    return int(np.array([val]).view(np.uint16)[0])


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


def ref_dequant_u4_bulk(
    packed_u32: np.ndarray, scales: np.ndarray, zeros: np.ndarray, group_size: int
) -> np.ndarray:
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
            output[base_idx + nibble] = np.float16((np.float16(val) - zero_point) * scale)
    return output


def ref_dequant_s4_bulk(
    packed_u32: np.ndarray, scales: np.ndarray, zeros: np.ndarray, group_size: int
) -> np.ndarray:
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


def ref_dequant_fp4_bulk(packed_u32: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
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
# FP8 E5M2 reference implementation
# ============================================================================


def ref_dequant_fp8_e5m2(code: int) -> np.float16:
    """Reference FP8 E5M2 -> FP16 dequantization via bitwise field extension.

    FP8 E5M2: [S:1][E:5][M:2]
      Normal (0 < E < 31): (-1)^S * 2^(E-15) * (1 + M/4)
      Subnormal (E=0, M>0): (-1)^S * 2^(-14) * (M/4)
      Zero (E=0, M=0): +/- 0.0
      Infinity (E=31, M=0): +/- Inf
      NaN (E=31, M!=0): NaN
    """
    assert 0 <= code <= 255
    S = (code >> 7) & 1
    E = (code >> 2) & 0x1F
    M = code & 0x3

    # Direct field placement: same exponent, mantissa left-aligned
    mant16 = M << 8
    fp16_bits = (S << 15) | (E << 10) | mant16
    return fp16_from_bits(fp16_bits)


def ref_dequant_fp8_e5m2_value(code: int) -> float:
    """Compute the mathematical value of an FP8 E5M2 code."""
    S = (code >> 7) & 1
    E = (code >> 2) & 0x1F
    M = code & 0x3
    sign = (-1.0) ** S

    if E == 0 and M == 0:
        return 0.0 * sign
    elif E == 0:
        return sign * (2.0**-14) * (M / 4.0)
    elif E == 31 and M == 0:
        return sign * float("inf")
    elif E == 31:
        return float("nan")
    else:
        return sign * (2.0 ** (E - 15)) * (1.0 + M / 4.0)


def ref_dequant_fp8_e5m2_x4(packed_u32: int, scale: float = 1.0) -> list[np.float16]:
    """Dequantize 4 FP8 E5M2 values from a packed uint32 with scale."""
    results = []
    for i in range(4):
        byte_val = (packed_u32 >> (i * 8)) & 0xFF
        val = ref_dequant_fp8_e5m2(byte_val)
        results.append(np.float16(float(val) * scale))
    return results


def build_fp8_e5m2_reference_table() -> np.ndarray:
    """Build the complete 256-entry FP8 E5M2 -> FP16 reference table."""
    table = np.zeros(256, dtype=np.uint16)
    for code in range(256):
        S = (code >> 7) & 1
        E = (code >> 2) & 0x1F
        M = code & 0x3

        if E == 31 and M != 0:
            table[code] = (S << 15) | (0x1F << 10) | (M << 8)
        elif E == 31 and M == 0:
            table[code] = (S << 15) | (0x1F << 10)
        elif E == 0 and M == 0:
            table[code] = S << 15
        elif E == 0:
            table[code] = (S << 15) | (M << 8)
        else:
            table[code] = (S << 15) | (E << 10) | (M << 8)

    return table


FP8_E5M2_REFERENCE = build_fp8_e5m2_reference_table()


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
    elif hasattr(contents, "__getitem__"):
        return b"".join(contents[i] for i in range(nbytes))
    else:
        ptr = ctypes.cast(contents, ctypes.POINTER(ctypes.c_char * nbytes))
        return bytes(ptr.contents)


def _compile_dequant_shader():
    """Compile the dequant.metal shader and return (device, library)."""
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


def _compile_fp8_shader():
    """Compile the dequant_fp8.metal shader and return (device, library)."""
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device is not None, "No Metal device found"

    shader_path = Path(__file__).parent.parent / "src" / "dequant_fp8.metal"
    source = shader_path.read_text()
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    assert err is None, f"Metal compile error: {err}"
    return device, library


def _compile_bf16_shader():
    """Compile the bf16_kernels.metal shader and return (device, library)."""
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device is not None, "No Metal device found"

    shader_path = Path(__file__).parent.parent / "src" / "bf16_kernels.metal"
    source = shader_path.read_text()

    # Handle include
    include_token = '#include "bf16_compat.metal"'
    if include_token in source:
        include_path = shader_path.parent / "bf16_compat.metal"
        if not include_path.exists():
            raise FileNotFoundError(f"Missing Metal include: {include_path}")
        include_source = include_path.read_text()
        source = source.replace(include_token, include_source)

    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    assert err is None, f"Metal compile error: {err}"
    return device, library


def _bf16_rne_bits(val: np.float32) -> np.uint16:
    bits = np.array([val], dtype=np.float32).view(np.uint32)[0]
    exp_bits = (bits >> 23) & 0xFF
    mantissa = bits & 0x007FFFFF

    if exp_bits == 0xFF:
        if mantissa != 0:
            return np.uint16((bits >> 16) | 0x0040)
        return np.uint16(bits >> 16)

    rounding = np.uint32(0x8000 + ((bits >> 16) & 1))
    rounded = np.uint32(bits + rounding)
    return np.uint16(rounded >> 16)


def _bf16_roundtrip(val: np.float32) -> np.float32:
    bf16_bits = np.uint32(_bf16_rne_bits(val)) << 16
    return np.array([bf16_bits], dtype=np.uint32).view(np.float32)[0]


def run_metal_bf16_direct_roundtrip(values: np.ndarray) -> np.ndarray:
    """Run bf16_roundtrip_direct_float8 kernel on Metal hardware."""
    import Metal

    values = np.asarray(values, dtype=np.float32)
    assert values.size % 8 == 0, "Input length must be a multiple of 8"

    device, library = _compile_bf16_shader()
    func = library.newFunctionWithName_("bf16_roundtrip_direct_float8")
    assert func is not None
    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    assert err is None

    input_bytes = values.tobytes()
    buf_input = device.newBufferWithBytes_length_options_(
        input_bytes, len(input_bytes), Metal.MTLResourceStorageModeShared
    )

    output_size = values.nbytes
    buf_output = device.newBufferWithLength_options_(
        output_size, Metal.MTLResourceStorageModeShared
    )

    scratch_size = values.size * 2
    buf_scratch = device.newBufferWithLength_options_(
        scratch_size, Metal.MTLResourceStorageModeShared
    )

    buf_num_elements = device.newBufferWithBytes_length_options_(
        np.array([values.size], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_input, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_scratch, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_num_elements, 0, 3)

    num_threads = values.size // 8
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(num_threads, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, output_size)
    return np.frombuffer(raw, dtype=np.float32).copy()


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
        output_size, Metal.MTLResourceStorageModeShared
    )

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 0)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(16, 1, 1)
    )
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
        packed_bytes, 4, Metal.MTLResourceStorageModeShared
    )
    buf_scale = device.newBufferWithBytes_length_options_(
        scale_bytes, 2, Metal.MTLResourceStorageModeShared
    )
    buf_output = device.newBufferWithLength_options_(16, Metal.MTLResourceStorageModeShared)

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 2)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, 16)
    return np.frombuffer(raw, dtype=np.float16).copy()


def run_metal_int4_dequant(
    packed_weights: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int,
    is_signed: bool,
) -> np.ndarray:
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
        packed_weights.tobytes(),
        packed_weights.nbytes,
        Metal.MTLResourceStorageModeShared,
    )
    buf_scales = device.newBufferWithBytes_length_options_(
        scales.tobytes(), scales.nbytes, Metal.MTLResourceStorageModeShared
    )
    buf_zeros = device.newBufferWithBytes_length_options_(
        zeros.tobytes(), zeros.nbytes, Metal.MTLResourceStorageModeShared
    )

    output_size = num_elements * 2
    buf_output = device.newBufferWithLength_options_(
        output_size, Metal.MTLResourceStorageModeShared
    )

    buf_num_elements = device.newBufferWithBytes_length_options_(
        np.array([num_elements], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    buf_group_size = device.newBufferWithBytes_length_options_(
        np.array([group_size], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )
    buf_is_signed = device.newBufferWithBytes_length_options_(
        np.array([1 if is_signed else 0], dtype=np.uint32).tobytes(),
        4,
        Metal.MTLResourceStorageModeShared,
    )

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
        Metal.MTLSizeMake(threads_per_group, 1, 1),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, output_size)
    return np.frombuffer(raw, dtype=np.float16).copy()


def run_metal_fp8_e5m2_all_codes() -> np.ndarray:
    """Run the test_fp8_e5m2_all_codes Metal kernel, returns 256 half values."""
    import Metal

    device, library = _compile_fp8_shader()

    func = library.newFunctionWithName_("test_fp8_e5m2_all_codes")
    assert func is not None, "Kernel 'test_fp8_e5m2_all_codes' not found"

    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    assert err is None, f"Pipeline error: {err}"

    output_size = 256 * 2
    buf_output = device.newBufferWithLength_options_(
        output_size, Metal.MTLResourceStorageModeShared
    )

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 0)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(256, 1, 1)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, output_size)
    return np.frombuffer(raw, dtype=np.float16).copy()


def run_metal_fp8_e5m2_packed_scaled(packed: np.uint32, scale: np.float16) -> np.ndarray:
    """Run test_fp8_e5m2_packed_scaled kernel: dequant 4 E5M2 values."""
    import Metal

    device, library = _compile_fp8_shader()

    func = library.newFunctionWithName_("test_fp8_e5m2_packed_scaled")
    assert func is not None

    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    assert err is None

    packed_bytes = np.array([packed], dtype=np.uint32).tobytes()
    scale_bytes = np.array([scale], dtype=np.float16).tobytes()

    buf_packed = device.newBufferWithBytes_length_options_(
        packed_bytes, 4, Metal.MTLResourceStorageModeShared
    )
    buf_scale = device.newBufferWithBytes_length_options_(
        scale_bytes, 2, Metal.MTLResourceStorageModeShared
    )
    buf_output = device.newBufferWithLength_options_(8, Metal.MTLResourceStorageModeShared)

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_output, 0, 2)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_output, 8)
    return np.frombuffer(raw, dtype=np.float16).copy()


# ============================================================================
# Test: FP4 E2M1 exact values
# ============================================================================


class TestFP4ExactValues:
    """FP4 E2M1 has exactly 16 representable values. Verify all of them."""

    @pytest.mark.parametrize(
        "code,expected_val",
        [
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
        ],
    )
    def test_fp4_code_produces_expected_value(self, code: int, expected_val: float):
        """Each FP4 code maps to exactly one value via bitwise construction."""
        result = dequant_fp4_scalar(code)
        if expected_val == 0.0:
            result_bits = np.array([result]).view(np.uint16)[0]
            expected_bits = FP4_E2M1_BITS[code]
            assert result_bits == expected_bits
        else:
            assert float(result) == pytest.approx(expected_val, abs=1e-5)

    @pytest.mark.parametrize("code", range(16))
    def test_fp4_bit_pattern_exact(self, code: int):
        """Verify the exact FP16 bit pattern for each FP4 code."""
        result = dequant_fp4_scalar(code)
        result_bits = np.array([result]).view(np.uint16)[0]
        expected_bits = FP4_E2M1_BITS[code]
        assert result_bits == expected_bits

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
            assert neg_val == -pos_val

    def test_monotonic_positive(self):
        """Positive FP4 values are monotonically increasing (codes 1-7)."""
        prev = 0.0
        for code in range(1, 8):
            val = float(dequant_fp4_scalar(code))
            assert val > prev
            prev = val

    def test_value_range(self):
        """All values lie in [-6.0, 6.0]."""
        for code in range(16):
            val = float(dequant_fp4_scalar(code))
            assert -6.0 <= val <= 6.0


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
            assert results[i] == expected

    def test_zero_scale_produces_zeros(self):
        """Zero scale should produce all zeros regardless of input."""
        packed = 0xFFFFFFFF
        results = dequant_fp4_x8(packed, scale=0.0)
        for i, r in enumerate(results):
            assert float(r) == 0.0

    def test_large_scale_within_fp16(self):
        """Scale producing values near FP16 max (65504)."""
        packed = pack_fp4_values([0b0111] * 8)
        scale = np.float16(5000.0)
        results = dequant_fp4_x8(packed, scale=float(scale))
        expected = np.float16(6.0 * scale)
        for r in results:
            assert abs(float(r) - float(expected)) < 200.0

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
            assert r_bits == e_bits


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
        packed = np.array(
            [
                pack_fp4_values([2, 2, 2, 2, 2, 2, 2, 2]),
                pack_fp4_values([2, 2, 2, 2, 2, 2, 2, 2]),
            ],
            dtype=np.uint32,
        )
        scales = np.array([1.0, 3.0], dtype=np.float16)
        result = ref_dequant_fp4_bulk(packed, scales, group_size=8)

        for i in range(8):
            assert float(result[i]) == pytest.approx(1.0)
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
        group_size = 32

        result = ref_dequant_fp4_bulk(packed, scales, group_size)
        assert result.shape == (128,)
        assert result.dtype == np.float16

        for i in range(num_packed):
            for j in range(8):
                idx = i * 8 + j
                group_idx = idx // group_size
                nibble = (int(packed[i]) >> (j * 4)) & 0xF
                expected = np.float16(float(dequant_fp4_scalar(nibble)) * float(scales[group_idx]))
                assert result[idx] == expected


# ============================================================================
# Test: INT4 unsigned (U4) dequantization
# ============================================================================


class TestU4Dequant:
    """Test unsigned INT4 dequantization using magic bias trick."""

    @pytest.mark.parametrize("val", range(16))
    def test_u4_identity(self, val: int):
        """Each U4 value [0,15] dequants to itself via magic trick."""
        result = magic_dequant_u4_scalar(val)
        assert result == pytest.approx(float(val), abs=1e-3)

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


# ============================================================================
# Test: INT4 signed (S4) dequantization
# ============================================================================


class TestS4Dequant:
    """Test signed INT4 dequantization (offset binary: stored = actual + 8)."""

    @pytest.mark.parametrize(
        "stored_val,expected_signed",
        [
            (0, -8.0),
            (1, -7.0),
            (7, -1.0),
            (8, 0.0),
            (9, 1.0),
            (15, 7.0),
        ],
    )
    def test_s4_individual_values(self, stored_val: int, expected_signed: float):
        """Verify offset binary mapping: stored - 8 = signed value."""
        result = magic_dequant_s4_scalar(stored_val)
        assert abs(result - expected_signed) < 0.05

    def test_s4_full_range(self):
        """All 16 stored values map to [-8, 7]."""
        for stored in range(16):
            result = magic_dequant_s4_scalar(stored)
            expected = float(stored - 8)
            assert abs(result - expected) < 0.05

    def test_s4x8_basic(self):
        """S4 x8 dequant with scale=1, zero=0."""
        packed = pack_fp4_values(list(range(8)))
        results = dequant_s4x8(packed, scale=1.0, zero_point=0.0)
        for i in range(8):
            expected = float(i - 8)
            assert abs(results[i] - expected) < 0.05


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

    def test_s4_full_range(self):
        """S4 bulk: two words covering all 16 stored values."""
        packed = np.array([0x76543210, 0xFEDCBA98], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        zeros = np.array([0.0], dtype=np.float16)
        result = ref_dequant_s4_bulk(packed, scales, zeros, group_size=128)
        expected = np.float16([i - 8.0 for i in range(16)])
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

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
        ref_results = ref_dequant_u4_bulk(packed_arr, scales_arr, zeros_arr, 128)

        for i in range(8):
            assert abs(magic_results[i] - float(ref_results[i])) < 0.1

    def test_magic_bias_is_1024(self):
        """Verify that 0x6400 in FP16 is exactly 1024.0."""
        bias = fp16_from_bits(0x6400)
        assert float(bias) == 1024.0


# ============================================================================
# Test: FP8 E5M2 bitwise construction
# ============================================================================


class TestFP8E5M2BitwiseConstruction:
    """Verify bitwise FP8 E5M2 -> FP16 construction matches mathematical values."""

    def test_all_256_codes_bit_exact(self):
        """Every FP8 E5M2 code [0x00..0xFF] matches the reference table."""
        for code in range(256):
            result = ref_dequant_fp8_e5m2(code)
            result_bits = float_to_fp16_bits(result)
            expected_bits = int(FP8_E5M2_REFERENCE[code])
            assert result_bits == expected_bits

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
        """Code for +1.0: S=0, E=15, M=0 -> byte = 0x3C."""
        val = ref_dequant_fp8_e5m2(0x3C)
        assert val == np.float16(1.0)

    def test_positive_infinity(self):
        """+Inf: S=0, E=31, M=0 -> byte = 0x7C."""
        val = ref_dequant_fp8_e5m2(0x7C)
        assert np.isinf(val) and val > 0

    def test_nan_codes(self):
        """E=31, M!=0 are NaN."""
        for code in [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]:
            val = ref_dequant_fp8_e5m2(code)
            assert np.isnan(val)

    def test_subnormals_positive(self):
        """Positive subnormals: E=0, M in {1,2,3}."""
        expected_values = [2.0**-16, 2.0**-15, 3.0 * 2.0**-16]
        for m, exp_val in enumerate(expected_values, start=1):
            code = m
            val = float(ref_dequant_fp8_e5m2(code))
            assert abs(val - exp_val) / exp_val < 1e-3

    def test_max_normal_positive(self):
        """Largest positive normal: S=0, E=30, M=3 -> 57344."""
        code = (30 << 2) | 3
        val = ref_dequant_fp8_e5m2(code)
        expected = np.float16(57344.0)
        assert val == expected

    def test_mathematical_values_match(self):
        """Cross-check: bitwise construction matches mathematical computation."""
        for code in range(256):
            bitwise_val = ref_dequant_fp8_e5m2(code)
            math_val = ref_dequant_fp8_e5m2_value(code)

            if np.isnan(bitwise_val):
                assert np.isnan(math_val) or math_val != math_val
                continue

            bitwise_f = float(bitwise_val)
            if np.isinf(bitwise_val):
                assert np.isinf(math_val) and np.sign(bitwise_f) == np.sign(math_val)
            elif bitwise_f == 0.0:
                assert math_val == 0.0
            else:
                assert abs(bitwise_f - math_val) / abs(math_val) < 1e-3


class TestFP8E5M2PackedDequant:
    """Test packed (x4) FP8 E5M2 dequantization."""

    def test_x4_identity(self):
        """Pack 4 known values and dequant with scale=1."""
        packed = 0x00 | (0x3C << 8) | (0xBC << 16) | (0x7C << 24)
        results = ref_dequant_fp8_e5m2_x4(packed, scale=1.0)

        assert results[0] == np.float16(0.0)
        assert results[1] == np.float16(1.0)
        assert results[2] == np.float16(-1.0)
        assert np.isinf(results[3]) and results[3] > 0

    def test_x4_with_scale(self):
        """Pack normal values and apply scale=0.5."""
        packed = 0x3C | (0x40 << 8) | (0x38 << 16) | (0x3E << 24)
        results = ref_dequant_fp8_e5m2_x4(packed, scale=0.5)

        expected = [0.5, 1.0, 0.25, 0.75]
        for i, exp in enumerate(expected):
            assert abs(float(results[i]) - exp) < 1e-2

    @pytest.mark.parametrize("seed", range(20))
    def test_random_packed(self, seed):
        """Random packed E5M2 values match element-wise dequant."""
        rng = np.random.default_rng(seed + 1000)
        bytes_val = rng.integers(0, 256, size=4, dtype=np.uint8)
        packed = (
            int(bytes_val[0])
            | (int(bytes_val[1]) << 8)
            | (int(bytes_val[2]) << 16)
            | (int(bytes_val[3]) << 24)
        )

        scale = float(np.float16(rng.uniform(0.01, 4.0)))
        results = ref_dequant_fp8_e5m2_x4(packed, scale=scale)

        for i in range(4):
            expected = ref_dequant_fp8_e5m2(int(bytes_val[i]))
            expected_scaled = np.float16(float(expected) * scale)

            r_bits = float_to_fp16_bits(results[i])
            e_bits = float_to_fp16_bits(expected_scaled)

            if np.isnan(results[i]):
                assert np.isnan(expected_scaled)
            else:
                assert r_bits == e_bits


# ============================================================================
# Test: NVIDIA FP8 E5M2 reference values
# ============================================================================


class TestFP8E5M2NvidiaReference:
    """Cross-validate against known NVIDIA FP8 E5M2 values."""

    NVIDIA_REFERENCE_VALUES = [
        (0x00, 0.0),
        (0x80, -0.0),
        (0x3C, 1.0),
        (0x40, 2.0),
        (0x44, 4.0),
        (0x48, 8.0),
        (0x38, 0.5),
        (0x34, 0.25),
        (0x30, 0.125),
        (0x3D, 1.25),
        (0x3E, 1.5),
        (0x3F, 1.75),
        (0x41, 2.5),
        (0x42, 3.0),
        (0x43, 3.5),
        (0x7B, 57344.0),
        (0x04, 2.0**-14),
        (0x01, 2.0**-16),
        (0x02, 2.0**-15),
        (0x03, 3.0 * 2.0**-16),
        (0xBC, -1.0),
        (0xC0, -2.0),
        (0xBF, -1.75),
    ]

    @pytest.mark.parametrize(
        "code,expected",
        NVIDIA_REFERENCE_VALUES,
        ids=[f"0x{c:02X}" for c, _ in NVIDIA_REFERENCE_VALUES],
    )
    def test_nvidia_reference_value(self, code: int, expected: float):
        """Each reference value matches the dequantized output."""
        result = ref_dequant_fp8_e5m2(code)
        result_f = float(result)

        if expected == 0.0:
            assert result_f == 0.0
            if code & 0x80:
                assert float_to_fp16_bits(result) == 0x8000
        else:
            assert abs(result_f - expected) / abs(expected) < 1e-3

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
            assert neg_bits == (pos_bits | 0x8000)

    def test_monotonicity_positive_normals(self):
        """Positive normal codes are monotonically increasing."""
        prev = -float("inf")
        for code in range(4, 0x7C):
            val = float(ref_dequant_fp8_e5m2(code))
            assert val > prev
            prev = val


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
                bits = np.array([dequant_fp4_scalar(code)]).view(np.uint16)[0]
                values.add(("zero", bits >> 15))
            else:
                values.add(val)
        assert len(values) == 16

    def test_exponent_bias_is_1(self):
        """E2M1 uses bias=1, so stored E=1 means actual exponent 0."""
        assert float(dequant_fp4_scalar(0b0010)) == 1.0
        assert float(dequant_fp4_scalar(0b0100)) == 2.0
        assert float(dequant_fp4_scalar(0b0110)) == 4.0

    def test_mantissa_adds_half(self):
        """Setting M=1 multiplies by 1.5 (adds 0.5 to implicit 1.0)."""
        assert float(dequant_fp4_scalar(0b0010)) == 1.0
        assert float(dequant_fp4_scalar(0b0011)) == 1.5
        assert float(dequant_fp4_scalar(0b0100)) == 2.0
        assert float(dequant_fp4_scalar(0b0101)) == 3.0

    def test_subnormal_is_half(self):
        """E=0, M=1 gives the subnormal value 0.5."""
        assert float(dequant_fp4_scalar(0b0001)) == 0.5
        assert float(dequant_fp4_scalar(0b1001)) == -0.5

    def test_spacing_doubles_per_binade(self):
        """Value spacing doubles with each exponent increment."""
        s1 = float(dequant_fp4_scalar(0b0011)) - float(dequant_fp4_scalar(0b0010))
        s2 = float(dequant_fp4_scalar(0b0101)) - float(dequant_fp4_scalar(0b0100))
        s3 = float(dequant_fp4_scalar(0b0111)) - float(dequant_fp4_scalar(0b0110))
        assert s1 == pytest.approx(0.5)
        assert s2 == pytest.approx(1.0)
        assert s3 == pytest.approx(2.0)


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

        packed = np.array([pack_fp4_values([2] * 8)] * num_packed, dtype=np.uint32)
        scales = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)

        result = ref_dequant_fp4_bulk(packed, scales, group_size)
        for g in range(num_groups):
            start = g * group_size
            expected = float(np.float16(1.0 * scales[g]))
            assert float(result[start]) == pytest.approx(expected, abs=0.01)

    def test_group_transition(self):
        """Elements at group boundary use correct neighboring scales."""
        group_size = 8
        packed = np.array(
            [
                pack_fp4_values([2] * 8),
                pack_fp4_values([2] * 8),
            ],
            dtype=np.uint32,
        )
        scales = np.array([1.0, 10.0], dtype=np.float16)

        result = ref_dequant_fp4_bulk(packed, scales, group_size)
        assert float(result[7]) == pytest.approx(1.0, abs=0.01)
        assert float(result[8]) == pytest.approx(10.0, abs=0.1)


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
                assert float(r) == 0.0

    def test_u4_zero_scale(self):
        """Zero scale for U4 produces all zeros."""
        packed = 0xFFFFFFFF
        results = dequant_u4x8(packed, scale=0.0, zero_point=0.0)
        for r in results:
            assert abs(r) < 1e-5

    def test_fp4_max_representable(self):
        """Largest FP4 value (6.0) with largest safe scale."""
        packed = pack_fp4_values([0b0111] * 8)
        results = dequant_fp4_x8(packed, scale=10000.0)
        for r in results:
            val = float(r)
            assert 50000.0 < val < 65504.0

    def test_alternating_zero_max(self):
        """Alternating 0 and max values."""
        packed = pack_fp4_values([0, 7, 0, 7, 0, 7, 0, 7])
        results = dequant_fp4_x8(packed, scale=1.0)
        expected = [0.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0, 6.0]
        for i, (r, e) in enumerate(zip(results, expected)):
            assert abs(float(r) - e) < 1e-3


# ============================================================================
# Test: FP16 precision boundaries
# ============================================================================


class TestFP16Precision:
    """Verify dequant behavior at FP16 precision limits."""

    def test_smallest_positive_subnormal(self):
        """FP4 code 0x1 (0.5) is the smallest non-zero positive value."""
        result = dequant_fp4_scalar(0x1)
        assert float(result) == 0.5
        bits = np.array([result]).view(np.uint16)[0]
        exp_field = (bits >> 10) & 0x1F
        assert exp_field > 0

    def test_fp4_values_are_fp16_exact(self):
        """All 16 FP4 values are exactly representable in FP16."""
        for code in range(16):
            val = dequant_fp4_scalar(code)
            val_f32 = np.float32(val)
            val_back = np.float16(val_f32)
            assert np.array([val]).view(np.uint16)[0] == np.array([val_back]).view(np.uint16)[0]

    def test_u4_values_are_fp16_exact(self):
        """All 16 U4 integer values [0,15] are exactly representable in FP16."""
        for val in range(16):
            result = magic_dequant_u4_scalar(val)
            assert result == float(int(result))


# ============================================================================
# Test: Metal kernels (skipped if PyObjC unavailable)
# ============================================================================


@pytest.mark.skipif(not _check_metal_available(), reason="Metal API (PyObjC) not available")
class TestMetalFP4Kernels:
    """Test Metal FP4 dequant kernels against reference implementation."""

    def test_all_16_codes(self):
        """Metal kernel produces exact LUT values for all 16 FP4 codes."""
        metal_out = run_metal_fp4_all_codes()
        assert len(metal_out) == 16

        for code in range(16):
            metal_bits = metal_out.view(np.uint16)[code]
            expected_bits = FP4_E2M1_BITS[code]
            assert metal_bits == expected_bits

    @pytest.mark.parametrize("scale", [0.125, 0.5, 1.0, 2.0, 4.0])
    def test_codes_0_to_7_various_scales(self, scale: float):
        """Positive FP4 codes with various scales."""
        packed = np.uint32(0x76543210)
        metal_out = run_metal_fp4_packed_scaled(packed, np.float16(scale))
        for i in range(8):
            expected = np.float16(float(FP4_E2M1_LUT[i]) * np.float16(scale))
            assert metal_out[i] == expected

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
            assert m_bits == e_bits


@pytest.mark.skipif(not _check_metal_available(), reason="Metal API (PyObjC) not available")
class TestMetalINT4Kernels:
    """Test Metal INT4 dequant kernels against reference implementation."""

    def test_u4_identity(self):
        """U4 with scale=1, zero=0 produces raw integer values."""
        packed = np.array([0x76543210], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        zeros = np.array([0.0], dtype=np.float16)

        metal_out = run_metal_int4_dequant(packed, scales, zeros, group_size=128, is_signed=False)
        ref_out = ref_dequant_u4_bulk(packed, scales, zeros, group_size=128)
        np.testing.assert_array_almost_equal(metal_out, ref_out, decimal=2)

    def test_s4_identity(self):
        """S4 with scale=1, zero=0 maps stored [0-7] to [-8, -1]."""
        packed = np.array([0x76543210], dtype=np.uint32)
        scales = np.array([1.0], dtype=np.float16)
        zeros = np.array([0.0], dtype=np.float16)

        metal_out = run_metal_int4_dequant(packed, scales, zeros, group_size=128, is_signed=True)
        ref_out = ref_dequant_s4_bulk(packed, scales, zeros, group_size=128)
        np.testing.assert_array_almost_equal(metal_out, ref_out, decimal=2)

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

        metal_out = run_metal_int4_dequant(
            packed, scales, zeros, group_size=group_size, is_signed=False
        )
        ref_out = ref_dequant_u4_bulk(packed, scales, zeros, group_size=group_size)
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

        metal_out = run_metal_int4_dequant(
            packed, scales, zeros, group_size=group_size, is_signed=False
        )
        ref_out = ref_dequant_u4_bulk(packed, scales, zeros, group_size=group_size)
        np.testing.assert_allclose(metal_out, ref_out, rtol=1e-2, atol=0.1)


@pytest.mark.skipif(not _check_metal_available(), reason="Metal API (PyObjC) not available")
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
                    f"({metal_out[code]}), expected=0x{expected_bits:04X}"
                )

        assert not mismatches, f"{len(mismatches)} mismatches out of 256 codes:\n" + "\n".join(
            mismatches[:20]
        )

    def test_zero_codes(self):
        """Verify +0 and -0."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        bits = metal_out.view(np.uint16)
        assert bits[0x00] == 0x0000
        assert bits[0x80] == 0x8000

    def test_infinity_codes(self):
        """Verify +Inf and -Inf."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        assert np.isinf(metal_out[0x7C]) and metal_out[0x7C] > 0
        assert np.isinf(metal_out[0xFC]) and metal_out[0xFC] < 0

    def test_nan_codes(self):
        """Verify NaN codes (E=31, M!=0)."""
        metal_out = run_metal_fp8_e5m2_all_codes()
        for code in [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]:
            assert np.isnan(metal_out[code])

    def test_packed_scale_1(self):
        """Packed codes with scale=1.0."""
        packed = np.uint32(0x00 | (0x3C << 8) | (0xBC << 16) | (0x40 << 24))
        metal_out = run_metal_fp8_e5m2_packed_scaled(packed, np.float16(1.0))

        assert metal_out[0] == np.float16(0.0)
        assert metal_out[1] == np.float16(1.0)
        assert metal_out[2] == np.float16(-1.0)
        assert metal_out[3] == np.float16(2.0)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_packed_e5m2(self, seed):
        """Random packed E5M2 values match element-wise reference."""
        rng = np.random.default_rng(seed + 2000)

        valid_codes = [c for c in range(256) if not ((c >> 2) & 0x1F) == 31 or (c & 0x3) == 0]
        bytes_val = rng.choice(valid_codes, size=4).astype(np.uint8)
        packed_val = (
            int(bytes_val[0])
            | (int(bytes_val[1]) << 8)
            | (int(bytes_val[2]) << 16)
            | (int(bytes_val[3]) << 24)
        )

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
                assert m_bits == e_bits


# ============================================================================
# Test: BF16 compatibility kernels (skipped if PyObjC unavailable)
# ============================================================================


@pytest.mark.skipif(not _check_metal_available(), reason="Metal API (PyObjC) not available")
class TestMetalBF16Compat:
    """Verify BF16 direct load/store conversions."""

    def test_direct_roundtrip_float8(self):
        values = np.array(
            [0.0, -0.0, 1.0, -2.5, 3.14159, 1e-8, np.inf, np.nan],
            dtype=np.float32,
        )
        expected = np.array([_bf16_roundtrip(v) for v in values], dtype=np.float32)
        metal_out = run_metal_bf16_direct_roundtrip(values)

        assert metal_out.shape == expected.shape
        for got, exp in zip(metal_out, expected):
            if np.isnan(exp):
                assert np.isnan(got)
            else:
                got_bits = np.array([got], dtype=np.float32).view(np.uint32)[0]
                exp_bits = np.array([exp], dtype=np.float32).view(np.uint32)[0]
                assert got_bits == exp_bits


# ============================================================================
# Sub-4-bit quantization tests (INT2, INT3, NF2, NF3)
# ============================================================================

# Import sub4bit functions if available
try:
    from metal_marlin.sub4bit import (
        INT2_LEVELS,
        INT3_LEVELS,
        NF2_LEVELS,
        NF3_LEVELS,
        compute_quantization_error,
        dequantize_int2,
        dequantize_int3,
        dequantize_nf2,
        dequantize_nf3,
        estimate_compression_ratio,
        get_int2_lut,
        get_int3_lut,
        get_nf2_lut,
        get_nf3_lut,
        quantize_int2,
        quantize_int3,
        quantize_nf2,
        quantize_nf3,
        select_sub4bit_format,
    )

    HAS_SUB4BIT = True
except ImportError:
    HAS_SUB4BIT = False


# Fixtures for sub4bit tests
@pytest.fixture
def gaussian_tensor_small():
    """Small Gaussian tensor for quick tests."""
    np.random.seed(42)
    return np.random.randn(64, 192).astype(np.float32) * 0.02


@pytest.fixture
def gaussian_tensor_medium():
    """Medium Gaussian tensor for accuracy tests."""
    np.random.seed(42)
    return np.random.randn(256, 384).astype(np.float32) * 0.02


@pytest.fixture
def uniform_tensor():
    """Uniform distribution tensor (non-Gaussian)."""
    np.random.seed(42)
    return np.random.uniform(-0.05, 0.05, (64, 192)).astype(np.float32)


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestINT2:
    """Test INT2 quantization (4 levels: -1.5, -0.5, 0.5, 1.5 scaled)."""

    def test_levels_correct(self):
        """Verify INT2 quantization levels."""
        expected = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(INT2_LEVELS, expected)
        assert len(INT2_LEVELS) == 4

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize, verify reasonable accuracy."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)

        assert packed.shape == (tensor.shape[0], tensor.shape[1] // 16)
        assert scales.shape == (tensor.shape[0], tensor.shape[1] // 64)
        assert reconstructed.shape == tensor.shape

        diff = tensor - reconstructed.astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.2 * tensor_range

    def test_packing_correctness(self):
        """Verify 16 INT2 values pack correctly into uint32."""
        np.random.seed(123)
        scale = 0.1
        values = np.array([INT2_LEVELS[i % 4] * scale for i in range(16)])
        tensor = values.reshape(1, 16).astype(np.float32)

        packed, scales = quantize_int2(tensor, group_size=16)

        packed_val = packed[0, 0]
        for i in range(16):
            expected_code = i % 4
            actual_code = (packed_val >> (i * 2)) & 0x3
            assert actual_code == expected_code

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 32), dtype=np.float32)
        packed, scales = quantize_int2(tensor, group_size=32)
        reconstructed = dequantize_int2(packed, scales, group_size=32)
        assert np.max(np.abs(reconstructed)) < 1e-5

    def test_max_values(self):
        """Test quantization of maximum magnitude values."""
        tensor = np.ones((16, 32), dtype=np.float32) * 1.5
        packed, scales = quantize_int2(tensor, group_size=32)
        reconstructed = dequantize_int2(packed, scales, group_size=32)
        np.testing.assert_allclose(reconstructed.astype(np.float32), tensor, rtol=1e-2)

    @pytest.mark.parametrize("shape", [(32, 64), (128, 320)])
    def test_different_shapes(self, shape):
        """Test various tensor shapes."""
        np.random.seed(42)
        tensor = np.random.randn(*shape).astype(np.float32) * 0.02
        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)
        assert reconstructed.shape == tensor.shape


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestINT3:
    """Test INT3 quantization (8 levels: -3.5 to +3.5 scaled)."""

    def test_levels_correct(self):
        """Verify INT3 quantization levels."""
        expected = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(INT3_LEVELS, expected)
        assert len(INT3_LEVELS) == 8

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize, verify better accuracy than INT2."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_int3(tensor, group_size=64)
        reconstructed = dequantize_int3(packed, scales, group_size=64)

        expected_packed_cols = (tensor.shape[1] + 9) // 10
        assert packed.shape == (tensor.shape[0], expected_packed_cols)

        min_feat = min(tensor.shape[1], reconstructed.shape[1])
        diff = tensor[:, :min_feat] - reconstructed[:, :min_feat].astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.1 * tensor_range

    def test_packing_correctness(self):
        """Verify 10 INT3 values pack correctly into uint32."""
        np.random.seed(123)
        scale = 0.1
        values = np.array([INT3_LEVELS[i % 8] * scale for i in range(10)])
        tensor = values.reshape(1, 10).astype(np.float32)

        packed, scales = quantize_int3(tensor, group_size=10)

        packed_val = packed[0, 0]
        for i in range(10):
            expected_code = i % 8
            actual_code = (packed_val >> (i * 3)) & 0x7
            assert actual_code == expected_code

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 30), dtype=np.float32)
        packed, scales = quantize_int3(tensor, group_size=30)
        reconstructed = dequantize_int3(packed, scales, group_size=30)
        assert np.max(np.abs(reconstructed)) < 1e-5


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestNF2:
    """Test NF2 (NormalFloat 2-bit) quantization with Gaussian quantile levels."""

    def test_levels_symmetric(self):
        """Verify NF2 levels are symmetric around 0."""
        assert len(NF2_LEVELS) == 4
        for i in range(2):
            np.testing.assert_almost_equal(NF2_LEVELS[i], -NF2_LEVELS[3 - i], decimal=5)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_levels_from_gaussian_quantiles(self):
        """Verify NF2 levels are based on Gaussian quantiles."""
        quantiles = [0.125, 0.375, 0.625, 0.875]
        expected_raw = stats.norm.ppf(quantiles)
        expected = expected_raw / np.max(np.abs(expected_raw))
        np.testing.assert_allclose(NF2_LEVELS, expected, rtol=1e-5)

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize Gaussian data; NF2 should be optimal."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_nf2(tensor, group_size=64)
        reconstructed = dequantize_nf2(packed, scales, group_size=64)

        assert packed.shape == (tensor.shape[0], tensor.shape[1] // 16)
        assert reconstructed.shape == tensor.shape

        diff = tensor - reconstructed.astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.2 * tensor_range

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 32), dtype=np.float32)
        packed, scales = quantize_nf2(tensor, group_size=32)
        reconstructed = dequantize_nf2(packed, scales, group_size=32)
        assert np.max(np.abs(reconstructed)) < 1e-5


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestNF3:
    """Test NF3 (NormalFloat 3-bit) quantization with Gaussian quantile levels."""

    def test_levels_symmetric(self):
        """Verify NF3 levels are symmetric around 0."""
        assert len(NF3_LEVELS) == 8
        for i in range(4):
            np.testing.assert_almost_equal(NF3_LEVELS[i], -NF3_LEVELS[7 - i], decimal=5)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_levels_from_gaussian_quantiles(self):
        """Verify NF3 levels are based on Gaussian quantiles."""
        n_levels = 8
        quantiles = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
        expected_raw = stats.norm.ppf(quantiles)
        expected = expected_raw / np.max(np.abs(expected_raw))
        np.testing.assert_allclose(NF3_LEVELS, expected, rtol=1e-5)

    def test_quantize_roundtrip_accuracy(self, gaussian_tensor_small):
        """Quantize and dequantize; NF3 should be more accurate than NF2."""
        tensor = gaussian_tensor_small
        packed, scales = quantize_nf3(tensor, group_size=64)
        reconstructed = dequantize_nf3(packed, scales, group_size=64)

        expected_packed_cols = (tensor.shape[1] + 9) // 10
        assert packed.shape == (tensor.shape[0], expected_packed_cols)

        min_feat = min(tensor.shape[1], reconstructed.shape[1])
        diff = tensor[:, :min_feat] - reconstructed[:, :min_feat].astype(np.float32)
        rmse = np.sqrt(np.mean(diff**2))
        tensor_range = tensor.max() - tensor.min()
        assert rmse < 0.1 * tensor_range

    def test_more_accurate_than_nf2(self, gaussian_tensor_medium):
        """NF3 should have lower quantization error than NF2."""
        tensor = gaussian_tensor_medium

        packed_nf2, scales_nf2 = quantize_nf2(tensor, group_size=64)
        recon_nf2 = dequantize_nf2(packed_nf2, scales_nf2, group_size=64)
        rmse_nf2 = np.sqrt(np.mean((tensor - recon_nf2.astype(np.float32)) ** 2))

        packed_nf3, scales_nf3 = quantize_nf3(tensor, group_size=64)
        recon_nf3 = dequantize_nf3(packed_nf3, scales_nf3, group_size=64)
        min_feat = min(tensor.shape[1], recon_nf3.shape[1])
        rmse_nf3 = np.sqrt(
            np.mean((tensor[:, :min_feat] - recon_nf3[:, :min_feat].astype(np.float32)) ** 2)
        )

        assert rmse_nf3 < rmse_nf2

    def test_zeros(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((16, 30), dtype=np.float32)
        packed, scales = quantize_nf3(tensor, group_size=30)
        reconstructed = dequantize_nf3(packed, scales, group_size=30)
        assert np.max(np.abs(reconstructed)) < 1e-5


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestSub4bitUtilityFunctions:
    """Test utility functions in sub4bit module."""

    @pytest.mark.parametrize("quant_type", ["int2", "int3", "nf2", "nf3"])
    def test_compute_quantization_error(self, quant_type, gaussian_tensor_small):
        """Test the error computation utility."""
        tensor = gaussian_tensor_small

        if quant_type == "int2":
            packed, scales = quantize_int2(tensor, group_size=64)
        elif quant_type == "int3":
            packed, scales = quantize_int3(tensor, group_size=64)
        elif quant_type == "nf2":
            packed, scales = quantize_nf2(tensor, group_size=64)
        else:
            packed, scales = quantize_nf3(tensor, group_size=64)

        errors = compute_quantization_error(tensor, packed, scales, quant_type, group_size=64)

        assert "mse" in errors
        assert "rmse" in errors
        assert "max_error" in errors
        assert errors["mse"] >= 0
        assert errors["rmse"] >= 0

    @pytest.mark.parametrize(
        "quant_type,expected_bits",
        [("int2", 2), ("int3", 3), ("nf2", 2), ("nf3", 3)],
    )
    def test_estimate_compression_ratio(self, quant_type, expected_bits):
        """Test compression ratio estimation."""
        shape = (4096, 4096)
        group_size = 64

        ratio = estimate_compression_ratio(shape, quant_type, group_size)

        expected_approx = 16 / expected_bits
        assert ratio > expected_approx * 0.8
        assert ratio < expected_approx * 1.1


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestSub4bitLUTExport:
    """Test lookup table export functions for Metal shaders."""

    def test_get_int2_lut(self):
        """Test INT2 LUT export."""
        lut = get_int2_lut()
        assert lut.shape == (4,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, INT2_LEVELS)

    def test_get_int3_lut(self):
        """Test INT3 LUT export."""
        lut = get_int3_lut()
        assert lut.shape == (8,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, INT3_LEVELS)

    def test_get_nf2_lut(self):
        """Test NF2 LUT export."""
        lut = get_nf2_lut()
        assert lut.shape == (4,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, NF2_LEVELS)

    def test_get_nf3_lut(self):
        """Test NF3 LUT export."""
        lut = get_nf3_lut()
        assert lut.shape == (8,)
        assert lut.dtype == np.float32
        np.testing.assert_array_almost_equal(lut, NF3_LEVELS)


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestSub4bitINTvsNFComparison:
    """Compare INT and NF formats at the same bit width."""

    def test_nf2_vs_int2_on_gaussian(self, gaussian_tensor_medium):
        """NF2 should outperform INT2 on Gaussian data."""
        tensor = gaussian_tensor_medium

        packed_int2, scales_int2 = quantize_int2(tensor, group_size=64)
        recon_int2 = dequantize_int2(packed_int2, scales_int2, group_size=64)
        rmse_int2 = np.sqrt(np.mean((tensor - recon_int2.astype(np.float32)) ** 2))

        packed_nf2, scales_nf2 = quantize_nf2(tensor, group_size=64)
        recon_nf2 = dequantize_nf2(packed_nf2, scales_nf2, group_size=64)
        rmse_nf2 = np.sqrt(np.mean((tensor - recon_nf2.astype(np.float32)) ** 2))

        assert rmse_nf2 <= rmse_int2 * 1.05


@pytest.mark.skipif(not HAS_SUB4BIT, reason="sub4bit module not available")
class TestSub4bitEdgeCases:
    """Test edge cases and boundary conditions for sub4bit."""

    def test_single_row(self):
        """Test quantization of single-row tensor."""
        tensor = np.random.randn(1, 64).astype(np.float32) * 0.02

        for quant_fn, dequant_fn in [
            (quantize_int2, dequantize_int2),
            (quantize_nf2, dequantize_nf2),
        ]:
            packed, scales = quant_fn(tensor, group_size=64)
            reconstructed = dequant_fn(packed, scales, group_size=64)
            assert reconstructed.shape == tensor.shape

    def test_single_group(self):
        """Test tensor with exactly one quantization group."""
        tensor = np.random.randn(8, 64).astype(np.float32) * 0.02

        packed, scales = quantize_int2(tensor, group_size=64)
        assert scales.shape == (8, 1)

        reconstructed = dequantize_int2(packed, scales, group_size=64)
        assert reconstructed.shape == tensor.shape

    def test_very_small_values(self):
        """Test quantization of very small values."""
        tensor = np.random.randn(16, 64).astype(np.float32) * 1e-10

        packed, scales = quantize_int2(tensor, group_size=64)
        reconstructed = dequantize_int2(packed, scales, group_size=64)
        assert np.max(np.abs(reconstructed)) < 1e-5

    def test_scale_dtype(self):
        """Verify scales are stored as float16."""
        tensor = np.random.randn(16, 64).astype(np.float32) * 0.02

        _, scales_int2 = quantize_int2(tensor, group_size=64)
        _, scales_int3 = quantize_int3(tensor, group_size=64)
        _, scales_nf2 = quantize_nf2(tensor, group_size=64)
        _, scales_nf3 = quantize_nf3(tensor, group_size=64)

        assert scales_int2.dtype == np.float16
        assert scales_int3.dtype == np.float16
        assert scales_nf2.dtype == np.float16
        assert scales_nf3.dtype == np.float16

    def test_packed_dtype(self):
        """Verify packed weights are stored as uint32."""
        tensor = np.random.randn(16, 64).astype(np.float32) * 0.02

        packed_int2, _ = quantize_int2(tensor, group_size=64)
        packed_int3, _ = quantize_int3(tensor, group_size=64)
        packed_nf2, _ = quantize_nf2(tensor, group_size=64)
        packed_nf3, _ = quantize_nf3(tensor, group_size=64)

        assert packed_int2.dtype == np.uint32
        assert packed_int3.dtype == np.uint32
        assert packed_nf2.dtype == np.uint32
        assert packed_nf3.dtype == np.uint32


# ============================================================================
# Main entry point
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
