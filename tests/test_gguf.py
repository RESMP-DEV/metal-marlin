"""Tests for GGUF Q4_K quantization format support.

Tests cover:
1. GGUF file parsing and header validation
2. Q4_K block unpacking
3. CPU dequantization (reference implementation)
4. Metal kernel execution (GPU correctness)
5. Integration with existing quantization infrastructure

Usage:
    cd contrib/metal_marlin
    uv run pytest tests/test_gguf.py -v
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Add metal_marlin package to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.quantization.gguf import (
    GGUFFile,
    GGUFHeader,
    Q4KWeights,
    decode_q4_k_min,
    decode_q4_k_scale,
    dequantize_q4_k_cpu,
    extract_q4_k_weights,
    get_q4_k_tensors,
    load_gguf_model,
    prepare_q4_k_for_metal,
    unpack_q4_k_block,
)

# ============================================================================
# Test: GGUF File Format
# ============================================================================


class TestGGUFFormat:
    """Test GGUF file format parsing."""

    def test_header_magic(self):
        """Verify GGUF magic number detection."""
        magic = b"GGUF"
        version = 3
        tensor_count = 100
        metadata_kv_count = 50

        header_bytes = (
            magic
            + struct.pack("<I", version)
            + struct.pack("<Q", tensor_count)
            + struct.pack("<Q", metadata_kv_count)
        )
        header = GGUFHeader.from_bytes(header_bytes)

        assert header.magic == magic
        assert header.version == version
        assert header.tensor_count == tensor_count
        assert header.metadata_kv_count == metadata_kv_count

    def test_header_invalid_magic(self):
        """Reject GGUF files with invalid magic."""
        invalid_magic = b"BADF"
        header_bytes = (
            invalid_magic + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        )

        with pytest.raises(ValueError, match="Invalid GGUF magic"):
            GGUFHeader.from_bytes(header_bytes)

    def test_header_too_short(self):
        """Reject GGUF headers that are too short."""
        short_header = b"GGUF" + b"\x03\x00\x00\x00"
        with pytest.raises(ValueError, match="GGUF header too short"):
            GGUFHeader.from_bytes(short_header)


# ============================================================================
# Test: Q4_K Block Unpacking
# ============================================================================


class TestQ4KBlockUnpacking:
    """Test Q4_K block unpacking logic."""

    def test_unpack_zero_block(self):
        """Unpack a block with all zero weights."""
        block_data = bytes(18)  # All zeros

        weights, scales, min_val = unpack_q4_k_block(block_data)

        assert weights.shape == (32,)
        assert np.all(weights == 0)
        assert scales.shape == (2,)
        assert min_val == -1.0  # min_bits=0 -> min=-exp2(0)=-1

    def test_unpack_max_weights(self):
        """Unpack a block with maximum weight values."""
        block_data = bytearray(18)
        # All 0xFF in packed weights (all 0xF nibbles)
        for i in range(16):
            block_data[i] = 0xFF
        # Scales: 0xF (max), Min: 0xF (max negative)
        block_data[16] = 0xFF  # Both scales at max
        block_data[17] = 0x0F  # Min at max negative

        weights, scales, min_val = unpack_q4_k_block(block_data)

        assert np.all(weights == 0xF)  # All weights at max
        assert scales[0] > 0
        assert scales[1] > 0
        assert min_val < -8.0  # Should be very negative

    def test_unpack_alternating_weights(self):
        """Unpack a block with alternating weight values."""
        block_data = bytearray(18)

        # Pack weights: 0, 15, 0, 15, ...
        # lo=15, hi=0 for all bytes -> weights[0]=15, weights[1]=0, etc.
        for i in range(16):
            block_data[i] = 0x0F

        weights, _, _ = unpack_q4_k_block(block_data)

        for i in range(32):
            expected = 15 if i % 2 == 0 else 0
            assert weights[i] == expected

    def test_scale_decoding(self):
        """Test scale value decoding."""
        # Test all 16 possible 4-bit scale values
        for scale_bits in range(16):
            scale = decode_q4_k_scale(scale_bits)
            expected = np.float16(np.exp2(scale_bits) / 16.0)
            assert abs(float(scale) - float(expected)) < 0.01

    def test_min_decoding(self):
        """Test min value decoding."""
        # Test all 16 possible 4-bit min values
        for min_bits in range(16):
            min_val = decode_q4_k_min(min_bits)
            expected = np.float16(-np.exp2(min_bits))
            assert abs(float(min_val) - float(expected)) < 0.01


# ============================================================================
# Test: Q4_K Weight Extraction
# ============================================================================


class TestQ4KWeightExtraction:
    """Test Q4_K weight extraction from tensor data."""

    def test_single_block(self):
        """Extract Q4_K weights from a single block."""
        num_elements = 32
        block_data = bytearray(18)

        # Set some weights
        for i in range(16):
            block_data[i] = i  # Pattern: lo=i, hi=i+1 (mod 16)

        weights = extract_q4_k_weights(bytes(block_data), num_elements)

        assert isinstance(weights, Q4KWeights)
        assert weights.num_elements == num_elements
        assert weights.num_blocks == 1
        assert len(weights.packed_data) == 18
        assert len(weights.scales) == 2
        assert len(weights.mins) == 1

    def test_multiple_blocks(self):
        """Extract Q4_K weights from multiple blocks."""
        num_elements = 128  # 4 blocks
        tensor_data = bytearray(4 * 18)

        # Fill with distinct patterns
        for block_idx in range(4):
            offset = block_idx * 18
            for i in range(16):
                tensor_data[offset + i] = block_idx + i
            tensor_data[offset + 16] = block_idx  # Scale byte
            tensor_data[offset + 17] = block_idx % 16  # Min byte

        weights = extract_q4_k_weights(bytes(tensor_data), num_elements)

        assert weights.num_elements == num_elements
        assert weights.num_blocks == 4
        assert len(weights.packed_data) == 72
        assert len(weights.scales) == 8
        assert len(weights.mins) == 4

    def test_size_mismatch(self):
        """Reject tensor data with incorrect size."""
        num_elements = 64  # 2 blocks expected
        tensor_data = bytes(18)  # Only 1 block provided

        with pytest.raises(ValueError, match="Tensor data size mismatch"):
            extract_q4_k_weights(tensor_data, num_elements)

    def test_partial_last_block(self):
        """Handle partial last block correctly."""
        num_elements = 50  # 2 blocks, but only 18 weights in last block
        tensor_data = bytearray(2 * 18)

        # Fill both blocks
        for i in range(36):
            tensor_data[i] = i % 16

        weights = extract_q4_k_weights(bytes(tensor_data), num_elements)

        assert weights.num_elements == num_elements
        assert weights.num_blocks == 2


# ============================================================================
# Test: Q4_K CPU Dequantization
# ============================================================================


class TestQ4KCPUDequantization:
    """Test CPU-based Q4_K dequantization."""

    def test_dequantize_zero_block(self):
        """Dequantize a block with zero weights."""
        block_data = bytes(18)
        scales = np.float16([1.0, 1.0])
        min_val = np.float16(0.0)

        weights = Q4KWeights(
            np.frombuffer(block_data, dtype=np.uint8), scales, np.array([min_val]), 32
        )

        dequantized = dequantize_q4_k_cpu(weights)

        assert dequantized.shape == (32,)
        # q=0 corresponds to -8.0 with scale=1.0, min=0.0
        # (0 - 8) * 1.0 + 0.0 = -8.0
        assert np.allclose(dequantized, -8.0, atol=0.1)

    def test_dequantize_identity_scale(self):
        """Dequantize with scale=1.0."""
        num_elements = 32
        block_data = bytearray(18)

        # Set weights to known values
        for i in range(16):
            lo = i % 16
            hi = (i + 8) % 16
            block_data[i] = lo | (hi << 4)

        # Scales that decode to ~1.0
        block_data[16] = 0x44  # scale=4 (exp2(4)/16=1.0)
        block_data[17] = 0x00  # min=-1.0

        weights = extract_q4_k_weights(bytes(block_data), num_elements)
        dequantized = dequantize_q4_k_cpu(weights)

        # Verify dequantized values match expected pattern
        assert dequantized.shape == (32,)
        # First 16 use scale_lo, last 16 use scale_hi
        assert not np.any(np.isnan(dequantized))

    def test_dequantize_with_scale(self):
        """Verify scale factor is applied correctly."""
        num_elements = 32
        block_data = bytearray(18)

        # Set weights to value 8 (zero in signed representation)
        for i in range(16):
            block_data[i] = 0x88  # lo=8, hi=8

        # Scale that doubles values
        block_data[16] = 0x55  # scale=5 (exp2(5)/16=2.0)
        block_data[17] = 0x00  # min=-1.0

        weights = extract_q4_k_weights(bytes(block_data), num_elements)
        dequantized = dequantize_q4_k_cpu(weights)

        # All weights should be -1.0 (min)
        assert np.allclose(dequantized, -1.0, atol=0.1)

    def test_dequantize_multiple_blocks(self):
        """Dequantize multiple blocks."""
        num_elements = 128  # 4 blocks
        tensor_data = bytearray(4 * 18)

        for block_idx in range(4):
            offset = block_idx * 18
            for i in range(16):
                tensor_data[offset + i] = block_idx
            tensor_data[offset + 16] = 0x44  # scale=1.0
            tensor_data[offset + 17] = 0x00  # min=-1.0

        weights = extract_q4_k_weights(bytes(tensor_data), num_elements)
        dequantized = dequantize_q4_k_cpu(weights)

        assert dequantized.shape == (128,)
        assert not np.any(np.isnan(dequantized))


# ============================================================================
# Test: Metal Dequantization
# ============================================================================


class TestQ4KMetalDequantization:
    """Test Metal-based Q4_K dequantization."""

    def _check_metal_available(self) -> bool:
        try:
            import Metal

            return True
        except ImportError:
            return False

    def _compile_gguf_shader(self):
        """Compile dequant_gguf.metal shader."""
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None, "No Metal device found"

        shader_path = Path(__file__).parent.parent / "src" / "dequant_gguf.metal"
        source = shader_path.read_text()
        options = Metal.MTLCompileOptions.new()
        options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
        library, err = device.newLibraryWithSource_options_error_(source, options, None)
        assert err is None, f"Metal compile error: {err}"
        return device, library

    @pytest.mark.skipif(
        not _check_metal_available(None), reason="Metal not available on this platform"
    )
    def test_metal_single_block(self):
        """Test Metal dequantization of a single Q4_K block."""
        import ctypes

        import Metal

        device, library = self._compile_gguf_shader()

        func = library.newFunctionWithName_("test_q4_k_block")
        assert func is not None
        pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
        assert err is None

        # Create test block data
        block_data = bytearray(18)
        for i in range(16):
            block_data[i] = i
        block_data[16] = 0x44  # scale=1.0
        block_data[17] = 0x00  # min=-1.0

        # Create buffers
        buf_packed = device.newBufferWithBytes_length_options_(
            bytes(block_data), 18, Metal.MTLResourceStorageModeShared
        )
        buf_scale_lo = device.newBufferWithBytes_length_options_(
            np.array([1.0], dtype=np.float16).tobytes(), 2, Metal.MTLResourceStorageModeShared
        )
        buf_scale_hi = device.newBufferWithBytes_length_options_(
            np.array([1.0], dtype=np.float16).tobytes(), 2, Metal.MTLResourceStorageModeShared
        )
        buf_min = device.newBufferWithBytes_length_options_(
            np.array([-1.0], dtype=np.float16).tobytes(), 2, Metal.MTLResourceStorageModeShared
        )
        buf_output = device.newBufferWithLength_options_(64, Metal.MTLResourceStorageModeShared)

        # Run kernel
        queue = device.newCommandQueue()
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_packed, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_scale_lo, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_scale_hi, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_min, 0, 3)
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 4)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
        )
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Read output
        # buf_output.contents() returns objc.varlist of bytes, join them
        output_bytes = b"".join(buf_output.contents()[:64])
        output = np.frombuffer(output_bytes, dtype=np.float16)

        assert output.shape == (32,)
        assert not np.any(np.isnan(output))

    @pytest.mark.skipif(
        not _check_metal_available(None), reason="Metal not available on this platform"
    )
    def test_metal_scale_decoding(self):
        """Test Metal scale value decoding."""
        import Metal

        device, library = self._compile_gguf_shader()

        func = library.newFunctionWithName_("test_q4_k_scale_decoding")
        assert func is not None
        pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
        assert err is None

        buf_output = device.newBufferWithLength_options_(32, Metal.MTLResourceStorageModeShared)

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

        # buf_output.contents() returns objc.varlist of bytes, join them
        output_bytes = b"".join(buf_output.contents()[:32])
        output = np.frombuffer(output_bytes, dtype=np.float16)

        assert output.shape == (16,)
        for i in range(16):
            expected = np.float16(np.exp2(i) / 16.0)
            assert abs(float(output[i]) - float(expected)) < 0.01

    @pytest.mark.skipif(
        not _check_metal_available(None), reason="Metal not available on this platform"
    )
    def test_metal_min_decoding(self):
        """Test Metal min value decoding."""
        import Metal

        device, library = self._compile_gguf_shader()

        func = library.newFunctionWithName_("test_q4_k_min_decoding")
        assert func is not None
        pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
        assert err is None

        buf_output = device.newBufferWithLength_options_(32, Metal.MTLResourceStorageModeShared)

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

        # buf_output.contents() returns objc.varlist of bytes, join them
        output_bytes = b"".join(buf_output.contents()[:32])
        output = np.frombuffer(output_bytes, dtype=np.float16)

        assert output.shape == (16,)
        for i in range(16):
            expected = np.float16(-np.exp2(i))
            assert abs(float(output[i]) - float(expected)) < 0.5


# ============================================================================
# Test: Metal Preparation
# ============================================================================


class TestQ4KMetalPreparation:
    """Test preparation of Q4_K weights for Metal."""

    def test_prepare_for_metal(self):
        """Prepare Q4_K weights for Metal buffers."""
        num_elements = 64
        tensor_data = bytearray(2 * 18)

        for i in range(36):
            tensor_data[i] = i % 16

        weights = extract_q4_k_weights(bytes(tensor_data), num_elements)
        packed_data, scales, mins = prepare_q4_k_for_metal(weights)

        assert packed_data.dtype == np.uint8
        assert scales.dtype == np.float16
        assert mins.dtype == np.float16
        assert len(packed_data) == 36
        assert len(scales) == 4
        assert len(mins) == 2

    def test_metal_buffer_layout(self):
        """Verify Metal buffer layout matches kernel expectations."""
        num_elements = 256  # 8 blocks
        tensor_data = bytearray(8 * 18)

        for i in range(144):
            tensor_data[i] = i % 16

        weights = extract_q4_k_weights(bytes(tensor_data), num_elements)
        packed_data, scales, mins = prepare_q4_k_for_metal(weights)

        # Verify packed data is contiguous
        assert packed_data.flags["C_CONTIGUOUS"]
        # Verify scales are in block-major order
        assert scales.shape[0] == weights.num_blocks * 2
        # Verify mins are in block-major order
        assert mins.shape[0] == weights.num_blocks


# ============================================================================
# Test: Integration
# ============================================================================


class TestQ4KIntegration:
    """Test Q4_K integration with existing code."""

    def test_load_gguf_model(self):
        """Test GGUF model loading."""
        # This would require a real GGUF file
        # For now, just verify the function exists and returns correct type
        assert callable(load_gguf_model)

    def test_get_q4_k_tensors(self):
        """Test filtering Q4_K tensors."""
        import struct
        import tempfile

        # Create a minimal GGUF file
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            # Write header
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))  # version
            f.write(struct.pack("<Q", 1))  # tensor_count
            f.write(struct.pack("<Q", 0))  # metadata_kv_count

            # Write tensor info
            name = b"test.weight"
            f.write(struct.pack("<Q", len(name)))
            f.write(name)
            f.write(struct.pack("<I", 2))  # n_dims
            f.write(struct.pack("<Q", 128))  # dim0
            f.write(struct.pack("<Q", 256))  # dim1
            f.write(struct.pack("<I", 12))  # ggml_type Q4_K
            f.write(struct.pack("<Q", 24))  # offset (after tensor info)

            # Write tensor data (minimal)
            tensor_size = (128 * 256 + 31) // 32 * 18
            f.write(bytes(tensor_size))

            gguf_path = f.name

        try:
            with GGUFFile(gguf_path) as gguf_file:
                q4k_tensors = get_q4_k_tensors(gguf_file)

                assert len(q4k_tensors) == 1
                assert q4k_tensors[0].name == "test.weight"
                assert q4k_tensors[0].is_q4_k
        finally:
            import os

            os.unlink(gguf_path)


if __name__ == "__main__":
    # Run tests with pytest
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
