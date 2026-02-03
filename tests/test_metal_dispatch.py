"""Test Metal shared memory (zero-copy) buffers and staging buffers."""

import pytest
import torch

# Check availability
try:
    from metal_marlin.metal_dispatch import (
        _ASYNC_TRANSFER_THRESHOLD,
        Metal,
        MetalKernelLibrary,
        StagingTransferHandle,
        _private_buffer_from_bytes,
        _private_buffer_from_tensor,
        mps_tensor_to_metal_buffer,
    )

    MetalKernelLibrary()
    HAS_METAL = True
except Exception:
    HAS_METAL = False

HAS_MPS = torch.backends.mps.is_available()


def _get_buffer_from_result(result):
    """Helper to get Metal buffer from result (handle or direct buffer)."""
    if isinstance(result, StagingTransferHandle):
        result.wait()
        return result.destination_buffer
    return result


@pytest.mark.skipif(not HAS_METAL, reason="Metal not available")
def test_shared_buffer_from_bytes():
    """Test that buffers are created from CPU bytes."""
    lib = MetalKernelLibrary()

    # Test various buffer sizes
    # Note: kept below 1MB to avoid async transfer path which returns StagingTransferHandle
    for size_kb in [1, 128, 512]:
        data = bytes(size_kb * 1024)
        buf = _private_buffer_from_bytes(lib, lib.device, data)

        # Buffer should be valid Metal buffer
        assert buf is not None
        assert hasattr(buf, "length")
        assert hasattr(buf, "contents")


@pytest.mark.skipif(not HAS_METAL, reason="Metal not available")
def test_shared_buffer_from_cpu_tensor():
    """Test that shared buffers are created from CPU tensors."""
    lib = MetalKernelLibrary()

    # Create CPU tensor
    cpu_tensor = torch.randn(1024, 1024, dtype=torch.float16, device="cpu")

    # Convert to shared buffer
    buf = _private_buffer_from_tensor(cpu_tensor, lib, lib.device, cache=False)

    # Should return just the buffer (zero-copy, no handles)
    assert not isinstance(buf, tuple)
    assert buf is not None


@pytest.mark.skipif(not HAS_METAL, reason="Metal not available")
@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_shared_buffer_from_mps_tensor():
    """Test that shared buffers are created from MPS tensors."""
    lib = MetalKernelLibrary()

    # Create MPS tensor
    mps_tensor = torch.randn(1024, 1024, dtype=torch.float16, device="mps")

    # Convert to shared buffer
    buf = _private_buffer_from_tensor(mps_tensor, lib, lib.device, cache=False)

    # Should return just the buffer (zero-copy, no handles)
    assert not isinstance(buf, tuple)
    assert buf is not None


@pytest.mark.skipif(not HAS_METAL, reason="Metal not available")
def test_storage_mode_shared():
    """Verify that buffers use StorageModeShared."""
    lib = MetalKernelLibrary()
    data = bytes(512 * 1024)  # 512KB
    buf = _private_buffer_from_bytes(lib, lib.device, data)

    # Check that buffer is in shared storage mode
    storage_mode = buf.storageMode()
    assert storage_mode == Metal.MTLResourceStorageModeShared, (
        f"Expected StorageModeShared, got {storage_mode}"
    )


@pytest.mark.skipif(not HAS_METAL, reason="Metal not available")
@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_mps_buffer_storage_mode():
    """Verify MPS tensor buffers use StorageModeShared."""
    lib = MetalKernelLibrary()
    mps_tensor = torch.randn(1024, 1024, dtype=torch.float16, device="mps")
    buf = mps_tensor_to_metal_buffer(mps_tensor, lib.device)

    # Check that buffer is in shared storage mode
    storage_mode = buf.storageMode()
    assert storage_mode == Metal.MTLResourceStorageModeShared, (
        f"Expected StorageModeShared, got {storage_mode}"
    )


@pytest.mark.skipif(not HAS_METAL, reason="Metal not available")
def test_zero_copy_content_access():
    """Verify CPU can access shared buffer contents directly."""
    lib = MetalKernelLibrary()

    # Create buffer with known data
    expected = b"hello world" * 1000
    buf = _private_buffer_from_bytes(lib, lib.device, expected)

    # Verify CPU can read the contents (zero-copy)
    contents = buf.contents()
    view = memoryview(contents.as_buffer(buf.length()))
    actual = bytes(view[: len(expected)])
    assert actual == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
