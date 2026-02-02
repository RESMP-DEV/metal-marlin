"""Tests for CPU to Metal buffer creation."""

import numpy as np
import pytest
import torch


@pytest.fixture
def metal_device():
    """Get Metal device."""
    import Metal
    return Metal.MTLCreateSystemDefaultDevice()


class TestCPUToMetalBuffer:
    """Tests for cpu_tensor_to_metal_buffer."""

    def test_float32_tensor(self, metal_device):
        """Test float32 CPU tensor to Metal buffer."""
        from metal_marlin.metal_dispatch import cpu_tensor_to_metal_buffer

        t = torch.randn(100, 100, dtype=torch.float32)
        buf = cpu_tensor_to_metal_buffer(t, metal_device)
        assert buf is not None
        assert buf.length() == t.numel() * t.element_size()

    def test_float16_tensor(self, metal_device):
        """Test float16 CPU tensor to Metal buffer."""
        from metal_marlin.metal_dispatch import cpu_tensor_to_metal_buffer

        t = torch.randn(100, 100, dtype=torch.float16)
        buf = cpu_tensor_to_metal_buffer(t, metal_device)
        assert buf is not None
        assert buf.length() == t.numel() * 2  # float16 = 2 bytes

    def test_uint8_tensor(self, metal_device):
        """Test uint8 CPU tensor to Metal buffer."""
        from metal_marlin.metal_dispatch import cpu_tensor_to_metal_buffer

        t = torch.randint(0, 256, (100, 100), dtype=torch.uint8)
        buf = cpu_tensor_to_metal_buffer(t, metal_device)
        assert buf is not None
        assert buf.length() == t.numel()

    def test_rejects_mps_tensor(self, metal_device):
        """Test that MPS tensors are rejected."""
        from metal_marlin.metal_dispatch import cpu_tensor_to_metal_buffer

        t = torch.randn(10, 10, device="mps")
        with pytest.raises(ValueError, match="must be on CPU"):
            cpu_tensor_to_metal_buffer(t, metal_device)

    def test_handles_non_contiguous(self, metal_device):
        """Test non-contiguous tensor is made contiguous."""
        from metal_marlin.metal_dispatch import cpu_tensor_to_metal_buffer

        t = torch.randn(100, 100).t()  # Transpose makes non-contiguous
        assert not t.is_contiguous()
        buf = cpu_tensor_to_metal_buffer(t, metal_device)
        assert buf is not None


class TestNumpyToMetalBuffer:
    """Tests for numpy array to Metal buffer via torch conversion."""

    def test_float32_array_via_torch(self, metal_device):
        """Test float32 numpy array to Metal buffer via torch tensor."""
        from metal_marlin.metal_dispatch import cpu_tensor_to_metal_buffer

        arr = np.random.randn(100, 100).astype(np.float32)
        t = torch.from_numpy(arr)
        buf = cpu_tensor_to_metal_buffer(t, metal_device)
        assert buf is not None
        assert buf.length() == arr.nbytes

    def test_uint8_array_via_torch(self, metal_device):
        """Test uint8 numpy array to Metal buffer via torch tensor."""
        from metal_marlin.metal_dispatch import cpu_tensor_to_metal_buffer

        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        t = torch.from_numpy(arr)
        buf = cpu_tensor_to_metal_buffer(t, metal_device)
        assert buf is not None
        assert buf.length() == arr.nbytes
