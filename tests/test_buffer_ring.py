"""Tests for ring buffer transient allocations."""

from __future__ import annotations

import gc

import pytest

HAS_MPS = False
try:
    import torch

    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    torch = None

try:
    from metal_marlin.buffer_ring import (
        RingBuffer,
        get_ring_buffer,
        reset_ring_buffer,
        ring_buffer_stats,
    )
except ImportError:
    RingBuffer = None


requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS required (Apple Silicon)")
requires_ring_buffer = pytest.mark.skipif(RingBuffer is None, reason="RingBuffer module required")


def clear_mps_memory():
    gc.collect()
    if HAS_MPS:
        torch.mps.empty_cache()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    clear_mps_memory()
    reset_ring_buffer()


@pytest.fixture
def metal_device():
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("Metal device not available")
    return device


@requires_ring_buffer
class TestRingBufferBasics:
    """Test basic ring buffer operations."""

    def test_initialization(self, metal_device):
        ring = RingBuffer(metal_device, capacity=1024 * 1024)

        assert ring.capacity > 0
        assert ring.used == 0
        assert ring.available == ring.capacity
        assert ring.high_water_mark == 0
        assert ring.buffer is not None

    def test_default_capacity(self, metal_device):
        ring = RingBuffer(metal_device)

        assert ring.capacity >= 256 * 1024 * 1024

    def test_single_allocation(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        buf, offset = ring.alloc(1024)

        assert offset == 0
        assert ring.used >= 1024
        assert ring.used <= 1024 + 127  # Account for alignment
        assert buf is ring.buffer

    def test_sequential_allocations(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        buf1, offset1 = ring.alloc(512)
        buf2, offset2 = ring.alloc(512)
        buf3, offset3 = ring.alloc(512)

        assert offset1 < offset2 < offset3
        assert buf1 is buf2 is buf3 is ring.buffer

    def test_reset_clears_offset(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        _, offset1 = ring.alloc(1024)
        assert offset1 == 0

        _, offset2 = ring.alloc(1024)
        assert offset2 > 0

        ring.reset()

        _, offset3 = ring.alloc(1024)
        assert offset3 == 0
        assert ring.used >= 1024

    def test_capacity_enforced(self, metal_device):
        ring = RingBuffer(metal_device, capacity=1024)

        _, offset1 = ring.alloc(512)
        assert offset1 == 0

        _, offset2 = ring.alloc(512)
        assert offset2 >= 512

        with pytest.raises(RuntimeError, match="overflow"):
            ring.alloc(512)

    def test_can_alloc_prediction(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        assert ring.can_alloc(2048) is True
        ring.alloc(2048)

        assert ring.can_alloc(1024) is True
        ring.alloc(1024)

        assert ring.can_alloc(2048) is False

    def test_high_water_mark(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        ring.alloc(1024)
        assert ring.high_water_mark >= 1024

        ring.alloc(512)
        assert ring.high_water_mark >= 1024 + 512

        ring.reset()
        ring.alloc(256)
        assert ring.high_water_mark >= 1024 + 512

    def test_allocation_count(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        assert ring.stats()["allocation_count"] == 0

        ring.alloc(512)
        assert ring.stats()["allocation_count"] == 1

        ring.alloc(512)
        ring.alloc(512)
        assert ring.stats()["allocation_count"] == 3

        ring.reset()
        ring.alloc(512)
        assert ring.stats()["allocation_count"] == 1


@requires_ring_buffer
class TestRingBufferAlignment:
    """Test cache line and page alignment."""

    def test_cache_line_alignment(self, metal_device):
        ring = RingBuffer(metal_device, capacity=8192)

        offsets = []
        for i in range(10):
            _, offset = ring.alloc(127 + i)
            offsets.append(offset)

        for i in range(len(offsets) - 1):
            gap = offsets[i + 1] - offsets[i]
            if i < 9:
                assert gap >= 128, f"Allocation {i} not cache-line aligned: gap={gap}"

    def test_small_buffer_alignment(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        _, offset1 = ring.alloc(1)
        assert offset1 == 0

        _, offset2 = ring.alloc(1)
        assert offset2 == 128

    def test_large_buffer_page_alignment(self, metal_device):
        ring = RingBuffer(metal_device, capacity=1024 * 1024)

        large_size = 128 * 1024  # 128KB, triggers page alignment
        _, offset = ring.alloc(large_size)

        assert offset % 16384 == 0, f"Large buffer not page-aligned: offset={offset}"


@requires_ring_buffer
class TestRingBufferAllocBytes:
    """Test alloc_bytes for CPU-side access."""

    def test_alloc_bytes_returns_memoryview(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        view, offset = ring.alloc_bytes(256)

        assert isinstance(view, memoryview)
        assert len(view) == 256
        assert offset == 0

    def test_alloc_bytes_writable(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        view, offset = ring.alloc_bytes(256)

        for i in range(256):
            view[i] = i % 256

        for i in range(256):
            assert view[i] == i % 256, f"Data corruption at byte {i}"

    def test_alloc_bytes_no_overlap(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        view1, offset1 = ring.alloc_bytes(256)
        view2, offset2 = ring.alloc_bytes(256)

        for i in range(256):
            view1[i] = 0xAA

        for i in range(256):
            view2[i] = 0xBB

        for i in range(256):
            assert view1[i] == 0xAA, f"view1 corrupted at {i}"
            assert view2[i] == 0xBB, f"view2 corrupted at {i}"

    def test_alloc_bytes_overflow(self, metal_device):
        ring = RingBuffer(metal_device, capacity=1024)

        ring.alloc_bytes(512)

        with pytest.raises(RuntimeError, match="overflow"):
            ring.alloc_bytes(513)


@requires_ring_buffer
class TestRingBufferStats:
    """Test ring buffer statistics."""

    def test_stats_structure(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        stats = ring.stats()

        assert "capacity_bytes" in stats
        assert "used_bytes" in stats
        assert "available_bytes" in stats
        assert "high_water_mark_bytes" in stats
        assert "allocation_count" in stats
        assert "utilization" in stats
        assert "peak_utilization" in stats

    def test_utilization_tracking(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        stats1 = ring.stats()
        assert stats1["utilization"] == 0.0
        assert stats1["peak_utilization"] == 0.0

        ring.alloc(1024)

        stats2 = ring.stats()
        assert stats2["utilization"] > 0.0
        assert stats2["peak_utilization"] > 0.0
        assert stats2["utilization"] <= 1.0

        ring.reset()

        stats3 = ring.stats()
        assert stats3["utilization"] == 0.0
        assert stats3["peak_utilization"] > 0.0  # Peak persists

    def test_capacity_consistency(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        assert ring.capacity == ring.stats()["capacity_bytes"]
        assert ring.available == ring.stats()["available_bytes"]
        assert ring.used == ring.stats()["used_bytes"]


@requires_ring_buffer
class TestGlobalRingBuffer:
    """Test global ring buffer singleton functions."""

    def test_get_ring_buffer_singleton(self, metal_device):
        ring1 = get_ring_buffer(metal_device, capacity=1024 * 1024)
        ring2 = get_ring_buffer(metal_device, capacity=1024 * 1024)

        assert ring1 is ring2

    def test_get_ring_buffer_different_device(self, metal_device):
        import Metal

        ring1 = get_ring_buffer(metal_device, capacity=1024 * 1024)

        device2 = Metal.MTLCreateSystemDefaultDevice()
        if device2 is None:
            pytest.skip("Second Metal device not available")

        ring2 = get_ring_buffer(device2, capacity=1024 * 1024)

        if id(metal_device) != id(device2):
            assert ring1 is not ring2

    def test_reset_ring_buffer_global(self, metal_device):
        ring = get_ring_buffer(metal_device, capacity=1024 * 1024)

        _, _ = ring.alloc(1024)
        assert ring.used > 0

        reset_ring_buffer()

        assert ring.used == 0

    def test_ring_buffer_stats_global(self, metal_device):
        reset_ring_buffer()
        ring = get_ring_buffer(metal_device, capacity=1024 * 1024)

        _, _ = ring.alloc(512)

        stats = ring_buffer_stats()
        assert stats is not None
        assert stats["used_bytes"] >= 512
        assert stats["capacity_bytes"] == 1024 * 1024
        assert 0 <= stats["utilization"] <= 1.0


@requires_ring_buffer
class TestRingBufferEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_size_allocation(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        _, offset = ring.alloc(0)

        assert offset == 0
        assert ring.used == 0  # Zero-size uses no space

    def test_exact_capacity_allocation(self, metal_device):
        ring = RingBuffer(metal_device, capacity=4096)

        _, _ = ring.alloc(4096)

        assert ring.available == 0

    def test_repeated_reset_no_leak(self, metal_device):
        ring = RingBuffer(metal_device, capacity=1024 * 1024)

        initial_capacity = ring.capacity

        for _ in range(100):
            ring.reset()
            _, offset = ring.alloc(1024)
            assert offset == 0

        assert ring.capacity == initial_capacity

    def test_large_number_of_allocations(self, metal_device):
        ring = RingBuffer(metal_device, capacity=1024 * 1024)

        ring.reset()

        for i in range(100):
            _, offset = ring.alloc(128)  # 100 * 128 = 12800 bytes
            if i == 0:
                assert offset == 0

        assert ring.stats()["allocation_count"] == 100

    def test_reset_between_forward_passes(self, metal_device):
        ring = RingBuffer(metal_device, capacity=1024 * 1024)

        def forward_pass(pass_id):
            ring.reset()

            allocations = []
            for i in range(10):
                buf, offset = ring.alloc(1024)
                allocations.append((buf, offset))

            assert allocations[0][1] == 0
            assert ring.used > 0

            return allocations

        pass1 = forward_pass(1)
        pass2 = forward_pass(2)
        pass3 = forward_pass(3)

        assert pass1[0][1] == pass2[0][1] == pass3[0][1] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
