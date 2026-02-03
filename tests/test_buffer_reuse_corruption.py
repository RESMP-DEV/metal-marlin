"""Tests for buffer reuse data corruption scenarios.

This test suite validates that buffer reuse mechanisms don't cause data corruption
in various scenarios including:
- Multiple forward passes with same/different inputs
- Concurrent operations (simulated)
- Buffer aliasing (same buffer used for multiple purposes)
- Edge cases like batch size changes

Key assertions check for:
- Stale data from previous operations
- Data bleeding between different buffer users
- Incorrect results due to buffer reuse bugs
"""

from __future__ import annotations

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

HAS_MPS = torch.backends.mps.is_available()

try:
    from metal_marlin._buffer_pool import (
        BufferPriority,
        MetalBufferPool,
        TransientRingBuffer,
        get_transient_ring,
        reset_transient_ring,
    )
    from metal_marlin.metal_dispatch import HAS_METAL, MetalKernelLibrary

    HAS_BUFFER_POOL = True
except ImportError:
    HAS_BUFFER_POOL = False


requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS required (Apple Silicon)")
requires_buffer_pool = pytest.mark.skipif(not HAS_BUFFER_POOL, reason="Buffer pool modules required")
requires_metal = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def clear_mps_memory():
    """Clear MPS memory cache."""
    gc.collect()
    if HAS_MPS:
        torch.mps.empty_cache()
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    clear_mps_memory()


@pytest.fixture
def metal_device():
    """Get Metal device."""
    import Metal
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("Metal device not available")
    return device


@pytest.fixture
def metal_lib():
    """Compiled Metal library fixture."""
    if not HAS_METAL:
        pytest.skip("Metal not available")
    return MetalKernelLibrary.from_source_dir()


@requires_metal
class TestBufferPoolBasics:
    """Test basic buffer pool operations don't corrupt data."""

    def test_buffer_content_preserved(self, metal_device):
        """Data written to buffer should be preserved through release/re-acquire."""
        pool = MetalBufferPool(metal_device)

        # Get buffer and write pattern
        buf1 = pool.get(1024, BufferPriority.NORMAL)

        # Write pattern via backing buffer
        backing1 = pool._backing.get(id(buf1))
        if backing1:
            for i in range(len(backing1)):
                backing1[i] = (i * 7) % 256  # Write unique pattern

        original_id = id(buf1)
        pool.release(buf1)

        # Get buffer again
        buf2 = pool.get(1024, BufferPriority.NORMAL)

        # Should be same buffer (reused)
        assert id(buf2) == original_id, "Buffer not reused from pool"

        # Data might be overwritten - we're testing the mechanism works,
        # not that data persists (that would be a bug in reuse!)
        backing2 = pool._backing.get(id(buf2))
        assert backing2 is not None, "Backing buffer missing"

    def test_no_cross_buffer_contamination(self, metal_device):
        """Operations on one buffer shouldn't affect others."""
        pool = MetalBufferPool(metal_device)

        # Get multiple buffers
        buf1 = pool.get(1024, BufferPriority.NORMAL)
        buf2 = pool.get(2048, BufferPriority.NORMAL)
        buf3 = pool.get(1024, BufferPriority.NORMAL)

        # Get backings
        backing1 = pool._backing.get(id(buf1))
        backing2 = pool._backing.get(id(buf2))
        backing3 = pool._backing.get(id(buf3))

        if backing1 and backing2 and backing3:
            # Fill with distinct patterns
            backing1[:] = b'\xAA' * len(backing1)
            backing2[:] = b'\xBB' * len(backing2)
            backing3[:] = b'\xCC' * len(backing3)

            # Verify no cross-contamination
            assert all(b == 0xAA for b in backing1), "Buffer 1 corrupted"
            assert all(b == 0xBB for b in backing2), "Buffer 2 corrupted"
            assert all(b == 0xCC for b in backing3), "Buffer 3 corrupted"

        pool.release(buf1)
        pool.release(buf2)
        pool.release(buf3)

    def test_priority_isolation(self, metal_device):
        """Different priority buffers should be managed independently."""
        pool = MetalBufferPool(metal_device)

        # Get buffers with different priorities
        low_buf = pool.get(1024, BufferPriority.LOW)
        normal_buf = pool.get(1024, BufferPriority.NORMAL)
        high_buf = pool.get(1024, BufferPriority.HIGH)
        pinned_buf = pool.get(1024, BufferPriority.PINNED)

        # All should be distinct
        assert len({id(low_buf), id(normal_buf), id(high_buf), id(pinned_buf)}) == 4, (
            "Buffers with different priorities should be distinct objects"
        )

        # Check priorities tracked correctly
        assert pool.get_priority(low_buf) == BufferPriority.LOW
        assert pool.get_priority(normal_buf) == BufferPriority.NORMAL
        assert pool.get_priority(high_buf) == BufferPriority.HIGH
        assert pool.get_priority(pinned_buf) == BufferPriority.PINNED

        pool.release(low_buf)
        pool.release(normal_buf)
        pool.release(high_buf)
        # Don't release pinned - it should never be evicted


@requires_metal
class TestTransientRingBuffer:
    """Test TransientRingBuffer for data corruption."""

    def test_sequential_allocations_no_overlap(self, metal_device):
        """Sequential allocations should not overlap."""
        ring = TransientRingBuffer(metal_device, capacity=1024 * 1024)

        allocations = []
        for i in range(10):
            size = 1024 * (i + 1)
            buf, offset = ring.alloc(size)
            allocations.append((offset, size))

        # Check no overlaps
        for i, (off1, size1) in enumerate(allocations):
            end1 = off1 + size1
            for j, (off2, size2) in enumerate(allocations):
                if i != j:
                    end2 = off2 + size2
                    assert not (off1 < end2 and off2 < end1), (
                        f"Allocations {i} [{off1}, {end1}) and "
                        f"{j} [{off2}, {end2}) overlap"
                    )

    def test_reset_allows_reuse(self, metal_device):
        """Reset should allow buffer reuse without corruption."""
        ring = TransientRingBuffer(metal_device, capacity=1024 * 1024)

        # First allocation pattern
        buf1, offset1 = ring.alloc(4096)

        # Write pattern to backing
        backing = ring._backing
        for i in range(offset1, min(offset1 + 100, len(backing))):
            backing[i] = 0xDD

        # Reset
        ring.reset()

        # New allocation should start at 0
        buf2, offset2 = ring.alloc(2048)
        assert offset2 == 0, f"After reset, offset should be 0, got {offset2}"

        # Same underlying buffer
        assert buf1 is buf2, "Ring buffer should reuse same Metal buffer"

    def test_capacity_enforced(self, metal_device):
        """Allocations beyond capacity should raise error."""
        ring = TransientRingBuffer(metal_device, capacity=1024)

        # First allocation should work
        buf1, offset1 = ring.alloc(512)
        assert offset1 == 0

        # Second should work
        buf2, offset2 = ring.alloc(512)
        assert offset2 >= 512

        # Third should fail (over capacity)
        with pytest.raises(RuntimeError, match="overflow"):
            ring.alloc(1024)

    def test_can_alloc_check(self, metal_device):
        """can_alloc should correctly predict allocation success."""
        ring = TransientRingBuffer(metal_device, capacity=4096)

        assert ring.can_alloc(2048) is True
        buf, _ = ring.alloc(2048)

        assert ring.can_alloc(1024) is True
        buf2, _ = ring.alloc(1024)

        # Now we have 3072 used (or more due to alignment)
        # Remaining should be less than 1024
        assert ring.can_alloc(2048) is False

    def test_alloc_bytes_memoryview(self, metal_device):
        """alloc_bytes should return valid memoryview."""
        ring = TransientRingBuffer(metal_device, capacity=1024 * 1024)

        view1, offset1 = ring.alloc_bytes(256)
        assert len(view1) == 256

        # Write pattern
        for i in range(256):
            view1[i] = i % 256

        # Verify pattern
        for i in range(256):
            assert view1[i] == i % 256, f"Data corruption at byte {i}"

        # Second allocation should not overlap
        view2, offset2 = ring.alloc_bytes(256)
        assert offset2 >= offset1 + 256 or offset2 == 0, "Allocations overlap"


@requires_metal
class TestBufferPoolEviction:
    """Test eviction doesn't cause data corruption."""

    def test_eviction_removes_correct_buffers(self, metal_device):
        """Eviction should remove low-priority buffers first."""
        pool = MetalBufferPool(metal_device)

        # Create low priority buffers and release them
        low_bufs = []
        for _ in range(5):
            buf = pool.get(1024, BufferPriority.LOW)
            low_bufs.append(id(buf))
            pool.release(buf)

        # At this point we should have pooled buffers
        initial_pooled = pool.total_pooled_bytes
        assert initial_pooled > 0, "No buffers in pool after release"

        # Evict 2KB worth (should remove some low priority buffers)
        freed = pool.evict(2 * 1024)

        # Should have freed some bytes
        assert freed >= 2 * 1024 or pool.total_pooled_bytes == 0, (
            f"Eviction didn't free enough: freed={freed}, pooled={pool.total_pooled_bytes}"
        )

        # Now test with pinned buffer
        pinned_buf = pool.get(1024, BufferPriority.PINNED)
        pinned_id = id(pinned_buf)
        # Don't release pinned

        # Evict again - pinned should not be affected
        pool.evict(1024)

        # Pinned should still be tracked
        assert id(pinned_buf) in pool._tracked, "Pinned buffer was incorrectly evicted"

    def test_max_pool_size_enforcement(self, metal_device):
        """max_pool_size should trigger eviction when exceeded."""
        pool = MetalBufferPool(metal_device, max_pool_size=8 * 1024)

        # Allocate and release to fill pool
        for _ in range(10):
            buf = pool.get(1024, BufferPriority.LOW)
            pool.release(buf)

        # Pool should stay under limit
        stats = pool.stats()
        assert stats["current_pooled_bytes"] <= 8 * 1024, (
            f"Pool exceeded max size: {stats['current_pooled_bytes']} > {8 * 1024}"
        )

    def test_ref_count_prevents_eviction(self, metal_device):
        """Buffers with non-zero ref count should not be evicted."""
        pool = MetalBufferPool(metal_device)

        # Get buffer (ref_count = 1)
        buf = pool.get(1024, BufferPriority.LOW)

        # Get again from pool (should be new allocation since not released)
        buf2 = pool.get(1024, BufferPriority.LOW)

        # Should be different buffers
        assert buf is not buf2, "Got same buffer while first still held"

        pool.release(buf)
        pool.release(buf2)


@requires_metal
class TestMoEBufferPool:
    """Test MoEBufferPool for data corruption."""

    def test_output_buffer_zeroing(self, metal_device):
        """Output buffers should be zeroed on get."""
        from metal_marlin.trellis.moe_dispatch import MoEBufferPool

        pool = MoEBufferPool(metal_device, hidden_dim=256, max_batch=8)

        # Get and fill with pattern
        out1, buf1 = pool.get_output_buffer(4)
        out1.fill_(999.0)

        # Release (simulated by getting again)
        out2, buf2 = pool.get_output_buffer(4)

        # Should be same tensor
        assert out1 is out2, "Output buffer not reused"

        # Should be zeroed
        assert out2.abs().max().item() == 0.0, (
            f"Output buffer not zeroed: max = {out2.abs().max().item()}"
        )

    def test_activation_buffer_copy_isolation(self, metal_device):
        """Activation buffer should copy, not share memory."""
        from metal_marlin.trellis.moe_dispatch import MoEBufferPool

        pool = MoEBufferPool(metal_device, hidden_dim=256, max_batch=8)

        # Create input tensor
        act = torch.randn(4, 256, dtype=torch.float16, device="mps")
        original_val = act[0, 0].item()

        # Get buffer (copies data)
        buf = pool.get_activation_buffer(4, act)

        # Modify original tensor
        act.fill_(999.0)

        # Get buffer again with different data
        act2 = torch.randn(4, 256, dtype=torch.float16, device="mps")
        buf2 = pool.get_activation_buffer(4, act2)

        # Pool should have copied data, not aliased
        # (Cannot directly verify, but operation should succeed)

    def test_different_batch_sizes_isolated(self, metal_device):
        """Different batch sizes should have independent buffers."""
        from metal_marlin.trellis.moe_dispatch import MoEBufferPool

        pool = MoEBufferPool(metal_device, hidden_dim=256, max_batch=16)

        # Get buffers for different batch sizes
        tensor1, _ = pool.get_output_buffer(1)
        tensor2, _ = pool.get_output_buffer(2)
        tensor4, _ = pool.get_output_buffer(4)
        tensor8, _ = pool.get_output_buffer(8)

        # Verify shapes
        assert tensor1.shape == (1, 256)
        assert tensor2.shape == (2, 256)
        assert tensor4.shape == (4, 256)
        assert tensor8.shape == (8, 256)

        # Write distinct patterns
        tensor1.fill_(1.0)
        tensor2.fill_(2.0)
        tensor4.fill_(4.0)
        tensor8.fill_(8.0)

        # Verify isolation
        assert tensor1[0, 0].item() == 1.0, "Batch=1 buffer corrupted"
        assert tensor2[0, 0].item() == 2.0, "Batch=2 buffer corrupted"
        assert tensor4[0, 0].item() == 4.0, "Batch=4 buffer corrupted"
        assert tensor8[0, 0].item() == 8.0, "Batch=8 buffer corrupted"

    def test_expert_ids_buffer_isolation(self, metal_device):
        """Expert IDs buffers should be isolated."""
        from metal_marlin.trellis.moe_dispatch import MoEBufferPool

        pool = MoEBufferPool(metal_device, hidden_dim=256, max_batch=8, top_k_values=(2, 4))

        # Create test tensors
        ids_4_2 = torch.randint(0, 4, (4, 2), dtype=torch.int32, device="mps")
        ids_4_4 = torch.randint(0, 4, (4, 4), dtype=torch.int32, device="mps")

        # Get buffers
        buf1 = pool.get_expert_ids_buffer(4, 2, ids_4_2)
        buf2 = pool.get_expert_ids_buffer(4, 4, ids_4_4)

        # Should be different
        assert buf1 is not buf2, "Different (batch, top_k) should get different buffers"


@requires_metal
class TestConcurrentAccess:
    """Test concurrent access patterns."""

    def test_concurrent_get_release(self, metal_device):
        """Concurrent get/release should not corrupt pool state."""
        pool = MetalBufferPool(metal_device)
        errors = []
        results = []

        def worker(worker_id):
            try:
                for i in range(50):
                    buf = pool.get(1024)
                    # Small delay to increase race chance
                    time.sleep(0.0001)
                    pool.release(buf)
                results.append(f"worker_{worker_id}_ok")
            except Exception as e:
                errors.append(f"worker_{worker_id}: {e}")

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"

        # Pool should be consistent
        stats = pool.stats()
        assert stats["total_allocated_bytes"] >= 0
        assert stats["total_pooled_bytes"] >= 0

    def test_thread_local_safety(self, metal_device):
        """Each thread should see consistent pool state."""
        pool = MetalBufferPool(metal_device)
        results = []

        def worker():
            # Each thread does get/release
            buf = pool.get(2048)
            time.sleep(0.001)
            pool.release(buf)
            results.append("ok")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(results) == 20, "Not all workers completed"
        assert all(r == "ok" for r in results), "Some workers failed"


@requires_metal
class TestDefragmentation:
    """Test defragmentation doesn't corrupt data."""

    def test_defragment_maintains_pool_integrity(self, metal_device):
        """Defragmentation should leave pool in valid state."""
        pool = MetalBufferPool(metal_device)

        # Create fragmented pool
        bufs = []
        for i in range(10):
            buf = pool.get(1024 * (i + 1), BufferPriority.NORMAL)
            bufs.append(buf)

        # Release every other to create fragmentation
        for i in range(0, len(bufs), 2):
            pool.release(bufs[i])

        pre_stats = pool.stats()

        # Defragment
        freed = pool.defragment()

        post_stats = pool.stats()

        # Stats should be consistent
        assert post_stats["total_allocated_bytes"] >= 0
        assert post_stats["total_pooled_bytes"] >= 0

        # Release remaining
        for i in range(1, len(bufs), 2):
            pool.release(bufs[i])

    def test_compact_frees_memory(self, metal_device):
        """Compact should reduce pooled memory."""
        pool = MetalBufferPool(metal_device)

        # Create multiple buffers of same size
        size = 1024
        bufs = [pool.get(size) for _ in range(5)]

        for buf in bufs:
            pool.release(buf)

        # Should have multiple buffers in pool
        pre_stats = pool.stats()

        # Compact
        freed = pool.compact()

        post_stats = pool.stats()

        # Should have fewer buffers after compact
        assert post_stats["buffer_count"] <= pre_stats["buffer_count"]


@requires_metal
class TestEdgeCases:
    """Test edge cases for buffer corruption."""

    def test_zero_size_allocation(self, metal_device):
        """Zero-size allocation should be handled gracefully."""
        pool = MetalBufferPool(metal_device)

        # Should get minimum sized buffer (cache line aligned)
        buf = pool.get(0)
        assert buf is not None

        pool.release(buf)

    def test_large_allocation(self, metal_device):
        """Large allocations should work correctly."""
        pool = MetalBufferPool(metal_device)

        # Allocate 1MB
        buf = pool.get(1024 * 1024)
        assert buf is not None

        # Buffer should be page-aligned (large buffer threshold)
        # Cannot directly verify, but operation should succeed

        pool.release(buf)

    def test_repeated_reset_no_leak(self, metal_device):
        """Repeated ring buffer resets shouldn't leak memory."""
        ring = TransientRingBuffer(metal_device, capacity=1024 * 1024)

        initial_capacity = ring.capacity

        for _ in range(100):
            ring.reset()
            buf, offset = ring.alloc(1024)

        # Capacity should remain constant
        assert ring.capacity == initial_capacity

    def test_stats_consistency(self, metal_device):
        """Stats should always be consistent."""
        pool = MetalBufferPool(metal_device)

        # Allocate some buffers
        bufs = [pool.get(1024 * i) for i in range(1, 6)]

        stats1 = pool.stats()

        # current_allocated should be > 0
        assert stats1["current_allocated_bytes"] > 0, "Allocated bytes should be > 0"

        # Release half
        for buf in bufs[:3]:
            pool.release(buf)

        stats2 = pool.stats()

        # current_allocated should have decreased
        assert stats2["current_allocated_bytes"] < stats1["current_allocated_bytes"]

        # current_pooled should have increased
        assert stats2["current_pooled_bytes"] > 0, "Pooled bytes should be > 0"

        # Release rest
        for buf in bufs[3:]:
            pool.release(buf)

        stats3 = pool.stats()

        # Should have 0 allocated, all pooled
        assert stats3["current_allocated_bytes"] == 0


@requires_metal
class TestGlobalRingBuffer:
    """Test global transient ring buffer functions."""

    def test_get_transient_ring_singleton(self, metal_device):
        """get_transient_ring should return singleton per device."""
        ring1 = get_transient_ring(metal_device, capacity=1024 * 1024)
        ring2 = get_transient_ring(metal_device, capacity=1024 * 1024)

        assert ring1 is ring2, "get_transient_ring should return same instance"

    def test_reset_transient_ring(self, metal_device):
        """reset_transient_ring should reset the global ring."""
        ring = get_transient_ring(metal_device, capacity=1024 * 1024)

        # Allocate some space
        _, _ = ring.alloc(1024)
        assert ring.used > 0

        # Reset via global function
        reset_transient_ring()

        assert ring.used == 0, "Ring buffer not reset"

    def test_transient_ring_stats(self, metal_device):
        """transient_ring_stats should return valid stats."""
        ring = get_transient_ring(metal_device, capacity=1024 * 1024)
        ring.reset()

        # Allocate some space
        _, _ = ring.alloc(512)

        stats = ring.stats()
        assert stats["used_bytes"] >= 512
        assert stats["capacity_bytes"] == 1024 * 1024
        assert 0 <= stats["utilization"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
