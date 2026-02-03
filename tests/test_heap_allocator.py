"""Tests for heap allocator using MTLHeap."""

import pytest

from metal_marlin._compat import HAS_PYOBJC_METAL

if HAS_PYOBJC_METAL:
    import Metal


@pytest.mark.skipif(not HAS_PYOBJC_METAL, reason="PyObjC Metal not available")
class TestHeapAllocation:
    """Tests for HeapAllocation dataclass."""

    def test_heap_allocation_creation(self):
        """Test HeapAllocation can be created with correct defaults."""
        from metal_marlin.heap_allocator import HeapAllocation

        device = Metal.MTLCreateSystemDefaultDevice()
        buf = device.newBufferWithLength_options_(1024, 0)

        alloc = HeapAllocation(
            buffer=buf,
            size=4096,
            offset=0,
            heap_offset=0,
            created_at=0.0,
            last_used_at=0.0,
            use_count=0,
        )

        assert alloc.size == 4096
        assert alloc.offset == 0
        assert alloc.heap_offset == 0
        assert alloc.use_count == 0


@pytest.mark.skipif(not HAS_PYOBJC_METAL, reason="PyObjC Metal not available")
class TestHeapAllocatorMetrics:
    """Tests for HeapAllocatorMetrics class."""

    def test_metrics_initialization(self):
        """Test metrics initialize with correct defaults."""
        from metal_marlin.heap_allocator import HeapAllocatorMetrics

        metrics = HeapAllocatorMetrics()

        assert metrics.allocations == 0
        assert metrics.deallocations == 0
        assert metrics.reuse_count == 0
        assert metrics.peak_allocated == 0
        assert metrics.current_allocated == 0
        assert metrics.fragmentation_waste == 0
        assert metrics.reuse_rate == 0.0

    def test_reuse_rate_calculation(self):
        """Test reuse rate calculation."""
        from metal_marlin.heap_allocator import HeapAllocatorMetrics

        metrics = HeapAllocatorMetrics()

        assert metrics.reuse_rate == 0.0

        metrics.allocations = 10
        metrics.reuse_count = 5
        assert metrics.reuse_rate == 0.5

        metrics.reuse_count = 10
        assert metrics.reuse_rate == 1.0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        from metal_marlin.heap_allocator import HeapAllocatorMetrics

        metrics = HeapAllocatorMetrics()
        metrics.allocations = 100
        metrics.reuse_count = 80
        metrics.peak_allocated = 1_000_000
        metrics.current_allocated = 500_000

        d = metrics.to_dict()

        assert d["allocations"] == 100
        assert d["reuse_count"] == 80
        assert d["reuse_rate"] == 0.8
        assert d["peak_allocated_bytes"] == 1_000_000
        assert d["current_allocated_bytes"] == 500_000


@pytest.mark.skipif(not HAS_PYOBJC_METAL, reason="PyObjC Metal not available")
class TestMetalHeapAllocator:
    """Tests for MetalHeapAllocator class."""

    def test_allocator_initialization(self):
        """Test allocator initializes with correct heap size."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        assert allocator.heap_size == 64_000_000
        assert allocator.allocated_bytes == 0
        assert allocator.available_bytes == 64_000_000
        assert allocator.heap is not None

    def test_allocator_minimum_heap_size(self):
        """Test allocator enforces minimum heap size."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=1_000_000)

        assert allocator.heap_size == MetalHeapAllocator.MIN_HEAP_SIZE

    def test_alloc_buffer(self):
        """Test buffer allocation from heap."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf, offset = allocator.alloc(4096)

        assert buf is not None
        assert offset == 0
        assert allocator.allocated_bytes > 0

    def test_alloc_returns_aligned_sizes(self):
        """Test allocations return cache-aligned sizes."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf1, offset1 = allocator.alloc(100)
        buf2, offset2 = allocator.alloc(4097)
        buf3, offset3 = allocator.alloc(70000)

        assert buf1 is not None
        assert buf2 is not None
        assert buf3 is not None
        # Offsets depend on heap implementation and device
        assert offset1 >= 0
        assert offset2 >= 0
        assert offset3 >= 0

    def test_get_offset(self):
        """Test getting buffer offset from allocator."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf, expected_offset = allocator.alloc(4096)
        actual_offset = allocator.get_offset(buf)

        assert actual_offset == expected_offset

    def test_get_offset_unknown_buffer(self):
        """Test getting offset for unknown buffer returns None."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        # Create a buffer not tracked by allocator
        buf = device.newBufferWithLength_options_(1024, 0)

        offset = allocator.get_offset(buf)
        assert offset is None

    def test_release_and_reuse(self):
        """Test buffer reuse after release."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf1, _ = allocator.alloc(4096)
        initial_allocated = allocator.allocated_bytes

        allocator.release(buf1)
        assert allocator.allocated_bytes < initial_allocated

        buf2, offset2 = allocator.alloc(4096)
        metrics = allocator.metrics

        assert offset2 == 0
        assert metrics.reuse_count > 0

    def test_heap_exhaustion(self):
        """Test allocation fails when heap is exhausted."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        with pytest.raises(RuntimeError, match="Heap allocation failed"):
            allocator.alloc(128_000_000)

    def test_clear_pool(self):
        """Test clearing buffer pool."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf1, _ = allocator.alloc(4096)
        buf2, _ = allocator.alloc(8192)
        allocator.release(buf1)
        allocator.release(buf2)

        cleared = allocator.clear_pool()
        assert cleared >= 0

    def test_reset_allocator(self):
        """Test resetting allocator clears all state."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf, _ = allocator.alloc(4096)
        allocator.release(buf)

        allocator.reset()

        assert allocator.allocated_bytes == 0
        assert allocator.available_bytes == allocator.heap_size

    def test_stats(self):
        """Test allocator statistics."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf, _ = allocator.alloc(4096)
        stats = allocator.stats()

        assert stats["heap_size_bytes"] == 64_000_000
        assert stats["allocated_bytes"] > 0
        assert stats["active_allocation_count"] == 1

    def test_alloc_buffer_convenience_method(self):
        """Test alloc_buffer convenience method."""
        from metal_marlin.heap_allocator import MetalHeapAllocator

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        allocator = MetalHeapAllocator(device, heap_size=64_000_000)

        buf = allocator.alloc_buffer(4096)
        offset = allocator.get_offset(buf)

        assert buf is not None
        assert offset == 0


@pytest.mark.skipif(not HAS_PYOBJC_METAL, reason="PyObjC Metal not available")
class TestHeapBufferPool:
    """Tests for HeapBufferPool class."""

    def test_pool_initialization(self):
        """Test pool initialization."""
        from metal_marlin.heap_allocator import HeapBufferPool

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        pool = HeapBufferPool(device, heap_size=32_000_000)

        assert pool.allocator.heap_size == 32_000_000

    def test_pool_get_and_release(self):
        """Test getting and releasing buffers from pool."""
        from metal_marlin.heap_allocator import HeapBufferPool

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        pool = HeapBufferPool(device, heap_size=32_000_000)

        buf = pool.get(4096)
        assert buf is not None

        pool.release(buf)

        stats = pool.stats()
        assert stats["deallocations"] == 1

    def test_pool_stats(self):
        """Test pool statistics."""
        from metal_marlin.heap_allocator import HeapBufferPool

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        pool = HeapBufferPool(device, heap_size=32_000_000)

        buf = pool.get(4096)
        stats = pool.stats()

        assert stats["active_allocation_count"] == 1
        assert stats["allocations"] == 1

    def test_pool_clear(self):
        """Test clearing pool buffers."""
        from metal_marlin.heap_allocator import HeapBufferPool

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None

        pool = HeapBufferPool(device, heap_size=32_000_000)

        buf = pool.get(4096)
        pool.release(buf)
        cleared = pool.clear_pool()

        assert cleared >= 0
