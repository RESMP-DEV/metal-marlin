"""Memory leak detection for buffer pool.

Tests for GPU memory growth across 100 forward passes.
"""

import gc
import weakref

import pytest
import torch

from metal_marlin._buffer_pool import MetalBufferPool
from metal_marlin.quantized_linear import QuantizedLinear
from metal_marlin.quantized_loader import QuantizedTensor


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
class TestMemoryLeaks:
    """Memory leak detection tests."""

    def test_buffer_pool_no_leak_across_forward_passes(self):
        """Test that buffer pool doesn't leak memory over 100 forward passes."""
        device = torch.device("mps")

        # Create a simple quantized layer
        batch_size = 4
        seq_len = 128
        in_features = 512
        out_features = 512

        # Create quantized weight (FP4 format)
        weight_data = torch.randint(0, 15, (out_features, in_features // 2),
                                    dtype=torch.uint8, device=device)
        weight_scales = torch.randn(out_features, in_features // 128,
                                    dtype=torch.float16, device=device)

        quantized_weight = QuantizedTensor(
            data=weight_data,
            scales=weight_scales,
            format="fp4",
            group_size=128,
            original_shape=(out_features, in_features),
        )

        layer = QuantizedLinear(quantized_weight).to(device)

        # Get initial memory
        torch.mps.synchronize()
        gc.collect()
        initial_memory = torch.mps.current_allocated_memory()

        # Track memory across 100 forward passes
        memory_samples = []

        for i in range(100):
            # Create input
            x = torch.randn(batch_size, seq_len, in_features,
                           dtype=torch.float16, device=device)

            # Forward pass
            out = layer(x)

            # Force completion
            torch.mps.synchronize()

            # Sample memory every 10 iterations
            if i % 10 == 0:
                gc.collect()
                current_memory = torch.mps.current_allocated_memory()
                memory_samples.append(current_memory)

        # Final memory check
        torch.mps.synchronize()
        gc.collect()
        final_memory = torch.mps.current_allocated_memory()

        # Calculate memory growth
        memory_growth = final_memory - initial_memory
        growth_percent = (memory_growth / initial_memory) * 100 if initial_memory > 0 else 0

        # Assert memory growth is less than 1%
        assert growth_percent < 1.0, (
            f"Memory leak detected: {growth_percent:.2f}% growth "
            f"({memory_growth / (1024**2):.2f} MB) over 100 forward passes. "
            f"Memory samples: {[m // (1024**2) for m in memory_samples]} MB"
        )

    def test_buffer_pool_weak_references(self):
        """Test that buffers are released when no strong references exist."""
        device = torch.device("mps")
        pool = MetalBufferPool.get_instance()

        # Clear pool
        pool.clear()

        # Allocate a buffer
        size = 1024 * 1024  # 1 MB
        buffer = pool.acquire(size, priority=1)

        # Create weak reference
        weak_ref = weakref.ref(buffer)

        # Verify buffer exists
        assert weak_ref() is not None

        # Release to pool
        pool.release(buffer, priority=1)

        # Buffer should still exist in pool
        assert weak_ref() is not None

        # Clear pool
        pool.clear()

        # Delete buffer reference
        del buffer
        gc.collect()

        # Weak reference should now be None (buffer was garbage collected)
        assert weak_ref() is None, "Buffer was not garbage collected (orphaned)"

    def test_buffer_pool_no_orphaned_buffers(self):
        """Test that no buffers are orphaned after many acquire/release cycles."""
        device = torch.device("mps")
        pool = MetalBufferPool.get_instance()

        # Clear pool
        pool.clear()

        # Track weak references
        weak_refs = []

        # Acquire and release buffers
        for i in range(50):
            size = (i + 1) * 1024  # Varying sizes
            buffer = pool.acquire(size, priority=1)
            weak_refs.append(weakref.ref(buffer))
            pool.release(buffer, priority=1)
            del buffer

        # Clear pool
        pool.clear()
        gc.collect()

        # Count orphaned buffers
        orphaned = sum(1 for ref in weak_refs if ref() is not None)

        assert orphaned == 0, (
            f"{orphaned}/50 buffers were orphaned and not released"
        )

    def test_buffer_pool_memory_growth_rate(self):
        """Test that memory growth rate is bounded across forward passes."""
        device = torch.device("mps")

        # Create a simple quantized layer
        batch_size = 4
        seq_len = 128
        in_features = 512
        out_features = 512

        # Create quantized weight (FP4 format)
        weight_data = torch.randint(0, 15, (out_features, in_features // 2),
                                    dtype=torch.uint8, device=device)
        weight_scales = torch.randn(out_features, in_features // 128,
                                    dtype=torch.float16, device=device)

        quantized_weight = QuantizedTensor(
            data=weight_data,
            scales=weight_scales,
            format="fp4",
            group_size=128,
            original_shape=(out_features, in_features),
        )

        layer = QuantizedLinear(quantized_weight).to(device)

        # Get baseline memory after first pass
        x = torch.randn(batch_size, seq_len, in_features,
                       dtype=torch.float16, device=device)
        _ = layer(x)
        torch.mps.synchronize()
        gc.collect()
        baseline_memory = torch.mps.current_allocated_memory()

        # Track memory across subsequent passes
        memory_samples = []

        for i in range(99):  # 99 more passes (100 total)
            x = torch.randn(batch_size, seq_len, in_features,
                           dtype=torch.float16, device=device)
            _ = layer(x)

            if i % 10 == 9:
                torch.mps.synchronize()
                gc.collect()
                current_memory = torch.mps.current_allocated_memory()
                memory_samples.append(current_memory)

        # Check that memory is stable (within 1% of baseline)
        for i, sample in enumerate(memory_samples):
            growth = sample - baseline_memory
            growth_percent = (growth / baseline_memory) * 100 if baseline_memory > 0 else 0

            assert growth_percent < 1.0, (
                f"Memory leak at pass {(i+1)*10}: {growth_percent:.2f}% growth "
                f"({growth / (1024**2):.2f} MB above baseline)"
            )

    def test_buffer_pool_no_fragmentation_leak(self):
        """Test that buffer pool fragmentation doesn't grow unbounded."""
        device = torch.device("mps")
        pool = MetalBufferPool.get_instance()

        # Clear pool
        pool.clear()

        # Get initial metrics
        initial_metrics = pool.get_metrics()
        initial_fragmentation = initial_metrics.fragmentation_ratio

        # Allocate and release buffers of varying sizes (worst case for fragmentation)
        for _ in range(100):
            sizes = [1024, 2048, 512, 4096, 256, 8192]  # Random sizes
            buffers = [pool.acquire(s, priority=1) for s in sizes]

            # Release in reverse order (creates fragmentation)
            for buf in reversed(buffers):
                pool.release(buf, priority=1)

        # Get final metrics
        final_metrics = pool.get_metrics()
        final_fragmentation = final_metrics.fragmentation_ratio

        # Fragmentation should not grow unbounded
        # Allow up to 50% fragmentation (pool should defragment)
        assert final_fragmentation < 0.5, (
            f"Fragmentation grew unbounded: {final_fragmentation:.2%} "
            f"(initial: {initial_fragmentation:.2%})"
        )
