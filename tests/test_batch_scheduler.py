"""Tests for BatchScheduler dynamic insertion capabilities.

Tests the BatchScheduler implementation in scheduler.py for:
- Dynamic single request insertion
- Batch insertion with multiple policies
- Priority insertion
- Queue capacity management
"""

import pytest

from metal_marlin.paged.allocator import BlockAllocator
from metal_marlin.serving.request import GenerationRequest, RequestStatus
from metal_marlin.serving.scheduler import (
    BatchScheduler,
    InsertionPolicy,
    QueueFullError,
    SchedulerConfig,
)


@pytest.fixture
def config():
    """Default scheduler configuration."""
    return SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=512,
        max_prefill_tokens=256,
        block_size=16,
    )


@pytest.fixture
def allocator():
    """Block allocator with sufficient capacity."""
    return BlockAllocator(num_blocks=128)


@pytest.fixture
def scheduler(config, allocator):
    """BatchScheduler instance for testing."""
    return BatchScheduler(config, allocator, max_queue_size=10)


def make_request(request_id: str, num_tokens: int = 4) -> GenerationRequest:
    """Helper to create a test request."""
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=list(range(num_tokens)),
        max_tokens=16,
    )


class TestBatchSchedulerDynamicInsertion:
    """Test dynamic insertion capabilities."""

    def test_single_request_insertion(self, scheduler):
        """Test adding a single request dynamically."""
        req = make_request("req-1")
        scheduler.add_request(req)

        assert scheduler.num_waiting == 1
        assert req.status == RequestStatus.PENDING
        assert scheduler.insertion_stats["total_inserted"] == 1

    def test_multiple_single_insertions(self, scheduler):
        """Test adding multiple requests one at a time."""
        for i in range(5):
            req = make_request(f"req-{i}")
            scheduler.add_request(req)

        assert scheduler.num_waiting == 5
        assert scheduler.insertion_stats["total_inserted"] == 5

    def test_batch_insertion_enqueue(self, scheduler):
        """Test batch insertion with ENQUEUE policy."""
        batch = [make_request(f"req-{i}") for i in range(3)]
        inserted = scheduler.insert_batch(batch, InsertionPolicy.ENQUEUE)

        assert inserted == 3
        assert scheduler.num_waiting == 3
        assert scheduler.insertion_stats["total_inserted"] == 3
        assert scheduler.insertion_stats["batch_insertions"] == 1

    def test_batch_insertion_merge(self, scheduler):
        """Test batch insertion with MERGE policy (front insertion)."""
        # Add some background requests
        for i in range(3):
            scheduler.add_request(make_request(f"bg-{i}"))

        # Insert batch at front
        batch = [make_request(f"priority-{i}") for i in range(2)]
        inserted = scheduler.insert_batch(batch, InsertionPolicy.MERGE)

        assert inserted == 2
        assert scheduler.num_waiting == 5

    def test_batch_insertion_drop_if_full(self, scheduler):
        """Test batch insertion with DROP_IF_FULL policy."""
        # Fill the queue to near capacity (max_queue_size = 10)
        for i in range(9):
            scheduler.add_request(make_request(f"req-{i}"))

        # Try to insert batch that won't fit
        batch = [make_request(f"overflow-{i}") for i in range(3)]
        inserted = scheduler.insert_batch(batch, InsertionPolicy.DROP_IF_FULL)

        assert inserted == 0  # None inserted because batch doesn't fit
        assert scheduler.num_waiting == 9  # Original queue unchanged
        assert scheduler.insertion_stats["total_dropped"] == 3

    def test_add_request_front(self, scheduler):
        """Test priority insertion at front of queue."""
        # Add background requests
        scheduler.add_request(make_request("bg-1"))
        scheduler.add_request(make_request("bg-2"))

        # Add priority request at front
        priority_req = make_request("priority")
        scheduler.add_request_front(priority_req)

        assert scheduler.num_waiting == 3
        # Note: Can't easily verify order without queue introspection

    def test_insert_batch_front(self, scheduler):
        """Test batch priority insertion."""
        scheduler.add_request(make_request("bg-1"))

        batch = [make_request(f"priority-{i}") for i in range(2)]
        inserted = scheduler.insert_batch_front(batch)

        assert inserted == 2
        assert scheduler.num_waiting == 3


class TestQueueCapacityManagement:
    """Test queue capacity limits and overflow handling."""

    def test_queue_full_single_insertion(self, scheduler):
        """Test that add_request raises error when queue is full."""
        # Fill queue to capacity (max_queue_size = 10)
        for i in range(10):
            scheduler.add_request(make_request(f"req-{i}"))

        # Next insertion should raise error
        with pytest.raises(QueueFullError):
            scheduler.add_request(make_request("overflow"))

    def test_queue_utilization(self, scheduler):
        """Test queue utilization metric."""
        assert scheduler.queue_utilization == 0.0

        # Add 5 requests (50% capacity)
        for i in range(5):
            scheduler.add_request(make_request(f"req-{i}"))

        assert scheduler.queue_utilization == 0.5

        # Fill to capacity
        for i in range(5, 10):
            scheduler.add_request(make_request(f"req-{i}"))

        assert scheduler.queue_utilization == 1.0

    def test_clear_waiting(self, scheduler):
        """Test clearing waiting queue."""
        for i in range(5):
            scheduler.add_request(make_request(f"req-{i}"))

        assert scheduler.num_waiting == 5

        cleared = scheduler.clear_waiting()
        assert cleared == 5
        assert scheduler.num_waiting == 0


class TestInsertionStatistics:
    """Test insertion statistics tracking."""

    def test_stats_single_insertions(self, scheduler):
        """Test stats for single request insertions."""
        for i in range(3):
            scheduler.add_request(make_request(f"req-{i}"))

        stats = scheduler.insertion_stats
        assert stats["total_inserted"] == 3
        assert stats["total_dropped"] == 0
        assert stats["batch_insertions"] == 0

    def test_stats_batch_insertions(self, scheduler):
        """Test stats for batch insertions."""
        batch1 = [make_request(f"b1-{i}") for i in range(3)]
        batch2 = [make_request(f"b2-{i}") for i in range(2)]

        scheduler.insert_batch(batch1)
        scheduler.insert_batch(batch2)

        stats = scheduler.insertion_stats
        assert stats["total_inserted"] == 5
        assert stats["batch_insertions"] == 2

    def test_stats_with_drops(self, scheduler):
        """Test stats tracking dropped requests."""
        # Fill queue
        for i in range(10):
            scheduler.add_request(make_request(f"req-{i}"))

        # Try to insert batch that will be dropped
        batch = [make_request(f"overflow-{i}") for i in range(3)]
        scheduler.insert_batch(batch, InsertionPolicy.DROP_IF_FULL)

        stats = scheduler.insertion_stats
        assert stats["total_inserted"] == 10
        assert stats["total_dropped"] == 3

    def test_reset_stats(self, scheduler):
        """Test resetting insertion statistics."""
        scheduler.add_request(make_request("req-1"))
        scheduler.insert_batch([make_request("req-2")])

        scheduler.reset_insertion_stats()

        stats = scheduler.insertion_stats
        assert stats["total_inserted"] == 0
        assert stats["total_dropped"] == 0
        assert stats["batch_insertions"] == 0


class TestSchedulingWithDynamicInsertion:
    """Test that scheduling works correctly with dynamically inserted requests."""

    def test_schedule_after_dynamic_insertion(self, scheduler):
        """Test scheduling after dynamic insertions."""
        # Add requests dynamically
        for i in range(3):
            scheduler.add_request(make_request(f"req-{i}", num_tokens=16))

        # Schedule should work normally
        output = scheduler.schedule()

        # Should have some prefill work
        assert len(output.prefill_requests) > 0
        assert output.num_prefill_tokens > 0

    def test_schedule_mixed_insertion_methods(self, scheduler):
        """Test scheduling with mixed insertion methods."""
        # Add via different methods
        scheduler.add_request(make_request("single-1", num_tokens=8))

        batch = [make_request(f"batch-{i}", num_tokens=8) for i in range(2)]
        scheduler.insert_batch(batch)

        scheduler.add_request_front(make_request("priority", num_tokens=8))

        # Schedule
        output = scheduler.schedule()
        assert scheduler.num_waiting < 4  # Some should be scheduled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
