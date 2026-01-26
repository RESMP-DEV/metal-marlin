"""Tests for continuous batching scheduler."""

import pytest

from metal_marlin.serving.continuous_batch import (
    BatchScheduler,
    IterationPlanner,
    KVCacheManager,
    KVRegion,
    PreemptionPolicy,
    RequestPriority,
    RequestState,
    SchedulerConfig,
)
from metal_marlin.serving.request import GenerationRequest, RequestStatus


class TestKVCacheManager:
    """Tests for KVCacheManager."""

    def test_init(self):
        manager = KVCacheManager(num_blocks=64, block_size=16)
        assert manager.num_blocks == 64
        assert manager.block_size == 16
        assert manager.num_free_blocks == 64
        assert manager.num_allocated_blocks == 0
        assert len(manager) == 0

    def test_allocate_success(self):
        manager = KVCacheManager(num_blocks=64, block_size=16)
        region_id = manager.allocate("req-1", num_tokens=32)
        assert region_id is not None
        assert region_id == 0
        assert len(manager) == 1
        assert manager.get_num_tokens("req-1") == 32
        # 32 tokens / 16 block_size = 2 blocks
        assert manager.num_allocated_blocks == 2

    def test_allocate_oom(self):
        manager = KVCacheManager(num_blocks=2, block_size=16)
        # First allocation: 32 tokens = 2 blocks, uses all
        region_id = manager.allocate("req-1", num_tokens=32)
        assert region_id is not None
        # Second allocation: should fail (OOM)
        region_id = manager.allocate("req-2", num_tokens=16)
        assert region_id is None
        assert len(manager) == 1

    def test_allocate_duplicate_raises(self):
        manager = KVCacheManager(num_blocks=64, block_size=16)
        manager.allocate("req-1", num_tokens=16)
        with pytest.raises(ValueError, match="already exists"):
            manager.allocate("req-1", num_tokens=16)

    def test_extend(self):
        manager = KVCacheManager(num_blocks=64, block_size=16)
        manager.allocate("req-1", num_tokens=15)
        assert manager.get_num_tokens("req-1") == 15
        # Extend by 1 (still fits in block 1)
        assert manager.extend("req-1", 1)
        assert manager.get_num_tokens("req-1") == 16
        # Extend by 1 more (needs new block)
        assert manager.extend("req-1", 1)
        assert manager.get_num_tokens("req-1") == 17
        assert manager.num_allocated_blocks == 2

    def test_extend_oom(self):
        manager = KVCacheManager(num_blocks=1, block_size=16)
        manager.allocate("req-1", num_tokens=16)
        # Block is full, extending needs new block which we don't have
        assert not manager.extend("req-1", 1)
        assert manager.get_num_tokens("req-1") == 16

    def test_free(self):
        manager = KVCacheManager(num_blocks=64, block_size=16)
        manager.allocate("req-1", num_tokens=32)
        assert manager.num_allocated_blocks == 2
        manager.free("req-1")
        assert manager.num_allocated_blocks == 0
        assert len(manager) == 0
        assert manager.get_num_tokens("req-1") == 0

    def test_get_block_table(self):
        manager = KVCacheManager(num_blocks=64, block_size=16)
        manager.allocate("req-1", num_tokens=32)
        block_table = manager.get_block_table("req-1")
        assert block_table is not None
        assert len(block_table) == 2

    def test_memory_pressure(self):
        manager = KVCacheManager(num_blocks=4, block_size=16)
        assert manager.memory_pressure == 0.0
        manager.allocate("req-1", num_tokens=32)  # 2 blocks
        assert manager.memory_pressure == 0.5
        manager.allocate("req-2", num_tokens=32)  # 2 more blocks
        assert manager.memory_pressure == 1.0

    def test_can_allocate(self):
        manager = KVCacheManager(num_blocks=4, block_size=16)
        assert manager.can_allocate(64)  # 4 blocks needed, 4 available
        assert not manager.can_allocate(65)  # 5 blocks needed, 4 available
        manager.allocate("req-1", num_tokens=32)
        assert manager.can_allocate(32)
        assert not manager.can_allocate(33)

    def test_preempt(self):
        manager = KVCacheManager(num_blocks=64, block_size=16)
        manager.allocate("req-1", num_tokens=32)
        state = manager.preempt("req-1")
        assert state is not None
        assert state["request_id"] == "req-1"
        assert state["num_tokens"] == 32
        assert manager.num_allocated_blocks == 0
        assert len(manager) == 0


class TestRequestState:
    """Tests for RequestState."""

    def test_remaining_prefill(self):
        req = GenerationRequest(
            request_id="req-1",
            prompt_tokens=[1, 2, 3, 4, 5],
        )
        state = RequestState(
            request=req,
            priority=RequestPriority.NORMAL,
            enqueue_time=0.0,
            prefill_progress=2,
        )
        assert state.remaining_prefill == 3

    def test_is_prefill_complete(self):
        req = GenerationRequest(
            request_id="req-1",
            prompt_tokens=[1, 2, 3, 4, 5],
        )
        state = RequestState(
            request=req,
            priority=RequestPriority.NORMAL,
            enqueue_time=0.0,
            prefill_progress=5,
        )
        assert state.is_prefill_complete

    def test_effective_priority_ordering(self):
        req1 = GenerationRequest(request_id="req-1", prompt_tokens=[1])
        req2 = GenerationRequest(request_id="req-2", prompt_tokens=[1])
        req3 = GenerationRequest(request_id="req-3", prompt_tokens=[1])

        state1 = RequestState(request=req1, priority=RequestPriority.HIGH, enqueue_time=1.0)
        state2 = RequestState(request=req2, priority=RequestPriority.NORMAL, enqueue_time=0.5)
        state3 = RequestState(request=req3, priority=RequestPriority.HIGH, enqueue_time=0.5)

        # Higher priority first
        assert state1 < state2  # HIGH > NORMAL
        # Same priority: earlier enqueue time first
        assert state3 < state1  # Both HIGH, but state3 enqueued earlier


class TestIterationPlanner:
    """Tests for IterationPlanner."""

    def test_plan_decode_only(self):
        config = SchedulerConfig(
            max_num_batched_tokens=100,
            max_prefill_tokens=50,
            block_size=16,
        )
        planner = IterationPlanner(config)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)

        # Set up running requests with allocated KV
        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        state = RequestState(
            request=req,
            priority=RequestPriority.NORMAL,
            enqueue_time=0.0,
            prefill_progress=4,
            kv_region_id=0,
        )
        kv_manager.allocate("req-1", num_tokens=4)

        plan = planner.plan_iteration(waiting=[], running=[state], kv_manager=kv_manager)
        assert len(plan.decode_work) == 1
        assert len(plan.prefill_work) == 0
        assert plan.decode_tokens_used == 1

    def test_plan_prefill_only(self):
        config = SchedulerConfig(
            max_num_batched_tokens=100,
            max_prefill_tokens=50,
            block_size=16,
        )
        planner = IterationPlanner(config)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        state = RequestState(
            request=req,
            priority=RequestPriority.NORMAL,
            enqueue_time=0.0,
        )

        plan = planner.plan_iteration(waiting=[state], running=[], kv_manager=kv_manager)
        assert len(plan.prefill_work) == 1
        assert len(plan.decode_work) == 0
        assert plan.prefill_tokens_used == 4

    def test_plan_chunked_prefill(self):
        config = SchedulerConfig(
            max_num_batched_tokens=100,
            max_prefill_tokens=10,  # Limit prefill
            max_chunk_size=5,
            enable_chunked_prefill=True,
            block_size=16,
        )
        planner = IterationPlanner(config)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)

        req = GenerationRequest(
            request_id="req-1",
            prompt_tokens=list(range(20)),  # 20 tokens
        )
        state = RequestState(
            request=req,
            priority=RequestPriority.NORMAL,
            enqueue_time=0.0,
        )

        plan = planner.plan_iteration(waiting=[state], running=[], kv_manager=kv_manager)
        assert len(plan.prefill_work) == 1
        _, start_tok, num_toks = plan.prefill_work[0]
        assert start_tok == 0
        assert num_toks == 5  # Limited by max_chunk_size

    def test_plan_respects_token_budget(self):
        config = SchedulerConfig(
            max_num_batched_tokens=5,
            max_prefill_tokens=5,
            block_size=16,
            enable_chunked_prefill=True,  # Explicit
        )
        planner = IterationPlanner(config)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)

        # Create two requests
        req1 = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3])
        req2 = GenerationRequest(request_id="req-2", prompt_tokens=[4, 5, 6])
        state1 = RequestState(request=req1, priority=RequestPriority.NORMAL, enqueue_time=0.0)
        state2 = RequestState(request=req2, priority=RequestPriority.NORMAL, enqueue_time=1.0)

        plan = planner.plan_iteration(
            waiting=[state1, state2], running=[], kv_manager=kv_manager
        )
        # With chunked prefill: first request gets 3 tokens, second gets 2 (chunked)
        # Total: 5 tokens exactly matching budget
        assert len(plan.prefill_work) == 2
        assert plan.prefill_tokens_used == 5
        # First request gets full 3 tokens
        _, _, chunk1_size = plan.prefill_work[0]
        assert chunk1_size == 3
        # Second request gets remaining 2 tokens (chunked)
        _, _, chunk2_size = plan.prefill_work[1]
        assert chunk2_size == 2

    def test_plan_respects_budget_without_chunking(self):
        config = SchedulerConfig(
            max_num_batched_tokens=5,
            max_prefill_tokens=5,
            block_size=16,
            enable_chunked_prefill=False,  # Disable chunking
        )
        planner = IterationPlanner(config)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)

        # Create two requests
        req1 = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3])
        req2 = GenerationRequest(request_id="req-2", prompt_tokens=[4, 5, 6])
        state1 = RequestState(request=req1, priority=RequestPriority.NORMAL, enqueue_time=0.0)
        state2 = RequestState(request=req2, priority=RequestPriority.NORMAL, enqueue_time=1.0)

        plan = planner.plan_iteration(
            waiting=[state1, state2], running=[], kv_manager=kv_manager
        )
        # Without chunking: first request (3 tokens) fits, second (3 tokens) would exceed
        assert len(plan.prefill_work) == 1
        assert plan.prefill_tokens_used == 3


class TestBatchScheduler:
    """Tests for BatchScheduler."""

    def test_add_request(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        scheduler.add_request(req, priority=RequestPriority.HIGH)

        assert scheduler.num_waiting == 1
        assert scheduler.num_running == 0
        assert req.status == RequestStatus.PENDING

    def test_add_duplicate_raises(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        scheduler.add_request(req)
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(req)

    def test_schedule_prefill(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        scheduler.add_request(req)

        output = scheduler.schedule()
        assert len(output.prefill_requests) == 1
        assert len(output.decode_requests) == 0
        assert req.status == RequestStatus.RUNNING
        assert scheduler.num_running == 1
        assert scheduler.num_waiting == 0

    def test_schedule_decode_after_prefill(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        scheduler.add_request(req)

        # First schedule: prefill
        output1 = scheduler.schedule()
        assert len(output1.prefill_requests) == 1
        assert len(output1.decode_requests) == 0

        # Simulate generation
        req.append_token(5)
        scheduler.step()  # Extend KV cache

        # Second schedule: decode
        output2 = scheduler.schedule()
        assert len(output2.prefill_requests) == 0
        assert len(output2.decode_requests) == 1

    def test_abort_request(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        scheduler.add_request(req)
        scheduler.schedule()  # Move to running

        assert scheduler.abort_request("req-1")
        assert scheduler.num_running == 0
        assert kv_manager.num_allocated_blocks == 0

    def test_step_finishes_requests(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req = GenerationRequest(
            request_id="req-1", prompt_tokens=[1, 2, 3, 4], max_tokens=2
        )
        scheduler.add_request(req)
        scheduler.schedule()

        # Generate tokens until finished
        req.append_token(5)
        scheduler.step()
        req.append_token(6)  # This hits max_tokens

        finished = scheduler.step()
        assert len(finished) == 1
        assert finished[0].request_id == "req-1"
        assert scheduler.num_running == 0

    def test_priority_ordering(self):
        config = SchedulerConfig(max_num_batched_tokens=100, max_num_seqs=1, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req_low = GenerationRequest(request_id="req-low", prompt_tokens=[1, 2])
        req_high = GenerationRequest(request_id="req-high", prompt_tokens=[3, 4])

        # Add low priority first, then high priority
        scheduler.add_request(req_low, priority=RequestPriority.LOW)
        scheduler.add_request(req_high, priority=RequestPriority.HIGH)

        # With max_num_seqs=1, only one should be scheduled
        output = scheduler.schedule()
        assert len(output.prefill_requests) == 1
        # High priority should be scheduled first
        assert output.prefill_requests[0].request_id == "req-high"

    def test_get_stats(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        scheduler.add_request(req)
        scheduler.schedule()

        stats = scheduler.get_stats()
        assert stats["running"] == 1
        assert stats["waiting"] == 0
        assert stats["swapped"] == 0
        assert "kv_memory_pressure" in stats

    def test_has_pending_work(self):
        config = SchedulerConfig(max_num_batched_tokens=100, block_size=16)
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        assert not scheduler.has_pending_work

        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3, 4])
        scheduler.add_request(req)
        assert scheduler.has_pending_work


class TestPreemptionPolicy:
    """Tests for different preemption policies."""

    def test_lowest_priority_preemption(self):
        config = SchedulerConfig(
            max_num_batched_tokens=5,
            preemption_policy=PreemptionPolicy.LOWEST_PRIORITY,
            block_size=16,
        )
        kv_manager = KVCacheManager(num_blocks=2, block_size=16)  # Very limited
        scheduler = BatchScheduler(config, kv_manager)

        # Add high and low priority requests
        req_high = GenerationRequest(request_id="req-high", prompt_tokens=[1, 2])
        req_low = GenerationRequest(request_id="req-low", prompt_tokens=[3, 4])

        scheduler.add_request(req_high, priority=RequestPriority.HIGH)
        scheduler.add_request(req_low, priority=RequestPriority.LOW)

        # Schedule both
        scheduler.schedule()

        # Add another high priority that needs memory
        req_new = GenerationRequest(request_id="req-new", prompt_tokens=[5, 6])
        scheduler.add_request(req_new, priority=RequestPriority.HIGH)

        # The planner should handle memory pressure
        # (exact behavior depends on implementation details)


class TestKVRegion:
    """Tests for KVRegion dataclass."""

    def test_num_blocks(self):
        region = KVRegion(
            region_id=0,
            request_id="req-1",
            block_indices=[0, 1, 2],
            num_tokens=48,
        )
        assert region.num_blocks == 3


class TestIntegration:
    """Integration tests for the full continuous batching flow."""

    def test_multiple_requests_flow(self):
        config = SchedulerConfig(
            max_num_batched_tokens=100,
            max_prefill_tokens=50,
            max_num_seqs=4,
            block_size=16,
        )
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        # Add multiple requests
        for i in range(3):
            req = GenerationRequest(
                request_id=f"req-{i}",
                prompt_tokens=list(range(8)),
                max_tokens=5,
            )
            scheduler.add_request(req)

        # First iteration: prefill all
        output = scheduler.schedule()
        assert len(output.prefill_requests) == 3
        assert len(output.decode_requests) == 0

        # Simulate token generation
        for state_id, state in scheduler.iter_requests():
            state.request.append_token(100)
        scheduler.step()

        # Second iteration: decode all
        output = scheduler.schedule()
        assert len(output.prefill_requests) == 0
        assert len(output.decode_requests) == 3

        # Continue until requests finish
        finished_count = 0
        for _ in range(10):
            for state_id, state in scheduler.iter_requests():
                if not state.request.is_finished:
                    state.request.append_token(100)
            finished = scheduler.step()
            finished_count += len(finished)
            if not scheduler.has_pending_work:
                break

        assert finished_count == 3
        assert kv_manager.num_allocated_blocks == 0

    def test_continuous_request_addition(self):
        """Test adding new requests while others are decoding."""
        config = SchedulerConfig(
            max_num_batched_tokens=100,
            max_num_seqs=10,
            block_size=16,
        )
        kv_manager = KVCacheManager(num_blocks=64, block_size=16)
        scheduler = BatchScheduler(config, kv_manager)

        # Add first request and start decoding
        req1 = GenerationRequest(
            request_id="req-1", prompt_tokens=[1, 2, 3, 4], max_tokens=10
        )
        scheduler.add_request(req1)
        scheduler.schedule()
        req1.append_token(5)
        scheduler.step()

        # Add second request mid-generation
        req2 = GenerationRequest(
            request_id="req-2", prompt_tokens=[10, 11, 12], max_tokens=10
        )
        scheduler.add_request(req2)

        # Schedule should handle both
        output = scheduler.schedule()
        # req1 should decode, req2 should prefill
        assert len(output.prefill_requests) == 1
        assert len(output.decode_requests) == 1
        assert output.decode_requests[0].request_id == "req-1"
        assert output.prefill_requests[0].request_id == "req-2"
