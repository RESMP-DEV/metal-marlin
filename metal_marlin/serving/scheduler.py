"""Batch schedulers with preemption and dynamic request insertion.

Follows vLLM's scheduler design with three queues:
- WAITING: New sequences pending prefill
- RUNNING: Sequences actively generating tokens
- SWAPPED: Preempted sequences (logically CPU-offloaded)

Each iteration, the scheduler:
1. Allocates decode budget for running sequences (1 token each)
2. Fills remaining budget with waiting sequences (prefill)
3. Preempts running sequences if memory is exhausted

Prefix Caching:
The scheduler supports prefix caching to reuse KV cache blocks for common
prompt prefixes across multiple requests. Cached blocks are identified by
hashing token sequences and can be shared across requests.

Usage:
    from metal_marlin.serving.scheduler import (
        FCFSScheduler, BatchScheduler, SchedulerConfig,
    )
    from metal_marlin.paged.allocator import BlockAllocator

    # FCFS scheduler with prefix caching
    alloc = BlockAllocator(num_blocks=512)
    sched = FCFSScheduler(SchedulerConfig(block_size=16, enable_prefix_cache=True), alloc)
    sched.add_request(request)
    output = sched.schedule()

    # Batch scheduler with dynamic insertion
    sched = BatchScheduler(SchedulerConfig(block_size=16), alloc)
    sched.add_request(request)  # Dynamic insertion at any time
    sched.insert_batch([req1, req2, req3])  # Batch insertion
    output = sched.schedule()
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256

from ..paged.allocator import BlockAllocator
from ..paged.page_table import PageTable
from .request import GenerationRequest, RequestStatus, SchedulerOutput


@dataclass
class CachedPrefix:
    """Represents a cached prefix with its KV blocks and metadata.

    Attributes:
        token_hash: Hash of the token sequence for identification
        tokens: The actual token sequence (for verification and extension)
        block_indices: List of physical block indices containing cached KV
        num_blocks: Number of blocks used by this prefix
        ref_count: Number of currently active requests using this cache
        last_used: Timestamp of last access (for LRU eviction)
        seq_id: The sequence ID that produced this cache entry
    """

    token_hash: int
    tokens: list[int]
    block_indices: list[int]
    ref_count: int = 0
    last_used: float = 0.0
    seq_id: int = 0  # Source sequence ID that produced this cache

    @property
    def num_blocks(self) -> int:
        return len(self.block_indices)


class PrefixCache:
    """Manages KV cache blocks for common prompt prefixes across requests.

    Implements prefix caching to avoid recomputing attention for identical
    prompt prefixes. Uses an LRU (Least Recently Used) eviction policy when
    the cache reaches capacity.

    The cache works by:
    1. Hashing token sequences to identify common prefixes
    2. Storing KV block indices along with the token sequence
    3. Reference counting to prevent eviction of active entries
    4. LRU eviction when freeing up space for new entries

    Args:
        allocator: BlockAllocator for managing physical block allocation.
        block_size: Number of tokens per block.
        max_cache_entries: Maximum number of cache entries to store.
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int = 16,
        max_cache_entries: int = 1000,
    ) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.max_cache_entries = max_cache_entries
        self._cache: OrderedDict[int, CachedPrefix] = OrderedDict()
        self._stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "entries_added": 0,
        }

    def get(self, tokens: list[int]) -> CachedPrefix | None:
        """Look up a cached prefix matching the given tokens.

        Args:
            tokens: The token sequence to look up.

        Returns:
            CachedPrefix if found, None otherwise.
        """
        token_hash = self._hash_tokens(tokens)
        cached = self._cache.get(token_hash)
        if cached:
            # Move to end (most recently used)
            self._cache.move_to_end(token_hash)
            cached.last_used = cached.last_used  # Updated externally
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        return cached

    def add(
        self,
        tokens: list[int],
        block_indices: list[int],
        seq_id: int,
    ) -> bool:
        """Add a new prefix to the cache.

        Args:
            tokens: The token sequence being cached.
            block_indices: Physical block indices containing cached KV.
            seq_id: The source sequence ID that produced this cache.

        Returns:
            True if successfully added, False if cache is full and cannot evict.
        """
        from time import time

        token_hash = self._hash_tokens(tokens)

        # Check if already exists
        if token_hash in self._cache:
            existing = self._cache[token_hash]
            existing.ref_count += 1
            self._cache.move_to_end(token_hash)
            return True

        # Check capacity, evict if necessary
        if len(self._cache) >= self.max_cache_entries:
            if not self._evict_one():
                return False  # Cannot add, no evictable entries

        # Create new cache entry

        cached = CachedPrefix(
            token_hash=token_hash,
            tokens=tokens.copy(),
            block_indices=block_indices.copy(),
            ref_count=1,
            last_used=time(),
            seq_id=seq_id,
        )
        self._cache[token_hash] = cached
        self._stats["entries_added"] += 1
        return True

    def acquire(self, cached: CachedPrefix) -> None:
        """Acquire a reference to a cached prefix.

        Increments the reference count to prevent eviction.

        Args:
            cached: The cached prefix to acquire.
        """
        cached.ref_count += 1

    def release(self, cached: CachedPrefix) -> None:
        """Release a reference to a cached prefix.

        Decrements the reference count. If ref_count reaches zero and
        the entry is old enough, it becomes eligible for eviction.

        Args:
            cached: The cached prefix to release.
        """
        cached.ref_count = max(0, cached.ref_count - 1)

    def remove(self, token_hash: int) -> CachedPrefix | None:
        """Remove a cache entry, freeing its blocks.

        Args:
            token_hash: Hash of the token sequence to remove.

        Returns:
            The removed CachedPrefix, or None if not found.
        """
        cached = self._cache.pop(token_hash, None)
        if cached:
            # Free blocks only if not referenced by page table
            if cached.ref_count == 0:
                for block_idx in cached.block_indices:
                    self.allocator.free(block_idx)
        return cached

    def _evict_one(self) -> bool:
        """Evict one unreferenced entry using LRU policy.

        Returns:
            True if an entry was evicted, False if no evictable entries exist.
        """
        for token_hash, cached in list(self._cache.items()):
            if cached.ref_count == 0:
                self.remove(token_hash)
                self._stats["evictions"] += 1
                return True
        return False

    def _hash_tokens(self, tokens: list[int]) -> int:
        """Compute a stable hash for a token sequence.

        Uses SHA-256 for minimal collision probability.
        """
        token_bytes = ",".join(str(t) for t in tokens).encode()
        return int(sha256(token_bytes).hexdigest(), 16)

    def clear(self) -> None:
        """Clear all cache entries, freeing associated blocks."""
        for token_hash in list(self._cache.keys()):
            cached = self._cache.pop(token_hash)
            if cached.ref_count == 0:
                for block_idx in cached.block_indices:
                    self.allocator.free(block_idx)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "entries_added": 0,
        }

    @property
    def size(self) -> int:
        """Current number of cache entries."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self._stats.copy()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total


@dataclass
class CacheInfo:
    """Information about prefix cache utilization for a request.

    Attributes:
        cached_blocks: Number of blocks reused from cache.
        cached_tokens: Number of tokens reused from cache.
        uncached_tokens: Number of tokens that need prefill.
        cache_hit: Whether any cache hit occurred.
    """

    cached_blocks: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    uncached_tokens: int = 0
    cache_hit: bool = False

    @property
    def total_tokens(self) -> int:
        return self.cached_tokens + self.uncached_tokens


class PrefillStrategy(Enum):
    """Strategy for prioritizing waiting sequences during prefill."""

    FCFS = "fcfs"  # First-come-first-served (default)
    SHORTEST_FIRST = "shortest_first"  # Prioritize shorter prompts (better for latency)
    LONGEST_FIRST = "longest_first"  # Prioritize longer prompts (better for throughput)


@dataclass
class SchedulerConfig:
    """Configuration for the FCFS batch scheduler."""

    max_num_seqs: int = 64  # Max concurrent sequences
    max_num_batched_tokens: int = 2048  # Max tokens per iteration
    max_prefill_tokens: int = 1024  # Limit prefill to control latency
    block_size: int = 16
    prefill_strategy: PrefillStrategy = PrefillStrategy.FCFS  # Pre-fill prioritization
    enable_prefix_cache: bool = False  # Enable prefix caching across requests
    max_cache_entries: int = 1000  # Maximum number of cached prefixes


class FCFSScheduler:
    """First-come-first-served scheduler with preemption.

    Scheduling policy:
    - Running sequences get priority (preserve KV cache investment)
    - Waiting sequences are admitted FCFS if budget and memory allow
    - When memory is exhausted, the sequence with the most generated tokens
      is preempted (it has consumed the most blocks but may be furthest
      from completion)

    Preempted sequences are placed in the swapped queue and will be
    re-prefilled when memory becomes available (swap-in requires full
    re-prefill since KV blocks are freed on preemption).
    """

    def __init__(
        self,
        config: SchedulerConfig,
        allocator: BlockAllocator,
    ) -> None:
        self.config = config
        self.allocator = allocator
        self.page_table = PageTable(allocator, config.block_size)

        self.waiting: deque[GenerationRequest] = deque()
        self.running: list[GenerationRequest] = []
        self.swapped: deque[GenerationRequest] = deque()

        # Prefix cache support
        self.prefix_cache: PrefixCache | None = None
        if config.enable_prefix_cache:
            self.prefix_cache = PrefixCache(
                allocator=allocator,
                block_size=config.block_size,
                max_cache_entries=config.max_cache_entries,
            )
        self._cache_info_map: dict[int, CacheInfo] = {}  # seq_id -> CacheInfo

    def add_request(self, request: GenerationRequest) -> None:
        """Queue new request for scheduling."""
        request.status = RequestStatus.PENDING
        self.waiting.append(request)

    def abort_request(self, request_id: str) -> bool:
        """Cancel a request by ID, freeing any allocated resources.

        Returns True if the request was found and aborted.
        """
        # Check waiting queue (no blocks allocated yet for these)
        for req in list(self.waiting):
            if req.request_id == request_id:
                self.waiting.remove(req)
                return True

        # Check running queue (has blocks allocated)
        for req in list(self.running):
            if req.request_id == request_id:
                seq_id = id(req)
                self.page_table.remove_sequence(seq_id)

                # Release prefix cache reference if applicable
                if self.prefix_cache is not None and seq_id in self._cache_info_map:
                    cache_info = self._cache_info_map.pop(seq_id)
                    if cache_info.cache_hit:
                        cached = self.prefix_cache.get(req.prompt_tokens)
                        if cached:
                            self.prefix_cache.release(cached)

                self.running.remove(req)
                return True

        # Check swapped queue (blocks already freed on preemption)
        for req in list(self.swapped):
            if req.request_id == request_id:
                self.swapped.remove(req)
                return True

        return False

    def schedule(self) -> SchedulerOutput:
        """Determine what to run in the next iteration.

        Returns a SchedulerOutput partitioning requests into prefill,
        decode, and preempted groups. The engine processes prefill and
        decode requests in a single fused forward pass.
        """
        prefill: list[GenerationRequest] = []
        decode: list[GenerationRequest] = []
        preempted: list[GenerationRequest] = []

        budget = self.config.max_num_batched_tokens

        # Phase 1: Schedule decode for running sequences.
        # Each running sequence consumes exactly 1 token of budget.
        running_to_keep: list[GenerationRequest] = []
        for req in self.running:
            if budget >= 1:
                decode.append(req)
                budget -= 1
                running_to_keep.append(req)
            else:
                # Overloaded: preempt excess running sequences
                victim = self._preempt(req)
                preempted.append(victim)

        self.running = running_to_keep

        # Phase 2: Schedule prefill for waiting sequences.
        # Apply prefill strategy to prioritize which requests to admit first.
        # Also check swapped queue first (they were interrupted, give priority).
        prefill_budget = min(budget, self.config.max_prefill_tokens)

        # Try to admit swapped sequences first (re-prefill)
        while self.swapped and prefill_budget > 0:
            req = self.swapped[0]
            # Swapped sequences need full re-prefill (KV was freed)
            tokens_needed = len(req.prompt_tokens) + len(req.output_tokens)

            if tokens_needed > prefill_budget:
                break

            blocks_needed = self._blocks_for_tokens(tokens_needed)
            if self.allocator.num_free < blocks_needed:
                break

            self.swapped.popleft()
            if not self._allocate_for_prefill(req, tokens_needed):
                # Put it back if allocation failed somehow
                self.swapped.appendleft(req)
                break

            prefill.append(req)
            prefill_budget -= tokens_needed
            budget -= tokens_needed

        # Apply prefill prioritization strategy to waiting queue
        # Note: this creates a view without mutating the deque
        prioritized_waiting = self._sort_waiting_by_strategy()

        # Admit new waiting sequences in prioritized order
        admitted_ids: set[int] = set()
        for req in prioritized_waiting:
            if id(req) in admitted_ids:
                continue  # Skip if already admitted in this iteration
            if prefill_budget <= 0:
                break

            tokens_needed = len(req.prompt_tokens)

            if tokens_needed > prefill_budget:
                # Can't fit this request; try chunked prefill
                tokens_needed = min(tokens_needed, prefill_budget)
                if tokens_needed <= 0:
                    break

            blocks_needed = self._blocks_for_tokens(tokens_needed)
            if self.allocator.num_free >= blocks_needed:
                self.waiting.popleft()
                if not self._allocate_for_prefill(req, tokens_needed):
                    # Memory allocation failed, try preemption
                    if self.running:
                        victim = self._select_preempt_victim()
                        preempted.append(victim)
                        # Retry this request
                        self.waiting.appendleft(req)
                        continue
                    else:
                        # No one to preempt, put it back
                        self.waiting.appendleft(req)
                        break

                prefill.append(req)
                prefill_budget -= tokens_needed
                budget -= tokens_needed

                # Enforce max concurrent sequences
                total_seqs = len(self.running) + len(prefill)
                if total_seqs >= self.config.max_num_seqs:
                    break
            else:
                # Not enough free blocks; try preemption
                if self.running:
                    victim = self._select_preempt_victim()
                    preempted.append(victim)
                    continue
                break

        # Move prefilled requests to running
        for req in prefill:
            req.status = RequestStatus.RUNNING
            self.running.append(req)

        return SchedulerOutput(
            prefill_requests=prefill,
            decode_requests=decode,
            preempted_requests=preempted,
        )

    def free_finished(self) -> list[GenerationRequest]:
        """Remove finished sequences and free their blocks.

        Should be called after each generation step to reclaim memory.
        Returns the list of finished requests (for output streaming).
        """
        finished: list[GenerationRequest] = []
        still_running: list[GenerationRequest] = []

        for req in self.running:
            if req.is_finished:
                seq_id = id(req)
                self.page_table.remove_sequence(seq_id)

                # Release prefix cache reference if applicable
                if self.prefix_cache is not None and seq_id in self._cache_info_map:
                    cache_info = self._cache_info_map.pop(seq_id)
                    if cache_info.cache_hit:
                        cached = self.prefix_cache.get(req.prompt_tokens)
                        if cached:
                            self.prefix_cache.release(cached)

                finished.append(req)
            else:
                still_running.append(req)

        self.running = still_running
        return finished

    def step_decode(self) -> None:
        """Advance page table by one token for all running sequences.

        Called after the model produces one new token per decode request.
        Allocates new blocks if any sequence fills its current tail block.
        """
        for req in self.running:
            if self.page_table.has_sequence(id(req)):
                success = self.page_table.append_token(id(req))
                if not success:
                    # Out of memory mid-decode; preempt this sequence
                    self._preempt(req)

    def _allocate_for_prefill(self, req: GenerationRequest, num_tokens: int) -> bool:
        """Allocate page table entries for a prefill.

        Registers the sequence and appends token slots.
        If prefix caching is enabled, attempts to reuse cached blocks.
        Returns False if allocation fails.
        """
        seq_id = id(req)
        cache_info = CacheInfo()

        # Try prefix cache lookup if enabled
        if self.prefix_cache is not None:
            cached_prefix = self.prefix_cache.get(req.prompt_tokens)
            if cached_prefix is not None:
                # Cache hit! Reuse cached blocks
                from time import time

                cached_prefix.last_used = time()
                cache_info.cached_blocks = cached_prefix.block_indices.copy()
                cache_info.cached_tokens = len(cached_prefix.tokens)
                cache_info.cache_hit = True

                # Register sequence and link cached blocks
                if not self.page_table.has_sequence(seq_id):
                    # Create sequence state directly with cached blocks
                    from ..paged.page_table import SequenceState

                    self.page_table.sequences[seq_id] = SequenceState(
                        seq_id=seq_id,
                        block_indices=cached_prefix.block_indices.copy(),
                        logical_len=cache_info.cached_tokens,
                    )

                    # Increment ref count for shared blocks
                    for block_idx in cached_prefix.block_indices:
                        self.allocator.blocks[block_idx].ref_count += 1

                # Acquire reference to prevent eviction
                self.prefix_cache.acquire(cached_prefix)

                # Only allocate for remaining tokens beyond cache
                remaining_tokens = num_tokens - cache_info.cached_tokens
                if remaining_tokens > 0:
                    cache_info.uncached_tokens = remaining_tokens
                    success = self.page_table.append_tokens(seq_id, remaining_tokens)
                    if not success:
                        # Allocation failed, cleanup and return
                        self.page_table.remove_sequence(seq_id)
                        self.prefix_cache.release(cached_prefix)
                        return False
            else:
                # Cache miss - normal allocation path
                cache_info.uncached_tokens = num_tokens
        else:
            # Caching disabled - normal allocation path
            cache_info.uncached_tokens = num_tokens

        # Normal allocation if no cache or cache miss
        if not cache_info.cache_hit:
            if not self.page_table.has_sequence(seq_id):
                success = self.page_table.add_sequence(seq_id)
                if not success:
                    return False

            # Append token slots (triggers block allocation as needed)
            success = self.page_table.append_tokens(seq_id, num_tokens)
            if not success:
                return False

            # Add to cache for future requests (if enabled)
            if self.prefix_cache is not None and len(req.prompt_tokens) > 0:
                blocks = self.page_table.get_block_table(seq_id)
                self.prefix_cache.add(
                    tokens=req.prompt_tokens,
                    block_indices=blocks,
                    seq_id=seq_id,
                )

        # Store cache info for this request
        self._cache_info_map[seq_id] = cache_info
        return True

    def _select_preempt_victim(self) -> GenerationRequest:
        """Select sequence to preempt.

        FCFS policy: preempt the sequence with the most generated tokens.
        Rationale: it has consumed the most memory and is potentially
        furthest from completion if generating long outputs.
        """
        victim = max(self.running, key=lambda r: len(r.output_tokens))
        return self._preempt(victim)

    def _preempt(self, req: GenerationRequest) -> GenerationRequest:
        """Preempt a running sequence: free blocks, move to swapped."""
        if req in self.running:
            self.running.remove(req)

        seq_id = id(req)
        self.page_table.remove_sequence(seq_id)

        # Release prefix cache reference if applicable
        if self.prefix_cache is not None and seq_id in self._cache_info_map:
            cache_info = self._cache_info_map.pop(seq_id)
            if cache_info.cache_hit:
                cached = self.prefix_cache.get(req.prompt_tokens)
                if cached:
                    self.prefix_cache.release(cached)

        req.status = RequestStatus.PREEMPTED
        self.swapped.append(req)
        return req

    def _blocks_for_tokens(self, num_tokens: int) -> int:
        """Calculate blocks needed for a token count."""
        return (num_tokens + self.config.block_size - 1) // self.config.block_size

    @property
    def num_waiting(self) -> int:
        """Number of requests in the waiting queue."""
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        """Number of actively running sequences."""
        return len(self.running)

    @property
    def num_swapped(self) -> int:
        """Number of preempted sequences."""
        return len(self.swapped)

    def _sort_waiting_by_strategy(self) -> list[GenerationRequest]:
        """Apply prefill prioritization strategy to waiting queue.

        Returns a prioritized list without mutating the original deque.
        """
        waiting_list = list(self.waiting)

        match self.config.prefill_strategy:
            case PrefillStrategy.FCFS:
                # Preserve original arrival order
                return waiting_list
            case PrefillStrategy.SHORTEST_FIRST:
                # Prioritize shorter prompts (better for latency)
                return sorted(waiting_list, key=lambda r: len(r.prompt_tokens))
            case PrefillStrategy.LONGEST_FIRST:
                # Prioritize longer prompts (better for throughput)
                return sorted(waiting_list, key=lambda r: -len(r.prompt_tokens))
            case _:
                return waiting_list

    @property
    def has_pending_work(self) -> bool:
        """Whether there's any work to do (requests in any queue)."""
        return bool(self.waiting or self.running or self.swapped)

    def get_cache_info(self, request: GenerationRequest) -> CacheInfo | None:
        """Get prefix cache information for a request.

        Args:
            request: The generation request to query.

        Returns:
            CacheInfo if caching is enabled and request was processed, None otherwise.
        """
        return self._cache_info_map.get(id(request))

    @property
    def cache_stats(self) -> dict[str, int | float] | None:
        """Get prefix cache statistics.

        Returns:
            Dictionary with cache hits, misses, hit rate, etc., or None if caching disabled.
        """
        if self.prefix_cache is None:
            return None
        return {
            **self.prefix_cache.stats,
            "hit_rate": self.prefix_cache.hit_rate,
            "cache_size": self.prefix_cache.size,
        }


class InsertionPolicy(Enum):
    """Policy for inserting batch requests into the scheduler."""

    ENQUEUE = "enqueue"  # Add to back of waiting queue (FCFS order)
    MERGE = "merge"  # Insert at front of waiting queue (higher priority)
    DROP_IF_FULL = "drop_if_full"  # Drop entire batch if queue full


class QueueFullError(Exception):
    """Raised when attempting to add to a full waiting queue."""

    pass


class BatchScheduler(FCFSScheduler):
    """Batch scheduler with dynamic request insertion and flexible admission.

    Extends FCFSScheduler to support:
    - Dynamic insertion of single requests at any time via add_request()
    - Batch insertion of multiple requests via insert_batch()
    - Priority-based admission control with configurable insertion policies

    The scheduler maintains the same three-queue structure (WAITING, RUNNING,
    SWAPPED) but provides enhanced methods for efficient batch processing.

    Insertion policies control how batch requests are added:
    - ENQUEUE: Add all to waiting queue (default, preserves order)
    - MERGE: Insert at front of waiting queue (higher priority)
    - DROP_IF_FULL: Silently drop if waiting queue is too large

    Usage:
        sched = BatchScheduler(SchedulerConfig(block_size=16), alloc)

        # Dynamic single insertion
        req = GenerationRequest(request_id="req-1", prompt_tokens=[1, 2, 3])
        sched.add_request(req)

        # Batch insertion
        batch = [
            GenerationRequest(request_id="req-2", prompt_tokens=[4, 5, 6]),
            GenerationRequest(request_id="req-3", prompt_tokens=[7, 8, 9]),
        ]
        sched.insert_batch(batch, policy=InsertionPolicy.ENQUEUE)

        # Normal scheduling cycle
        output = sched.schedule()
    """

    def __init__(
        self,
        config: SchedulerConfig,
        allocator: BlockAllocator,
        max_queue_size: int = 1000,
    ) -> None:
        """Initialize the batch scheduler.

        Args:
            config: Scheduler configuration parameters.
            allocator: Block allocator for KV cache memory management.
            max_queue_size: Maximum size of waiting queue before dropping
                requests under DROP_IF_FULL policy.
        """
        super().__init__(config, allocator)
        self.max_queue_size = max_queue_size
        self._insertion_stats: dict[str, int] = {
            "total_inserted": 0,
            "total_dropped": 0,
            "batch_insertions": 0,
        }

    def add_request(self, request: GenerationRequest) -> None:
        """Queue new request for scheduling with stats tracking.

        Overrides FCFSScheduler.add_request to track insertion statistics.

        Args:
            request: Generation request to add.

        Raises:
            QueueFullError: If waiting queue is at max capacity.
        """
        if not self._check_queue_capacity():
            raise QueueFullError(
                f"Cannot add request {request.request_id}: "
                f"waiting queue full ({len(self.waiting)} >= {self.max_queue_size})"
            )
        super().add_request(request)
        self._insertion_stats["total_inserted"] += 1

    def insert_batch(
        self,
        requests: list[GenerationRequest],
        policy: InsertionPolicy = InsertionPolicy.ENQUEUE,
    ) -> int:
        """Insert a batch of requests into the scheduler.

        Args:
            requests: List of generation requests to insert.
            policy: Insertion policy controlling how requests are added.

        Returns:
            Number of requests successfully inserted. Some may be dropped
            under DROP_IF_FULL policy if the queue is at capacity.

        Raises:
            ValueError: If an invalid insertion policy is provided.
        """
        self._insertion_stats["batch_insertions"] += 1
        inserted = 0

        match policy:
            case InsertionPolicy.ENQUEUE:
                for req in requests:
                    if self._check_queue_capacity():
                        self.add_request(req)
                        inserted += 1
                    else:
                        self._insertion_stats["total_dropped"] += 1

            case InsertionPolicy.MERGE:
                # Insert at front of waiting queue with reversed order
                # to preserve batch order within merged section
                for req in reversed(requests):
                    if self._check_queue_capacity():
                        req.status = RequestStatus.PENDING
                        self.waiting.appendleft(req)
                        inserted += 1
                    else:
                        self._insertion_stats["total_dropped"] += 1

            case InsertionPolicy.DROP_IF_FULL:
                # Only insert if queue has room, otherwise drop all
                queue_available = self.max_queue_size - len(self.waiting)
                if queue_available >= len(requests):
                    for req in requests:
                        self.add_request(req)
                        inserted += 1
                else:
                    self._insertion_stats["total_dropped"] += len(requests)

            case _:
                raise ValueError(f"Invalid insertion policy: {policy}")

        self._insertion_stats["total_inserted"] += inserted
        return inserted

    def insert_batch_front(self, requests: list[GenerationRequest]) -> int:
        """Insert batch at front of waiting queue (high priority).

        This is a convenience method for inserting batch requests with
        priority over existing waiting requests.

        Args:
            requests: List of generation requests to insert.

        Returns:
            Number of requests successfully inserted.
        """
        return self.insert_batch(requests, InsertionPolicy.MERGE)

    def add_request_front(self, request: GenerationRequest) -> None:
        """Add a single request at front of waiting queue (high priority).

        Args:
            request: Generation request to add.

        Raises:
            QueueFullError: If waiting queue is at max capacity.
        """
        if not self._check_queue_capacity():
            raise QueueFullError(
                f"Cannot add request {request.request_id}: "
                f"waiting queue full ({len(self.waiting)} >= {self.max_queue_size})"
            )
        request.status = RequestStatus.PENDING
        self.waiting.appendleft(request)
        self._insertion_stats["total_inserted"] += 1

    def _check_queue_capacity(self) -> bool:
        """Check if there's room in the waiting queue.

        Returns:
            True if waiting queue is below max_queue_size, False otherwise.
        """
        return len(self.waiting) < self.max_queue_size

    @property
    def insertion_stats(self) -> dict[str, int]:
        """Get statistics about request insertions.

        Returns:
            Dictionary with keys:
            - total_inserted: Total number of requests inserted
            - total_dropped: Total number of requests dropped
            - batch_insertions: Number of batch insertion operations
        """
        return self._insertion_stats.copy()

    def reset_insertion_stats(self) -> None:
        """Reset insertion statistics counters."""
        self._insertion_stats = {
            "total_inserted": 0,
            "total_dropped": 0,
            "batch_insertions": 0,
        }

    @property
    def queue_utilization(self) -> float:
        """Get current waiting queue utilization ratio.

        Returns:
            Float in range [0.0, 1.0] representing how full the waiting
            queue is. Returns 0.0 if max_queue_size is 0 (unlimited).
        """
        if self.max_queue_size == 0:
            return 0.0
        return len(self.waiting) / self.max_queue_size

    def clear_waiting(self) -> int:
        """Clear all waiting requests (useful for recovery or reset).

        Returns:
            Number of requests that were cleared.
        """
        count = len(self.waiting)
        self.waiting.clear()
        return count
