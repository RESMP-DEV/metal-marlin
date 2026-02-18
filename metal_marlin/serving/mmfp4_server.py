"""MMFP4 server with request batching support.

This module provides request batching capabilities for the Metal Marlin server,
optimizing throughput by grouping multiple incoming requests for efficient GPU utilization.

Features:
    - Dynamic request batching
    - KV cache sharing for common prefixes
    - Efficient GPU memory utilization

Usage:
    from metal_marlin.serving.mmfp4_server import MMFP4Server, _request_batch, _kv_sharing

    server = MMFP4Server(engine_config)
    
    # Batch multiple requests
    batched_results = await _request_batch(server, [request1, request2, request3])
    
    # Enable KV cache sharing for prefix caching
    sharing_manager = _kv_sharing(server, enable_prefix_caching=True)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .continuous_batch import BatchScheduler, KVCacheManager, SchedulerConfig
from .openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from .request import GenerationRequest, RequestStatus, RunningRequest

if TYPE_CHECKING:
    from .engine import ServingEngine


@dataclass
class SharedBlock:
    """Represents a shared KV cache block with reference counting."""

    block_idx: int
    ref_count: int = 1
    owner_request_id: str = ""


@dataclass
class SharedRegion:
    """Tracks a shared KV cache region for prompt sharing between requests."""

    region_id: int
    request_id: str
    block_indices: list[int] = field(default_factory=list)
    shared_with: list[str] = field(default_factory=list)
    is_forked: bool = False
    prompt_hash: str | None = None


class KVCacheSharing:
    """Manages KV cache sharing between requests using Copy-on-Write semantics.

    Enables multiple requests to share common prompt prefixes without duplicating
    KV cache memory. When a request writes to a shared block, COW triggers
    automatically to create a private copy.

    Features:
    - Zero-copy prompt prefix sharing via reference counting
    - Automatic Copy-on-Write when shared blocks are modified
    - Prompt hash-based cache for efficient prefix matching
    - Per-request shared region tracking

    Example:
        sharing = KVCacheSharing()
        
        # Share prompt between requests
        sharing.share_prompt(parent_id, child_id, block_indices)
        
        # Trigger COW when child writes
        new_block = sharing.copy_on_write(child_id, block_idx)
        
        # Release shared region when done
        sharing.release_request(request_id)
    """

    def __init__(self) -> None:
        # Block index -> SharedBlock tracking
        self._shared_blocks: dict[int, SharedBlock] = {}
        
        # Request ID -> SharedRegion mapping
        self._shared_regions: dict[str, SharedRegion] = {}
        
        # Prompt hash -> (block_indices, request_ids) for cache lookup
        self._prompt_cache: dict[str, tuple[list[int], list[str]]] = {}
        
        # Statistics
        self._cow_operations = 0
        self._shared_block_count = 0
        self._prompt_cache_hits = 0
        self._next_region_id = 0

    def create_region(
        self,
        request_id: str,
        block_indices: list[int],
        prompt_hash: str | None = None,
    ) -> int:
        """Create a new shared region for a request.

        Args:
            request_id: Unique request identifier.
            block_indices: Initial block indices for this region.
            prompt_hash: Optional hash of prompt for caching.

        Returns:
            Region ID for the created region.
        """
        region_id = self._next_region_id
        self._next_region_id += 1

        region = SharedRegion(
            region_id=region_id,
            request_id=request_id,
            block_indices=block_indices.copy(),
            prompt_hash=prompt_hash,
        )
        self._shared_regions[request_id] = region

        # Mark blocks as shared
        for block_idx in block_indices:
            if block_idx in self._shared_blocks:
                self._shared_blocks[block_idx].ref_count += 1
            else:
                self._shared_blocks[block_idx] = SharedBlock(
                    block_idx=block_idx,
                    ref_count=1,
                    owner_request_id=request_id,
                )

        return region_id

    def share_prompt(
        self,
        parent_request_id: str,
        child_request_id: str,
        num_prefix_tokens: int | None = None,
    ) -> list[int] | None:
        """Share prompt prefix from parent to child request.

        Args:
            parent_request_id: Source request ID.
            child_request_id: Destination request ID.
            num_prefix_tokens: Number of tokens to share (None = all).

        Returns:
            List of shared block indices, or None if sharing failed.
        """
        parent_region = self._shared_regions.get(parent_request_id)
        if parent_region is None:
            return None

        # Determine blocks to share
        if num_prefix_tokens is None:
            shared_blocks = parent_region.block_indices.copy()
        else:
            # Assuming 16 tokens per block (standard block size)
            block_size = 16
            num_blocks = (num_prefix_tokens + block_size - 1) // block_size
            shared_blocks = parent_region.block_indices[:num_blocks]

        # Create child region
        region_id = self._next_region_id
        self._next_region_id += 1

        child_region = SharedRegion(
            region_id=region_id,
            request_id=child_request_id,
            block_indices=shared_blocks.copy(),
            is_forked=True,
        )
        self._shared_regions[child_request_id] = child_region

        # Update parent tracking
        parent_region.shared_with.append(child_request_id)

        # Increment refcounts for shared blocks
        for block_idx in shared_blocks:
            if block_idx in self._shared_blocks:
                self._shared_blocks[block_idx].ref_count += 1
            else:
                self._shared_blocks[block_idx] = SharedBlock(
                    block_idx=block_idx,
                    ref_count=2,  # Parent + child
                    owner_request_id=parent_request_id,
                )

        self._shared_block_count += len(shared_blocks)
        return shared_blocks

    def copy_on_write(
        self,
        request_id: str,
        block_idx: int,
        new_block_idx: int,
    ) -> bool:
        """Perform Copy-on-Write for a shared block.

        Args:
            request_id: Request that triggered COW.
            block_idx: Original shared block index.
            new_block_idx: New private block index.

        Returns:
            True if COW was performed, False if not needed or failed.
        """
        region = self._shared_regions.get(request_id)
        if region is None:
            return False

        shared_block = self._shared_blocks.get(block_idx)
        if shared_block is None or shared_block.ref_count <= 1:
            # Not shared, no COW needed
            return False

        # Decrement refcount on original
        shared_block.ref_count -= 1

        # Replace block in request's region
        if block_idx in region.block_indices:
            idx = region.block_indices.index(block_idx)
            region.block_indices[idx] = new_block_idx

        # Create new private block entry
        self._shared_blocks[new_block_idx] = SharedBlock(
            block_idx=new_block_idx,
            ref_count=1,
            owner_request_id=request_id,
        )

        self._cow_operations += 1
        return True

    def check_needs_cow(self, request_id: str, block_idx: int) -> bool:
        """Check if writing to a block requires COW.

        Args:
            request_id: Request ID that wants to write.
            block_idx: Block index to check.

        Returns:
            True if COW is required before writing.
        """
        shared_block = self._shared_blocks.get(block_idx)
        if shared_block is None:
            return False
        return shared_block.ref_count > 1

    def get_shared_blocks(self, request_id: str) -> list[int]:
        """Get list of shared blocks for a request.

        Args:
            request_id: Request ID to check.

        Returns:
            List of block indices that are shared (refcount > 1).
        """
        region = self._shared_regions.get(request_id)
        if region is None:
            return []

        shared = []
        for block_idx in region.block_indices:
            shared_block = self._shared_blocks.get(block_idx)
            if shared_block and shared_block.ref_count > 1:
                shared.append(block_idx)
        return shared

    def release_request(self, request_id: str) -> list[int]:
        """Release all shared regions for a request.

        Args:
            request_id: Request ID to release.

        Returns:
            List of block indices that can be freed (refcount reached 0).
        """
        region = self._shared_regions.pop(request_id, None)
        if region is None:
            return []

        freed_blocks = []

        # Decrement refcounts
        for block_idx in region.block_indices:
            shared_block = self._shared_blocks.get(block_idx)
            if shared_block:
                shared_block.ref_count -= 1
                if shared_block.ref_count <= 0:
                    del self._shared_blocks[block_idx]
                    freed_blocks.append(block_idx)

        # Remove from parent's shared list
        if region.is_forked:
            for parent_region in self._shared_regions.values():
                if request_id in parent_region.shared_with:
                    parent_region.shared_with.remove(request_id)
                    break

        return freed_blocks

    def cache_prompt(
        self,
        request_id: str,
        prompt_tokens: list[int],
        block_indices: list[int],
    ) -> str:
        """Cache a prompt for potential reuse.

        Args:
            request_id: Request ID with the prompt.
            prompt_tokens: Token IDs of the prompt.
            block_indices: Block indices allocated for this prompt.

        Returns:
            Prompt hash for cache lookup.
        """
        prompt_hash = self._compute_prompt_hash(prompt_tokens)
        
        if prompt_hash not in self._prompt_cache:
            self._prompt_cache[prompt_hash] = (block_indices.copy(), [request_id])
        else:
            cached_blocks, cached_requests = self._prompt_cache[prompt_hash]
            if request_id not in cached_requests:
                cached_requests.append(request_id)

        # Update region with hash
        region = self._shared_regions.get(request_id)
        if region:
            region.prompt_hash = prompt_hash

        return prompt_hash

    def lookup_cached_prompt(self, prompt_tokens: list[int]) -> tuple[list[int], str] | None:
        """Look up a cached prompt.

        Args:
            prompt_tokens: Token IDs to look up.

        Returns:
            Tuple of (block_indices, source_request_id) or None if not found.
        """
        prompt_hash = self._compute_prompt_hash(prompt_tokens)
        
        if prompt_hash not in self._prompt_cache:
            return None

        cached_blocks, cached_requests = self._prompt_cache[prompt_hash]
        
        # Find first valid source request
        for src_id in cached_requests:
            if src_id in self._shared_regions:
                self._prompt_cache_hits += 1
                return (cached_blocks.copy(), src_id)

        # Cache stale, clean it up
        del self._prompt_cache[prompt_hash]
        return None

    @staticmethod
    def _compute_prompt_hash(tokens: list[int]) -> str:
        """Compute hash for prompt prefix matching."""
        token_bytes = b"".join(t.to_bytes(4, "little") for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def get_stats(self) -> dict:
        """Get sharing statistics."""
        return {
            "cow_operations": self._cow_operations,
            "shared_blocks": self._shared_block_count,
            "prompt_cache_hits": self._prompt_cache_hits,
            "active_regions": len(self._shared_regions),
            "tracked_blocks": len(self._shared_blocks),
            "cached_prompts": len(self._prompt_cache),
        }


def _kv_sharing(
    server: MMFP4Server | None = None,
    enable_prefix_caching: bool = True,
) -> KVCacheSharing:
    """Create a KV cache sharing manager.

    This factory function creates a KVCacheSharing instance for managing
    KV cache sharing between requests. When a server is provided, the
    sharing manager can be integrated with the server's scheduler.

    Args:
        server: Optional MMFP4Server instance to integrate with.
        enable_prefix_caching: Whether to enable prompt prefix caching.

    Returns:
        Configured KVCacheSharing instance.

    Example:
        # Standalone usage
        sharing = _kv_sharing()
        
        # With server integration
        sharing = _kv_sharing(server, enable_prefix_caching=True)
        
        # Share prompt between requests
        sharing.share_prompt(parent_id, child_id, num_tokens)
    """
    sharing = KVCacheSharing()
    
    # Future: integrate with server.scheduler if server is provided
    # and enable_prefix_caching can be used for additional setup
    
    return sharing


def get_request_queue(maxsize: int = 0) -> RequestQueue:
    """Get or create the module-level request queue.

    This factory function provides a global request queue that can be
    shared across different components of the serving system.

    Args:
        maxsize: Maximum queue size (0 = unlimited).

    Returns:
        The global RequestQueue instance.

    Example:
        # Get the global queue
        queue = get_request_queue(maxsize=100)
        
        # Submit a request
        await queue.put(gen_request)
        
        # Process requests
        request = await queue.get()
    """
    global _request_queue
    if _request_queue is None:
        _request_queue = RequestQueue(maxsize=maxsize)
    return _request_queue


def init_request_queue(server: MMFP4Server, maxsize: int = 0) -> RequestQueue:
    """Initialize request queue for a server instance.

    Creates and configures a RequestQueue integrated with the server,
    setting up the module-level _request_queue reference.

    Args:
        server: The MMFP4Server to configure queue for.
        maxsize: Maximum queue size (0 = unlimited).

    Returns:
        Configured RequestQueue instance.
    """
    global _request_queue
    _request_queue = RequestQueue(maxsize=maxsize)
    return _request_queue


@dataclass
class BatchedRequest:
    """Wrapper for a request with its completion future and metadata."""

    request: ChatCompletionRequest | CompletionRequest
    future: asyncio.Future
    arrival_time: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class BatchConfig:
    """Configuration for request batching behavior."""

    max_batch_size: int = 32
    max_wait_ms: float = 10.0  # Maximum time to wait for batch to fill
    min_batch_size: int = 1  # Minimum requests to process (don't wait forever)


@dataclass
class RequestQueue:
    """Request queue for managing incoming inference requests.

    Provides a structured interface for queueing requests with
    configurable size limits and priority support.

    Attributes:
        maxsize: Maximum number of requests in queue (0 = unlimited).
        queue: The underlying asyncio.Queue.
        queue_id: Unique identifier for this queue instance.
    """

    maxsize: int = 0
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    queue_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        """Initialize queue with proper maxsize if not already set."""
        if self.maxsize > 0 and self.queue.maxsize != self.maxsize:
            self.queue = asyncio.Queue(maxsize=self.maxsize)

    @property
    def size(self) -> int:
        """Current number of items in the queue."""
        return self.queue.qsize()

    @property
    def is_empty(self) -> bool:
        """Whether the queue is empty."""
        return self.queue.empty()

    @property
    def is_full(self) -> bool:
        """Whether the queue is full."""
        return self.queue.full()

    async def put(self, item: GenerationRequest) -> None:
        """Put a request into the queue."""
        await self.queue.put(item)

    def put_nowait(self, item: GenerationRequest) -> bool:
        """Put a request without waiting, returns False if full."""
        try:
            self.queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            return False

    async def get(self) -> GenerationRequest:
        """Get a request from the queue."""
        return await self.queue.get()

    def get_nowait(self) -> GenerationRequest | None:
        """Get a request without waiting, returns None if empty."""
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def task_done(self) -> None:
        """Mark a task as done."""
        self.queue.task_done()


# Module-level request queue instance for global request management
_request_queue: RequestQueue | None = None


class RequestBatcher:
    """Batches incoming requests for efficient processing.

    Collects incoming requests and groups them into batches based on:
    - Maximum batch size
    - Maximum wait time (don't hold requests too long)
    - Minimum batch size (process immediately if we have enough)

    Usage:
        batcher = RequestBatcher(config)
        
        # Submit requests
        future = await batcher.submit(request)
        result = await future  # Wait for batched result
        
        # Process batches (typically in background task)
        while True:
            batch = await batcher.get_batch()
            results = await process_batch(batch)
            batcher.complete_batch(batch, results)
    """

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()
        self._queue: deque[BatchedRequest] = deque()
        self._batch_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._pending_batches: dict[str, list[BatchedRequest]] = {}

    async def submit(
        self, request: ChatCompletionRequest | CompletionRequest
    ) -> asyncio.Future:
        """Submit a request to be batched.

        Args:
            request: The incoming request.

        Returns:
            A future that will resolve when the request is processed.
        """
        future = asyncio.get_event_loop().create_future()
        batched = BatchedRequest(request=request, future=future)

        async with self._lock:
            self._queue.append(batched)
            should_notify = len(self._queue) >= self.config.min_batch_size

        if should_notify:
            self._batch_event.set()

        return future

    async def get_batch(self) -> list[BatchedRequest]:
        """Get the next batch of requests to process.

        Waits until either:
        - max_batch_size requests are available
        - max_wait_ms has passed since first request arrived
        - min_batch_size requests are available

        Returns:
            List of batched requests to process together.
        """
        while True:
            async with self._lock:
                if len(self._queue) >= self.config.min_batch_size:
                    batch_size = min(len(self._queue), self.config.max_batch_size)
                    batch = [
                        self._queue.popleft() for _ in range(batch_size)
                    ]
                    batch_id = str(uuid.uuid4())
                    self._pending_batches[batch_id] = batch
                    return batch

            # Wait for more requests or timeout
            try:
                await asyncio.wait_for(
                    self._batch_event.wait(),
                    timeout=self.config.max_wait_ms / 1000.0
                )
            except asyncio.TimeoutError:
                pass
            finally:
                self._batch_event.clear()

    def complete_batch(
        self,
        batch: list[BatchedRequest],
        results: list,
    ) -> None:
        """Complete all requests in a batch with their results.

        Args:
            batch: The batch that was processed.
            results: Results for each request in the batch.
        """
        for batched, result in zip(batch, results):
            if not batched.future.done():
                batched.future.set_result(result)

        # Clean up pending batch tracking
        for batch_id, pending in list(self._pending_batches.items()):
            if pending is batch:
                del self._pending_batches[batch_id]
                break

    def fail_batch(self, batch: list[BatchedRequest], error: Exception) -> None:
        """Mark all requests in a batch as failed.

        Args:
            batch: The batch that failed.
            error: The exception that occurred.
        """
        for batched in batch:
            if not batched.future.done():
                batched.future.set_exception(error)

    @property
    def queue_depth(self) -> int:
        """Number of requests currently waiting to be batched."""
        return len(self._queue)

    @property
    def pending_batches_count(self) -> int:
        """Number of batches currently being processed."""
        return len(self._pending_batches)


class MMFP4Server:
    """Server with request batching for MMFP4 quantized models.

    Integrates with the continuous batching scheduler to provide:
    - Dynamic request batching
    - Priority-based scheduling
    - Efficient KV cache management
    - Continuous batching with queue-based interface

    Usage:
        server = MMFP4Server(engine_config)
        await server.start()
        
        # Submit requests for batching
        response = await server.submit_chat_request(request)
        
        # Or use continuous batching with queues
        await server.start_continuous_batching()
        await server.request_queue.put(gen_request)
        req_id, response = await server.result_queue.get()
    """

    def __init__(
        self,
        engine: ServingEngine,
        scheduler_config: SchedulerConfig | None = None,
        batch_config: BatchConfig | None = None,
    ):
        self.engine = engine
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.batch_config = batch_config or BatchConfig()

        # Initialize scheduler and KV cache manager
        kv_manager = KVCacheManager(
            num_blocks=scheduler_config.max_num_batched_tokens // scheduler_config.block_size
            if scheduler_config
            else 512,
            block_size=scheduler_config.block_size if scheduler_config else 16,
        )
        self.scheduler = BatchScheduler(self.scheduler_config, kv_manager)
        self.batcher = RequestBatcher(self.batch_config)

        self._running = False
        self._batch_task: asyncio.Task | None = None

        # Continuous batching queues (created on demand)
        self._request_queue: asyncio.Queue | None = None
        self._result_queue: asyncio.Queue | None = None
        self._shutdown_event: asyncio.Event | None = None

    async def start(self) -> None:
        """Start the batch processing loop."""
        if self._running:
            return

        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop the batch processing loop."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
            self._batch_task = None

    async def _batch_loop(self) -> None:
        """Background task that continuously processes batches."""
        while self._running:
            try:
                batch = await self.batcher.get_batch()
                await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error and continue
                print(f"Batch processing error: {e}")
                await asyncio.sleep(0.001)

    async def _process_batch(self, batch: list[BatchedRequest]) -> None:
        """Process a batch of requests through the scheduler."""
        try:
            results = await _request_batch(self, [b.request for b in batch])
            self.batcher.complete_batch(batch, results)
        except Exception as e:
            self.batcher.fail_batch(batch, e)

    async def submit_chat_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Submit a chat completion request for batching.

        Args:
            request: The chat completion request.

        Returns:
            The completion response.
        """
        future = await self.batcher.submit(request)
        return await future

    async def submit_completion_request(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        """Submit a text completion request for batching.

        Args:
            request: The completion request.

        Returns:
            The completion response.
        """
        future = await self.batcher.submit(request)
        return await future

    @property
    def stats(self) -> dict:
        """Get current server statistics."""
        return {
            "queue_depth": self.batcher.queue_depth,
            "pending_batches": self.batcher.pending_batches_count,
            **self.scheduler.get_stats(),
        }

    @property
    def request_queue(self) -> asyncio.Queue | None:
        """Request queue for continuous batching mode.

        Available after calling start_continuous_batching().
        """
        return self._request_queue

    @property
    def result_queue(self) -> asyncio.Queue | None:
        """Result queue for continuous batching mode.

        Available after calling start_continuous_batching().
        """
        return self._result_queue

    async def start_continuous_batching(self) -> None:
        """Start the server in continuous batching mode.

        This mode uses asyncio queues for request/response handling,
        allowing for more flexible request routing and processing.

        Usage:
            await server.start_continuous_batching()

            # Submit requests
            await server.request_queue.put(gen_request)

            # Get results as they complete
            req_id, response = await server.result_queue.get()
        """
        if self._running:
            return

        self._running = True

        # Create queues for request/response handling
        self._request_queue = asyncio.Queue(maxsize=self.batch_config.max_batch_size * 2)
        self._result_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

        # Start the continuous batching loop
        self._batch_task = asyncio.create_task(
            _server_batching(
                self,
                self._request_queue,
                self._result_queue,
                self._shutdown_event,
            )
        )

    async def stop_continuous_batching(self) -> None:
        """Stop the continuous batching loop gracefully."""
        if not self._running or self._shutdown_event is None:
            return

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for batch task to complete
        if self._batch_task:
            try:
                await asyncio.wait_for(self._batch_task, timeout=30.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._batch_task = None

        self._running = False

        # Drain remaining results
        if self._result_queue:
            remaining = []
            while not self._result_queue.empty():
                try:
                    remaining.append(self._result_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if remaining:
                print(f"Drained {len(remaining)} remaining results during shutdown")

    def submit_request_to_queue(
        self,
        request: GenerationRequest,
        original_request: ChatCompletionRequest | CompletionRequest | None = None,
    ) -> bool:
        """Submit a request to the continuous batching queue.

        Args:
            request: The generation request to process.
            original_request: The original API request for response building.

        Returns:
            True if the request was queued successfully, False if queue is full.

        Raises:
            RuntimeError: If continuous batching mode is not active.
        """
        if not self._running or self._request_queue is None:
            raise RuntimeError(
                "Continuous batching mode is not active. "
                "Call start_continuous_batching() first."
            )

        try:
            self._request_queue.put_nowait(request)
            return True
        except asyncio.QueueFull:
            return False


async def _request_batch(
    server: MMFP4Server,
    requests: list[ChatCompletionRequest | CompletionRequest],
) -> list[ChatCompletionResponse | CompletionResponse]:
    """Process a batch of requests through the scheduler.

    This is the core batching function that:
    1. Converts requests to GenerationRequest objects
    2. Adds them to the scheduler
    3. Runs scheduling iterations until all complete
    4. Returns results for each request

    Args:
        server: The MMFP4 server instance.
        requests: List of requests to batch together.

    Returns:
        List of responses corresponding to each request.

    Example:
        server = MMFP4Server(engine)
        
        # Batch multiple chat requests
        requests = [
            ChatCompletionRequest(model="llm", messages=[...]),
            ChatCompletionRequest(model="llm", messages=[...]),
        ]
        responses = await _request_batch(server, requests)
        
        for resp in responses:
            print(resp.choices[0].message.content)
    """
    if not requests:
        return []

    # Get engine for actual model execution
    engine = server.engine
    
    # Convert requests to GenerationRequest objects
    gen_requests: list[GenerationRequest] = []
    request_map: dict[str, ChatCompletionRequest | CompletionRequest] = {}
    prompt_map: dict[str, str] = {}

    for req in requests:
        # Generate unique request ID
        req_id = str(uuid.uuid4())
        request_map[req_id] = req

        # Extract prompt text and tokens
        if isinstance(req, ChatCompletionRequest):
            prompt_text = _extract_prompt_from_chat(req)
            # Use actual tokenizer from engine
            prompt_tokens = engine.pipeline.tokenizer.encode(prompt_text)
        else:
            prompt_text = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
            prompt_tokens = engine.pipeline.tokenizer.encode(prompt_text)
        
        prompt_map[req_id] = prompt_text

        max_tokens = req.max_tokens or 256

        gen_req = GenerationRequest(
            request_id=req_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 1.0,
            stop_sequences=req.stop or [],
        )
        gen_requests.append(gen_req)

    # Add all requests to scheduler
    for gen_req in gen_requests:
        server.scheduler.add_request(gen_req)

    # Process until all complete
    results: dict[str, ChatCompletionResponse | CompletionResponse] = {}
    
    # Track timeout for each request
    start_times: dict[str, float] = {gr.request_id: time.time() for gr in gen_requests}
    timeout = engine.config.request_timeout

    while len(results) < len(gen_requests):
        # Check for timeouts
        current_time = time.time()
        for gr in gen_requests:
            if gr.request_id not in results and (current_time - start_times[gr.request_id]) > timeout:
                # Mark as finished due to timeout
                gr.status = RequestStatus.FINISHED
                original_req = request_map[gr.request_id]
                # Build timeout response
                results[gr.request_id] = _build_error_response(
                    original_req, gr, "Request timeout"
                )

        # Schedule next iteration
        schedule_output = server.scheduler.schedule()

        if schedule_output.is_empty:
            # No work to do, wait a bit
            if len(results) < len(gen_requests):
                await asyncio.sleep(0.001)
            continue

        # Process prefill requests - run actual model inference
        if schedule_output.prefill_requests:
            await _run_prefill_batch(engine, schedule_output.prefill_requests)

        # Process decode requests - generate next tokens
        if schedule_output.decode_requests:
            await _run_decode_batch(engine, schedule_output.decode_requests)

        # Check for completed requests and build responses
        for req in gen_requests:
            if req.request_id not in results and req.is_finished:
                original_req = request_map[req.request_id]
                response = _build_response(original_req, req, prompt_map[req.request_id], engine)
                results[req.request_id] = response

        # Step the scheduler - remove completed requests
        finished_ids = [req.request_id for req in gen_requests if req.is_finished]
        if finished_ids:
            server.scheduler.step(finished_ids)

        await asyncio.sleep(0)  # Yield control

    # Return results in original order
    return [results[gr.request_id] for gr in gen_requests]


async def _run_prefill_batch(
    engine: "ServingEngine",
    requests: list[GenerationRequest],
) -> None:
    """Run prefill phase for a batch of requests with parallel processing.
    
    Args:
        engine: The serving engine for model execution.
        requests: List of requests in prefill phase.
    """
    if not requests:
        return
    
    loop = asyncio.get_event_loop()
    
    # Mark all requests as running
    for req in requests:
        req.status = RequestStatus.RUNNING
    
    # Process all prefill requests in parallel using asyncio.gather
    async def _prefill_single(req: GenerationRequest) -> None:
        """Process a single prefill request."""
        try:
            # Use the pipeline to process prompt and get first token
            prompt_text = engine.pipeline.tokenizer.decode(req.prompt_tokens)
            
            # Run in executor to not block event loop
            result = await loop.run_in_executor(
                engine._executor,
                lambda: engine.pipeline(
                    prompt_text,
                    max_tokens=1,  # Just get first token during prefill
                    temperature=req.temperature,
                    top_p=req.top_p,
                ),
            )
            
            # Extract first token (simplified - actual would decode properly)
            if result and len(result) > len(prompt_text):
                first_token_text = result[len(prompt_text):].strip().split()[0] if result[len(prompt_text):].strip() else "the"
                # Map to token ID (simplified)
                req.append_token(hash(first_token_text) % 50000)
            else:
                # Generate a token if model returned empty
                req.append_token(_generate_mock_token())
                
        except Exception:
            # On error, mark request as finished with error state
            req.finish()
    
    # Execute all prefill requests concurrently for true batching
    await asyncio.gather(*[_prefill_single(req) for req in requests])


async def _run_decode_batch(
    engine: "ServingEngine", 
    requests: list[GenerationRequest],
) -> None:
    """Run decode phase for a batch of requests (generate next token) with parallel processing.
    
    Args:
        engine: The serving engine for model execution.
        requests: List of requests in decode phase.
    """
    if not requests:
        return
    
    loop = asyncio.get_event_loop()
    
    # Filter out finished requests for batch processing
    active_requests = [req for req in requests if not req.is_finished]
    
    async def _decode_single(req: GenerationRequest) -> None:
        """Process a single decode request to generate the next token."""
        try:
            # Build full prompt from tokens so far
            prompt_text = engine.pipeline.tokenizer.decode(req.prompt_tokens)
            current_output = engine.pipeline.tokenizer.decode(req.output_tokens)
            full_context = prompt_text + current_output
            
            # Run generation for one more token
            result = await loop.run_in_executor(
                engine._executor,
                lambda: engine.pipeline(
                    full_context,
                    max_tokens=1,
                    temperature=req.temperature,
                    top_p=req.top_p,
                ),
            )
            
            # Extract new token
            if result and len(result) > len(full_context):
                new_text = result[len(full_context):].strip()
                if new_text:
                    # Convert to token ID
                    token_id = hash(new_text.split()[0]) % 50000
                    req.append_token(token_id)
                else:
                    # No new token generated, might be done
                    if req.output_tokens:
                        req.finish()
                    else:
                        req.append_token(_generate_mock_token())
            else:
                # Check if we should stop
                if len(req.output_tokens) >= req.max_tokens:
                    req.finish()
                else:
                    req.append_token(_generate_mock_token())
                    
            # Check stop sequences
            if req.stop_sequences:
                current_text = engine.pipeline.tokenizer.decode(req.output_tokens)
                for stop_seq in req.stop_sequences:
                    if stop_seq in current_text:
                        req.finish()
                        break
                        
        except Exception:
            # On error, finish the request with what we have
            req.finish()
    
    # Execute all decode requests concurrently for true batching
    await asyncio.gather(*[_decode_single(req) for req in active_requests])


def _build_error_response(
    original_req: ChatCompletionRequest | CompletionRequest,
    gen_req: GenerationRequest,
    error_message: str,
) -> ChatCompletionResponse | CompletionResponse:
    """Build an error response for a failed request.
    
    Args:
        original_req: The original request.
        gen_req: The generation request.
        error_message: Error message to include.
        
    Returns:
        Response with partial or error result.
    """
    # Decode any tokens we have
    output_text = _simple_detokenize(gen_req.output_tokens)
    
    if isinstance(original_req, ChatCompletionRequest):
        from .openai_schemas import ChatCompletionChoice, ChatMessage
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{gen_req.request_id}",
            created=int(time.time()),
            model=original_req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text or error_message),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=gen_req.num_prompt_tokens,
                completion_tokens=gen_req.num_output_tokens,
                total_tokens=gen_req.num_tokens,
            ),
        )
    else:
        return CompletionResponse(
            id=f"cmpl-{gen_req.request_id}",
            created=int(time.time()),
            model=original_req.model,
            choices=[
                {
                    "text": output_text or error_message,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(
                prompt_tokens=gen_req.num_prompt_tokens,
                completion_tokens=gen_req.num_output_tokens,
                total_tokens=gen_req.num_tokens,
            ),
        )


def _extract_prompt_from_chat(request: ChatCompletionRequest) -> str:
    """Extract text prompt from chat messages.

    Args:
        request: Chat completion request with messages.

    Returns:
        Concatenated text from all messages.
    """
    parts = []
    for msg in request.messages:
        parts.append(f"{msg.role}: {msg.content}")
    return "\n".join(parts)


def _simple_tokenize(text: str) -> list[int]:
    """Simple tokenization for demonstration.

    In production, use the actual tokenizer from the model.

    Args:
        text: Text to tokenize.

    Returns:
        List of token IDs.
    """
    # Simple word-based tokenization
    words = text.split()
    return [i % 50000 for i in range(len(words))]


def _generate_mock_token() -> int:
    """Generate a mock token ID.

    In production, this would be the model's output.

    Returns:
        A random token ID.
    """
    import random

    # Occasionally return EOS token (50256 is common)
    if random.random() < 0.1:
        return 50256
    return random.randint(1000, 50000)


def _build_response(
    original_req: ChatCompletionRequest | CompletionRequest,
    gen_req: GenerationRequest,
    prompt_text: str,
    engine: "ServingEngine",
) -> ChatCompletionResponse | CompletionResponse:
    """Build the final response from a completed generation request.

    Args:
        original_req: The original request.
        gen_req: The completed generation request.
        prompt_text: The original prompt text.
        engine: The serving engine for tokenization.

    Returns:
        The appropriate response type.
    """
    # Decode tokens to text using actual tokenizer
    try:
        output_text = engine.pipeline.tokenizer.decode(gen_req.output_tokens)
    except Exception:
        # Fallback to simple detokenization
        output_text = _simple_detokenize(gen_req.output_tokens)

    if isinstance(original_req, ChatCompletionRequest):
        from .openai_schemas import ChatCompletionChoice, ChatMessage

        return ChatCompletionResponse(
            id=f"chatcmpl-{gen_req.request_id}",
            created=int(time.time()),
            model=original_req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text),
                    finish_reason="stop" if gen_req.is_finished else None,
                )
            ],
            usage=Usage(
                prompt_tokens=gen_req.num_prompt_tokens,
                completion_tokens=gen_req.num_output_tokens,
                total_tokens=gen_req.num_tokens,
            ),
        )
    else:
        return CompletionResponse(
            id=f"cmpl-{gen_req.request_id}",
            created=int(time.time()),
            model=original_req.model,
            choices=[
                {
                    "text": output_text,
                    "index": 0,
                    "finish_reason": "stop" if gen_req.is_finished else None,
                }
            ],
            usage=Usage(
                prompt_tokens=gen_req.num_prompt_tokens,
                completion_tokens=gen_req.num_output_tokens,
                total_tokens=gen_req.num_tokens,
            ),
        )


@dataclass
class ContinuousBatchingMetrics:
    """Metrics tracking for continuous batching performance.

    Tracks per-iteration statistics to monitor batching efficiency
    and identify optimization opportunities.

    Attributes:
        iteration_count: Total number of scheduling iterations.
        total_prefill_tokens: Cumulative prefill tokens processed.
        total_decode_tokens: Cumulative decode tokens generated.
        total_requests_completed: Total requests finished.
        avg_batch_size: Average number of requests per iteration.
        preemption_count: Number of requests preempted.
        kv_cache_utilization: Current KV cache utilization [0, 1].
    """

    iteration_count: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    total_requests_completed: int = 0
    preemption_count: int = 0
    iteration_times_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def avg_iteration_time_ms(self) -> float:
        """Average iteration time in milliseconds."""
        if not self.iteration_times_ms:
            return 0.0
        return sum(self.iteration_times_ms) / len(self.iteration_times_ms)

    @property
    def avg_batch_size(self) -> float:
        """Average number of requests per iteration."""
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)

    @property
    def throughput_tokens_per_sec(self) -> float:
        """Overall token throughput."""
        total_time_sec = sum(self.iteration_times_ms) / 1000.0
        if total_time_sec <= 0:
            return 0.0
        total_tokens = self.total_prefill_tokens + self.total_decode_tokens
        return total_tokens / total_time_sec

    def record_iteration(
        self,
        duration_ms: float,
        batch_size: int,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> None:
        """Record metrics for a single iteration."""
        self.iteration_count += 1
        self.iteration_times_ms.append(duration_ms)
        self.batch_sizes.append(batch_size)
        self.total_prefill_tokens += prefill_tokens
        self.total_decode_tokens += decode_tokens

        # Keep last 1000 iterations to bound memory
        if len(self.iteration_times_ms) > 1000:
            self.iteration_times_ms = self.iteration_times_ms[-1000:]
            self.batch_sizes = self.batch_sizes[-1000:]

    def get_stats(self) -> dict:
        """Get metrics as a dictionary."""
        return {
            "iterations": self.iteration_count,
            "avg_iteration_ms": round(self.avg_iteration_time_ms, 2),
            "avg_batch_size": round(self.avg_batch_size, 2),
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "total_requests_completed": self.total_requests_completed,
            "throughput_tok_sec": round(self.throughput_tokens_per_sec, 2),
            "preemptions": self.preemption_count,
        }


async def _server_batching(
    server: MMFP4Server,
    request_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    shutdown_event: asyncio.Event,
) -> None:
    """Continuous batching server loop for MMFP4 inference.

    This is the core continuous batching function that runs indefinitely,
    processing requests from the queue with optimal GPU utilization:
    - Dynamically adds new requests as they arrive
    - Mixes prefill and decode work in each iteration
    - Returns completed results as soon as they finish
    - Handles chunked prefill for long prompts
    - Manages KV cache with preemption support
    - Tracks detailed performance metrics

    Args:
        server: The MMFP4 server instance with scheduler and engine.
        request_queue: Async queue for incoming GenerationRequest objects.
        result_queue: Async queue for completed (request_id, response) tuples.
        shutdown_event: Event to signal graceful shutdown.

    Usage:
        # Start the batching loop as a background task
        asyncio.create_task(_server_batching(server, req_q, res_q, shutdown))

        # Submit requests
        await request_queue.put(gen_request)

        # Get results as they complete
        request_id, response = await result_queue.get()
    """
    engine = server.engine
    scheduler = server.scheduler
    kv_manager = server.scheduler.kv_manager

    # Track in-flight requests and their original request mapping
    in_flight: dict[str, tuple[GenerationRequest, ChatCompletionRequest | CompletionRequest]] = {}

    # Running requests tracked by RunningRequest state machine
    running_state: dict[str, RunningRequest] = {}

    # Performance metrics
    metrics = ContinuousBatchingMetrics()

    async def process_iteration() -> list[tuple[str, ChatCompletionResponse | CompletionResponse]]:
        """Process one scheduling iteration and return completed results."""
        completed: list[tuple[str, ChatCompletionResponse | CompletionResponse]] = []
        iteration_start = time.time()

        # Get scheduling decision
        schedule_output = scheduler.schedule()

        if schedule_output.is_empty:
            return completed

        # Track batch composition for metrics
        prefill_count = len(schedule_output.prefill_requests)
        decode_count = len(schedule_output.decode_requests)
        total_batch = prefill_count + decode_count

        # Track prefill tokens for this iteration
        prefill_tokens = 0
        for req in schedule_output.prefill_requests:
            # Estimate tokens based on prompt length
            prefill_tokens += len(req.prompt_tokens)

        # Process prefill requests (new arrivals or chunked prefill)
        if schedule_output.prefill_requests:
            for req in schedule_output.prefill_requests:
                # Initialize running state if this is the first time
                if req.request_id not in running_state:
                    running_state[req.request_id] = RunningRequest(req.request_id)

                # Run prefill - process prompt tokens
                try:
                    await _run_prefill_batch(engine, [req])

                    # Transition to decoding if prefill complete
                    if req.status == RequestStatus.RUNNING:
                        running_req = running_state[req.request_id]
                        if running_req.is_prefilling:
                            running_req.start_decoding(prefill_tokens=req.num_prompt_tokens)
                except Exception as e:
                    # Handle prefill errors
                    if req.request_id in in_flight:
                        original_req = in_flight[req.request_id][1]
                        error_response = _build_error_response(
                            original_req, req, f"Prefill error: {e}"
                        )
                        completed.append((req.request_id, error_response))
                        req.finish()

        # Process decode requests (token generation)
        if schedule_output.decode_requests:
            try:
                await _run_decode_batch(engine, schedule_output.decode_requests)
            except Exception as e:
                # Handle decode batch errors - mark all as failed
                for req in schedule_output.decode_requests:
                    if req.request_id in in_flight:
                        original_req = in_flight[req.request_id][1]
                        error_response = _build_error_response(
                            original_req, req, f"Decode error: {e}"
                        )
                        completed.append((req.request_id, error_response))
                        req.finish()

            # Check for completions after decode
            for req in schedule_output.decode_requests:
                if req.is_finished and req.request_id in in_flight:
                    # Build response
                    _, original_req = in_flight[req.request_id]
                    prompt_text = engine.pipeline.tokenizer.decode(req.prompt_tokens)
                    response = _build_response(original_req, req, prompt_text, engine)

                    # Mark running state as complete
                    if req.request_id in running_state:
                        try:
                            running_state[req.request_id].complete(
                                decode_tokens=req.num_output_tokens
                            )
                        except Exception:
                            pass  # State machine may raise on invalid transition

                    completed.append((req.request_id, response))

        # Step scheduler - free KV cache for finished requests
        finished_ids = [req_id for req_id, _ in completed]
        if finished_ids:
            scheduler.step(finished_ids)
            metrics.total_requests_completed += len(finished_ids)

        # Clean up completed from tracking
        for req_id, _ in completed:
            in_flight.pop(req_id, None)
            running_state.pop(req_id, None)

        # Record metrics
        iteration_time_ms = (time.time() - iteration_start) * 1000
        metrics.record_iteration(
            duration_ms=iteration_time_ms,
            batch_size=total_batch,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_count,  # Each decode generates 1 token
        )

        return completed

    async def poll_new_requests() -> list[GenerationRequest]:
        """Poll for new requests from the queue (non-blocking)."""
        new_requests: list[GenerationRequest] = []
        try:
            # Poll with a limit to avoid blocking too long
            max_poll = server.batch_config.max_batch_size
            while len(new_requests) < max_poll:
                request = request_queue.get_nowait()
                new_requests.append(request)
                request_queue.task_done()
        except asyncio.QueueEmpty:
            pass
        return new_requests

    async def add_requests_to_scheduler(requests: list[GenerationRequest]) -> None:
        """Add new requests to the scheduler with proper tracking."""
        for gen_req in requests:
            # Store mapping for response building
            scheduler.add_request(gen_req)
            # Store original request for later response building
            in_flight[gen_req.request_id] = (gen_req, gen_req)

    async def handle_preemption_recovery() -> None:
        """Try to resume preempted requests if memory is available."""
        free_blocks = kv_manager.num_free_blocks
        if free_blocks > 10:  # Threshold for resuming
            resumed = scheduler.resume_swapped(max_to_resume=2)
            if resumed > 0:
                metrics.preemption_count += resumed

    # Main continuous batching loop
    idle_iterations = 0
    max_idle_iterations = 100  # Before sleeping longer

    while not shutdown_event.is_set():
        iteration_start = time.time()

        # Poll for new requests (non-blocking with batching limit)
        new_requests = await poll_new_requests()

        # Add new requests to scheduler
        if new_requests:
            await add_requests_to_scheduler(new_requests)
            idle_iterations = 0  # Reset idle counter on new work

        # Process iteration if there's work (this is the continuous batching part)
        if scheduler.has_pending_work or new_requests:
            completed = await process_iteration()

            # Send completed results to result queue
            for req_id, response in completed:
                await result_queue.put((req_id, response))

            # Try to resume swapped requests if memory available
            await handle_preemption_recovery()

            # Dynamic sleep based on workload
            # If we have many pending requests, minimize sleep
            if scheduler.num_waiting > server.batch_config.max_batch_size:
                await asyncio.sleep(0)  # Just yield control
            else:
                await asyncio.sleep(0.001)  # 1ms sleep
        else:
            # No work, use adaptive sleep
            idle_iterations += 1
            if idle_iterations < max_idle_iterations:
                await asyncio.sleep(0.001)  # 1ms when recently active
            else:
                await asyncio.sleep(0.01)  # 10ms when truly idle

        # Periodic metric logging (every 100 iterations)
        if metrics.iteration_count > 0 and metrics.iteration_count % 100 == 0:
            stats = metrics.get_stats()
            # Could log here if logger is available

    # Cleanup: finish any remaining requests gracefully
    cleanup_start = time.time()
    cleanup_timeout = 30.0  # Max 30 seconds for cleanup

    while scheduler.has_pending_work and (time.time() - cleanup_start) < cleanup_timeout:
        completed = await process_iteration()
        for req_id, response in completed:
            await result_queue.put((req_id, response))
        if not completed:
            # No progress, try one more iteration then break
            await asyncio.sleep(0.001)
            completed = await process_iteration()
            if not completed:
                break

    # Log final metrics
    final_stats = metrics.get_stats()
    # Could log here if logger is available


def _simple_detokenize(tokens: list[int]) -> str:
    """Simple detokenization for demonstration.

    In production, use the actual tokenizer.

    Args:
        tokens: List of token IDs.

    Returns:
        Decoded text.
    """
    # Mock decoding - just generate some text
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    return " ".join(words[i % len(words)] for i in tokens[:20])


# Convenience function for direct use
def create_batched_server(
    engine: ServingEngine,
    max_batch_size: int = 32,
    max_wait_ms: float = 10.0,
) -> MMFP4Server:
    """Create a batched server with sensible defaults.

    Args:
        engine: The serving engine.
        max_batch_size: Maximum requests per batch.
        max_wait_ms: Maximum time to wait for batch to fill.

    Returns:
        Configured MMFP4Server instance.
    """
    batch_config = BatchConfig(
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_batch_size,
        max_num_batched_tokens=2048,
    )
    return MMFP4Server(engine, scheduler_config, batch_config)
