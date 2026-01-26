"""First-come-first-served batch scheduler with preemption.

Follows vLLM's scheduler design with three queues:
- WAITING: New sequences pending prefill
- RUNNING: Sequences actively generating tokens
- SWAPPED: Preempted sequences (logically CPU-offloaded)

Each iteration, the scheduler:
1. Allocates decode budget for running sequences (1 token each)
2. Fills remaining budget with waiting sequences (prefill)
3. Preempts running sequences if memory is exhausted

Usage:
    from metal_marlin.serving.scheduler import FCFSScheduler, SchedulerConfig
    from metal_marlin.paged.allocator import BlockAllocator

    alloc = BlockAllocator(num_blocks=512)
    sched = FCFSScheduler(SchedulerConfig(block_size=16), alloc)

    sched.add_request(request)
    output = sched.schedule()
    # output.prefill_requests, output.decode_requests, output.preempted_requests
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..paged.allocator import BlockAllocator
from ..paged.page_table import PageTable
from .request import GenerationRequest, RequestStatus, SchedulerOutput


@dataclass
class SchedulerConfig:
    """Configuration for the FCFS batch scheduler."""

    max_num_seqs: int = 64  # Max concurrent sequences
    max_num_batched_tokens: int = 2048  # Max tokens per iteration
    max_prefill_tokens: int = 1024  # Limit prefill to control latency
    block_size: int = 16


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
                self.page_table.remove_sequence(id(req))
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

        # Phase 2: Schedule prefill for waiting sequences (FCFS order).
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

        # Admit new waiting sequences
        while self.waiting and prefill_budget > 0:
            req = self.waiting[0]
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
                self.page_table.remove_sequence(id(req))
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
        Returns False if allocation fails.
        """
        seq_id = id(req)
        if not self.page_table.has_sequence(seq_id):
            success = self.page_table.add_sequence(seq_id)
            if not success:
                return False

        # Append token slots (triggers block allocation as needed)
        return self.page_table.append_tokens(seq_id, num_tokens)

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
        self.page_table.remove_sequence(id(req))
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

    @property
    def has_pending_work(self) -> bool:
        """Whether there's any work to do (requests in any queue)."""
        return bool(self.waiting or self.running or self.swapped)
