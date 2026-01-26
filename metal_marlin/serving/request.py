"""
Request state tracking for continuous batching.

Each incoming generation request is tracked through its lifecycle:
PENDING -> RUNNING -> FINISHED (or PREEMPTED -> RUNNING -> FINISHED).

The scheduler uses GenerationRequest state to decide which requests to
prefill, which to continue decoding, and which to preempt when memory
pressure requires swapping KV cache blocks.

Usage:
    from metal_marlin.serving.request import GenerationRequest, RequestStatus

    req = GenerationRequest(
        request_id="req-001",
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=128,
        temperature=0.7,
    )
    assert req.status == RequestStatus.PENDING
    assert req.num_tokens == 4  # prompt only, no output yet
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import time


class RequestStatus(Enum):
    """Lifecycle states for a generation request."""

    PENDING = auto()    # Waiting for prefill
    RUNNING = auto()    # Actively generating tokens
    PREEMPTED = auto()  # Swapped out due to memory pressure
    FINISHED = auto()   # Completed (EOS, max_tokens, or stop sequence)


@dataclass
class GenerationRequest:
    """Tracks state of a single generation request in the continuous batch.

    Manages prompt/output tokens, block table indices for paged attention,
    generation parameters, and timing metrics for latency tracking.
    """

    request_id: str
    prompt_tokens: list[int]

    # Generation config
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)

    # State
    status: RequestStatus = RequestStatus.PENDING
    output_tokens: list[int] = field(default_factory=list)
    block_indices: list[int] = field(default_factory=list)

    # Timing
    arrival_time: float = field(default_factory=time)
    first_token_time: float | None = None
    completion_time: float | None = None

    @property
    def num_tokens(self) -> int:
        """Total tokens currently held (prompt + generated)."""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def num_prompt_tokens(self) -> int:
        """Number of prompt tokens."""
        return len(self.prompt_tokens)

    @property
    def num_output_tokens(self) -> int:
        """Number of generated output tokens."""
        return len(self.output_tokens)

    @property
    def time_to_first_token(self) -> float | None:
        """Latency from arrival to first generated token, in seconds."""
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def generation_time(self) -> float | None:
        """Total time from arrival to completion, in seconds."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None

    @property
    def decode_throughput(self) -> float | None:
        """Tokens per second during decode phase (after first token)."""
        if (
            self.completion_time is not None
            and self.first_token_time is not None
            and self.num_output_tokens > 1
        ):
            decode_time = self.completion_time - self.first_token_time
            if decode_time > 0:
                return (self.num_output_tokens - 1) / decode_time
        return None

    @property
    def is_finished(self) -> bool:
        """Whether this request has reached a stopping condition."""
        if self.status == RequestStatus.FINISHED:
            return True
        if len(self.output_tokens) >= self.max_tokens:
            return True
        return False

    def append_token(self, token_id: int) -> None:
        """Append a generated token and update timing/status."""
        if self.status == RequestStatus.PENDING:
            self.status = RequestStatus.RUNNING
            self.first_token_time = time()

        self.output_tokens.append(token_id)

        if self.is_finished:
            self.status = RequestStatus.FINISHED
            self.completion_time = time()

    def preempt(self) -> None:
        """Mark request as preempted (KV blocks will be swapped out)."""
        self.status = RequestStatus.PREEMPTED

    def resume(self) -> None:
        """Resume a preempted request."""
        if self.status == RequestStatus.PREEMPTED:
            self.status = RequestStatus.RUNNING

    def finish(self) -> None:
        """Explicitly mark request as finished (e.g., EOS or stop sequence)."""
        self.status = RequestStatus.FINISHED
        if self.completion_time is None:
            self.completion_time = time()


@dataclass
class SchedulerOutput:
    """Result of a scheduling decision for one iteration.

    The scheduler partitions active requests into three groups each step:
    - prefill_requests: new arrivals that need full prompt processing
    - decode_requests: running requests generating their next token
    - preempted_requests: requests being swapped out to free blocks
    """

    prefill_requests: list[GenerationRequest] = field(default_factory=list)
    decode_requests: list[GenerationRequest] = field(default_factory=list)
    preempted_requests: list[GenerationRequest] = field(default_factory=list)

    @property
    def num_prefill_tokens(self) -> int:
        """Total tokens to process in prefill phase."""
        return sum(r.num_prompt_tokens for r in self.prefill_requests)

    @property
    def num_decode_tokens(self) -> int:
        """Total tokens to process in decode phase (1 per request)."""
        return len(self.decode_requests)

    @property
    def total_tokens(self) -> int:
        """Total tokens this iteration will process."""
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        """Whether there's nothing to schedule."""
        return not self.prefill_requests and not self.decode_requests
