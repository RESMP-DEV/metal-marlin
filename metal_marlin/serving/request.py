"""
Request state tracking for continuous batching.

Each incoming generation request is tracked through its lifecycle:
PENDING -> RUNNING -> FINISHED (or PREEMPTED -> RUNNING -> FINISHED).

The scheduler uses GenerationRequest state to decide which requests to
prefill, which to continue decoding, and which to preempt when memory
pressure requires swapping KV cache blocks.

For fine-grained tracking of actively running requests, RunningRequest
provides a state machine that tracks PREFILLING -> DECODING -> COMPLETED
transitions with validation and state history.

Usage:
    from metal_marlin.serving.request import (
        GenerationRequest,
        RequestStatus,
        RunningRequest,
        RunningRequestState,
    )

    req = GenerationRequest(
        request_id="req-001",
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=128,
        temperature=0.7,
    )
    assert req.status == RequestStatus.PENDING
    assert req.num_tokens == 4  # prompt only, no output yet

    # RunningRequest state machine for fine-grained tracking
    running = RunningRequest(request_id="req-001")
    assert running.state == RunningRequestState.PREFILLING
    running.start_decoding(prefill_tokens=4)
    assert running.state == RunningRequestState.DECODING
    running.complete(decode_tokens=10)
    assert running.state == RunningRequestState.COMPLETED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import time


class RequestStatus(Enum):
    """Lifecycle states for a generation request."""

    PENDING = auto()  # Waiting for prefill
    RUNNING = auto()  # Actively generating tokens
    PREEMPTED = auto()  # Swapped out due to memory pressure
    FINISHED = auto()  # Completed (EOS, max_tokens, or stop sequence)


class RunningRequestState(Enum):
    """Granular states for a request that is actively running.

    This state machine tracks the internal progression of a request
    through its execution lifecycle, independent of the high-level
    RequestStatus used by the scheduler.
    """

    PREFILLING = auto()  # Processing prompt tokens
    DECODING = auto()  # Generating output tokens one by one
    COMPLETED = auto()  # All tokens generated
    FAILED = auto()  # Error occurred during generation
    CANCELLED = auto()  # Request cancelled by user


class RunningRequestTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class RunningRequest:
    """State machine for tracking a running request's completion lifecycle.

    This class provides a finite state machine for managing the detailed
    progression of a request through its execution, with validation of
    valid transitions and tracking of state history.

    State transitions:
        PREFILLING -> DECODING (on prefill completion)
        DECODING -> COMPLETED (on EOS or max_tokens)
        DECODING -> FAILED (on error)
        PREFILLING -> CANCELLED (user cancellation)
        DECODING -> CANCELLED (user cancellation)

    Example:
        req = RunningRequest(request_id="req-001")
        assert req.state == RunningRequestState.PREFILLING
        req.start_decoding()
        assert req.state == RunningRequestState.DECODING
        req.complete()
        assert req.state == RunningRequestState.COMPLETED
        assert req.is_terminal
    """

    # Valid state transitions: from_state -> {to_states}
    _VALID_TRANSITIONS: dict[RunningRequestState, set[RunningRequestState]] = {
        RunningRequestState.PREFILLING: {
            RunningRequestState.DECODING,
            RunningRequestState.CANCELLED,
        },
        RunningRequestState.DECODING: {
            RunningRequestState.COMPLETED,
            RunningRequestState.FAILED,
            RunningRequestState.CANCELLED,
        },
        RunningRequestState.COMPLETED: set(),  # Terminal
        RunningRequestState.FAILED: set(),  # Terminal
        RunningRequestState.CANCELLED: set(),  # Terminal
    }

    def __init__(self, request_id: str) -> None:
        """Initialize a new running request in PREFILLING state."""
        self._request_id: str = request_id
        self._state: RunningRequestState = RunningRequestState.PREFILLING
        self._state_history: list[tuple[float, RunningRequestState]] = [
            (time(), RunningRequestState.PREFILLING)
        ]
        self._error_message: str | None = None
        self._prefill_tokens_processed: int = 0
        self._decode_tokens_generated: int = 0

    @property
    def request_id(self) -> str:
        """Unique identifier for this request."""
        return self._request_id

    @property
    def state(self) -> RunningRequestState:
        """Current state of the request."""
        return self._state

    @property
    def is_terminal(self) -> bool:
        """Whether the request has reached a terminal state."""
        return self._state in (
            RunningRequestState.COMPLETED,
            RunningRequestState.FAILED,
            RunningRequestState.CANCELLED,
        )

    @property
    def is_prefilling(self) -> bool:
        """Whether the request is currently prefilling."""
        return self._state == RunningRequestState.PREFILLING

    @property
    def is_decoding(self) -> bool:
        """Whether the request is currently decoding."""
        return self._state == RunningRequestState.DECODING

    @property
    def error_message(self) -> str | None:
        """Error message if in FAILED state, None otherwise."""
        return self._error_message

    @property
    def prefill_tokens_processed(self) -> int:
        """Number of prompt tokens processed during prefill."""
        return self._prefill_tokens_processed

    @property
    def decode_tokens_generated(self) -> int:
        """Number of output tokens generated during decode."""
        return self._decode_tokens_generated

    @property
    def state_history(self) -> list[tuple[float, RunningRequestState]]:
        """History of state transitions with timestamps."""
        return self._state_history.copy()

    def start_decoding(self, prefill_tokens: int) -> None:
        """Transition from PREFILLING to DECODING state.

        Args:
            prefill_tokens: Number of prompt tokens processed.

        Raises:
            RunningRequestTransitionError: If not currently in PREFILLING state.
        """
        self._transition(RunningRequestState.DECODING)
        self._prefill_tokens_processed = prefill_tokens

    def complete(self, decode_tokens: int) -> None:
        """Transition from DECODING to COMPLETED state.

        Args:
            decode_tokens: Total number of output tokens generated.

        Raises:
            RunningRequestTransitionError: If not currently in DECODING state.
        """
        self._transition(RunningRequestState.COMPLETED)
        self._decode_tokens_generated = decode_tokens

    def fail(self, message: str) -> None:
        """Transition to FAILED state.

        Args:
            message: Error message describing the failure.

        Raises:
            RunningRequestTransitionError: If already in a terminal state.
        """
        self._transition(RunningRequestState.FAILED)
        self._error_message = message

    def cancel(self) -> None:
        """Transition to CANCELLED state.

        Can be called from PREFILLING or DECODING states.

        Raises:
            RunningRequestTransitionError: If already in a terminal state.
        """
        self._transition(RunningRequestState.CANCELLED)

    def _transition(self, new_state: RunningRequestState) -> None:
        """Internal method to perform state transition with validation.

        Args:
            new_state: The state to transition to.

        Raises:
            RunningRequestTransitionError: If the transition is invalid.
        """
        if new_state not in self._VALID_TRANSITIONS[self._state]:
            raise RunningRequestTransitionError(
                f"Invalid transition from {self._state.name} "
                f"to {new_state.name} for request {self._request_id}. "
                f"Valid transitions from {self._state.name}: "
                f"{', '.join(s.name for s in self._VALID_TRANSITIONS[self._state])}"
            )

        self._state = new_state
        self._state_history.append((time(), new_state))

    def __repr__(self) -> str:
        return (
            f"RunningRequest(id={self._request_id}, "
            f"state={self._state.name}, "
            f"prefill_tokens={self._prefill_tokens_processed}, "
            f"decode_tokens={self._decode_tokens_generated})"
        )


class RequestTimeoutError(Exception):
    """Raised when a generation request exceeds the configured timeout."""

    pass


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
