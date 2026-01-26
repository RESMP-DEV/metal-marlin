"""Base classes and utilities for logit processing in guided generation.

This module provides the foundational abstractions for constraining LLM
output during autoregressive decoding. The core pattern:

1. LogitProcessor protocol: Any callable that transforms logits given context
2. LogitProcessorList: Chains multiple processors for composed constraints
3. apply_logit_mask: Efficient token masking (with optional Metal acceleration)

The key insight from outlines/guidance: most structured generation can be
expressed as "which tokens are valid next?" This yields a mask over the
vocabulary that we apply to logits before sampling.

Performance considerations:
- Token masks can be precomputed per-state for DFA-based constraints
- GPU-based masking is beneficial for large vocabularies (32k+)
- Incremental state updates avoid recomputing from scratch each step
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .._compat import HAS_MLX

if HAS_MLX:
    import mlx.core as mx
elif TYPE_CHECKING:
    import mlx.core as mx


class MaskingMode(Enum):
    """How to apply the token mask to logits."""

    NEGATIVE_INF = "negative_inf"  # Set invalid tokens to -inf
    LARGE_NEGATIVE = "large_negative"  # Set to -1e9 (avoids inf issues)
    ADDITIVE = "additive"  # Add large negative value (preserves gradients)


@runtime_checkable
class LogitProcessor(Protocol):
    """Protocol for logit processors that constrain generation.

    A LogitProcessor receives logits and context (generated token IDs so far),
    and returns modified logits with invalid tokens masked out.

    Implementations must be stateful when the constraint depends on what has
    been generated (e.g., JSON structure tracking). Call reset() between
    independent generations.
    """

    def __call__(
        self,
        logits: mx.array,
        generated_ids: list[int],
    ) -> mx.array:
        """Apply constraint to logits.

        Args:
            logits: Unnormalized logits [vocab_size] or [1, vocab_size]
            generated_ids: Previously generated token IDs (excluding prompt)

        Returns:
            Modified logits with invalid tokens masked to -inf (or similar)
        """
        ...

    def reset(self) -> None:
        """Reset internal state for a new generation sequence."""
        ...

    def get_valid_tokens(self, generated_ids: list[int]) -> set[int]:
        """Return the set of valid token IDs given current context.

        This is the core constraint logic. Subclasses implement this to
        determine which tokens can legally follow the current generation.
        """
        ...


class BaseLogitProcessor(ABC):
    """Abstract base class for logit processors with common functionality.

    Provides:
    - Vocabulary size tracking
    - Masking mode configuration
    - Efficient mask application
    - State reset interface
    """

    def __init__(
        self,
        vocab_size: int,
        masking_mode: MaskingMode = MaskingMode.NEGATIVE_INF,
    ) -> None:
        """Initialize processor with vocabulary size.

        Args:
            vocab_size: Size of the model's vocabulary
            masking_mode: How to mask invalid tokens
        """
        self.vocab_size = vocab_size
        self.masking_mode = masking_mode

    @abstractmethod
    def get_valid_tokens(self, generated_ids: list[int]) -> set[int]:
        """Return valid token IDs for current state.

        Subclasses implement the constraint logic here.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new generation."""
        ...

    def __call__(
        self,
        logits: mx.array,
        generated_ids: list[int],
    ) -> mx.array:
        """Apply constraint by masking invalid tokens.

        Args:
            logits: Logits tensor, shape [vocab_size] or [batch, vocab_size]
            generated_ids: Previously generated token IDs

        Returns:
            Masked logits tensor
        """
        valid_tokens = self.get_valid_tokens(generated_ids)
        return apply_logit_mask(
            logits,
            valid_tokens,
            self.vocab_size,
            mode=self.masking_mode,
        )


class LogitProcessorList:
    """Chain multiple logit processors together.

    Processors are applied in order. The output of each becomes the input
    to the next. All processors must operate on the same vocabulary size.

    This enables composing constraints, e.g.:
    - JSONSchemaProcessor (structure)
    - RegexProcessor (format within strings)
    - RepetitionPenaltyProcessor (quality)
    """

    def __init__(self, processors: list[LogitProcessor] | None = None) -> None:
        """Initialize with optional list of processors.

        Args:
            processors: Initial processors (can add more later)
        """
        self._processors: list[LogitProcessor] = processors or []

    def append(self, processor: LogitProcessor) -> None:
        """Add a processor to the chain."""
        self._processors.append(processor)

    def __len__(self) -> int:
        return len(self._processors)

    def __iter__(self):
        return iter(self._processors)

    def __call__(
        self,
        logits: mx.array,
        generated_ids: list[int],
    ) -> mx.array:
        """Apply all processors in sequence.

        Args:
            logits: Input logits tensor
            generated_ids: Previously generated token IDs

        Returns:
            Logits after all processors have been applied
        """
        for processor in self._processors:
            logits = processor(logits, generated_ids)
        return logits

    def reset(self) -> None:
        """Reset all processors for a new generation."""
        for processor in self._processors:
            processor.reset()

    def get_valid_tokens(self, generated_ids: list[int]) -> set[int]:
        """Get intersection of valid tokens from all processors.

        Returns tokens that are valid according to ALL constraints.
        """
        if not self._processors:
            return set()

        valid = self._processors[0].get_valid_tokens(generated_ids)
        for processor in self._processors[1:]:
            valid &= processor.get_valid_tokens(generated_ids)
        return valid


# ---------------------------------------------------------------------------
# Logit masking utilities
# ---------------------------------------------------------------------------


def apply_logit_mask(
    logits: mx.array,
    valid_tokens: set[int],
    vocab_size: int,
    mode: MaskingMode = MaskingMode.NEGATIVE_INF,
) -> mx.array:
    """Apply a token validity mask to logits.

    Creates a mask where valid tokens have 0 and invalid tokens have a large
    negative value, then adds this to logits. This is numerically stable and
    works with softmax (masked positions get ~0 probability).

    Args:
        logits: Logits tensor, shape [vocab_size] or [batch, vocab_size]
        valid_tokens: Set of token IDs that are allowed
        vocab_size: Total vocabulary size
        mode: How to apply the mask

    Returns:
        Masked logits tensor (same shape as input)
    """
    if not valid_tokens:
        # No valid tokens - return logits unchanged (will likely cause issues)
        return logits

    # Build boolean mask: True = valid, False = invalid
    mask_np = np.zeros(vocab_size, dtype=np.bool_)
    valid_list = list(valid_tokens)
    mask_np[valid_list] = True

    # Convert to MLX and expand to match logits shape
    mask = mx.array(mask_np)
    if logits.ndim > 1:
        # Broadcast for batch dimension
        mask = mx.broadcast_to(mask, logits.shape)

    # Determine mask value
    if mode == MaskingMode.NEGATIVE_INF:
        mask_value = float("-inf")
    elif mode == MaskingMode.LARGE_NEGATIVE:
        mask_value = -1e9
    else:  # ADDITIVE
        mask_value = -1e9

    # Apply mask: where mask is True, keep logits; else apply mask_value
    if HAS_MLX and mask.size == logits.size:
        return apply_logit_mask_metal(logits, mask, mask_value=mask_value)

    return mx.where(mask, logits, mx.array(mask_value))


def apply_logit_mask_vectorized(
    logits: mx.array,
    valid_token_ids: mx.array,
    vocab_size: int,
    mode: MaskingMode = MaskingMode.NEGATIVE_INF,
) -> mx.array:
    """Apply mask using vectorized MLX operations (no numpy).

    This variant takes token IDs as an MLX array, avoiding numpy conversion.
    Useful when the valid token set is already computed on GPU.

    Args:
        logits: Logits tensor
        valid_token_ids: 1D array of valid token IDs
        vocab_size: Total vocabulary size
        mode: How to apply the mask

    Returns:
        Masked logits tensor
    """
    # Create mask via scatter: start with all False, set valid positions True
    mask = mx.zeros(vocab_size, dtype=mx.bool_)

    # MLX doesn't have direct scatter, so we use index update
    # This is a workaround - for efficiency, use the numpy path above
    valid_np = np.zeros(vocab_size, dtype=np.bool_)
    valid_ids = valid_token_ids.tolist() if hasattr(valid_token_ids, "tolist") else list(valid_token_ids)
    valid_np[valid_ids] = True
    mask = mx.array(valid_np)

    if logits.ndim > 1:
        mask = mx.broadcast_to(mask, logits.shape)

    mask_value = float("-inf") if mode == MaskingMode.NEGATIVE_INF else -1e9
    return mx.where(mask, logits, mx.array(mask_value))


# ---------------------------------------------------------------------------
# Token mask caching for state-based constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedMask:
    """Precomputed token mask for a constraint state.

    For DFA-based constraints, we can precompute which tokens are valid
    from each state. This caches the mask array to avoid recomputation.
    """

    state_id: int
    valid_token_ids: frozenset[int]
    _mask_array: mx.array | None = None

    def get_mask_array(self, vocab_size: int) -> mx.array:
        """Get or create the boolean mask array."""
        if self._mask_array is not None:
            return self._mask_array

        mask_np = np.zeros(vocab_size, dtype=np.bool_)
        for tid in self.valid_token_ids:
            mask_np[tid] = True
        return mx.array(mask_np)


class MaskCache:
    """LRU cache for token masks indexed by constraint state.

    DFA-based constraints (regex, JSON structure) have a finite number of
    states. Caching masks avoids rebuilding them on repeated visits.
    """

    def __init__(self, max_size: int = 1024, vocab_size: int = 32000) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of cached masks
            vocab_size: Vocabulary size for mask arrays
        """
        self.max_size = max_size
        self.vocab_size = vocab_size
        self._cache: dict[int, mx.array] = {}
        self._access_order: list[int] = []

    def get(self, state_id: int) -> mx.array | None:
        """Get cached mask for state, or None if not cached."""
        if state_id in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(state_id)
            self._access_order.append(state_id)
            return self._cache[state_id]
        return None

    def put(self, state_id: int, valid_tokens: set[int]) -> mx.array:
        """Cache mask for state and return it.

        Args:
            state_id: Unique identifier for the constraint state
            valid_tokens: Set of valid token IDs

        Returns:
            The mask array (cached for future use)
        """
        # Build mask
        mask_np = np.zeros(self.vocab_size, dtype=np.bool_)
        for tid in valid_tokens:
            mask_np[tid] = True
        mask = mx.array(mask_np)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[state_id] = mask
        self._access_order.append(state_id)
        return mask

    def get_or_compute(
        self,
        state_id: int,
        valid_tokens_fn: callable,
    ) -> mx.array:
        """Get cached mask or compute and cache it.

        Args:
            state_id: State identifier
            valid_tokens_fn: Function that returns set[int] of valid tokens

        Returns:
            Mask array
        """
        mask = self.get(state_id)
        if mask is None:
            valid_tokens = valid_tokens_fn()
            mask = self.put(state_id, valid_tokens)
        return mask

    def clear(self) -> None:
        """Clear all cached masks."""
        self._cache.clear()
        self._access_order.clear()


# ---------------------------------------------------------------------------
# Metal-accelerated masking kernel (when vocabulary is large)
# ---------------------------------------------------------------------------

# The kernel source for parallel logit masking
_LOGIT_MASK_KERNEL_SOURCE = """
// Apply token validity mask to logits in parallel
// Each thread handles one vocabulary position

uint idx = tid;
if (idx >= vocab_size) return;

// Load mask value (1 = valid, 0 = invalid)
bool is_valid = mask[idx] > 0;

// Load logit and apply mask
float logit = logits[idx];
output[idx] = is_valid ? logit : mask_value;
"""

_LOGIT_MASK_KERNEL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _build_mask_kernel():
    """Build the Metal kernel for logit masking.

    Returns None if MLX is unavailable.
    """
    if not HAS_MLX:
        return None

    try:
        kernel = mx.fast.metal_kernel(
            name="logit_mask",
            input_names=["logits", "mask"],
            output_names=["output"],
            source=_LOGIT_MASK_KERNEL_SOURCE,
            header=_LOGIT_MASK_KERNEL_HEADER,
            ensure_row_contiguous=True,
        )
        return kernel
    except Exception:
        # Fall back to Python implementation if kernel compilation fails
        return None


# Lazily compiled kernel
_mask_kernel = None


def apply_logit_mask_metal(
    logits: mx.array,
    mask: mx.array,
    mask_value: float = float("-inf"),
) -> mx.array:
    """Apply logit mask using Metal kernel.

    This is faster than the numpy-based approach for large vocabularies
    because it avoids CPU-GPU data transfer for the mask.

    Args:
        logits: Logits tensor [vocab_size] or [batch, vocab_size]
        mask: Boolean mask [vocab_size] (True = valid)
        mask_value: Value to assign to invalid positions

    Returns:
        Masked logits
    """
    global _mask_kernel

    if _mask_kernel is None:
        _mask_kernel = _build_mask_kernel()

    if _mask_kernel is None:
        # Fall back to mx.where
        return mx.where(mask, logits, mx.array(mask_value))

    # Flatten for kernel dispatch
    flat_logits = logits.reshape(-1)
    vocab_size = flat_logits.shape[0]

    # Dispatch kernel
    output = _mask_kernel(
        inputs=[flat_logits, mask.astype(mx.float32)],
        template=[("vocab_size", vocab_size), ("mask_value", mask_value)],
        grid=(vocab_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(vocab_size,)],
        output_dtypes=[logits.dtype],
    )[0]

    return output.reshape(logits.shape)
