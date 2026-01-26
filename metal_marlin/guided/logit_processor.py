"""Base classes and utilities for logit processing in guided generation.

This module provides the foundational abstractions for constraining LLM
output during autoregressive decoding. The core pattern:

1. LogitProcessor protocol: Any callable that transforms logits given context
2. LogitProcessorList: Chains multiple processors for composed constraints
3. apply_logit_mask: Efficient token masking using PyTorch

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

from .._compat import HAS_TORCH, require_torch

if HAS_TORCH:
    import torch
elif TYPE_CHECKING:
    import torch


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
        logits: torch.Tensor,
        generated_ids: list[int],
    ) -> torch.Tensor:
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
        require_torch("guided generation")
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
        logits: torch.Tensor,
        generated_ids: list[int],
    ) -> torch.Tensor:
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
        logits: torch.Tensor,
        generated_ids: list[int],
    ) -> torch.Tensor:
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
    logits: torch.Tensor,
    valid_tokens: set[int],
    vocab_size: int,
    mode: MaskingMode = MaskingMode.NEGATIVE_INF,
) -> torch.Tensor:
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
    require_torch("logit masking")

    if not valid_tokens:
        # No valid tokens - return logits unchanged (will likely cause issues)
        return logits

    # Build boolean mask: True = valid, False = invalid
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    valid_indices = torch.tensor(list(valid_tokens), dtype=torch.long, device=logits.device)
    mask[valid_indices] = True

    # Expand mask to match logits shape
    if logits.ndim > 1:
        mask = mask.unsqueeze(0).expand(logits.shape[0], -1)

    # Determine mask value
    if mode == MaskingMode.NEGATIVE_INF:
        mask_value = float("-inf")
    elif mode == MaskingMode.LARGE_NEGATIVE:
        mask_value = -1e9
    else:  # ADDITIVE
        mask_value = -1e9

    # Apply mask: where mask is True, keep logits; else apply mask_value
    return torch.where(
        mask, logits, torch.tensor(mask_value, device=logits.device, dtype=logits.dtype)
    )


def apply_logit_mask_vectorized(
    logits: torch.Tensor,
    valid_token_ids: torch.Tensor,
    vocab_size: int,
    mode: MaskingMode = MaskingMode.NEGATIVE_INF,
) -> torch.Tensor:
    """Apply mask using vectorized PyTorch operations.

    This variant takes token IDs as a tensor, avoiding list conversion.
    Useful when the valid token set is already computed on GPU.

    Args:
        logits: Logits tensor
        valid_token_ids: 1D tensor of valid token IDs
        vocab_size: Total vocabulary size
        mode: How to apply the mask

    Returns:
        Masked logits tensor
    """
    require_torch("vectorized logit masking")

    # Create mask via scatter
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    mask.scatter_(0, valid_token_ids, True)

    if logits.ndim > 1:
        mask = mask.unsqueeze(0).expand(logits.shape[0], -1)

    mask_value = float("-inf") if mode == MaskingMode.NEGATIVE_INF else -1e9
    return torch.where(
        mask, logits, torch.tensor(mask_value, device=logits.device, dtype=logits.dtype)
    )


# ---------------------------------------------------------------------------
# Token mask caching for state-based constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedMask:
    """Precomputed token mask for a constraint state.

    For DFA-based constraints, we can precompute which tokens are valid
    from each state. This caches the mask tensor to avoid recomputation.
    """

    state_id: int
    valid_token_ids: frozenset[int]
    _mask_tensor: torch.Tensor | None = None

    def get_mask_tensor(self, vocab_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Get or create the boolean mask tensor."""
        if self._mask_tensor is not None and self._mask_tensor.device == torch.device(device):
            return self._mask_tensor

        require_torch("mask tensor creation")
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if self.valid_token_ids:
            valid_indices = torch.tensor(
                list(self.valid_token_ids), dtype=torch.long, device=device
            )
            mask[valid_indices] = True
        return mask


class MaskCache:
    """LRU cache for token masks indexed by constraint state.

    DFA-based constraints (regex, JSON structure) have a finite number of
    states. Caching masks avoids rebuilding them on repeated visits.
    """

    def __init__(
        self,
        max_size: int = 1024,
        vocab_size: int = 32000,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of cached masks
            vocab_size: Vocabulary size for mask tensors
            device: Device for mask tensors
        """
        require_torch("mask caching")
        self.max_size = max_size
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self._cache: dict[int, torch.Tensor] = {}
        self._access_order: list[int] = []

    def get(self, state_id: int) -> torch.Tensor | None:
        """Get cached mask for state, or None if not cached."""
        if state_id in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(state_id)
            self._access_order.append(state_id)
            return self._cache[state_id]
        return None

    def put(self, state_id: int, valid_tokens: set[int]) -> torch.Tensor:
        """Cache mask for state and return it.

        Args:
            state_id: Unique identifier for the constraint state
            valid_tokens: Set of valid token IDs

        Returns:
            The mask tensor (cached for future use)
        """
        # Build mask
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        if valid_tokens:
            valid_indices = torch.tensor(list(valid_tokens), dtype=torch.long, device=self.device)
            mask[valid_indices] = True

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
    ) -> torch.Tensor:
        """Get cached mask or compute and cache it.

        Args:
            state_id: State identifier
            valid_tokens_fn: Function that returns set[int] of valid tokens

        Returns:
            Mask tensor
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

    def to(self, device: torch.device | str) -> MaskCache:
        """Move all cached masks to a new device.

        Args:
            device: Target device

        Returns:
            self (for method chaining)
        """
        new_device = torch.device(device)
        if new_device != self.device:
            self._cache = {k: v.to(new_device) for k, v in self._cache.items()}
            self.device = new_device
        return self
