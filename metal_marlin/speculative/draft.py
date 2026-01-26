"""
Draft model interface for speculative decoding.

Speculative decoding (Leviathan et al., "Fast Inference from Transformers
via Speculative Decoding", ICML 2023) uses a small draft model to propose
tokens that a larger target model verifies in a single forward pass.

The draft model generates k candidate tokens autoregressively (cheap),
then the target model scores all k+1 positions in one pass (parallelized).
Accepted tokens skip target-model decode steps; rejected tokens fall back
to the target model's own prediction at the rejection point.

This module defines the abstract DraftModel interface and two concrete
implementations: SmallModelDraft (uses a smaller transformer) and
NGramDraft (uses online n-gram statistics for zero-cost drafting).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from ..kv_cache import KVCache


class CausalLMDraft(Protocol):
    """Protocol for models usable as draft models."""

    def __call__(self, input_ids: mx.array, kv_cache: KVCache | None = None) -> mx.array: ...
    def create_kv_cache(self) -> KVCache: ...


@dataclass
class DraftOutput:
    """Output from a draft model's speculative generation.

    Attributes:
        tokens: Proposed token IDs, shape [batch, num_speculative].
        probs: Full probability distributions for each proposed position,
            shape [batch, num_speculative, vocab_size]. Used by the verifier
            to compute acceptance probabilities.
    """

    tokens: mx.array  # [batch, num_speculative]
    probs: mx.array   # [batch, num_speculative, vocab_size]


class DraftModel(ABC):
    """Abstract interface for draft models in speculative decoding.

    Implementations must produce both token predictions and their full
    probability distributions. The distributions are needed for the
    modified rejection sampling used in verification.
    """

    @abstractmethod
    def speculate(
        self,
        input_ids: mx.array,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        """Generate speculative token proposals.

        Args:
            input_ids: Current context token IDs, shape [batch, seq_len].
                For decode-phase usage this is typically [batch, 1] containing
                the last accepted token.
            kv_cache: Target model's KV cache (for context). Draft models
                with their own caches may ignore this.
            num_tokens: Number of tokens to speculatively generate.

        Returns:
            DraftOutput with proposed tokens and their probability distributions.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (caches, statistics) for a new sequence."""
        ...


class SmallModelDraft(DraftModel):
    """Use a smaller causal LM (e.g., 1B params) for drafting.

    The draft model runs its own KV cache independently of the target.
    On each call to speculate(), it generates num_tokens autoregressively
    using greedy decoding (argmax) for maximum throughput.

    The draft model's cache is synchronized: after the target model accepts
    n tokens, the caller should advance the draft cache by feeding those
    tokens or simply reset and re-prefill.
    """

    def __init__(self, model: CausalLMDraft, max_speculative: int = 4):
        self.model = model
        self.max_speculative = max_speculative
        self._cache: KVCache | None = None
        self._cache_seq_len: int = 0

    def speculate(
        self,
        input_ids: mx.array,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        num_tokens = min(num_tokens, self.max_speculative)
        batch_size = input_ids.shape[0]

        # Initialize draft cache if needed
        if self._cache is None:
            self._cache = self.model.create_kv_cache()

        tokens: list[mx.array] = []
        probs: list[mx.array] = []

        # Run the draft model autoregressively
        current_ids = input_ids
        for _ in range(num_tokens):
            logits = self.model(current_ids, kv_cache=self._cache)
            # Take logits for last position: [batch, vocab]
            last_logits = logits[:, -1, :]

            next_probs = mx.softmax(last_logits, axis=-1)
            next_token = mx.argmax(next_probs, axis=-1)  # [batch]

            tokens.append(next_token)
            probs.append(next_probs)

            # Advance draft cache
            self._cache.advance(current_ids.shape[1])
            current_ids = next_token.reshape(batch_size, 1)

        return DraftOutput(
            tokens=mx.stack(tokens, axis=1),   # [batch, num_tokens]
            probs=mx.stack(probs, axis=1),     # [batch, num_tokens, vocab]
        )

    def reset(self) -> None:
        """Reset the draft model's KV cache for a new sequence."""
        if self._cache is not None:
            self._cache.reset()
        self._cache = None
        self._cache_seq_len = 0

    def sync_after_accept(self, accepted_tokens: mx.array) -> None:
        """Feed accepted tokens into draft cache to keep it synchronized.

        After the target model accepts some subset of the draft's proposals,
        call this so the draft cache reflects the true sequence state.

        In practice it's often simpler to reset and re-prefill the draft
        model, but for long sequences this incremental sync saves compute.

        Args:
            accepted_tokens: The tokens accepted by the target, [batch, n_accepted].
        """
        if self._cache is None:
            self._cache = self.model.create_kv_cache()
        # Run accepted tokens through draft model to populate its cache
        self.model(accepted_tokens, kv_cache=self._cache)
        self._cache.advance(accepted_tokens.shape[1])


class NGramDraft(DraftModel):
    """Use online n-gram statistics for zero-cost speculative drafting.

    This approach requires no additional model parameters. It builds n-gram
    frequency tables from the generated text so far, then uses the most
    frequent continuations as draft proposals.

    Works best for repetitive or formulaic text (code, structured data).
    Falls back to uniform distributions when no n-gram match exists.

    The probability estimates are crude (based on count ratios) but sufficient
    for the verifier's rejection sampling since mismatches just result in
    rejection with fallback to the target model's own prediction.
    """

    def __init__(self, ngram_size: int = 3, vocab_size: int = 32000):
        self.ngram_size = ngram_size
        self.vocab_size = vocab_size
        self.ngram_counts: dict[tuple[int, ...], dict[int, int]] = {}
        self._history: list[int] = []

    def update_ngrams(self, token_ids: list[int]) -> None:
        """Update n-gram statistics from newly generated tokens.

        Should be called after each verification step with the accepted tokens.

        Args:
            token_ids: Sequence of token IDs to learn from.
        """
        self._history.extend(token_ids)
        # Build n-grams from the extended history
        for i in range(max(0, len(self._history) - self.ngram_size - len(token_ids)),
                       len(self._history) - self.ngram_size):
            context = tuple(self._history[i:i + self.ngram_size])
            next_token = self._history[i + self.ngram_size]
            if context not in self.ngram_counts:
                self.ngram_counts[context] = {}
            self.ngram_counts[context][next_token] = (
                self.ngram_counts[context].get(next_token, 0) + 1
            )

    def speculate(
        self,
        input_ids: mx.array,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        batch_size = input_ids.shape[0]

        # Extract the last ngram_size tokens from input as initial context
        seq_len = input_ids.shape[1]
        if seq_len < self.ngram_size:
            # Not enough context; return uniform guesses (token 0)
            return self._uniform_fallback(batch_size, num_tokens)

        # Work with first batch element (n-gram draft is not naturally batched)
        context_ids = input_ids[0, -self.ngram_size:].tolist()

        tokens: list[int] = []
        probs_list: list[mx.array] = []

        for _ in range(num_tokens):
            context = tuple(context_ids[-self.ngram_size:])
            counts = self.ngram_counts.get(context)

            if counts:
                # Build probability distribution from counts
                total = sum(counts.values())
                prob_dist = mx.zeros((self.vocab_size,), dtype=mx.float32)
                for tok, count in counts.items():
                    if tok < self.vocab_size:
                        prob_dist[tok] = count / total

                # Pick the most likely token
                best_token = max(counts, key=counts.get)  # type: ignore[arg-type]
                if best_token >= self.vocab_size:
                    best_token = 0
                    prob_dist = mx.ones((self.vocab_size,), dtype=mx.float32) / self.vocab_size
            else:
                # No n-gram match: uniform distribution, predict token 0
                prob_dist = mx.ones((self.vocab_size,), dtype=mx.float32) / self.vocab_size
                best_token = 0

            tokens.append(best_token)
            probs_list.append(prob_dist)

            # Extend context for next prediction
            context_ids.append(best_token)

        # Stack into batch-compatible shapes
        tokens_arr = mx.array(tokens, dtype=mx.uint32).reshape(1, num_tokens)
        probs_arr = mx.stack(probs_list, axis=0).reshape(1, num_tokens, self.vocab_size)

        # Broadcast to batch dimension if needed
        if batch_size > 1:
            tokens_arr = mx.broadcast_to(tokens_arr, (batch_size, num_tokens))
            probs_arr = mx.broadcast_to(probs_arr, (batch_size, num_tokens, self.vocab_size))

        return DraftOutput(tokens=tokens_arr, probs=probs_arr)

    def reset(self) -> None:
        """Clear n-gram statistics and history for a new sequence."""
        self.ngram_counts.clear()
        self._history.clear()

    def _uniform_fallback(self, batch_size: int, num_tokens: int) -> DraftOutput:
        """Return uniform distributions when no context is available."""
        tokens = mx.zeros((batch_size, num_tokens), dtype=mx.uint32)
        probs = mx.broadcast_to(
            mx.ones((1, 1, self.vocab_size), dtype=mx.float32) / self.vocab_size,
            (batch_size, num_tokens, self.vocab_size),
        )
        return DraftOutput(tokens=tokens, probs=probs)
