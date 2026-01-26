"""
Token generation loop with sampling strategies for autoregressive inference.

Supports temperature scaling, top-p (nucleus) sampling, top-k sampling,
repetition penalty, and streaming generation.

Usage:
    from metal_marlin.python.generate import generate, GenerationConfig

    config = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=0.9)
    output_ids = generate(model, input_ids, config=config)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx

from .kv_cache import KVCache

if TYPE_CHECKING:
    pass


class CausalLM(Protocol):
    """Protocol for causal language models compatible with this generator."""

    def __call__(self, input_ids: mx.array, kv_cache: KVCache | None = None) -> mx.array: ...
    def create_kv_cache(self) -> KVCache: ...


@dataclass
class GenerationConfig:
    """Configuration for token generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    eos_token_id: int = 2
    pad_token_id: int = 0
    do_sample: bool = True


def sample_top_p(logits: mx.array, p: float) -> mx.array:
    """
    Nucleus (top-p) sampling.

    Sorts tokens by probability, accumulates until cumulative probability
    exceeds p, then samples from the retained set.

    Args:
        logits: Unnormalized logits [vocab_size] or [1, vocab_size]
        p: Cumulative probability threshold

    Returns:
        Sampled token index as scalar array
    """
    # Ensure 1D for processing
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[None, :]

    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

    # Keep tokens where cumulative prob hasn't exceeded p yet,
    # always keeping at least the top token
    sorted_mask = mx.concatenate(
        [mx.ones(cumulative_probs[..., :1].shape, dtype=mx.bool_),
         cumulative_probs[..., :-1] < p],
        axis=-1,
    )

    # Mask out tokens beyond the nucleus
    masked_logits = mx.where(sorted_mask, sorted_logits, mx.array(float("-inf")))

    # Sample from filtered distribution using categorical (takes unnormalized logits)
    sampled_idx = mx.random.categorical(masked_logits)

    # Map back to original vocabulary indices
    token = mx.take_along_axis(sorted_indices, sampled_idx[..., None].astype(mx.uint32), axis=-1)
    result = token.reshape(-1)[0]
    return result


def sample_top_k(logits: mx.array, k: int) -> mx.array:
    """
    Top-k sampling.

    Retains only the k highest-logit tokens, then samples from that set.

    Args:
        logits: Unnormalized logits [vocab_size] or [1, vocab_size]
        k: Number of top tokens to retain

    Returns:
        Sampled token index as scalar array
    """
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[None, :]

    # Get indices of top-k by sorting
    sorted_indices = mx.argsort(-logits, axis=-1)
    top_k_indices = sorted_indices[..., :k]
    top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)

    # Sample from top-k distribution
    sampled_idx = mx.random.categorical(top_k_logits)

    # Map back to original vocabulary indices
    token = mx.take_along_axis(top_k_indices, sampled_idx[..., None].astype(mx.uint32), axis=-1)
    result = token.reshape(-1)[0]
    return result


def apply_repetition_penalty(
    logits: mx.array,
    generated_ids: list[int],
    penalty: float,
) -> mx.array:
    """
    Apply repetition penalty to already-generated tokens.

    Tokens that have appeared before get their logits divided (if positive)
    or multiplied (if negative) by the penalty factor, discouraging repetition.

    Args:
        logits: Logits tensor [1, 1, vocab_size] or [1, vocab_size]
        generated_ids: List of previously generated token IDs
        penalty: Multiplicative penalty (1.0 = no penalty)

    Returns:
        Modified logits with repetition penalty applied
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    # Deduplicate token IDs
    unique_ids = list(set(generated_ids))
    if not unique_ids:
        return logits

    # Build penalty mask using vectorized operations
    # Extract the logits for tokens we've already generated
    token_indices = mx.array(unique_ids, dtype=mx.uint32)

    # Get the relevant logits
    flat_logits = logits.reshape(-1, logits.shape[-1])
    token_logits = flat_logits[0, token_indices]

    # Apply penalty: divide positive logits, multiply negative logits
    penalized = mx.where(
        token_logits > 0,
        token_logits / penalty,
        token_logits * penalty,
    )

    # Scatter back into the logits tensor
    result = mx.array(flat_logits)
    for i, tid in enumerate(unique_ids):
        result[0, tid] = penalized[i]

    return result.reshape(logits.shape)


def generate(
    model: CausalLM,
    input_ids: mx.array,
    config: GenerationConfig = GenerationConfig(),
    kv_cache: KVCache | None = None,
    streamer: Callable[[int], None] | None = None,
) -> mx.array:
    """
    Generate tokens autoregressively.

    Processes the prompt through the model (prefill), then generates
    tokens one at a time using the configured sampling strategy.

    Args:
        model: A causal language model implementing the CausalLM protocol
        input_ids: Prompt token IDs [1, seq_len]
        config: Generation configuration
        kv_cache: Optional pre-created cache (created if None)
        streamer: Optional callback invoked with each generated token ID

    Returns:
        Full sequence (prompt + generated) as [1, total_len]
    """
    if kv_cache is None:
        kv_cache = model.create_kv_cache()

    # Prefill: process entire prompt
    logits = model(input_ids, kv_cache=kv_cache)
    kv_cache.advance(input_ids.shape[1])

    # Track generated tokens for repetition penalty
    generated_ids: list[int] = []
    all_ids = input_ids.tolist()[0]

    for _ in range(config.max_new_tokens):
        # Logits for the last position only
        next_logits = logits[:, -1:, :]  # [1, 1, vocab_size]

        # Temperature scaling
        if config.temperature > 0:
            next_logits = next_logits / config.temperature

        # Repetition penalty
        if config.repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(
                next_logits, generated_ids, config.repetition_penalty
            )

        # Sampling
        if config.do_sample:
            # Flatten to [vocab_size] for sampling functions
            flat_logits = next_logits.reshape(-1)

            if config.top_p < 1.0:
                next_token_id = int(sample_top_p(flat_logits, config.top_p).item())
            elif config.top_k > 0:
                next_token_id = int(sample_top_k(flat_logits, config.top_k).item())
            else:
                # Pure temperature sampling (no filtering)
                sampled = mx.random.categorical(flat_logits[None, :])
                next_token_id = int(sampled[0].item())
        else:
            # Greedy decoding
            next_token_id = int(mx.argmax(next_logits.reshape(-1)).item())

        # Stream callback
        if streamer:
            streamer(next_token_id)

        # EOS check
        if next_token_id == config.eos_token_id:
            break

        # Update tracking
        generated_ids.append(next_token_id)
        all_ids.append(next_token_id)

        # Evaluate to avoid graph buildup
        mx.eval(mx.array(0))

        # Decode step: single token forward
        next_input = mx.array([[next_token_id]])
        logits = model(next_input, kv_cache=kv_cache)
        kv_cache.advance(1)

    return mx.array([all_ids])


def generate_stream(
    model: CausalLM,
    input_ids: mx.array,
    config: GenerationConfig = GenerationConfig(),
) -> Iterator[int]:
    """
    Streaming generator that yields token IDs as they are produced.

    Useful for real-time text display. Each yield triggers an mx.eval
    to materialize the token before returning it.

    Args:
        model: A causal language model implementing the CausalLM protocol
        input_ids: Prompt token IDs [1, seq_len]
        config: Generation configuration

    Yields:
        Individual generated token IDs (excluding prompt and EOS)
    """
    kv_cache = model.create_kv_cache()

    # Prefill
    logits = model(input_ids, kv_cache=kv_cache)
    kv_cache.advance(input_ids.shape[1])
    generated_ids: list[int] = []

    for _ in range(config.max_new_tokens):
        next_logits = logits[:, -1:, :]

        # Temperature (clamp to avoid division by zero)
        temp = max(config.temperature, 1e-7)
        next_logits = next_logits / temp

        # Repetition penalty
        if config.repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(
                next_logits, generated_ids, config.repetition_penalty
            )

        flat_logits = next_logits.reshape(-1)

        if config.do_sample:
            if config.top_p < 1.0:
                next_token_id = int(sample_top_p(flat_logits, config.top_p).item())
            elif config.top_k > 0:
                next_token_id = int(sample_top_k(flat_logits, config.top_k).item())
            else:
                sampled = mx.random.categorical(flat_logits[None, :])
                next_token_id = int(sampled[0].item())
        else:
            next_token_id = int(mx.argmax(flat_logits).item())

        if next_token_id == config.eos_token_id:
            break

        yield next_token_id
        generated_ids.append(next_token_id)

        # Decode step
        next_input = mx.array([[next_token_id]])
        logits = model(next_input, kv_cache=kv_cache)
        kv_cache.advance(1)
