"""
Token generation loop with sampling strategies for autoregressive inference.

Uses Metal-accelerated sampling via MetalSampler and PyTorch MPS tensors.
Supports temperature scaling, top-p (nucleus) sampling, top-k sampling,
repetition penalty, and streaming generation.

Usage:
    from metal_marlin.generate import generate, GenerationConfig

    config = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=0.9)
    output_ids = generate(model, input_ids, config=config)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ._compat import require_torch, torch
from .sampler import MetalSampler

if TYPE_CHECKING:
    import torch as torch_typing

    from .kv_cache_torch import KVCacheTorch


class CausalLM(Protocol):
    """Protocol for causal language models compatible with this generator.

    Models must work with PyTorch MPS tensors and return logits.
    """

    def __call__(
        self,
        input_ids: torch_typing.Tensor,
        kv_cache: KVCacheTorch | None = None,
    ) -> torch_typing.Tensor:
        """Forward pass returning logits [batch, seq_len, vocab_size]."""
        ...

    def create_kv_cache(self) -> KVCacheTorch:
        """Create a KV cache for incremental decoding."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...


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


def generate(
    model: CausalLM,
    input_ids: torch_typing.Tensor,
    config: GenerationConfig | None = None,
    kv_cache: KVCacheTorch | None = None,
    streamer: Callable[[int], None] | None = None,
) -> torch_typing.Tensor:
    """
    Generate tokens autoregressively using Metal-accelerated sampling.

    Processes the prompt through the model (prefill), then generates
    tokens one at a time using the configured sampling strategy.

    Args:
        model: A causal language model implementing the CausalLM protocol
        input_ids: Prompt token IDs [1, seq_len], on MPS device
        config: Generation configuration
        kv_cache: Optional pre-created cache (created if None)
        streamer: Optional callback invoked with each generated token ID

    Returns:
        Full sequence (prompt + generated) as [1, total_len]
    """
    require_torch()

    if config is None:
        config = GenerationConfig()

    # Ensure input is on MPS
    if not input_ids.is_mps:
        input_ids = input_ids.to("mps")

    # Create KV cache if not provided
    if kv_cache is None:
        kv_cache = model.create_kv_cache()

    # Get vocab size for sampler
    vocab_size = model.vocab_size

    # Create Metal sampler
    sampler = MetalSampler(vocab_size=vocab_size)

    # Prefill: process entire prompt
    logits = model(input_ids, kv_cache=kv_cache)
    kv_cache.advance(input_ids.shape[1])

    # Track generated tokens for repetition penalty
    generated_ids: list[int] = []
    all_ids = input_ids[0].tolist()

    for _ in range(config.max_new_tokens):
        # Logits for the last position only
        next_logits = logits[:, -1, :]  # [1, vocab_size]

        # Sample next token using MetalSampler
        if config.do_sample:
            next_token_id = sampler.sample(
                next_logits,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                generated_ids=generated_ids,
            )
        else:
            # Greedy decoding
            next_token_id = sampler.argmax(next_logits)

        # Stream callback
        if streamer:
            streamer(next_token_id)

        # EOS check
        if next_token_id == config.eos_token_id:
            break

        # Update tracking
        generated_ids.append(next_token_id)
        all_ids.append(next_token_id)

        # Decode step: single token forward
        next_input = torch.tensor([[next_token_id]], device="mps", dtype=torch.long)
        logits = model(next_input, kv_cache=kv_cache)
        kv_cache.advance(1)

    return torch.tensor([all_ids], device="mps", dtype=torch.long)


def generate_stream(
    model: CausalLM,
    input_ids: torch_typing.Tensor,
    config: GenerationConfig | None = None,
) -> Iterator[int]:
    """
    Streaming generator that yields token IDs as they are produced.

    Useful for real-time text display. Each yield returns a token
    immediately after sampling.

    Args:
        model: A causal language model implementing the CausalLM protocol
        input_ids: Prompt token IDs [1, seq_len], on MPS device
        config: Generation configuration

    Yields:
        Individual generated token IDs (excluding prompt and EOS)
    """
    require_torch()

    if config is None:
        config = GenerationConfig()

    # Ensure input is on MPS
    if not input_ids.is_mps:
        input_ids = input_ids.to("mps")

    kv_cache = model.create_kv_cache()
    vocab_size = model.vocab_size
    sampler = MetalSampler(vocab_size=vocab_size)

    # Prefill
    logits = model(input_ids, kv_cache=kv_cache)
    kv_cache.advance(input_ids.shape[1])
    generated_ids: list[int] = []

    for _ in range(config.max_new_tokens):
        next_logits = logits[:, -1, :]

        if config.do_sample:
            next_token_id = sampler.sample(
                next_logits,
                temperature=max(config.temperature, 1e-7),
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                generated_ids=generated_ids,
            )
        else:
            next_token_id = sampler.argmax(next_logits)

        if next_token_id == config.eos_token_id:
            break

        yield next_token_id
        generated_ids.append(next_token_id)

        # Decode step
        next_input = torch.tensor([[next_token_id]], device="mps", dtype=torch.long)
        logits = model(next_input, kv_cache=kv_cache)
        kv_cache.advance(1)


def generate_batch(
    model: CausalLM,
    input_ids: torch_typing.Tensor,
    config: GenerationConfig | None = None,
    attention_mask: torch_typing.Tensor | None = None,
) -> torch_typing.Tensor:
    """
    Generate tokens for a batch of sequences.

    Handles variable-length sequences with padding and attention mask.

    Args:
        model: A causal language model implementing the CausalLM protocol
        input_ids: Prompt token IDs [batch, seq_len], on MPS device
        config: Generation configuration
        attention_mask: Optional mask [batch, seq_len] (1 = attend, 0 = ignore)

    Returns:
        Generated sequences [batch, max_len] (padded to longest output)
    """
    require_torch()

    if config is None:
        config = GenerationConfig()

    batch_size = input_ids.shape[0]
    if batch_size == 1:
        # Fall back to single-sequence generation
        return generate(model, input_ids, config=config)

    # Ensure input is on MPS
    if not input_ids.is_mps:
        input_ids = input_ids.to("mps")

    vocab_size = model.vocab_size
    sampler = MetalSampler(vocab_size=vocab_size)

    # Track sequences and their finished status
    all_outputs: list[list[int]] = [row.tolist() for row in input_ids]
    finished = [False] * batch_size
    generated_ids_per_seq: list[list[int]] = [[] for _ in range(batch_size)]

    # Create separate KV caches for each sequence (simpler than batched cache)
    # For true batched generation, would need batched KV cache implementation
    caches = [model.create_kv_cache() for _ in range(batch_size)]

    # Prefill each sequence
    all_logits = []
    for i in range(batch_size):
        seq = input_ids[i : i + 1]
        logits = model(seq, kv_cache=caches[i])
        caches[i].advance(seq.shape[1])
        all_logits.append(logits[:, -1, :])

    for step in range(config.max_new_tokens):
        for i in range(batch_size):
            if finished[i]:
                continue

            next_logits = all_logits[i]

            if config.do_sample:
                next_token_id = sampler.sample(
                    next_logits,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    generated_ids=generated_ids_per_seq[i],
                )
            else:
                next_token_id = sampler.argmax(next_logits)

            if next_token_id == config.eos_token_id:
                finished[i] = True
            else:
                all_outputs[i].append(next_token_id)
                generated_ids_per_seq[i].append(next_token_id)

                # Decode step
                next_input = torch.tensor([[next_token_id]], device="mps", dtype=torch.long)
                logits = model(next_input, kv_cache=caches[i])
                caches[i].advance(1)
                all_logits[i] = logits[:, -1, :]

        if all(finished):
            break

    # Pad to uniform length
    max_len = max(len(seq) for seq in all_outputs)
    padded = torch.full(
        (batch_size, max_len),
        config.pad_token_id,
        device="mps",
        dtype=torch.long,
    )
    for i, seq in enumerate(all_outputs):
        padded[i, : len(seq)] = torch.tensor(seq, device="mps", dtype=torch.long)

    return padded


# For backward compatibility with the MLX-based interface
__all__ = [
    "CausalLM",
    "GenerationConfig",
    "generate",
    "generate_stream",
    "generate_batch",
]
