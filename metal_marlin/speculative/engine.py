"""
Speculative decoding engine: orchestrates draft-then-verify generation.

The engine coordinates a draft model (cheap, autoregressive) with a target model
(expensive, but verifies K positions in parallel). Expected speedup is 2-4x for
well-matched draft/target pairs, with the exact ratio depending on acceptance rate.

Key design points:
- Adaptive speculation: dynamically adjusts the number of draft tokens based
  on recent acceptance rates. High acceptance -> speculate more; low -> less.
- Correct distribution: the verify step guarantees output matches the target
  model's distribution exactly, regardless of draft quality.
- One target forward per step: K draft tokens are verified in a single target
  model evaluation, amortizing the expensive computation.
- Draft sync: after verification, the draft model's cache is synchronized
  with accepted tokens to avoid re-prefilling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from ..kv_cache import KVCache
from .draft import DraftModel, DraftOutput, NGramDraft, SmallModelDraft
from .verify import VerifyResult, verify_speculative


class TargetModel:
    """Protocol for target models used in speculative decoding.

    Any model with __call__(input_ids, kv_cache) -> logits and
    create_kv_cache() is compatible. We use a plain class rather than
    typing.Protocol to avoid runtime overhead.
    """

    def __call__(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        """Forward pass returning logits [batch, seq_len, vocab]."""
        raise NotImplementedError

    def create_kv_cache(self) -> KVCache:
        """Create a fresh KV cache for this model."""
        raise NotImplementedError


@dataclass
class SpeculativeConfig:
    """Configuration for the speculative decoding engine.

    Attributes:
        num_speculative_tokens: Initial (and target) number of tokens to speculate.
        min_speculative_tokens: Adaptive floor.
        max_speculative_tokens: Adaptive ceiling (can exceed initial if acceptance is high).
        temperature: Sampling temperature for target model verification.
        top_p: Nucleus sampling threshold.
        acceptance_threshold: Reduce K when windowed rate drops below this.
        increase_threshold: Increase K when windowed rate exceeds this.
        history_window: Number of recent steps for adaptive averaging.
        eos_token_id: End-of-sequence token ID.
    """

    num_speculative_tokens: int = 4
    min_speculative_tokens: int = 1
    max_speculative_tokens: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    acceptance_threshold: float = 0.4
    increase_threshold: float = 0.8
    history_window: int = 10
    eos_token_id: int = 2


@dataclass
class StepResult:
    """Result from a single speculative generation step.

    Attributes:
        new_tokens: Produced tokens (accepted + next), shape [batch, max_new].
            Padded with zeros beyond num_new_tokens[b] for each batch element.
        num_new_tokens: Valid new token count per batch element, [batch].
        num_target_calls: Target model forward passes used (always 1).
        num_draft_tokens: Tokens proposed by draft model this step.
        num_accepted: Draft tokens accepted per batch element, [batch].
        acceptance_rate: Average acceptance fraction across batch.
    """

    new_tokens: Tensor  # [batch, max_new]
    num_new_tokens: Tensor  # [batch]
    num_target_calls: int
    num_draft_tokens: int
    num_accepted: Tensor  # [batch]
    acceptance_rate: float


@dataclass
class GenerationStats:
    """Cumulative statistics for a speculative generation run."""

    total_tokens: int = 0
    total_target_calls: int = 0
    total_draft_steps: int = 0
    total_accepted: int = 0
    total_proposed: int = 0

    @property
    def tokens_per_target_call(self) -> float:
        """Average tokens produced per expensive target forward pass."""
        if self.total_target_calls == 0:
            return 0.0
        return self.total_tokens / self.total_target_calls

    @property
    def overall_acceptance_rate(self) -> float:
        """Fraction of all proposed draft tokens that were accepted."""
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed


class SpeculativeEngine:
    """Orchestrates speculative decoding for faster generation.

    Coordinates between a cheap draft model and expensive target model.
    The draft proposes K tokens autoregressively, then the target verifies
    all K in a single forward pass. Adaptive speculation adjusts K based
    on observed acceptance rates.

    The output distribution is mathematically identical to sampling from
    the target model alone; the draft model only affects throughput.

    Usage:
        engine = SpeculativeEngine(target_model, draft_model, config)

        # Iterator interface (recommended for streaming)
        for step in engine.generate(input_ids, max_tokens=128):
            tokens = step.new_tokens[0, :step.num_new_tokens[0]]

        # Batch interface
        output_ids = engine.generate_all(input_ids, max_tokens=128)
    """

    def __init__(
        self,
        target_model,
        draft_model: DraftModel,
        config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        self.target = target_model
        self.draft = draft_model
        self.config = config or SpeculativeConfig()
        self.device = device or torch.device("cpu")

        # Adaptive state
        self._acceptance_history: list[float] = []
        self._current_num_spec = self.config.num_speculative_tokens
        self._stats = GenerationStats()

    @property
    def current_num_spec(self) -> int:
        """Current adaptive speculation length."""
        return self._current_num_spec

    @property
    def stats(self) -> GenerationStats:
        """Cumulative generation statistics for the current run."""
        return self._stats

    def reset(self) -> None:
        """Reset engine state for a new sequence."""
        self._acceptance_history.clear()
        self._current_num_spec = self.config.num_speculative_tokens
        self._stats = GenerationStats()
        self.draft.reset()

    def generate_step(
        self,
        input_ids: Tensor,
        target_cache: KVCache | None = None,
    ) -> StepResult:
        """Execute one step of speculative generation.

        1. Draft model proposes current_num_spec tokens.
        2. Target model evaluates all proposed positions in one forward pass.
        3. Verification accepts/rejects via rejection sampling.
        4. Adaptive mechanism adjusts speculation length for next step.

        The caller is responsible for advancing target_cache after this call
        by the number of new tokens produced (num_new_tokens).

        Args:
            input_ids: Current context, shape [batch, seq_len].
                For decode phase this is typically [batch, 1] with the last token.
            target_cache: Target model's KV cache. If None, no caching is used.

        Returns:
            StepResult with new tokens, counts, and diagnostics.
        """
        batch_size = input_ids.shape[0]
        num_spec = self._current_num_spec

        # 1. Draft proposes K tokens
        draft_out: DraftOutput = self.draft.speculate(
            input_ids, kv_cache=target_cache, num_tokens=num_spec
        )
        self._stats.total_draft_steps += 1

        # 2. Target evaluates: original context + draft tokens in one pass
        spec_input = torch.cat([input_ids, draft_out.tokens], dim=1)
        target_logits = self.target(spec_input, kv_cache=target_cache)
        self._stats.total_target_calls += 1
        # target_logits: [batch, input_len + num_spec, vocab]

        # Extract the K+1 logits needed for verification:
        # Position input_len-1 predicts the first draft token
        # Position input_len+K-1 predicts the bonus token after all accepted
        input_len = input_ids.shape[1]
        start = input_len - 1
        end = start + num_spec + 1
        verify_logits = target_logits[:, start:end, :]  # [batch, num_spec+1, vocab]

        # 3. Verify draft tokens against target
        result: VerifyResult = verify_speculative(
            draft_out.tokens,
            draft_out.probs,
            verify_logits,
            self.config.temperature,
            self.config.top_p,
        )

        # 4. Assemble output: accepted_tokens + next_token per batch element
        max_new = num_spec + 1
        new_tokens = torch.zeros(
            batch_size, max_new, dtype=draft_out.tokens.dtype, device=self.device
        )
        num_new_tokens = result.num_accepted + 1  # Each gets at least next_token

        for b in range(batch_size):
            n_acc = int(result.num_accepted[b].item())
            # Copy accepted tokens
            if n_acc > 0:
                new_tokens[b, :n_acc] = result.accepted_tokens[b, :n_acc]
            # Append next token (correction or bonus)
            new_tokens[b, n_acc] = result.next_token[b]

        # 5. Update cumulative stats
        total_acc = int(result.num_accepted.sum().item())
        total_new = int(num_new_tokens.sum().item())
        self._stats.total_accepted += total_acc
        self._stats.total_proposed += batch_size * num_spec
        self._stats.total_tokens += total_new

        # 6. Update adaptive speculation
        avg_accepted = float(result.num_accepted.float().mean().item())
        acceptance_rate = avg_accepted / num_spec if num_spec > 0 else 0.0
        self._update_num_spec(acceptance_rate)

        return StepResult(
            new_tokens=new_tokens,
            num_new_tokens=num_new_tokens,
            num_target_calls=1,
            num_draft_tokens=num_spec,
            num_accepted=result.num_accepted,
            acceptance_rate=acceptance_rate,
        )

    def generate(
        self,
        input_ids: Tensor,
        max_tokens: int = 256,
        target_cache: KVCache | None = None,
    ) -> Iterator[StepResult]:
        """Generate tokens via speculative decoding, yielding per-step results.

        Manages the generation loop including prefill, cache advancement,
        EOS detection, and draft model synchronization.

        Args:
            input_ids: Prompt token IDs, shape [batch, prompt_len].
            max_tokens: Maximum new tokens to generate.
            target_cache: Optional pre-created target KV cache.

        Yields:
            StepResult per generation step. Each contains 1 to K+1 new tokens.
        """
        self.reset()
        batch_size = input_ids.shape[0]

        if target_cache is None:
            target_cache = self.target.create_kv_cache()

        # Prefill: run prompt through target to populate cache
        prefill_logits = self.target(input_ids, kv_cache=target_cache)
        target_cache.advance(input_ids.shape[1])

        # Sample first token from prefill
        first_logits = prefill_logits[:, -1, :]  # [batch, vocab]
        first_token = _sample_token(first_logits, self.config.temperature, self.device)
        # [batch]

        # Yield first token as a degenerate StepResult
        first_step = StepResult(
            new_tokens=first_token.reshape(batch_size, 1),
            num_new_tokens=torch.ones(batch_size, dtype=torch.int32, device=self.device),
            num_target_calls=1,
            num_draft_tokens=0,
            num_accepted=torch.zeros(batch_size, dtype=torch.int32, device=self.device),
            acceptance_rate=0.0,
        )
        self._stats.total_target_calls += 1
        self._stats.total_tokens += batch_size
        yield first_step

        # Check EOS on first token
        if _contains_eos(first_token, self.config.eos_token_id):
            return

        target_cache.advance(1)
        current_input = first_token.reshape(batch_size, 1)
        total_generated = 1

        while total_generated < max_tokens:
            step = self.generate_step(current_input, target_cache=target_cache)

            # Check for EOS in produced tokens
            eos_pos = _find_eos(step.new_tokens, step.num_new_tokens, self.config.eos_token_id)
            if eos_pos is not None:
                # Truncate at EOS
                b, pos = eos_pos
                # Adjust num_new_tokens to stop at EOS (exclusive)
                step = StepResult(
                    new_tokens=step.new_tokens,
                    num_new_tokens=torch.minimum(
                        step.num_new_tokens,
                        torch.full((batch_size,), pos, dtype=torch.int32, device=self.device),
                    ),
                    num_target_calls=step.num_target_calls,
                    num_draft_tokens=step.num_draft_tokens,
                    num_accepted=step.num_accepted,
                    acceptance_rate=step.acceptance_rate,
                )
                yield step
                return

            yield step

            # Advance cache and prepare next input
            max_new = int(step.num_new_tokens.max().item())
            total_generated += max_new
            target_cache.advance(max_new)

            # Next input: last produced token per batch element
            last_tokens = torch.zeros(batch_size, 1, dtype=current_input.dtype, device=self.device)
            for b in range(batch_size):
                n_new = int(step.num_new_tokens[b].item())
                last_tokens[b, 0] = step.new_tokens[b, n_new - 1]
            current_input = last_tokens

            # Synchronize draft model with accepted tokens
            self._sync_draft(step)

    def generate_all(
        self,
        input_ids: Tensor,
        max_tokens: int = 256,
        target_cache: KVCache | None = None,
        streamer: Callable[[int], None] | None = None,
    ) -> Tensor:
        """Generate a complete sequence, returning prompt + generated tokens.

        Convenience wrapper around generate() that collects all tokens into
        a single array. Optionally calls a streamer callback per token.

        Args:
            input_ids: Prompt token IDs, [1, prompt_len].
            max_tokens: Maximum new tokens to generate.
            target_cache: Optional pre-created target KV cache.
            streamer: Optional callback invoked with each new token ID.

        Returns:
            Full sequence (prompt + generated) as [1, total_len].
        """
        all_tokens: list[int] = input_ids[0].tolist()

        for step in self.generate(input_ids, max_tokens=max_tokens, target_cache=target_cache):
            # Extract valid tokens from first batch element
            n_new = int(step.num_new_tokens[0].item())
            for j in range(n_new):
                tok = int(step.new_tokens[0, j].item())
                if tok == self.config.eos_token_id:
                    return torch.tensor([all_tokens], dtype=torch.long, device=self.device)
                all_tokens.append(tok)
                if streamer:
                    streamer(tok)

        return torch.tensor([all_tokens], dtype=torch.long, device=self.device)

    def _sync_draft(self, step: StepResult) -> None:
        """Synchronize draft model state after a verification step.

        For SmallModelDraft: feed accepted tokens into the draft cache.
        For NGramDraft: update n-gram statistics with produced tokens.
        """
        if isinstance(self.draft, SmallModelDraft):
            # Feed accepted tokens to keep draft cache in sync
            batch_size = step.new_tokens.shape[0]
            for b in range(batch_size):
                n_acc = int(step.num_accepted[b].item())
                if n_acc > 0:
                    accepted = step.new_tokens[b : b + 1, :n_acc]
                    self.draft.sync_after_accept(accepted)

        elif isinstance(self.draft, NGramDraft):
            # Update n-gram frequency tables with produced tokens
            batch_size = step.new_tokens.shape[0]
            for b in range(batch_size):
                n_new = int(step.num_new_tokens[b].item())
                if n_new > 0:
                    token_list = [int(step.new_tokens[b, j].item()) for j in range(n_new)]
                    self.draft.update_ngrams(token_list)

    def _update_num_spec(self, acceptance_rate: float) -> None:
        """Adapt speculation length based on windowed acceptance average.

        Conservative: decrease by 1 on sustained low acceptance,
        increase by 1 on sustained high acceptance.
        """
        self._acceptance_history.append(acceptance_rate)

        window = self._acceptance_history[-self.config.history_window :]
        avg_rate = sum(window) / len(window)

        if avg_rate < self.config.acceptance_threshold:
            self._current_num_spec = max(
                self.config.min_speculative_tokens,
                self._current_num_spec - 1,
            )
        elif avg_rate > self.config.increase_threshold:
            self._current_num_spec = min(
                self.config.num_speculative_tokens,
                self._current_num_spec + 1,
            )


def _sample_token(logits: Tensor, temperature: float, device: torch.device) -> Tensor:
    """Sample a token from logits with temperature.

    Args:
        logits: [batch, vocab] raw logits.
        temperature: Sampling temperature. <= 0 means greedy.
        device: Device for output tensor.

    Returns:
        [batch] sampled token IDs.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1).long()
    scaled = logits / max(temperature, 1e-8)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1).long()


def _contains_eos(tokens: Tensor, eos_id: int) -> bool:
    """Check if any element in a 1D token array equals eos_id."""
    return bool((tokens == eos_id).any().item())


def _find_eos(
    new_tokens: Tensor,
    num_new_tokens: Tensor,
    eos_id: int,
) -> tuple[int, int] | None:
    """Find first EOS position in a step's output tokens.

    Returns (batch_idx, position) of first EOS within valid token range,
    or None if no EOS found.
    """
    batch_size = new_tokens.shape[0]
    for b in range(batch_size):
        n_new = int(num_new_tokens[b].item())
        for j in range(n_new):
            if int(new_tokens[b, j].item()) == eos_id:
                return (b, j)
    return None
