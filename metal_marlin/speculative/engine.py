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
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from ..kv_cache import KVCache
from .draft import DraftModel, DraftOutput, EagleHead, NGramDraft, SmallModelDraft
from .token_acceptance import TokenAcceptanceTracker, create_acceptance_report
from .verify import VerifyResult, verify_speculative

if TYPE_CHECKING:
    from typing import Any


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
        draft_type: Type of draft model to use:
            - "small_model": Use a separate smaller LM for drafting.
            - "ngram": Use online n-gram statistics (zero extra params).
            - "eagle": Use EAGLE v3 tree attention for high-acceptance drafting.
        num_speculative_tokens: Initial (and target) number of tokens to speculate.
        min_speculative_tokens: Adaptive floor.
        max_speculative_tokens: Adaptive ceiling (capped at num_speculative_tokens).
        temperature: Sampling temperature for target model verification.
        top_p: Nucleus sampling threshold.
        acceptance_threshold: Reduce K when windowed rate drops below this.
        increase_threshold: Increase K when windowed rate exceeds this.
        history_window: Number of recent steps for adaptive averaging.
        eos_token_id: End-of-sequence token ID.
        eagle_tree_width: Number of candidates to explore per tree node (Eagle only).
        eagle_max_depth: Maximum depth of EAGLE speculation tree.
        eagle_adaptive: Enable adaptive tree width based on acceptance (Eagle only).
        adaptive_depth: Enable adaptive speculation depth based on running acceptance.
        adaptive_depth_alpha: EMA smoothing factor for adaptive depth (0-1).
    """

    draft_type: Literal["small_model", "ngram", "eagle"] = "eagle"
    num_speculative_tokens: int = 4
    min_speculative_tokens: int = 1
    max_speculative_tokens: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    acceptance_threshold: float = 0.4
    increase_threshold: float = 0.8
    history_window: int = 10
    eos_token_id: int = 2
    # Eagle-specific settings
    eagle_tree_width: int = 3
    eagle_max_depth: int = 5
    eagle_adaptive: bool = True
    # Adaptive depth settings
    adaptive_depth: bool = True
    adaptive_depth_alpha: float = 0.2


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
    acceptance_result: AcceptanceResult | None = None


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

    Supports three draft model types:
    - small_model: A separate smaller LM for drafting
    - ngram: Online n-gram statistics (zero additional parameters)
    - eagle: EAGLE v3 tree attention for high-acceptance rates

    Usage:
        # Using Eagle (default)
        config = SpeculativeConfig(draft_type="eagle", eagle_tree_width=3)
        engine = SpeculativeEngine.from_config(target_model, config)

        # Or with explicit draft model
        engine = SpeculativeEngine(target_model, draft_model, config)

        # Iterator interface (recommended for streaming)
        for step in engine.generate(input_ids, max_tokens=128):
            tokens = step.new_tokens[0, :step.num_new_tokens[0]]

        # Batch interface
        output_ids = engine.generate_all(input_ids, max_tokens=128)
    """

    def __init__(
        self,
        target_model: Any,
        draft_model: DraftModel | None = None,
        config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        self.target = target_model
        self.config = config or SpeculativeConfig()
        self.device = device or torch.device("cpu")

        # Create draft model if not provided
        if draft_model is None:
            draft_model = self._create_draft_model()
        self.draft = draft_model

        # Adaptive state
        self._current_num_spec = self.config.num_speculative_tokens
        self._stats = GenerationStats()

        # Token acceptance tracker for detailed acceptance management
        self._acceptance_tracker = TokenAcceptanceTracker(track_history=True)

        # Exponential moving average for adaptive depth
        self._ema_acceptance: float = 0.5  # Start at 50% estimate

    def _create_draft_model(self) -> DraftModel:
        """Create draft model based on config.draft_type."""
        if self.config.draft_type == "eagle":
            return EagleHead.from_target_model(
                self.target,
                tree_width=self.config.eagle_tree_width,
                max_depth=self.config.eagle_max_depth,
                adaptive_width=self.config.eagle_adaptive,
                device=self.device,
            )
        elif self.config.draft_type == "ngram":
            # Get vocab size from target model if available
            vocab_size = 32000
            config = getattr(self.target, "config", None)
            if config is not None:
                vocab_size = getattr(config, "vocab_size", 32000)
            return NGramDraft(ngram_size=3, vocab_size=vocab_size, device=self.device)
        elif self.config.draft_type == "small_model":
            raise ValueError(
                "small_model draft_type requires passing a draft model to __init__(). "
                "Use SpeculativeEngine(target, SmallModelDraft(small_model), config)."
            )
        else:
            raise ValueError(f"Unknown draft_type: {self.config.draft_type}")

    @classmethod
    def from_config(
        cls,
        target_model: Any,
        config: SpeculativeConfig,
        device: torch.device | None = None,
    ) -> SpeculativeEngine:
        """Create engine with draft model auto-created from config.

        This is the recommended way to create an engine when using Eagle
        or n-gram drafting, as it handles draft model creation automatically.

        Args:
            target_model: The target language model.
            config: Speculative decoding configuration.
            device: Device for tensors.

        Returns:
            Configured SpeculativeEngine instance.
        """
        return cls(target_model=target_model, config=config, device=device)

    @property
    def current_num_spec(self) -> int:
        """Current adaptive speculation length."""
        return self._current_num_spec

    @property
    def stats(self) -> GenerationStats:
        """Cumulative generation statistics for the current run."""
        return self._stats

    @property
    def acceptance_stats(self) -> dict:
        """Detailed token acceptance statistics.

        Returns:
            Dictionary with acceptance metrics including:
            - total_accepted: Total draft tokens accepted
            - total_rejected: Total draft tokens rejected
            - total_proposed: Total draft tokens proposed
            - overall_acceptance_rate: Fraction of tokens accepted
            - average_acceptance_per_step: Average tokens accepted per step
            - recent_acceptance_rate: Acceptance rate over recent steps
            - acceptance_trend: "improving", "declining", or "stable"
        """
        return create_acceptance_report(self._acceptance_tracker)

    def get_accepted_sequence(self, input_ids: torch.Tensor, step_result: StepResult) -> torch.Tensor:
        """Get the full sequence including accepted draft tokens.

        Uses the TokenAcceptanceTracker to properly assemble the sequence
        with accepted tokens from the draft model.

        Args:
            input_ids: Original input token IDs, shape [batch, seq_len].
            step_result: Result from generate_step.

        Returns:
            Full sequence with accepted tokens appended, shape [batch, seq_len + num_new].
        """
        # If we have detailed acceptance result, use tracker to assemble
        if step_result.acceptance_result is not None:
            return self._acceptance_tracker.assemble_sequence(
                input_ids, step_result.acceptance_result, include_next=True
            )

        # Fallback for when acceptance_result is not available (e.g. initial step)
        # or if StepResult was created manually without it.
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Build acceptance result from step result
        num_accepted = step_result.num_accepted

        # Extract accepted tokens from step result
        max_accepted = int(num_accepted.max().item())
        if max_accepted > 0:
            # Reconstruct accepted tokens from step result
            accepted_mask = (
                torch.arange(step_result.new_tokens.shape[1], device=device)
                .unsqueeze(0)
                .expand(batch_size, -1) < num_accepted.unsqueeze(-1)
            )
            accepted_tokens = torch.where(
                accepted_mask,
                step_result.new_tokens,
                torch.zeros_like(step_result.new_tokens),
            )[:, :max_accepted]
        else:
            accepted_tokens = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        # Assemble sequence: input + accepted tokens + next token
        output_parts: list[torch.Tensor] = [input_ids]

        for b in range(batch_size):
            n_acc = int(num_accepted[b].item())
            if n_acc > 0:
                output_parts.append(accepted_tokens[b:b + 1, :n_acc])

        # Add next token (the last token in new_tokens for each batch element)
        for b in range(batch_size):
            n_new = int(step_result.num_new_tokens[b].item())
            if n_new > 0:
                next_tok = step_result.new_tokens[b, n_new - 1:n_new]
                output_parts.append(next_tok.unsqueeze(0))

        # Concatenate all parts per batch element
        result_parts: list[torch.Tensor] = []
        for b in range(batch_size):
            parts: list[torch.Tensor] = [input_ids[b:b + 1]]

            n_acc = int(num_accepted[b].item())
            if n_acc > 0:
                parts.append(accepted_tokens[b:b + 1, :n_acc])

            n_new = int(step_result.num_new_tokens[b].item())
            if n_new > 0:
                next_tok = step_result.new_tokens[b, n_new - 1:n_new]
                parts.append(next_tok.unsqueeze(0))

            result_parts.append(torch.cat(parts, dim=1))

        # Pad to same length and stack
        max_len = max(r.shape[1] for r in result_parts)
        padded_results = []
        for r in result_parts:
            if r.shape[1] < max_len:
                padding = torch.zeros(
                    r.shape[0], max_len - r.shape[1], dtype=r.dtype, device=r.device
                )
                r = torch.cat([r, padding], dim=1)
            padded_results.append(r)

        return torch.cat(padded_results, dim=0)

    def reset(self) -> None:
        """Reset engine state for a new sequence."""
        self._current_num_spec = self.config.num_speculative_tokens
        self._stats = GenerationStats()
        self._acceptance_tracker.reset()
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

        # 1. Draft model generation loop: proposes K tokens autoregressively
        # The draft.speculate() method runs an autoregressive generation loop
        # that produces K candidate tokens using the cheap draft model.
        # Each iteration: forward(current_token) -> logits -> softmax -> argmax -> next_token
        # The loop maintains its own KV cache and generates tokens one-by-one.
        # See draft.py:SmallModelDraft.speculate() for the generation loop implementation.
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

        # 4. Compute detailed acceptance result using TokenAcceptanceTracker
        acceptance_result = self._acceptance_tracker.compute_acceptance(
            draft_tokens=draft_out.tokens,
            num_accepted=result.num_accepted,
            next_token=result.next_token,
            num_speculative=num_spec,
        )

        # Update acceptance statistics
        self._acceptance_tracker.update_stats(acceptance_result, num_spec)

        # 5. Assemble output: accepted_tokens + next_token per batch element
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

        # 6. Update cumulative stats
        total_acc = int(result.num_accepted.sum().item())
        total_new = int(num_new_tokens.sum().item())
        self._stats.total_accepted += total_acc
        self._stats.total_proposed += batch_size * num_spec
        self._stats.total_tokens += total_new

        # 7. Update adaptive speculation
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
            acceptance_result=acceptance_result,
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
                        torch.full((batch_size,), pos + 1, dtype=torch.int32, device=self.device),
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
        For EagleHead: update adaptive width based on acceptance rate.
        """
        if isinstance(self.draft, SmallModelDraft):
            # Feed accepted tokens to keep draft cache in sync
            # Use tracker utility if acceptance result is available
            if step.acceptance_result is not None:
                sync_tokens = self._acceptance_tracker.get_draft_sync_tokens(step.acceptance_result)
                for b, tokens in enumerate(sync_tokens):
                    if tokens.shape[1] > 0:
                        self.draft.sync_after_accept(tokens)
            else:
                # Fallback manual extraction
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

        elif isinstance(self.draft, EagleHead):
            # Update Eagle's adaptive tree width
            self.draft.update_acceptance(step.acceptance_rate)

    def _update_num_spec(self, acceptance_rate: float) -> None:
        """Adapt speculation length based on acceptance rate.

        Two modes:
        1. Windowed average (default): Conservative step changes based on
           sustained low/high acceptance.
        2. Adaptive depth (enabled via config.adaptive_depth): Uses EMA
           to smoothly track optimal speculation depth based on acceptance.
        """
        # Note: self._acceptance_tracker tracks acceptance history internally

        effective_max = min(self.config.max_speculative_tokens, self.config.num_speculative_tokens)

        if self.config.adaptive_depth:
            # Exponential moving average for smooth adaptation
            alpha = self.config.adaptive_depth_alpha
            self._ema_acceptance = alpha * acceptance_rate + (1 - alpha) * self._ema_acceptance

            # Compute optimal depth from EMA acceptance rate
            # Higher acceptance -> more speculation is beneficial
            # Formula: optimal_k ≈ 1 / (1 - acceptance) for geometric distribution
            # We clamp this to reasonable bounds
            if self._ema_acceptance < 0.1:
                optimal_depth = self.config.min_speculative_tokens
            elif self._ema_acceptance > 0.95:
                optimal_depth = effective_max
            else:
                # Geometric series expected length: 1/(1-p) - 1 accepted before rejection
                # We use a slightly conservative estimate
                raw_optimal = min(1.0 / (1.0 - self._ema_acceptance), 10.0)
                optimal_depth = int(raw_optimal * 0.8)  # 80% of theoretical optimum

            # Gradually move toward optimal (1 step at a time for stability)
            if optimal_depth > self._current_num_spec:
                self._current_num_spec = min(
                    self._current_num_spec + 1,
                    effective_max,
                )
            elif optimal_depth < self._current_num_spec:
                self._current_num_spec = max(
                    self._current_num_spec - 1,
                    self.config.min_speculative_tokens,
                )
        else:
            # Use TokenAcceptanceTracker for windowed step adjustment
            self._current_num_spec = self._acceptance_tracker.compute_speculation_length_step(
                current_length=self._current_num_spec,
                min_length=self.config.min_speculative_tokens,
                max_length=effective_max,
                decrease_threshold=self.config.acceptance_threshold,
                increase_threshold=self.config.increase_threshold,
                window=self.config.history_window,
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
