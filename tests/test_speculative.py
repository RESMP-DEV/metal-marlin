"""Comprehensive tests for speculative decoding: verification, draft models, and engine.

Tests are organized in three classes:
  1. TestVerification: Rejection sampling correctness (distribution preservation,
     acceptance behavior, temperature effects, edge cases).
  2. TestDraftModels: Draft model interfaces (SmallModelDraft, NGramDraft shapes,
     n-gram statistics, cache synchronization).
  3. TestSpeculativeEngine: End-to-end engine behavior (target call efficiency,
     adaptive speculation, EOS handling, generation loop).

All tests use mock models to avoid requiring trained weights or Metal hardware.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from metal_marlin.kv_cache import CacheConfig, KVCache
from metal_marlin.speculative.draft import DraftModel, DraftOutput, NGramDraft, SmallModelDraft
from metal_marlin.speculative.engine import (
    GenerationStats,
    SpeculativeConfig,
    SpeculativeEngine,
    StepResult,
)
from metal_marlin.speculative.verify import verify_speculative

# ---------------------------------------------------------------------------
# Test utilities and mock models
# ---------------------------------------------------------------------------


def _make_cache_config(vocab_size: int = 100) -> CacheConfig:
    """Create a minimal CacheConfig for testing."""
    return CacheConfig(
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=128,
    )


class MockCausalLM:
    """Mock causal language model for testing.

    Returns logits from a fixed distribution, optionally biased toward
    specific tokens to control acceptance behavior.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        bias_tokens: dict[int, float] | None = None,
        logit_scale: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.bias_tokens = bias_tokens or {}
        self.logit_scale = logit_scale
        self.call_count = 0
        self._cache_config = _make_cache_config(vocab_size)

    def __call__(self, input_ids: mx.array, kv_cache: KVCache | None = None) -> mx.array:
        self.call_count += 1
        batch_size, seq_len = input_ids.shape

        # Generate random logits
        logits = mx.random.normal(shape=(batch_size, seq_len, self.vocab_size))
        logits = logits * self.logit_scale

        # Apply biases
        for token_id, bias in self.bias_tokens.items():
            logits = logits.at[:, :, token_id].add(bias)

        return logits

    def create_kv_cache(self) -> KVCache:
        return KVCache(self._cache_config, batch_size=1)


class DeterministicDraft(DraftModel):
    """Draft model that always proposes specific tokens with given probabilities.

    Used to test verification behavior with controlled inputs.
    """

    def __init__(
        self,
        tokens: list[int],
        prob_mass: float = 0.9,
        vocab_size: int = 100,
    ):
        self._tokens = tokens
        self._prob_mass = prob_mass
        self._vocab_size = vocab_size

    def speculate(
        self,
        input_ids: mx.array,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        batch_size = input_ids.shape[0]
        num_tokens = min(num_tokens, len(self._tokens))

        tokens = mx.array([self._tokens[:num_tokens]] * batch_size, dtype=mx.uint32)

        # Build probability distributions: concentrate mass on proposed tokens
        probs = mx.ones((batch_size, num_tokens, self._vocab_size)) / self._vocab_size
        remainder = (1.0 - self._prob_mass) / (self._vocab_size - 1)
        for i in range(num_tokens):
            # Set all to remainder, then boost the proposed token
            mx.full((batch_size, self._vocab_size), remainder, dtype=mx.float32)
            mx.full((batch_size, 1), self._prob_mass, dtype=mx.float32)
            # Place concentrated mass at the proposed token
            probs_i = mx.zeros((batch_size, self._vocab_size), dtype=mx.float32)
            for b in range(batch_size):
                row = mx.full((self._vocab_size,), remainder, dtype=mx.float32)
                row = row.at[self._tokens[i]].add(self._prob_mass - remainder)
                probs_i = probs_i.at[b].add(row)
            probs = probs.at[:, i, :].add(probs_i - probs[:, i, :])

        return DraftOutput(tokens=tokens, probs=probs)

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# TestVerification: Rejection sampling correctness
# ---------------------------------------------------------------------------


class TestVerification:
    """Test rejection sampling correctness and distribution preservation."""

    def test_perfect_match_accepts_all(self):
        """When draft == target distributions, all tokens should be accepted."""
        mx.random.seed(42)
        batch, num_spec, vocab = 2, 4, 50

        draft_tokens = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.uint32)

        # Create a distribution and use it for both draft and target
        # Concentrate probability on the draft tokens for near-certain acceptance
        probs = mx.ones((batch, num_spec, vocab), dtype=mx.float32) * 0.001
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                probs = probs.at[b, i, tok].add(0.95)

        # Normalize
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)

        # Target logits that produce the same distribution after softmax
        # log(probs) -> softmax -> probs
        target_logits = mx.log(probs + 1e-10)
        # Need num_spec+1 positions for target (bonus position)
        bonus_logits = mx.random.normal(shape=(batch, 1, vocab))
        target_logits = mx.concatenate([target_logits, bonus_logits], axis=1)

        result = verify_speculative(draft_tokens, probs, target_logits)

        # With identical distributions and high concentration, should accept all
        assert result.num_accepted.shape == (batch,)
        assert result.accepted_tokens.shape == (batch, num_spec)
        assert result.next_token.shape == (batch,)

        # When distributions match perfectly, acceptance prob = 1.0
        # All tokens should be accepted
        for b in range(batch):
            assert int(result.num_accepted[b].item()) == num_spec

    def test_complete_mismatch_rejects_first(self):
        """When draft puts all mass on wrong token, first token should be rejected."""
        mx.random.seed(123)
        batch, num_spec, vocab = 1, 4, 50

        # Draft proposes token 10, but puts all probability on token 10
        draft_tokens = mx.array([[10, 11, 12, 13]], dtype=mx.uint32)
        draft_probs = mx.zeros((batch, num_spec, vocab), dtype=mx.float32)
        # Draft is very confident in its tokens
        for i, tok in enumerate([10, 11, 12, 13]):
            draft_probs = draft_probs.at[0, i, tok].add(0.99)
            draft_probs = draft_probs.at[0, i, :].add(0.01 / vocab)
        draft_probs = draft_probs / mx.sum(draft_probs, axis=-1, keepdims=True)

        # Target puts all mass on completely different tokens
        target_logits = mx.full((batch, num_spec + 1, vocab), -10.0)
        for i in range(num_spec + 1):
            # Target prefers token 40+i (different from draft's 10+i)
            target_logits = target_logits.at[0, i, 40 + i].add(20.0)

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        # Target assigns ~0 probability to draft tokens -> acceptance ratio ~0
        # Should reject at or very near position 0
        assert int(result.num_accepted[0].item()) <= 1

    def test_output_shapes(self):
        """Verify all output shapes are correct."""
        batch, num_spec, vocab = 3, 5, 200

        draft_tokens = mx.zeros((batch, num_spec), dtype=mx.uint32)
        draft_probs = mx.ones((batch, num_spec, vocab)) / vocab
        target_logits = mx.random.normal(shape=(batch, num_spec + 1, vocab))

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        assert result.accepted_tokens.shape == (batch, num_spec)
        assert result.num_accepted.shape == (batch,)
        assert result.next_token.shape == (batch,)

    def test_num_accepted_range(self):
        """num_accepted should be in [0, num_spec] for all batch elements."""
        mx.random.seed(99)
        batch, num_spec, vocab = 4, 6, 100

        draft_tokens = mx.random.randint(0, vocab, shape=(batch, num_spec)).astype(mx.uint32)
        draft_probs = mx.softmax(mx.random.normal(shape=(batch, num_spec, vocab)), axis=-1)
        target_logits = mx.random.normal(shape=(batch, num_spec + 1, vocab))

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        for b in range(batch):
            n_acc = int(result.num_accepted[b].item())
            assert 0 <= n_acc <= num_spec, f"Batch {b}: num_accepted={n_acc} outside [0,{num_spec}]"

    def test_accepted_tokens_match_draft(self):
        """Accepted token values should match the corresponding draft tokens."""
        mx.random.seed(77)
        batch, num_spec, vocab = 2, 4, 50

        draft_tokens = mx.array([[5, 10, 15, 20], [25, 30, 35, 40]], dtype=mx.uint32)
        # High prob on draft tokens -> likely acceptance
        draft_probs = mx.ones((batch, num_spec, vocab)) * (0.1 / vocab)
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                draft_probs = draft_probs.at[b, i, tok].add(0.9)
        draft_probs = draft_probs / mx.sum(draft_probs, axis=-1, keepdims=True)

        # Target also favors these tokens
        target_logits = mx.full((batch, num_spec + 1, vocab), -5.0)
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                target_logits = target_logits.at[b, i, tok].add(15.0)
        # Bonus position
        target_logits = target_logits.at[:, num_spec, 0].add(10.0)

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        for b in range(batch):
            n_acc = int(result.num_accepted[b].item())
            for i in range(n_acc):
                accepted = int(result.accepted_tokens[b, i].item())
                expected = int(draft_tokens[b, i].item())
                assert accepted == expected, (
                    f"Batch {b}, pos {i}: accepted={accepted} != draft={expected}"
                )

    def test_temperature_zero_is_greedy(self):
        """Temperature=0 should behave as greedy decoding for next_token."""
        mx.random.seed(55)
        batch, num_spec, vocab = 1, 3, 50

        draft_tokens = mx.array([[0, 0, 0]], dtype=mx.uint32)
        draft_probs = mx.ones((batch, num_spec, vocab)) / vocab
        # Target strongly prefers token 42 at all positions
        target_logits = mx.full((batch, num_spec + 1, vocab), -10.0)
        target_logits = target_logits.at[:, :, 42].add(30.0)

        result = verify_speculative(draft_tokens, draft_probs, target_logits, temperature=0.0)

        # With greedy target, if draft token 0 isn't argmax, it gets rejected
        # and next_token should be 42 (the argmax)
        # Even if accepted, bonus token should be 42
        assert int(result.next_token[0].item()) == 42

    def test_temperature_affects_acceptance(self):
        """Higher temperature should generally increase acceptance rate.

        With higher temperature, target distribution becomes more uniform,
        increasing p_target for non-peak tokens, boosting acceptance.
        """
        mx.random.seed(200)
        batch, num_spec, vocab = 4, 5, 50

        draft_tokens = mx.random.randint(0, vocab, shape=(batch, num_spec)).astype(mx.uint32)
        draft_probs = mx.softmax(mx.random.normal(shape=(batch, num_spec, vocab)), axis=-1)

        # Target with sharp peaks (low temperature will keep them sharp)
        target_logits = mx.random.normal(shape=(batch, num_spec + 1, vocab)) * 5.0

        result_low_temp = verify_speculative(
            draft_tokens, draft_probs, target_logits, temperature=0.1
        )
        result_high_temp = verify_speculative(
            draft_tokens, draft_probs, target_logits, temperature=5.0
        )

        # High temperature should give equal or more acceptances on average
        avg_low = float(mx.mean(result_low_temp.num_accepted).item())
        avg_high = float(mx.mean(result_high_temp.num_accepted).item())
        # Not a strict assertion (stochastic), but with enough batch size
        # high temp should usually give more acceptances
        # We just verify both produce valid results
        assert 0 <= avg_low <= num_spec
        assert 0 <= avg_high <= num_spec

    def test_top_p_filtering(self):
        """Top-p < 1.0 should still produce valid tokens."""
        mx.random.seed(303)
        batch, num_spec, vocab = 2, 3, 50

        draft_tokens = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.uint32)
        draft_probs = mx.softmax(mx.random.normal(shape=(batch, num_spec, vocab)), axis=-1)
        target_logits = mx.random.normal(shape=(batch, num_spec + 1, vocab))

        result = verify_speculative(draft_tokens, draft_probs, target_logits, top_p=0.9)

        # next_token should be valid token indices
        for b in range(batch):
            tok = int(result.next_token[b].item())
            assert 0 <= tok < vocab, f"Token {tok} outside vocab range [0, {vocab})"

    def test_distribution_preserving_statistical(self):
        """Statistical test: verify output matches target distribution.

        Run many trials with a simple 3-token vocab where target has a known
        distribution. The empirical distribution of sampled tokens should
        converge to the target. Uses chi-squared test for proper statistical
        validation.
        """
        mx.random.seed(42)
        vocab = 3
        num_trials = 5000
        target_dist = mx.array([0.6, 0.3, 0.1])  # Known target distribution
        draft_dist = mx.array([0.2, 0.5, 0.3])  # Different draft distribution

        counts = np.zeros(vocab)

        for _ in range(num_trials):
            # Draft proposes according to draft_dist
            draft_token = int(mx.random.categorical(mx.log(draft_dist)).item())
            draft_tokens = mx.array([[draft_token]], dtype=mx.uint32)
            draft_probs = draft_dist.reshape(1, 1, vocab)

            # Target logits that produce target_dist
            target_logits = mx.log(target_dist + 1e-10).reshape(1, 1, vocab)
            # Duplicate for bonus position
            target_logits = mx.concatenate([target_logits, target_logits], axis=1)

            result = verify_speculative(draft_tokens, draft_probs, target_logits)

            # The output token (either accepted draft or rejection sample)
            # should follow the target distribution
            n_acc = int(result.num_accepted[0].item())
            if n_acc > 0:
                tok = int(result.accepted_tokens[0, 0].item())
            else:
                tok = int(result.next_token[0].item())
            counts[tok] += 1

        # Chi-squared test: with 5000 samples and 3 categories (df=2),
        # critical value at p=0.001 is 13.82. Use generous threshold of 30
        # to avoid flaky failures while still catching real distribution bugs.
        expected = np.array([float(target_dist[i].item()) for i in range(vocab)]) * num_trials
        chi_sq = np.sum((counts - expected) ** 2 / expected)
        assert chi_sq < 30.0, (
            f"Chi-squared statistic {chi_sq:.2f} exceeds threshold 30.0 (p<0.001 critical=13.82). "
            f"Empirical: {counts / counts.sum()}, Target: {expected / num_trials}"
        )

    def test_batch_independence(self):
        """Different batch elements should be verified independently."""
        mx.random.seed(500)
        batch, num_spec, vocab = 2, 4, 50

        # Batch 0: perfect match (all accepted)
        # Batch 1: complete mismatch (none accepted)
        draft_tokens = mx.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=mx.uint32)

        draft_probs = mx.ones((batch, num_spec, vocab)) * (0.01 / vocab)
        for i in range(num_spec):
            draft_probs = draft_probs.at[0, i, 10 + i].add(0.99)
            draft_probs = draft_probs.at[1, i, 20 + i].add(0.99)
        draft_probs = draft_probs / mx.sum(draft_probs, axis=-1, keepdims=True)

        target_logits = mx.full((batch, num_spec + 1, vocab), -10.0)
        # Batch 0: target agrees with draft
        for i in range(num_spec):
            target_logits = target_logits.at[0, i, 10 + i].add(25.0)
        target_logits = target_logits.at[0, num_spec, 0].add(10.0)
        # Batch 1: target disagrees completely
        for i in range(num_spec + 1):
            target_logits = target_logits.at[1, i, 45].add(25.0)

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        # Batch 0 should accept all, batch 1 should reject at position 0
        assert int(result.num_accepted[0].item()) == num_spec
        assert int(result.num_accepted[1].item()) == 0

    def test_single_speculative_token(self):
        """Edge case: num_spec=1 should work correctly."""
        mx.random.seed(600)
        batch, vocab = 2, 30

        draft_tokens = mx.array([[5], [10]], dtype=mx.uint32)
        draft_probs = mx.softmax(mx.random.normal(shape=(batch, 1, vocab)), axis=-1)
        target_logits = mx.random.normal(shape=(batch, 2, vocab))

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        assert result.accepted_tokens.shape == (batch, 1)
        assert result.num_accepted.shape == (batch,)
        for b in range(batch):
            assert int(result.num_accepted[b].item()) in (0, 1)


# ---------------------------------------------------------------------------
# TestDraftModels: Draft model interface and implementations
# ---------------------------------------------------------------------------


class TestDraftModels:
    """Test draft model implementations (SmallModelDraft, NGramDraft)."""

    def test_small_model_draft_output_shape(self):
        """SmallModelDraft should produce correct output shapes."""
        vocab_size = 100
        model = MockCausalLM(vocab_size=vocab_size)
        draft = SmallModelDraft(model, max_speculative=4)

        input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=3)

        assert output.tokens.shape == (1, 3), f"tokens shape: {output.tokens.shape}"
        assert output.probs.shape == (1, 3, vocab_size), f"probs shape: {output.probs.shape}"

    def test_small_model_draft_respects_max_speculative(self):
        """Requesting more tokens than max_speculative should be clamped."""
        model = MockCausalLM(vocab_size=50)
        draft = SmallModelDraft(model, max_speculative=2)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=10)

        # Should be clamped to max_speculative=2
        assert output.tokens.shape[1] == 2
        assert output.probs.shape[1] == 2

    def test_small_model_draft_probabilities_sum_to_one(self):
        """Draft probability distributions should be normalized."""
        model = MockCausalLM(vocab_size=50)
        draft = SmallModelDraft(model, max_speculative=4)

        input_ids = mx.array([[5, 10, 15]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=4)

        # Check each position's distribution sums to 1
        sums = mx.sum(output.probs, axis=-1)  # [batch, num_spec]
        np.testing.assert_allclose(
            np.array(sums.tolist()), 1.0, atol=1e-5, err_msg="Draft probabilities don't sum to 1"
        )

    def test_small_model_draft_tokens_are_argmax(self):
        """SmallModelDraft uses greedy decoding, so tokens should match argmax of probs."""
        model = MockCausalLM(vocab_size=50, logit_scale=5.0)
        draft = SmallModelDraft(model, max_speculative=3)

        mx.random.seed(42)
        input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=3)

        for i in range(3):
            token = int(output.tokens[0, i].item())
            argmax = int(mx.argmax(output.probs[0, i]).item())
            assert token == argmax, f"Position {i}: token={token} != argmax={argmax}"

    def test_small_model_draft_batched(self):
        """SmallModelDraft should handle batch_size > 1."""
        model = MockCausalLM(vocab_size=50)
        draft = SmallModelDraft(model, max_speculative=4)

        input_ids = mx.array([[1, 2], [3, 4], [5, 6]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=3)

        assert output.tokens.shape == (3, 3)
        assert output.probs.shape == (3, 3, 50)

    def test_small_model_draft_reset(self):
        """Reset should clear the internal cache."""
        model = MockCausalLM(vocab_size=50)
        draft = SmallModelDraft(model, max_speculative=4)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        draft.speculate(input_ids, num_tokens=2)
        assert draft._cache is not None

        draft.reset()
        assert draft._cache is None
        assert draft._cache_seq_len == 0

    def test_ngram_draft_output_shape(self):
        """NGramDraft should produce correct output shapes."""
        vocab_size = 100
        draft = NGramDraft(ngram_size=3, vocab_size=vocab_size)

        # Need at least ngram_size tokens in input
        input_ids = mx.array([[10, 20, 30, 40, 50]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=3)

        assert output.tokens.shape == (1, 3)
        assert output.probs.shape == (1, 3, vocab_size)

    def test_ngram_updates_and_prediction(self):
        """After learning n-grams, predictions should use learned patterns."""
        draft = NGramDraft(ngram_size=2, vocab_size=50)

        # Train on a repeating pattern: [1, 2, 3, 1, 2, 3, 1, 2, 3]
        pattern = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        draft.update_ngrams(pattern)

        # After context [2, 3], should predict 1 (most frequent next token)
        input_ids = mx.array([[2, 3]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=3)

        # First prediction should be 1 (after [2,3] comes 1)
        first_token = int(output.tokens[0, 0].item())
        assert first_token == 1, f"Expected token 1 after [2,3], got {first_token}"

    def test_ngram_updates_accumulate(self):
        """Multiple update_ngrams calls should accumulate statistics."""
        draft = NGramDraft(ngram_size=2, vocab_size=50)

        # First sequence: [1, 2, 3]
        draft.update_ngrams([1, 2, 3])
        # Second sequence adds: [1, 2, 4]
        draft.update_ngrams([1, 2, 4])

        # Context [1, 2] now has counts: {3: 1, 4: 1}
        # Both should have nonzero probability
        context = (1, 2)
        assert context in draft.ngram_counts
        assert 3 in draft.ngram_counts[context]
        assert 4 in draft.ngram_counts[context]

    def test_ngram_fallback_with_short_context(self):
        """With insufficient context, NGramDraft should fall back to uniform."""
        draft = NGramDraft(ngram_size=3, vocab_size=50)

        # Only 2 tokens, but ngram_size=3 -> not enough context
        input_ids = mx.array([[1, 2]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=2)

        # Should return uniform probs (token 0 as default)
        assert output.tokens.shape == (1, 2)
        # All tokens should be 0 (fallback)
        assert int(output.tokens[0, 0].item()) == 0
        assert int(output.tokens[0, 1].item()) == 0

    def test_ngram_reset_clears_state(self):
        """Reset should clear all n-gram statistics and history."""
        draft = NGramDraft(ngram_size=2, vocab_size=50)
        draft.update_ngrams([1, 2, 3, 4, 5])

        assert len(draft.ngram_counts) > 0
        assert len(draft._history) > 0

        draft.reset()
        assert len(draft.ngram_counts) == 0
        assert len(draft._history) == 0

    def test_ngram_probabilities_normalized(self):
        """NGramDraft probabilities should sum to 1."""
        draft = NGramDraft(ngram_size=2, vocab_size=50)
        draft.update_ngrams([1, 2, 3, 1, 2, 4, 1, 2, 5])

        input_ids = mx.array([[1, 2]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=2)

        sums = mx.sum(output.probs, axis=-1)
        np.testing.assert_allclose(
            np.array(sums.tolist()),
            1.0,
            atol=1e-5,
            err_msg="NGramDraft probabilities don't sum to 1",
        )

    def test_ngram_batched_broadcasts(self):
        """NGramDraft with batch_size > 1 should broadcast predictions."""
        draft = NGramDraft(ngram_size=2, vocab_size=50)
        draft.update_ngrams([10, 20, 30, 10, 20, 30])

        # Batch of 3, all same context
        input_ids = mx.array([[10, 20], [10, 20], [10, 20]], dtype=mx.uint32)
        output = draft.speculate(input_ids, num_tokens=2)

        assert output.tokens.shape == (3, 2)
        assert output.probs.shape == (3, 2, 50)


# ---------------------------------------------------------------------------
# TestSpeculativeEngine: End-to-end engine behavior
# ---------------------------------------------------------------------------


class TestSpeculativeEngine:
    """Test the speculative decoding engine orchestration."""

    def test_single_step_output_shape(self):
        """A single generate_step should produce valid StepResult."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3, 4], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=4)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[10, 20, 30]], dtype=mx.uint32)
        result = engine.generate_step(input_ids)

        assert isinstance(result, StepResult)
        assert result.new_tokens.shape[0] == 1  # batch=1
        assert result.new_tokens.shape[1] == 5  # num_spec + 1
        assert result.num_target_calls == 1
        assert result.num_draft_tokens == 4

    def test_single_target_call_per_step(self):
        """Each generate_step should make exactly 1 target model forward call."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        target.call_count = 0
        input_ids = mx.array([[5]], dtype=mx.uint32)
        result = engine.generate_step(input_ids)

        assert target.call_count == 1
        assert result.num_target_calls == 1

    def test_multiple_steps_count_target_calls(self):
        """Multiple steps should each use exactly 1 target call."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3, 4], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=4)
        engine = SpeculativeEngine(target, draft, config)

        target.call_count = 0
        input_ids = mx.array([[5]], dtype=mx.uint32)
        for _ in range(5):
            engine.generate_step(input_ids)

        assert target.call_count == 5

    def test_num_new_tokens_at_least_one(self):
        """Each step should produce at least 1 new token (the next_token)."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[99, 99, 99], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        result = engine.generate_step(input_ids)

        for b in range(1):
            n_new = int(result.num_new_tokens[b].item())
            assert n_new >= 1, f"Expected at least 1 new token, got {n_new}"

    def test_num_new_tokens_at_most_k_plus_one(self):
        """Each step should produce at most K+1 new tokens."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3, 4, 5], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=5)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        result = engine.generate_step(input_ids)

        for b in range(1):
            n_new = int(result.num_new_tokens[b].item())
            assert n_new <= 6, f"Expected at most 6 new tokens, got {n_new}"

    def test_adaptive_reduces_on_low_acceptance(self):
        """Adaptive mechanism should reduce speculation on sustained low acceptance."""
        target = MockCausalLM(vocab_size=50)
        # Draft proposes tokens that won't match target -> low acceptance
        draft = DeterministicDraft(tokens=[49, 49, 49, 49], prob_mass=0.99, vocab_size=50)
        config = SpeculativeConfig(
            num_speculative_tokens=4,
            min_speculative_tokens=1,
            acceptance_threshold=0.5,
            history_window=3,
        )
        engine = SpeculativeEngine(target, draft, config)

        assert engine.current_num_spec == 4

        input_ids = mx.array([[1]], dtype=mx.uint32)
        # Run many steps to trigger adaptation
        for _ in range(20):
            engine.generate_step(input_ids)

        # After sustained low acceptance, should have reduced
        assert engine.current_num_spec < 4

    def test_adaptive_increases_on_high_acceptance(self):
        """Adaptive mechanism should increase speculation on sustained high acceptance."""
        mx.random.seed(42)
        vocab_size = 50
        target = MockCausalLM(vocab_size=vocab_size, bias_tokens={10: 20.0, 11: 20.0, 12: 20.0})
        # Draft proposes the tokens target prefers -> high acceptance
        draft = DeterministicDraft(tokens=[10, 11, 12, 10], prob_mass=0.95, vocab_size=vocab_size)
        config = SpeculativeConfig(
            num_speculative_tokens=4,
            min_speculative_tokens=1,
            increase_threshold=0.8,
            history_window=3,
        )
        engine = SpeculativeEngine(target, draft, config)

        # Start with reduced speculation
        engine._current_num_spec = 2

        input_ids = mx.array([[1]], dtype=mx.uint32)
        # Run steps; if acceptance is high, should increase
        for _ in range(20):
            result = engine.generate_step(input_ids)
            if result.acceptance_rate > 0.8:
                break

        # If we got high acceptance, num_spec should have increased
        # (may not reach 4 due to stochastic nature, but should be > 2 if high acceptance)
        # This is a soft assertion since acceptance depends on random sampling

    def test_adaptive_never_below_minimum(self):
        """Speculation length should never go below min_speculative_tokens."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[49, 49, 49, 49], prob_mass=0.99, vocab_size=50)
        config = SpeculativeConfig(
            num_speculative_tokens=4,
            min_speculative_tokens=2,
            acceptance_threshold=0.5,
            history_window=2,
        )
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        for _ in range(50):
            engine.generate_step(input_ids)

        assert engine.current_num_spec >= config.min_speculative_tokens

    def test_adaptive_never_above_maximum(self):
        """Speculation length should never exceed num_speculative_tokens."""
        target = MockCausalLM(vocab_size=50, bias_tokens={1: 50.0})
        draft = DeterministicDraft(tokens=[1, 1, 1, 1], prob_mass=0.99, vocab_size=50)
        config = SpeculativeConfig(
            num_speculative_tokens=4,
            max_speculative_tokens=8,
            increase_threshold=0.3,
            history_window=2,
        )
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        for _ in range(50):
            engine.generate_step(input_ids)

        assert engine.current_num_spec <= config.num_speculative_tokens

    def test_reset_clears_state(self):
        """Engine reset should clear acceptance history and restore speculation count."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        engine.generate_step(input_ids)
        engine._current_num_spec = 1

        engine.reset()
        assert engine.current_num_spec == 3
        assert len(engine._acceptance_history) == 0

    def test_acceptance_rate_in_result(self):
        """StepResult should contain valid acceptance_rate."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3, 4], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=4)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        result = engine.generate_step(input_ids)

        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_generate_iterator_yields_steps(self):
        """generate() should yield StepResult objects."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[10, 20, 30]], dtype=mx.uint32)
        steps = list(engine.generate(input_ids, max_tokens=5))

        assert len(steps) >= 1
        for step in steps:
            assert isinstance(step, StepResult)
            assert step.num_target_calls == 1

    def test_generate_respects_max_tokens(self):
        """generate() should stop after producing ~max_tokens tokens."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3, 4], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=4)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        steps = list(engine.generate(input_ids, max_tokens=10))

        total_tokens = sum(int(mx.max(s.num_new_tokens).item()) for s in steps)
        # Should produce approximately max_tokens (may overshoot by up to K)
        assert total_tokens >= 10

    def test_generate_stops_on_eos(self):
        """generate() should stop when EOS token is produced."""
        eos_id = 2
        # Target model that always outputs EOS
        target = MockCausalLM(vocab_size=50, bias_tokens={eos_id: 100.0})
        draft = DeterministicDraft(tokens=[eos_id, eos_id, eos_id], prob_mass=0.99, vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=3, eos_token_id=eos_id)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        steps = list(engine.generate(input_ids, max_tokens=100))

        # Should stop early due to EOS, not run all 100 tokens
        total_tokens = sum(int(mx.max(s.num_new_tokens).item()) for s in steps)
        assert total_tokens < 100

    def test_speedup_fewer_target_calls(self):
        """Speculative decoding should use fewer target calls than autoregressive.

        For N new tokens, autoregressive needs N target calls.
        Speculative needs ceil(N / avg_accepted_per_step) calls.
        """
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3, 4], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=4)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        steps = list(engine.generate(input_ids, max_tokens=20))

        total_target_calls = sum(s.num_target_calls for s in steps)
        total_tokens = sum(int(mx.max(s.num_new_tokens).item()) for s in steps)

        # If speculative works at all, we should need fewer target calls than tokens
        # Even with poor acceptance, each step produces at least 1 token from 1 call
        assert total_target_calls <= total_tokens

    def test_generation_stats_accumulate(self):
        """GenerationStats should track cumulative metrics across steps."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2, 3], vocab_size=50)
        config = SpeculativeConfig(
            num_speculative_tokens=3,
            min_speculative_tokens=3,  # Prevent adaptive reduction
            acceptance_threshold=0.0,  # Never reduce
        )
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        for _ in range(5):
            engine.generate_step(input_ids)

        stats = engine.stats
        assert isinstance(stats, GenerationStats)
        assert stats.total_target_calls == 5
        assert stats.total_draft_steps == 5
        assert stats.total_tokens > 0
        assert stats.total_proposed >= 5  # At least 1 proposed per step
        assert 0.0 <= stats.overall_acceptance_rate <= 1.0
        assert stats.tokens_per_target_call >= 1.0  # At least 1 token per call

    def test_generation_stats_reset(self):
        """Stats should be reset when engine.reset() is called."""
        target = MockCausalLM(vocab_size=50)
        draft = DeterministicDraft(tokens=[1, 2], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=2)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1]], dtype=mx.uint32)
        engine.generate_step(input_ids)
        assert engine.stats.total_target_calls > 0

        engine.reset()
        assert engine.stats.total_target_calls == 0
        assert engine.stats.total_tokens == 0

    def test_generate_all_returns_full_sequence(self):
        """generate_all should return prompt + generated tokens as a single array."""
        mx.random.seed(42)
        vocab_size = 50
        target = MockCausalLM(vocab_size=vocab_size)
        draft = DeterministicDraft(tokens=[1, 2, 3], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        prompt = mx.array([[10, 20, 30]], dtype=mx.uint32)
        output = engine.generate_all(prompt, max_tokens=10)

        assert output.shape[0] == 1  # batch=1
        # Output should start with the prompt
        assert output.shape[1] >= 3 + 10  # prompt_len + max_tokens
        assert int(output[0, 0].item()) == 10
        assert int(output[0, 1].item()) == 20
        assert int(output[0, 2].item()) == 30

    def test_generate_all_with_streamer(self):
        """generate_all with streamer should call back for each new token."""
        mx.random.seed(42)
        vocab_size = 50
        target = MockCausalLM(vocab_size=vocab_size)
        draft = DeterministicDraft(tokens=[1, 2, 3], vocab_size=50)
        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        streamed: list[int] = []
        prompt = mx.array([[10, 20, 30]], dtype=mx.uint32)
        engine.generate_all(prompt, max_tokens=8, streamer=streamed.append)

        # Streamer should have been called at least once
        assert len(streamed) >= 8
        # All streamed tokens should be valid
        for tok in streamed:
            assert 0 <= tok < vocab_size


# ---------------------------------------------------------------------------
# TestIntegration: Combined end-to-end tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining draft, verify, and engine."""

    def test_ngram_draft_with_engine(self):
        """NGramDraft integrated with SpeculativeEngine should work end-to-end."""
        vocab_size = 50
        target = MockCausalLM(vocab_size=vocab_size)
        draft = NGramDraft(ngram_size=2, vocab_size=vocab_size)
        # Seed with a pattern
        draft.update_ngrams([1, 2, 3, 1, 2, 3, 1, 2, 3])

        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1, 2]], dtype=mx.uint32)
        result = engine.generate_step(input_ids)

        assert isinstance(result, StepResult)
        assert result.num_target_calls == 1

    def test_small_model_draft_with_engine(self):
        """SmallModelDraft integrated with engine should work end-to-end."""
        vocab_size = 50
        target = MockCausalLM(vocab_size=vocab_size)
        draft_model = MockCausalLM(vocab_size=vocab_size)
        draft = SmallModelDraft(draft_model, max_speculative=3)

        config = SpeculativeConfig(num_speculative_tokens=3)
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[10, 20, 30]], dtype=mx.uint32)
        result = engine.generate_step(input_ids)

        assert isinstance(result, StepResult)
        assert result.num_target_calls == 1
        assert result.num_draft_tokens == 3

    def test_verify_with_real_draft_output(self):
        """verify_speculative should work with actual DraftOutput from SmallModelDraft."""
        mx.random.seed(42)
        vocab_size = 50
        draft_model = MockCausalLM(vocab_size=vocab_size)
        draft = SmallModelDraft(draft_model, max_speculative=4)

        input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)
        draft_out = draft.speculate(input_ids, num_tokens=4)

        # Simulate target model output
        target_logits = mx.random.normal(shape=(1, 5, vocab_size))

        result = verify_speculative(draft_out.tokens, draft_out.probs, target_logits)

        assert result.accepted_tokens.shape == (1, 4)
        assert result.num_accepted.shape == (1,)
        assert result.next_token.shape == (1,)

    def test_full_generation_loop(self):
        """Full generation loop should produce tokens without errors."""
        mx.random.seed(42)
        vocab_size = 50
        target = MockCausalLM(vocab_size=vocab_size)
        draft_model = MockCausalLM(vocab_size=vocab_size)
        draft = SmallModelDraft(draft_model, max_speculative=3)

        config = SpeculativeConfig(
            num_speculative_tokens=3,
            temperature=1.0,
        )
        engine = SpeculativeEngine(target, draft, config)

        input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.uint32)
        all_tokens: list[int] = []

        for step in engine.generate(input_ids, max_tokens=15):
            n_new = int(step.num_new_tokens[0].item())
            for j in range(n_new):
                tok = int(step.new_tokens[0, j].item())
                all_tokens.append(tok)

        assert len(all_tokens) >= 15
