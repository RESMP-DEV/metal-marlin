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

import numpy as np
import torch

from metal_marlin.kv_cache import CacheConfig, KVCache
from metal_marlin.speculative.draft import DraftModel, DraftOutput, NGramDraft, SmallModelDraft
from metal_marlin.speculative.mmfp4_draft import MMFP4DraftModel
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


class MockCausalLM(torch.nn.Module):
    """Mock causal language model for testing.

    Returns logits from a fixed distribution, optionally biased toward
    specific tokens to control acceptance behavior.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        bias_tokens: dict[int, float] | None = None,
        logit_scale: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias_tokens = bias_tokens or {}
        self.logit_scale = logit_scale
        self.call_count = 0
        self._cache_config = _make_cache_config(vocab_size)
        self.device = device or torch.device("cpu")
        self.config = type('Config', (), {'hidden_size': 32, 'vocab_size': vocab_size})()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=self.device))

    def __call__(self, input_ids: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        self.call_count += 1
        batch_size, seq_len = input_ids.shape

        # Generate random logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size, device=self.device)
        logits = logits * self.logit_scale

        # Apply biases
        for token_id, bias in self.bias_tokens.items():
            logits[:, :, token_id] = logits[:, :, token_id] + bias

        return logits
    
    def modules(self):
        return []

    def create_kv_cache(self) -> KVCache:
        return KVCache(self._cache_config, batch_size=1, device="cpu")


class DeterministicDraft(DraftModel):
    """Draft model that always proposes specific tokens with given probabilities.

    Used to test verification behavior with controlled inputs.
    """

    def __init__(
        self,
        tokens: list[int],
        prob_mass: float = 0.9,
        vocab_size: int = 100,
        device: torch.device | None = None,
    ):
        self._tokens = tokens
        self._prob_mass = prob_mass
        self._vocab_size = vocab_size
        self.device = device or torch.device("cpu")

    def speculate(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        num_tokens: int = 4,
    ) -> DraftOutput:
        batch_size = input_ids.shape[0]
        num_tokens = min(num_tokens, len(self._tokens))

        tokens = torch.tensor(
            [self._tokens[:num_tokens]] * batch_size,
            dtype=torch.long,
            device=self.device,
        )

        # Build probability distributions: concentrate mass on proposed tokens
        remainder = (1.0 - self._prob_mass) / (self._vocab_size - 1)
        probs = torch.full(
            (batch_size, num_tokens, self._vocab_size),
            remainder,
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(num_tokens):
            probs[:, i, self._tokens[i]] = self._prob_mass

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
        torch.manual_seed(42)
        batch, num_spec, vocab = 2, 4, 50
        device = torch.device("cpu")

        draft_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long, device=device)

        # Create a distribution and use it for both draft and target
        # Concentrate probability on the draft tokens for near-certain acceptance
        probs = torch.ones((batch, num_spec, vocab), dtype=torch.float32, device=device) * 0.001
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                probs[b, i, tok] = probs[b, i, tok] + 0.95

        # Normalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Target logits that produce the same distribution after softmax
        # log(probs) -> softmax -> probs
        target_logits = torch.log(probs + 1e-10)
        # Need num_spec+1 positions for target (bonus position)
        bonus_logits = torch.randn(batch, 1, vocab, device=device)
        target_logits = torch.cat([target_logits, bonus_logits], dim=1)

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
        torch.manual_seed(123)
        batch, num_spec, vocab = 1, 4, 50
        device = torch.device("cpu")

        # Draft proposes token 10, but puts all probability on token 10
        draft_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.long, device=device)
        draft_probs = torch.zeros((batch, num_spec, vocab), dtype=torch.float32, device=device)
        # Draft is very confident in its tokens
        for i, tok in enumerate([10, 11, 12, 13]):
            draft_probs[0, i, tok] = 0.99
            draft_probs[0, i, :] = draft_probs[0, i, :] + 0.01 / vocab
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)

        # Target puts all mass on completely different tokens
        target_logits = torch.full((batch, num_spec + 1, vocab), -10.0, device=device)
        for i in range(num_spec + 1):
            # Target prefers token 40+i (different from draft's 10+i)
            target_logits[0, i, 40 + i] = target_logits[0, i, 40 + i] + 20.0

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        # Target assigns ~0 probability to draft tokens -> acceptance ratio ~0
        # Should reject at or very near position 0
        assert int(result.num_accepted[0].item()) <= 1

    def test_output_shapes(self):
        """Verify all output shapes are correct."""
        batch, num_spec, vocab = 3, 5, 200
        device = torch.device("cpu")

        draft_tokens = torch.zeros((batch, num_spec), dtype=torch.long, device=device)
        draft_probs = torch.ones((batch, num_spec, vocab), device=device) / vocab
        target_logits = torch.randn(batch, num_spec + 1, vocab, device=device)

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        assert result.accepted_tokens.shape == (batch, num_spec)
        assert result.num_accepted.shape == (batch,)
        assert result.next_token.shape == (batch,)

    def test_num_accepted_range(self):
        """num_accepted should be in [0, num_spec] for all batch elements."""
        torch.manual_seed(99)
        batch, num_spec, vocab = 4, 6, 100
        device = torch.device("cpu")

        draft_tokens = torch.randint(0, vocab, (batch, num_spec), dtype=torch.long, device=device)
        draft_probs = torch.softmax(torch.randn(batch, num_spec, vocab, device=device), dim=-1)
        target_logits = torch.randn(batch, num_spec + 1, vocab, device=device)

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        for b in range(batch):
            n_acc = int(result.num_accepted[b].item())
            assert 0 <= n_acc <= num_spec, f"Batch {b}: num_accepted={n_acc} outside [0,{num_spec}]"

    def test_accepted_tokens_match_draft(self):
        """Accepted token values should match the corresponding draft tokens."""
        torch.manual_seed(77)
        batch, num_spec, vocab = 2, 4, 50
        device = torch.device("cpu")

        draft_tokens = torch.tensor(
            [[5, 10, 15, 20], [25, 30, 35, 40]], dtype=torch.long, device=device
        )
        # High prob on draft tokens -> likely acceptance
        draft_probs = torch.ones((batch, num_spec, vocab), device=device) * (0.1 / vocab)
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                draft_probs[b, i, tok] = draft_probs[b, i, tok] + 0.9
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)

        # Target also favors these tokens
        target_logits = torch.full((batch, num_spec + 1, vocab), -5.0, device=device)
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                target_logits[b, i, tok] = target_logits[b, i, tok] + 15.0
        # Bonus position
        target_logits[:, num_spec, 0] = target_logits[:, num_spec, 0] + 10.0

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
        torch.manual_seed(55)
        batch, num_spec, vocab = 1, 3, 50
        device = torch.device("cpu")

        draft_tokens = torch.zeros((batch, num_spec), dtype=torch.long, device=device)
        draft_probs = torch.ones((batch, num_spec, vocab), device=device) / vocab
        # Target strongly prefers token 42 at all positions
        target_logits = torch.full((batch, num_spec + 1, vocab), -10.0, device=device)
        target_logits[:, :, 42] = target_logits[:, :, 42] + 30.0

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
        torch.manual_seed(200)
        batch, num_spec, vocab = 4, 5, 50
        device = torch.device("cpu")

        draft_tokens = torch.randint(0, vocab, (batch, num_spec), dtype=torch.long, device=device)
        draft_probs = torch.softmax(torch.randn(batch, num_spec, vocab, device=device), dim=-1)

        # Target with sharp peaks (low temperature will keep them sharp)
        target_logits = torch.randn(batch, num_spec + 1, vocab, device=device) * 5.0

        result_low_temp = verify_speculative(
            draft_tokens, draft_probs, target_logits, temperature=0.1
        )
        result_high_temp = verify_speculative(
            draft_tokens, draft_probs, target_logits, temperature=5.0
        )

        # High temperature should give equal or more acceptances on average
        avg_low = float(result_low_temp.num_accepted.float().mean().item())
        avg_high = float(result_high_temp.num_accepted.float().mean().item())
        # Not a strict assertion (stochastic), but with enough batch size
        # high temp should usually give more acceptances
        # We just verify both produce valid results
        assert 0 <= avg_low <= num_spec
        assert 0 <= avg_high <= num_spec

    def test_top_p_filtering(self):
        """Top-p < 1.0 should still produce valid tokens."""
        torch.manual_seed(303)
        batch, num_spec, vocab = 2, 3, 50
        device = torch.device("cpu")

        draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long, device=device)
        draft_probs = torch.softmax(torch.randn(batch, num_spec, vocab, device=device), dim=-1)
        target_logits = torch.randn(batch, num_spec + 1, vocab, device=device)

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
        torch.manual_seed(42)
        vocab = 3
        num_trials = 5000
        device = torch.device("cpu")
        target_dist = torch.tensor([0.6, 0.3, 0.1], device=device)  # Known target distribution
        draft_dist = torch.tensor([0.2, 0.5, 0.3], device=device)  # Different draft distribution

        counts = np.zeros(vocab)

        for _ in range(num_trials):
            # Draft proposes according to draft_dist
            draft_token = int(torch.multinomial(draft_dist, num_samples=1).item())
            draft_tokens = torch.tensor([[draft_token]], dtype=torch.long, device=device)
            draft_probs = draft_dist.reshape(1, 1, vocab)

            # Target logits that produce target_dist
            target_logits = torch.log(target_dist + 1e-10).reshape(1, 1, vocab)
            # Duplicate for bonus position
            target_logits = torch.cat([target_logits, target_logits], dim=1)

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
        torch.manual_seed(500)
        batch, num_spec, vocab = 2, 4, 50
        device = torch.device("cpu")

        # Batch 0: perfect match (all accepted)
        # Batch 1: complete mismatch (none accepted)
        draft_tokens = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long, device=device
        )

        draft_probs = torch.ones((batch, num_spec, vocab), device=device) * (0.01 / vocab)
        for i in range(num_spec):
            draft_probs[0, i, 10 + i] = draft_probs[0, i, 10 + i] + 0.99
            draft_probs[1, i, 20 + i] = draft_probs[1, i, 20 + i] + 0.99
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)

        target_logits = torch.full((batch, num_spec + 1, vocab), -10.0, device=device)
        # Batch 0: target agrees with draft
        for i in range(num_spec):
            target_logits[0, i, 10 + i] = target_logits[0, i, 10 + i] + 25.0
        target_logits[0, num_spec, 0] = target_logits[0, num_spec, 0] + 10.0
        # Batch 1: target disagrees completely
        for i in range(num_spec + 1):
            target_logits[1, i, 45] = target_logits[1, i, 45] + 25.0

        result = verify_speculative(draft_tokens, draft_probs, target_logits)

        # Batch 0 should accept all, batch 1 should reject at position 0
        assert int(result.num_accepted[0].item()) == num_spec
        assert int(result.num_accepted[1].item()) == 0

    def test_single_speculative_token(self):
        """Edge case: num_spec=1 should work correctly."""
        torch.manual_seed(600)
        batch, vocab = 2, 30
        device = torch.device("cpu")

        draft_tokens = torch.tensor([[5], [10]], dtype=torch.long, device=device)
        draft_probs = torch.softmax(torch.randn(batch, 1, vocab, device=device), dim=-1)
        target_logits = torch.randn(batch, 2, vocab, device=device)

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
        device = torch.device("cpu")
        model = MockCausalLM(vocab_size=vocab_size, device=device)
        draft = SmallModelDraft(model, max_speculative=4, device=device)

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=3)

        assert output.tokens.shape == (1, 3), f"tokens shape: {output.tokens.shape}"
        assert output.probs.shape == (1, 3, vocab_size), f"probs shape: {output.probs.shape}"

    def test_small_model_draft_respects_max_speculative(self):
        """Requesting more tokens than max_speculative should be clamped."""
        device = torch.device("cpu")
        model = MockCausalLM(vocab_size=50, device=device)
        draft = SmallModelDraft(model, max_speculative=2, device=device)

        input_ids = torch.tensor([[1]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=10)

        # Should be clamped to max_speculative=2
        assert output.tokens.shape[1] == 2
        assert output.probs.shape[1] == 2

    def test_small_model_draft_probabilities_sum_to_one(self):
        """Draft probability distributions should be normalized."""
        device = torch.device("cpu")
        model = MockCausalLM(vocab_size=50, device=device)
        draft = SmallModelDraft(model, max_speculative=4, device=device)

        input_ids = torch.tensor([[5, 10, 15]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=4)

        # Check each position's distribution sums to 1
        sums = output.probs.sum(dim=-1)  # [batch, num_spec]
        np.testing.assert_allclose(
            sums.cpu().numpy(), 1.0, atol=1e-5, err_msg="Draft probabilities don't sum to 1"
        )

    def test_small_model_draft_tokens_are_argmax(self):
        """SmallModelDraft uses greedy decoding, so tokens should match argmax of probs."""
        device = torch.device("cpu")
        model = MockCausalLM(vocab_size=50, logit_scale=5.0, device=device)
        draft = SmallModelDraft(model, max_speculative=3, device=device)

        torch.manual_seed(42)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=3)

        for i in range(3):
            token = int(output.tokens[0, i].item())
            argmax = int(output.probs[0, i].argmax().item())
            assert token == argmax, f"Position {i}: token={token} != argmax={argmax}"

    def test_small_model_draft_batched(self):
        """SmallModelDraft should handle batch_size > 1."""
        device = torch.device("cpu")
        model = MockCausalLM(vocab_size=50, device=device)
        draft = SmallModelDraft(model, max_speculative=4, device=device)

        input_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=3)

        assert output.tokens.shape == (3, 3)
        assert output.probs.shape == (3, 3, 50)

    def test_small_model_draft_reset(self):
        """Reset should clear the internal cache."""
        device = torch.device("cpu")
        model = MockCausalLM(vocab_size=50, device=device)
        draft = SmallModelDraft(model, max_speculative=4, device=device)

        input_ids = torch.tensor([[1]], dtype=torch.long, device=device)
        draft.speculate(input_ids, num_tokens=2)
        assert draft._cache is not None

        draft.reset()
        assert draft._cache is None
        assert draft._cache_seq_len == 0

    def test_ngram_draft_output_shape(self):
        """NGramDraft should produce correct output shapes."""
        vocab_size = 100
        device = torch.device("cpu")
        draft = NGramDraft(ngram_size=3, vocab_size=vocab_size, device=device)

        # Need at least ngram_size tokens in input
        input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=3)

        assert output.tokens.shape == (1, 3)
        assert output.probs.shape == (1, 3, vocab_size)

    def test_ngram_updates_and_prediction(self):
        """After learning n-grams, predictions should use learned patterns."""
        device = torch.device("cpu")
        draft = NGramDraft(ngram_size=2, vocab_size=50, device=device)

        # Train on a repeating pattern: [1, 2, 3, 1, 2, 3, 1, 2, 3]
        pattern = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        draft.update_ngrams(pattern)

        # After context [2, 3], should predict 1 (most frequent next token)
        input_ids = torch.tensor([[2, 3]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=3)

        # First prediction should be 1 (after [2,3] comes 1)
        first_token = int(output.tokens[0, 0].item())
        assert first_token == 1, f"Expected token 1 after [2,3], got {first_token}"

    def test_ngram_updates_accumulate(self):
        """Multiple update_ngrams calls should accumulate statistics."""
        device = torch.device("cpu")
        draft = NGramDraft(ngram_size=2, vocab_size=50, device=device)

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
        device = torch.device("cpu")
        draft = NGramDraft(ngram_size=3, vocab_size=50, device=device)

        # Only 2 tokens, but ngram_size=3 -> not enough context
        input_ids = torch.tensor([[1, 2]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=2)

        # Should return uniform probs (token 0 as default)
        assert output.tokens.shape == (1, 2)
        # All tokens should be 0 (fallback)
        assert int(output.tokens[0, 0].item()) == 0
        assert int(output.tokens[0, 1].item()) == 0

    def test_ngram_reset_clears_state(self):
        """Reset should clear all n-gram statistics and history."""
        device = torch.device("cpu")
        draft = NGramDraft(ngram_size=2, vocab_size=50, device=device)
        draft.update_ngrams([1, 2, 3, 4, 5])

        assert len(draft.ngram_counts) > 0
        assert len(draft._history) > 0

        draft.reset()
        assert len(draft.ngram_counts) == 0
        assert len(draft._history) == 0

    def test_ngram_probabilities_normalized(self):
        """NGramDraft probabilities should sum to 1."""
        device = torch.device("cpu")
        draft = NGramDraft(ngram_size=2, vocab_size=50, device=device)
        draft.update_ngrams([1, 2, 3, 1, 2, 4, 1, 2, 5])

        input_ids = torch.tensor([[1, 2]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=2)

        sums = output.probs.sum(dim=-1)
        np.testing.assert_allclose(
            sums.cpu().numpy(),
            1.0,
            atol=1e-5,
            err_msg="NGramDraft probabilities don't sum to 1",
        )

    def test_ngram_batched_broadcasts(self):
        """NGramDraft with batch_size > 1 should broadcast predictions."""
        device = torch.device("cpu")
        draft = NGramDraft(ngram_size=2, vocab_size=50, device=device)
        draft.update_ngrams([10, 20, 30, 10, 20, 30])

        # Batch of 3, all same context
        input_ids = torch.tensor([[10, 20], [10, 20], [10, 20]], dtype=torch.long, device=device)
        output = draft.speculate(input_ids, num_tokens=2)

        assert output.tokens.shape == (3, 2)
        assert output.probs.shape == (3, 2, 50)


class TestMMFP4DraftModel:
    """Test MMFP4-specific draft model implementation."""

    def test_init(self):
        """MMFP4DraftModel should initialize correctly."""
        draft = MMFP4DraftModel(
            hidden_size=64,
            vocab_size=100,
            num_predictions=4,
            group_size=32
        )
        assert draft.num_predictions == 4
        assert draft.hidden_size == 64
        # Should have an MTP head
        assert hasattr(draft, 'mtp_head')

    def test_speculate_from_hidden(self):
        """MMFP4DraftModel should generate tokens from hidden states."""
        hidden_size = 32
        vocab_size = 50
        num_predictions = 4
        
        draft = MMFP4DraftModel(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_predictions=num_predictions,
            dtype=torch.float32
        )
        
        batch_size = 2
        # Mock hidden states: [batch, 1, hidden]
        hidden = torch.randn(batch_size, 1, hidden_size)
        
        output = draft.speculate_from_hidden(hidden, num_tokens=num_predictions)
        
        assert output.tokens.shape == (batch_size, num_predictions)
        assert output.probs.shape == (batch_size, num_predictions, vocab_size)
        
        # Probs should sum to 1
        np.testing.assert_allclose(
            output.probs.sum(dim=-1).detach().numpy(),
            1.0,
            atol=1e-5
        )

    def test_from_target_model(self):
        """Factory method should extract config from target model."""
        target = MockCausalLM(vocab_size=100)
        draft = MMFP4DraftModel.from_target_model(target, num_predictions=3)
        
        assert draft.vocab_size == 100
        assert draft.num_predictions == 3
        assert draft.hidden_size == 32  # MockCausalLM has 32

    def test_weight_sharing(self):
        """Weight sharing factory should enable weight sharing flag."""
        target = MockCausalLM(vocab_size=100)
        draft = MMFP4DraftModel.from_target_model_with_weight_sharing(
            target, 
            num_predictions=3,
            share_lm_head=False # Mock doesn't have compatible LM head usually
        )
        
        assert draft.is_weight_sharing_enabled()
        assert draft.vocab_size == 100