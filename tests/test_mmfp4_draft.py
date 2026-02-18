"""Tests for MMFP4 draft model architecture for speculative decoding."""

from __future__ import annotations

import pytest
import torch

from metal_marlin.layers.mmfp4_mtp_head import MMFP4MTPHead, verify_kernel
from metal_marlin.speculative.mmfp4_draft import MMFP4DraftModel, MMFP4DraftModelWithTarget
from metal_marlin.speculative.mtp_draft import MTPDraft
from metal_marlin.speculative.engine import SpeculativeConfig, SpeculativeEngine
from metal_marlin.speculative.draft import DraftOutput


def _get_default_dtype():
    """Get default dtype for tests."""
    return torch.float32  # Use float32 for CPU tests


class MockTargetModel:
    """Mock target model for testing."""
    
    def __init__(self, vocab_size: int = 100, hidden_size: int = 256):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.config = type("Config", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
        self.call_count = 0
    
    def __call__(self, input_ids: torch.Tensor, kv_cache=None) -> torch.Tensor:
        self.call_count += 1
        batch, seq = input_ids.shape
        return torch.randn(batch, seq, self.vocab_size)
    
    def create_kv_cache(self):
        from metal_marlin.kv_cache import CacheConfig, KVCache
        config = CacheConfig(
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=128,
        )
        return KVCache(config, batch_size=1, device="cpu")
    
    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        # Return hidden states if requested
        if output_hidden_states:
            hidden = torch.randn(batch, seq, self.hidden_size)
            class Output:
                pass
            out = Output()
            out.hidden_states = [hidden]
            return out
        return torch.randn(batch, seq, self.vocab_size)
    
    def parameters(self):
        return iter([torch.randn(1)])


class TestMMFP4MTPHead:
    """Test MMFP4 MTP Head for multi-token prediction."""
    
    def test_mtp_head_output_shape(self):
        """MTP head should predict N future tokens."""
        batch, seq, hidden, vocab = 2, 10, 256, 100
        num_predictions = 4
        dtype = _get_default_dtype()
        
        head = MMFP4MTPHead(
            hidden_size=hidden,
            vocab_size=vocab,
            num_predictions=num_predictions,
        ).to(dtype=dtype)
        
        hidden_states = torch.randn(batch, seq, hidden, dtype=dtype)
        output = head(hidden_states)
        
        # Output should be [batch, num_predictions, vocab]
        assert output.shape == (batch, num_predictions, vocab)
    
    def test_mtp_head_speculate(self):
        """MTP head speculate should return tokens and probs."""
        batch, seq, hidden, vocab = 1, 5, 256, 100
        num_predictions = 4
        dtype = _get_default_dtype()
        
        head = MMFP4MTPHead(
            hidden_size=hidden,
            vocab_size=vocab,
            num_predictions=num_predictions,
        ).to(dtype=dtype)
        
        hidden_states = torch.randn(batch, seq, hidden, dtype=dtype)
        tokens, probs = head.speculate(hidden_states)
        
        assert tokens.shape == (batch, num_predictions)
        assert probs.shape == (batch, num_predictions, vocab)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch, num_predictions), atol=1e-5)
    
    def test_verify_kernel(self):
        """Verify kernel should accept/reject draft tokens correctly."""
        batch, num_spec, vocab = 2, 4, 50
        
        draft_tokens = torch.randint(0, vocab, (batch, num_spec))
        target_logits = torch.randn(batch, num_spec + 1, vocab)  # +1 for bonus token
        draft_probs = torch.softmax(torch.randn(batch, num_spec, vocab), dim=-1)
        
        num_accepted, accepted_mask, next_token = verify_kernel(
            draft_tokens, target_logits, draft_probs, temperature=1.0
        )
        
        assert num_accepted.shape == (batch,)
        assert accepted_mask.shape == (batch, num_spec)
        assert next_token.shape == (batch,)
        assert (num_accepted >= 0).all() and (num_accepted <= num_spec).all()


class TestMMFP4DraftModel:
    """Test MMFP4 draft model for speculative decoding."""
    
    def test_draft_model_creation(self):
        """Draft model should initialize correctly."""
        draft = MMFP4DraftModel(
            hidden_size=256,
            vocab_size=100,
            num_predictions=4,
        )
        
        assert draft.hidden_size == 256
        assert draft.vocab_size == 100
        assert draft.num_predictions == 4
        assert draft.mtp_head is not None
    
    def test_draft_model_from_target(self):
        """Draft model should create from target model."""
        target = MockTargetModel(vocab_size=100, hidden_size=256)
        draft = MMFP4DraftModel.from_target_model(target, num_predictions=4)
        
        assert draft.hidden_size == 256
        assert draft.vocab_size == 100
        assert draft.num_predictions == 4
    
    def test_draft_model_speculate_from_hidden(self):
        """Draft model should generate tokens from hidden states."""
        batch, seq, hidden, vocab = 1, 5, 256, 100
        num_predictions = 4
        dtype = _get_default_dtype()
        
        draft = MMFP4DraftModel(
            hidden_size=hidden,
            vocab_size=vocab,
            num_predictions=num_predictions,
            dtype=dtype,
        )
        
        hidden_states = torch.randn(batch, seq, hidden, dtype=dtype)
        output = draft.speculate_from_hidden(hidden_states, num_tokens=4)
        
        assert isinstance(output, DraftOutput)
        assert output.tokens.shape == (batch, 4)
        assert output.probs.shape == (batch, 4, vocab)
        # Probabilities should sum to 1
        assert torch.allclose(output.probs.sum(dim=-1), torch.ones(batch, 4), atol=1e-5)
    
    def test_draft_model_speculate_with_cache(self):
        """Draft model should use cached hidden states."""
        batch, seq, hidden, vocab = 1, 5, 256, 100
        dtype = _get_default_dtype()
        
        draft = MMFP4DraftModel(
            hidden_size=hidden,
            vocab_size=vocab,
            num_predictions=4,
            dtype=dtype,
        )
        
        # Set hidden states
        hidden_states = torch.randn(batch, seq, hidden, dtype=dtype)
        draft.set_hidden_states(hidden_states)
        
        # Create dummy input_ids
        input_ids = torch.randint(0, vocab, (batch, 3))
        output = draft.speculate(input_ids, num_tokens=4)
        
        assert isinstance(output, DraftOutput)
        assert output.tokens.shape == (batch, 4)
    
    def test_draft_model_reset(self):
        """Draft model should reset cache correctly."""
        draft = MMFP4DraftModel(
            hidden_size=256,
            vocab_size=100,
            num_predictions=4,
        )
        
        draft.set_hidden_states(torch.randn(1, 5, 256))
        assert draft._cached_hidden is not None
        
        draft.reset()
        assert draft._cached_hidden is None
        assert draft._cache_seq_len == 0
    
    def test_draft_model_fallback(self):
        """Draft model should fallback when no hidden states."""
        batch, vocab = 2, 100
        dtype = _get_default_dtype()
        
        draft = MMFP4DraftModel(
            hidden_size=256,
            vocab_size=vocab,
            num_predictions=4,
            dtype=dtype,
        )
        
        input_ids = torch.randint(0, vocab, (batch, 3))
        output = draft.speculate(input_ids, num_tokens=4)
        
        assert isinstance(output, DraftOutput)
        assert output.tokens.shape == (batch, 4)
        assert output.probs.shape == (batch, 4, vocab)


class TestMTPDraft:
    """Test MTPDraft adapter for speculative engine."""
    
    def test_mtp_draft_creation(self):
        """MTPDraft should wrap MTP head."""
        dtype = _get_default_dtype()
        head = MMFP4MTPHead(hidden_size=256, vocab_size=100, num_predictions=4).to(dtype=dtype)
        draft = MTPDraft(mtp_head=head)
        
        assert draft.head is head
        assert draft.device is not None
    
    def test_mtp_draft_speculate(self):
        """MTPDraft should generate speculative tokens."""
        dtype = _get_default_dtype()
        head = MMFP4MTPHead(hidden_size=256, vocab_size=100, num_predictions=4).to(dtype=dtype)
        draft = MTPDraft(mtp_head=head)
        
        # Set hidden states
        hidden_states = torch.randn(1, 5, 256, dtype=dtype)
        draft.set_hidden_states(hidden_states)
        
        input_ids = torch.randint(0, 100, (1, 3))
        output = draft.speculate(input_ids, num_tokens=4)
        
        assert isinstance(output, DraftOutput)
        assert output.tokens.shape == (1, 4)
        assert output.probs.shape == (1, 4, 100)
    
    def test_mtp_draft_fallback(self):
        """MTPDraft should fallback without hidden states."""
        dtype = _get_default_dtype()
        head = MMFP4MTPHead(hidden_size=256, vocab_size=100, num_predictions=4).to(dtype=dtype)
        draft = MTPDraft(mtp_head=head)
        
        # No hidden states set - should fallback
        input_ids = torch.randint(0, 100, (1, 3))
        output = draft.speculate(input_ids, num_tokens=4)
        
        assert isinstance(output, DraftOutput)
        assert output.tokens.shape == (1, 4)


class TestMMFP4SpeculativeIntegration:
    """Integration tests for MMFP4 speculative decoding."""
    
    def test_mmfp4_draft_with_speculative_engine(self):
        """MMFP4 draft model should work with SpeculativeEngine."""
        vocab_size = 100
        hidden_size = 256
        dtype = _get_default_dtype()
        
        target = MockTargetModel(vocab_size=vocab_size, hidden_size=hidden_size)
        draft = MMFP4DraftModel(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_predictions=4,
            dtype=dtype,
        )
        
        config = SpeculativeConfig(num_speculative_tokens=4)
        engine = SpeculativeEngine(target, draft, config)
        
        # Need to set hidden states for draft model
        hidden_states = torch.randn(1, 5, hidden_size, dtype=dtype)
        draft.set_hidden_states(hidden_states)
        
        input_ids = torch.randint(0, vocab_size, (1, 5))
        result = engine.generate_step(input_ids)
        
        assert result.num_target_calls == 1
        assert result.num_draft_tokens == 4
    
    def test_speedup_estimate(self):
        """Draft model should provide speedup estimates."""
        draft = MMFP4DraftModel(
            hidden_size=256,
            vocab_size=100,
            num_predictions=4,
        )
        
        # 100% acceptance should give >2x speedup
        speedup = draft.get_speedup_estimate(acceptance_rate=1.0)
        assert speedup > 2.0, f"Expected >2x speedup with 100% acceptance, got {speedup}"
        
        # 50% acceptance should give some speedup
        speedup = draft.get_speedup_estimate(acceptance_rate=0.5)
        assert speedup > 1.0, f"Expected some speedup with 50% acceptance, got {speedup}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
