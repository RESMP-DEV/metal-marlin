import torch
import pytest
from metal_marlin.layers.mmfp4_mtp_head import verify_kernel

class TestVerifyKernel:
    """Test verify_kernel from mmfp4_mtp_head.py."""

    def test_perfect_match_accepts_all(self):
        """When draft == target distributions, all tokens should be accepted."""
        torch.manual_seed(42)
        batch, num_spec, vocab = 2, 4, 50
        device = torch.device("cpu")

        draft_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long, device=device)

        # Create a distribution and use it for both draft and target
        # Concentrate probability on the draft tokens for near-certain acceptance
        # verify_kernel uses draft_probs=None by default (greedy draft)
        # So we ensure target gives high prob to draft tokens.
        
        # Target logits: high value at draft tokens
        target_logits = torch.full((batch, num_spec + 1, vocab), -10.0, device=device)
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                target_logits[b, i, tok] = 10.0
        
        # Bonus position
        target_logits[:, num_spec, 0] = 10.0

        # Run verify_kernel
        # Note: verify_kernel returns (num_accepted, accepted_mask, next_token)
        num_accepted, accepted_mask, next_token = verify_kernel(
            draft_tokens, target_logits, temperature=1.0
        )

        # With high concentration, should accept all
        assert num_accepted.shape == (batch,)
        assert accepted_mask.shape == (batch, num_spec)
        assert next_token.shape == (batch,)

        for b in range(batch):
            assert int(num_accepted[b].item()) == num_spec
            assert accepted_mask[b].all()

    def test_complete_mismatch_rejects_first(self):
        """When target mismatches draft, should reject."""
        torch.manual_seed(123)
        batch, num_spec, vocab = 1, 4, 50
        device = torch.device("cpu")

        draft_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.long, device=device)
        
        # Target puts mass on different tokens
        target_logits = torch.full((batch, num_spec + 1, vocab), -10.0, device=device)
        for i in range(num_spec + 1):
            target_logits[0, i, 40 + i] = 10.0

        num_accepted, accepted_mask, next_token = verify_kernel(
            draft_tokens, target_logits, temperature=1.0
        )

        # Should reject first token
        assert int(num_accepted[0].item()) == 0
        assert not accepted_mask[0, 0]

    def test_output_shapes(self):
        batch, num_spec, vocab = 3, 5, 200
        device = torch.device("cpu")

        draft_tokens = torch.zeros((batch, num_spec), dtype=torch.long, device=device)
        target_logits = torch.randn(batch, num_spec + 1, vocab, device=device)

        num_accepted, accepted_mask, next_token = verify_kernel(
            draft_tokens, target_logits
        )

        assert num_accepted.shape == (batch,)
        assert accepted_mask.shape == (batch, num_spec)
        assert next_token.shape == (batch,)

    def test_temperature_scaling(self):
        """Test that temperature argument is accepted and affects results."""
        # This mostly verifies the function signature and execution path
        batch, num_spec, vocab = 1, 4, 50
        device = torch.device("cpu")
        draft_tokens = torch.zeros((batch, num_spec), dtype=torch.long, device=device)
        target_logits = torch.randn(batch, num_spec + 1, vocab, device=device)
        
        verify_kernel(draft_tokens, target_logits, temperature=0.5)
        verify_kernel(draft_tokens, target_logits, temperature=2.0)