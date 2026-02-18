
import torch
import pytest
from metal_marlin.layers.mmfp4_mtp_head import verify_kernel

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
class TestVerifyKernelMPS:
    """Test verify_kernel from mmfp4_mtp_head.py on MPS."""

    def test_perfect_match_accepts_all(self):
        """When draft == target distributions, all tokens should be accepted."""
        torch.manual_seed(42)
        batch, num_spec, vocab = 2, 4, 50
        device = torch.device("mps")

        draft_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long, device=device)

        # Target logits: high value at draft tokens
        target_logits = torch.full((batch, num_spec + 1, vocab), -10.0, device=device)
        for b in range(batch):
            for i in range(num_spec):
                tok = int(draft_tokens[b, i].item())
                target_logits[b, i, tok] = 10.0
        
        # Bonus position
        target_logits[:, num_spec, 0] = 10.0

        # Run verify_kernel
        num_accepted, accepted_mask, next_token = verify_kernel(
            draft_tokens, target_logits, temperature=1.0
        )

        assert num_accepted.device.type == "mps"
        assert accepted_mask.device.type == "mps"
        assert next_token.device.type == "mps"

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
        device = torch.device("mps")

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

    def test_verify_speed(self):
        """Simple check to ensure it doesn't crash and is reasonable."""
        batch, num_spec, vocab = 1, 4, 32000
        device = torch.device("mps")
        draft_tokens = torch.randint(0, vocab, (batch, num_spec), device=device)
        target_logits = torch.randn(batch, num_spec + 1, vocab, device=device)
        
        # Warmup
        verify_kernel(draft_tokens, target_logits)
        
        # Just run it
        verify_kernel(draft_tokens, target_logits)
