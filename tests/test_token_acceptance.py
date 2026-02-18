
import pytest
import torch
from metal_marlin.speculative.token_acceptance import (
    AcceptanceResult,
    TokenAcceptanceTracker,
    compute_acceptance_probabilities,
    estimate_optimal_speculation_length,
)

class TestTokenAcceptanceTracker:
    def test_initialization(self):
        tracker = TokenAcceptanceTracker()
        assert tracker.stats.total_accepted == 0
        assert tracker.stats.step_count == 0
        assert tracker.track_history is True

    def test_compute_acceptance(self):
        tracker = TokenAcceptanceTracker()
        batch_size = 2
        num_spec = 4
        device = torch.device("cpu")

        draft_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
        num_accepted = torch.tensor([2, 4], device=device)
        next_token = torch.tensor([10, 11], device=device)

        result = tracker.compute_acceptance(
            draft_tokens, num_accepted, next_token, num_spec
        )

        assert isinstance(result, AcceptanceResult)
        assert torch.equal(result.num_accepted, num_accepted)
        assert torch.equal(result.next_token, next_token)
        
        # Check accepted tokens
        # Batch 0: 2 accepted -> [1, 2]
        assert torch.equal(result.accepted_tokens[0, :2], torch.tensor([1, 2], device=device))
        # Batch 1: 4 accepted -> [5, 6, 7, 8]
        assert torch.equal(result.accepted_tokens[1], torch.tensor([5, 6, 7, 8], device=device))

        # Check accepted mask
        assert result.accepted_mask[0, 0].item() is True
        assert result.accepted_mask[0, 1].item() is True
        assert result.accepted_mask[0, 2].item() is False
        assert result.accepted_mask[1, 3].item() is True

    def test_update_stats(self):
        tracker = TokenAcceptanceTracker()
        batch_size = 1
        num_spec = 4
        device = torch.device("cpu")

        draft_tokens = torch.zeros(batch_size, num_spec, device=device)
        num_accepted = torch.tensor([3], device=device) # 75% acceptance
        next_token = torch.zeros(batch_size, device=device)

        result = tracker.compute_acceptance(
            draft_tokens, num_accepted, next_token, num_spec
        )

        tracker.update_stats(result, num_spec)

        assert tracker.stats.total_accepted == 3
        assert tracker.stats.total_proposed == 4
        assert tracker.stats.total_rejected == 1
        assert tracker.stats.step_count == 1
        assert len(tracker.stats.acceptance_history) == 1
        assert tracker.stats.acceptance_history[0] == 0.75

    def test_assemble_sequence(self):
        tracker = TokenAcceptanceTracker()
        device = torch.device("cpu")
        
        input_ids = torch.tensor([[100]], device=device)
        # Accepted: [1, 2], Next: [10]
        # Result should be [100, 1, 2, 10]
        
        draft_tokens = torch.tensor([[1, 2, 3, 4]], device=device)
        num_accepted = torch.tensor([2], device=device)
        next_token = torch.tensor([10], device=device)
        
        result = tracker.compute_acceptance(
            draft_tokens, num_accepted, next_token, num_speculative=4
        )
        
        output = tracker.assemble_sequence(input_ids, result, include_next=True)
        
        expected = torch.tensor([[100, 1, 2, 10]], device=device)
        assert torch.equal(output, expected)

    def test_adaptive_speculation_length(self):
        tracker = TokenAcceptanceTracker()
        # Mock history with high acceptance
        tracker.stats.acceptance_history = [0.9] * 10
        
        current = 4
        new_len = tracker.compute_adaptive_speculation_length(
            current, target_acceptance_rate=0.6
        )
        assert new_len > current

        # Mock history with low acceptance
        tracker.stats.acceptance_history = [0.1] * 10
        new_len = tracker.compute_adaptive_speculation_length(
            current, target_acceptance_rate=0.6
        )
        assert new_len < current

def test_compute_acceptance_probabilities():
    batch = 1
    num_spec = 2
    vocab = 4
    device = torch.device("cpu")
    
    draft_probs = torch.zeros(batch, num_spec, vocab, device=device)
    target_probs = torch.zeros(batch, num_spec, vocab, device=device)
    draft_tokens = torch.zeros(batch, num_spec, dtype=torch.long, device=device)
    
    # Token 0: draft prob 0.5, target prob 0.2 -> ratio 0.4
    draft_probs[0, 0, 0] = 0.5
    target_probs[0, 0, 0] = 0.2
    
    # Token 1: draft prob 0.2, target prob 0.5 -> ratio > 1 -> 1.0
    draft_probs[0, 1, 0] = 0.2
    target_probs[0, 1, 0] = 0.5
    
    acc_probs = compute_acceptance_probabilities(draft_probs, target_probs, draft_tokens)
    
    assert torch.isclose(acc_probs[0, 0], torch.tensor(0.4))
    assert torch.isclose(acc_probs[0, 1], torch.tensor(1.0))

def test_estimate_optimal_speculation_length():
    # If acceptance is 1.0 everywhere, optimal length should be high
    probs = torch.ones(1, 10)
    length = estimate_optimal_speculation_length(probs)
    assert length == 10
    
    # If acceptance is 0.0, optimal length should be 1 (minimum)
    probs = torch.zeros(1, 10)
    length = estimate_optimal_speculation_length(probs)
    assert length == 1
