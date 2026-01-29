import numpy as np
import pytest
import torch

from metal_marlin.sampler import MetalSampler


@pytest.fixture
def sampler():
    return MetalSampler(vocab_size=32000)


def test_argmax_correctness(sampler):
    """Metal argmax should match torch.argmax."""
    logits = torch.randn(1, 32000, device="mps")

    metal_result = sampler.argmax(logits)
    torch_result = torch.argmax(logits, dim=-1).item()

    assert metal_result == torch_result


def test_top_p_distribution(sampler):
    """Top-p sampling should produce valid distribution."""
    torch.manual_seed(42)
    logits = torch.randn(1, 32000, device="mps")

    # Sample many times and check distribution
    samples = [sampler.sample_top_p(logits, p=0.9, temperature=1.0) for _ in range(100)]

    # All samples should be valid token IDs
    assert all(0 <= s < 32000 for s in samples)

    # Should have some variety (not all same token)
    assert len(set(samples)) > 1


def test_top_k_restricts_to_k(sampler):
    """Top-k should only sample from top k tokens."""
    # Create logits where top-5 are clearly dominant
    logits = torch.full((1, 32000), -100.0, device="mps")
    logits[0, :5] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])

    samples = [sampler.sample_top_k(logits, k=5, temperature=1.0) for _ in range(50)]

    # All samples should be in top-5
    assert all(s < 5 for s in samples)


def test_temperature_effect(sampler):
    """Higher temperature should increase entropy."""
    logits = torch.randn(1, 32000, device="mps")

    # Low temperature = peaky distribution (low entropy)
    low_temp_samples = [sampler.sample_categorical(logits, temperature=0.1) for _ in range(100)]

    # High temperature = flat distribution (high entropy)
    high_temp_samples = [sampler.sample_categorical(logits, temperature=2.0) for _ in range(100)]

    # High temp should have more unique samples
    assert len(set(high_temp_samples)) > len(set(low_temp_samples))
