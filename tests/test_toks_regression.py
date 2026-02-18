"""Regression tests for token throughput (tok/s).

Target: _toks_tests
"""
import pytest
import torch
from .fixtures.synthetic_mixed_moe import create_synthetic_model, benchmark_forward

# Thresholds for synthetic model on MPS
# Note: Synthetic model uses FakeTrellisLinear (F.linear) so it should be fast.
# This establishes a baseline for the test infrastructure itself and upper bound.
THRESHOLDS = {
    "synthetic_decode_tok_s": 400.0,
    "synthetic_prefill_tok_s": 1000.0,
}

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
class TestToksRegression:
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model once for the class."""
        return create_synthetic_model(device="mps")

    def test_synthetic_decode_throughput(self, model):
        """Test decode throughput (batch=1, seq_len=1)."""
        result = benchmark_forward(
            model,
            batch_size=1,
            seq_len=1,
            warmup=5,
            iterations=20,
            device="mps"
        )
        print(f"\nSynthetic Decode: {result.throughput_tokens_per_sec:.1f} tok/s")
        
        # Log for observability
        print(f"OBSERVABILITY: metric=decode_tok_s value={result.throughput_tokens_per_sec:.2f}")

        assert result.throughput_tokens_per_sec > THRESHOLDS["synthetic_decode_tok_s"], (
            f"Decode throughput {result.throughput_tokens_per_sec:.1f} < {THRESHOLDS['synthetic_decode_tok_s']}"
        )

    def test_synthetic_prefill_throughput(self, model):
        """Test prefill throughput (batch=1, seq_len=128)."""
        result = benchmark_forward(
            model,
            batch_size=1,
            seq_len=128,
            warmup=2,
            iterations=5,
            device="mps"
        )
        print(f"\nSynthetic Prefill: {result.throughput_tokens_per_sec:.1f} tok/s")
        
        # Log for observability
        print(f"OBSERVABILITY: metric=prefill_tok_s value={result.throughput_tokens_per_sec:.2f}")
        
        assert result.throughput_tokens_per_sec > THRESHOLDS["synthetic_prefill_tok_s"], (
            f"Prefill throughput {result.throughput_tokens_per_sec:.1f} < {THRESHOLDS['synthetic_prefill_tok_s']}"
        )