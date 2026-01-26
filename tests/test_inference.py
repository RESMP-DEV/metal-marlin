"""Tests for inference components using generic model.

These tests use a simple transformer model for testing the inference
infrastructure (KV cache, generation loop) without hardcoding a specific
architecture like Llama.
"""

import pytest

from metal_marlin._compat import HAS_MLX

# Skip entire module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="Requires MLX (Apple Silicon only)")

# Import MLX modules only after skip check
if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn

    from metal_marlin.generate import GenerationConfig, generate
    from metal_marlin.kv_cache import CacheConfig, KVCache


class SimpleTransformer(nn.Module):
    """Minimal transformer for testing inference infrastructure.

    This is NOT a real model - it's a test fixture that implements
    the interface expected by generate() without being architecture-specific.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4)
            for _ in range(num_layers)
        ]
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        kv_cache: KVCache | None = None,
    ) -> mx.array:
        """Forward pass returning logits."""
        h = self.embed(input_ids)

        # Create causal mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        for layer in self.layers:
            h = layer(h, mask=mask)

        logits = self.lm_head(h)
        return logits

    def create_kv_cache(self, batch_size: int = 1) -> KVCache:
        """Create KV cache for this model."""
        config = CacheConfig(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=512,
        )
        return KVCache(config, batch_size=batch_size)


@pytest.fixture
def small_model():
    """Create tiny model for testing."""
    return SimpleTransformer(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
    )


class TestKVCache:
    def test_cache_creation(self):
        config = CacheConfig(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=512,
        )
        cache = KVCache(config, batch_size=1)

        assert cache.seq_len == 0
        assert len(cache.k_cache) == 2

    def test_cache_update(self):
        config = CacheConfig(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=512,
        )
        cache = KVCache(config, batch_size=1)

        # Add some KV
        k = mx.random.normal((1, 4, 10, 64))
        v = mx.random.normal((1, 4, 10, 64))

        k_full, v_full = cache.update(0, k, v)

        assert k_full.shape == (1, 4, 10, 64)

        cache.advance(10)
        assert cache.seq_len == 10


class TestGeneration:
    def test_greedy_generation(self, small_model):
        input_ids = mx.array([[1, 2, 3, 4, 5]])  # Dummy prompt

        config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,  # Greedy
            eos_token_id=999,  # Won't hit this
        )

        output = generate(small_model, input_ids, config)

        # Should have prompt + generated tokens
        assert output.shape[1] == 5 + 10

    def test_sampling_generation(self, small_model):
        input_ids = mx.array([[1, 2, 3]])

        config = GenerationConfig(
            max_new_tokens=5,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )

        output = generate(small_model, input_ids, config)
        assert output.shape[1] >= 3  # At least prompt

    def test_eos_stopping(self, small_model):
        input_ids = mx.array([[1, 2, 3]])

        # Test EOS detection mechanism exists
        # Note: Whether EOS actually triggers depends on model outputs
        config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=999,  # Unlikely to be generated
        )

        output = generate(small_model, input_ids, config)
        # Should complete with prompt + max_new_tokens
        assert output.shape[1] == 3 + 10


class TestModel:
    def test_forward_pass(self, small_model):
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        logits = small_model(input_ids)

        # [batch, seq, vocab]
        assert logits.shape == (1, 5, 1000)

    def test_with_cache(self, small_model):
        cache = small_model.create_kv_cache()

        # Prefill
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = small_model(input_ids, kv_cache=cache)
        cache.advance(5)

        # Decode
        next_input = mx.array([[6]])
        logits = small_model(next_input, kv_cache=cache)

        assert logits.shape == (1, 1, 1000)
        assert cache.seq_len == 5  # Not yet advanced
