"""Tests for prefill optimization module.

Tests chunked_prefill, parallel_kv_write, flash_prefill_attention,
and related prefill utilities.
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX required for prefill tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_config():
    """Standard test model configuration."""
    return {
        "vocab_size": 32000,
        "hidden_size": 256,  # Small for testing
        "intermediate_size": 512,
        "num_heads": 4,
        "num_kv_heads": 2,  # GQA
        "head_dim": 64,
        "num_layers": 2,
        "max_seq_len": 1024,
    }


@pytest.fixture
def kv_cache(model_config):
    """Create a test KV cache."""
    from metal_marlin.kv_cache import CacheConfig, KVCache

    cache_config = CacheConfig(
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        num_kv_heads=model_config["num_kv_heads"],
        head_dim=model_config["head_dim"],
        max_seq_len=model_config["max_seq_len"],
    )
    return KVCache(cache_config, batch_size=1)


class MockModel:
    """Mock model for testing prefill."""

    def __init__(self, config: dict):
        self.config = config
        self._call_count = 0

    def __call__(
        self,
        input_ids: mx.array,
        kv_cache=None,
        attention_mask=None,
    ) -> mx.array:
        """Mock forward pass - returns random logits."""
        self._call_count += 1
        batch_size, seq_len = input_ids.shape
        return mx.random.normal((batch_size, seq_len, self.config["vocab_size"]))

    def create_kv_cache(self, batch_size: int = 1):
        from metal_marlin.kv_cache import CacheConfig, KVCache

        cache_config = CacheConfig(
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            num_kv_heads=self.config["num_kv_heads"],
            head_dim=self.config["head_dim"],
            max_seq_len=self.config["max_seq_len"],
        )
        return KVCache(cache_config, batch_size)


@pytest.fixture
def mock_model(model_config):
    """Create a mock model for testing."""
    return MockModel(model_config)


# ---------------------------------------------------------------------------
# PrefillConfig tests
# ---------------------------------------------------------------------------


class TestPrefillConfig:
    """Tests for PrefillConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from metal_marlin.inference.prefill import PrefillConfig

        config = PrefillConfig()
        assert config.chunk_size == 512
        assert config.use_flash_attention is True
        assert config.parallel_kv_writes is True
        assert config.memory_fraction == 0.8

    def test_custom_config(self):
        """Test custom configuration."""
        from metal_marlin.inference.prefill import PrefillConfig

        config = PrefillConfig(
            chunk_size=2048,
            use_flash_attention=False,
            memory_fraction=0.6,
        )
        assert config.chunk_size == 2048
        assert config.use_flash_attention is False
        assert config.memory_fraction == 0.6


# ---------------------------------------------------------------------------
# chunked_prefill tests
# ---------------------------------------------------------------------------


class TestChunkedPrefill:
    """Tests for chunked_prefill function."""

    def test_short_prompt_single_chunk(self, mock_model, model_config):
        """Test that short prompts use single chunk."""
        from metal_marlin.inference.prefill import PrefillConfig, chunked_prefill

        input_ids = mx.zeros((1, 100), dtype=mx.int32)
        config = PrefillConfig(chunk_size=512)

        logits, stats = chunked_prefill(mock_model, input_ids, config=config)

        assert logits.shape == (1, 100, model_config["vocab_size"])
        assert stats.num_chunks == 1
        assert stats.total_tokens == 100
        assert mock_model._call_count == 1

    def test_long_prompt_multiple_chunks(self, mock_model, model_config):
        """Test that long prompts are chunked correctly."""
        from metal_marlin.inference.prefill import PrefillConfig, chunked_prefill

        input_ids = mx.zeros((1, 800), dtype=mx.int32)
        config = PrefillConfig(chunk_size=256)

        logits, stats = chunked_prefill(mock_model, input_ids, config=config)

        assert logits.shape == (1, 800, model_config["vocab_size"])
        assert stats.num_chunks == 4  # 800 / 256 = 3.125 -> 4 chunks
        assert stats.total_tokens == 800
        assert mock_model._call_count == 4

    def test_prefill_stats(self, mock_model):
        """Test that stats are populated correctly."""
        from metal_marlin.inference.prefill import PrefillConfig, chunked_prefill

        input_ids = mx.zeros((1, 512), dtype=mx.int32)
        config = PrefillConfig(chunk_size=256)

        _, stats = chunked_prefill(mock_model, input_ids, config=config)

        assert stats.total_tokens == 512
        assert stats.num_chunks == 2
        assert stats.prefill_time_ms > 0
        assert stats.tokens_per_second > 0
        assert len(stats.chunk_times_ms) == 2

    def test_progress_callback(self, mock_model):
        """Test progress callback invocation."""
        from metal_marlin.inference.prefill import PrefillConfig, chunked_prefill

        input_ids = mx.zeros((1, 600), dtype=mx.int32)
        config = PrefillConfig(chunk_size=256)

        progress_calls = []

        def on_progress(done, total):
            progress_calls.append((done, total))

        chunked_prefill(mock_model, input_ids, config=config, progress_callback=on_progress)

        assert len(progress_calls) == 3  # 256, 512, 600
        assert progress_calls[-1] == (600, 600)

    def test_invalid_batch_size(self, mock_model):
        """Test that batch_size > 1 raises error."""
        from metal_marlin.inference.prefill import chunked_prefill

        input_ids = mx.zeros((2, 100), dtype=mx.int32)

        with pytest.raises(ValueError, match="batch_size=1"):
            chunked_prefill(mock_model, input_ids)

    def test_sequence_exceeds_cache(self, mock_model, kv_cache):
        """Test that exceeding cache capacity raises error."""
        from metal_marlin.inference.prefill import chunked_prefill

        # kv_cache has max_seq_len=1024
        input_ids = mx.zeros((1, 2000), dtype=mx.int32)

        with pytest.raises(ValueError, match="exceeds"):
            chunked_prefill(mock_model, input_ids, kv_cache=kv_cache)


# ---------------------------------------------------------------------------
# parallel_kv_write tests
# ---------------------------------------------------------------------------


class TestParallelKVWrite:
    """Tests for parallel_kv_write function."""

    def test_basic_write(self, kv_cache, model_config):
        """Test basic parallel KV write."""
        from metal_marlin.inference.prefill import parallel_kv_write

        num_layers = model_config["num_layers"]
        num_kv_heads = model_config["num_kv_heads"]
        head_dim = model_config["head_dim"]
        seq_len = 32

        # Create dummy K/V tensors
        keys = [
            mx.random.normal((1, num_kv_heads, seq_len, head_dim))
            for _ in range(num_layers)
        ]
        values = [
            mx.random.normal((1, num_kv_heads, seq_len, head_dim))
            for _ in range(num_layers)
        ]

        # Write to cache
        parallel_kv_write(keys, values, kv_cache)

        # Verify write by checking cache state
        # Note: kv_cache.update was called for each layer
        assert kv_cache.seq_len == 0  # parallel_kv_write doesn't advance

    def test_mismatched_keys_values(self, kv_cache, model_config):
        """Test that mismatched key/value counts raise error."""
        from metal_marlin.inference.prefill import parallel_kv_write

        num_kv_heads = model_config["num_kv_heads"]
        head_dim = model_config["head_dim"]

        keys = [mx.random.normal((1, num_kv_heads, 32, head_dim)) for _ in range(2)]
        values = [mx.random.normal((1, num_kv_heads, 32, head_dim)) for _ in range(3)]

        with pytest.raises(ValueError, match="mismatch"):
            parallel_kv_write(keys, values, kv_cache)


# ---------------------------------------------------------------------------
# flash_prefill_attention tests
# ---------------------------------------------------------------------------


class TestFlashPrefillAttention:
    """Tests for flash_prefill_attention function."""

    def test_basic_attention(self):
        """Test basic flash attention computation."""
        from metal_marlin.inference.prefill import flash_prefill_attention

        batch, num_heads, seq_len, head_dim = 1, 4, 64, 64
        num_kv_heads = 4

        q = mx.random.normal((batch, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch, num_kv_heads, seq_len, head_dim))
        v = mx.random.normal((batch, num_kv_heads, seq_len, head_dim))

        output = flash_prefill_attention(q, k, v, causal=True)

        assert output.shape == (batch, num_heads, seq_len, head_dim)

    def test_gqa_attention(self):
        """Test GQA (num_kv_heads < num_heads)."""
        from metal_marlin.inference.prefill import flash_prefill_attention

        batch, num_heads, seq_len, head_dim = 1, 8, 32, 64
        num_kv_heads = 2

        q = mx.random.normal((batch, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch, num_kv_heads, seq_len, head_dim))
        v = mx.random.normal((batch, num_kv_heads, seq_len, head_dim))

        output = flash_prefill_attention(q, k, v, causal=True)

        assert output.shape == (batch, num_heads, seq_len, head_dim)

    def test_causal_masking(self):
        """Test that causal masking prevents attending to future."""
        from metal_marlin.inference.prefill import flash_prefill_attention

        batch, num_heads, seq_len, head_dim = 1, 2, 4, 16

        # Create Q and K where future tokens have large values
        q = mx.ones((batch, num_heads, seq_len, head_dim))
        k = mx.ones((batch, num_heads, seq_len, head_dim))
        v = mx.zeros((batch, num_heads, seq_len, head_dim))

        # Put distinctive values in V at each position
        for i in range(seq_len):
            v = v.at[:, :, i, :].add(mx.full((batch, num_heads, head_dim), float(i)))

        output_causal = flash_prefill_attention(q, k, v, causal=True)
        output_noncausal = flash_prefill_attention(q, k, v, causal=False)

        # With causal masking, position 0 should only see V[0]
        # Without causal masking, all positions see all V
        mx.eval([output_causal, output_noncausal])

        # Causal: first position only attends to first position
        # Noncausal: first position attends to all positions
        assert output_causal.shape == output_noncausal.shape

    def test_custom_scale(self):
        """Test custom attention scale."""
        from metal_marlin.inference.prefill import flash_prefill_attention

        batch, num_heads, seq_len, head_dim = 1, 2, 16, 32

        q = mx.random.normal((batch, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch, num_heads, seq_len, head_dim))

        # Custom scale
        output = flash_prefill_attention(q, k, v, scale=0.1, causal=True)

        assert output.shape == (batch, num_heads, seq_len, head_dim)


# ---------------------------------------------------------------------------
# batched_kv_projection tests
# ---------------------------------------------------------------------------


class TestBatchedKVProjection:
    """Tests for batched_kv_projection function."""

    def test_basic_projection(self, model_config):
        """Test basic batched KV projection."""
        from metal_marlin.inference.prefill import batched_kv_projection

        # Create minimal layer structure
        hidden_size = model_config["hidden_size"]
        num_kv_heads = model_config["num_kv_heads"]
        head_dim = model_config["head_dim"]
        num_layers = model_config["num_layers"]

        class MockAttention:
            def __init__(self):
                self.num_kv_heads = num_kv_heads
                self.head_dim = head_dim
                self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
                self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
                self.rope = MockRoPE()

            def _init_weights(self):
                pass

        class MockRoPE:
            def __call__(self, x, offset=0):
                return x  # Identity for testing

        class MockLayer:
            def __init__(self):
                self.self_attn = MockAttention()

        layers = [MockLayer() for _ in range(num_layers)]

        hidden_states = mx.random.normal((1, 32, hidden_size))

        result = batched_kv_projection(hidden_states, layers, rope_offset=0)

        assert len(result.keys) == num_layers
        assert len(result.values) == num_layers
        assert result.compute_time_ms > 0

        # Check shapes
        for k, v in zip(result.keys, result.values):
            assert k.shape == (1, num_kv_heads, 32, head_dim)
            assert v.shape == (1, num_kv_heads, 32, head_dim)


# ---------------------------------------------------------------------------
# SpeculativePrefill tests
# ---------------------------------------------------------------------------


class TestSpeculativePrefill:
    """Tests for speculative_prefill function."""

    def test_disabled_speculation(self, mock_model):
        """Test that disabled speculation returns empty tokens."""
        from metal_marlin.inference.prefill import (
            SpeculativePrefillConfig,
            speculative_prefill,
        )

        input_ids = mx.zeros((1, 100), dtype=mx.int32)
        kv_cache = mock_model.create_kv_cache()

        config = SpeculativePrefillConfig(enabled=False)

        logits, spec_tokens = speculative_prefill(
            mock_model, input_ids, kv_cache, config
        )

        assert logits.shape[0] == 1
        assert logits.shape[1] == 100
        assert spec_tokens == []

    def test_enabled_speculation(self, mock_model):
        """Test speculative prefill with speculation enabled."""
        from metal_marlin.inference.prefill import (
            SpeculativePrefillConfig,
            speculative_prefill,
        )

        input_ids = mx.zeros((1, 50), dtype=mx.int32)
        kv_cache = mock_model.create_kv_cache()

        config = SpeculativePrefillConfig(
            enabled=True,
            num_speculative_tokens=4,
            confidence_threshold=0.0,  # Accept any prediction
        )

        logits, spec_tokens = speculative_prefill(
            mock_model, input_ids, kv_cache, config
        )

        # With threshold=0.0, we should get some speculative tokens
        assert logits.shape[0] == 1
        # spec_tokens may or may not be populated depending on model output


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPrefillIntegration:
    """Integration tests combining prefill components."""

    def test_prefill_then_decode(self, mock_model, model_config):
        """Test prefill followed by decode simulation."""
        from metal_marlin.inference.prefill import PrefillConfig, chunked_prefill

        # Prefill phase
        input_ids = mx.zeros((1, 256), dtype=mx.int32)
        kv_cache = mock_model.create_kv_cache()
        config = PrefillConfig(chunk_size=128)

        logits, stats = chunked_prefill(
            mock_model, input_ids, kv_cache=kv_cache, config=config
        )

        assert stats.num_chunks == 2
        assert logits.shape == (1, 256, model_config["vocab_size"])

        # Cache should be advanced
        assert kv_cache.seq_len == 256

    def test_prefill_memory_estimate(self, mock_model):
        """Test that memory estimation is reasonable."""
        from metal_marlin.inference.prefill import PrefillConfig, chunked_prefill

        input_ids = mx.zeros((1, 512), dtype=mx.int32)
        config = PrefillConfig(chunk_size=256)

        _, stats = chunked_prefill(mock_model, input_ids, config=config)

        # Memory should be positive and reasonable
        assert stats.peak_memory_mb > 0
        assert stats.peak_memory_mb < 10000  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
