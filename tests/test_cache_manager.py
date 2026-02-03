"""Tests for PagedKVCache tracker."""

import numpy as np
import pytest

from metal_marlin.paged import CacheStats, KVBlockConfig, PagedKVCache


class TestPagedKVCacheBasics:
    def test_init_default(self):
        cache = PagedKVCache(num_blocks=100)
        assert cache.num_blocks == 100
        assert cache.num_sequences == 0
        assert not cache.use_multimodal

    def test_init_with_config(self):
        config = KVBlockConfig(block_size=8, num_heads=16, head_dim=64)
        cache = PagedKVCache(config=config, num_blocks=50)
        assert cache.config.block_size == 8
        assert cache.config.num_heads == 16
        assert cache.config.head_dim == 64
        assert cache.num_blocks == 50

    def test_add_sequence(self):
        cache = PagedKVCache(num_blocks=10)
        success = cache.add_sequence(seq_id=0)
        assert success
        assert cache.num_sequences == 1
        assert cache.has_sequence(0)

    def test_add_multiple_sequences(self):
        cache = PagedKVCache(num_blocks=10)
        for i in range(5):
            assert cache.add_sequence(seq_id=i)
        assert cache.num_sequences == 5
        assert cache.sequence_ids() == [0, 1, 2, 3, 4]

    def test_add_sequence_oom(self):
        # Only 2 blocks, each sequence needs 1 initial block
        cache = PagedKVCache(num_blocks=2)
        assert cache.add_sequence(0)
        assert cache.add_sequence(1)
        # Third should fail
        assert not cache.add_sequence(2)

    def test_remove_sequence(self):
        cache = PagedKVCache(num_blocks=10)
        cache.add_sequence(0)
        cache.add_sequence(1)
        assert cache.num_sequences == 2

        cache.remove_sequence(0)
        assert cache.num_sequences == 1
        assert not cache.has_sequence(0)
        assert cache.has_sequence(1)


class TestPagedKVCacheAppend:
    def test_append_single_kv(self):
        config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        cache = PagedKVCache(config=config, num_blocks=10)
        cache.add_sequence(0)

        k = np.random.randn(2, 8).astype(np.float16)
        v = np.random.randn(2, 8).astype(np.float16)

        success = cache.append_kv(seq_id=0, key=k, value=v)
        assert success

        # Verify retrieval
        keys, values = cache.get_kv(0)
        assert keys.shape == (1, 2, 8)
        assert values.shape == (1, 2, 8)
        np.testing.assert_allclose(keys[0], k, rtol=1e-3)
        np.testing.assert_allclose(values[0], v, rtol=1e-3)

    def test_append_multiple_kv(self):
        config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        cache = PagedKVCache(config=config, num_blocks=10)
        cache.add_sequence(0)

        # Append 3 tokens
        for i in range(3):
            k = np.full((2, 8), fill_value=float(i), dtype=np.float16)
            v = np.full((2, 8), fill_value=float(i + 10), dtype=np.float16)
            assert cache.append_kv(0, k, v)

        keys, values = cache.get_kv(0)
        assert keys.shape == (3, 2, 8)
        assert values.shape == (3, 2, 8)

        # Check values
        for i in range(3):
            assert np.allclose(keys[i], float(i), rtol=1e-3)
            assert np.allclose(values[i], float(i + 10), rtol=1e-3)

    def test_append_kv_batch(self):
        config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        cache = PagedKVCache(config=config, num_blocks=10)
        cache.add_sequence(0)

        # Append 5 tokens at once (spans 2 blocks)
        keys = np.random.randn(5, 2, 8).astype(np.float16)
        values = np.random.randn(5, 2, 8).astype(np.float16)

        success = cache.append_kv_batch(0, keys, values)
        assert success

        # Verify
        k_out, v_out = cache.get_kv(0)
        assert k_out.shape == (5, 2, 8)
        assert v_out.shape == (5, 2, 8)
        np.testing.assert_allclose(k_out, keys, rtol=1e-3)
        np.testing.assert_allclose(v_out, values, rtol=1e-3)

    def test_append_across_block_boundary(self):
        config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        cache = PagedKVCache(config=config, num_blocks=10)
        cache.add_sequence(0)

        # Fill first block
        for _ in range(4):
            k = np.random.randn(2, 8).astype(np.float16)
            v = np.random.randn(2, 8).astype(np.float16)
            assert cache.append_kv(0, k, v)

        # This should allocate second block
        k = np.random.randn(2, 8).astype(np.float16)
        v = np.random.randn(2, 8).astype(np.float16)
        assert cache.append_kv(0, k, v)

        # Should have 2 blocks now
        block_table = cache.get_block_table(0)
        assert len(block_table) == 2

        keys, values = cache.get_kv(0)
        assert keys.shape == (5, 2, 8)


class TestPagedKVCacheFork:
    def test_fork_sequence(self):
        config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        cache = PagedKVCache(config=config, num_blocks=10)
        cache.add_sequence(0)

        # Add some data
        for i in range(3):
            k = np.full((2, 8), fill_value=float(i), dtype=np.float16)
            v = np.full((2, 8), fill_value=float(i + 10), dtype=np.float16)
            cache.append_kv(0, k, v)

        # Fork sequence
        success = cache.fork_sequence(src_id=0, dst_id=1)
        assert success
        assert cache.num_sequences == 2

        # Both should have same data
        k0, v0 = cache.get_kv(0)
        k1, v1 = cache.get_kv(1)
        np.testing.assert_allclose(k0, k1, rtol=1e-3)
        np.testing.assert_allclose(v0, v1, rtol=1e-3)

        # Block tables should share blocks initially
        table0 = cache.get_block_table(0)
        table1 = cache.get_block_table(1)
        assert table0 == table1

    def test_fork_sequence_nonexistent(self):
        cache = PagedKVCache(num_blocks=10)
        success = cache.fork_sequence(src_id=99, dst_id=1)
        assert not success


class TestPagedKVCacheStats:
    def test_stats_empty(self):
        cache = PagedKVCache(num_blocks=100)
        stats = cache.get_stats()

        assert stats.num_blocks_total == 100
        assert stats.num_blocks_free == 100
        assert stats.num_blocks_allocated == 0
        assert stats.num_sequences == 0
        assert stats.total_tokens == 0

    def test_stats_with_sequences(self):
        config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        cache = PagedKVCache(config=config, num_blocks=100)

        # Add 2 sequences
        cache.add_sequence(0)
        cache.add_sequence(1)

        # Add 3 tokens to seq 0
        for _ in range(3):
            k = np.random.randn(2, 8).astype(np.float16)
            v = np.random.randn(2, 8).astype(np.float16)
            cache.append_kv(0, k, v)

        # Add 5 tokens to seq 1 (will need 2 blocks)
        for _ in range(5):
            k = np.random.randn(2, 8).astype(np.float16)
            v = np.random.randn(2, 8).astype(np.float16)
            cache.append_kv(1, k, v)

        stats = cache.get_stats()
        assert stats.num_sequences == 2
        assert stats.total_tokens == 8  # 3 + 5
        assert stats.num_blocks_allocated == 3  # 1 for seq0, 2 for seq1
        assert stats.num_blocks_free == 97

    def test_memory_accounting(self):
        config = KVBlockConfig(block_size=16, num_heads=32, head_dim=128)
        cache = PagedKVCache(config=config, num_blocks=100)

        stats = cache.get_stats()
        # Each block: 2 * 16 * 32 * 128 * 2 bytes (fp16) = 262144 bytes
        expected_total = 100 * 262144
        assert stats.memory_total_bytes == expected_total
        assert stats.memory_used_bytes == 0

        # Add one sequence (1 block)
        cache.add_sequence(0)
        stats = cache.get_stats()
        assert stats.memory_used_bytes == 262144


class TestPagedKVCacheEdgeCases:
    def test_get_kv_empty_sequence(self):
        cache = PagedKVCache(num_blocks=10)
        cache.add_sequence(0)

        keys, values = cache.get_kv(0)
        assert keys.shape == (0, 32, 128)  # Default config
        assert values.shape == (0, 32, 128)

    def test_get_kv_nonexistent_sequence(self):
        cache = PagedKVCache(num_blocks=10)
        with pytest.raises(ValueError, match="not registered"):
            cache.get_kv(99)

    def test_append_kv_nonexistent_sequence(self):
        cache = PagedKVCache(num_blocks=10)
        k = np.random.randn(32, 128).astype(np.float16)
        v = np.random.randn(32, 128).astype(np.float16)

        with pytest.raises(ValueError, match="not registered"):
            cache.append_kv(99, k, v)

    def test_repr(self):
        cache = PagedKVCache(num_blocks=100)
        repr_str = repr(cache)
        assert "PagedKVCache" in repr_str
        assert "blocks=" in repr_str
        assert "sequences=" in repr_str


class TestPagedKVCacheMultimodal:
    def test_init_multimodal(self):
        cache = PagedKVCache(num_blocks=100, use_multimodal=True)
        assert cache.use_multimodal
        from metal_marlin.paged.allocator import MultimodalBlockAllocator

        assert isinstance(cache.allocator, MultimodalBlockAllocator)

    def test_add_sequence_multimodal(self):
        cache = PagedKVCache(num_blocks=10, use_multimodal=True)
        success = cache.add_sequence(0)
        assert success
        assert cache.num_sequences == 1

        # Verify sequence registered with allocator
        modality = cache.allocator.get_sequence_modality(0)
        assert modality is not None
        assert modality.seq_id == 0
