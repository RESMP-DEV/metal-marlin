"""Tests for prompt/prefix caching functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from metal_marlin._compat import HAS_TORCH, torch
from metal_marlin.cache.prompt_cache import (
    CacheMetrics,
    PrefixCache,
    PrefixCacheConfig,
    PrefixMatch,
    RadixPrefixCache,
    StorageTier,
    hash_prefix,
    hash_tokens,
)


class TestHashFunctions:
    """Test token hashing functions."""

    def test_hash_tokens_deterministic(self):
        """Same tokens produce same hash."""
        tokens = [1, 2, 3, 4, 5]
        h1 = hash_tokens(tokens)
        h2 = hash_tokens(tokens)
        assert h1 == h2

    def test_hash_tokens_different_input(self):
        """Different tokens produce different hash."""
        tokens1 = [1, 2, 3, 4, 5]
        tokens2 = [1, 2, 3, 4, 6]
        h1 = hash_tokens(tokens1)
        h2 = hash_tokens(tokens2)
        assert h1 != h2

    def test_hash_tokens_algorithm_md5(self):
        """MD5 hash algorithm works."""
        tokens = [1, 2, 3, 4, 5]
        h = hash_tokens(tokens, algorithm="md5")
        assert len(h) == 32  # MD5 produces 32 hex chars

    def test_hash_tokens_algorithm_sha256(self):
        """SHA256 hash algorithm works."""
        tokens = [1, 2, 3, 4, 5]
        h = hash_tokens(tokens, algorithm="sha256")
        assert len(h) == 64  # SHA256 produces 64 hex chars

    def test_hash_prefix_block_aligned(self):
        """Hash prefix produces one hash per complete block."""
        tokens = list(range(48))  # 3 blocks of 16
        block_size = 16
        hashes = hash_prefix(tokens, block_size)
        assert len(hashes) == 3

    def test_hash_prefix_partial_block_ignored(self):
        """Partial blocks at the end are not hashed."""
        tokens = list(range(40))  # 2 complete blocks + 8 extra
        block_size = 16
        hashes = hash_prefix(tokens, block_size)
        assert len(hashes) == 2

    def test_hash_prefix_chain_dependency(self):
        """Block hashes depend on previous blocks (chain hashing)."""
        # Two sequences with same first block but different second
        tokens1 = list(range(32))
        tokens2 = list(range(16)) + list(range(100, 116))

        block_size = 16
        hashes1 = hash_prefix(tokens1, block_size)
        hashes2 = hash_prefix(tokens2, block_size)

        # First block same
        assert hashes1[0] == hashes2[0]
        # Second block different (even though chain hashing)
        assert hashes1[1] != hashes2[1]

    def test_hash_prefix_shared_prefix(self):
        """Shared prefixes have identical hash chains up to divergence."""
        # System prompt + different user messages
        system_prompt = list(range(32))  # 2 blocks
        user1 = list(range(200, 216))  # 1 block
        user2 = list(range(300, 316))  # 1 block

        seq1 = system_prompt + user1
        seq2 = system_prompt + user2

        block_size = 16
        hashes1 = hash_prefix(seq1, block_size)
        hashes2 = hash_prefix(seq2, block_size)

        # First 2 blocks (system prompt) should match
        assert hashes1[0] == hashes2[0]
        assert hashes1[1] == hashes2[1]
        # Third block (user message) should differ
        assert hashes1[2] != hashes2[2]


class TestPrefixCacheConfig:
    """Test cache configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = PrefixCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        assert config.block_size == 16
        assert config.max_gpu_blocks == 512
        assert config.max_ram_blocks == 2048

    def test_bytes_per_block(self):
        """Bytes per block calculation is correct."""
        config = PrefixCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=16,
        )
        # 2 (K+V) * 16 (tokens) * 32 (heads) * 128 (dim) * 2 (bytes) * 32 (layers)
        expected = 2 * 16 * 32 * 128 * 2 * 32
        assert config.bytes_per_block == expected


class TestPrefixCache:
    """Test basic prefix cache operations."""

    @pytest.fixture
    def simple_config(self) -> PrefixCacheConfig:
        """Simple cache config for testing."""
        return PrefixCacheConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            block_size=16,
            max_gpu_blocks=10,
            max_ram_blocks=20,
        )

    @pytest.fixture
    def cache(self, simple_config: PrefixCacheConfig) -> PrefixCache:
        """Create a prefix cache for testing."""
        return PrefixCache(simple_config)

    def test_cache_creation(self, cache: PrefixCache):
        """Cache is created in empty state."""
        assert len(cache._gpu_blocks) == 0
        assert len(cache._ram_blocks) == 0
        assert cache.metrics.hits == 0

    def test_match_empty_cache(self, cache: PrefixCache):
        """Empty cache returns no matches."""
        tokens = list(range(32))
        match = cache.match_prefix(tokens)
        assert match.num_matched_tokens == 0
        assert match.num_matched_blocks == 0
        assert cache.metrics.misses == 1

    def test_match_short_input(self, cache: PrefixCache):
        """Input shorter than block size returns no match."""
        tokens = list(range(10))  # < 16 block size
        match = cache.match_prefix(tokens)
        assert match.num_matched_tokens == 0
        assert cache.metrics.misses == 1

    def test_store_and_match(self, cache: PrefixCache, simple_config: PrefixCacheConfig):
        """Store blocks and retrieve them."""
        tokens = list(range(32))  # 2 blocks

        # Create fake KV data for 2 blocks
        kv_blocks = []
        for _ in range(2):
            # Per layer: [2, block_size, num_heads, head_dim]
            layer_data = [
                np.random.randn(2, 16, 8, 64).astype(np.float16)
                for _ in range(simple_config.num_layers)
            ]
            kv_blocks.append(layer_data)

        # Store
        stored_hashes = cache.store_prefix(tokens, kv_blocks)
        assert len(stored_hashes) == 2

        # Match same tokens
        match = cache.match_prefix(tokens)
        assert match.num_matched_tokens == 32
        assert match.num_matched_blocks == 2
        assert cache.metrics.hits == 1

    def test_partial_match(self, cache: PrefixCache, simple_config: PrefixCacheConfig):
        """Partial prefix match works correctly."""
        # Store 2 blocks
        tokens_short = list(range(32))
        kv_blocks = [
            [np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)] for _ in range(2)
        ]
        cache.store_prefix(tokens_short, kv_blocks)

        # Query with 3 blocks (only first 2 will match)
        tokens_long = list(range(48))
        match = cache.match_prefix(tokens_long)

        assert match.num_matched_blocks == 2
        assert match.num_matched_tokens == 32
        assert cache.metrics.partial_hits == 1

    def test_get_blocks(self, cache: PrefixCache, simple_config: PrefixCacheConfig):
        """Retrieved blocks match stored data."""
        tokens = list(range(16))  # 1 block
        original_data = [[np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)]]
        stored_hashes = cache.store_prefix(tokens, original_data)

        retrieved = cache.get_blocks(stored_hashes)
        assert len(retrieved) == 1

        # Compare data
        for layer_idx in range(4):
            np.testing.assert_array_almost_equal(
                retrieved[0][layer_idx], original_data[0][layer_idx]
            )

    def test_release_blocks(self, cache: PrefixCache, simple_config: PrefixCacheConfig):
        """Releasing blocks decrements ref count."""
        tokens = list(range(16))
        kv_blocks = [[np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)]]
        stored_hashes = cache.store_prefix(tokens, kv_blocks)

        # Initial ref count is 1
        assert cache._gpu_blocks[stored_hashes[0]].ref_count == 1

        # Release
        cache.release_blocks(stored_hashes)
        assert cache._gpu_blocks[stored_hashes[0]].ref_count == 0

    def test_extend_from_cache_numpy(self, cache: PrefixCache):
        """extend_from_cache concatenates numpy arrays correctly."""
        # Prefix KV: 2 blocks
        prefix_kv = [
            [np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)],
            [np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)],
        ]

        # New KV: 10 tokens
        new_kv = [np.random.randn(2, 10, 8, 64).astype(np.float16) for _ in range(4)]

        combined = cache.extend_from_cache(prefix_kv, new_kv)

        # Should have 4 layers with seq_len = 32 + 10 = 42
        assert len(combined) == 4
        assert combined[0].shape == (2, 42, 8, 64)

    def test_memory_usage(self, cache: PrefixCache, simple_config: PrefixCacheConfig):
        """Memory usage tracking works."""
        tokens = list(range(32))
        kv_blocks = [
            [np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)] for _ in range(2)
        ]
        cache.store_prefix(tokens, kv_blocks)

        usage = cache.memory_usage()
        assert usage["gpu"] > 0
        assert usage["ram"] == 0
        assert usage["disk_blocks"] == 0


class TestCacheEviction:
    """Test cache eviction policies."""

    @pytest.fixture
    def small_cache(self) -> PrefixCache:
        """Cache with very limited capacity for eviction testing."""
        config = PrefixCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            block_size=8,
            max_gpu_blocks=2,  # Very small
            max_ram_blocks=4,
        )
        return PrefixCache(config)

    def test_evict_to_ram(self, small_cache: PrefixCache):
        """Blocks evict from GPU to RAM when full."""
        # Store 3 blocks (exceeds max_gpu_blocks=2)
        for i in range(3):
            tokens = list(range(i * 100, i * 100 + 8))
            kv = [[np.random.randn(2, 8, 4, 32).astype(np.float16) for _ in range(2)]]
            small_cache.store_prefix(tokens, kv)

        # 2 in GPU, 1 evicted to RAM
        assert len(small_cache._gpu_blocks) == 2
        assert len(small_cache._ram_blocks) == 1
        assert small_cache.metrics.evictions == 1
        assert small_cache.metrics.gpu_to_ram_moves == 1

    def test_evict_to_disk(self):
        """Blocks evict from RAM to disk when full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PrefixCacheConfig(
                num_layers=2,
                num_heads=4,
                head_dim=32,
                block_size=8,
                max_gpu_blocks=1,
                max_ram_blocks=1,
                disk_cache_path=Path(tmpdir),
            )
            cache = PrefixCache(config)

            # Store 3 blocks (GPU=1, RAM=1, disk=1)
            for i in range(3):
                tokens = list(range(i * 100, i * 100 + 8))
                kv = [[np.random.randn(2, 8, 4, 32).astype(np.float16) for _ in range(2)]]
                cache.store_prefix(tokens, kv)

            assert len(cache._gpu_blocks) == 1
            assert len(cache._ram_blocks) == 1
            assert len(cache._disk_index) == 1
            assert cache.metrics.ram_to_disk_moves == 1

    def test_load_from_disk(self):
        """Blocks can be loaded back from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PrefixCacheConfig(
                num_layers=2,
                num_heads=4,
                head_dim=32,
                block_size=8,
                max_gpu_blocks=1,
                max_ram_blocks=1,
                disk_cache_path=Path(tmpdir),
            )
            cache = PrefixCache(config)

            # Store first block (will end up on disk after 2 more)
            tokens1 = list(range(8))
            kv1 = [[np.random.randn(2, 8, 4, 32).astype(np.float16) for _ in range(2)]]
            hashes1 = cache.store_prefix(tokens1, kv1)
            original_data = kv1[0][0].copy()

            # Store 2 more to push first to disk
            for i in range(1, 3):
                tokens = list(range(i * 100, i * 100 + 8))
                kv = [[np.random.randn(2, 8, 4, 32).astype(np.float16) for _ in range(2)]]
                cache.store_prefix(tokens, kv)

            # First block should be on disk
            assert hashes1[0] in cache._disk_index

            # Retrieve it (should load from disk)
            retrieved = cache.get_blocks(hashes1)
            np.testing.assert_array_almost_equal(retrieved[0][0], original_data)
            assert cache.metrics.disk_loads == 1


class TestRadixPrefixCache:
    """Test radix tree-based prefix cache."""

    @pytest.fixture
    def radix_cache(self) -> RadixPrefixCache:
        """Create a radix prefix cache for testing."""
        config = PrefixCacheConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            block_size=16,
            max_gpu_blocks=20,
        )
        return RadixPrefixCache(config)

    def test_radix_store_and_match(self, radix_cache: RadixPrefixCache):
        """Radix cache stores and matches prefixes."""
        tokens = list(range(32))
        kv_blocks = [
            [np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)] for _ in range(2)
        ]
        radix_cache.store_prefix(tokens, kv_blocks)

        match = radix_cache.match_prefix(tokens)
        assert match.num_matched_blocks == 2

    def test_radix_partial_match(self, radix_cache: RadixPrefixCache):
        """Radix cache handles partial matches."""
        # Store system prompt
        system = list(range(32))
        kv_system = [
            [np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)] for _ in range(2)
        ]
        radix_cache.store_prefix(system, kv_system)

        # Query with system + user (only system matches)
        query = system + list(range(100, 116))
        match = radix_cache.match_prefix(query)

        assert match.num_matched_blocks == 2  # Only system blocks
        assert match.num_matched_tokens == 32

    def test_radix_clear(self, radix_cache: RadixPrefixCache):
        """Clear removes radix tree nodes."""
        tokens = list(range(32))
        kv_blocks = [
            [np.random.randn(2, 16, 8, 64).astype(np.float16) for _ in range(4)] for _ in range(2)
        ]
        radix_cache.store_prefix(tokens, kv_blocks)
        radix_cache.clear()

        match = radix_cache.match_prefix(tokens)
        assert match.num_matched_blocks == 0
        assert len(radix_cache._radix_root) == 0


class TestCacheMetrics:
    """Test cache metrics tracking."""

    def test_hit_rate_calculation(self):
        """Hit rate calculation is correct."""
        metrics = CacheMetrics(hits=80, misses=10, partial_hits=10)
        assert metrics.hit_rate == 0.8

    def test_hit_rate_empty(self):
        """Hit rate is 0 when no accesses."""
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0

    def test_metrics_reset(self):
        """Reset clears all metrics."""
        metrics = CacheMetrics(hits=10, misses=5, evictions=3)
        metrics.reset()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0


class TestThreadSafety:
    """Test thread-safe cache operations."""

    def test_concurrent_access(self):
        """Cache handles concurrent access."""
        import threading

        config = PrefixCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            block_size=8,
            max_gpu_blocks=100,
        )
        cache = PrefixCache(config)

        errors = []

        def store_worker(thread_id: int):
            try:
                for i in range(10):
                    tokens = list(range(thread_id * 1000 + i * 100, thread_id * 1000 + i * 100 + 8))
                    kv = [[np.random.randn(2, 8, 4, 32).astype(np.float16) for _ in range(2)]]
                    cache.store_prefix(tokens, kv)
            except Exception as e:
                errors.append(e)

        def match_worker(thread_id: int):
            try:
                for i in range(10):
                    tokens = list(range(thread_id * 1000 + i * 100, thread_id * 1000 + i * 100 + 8))
                    cache.match_prefix(tokens)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=store_worker, args=(i,)))
            threads.append(threading.Thread(target=match_worker, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestTorchIntegration:
    """Test PyTorch integration when available."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        return HAS_TORCH

    def test_extend_from_cache_torch(self, torch_available):
        """extend_from_cache works with PyTorch tensors."""
        if not torch_available or torch is None:
            pytest.skip("PyTorch not available")

        config = PrefixCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            block_size=8,
        )
        cache = PrefixCache(config)

        # PyTorch prefix KV
        prefix_kv = [
            [torch.randn(2, 8, 4, 32, dtype=torch.float16) for _ in range(2)],
        ]

        # PyTorch new KV
        new_kv = [torch.randn(2, 4, 4, 32, dtype=torch.float16) for _ in range(2)]

        # Custom concat function for PyTorch tensors
        def torch_concat(a, b):
            return torch.cat([a, b], dim=1)

        combined = cache.extend_from_cache(prefix_kv, new_kv, concat_fn=torch_concat)

        assert len(combined) == 2
        assert combined[0].shape == (2, 12, 4, 32)
        assert isinstance(combined[0], torch.Tensor)

    def test_store_and_retrieve_torch_tensors(self, torch_available):
        """Store PyTorch tensors (converted to numpy) and retrieve them."""
        if not torch_available or torch is None:
            pytest.skip("PyTorch not available")

        config = PrefixCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            block_size=8,
            max_gpu_blocks=10,
        )
        cache = PrefixCache(config)

        tokens = list(range(8))

        # Create KV data as PyTorch tensors, then convert to numpy for storage
        # (The cache internally stores numpy arrays)
        original_tensors = [torch.randn(2, 8, 4, 32, dtype=torch.float16) for _ in range(2)]
        kv_numpy = [[t.numpy() for t in original_tensors]]

        stored_hashes = cache.store_prefix(tokens, kv_numpy)
        retrieved = cache.get_blocks(stored_hashes)

        assert len(retrieved) == 1
        for layer_idx in range(2):
            np.testing.assert_array_almost_equal(
                retrieved[0][layer_idx], original_tensors[layer_idx].numpy()
            )

    def test_torch_mps_compatibility(self, torch_available):
        """Test that tensors on MPS device can be converted for cache use."""
        if not torch_available or torch is None:
            pytest.skip("PyTorch not available")

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        config = PrefixCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            block_size=8,
        )
        cache = PrefixCache(config)

        # Create tensor on MPS
        mps_tensor = torch.randn(2, 8, 4, 32, dtype=torch.float16, device="mps")

        # Convert to CPU numpy for storage
        cpu_numpy = mps_tensor.cpu().numpy()

        tokens = list(range(8))
        kv_blocks = [[[cpu_numpy] * 2]]

        # Should store without error
        stored_hashes = cache.store_prefix(tokens, kv_blocks)
        assert len(stored_hashes) == 1
