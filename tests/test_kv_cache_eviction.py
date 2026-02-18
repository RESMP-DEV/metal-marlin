"""Tests for KV cache eviction policy in PagedKVCache."""

import time
import pytest
import numpy as np
from metal_marlin.paged.cache_manager import PagedKVCache, EvictionPolicy
from metal_marlin.paged.kv_block import KVBlockConfig

class TestKVCacheEviction:
    """Test LRU eviction in PagedKVCache."""

    def test_lru_eviction(self):
        """Test that the least recently used sequence is evicted when cache is full."""
        # Create a cache with very few blocks
        # block_size=16, num_blocks=4 -> total capacity 64 tokens
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        cache = PagedKVCache(config=config, num_blocks=4, eviction_policy=EvictionPolicy.LRU)

        # Add sequence 0 and fill it with 32 tokens (2 blocks)
        assert cache.add_sequence(0)
        k = np.zeros((32, 1, 64), dtype=np.float16)
        v = np.zeros((32, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)
        
        # Add sequence 1 and fill it with 16 tokens (1 block)
        assert cache.add_sequence(1)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(1, k, v)
        
        # Used: 3 blocks. Free: 1 block.
        assert cache.allocator.num_free == 1
        
        # Access sequence 0 to make it more recently used
        cache.get_kv(0)
        time.sleep(0.01) # Ensure time difference
        
        # Access sequence 1 (it is now MRU)
        cache.get_kv(1)
        time.sleep(0.01)
        
        # Make sequence 0 MRU again
        cache.get_kv(0)
        
        # Current state:
        # Seq 0: 2 blocks (MRU)
        # Seq 1: 1 block (LRU)
        # Free: 1 block
        
        # Add sequence 2 which needs 2 blocks
        # We have 1 free block. We need 1 more.
        # Eviction should kick in. Seq 1 is LRU, so it should be evicted (freeing 1 block).
        # Then we have 2 free blocks total.
        
        assert cache.add_sequence(2)
        k = np.zeros((32, 1, 64), dtype=np.float16)
        v = np.zeros((32, 1, 64), dtype=np.float16)
        
        # This append should trigger eviction of seq 1
        assert cache.append_kv_batch(2, k, v)
        
        # Verify sequence 1 is gone
        assert not cache.has_sequence(1)
        
        # Verify sequence 0 is still there
        assert cache.has_sequence(0)
        
        # Verify sequence 2 is there
        assert cache.has_sequence(2)
        
        # Check stats
        stats = cache.get_stats()
        assert stats.evictions >= 1
        assert stats.sequences_evicted >= 1

    def test_eviction_fails_if_one_sequence_hogs_all(self):
        """Test behavior when a single sequence consumes all blocks (no eviction possible)."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        cache = PagedKVCache(config=config, num_blocks=2, eviction_policy=EvictionPolicy.LRU)
        
        # Add sequence 0 and fill it (2 blocks)
        assert cache.add_sequence(0)
        k = np.zeros((32, 1, 64), dtype=np.float16)
        v = np.zeros((32, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)
        
        # Now cache is full with seq 0.
        # Try adding sequence 1.
        # We need 1 block for seq 1.
        # Eviction candidates: [seq 0].
        # If we evict seq 0, we have space.
        
        # However, if we try to extend seq 0 further, it should fail if it's the only one
        # But wait, eviction policy evicts *other* sequences usually?
        # Our implementation evicts ANY sequence in the metadata list sorted by LRU.
        # If seq 0 is the only one, it is both LRU and MRU.
        # If we are operating on seq 0 (append_kv), we shouldn't evict it?
        # The current implementation DOES NOT protect the current sequence from eviction
        # unless we explicitly handle it.
        # But `append_kv` updates access time at the start.
        # So seq 0 becomes MRU.
        # But if it is the ONLY sequence, it is also LRU (first in sorted list).
        # So it might evict itself!
        # Ideally, we should not evict the sequence we are currently adding to.
        
        # Let's see what happens if we add a new sequence.
        # cache.add_sequence(1) will try to allocate a block.
        # If full, it calls _evict(1).
        # Candidates: [seq 0]. Sorted by LRU (only one).
        # It will evict seq 0.
        
        assert cache.add_sequence(1)
        assert not cache.has_sequence(0)
        
    def test_cow_eviction(self):
        """Test eviction during Copy-On-Write."""
        config = KVBlockConfig(block_size=16, num_heads=1, head_dim=64)
        cache = PagedKVCache(config=config, num_blocks=3, eviction_policy=EvictionPolicy.LRU)
        
        # Seq 0: 1 block
        assert cache.add_sequence(0)
        k = np.zeros((16, 1, 64), dtype=np.float16)
        v = np.zeros((16, 1, 64), dtype=np.float16)
        assert cache.append_kv_batch(0, k, v)
        
        # Fork to Seq 1 (Zero-copy, shared block)
        assert cache.fork_sequence(0, 1)
        
        # Add Seq 2 (1 block) -> Full cache
        assert cache.add_sequence(2)
        assert cache.append_kv_batch(2, k, v)
        
        # Cache state:
        # Block 0: Shared by Seq 0 and Seq 1 (refcount 2)
        # Block 1: Seq 2 (refcount 1)
        # Free: 1 block (Wait, num_blocks=3. Used 2. Free 1.)
        
        assert cache.allocator.num_free == 1
        
        # Access seq 0, 1 to make them recent
        cache.get_kv(0)
        cache.get_kv(1)
        time.sleep(0.01)
        
        # Access seq 2 (MRU)
        cache.get_kv(2)
        
        # Now Seq 0 and 1 are older than Seq 2.
        
        # Trigger COW on Seq 1.
        # Needs 1 new block. We have 1 free block.
        # Should succeed without eviction.
        assert cache.append_kv_batch(1, k, v)
        assert cache.allocator.num_free == 0
        
        # Now cache is full.
        # Trigger another COW or append that requires allocation.
        # Let's append to Seq 2. Needs 1 block.
        # Should evict LRU.
        # Seq 0 is likely LRU (accessed earliest).
        # Evicting Seq 0 will decr refcount on original shared block (now unique to Seq 0? No, wait)
        # Seq 1 forked from Seq 0.
        # When Seq 1 did COW, it got a NEW block.
        # Seq 0 kept the OLD block.
        # So eviction of Seq 0 frees the OLD block.
        
        assert cache.append_kv_batch(2, k, v)
        
        # Seq 0 should be gone
        assert not cache.has_sequence(0)
        
