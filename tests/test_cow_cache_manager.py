"""Test copy-on-write prompt sharing in cache manager.

Standalone test that doesn't require torch/metal dependencies.
"""

import numpy as np
import pytest
from metal_marlin.paged.cache_manager import PagedKVCache

from metal_marlin.paged.allocator import BlockAllocator
from metal_marlin.paged.kv_block import KVBlockConfig


def test_prompt_sharing_basic():
    """Test basic prompt sharing between two sequences."""
    config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
    cache = PagedKVCache(config, num_blocks=16)
    
    # Create first sequence with prompt
    prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    assert cache.add_sequence_with_prompt(seq_id=0, prompt_tokens=prompt_tokens)
    
    # Add some KV data
    for _ in range(8):
        k = np.random.randn(2, 8).astype(np.float16)
        v = np.random.randn(2, 8).astype(np.float16)
        assert cache.append_kv(seq_id=0, key=k, value=v)
    
    # Create second sequence sharing the prompt
    assert cache.add_sequence_with_prompt(seq_id=1, prompt_tokens=prompt_tokens)
    
    # Check statistics
    stats = cache.get_stats()
    assert stats.prompt_cache_hits == 1, "Should have one cache hit"
    assert stats.active_sequences == 2
    
    # Verify blocks are shared (refcount > 1)
    shared_blocks = cache.get_shared_blocks(seq_id=1)
    assert len(shared_blocks) > 0, "Sequences should share blocks"


def test_cow_on_divergence():
    """Test copy-on-write when sequences diverge."""
    config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
    cache = PagedKVCache(config, num_blocks=16)
    
    # Create parent sequence
    prompt_tokens = [1, 2, 3, 4]
    assert cache.add_sequence_with_prompt(seq_id=0, prompt_tokens=prompt_tokens)
    
    # Fill first block
    for _ in range(4):
        k = np.ones((2, 8), dtype=np.float16)
        v = np.ones((2, 8), dtype=np.float16)
        assert cache.append_kv(seq_id=0, key=k, value=v)
    
    # Fork sequence (share blocks)
    assert cache.fork_sequence(src_id=0, dst_id=1)
    
    # Check initial shared state
    shared_before = cache.get_shared_blocks(seq_id=1)
    assert len(shared_before) > 0
    
    # Write to child - should trigger COW
    k_new = np.ones((2, 8), dtype=np.float16) * 2
    v_new = np.ones((2, 8), dtype=np.float16) * 2
    assert cache.append_kv(seq_id=1, key=k_new, value=v_new)
    
    # Check COW happened
    stats = cache.get_stats()
    assert stats.cow_operations > 0, "Should have triggered COW"
    
    # Verify data diverged
    k0, v0 = cache.get_kv(seq_id=0)
    k1, v1 = cache.get_kv(seq_id=1)
    assert k1.shape[0] > k0.shape[0], "Child should have more tokens"


def test_share_prompt_blocks_batch():
    """Test sharing prompt across multiple sequences."""
    config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
    cache = PagedKVCache(config, num_blocks=32)
    
    # Create source sequence
    assert cache.add_sequence(seq_id=0)
    
    # Fill two blocks (8 tokens)
    for _ in range(8):
        k = np.random.randn(2, 8).astype(np.float16)
        v = np.random.randn(2, 8).astype(np.float16)
        assert cache.append_kv(seq_id=0, key=k, value=v)
    
    # Share with 4 new sequences
    dst_ids = [1, 2, 3, 4]
    results = cache.share_prompt_blocks(
        src_seq_id=0,
        dst_seq_ids=dst_ids,
        num_prefix_tokens=8,
    )
    
    assert all(results), "All shares should succeed"
    
    # Verify all sequences share blocks
    for seq_id in dst_ids:
        shared = cache.get_shared_blocks(seq_id)
        assert len(shared) == 2, f"Seq {seq_id} should share 2 blocks"
    
    # Check statistics
    stats = cache.get_stats()
    assert stats.shared_prompt_blocks >= 8, "Should track shared blocks"


def test_prompt_cache_invalidation():
    """Test that stale prompt cache entries are cleaned up."""
    config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
    cache = PagedKVCache(config, num_blocks=16)
    
    # Create and cache first sequence
    prompt_tokens = [1, 2, 3, 4]
    assert cache.add_sequence_with_prompt(seq_id=0, prompt_tokens=prompt_tokens)
    
    # Remove sequence
    cache.remove_sequence(seq_id=0)
    
    # Try to create new sequence with same prompt
    # Should not hit cache (source sequence gone)
    assert cache.add_sequence_with_prompt(seq_id=1, prompt_tokens=prompt_tokens)
    
    stats = cache.get_stats()
    assert stats.prompt_cache_hits == 0, "Stale cache should not hit"


def test_cow_preserves_block_data():
    """Verify COW correctly copies block data."""
    config = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
    cache = PagedKVCache(config, num_blocks=16)
    
    # Create parent with known data
    assert cache.add_sequence(seq_id=0)
    
    # Fill with predictable pattern
    for i in range(4):
        k = np.full((2, 8), i, dtype=np.float16)
        v = np.full((2, 8), i * 10, dtype=np.float16)
        assert cache.append_kv(seq_id=0, key=k, value=v)
    
    # Fork
    assert cache.fork_sequence(src_id=0, dst_id=1)
    
    # Verify data matches before divergence
    k0, v0 = cache.get_kv(seq_id=0)
    k1, v1 = cache.get_kv(seq_id=1)
    
    assert np.allclose(k0, k1), "Keys should match before divergence"
    assert np.allclose(v0, v1), "Values should match before divergence"
    
    # Write to child
    k_new = np.full((2, 8), 99, dtype=np.float16)
    v_new = np.full((2, 8), 999, dtype=np.float16)
    assert cache.append_kv(seq_id=1, key=k_new, value=v_new)
    
    # Verify parent unchanged
    k0_after, v0_after = cache.get_kv(seq_id=0)
    assert np.allclose(k0, k0_after), "Parent should be unchanged"
    assert np.allclose(v0, v0_after), "Parent should be unchanged"
    
    # Verify child has new data
    k1_after, v1_after = cache.get_kv(seq_id=1)
    assert k1_after[-1, 0, 0] == 99, "Child should have new key"
    assert v1_after[-1, 0, 0] == 999, "Child should have new value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
