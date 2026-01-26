"""Tests for multimodal KV cache extensions.

Tests the VLM-specific KV cache features:
- MultimodalBlockAllocator with modality tracking
- Prefix caching for repeated image contexts
- VisionEncoderCache for encoder output caching
- Dynamic image token count handling
"""

from __future__ import annotations

import pytest

from metal_marlin.paged import (
    ImageRegion,
    MultimodalBlockAllocator,
    MultimodalBlockState,
    SequenceModality,
    TokenModality,
    VisionEncoderCache,
)


class TestTokenModality:
    """Tests for TokenModality enum."""

    def test_enum_values(self):
        assert TokenModality.TEXT.value == "text"
        assert TokenModality.IMAGE.value == "image"
        assert TokenModality.CROSS_ATTENTION.value == "cross_attention"


class TestMultimodalBlockState:
    """Tests for MultimodalBlockState dataclass."""

    def test_defaults(self):
        state = MultimodalBlockState(block_idx=0)
        assert state.block_idx == 0
        assert state.ref_count == 0
        assert state.is_free is True
        assert state.modality == TokenModality.TEXT
        assert state.content_hash is None
        assert state.valid_start == 0
        assert state.valid_end == 0

    def test_with_image_modality(self):
        state = MultimodalBlockState(
            block_idx=5,
            ref_count=2,
            is_free=False,
            modality=TokenModality.IMAGE,
            content_hash="abc123",
            valid_start=0,
            valid_end=16,
        )
        assert state.modality == TokenModality.IMAGE
        assert state.content_hash == "abc123"


class TestImageRegion:
    """Tests for ImageRegion dataclass."""

    def test_defaults(self):
        region = ImageRegion(
            start_pos=10,
            num_tokens=576,  # Typical 24x24 patch grid
            image_hash="deadbeef",
        )
        assert region.start_pos == 10
        assert region.num_tokens == 576
        assert region.image_hash == "deadbeef"
        assert region.block_indices == []

    def test_with_blocks(self):
        region = ImageRegion(
            start_pos=0,
            num_tokens=32,
            image_hash="hash123",
            block_indices=[0, 1],
        )
        assert region.block_indices == [0, 1]


class TestSequenceModality:
    """Tests for SequenceModality tracking."""

    def test_init(self):
        sm = SequenceModality(seq_id=42)
        assert sm.seq_id == 42
        assert sm.image_regions == []
        assert sm.text_ranges == []

    def test_add_image_region(self):
        sm = SequenceModality(seq_id=1)
        sm.add_image_region(
            start_pos=5,
            num_tokens=100,
            image_hash="img1",
            block_indices=[0, 1, 2, 3, 4, 5, 6],
        )
        assert len(sm.image_regions) == 1
        assert sm.image_regions[0].start_pos == 5
        assert sm.image_regions[0].num_tokens == 100

    def test_add_text_range(self):
        sm = SequenceModality(seq_id=1)
        sm.add_text_range(0, 10)
        sm.add_text_range(110, 150)
        assert sm.text_ranges == [(0, 10), (110, 150)]

    def test_total_tokens(self):
        sm = SequenceModality(seq_id=1)
        sm.add_text_range(0, 5)  # 5 tokens
        sm.add_image_region(5, 100, "img", [])  # 100 tokens
        sm.add_text_range(105, 125)  # 20 tokens

        assert sm.total_text_tokens == 25
        assert sm.total_image_tokens == 100

    def test_get_modality_at(self):
        sm = SequenceModality(seq_id=1)
        sm.add_text_range(0, 10)
        sm.add_image_region(10, 50, "img", [])
        sm.add_text_range(60, 100)

        # Text positions
        assert sm.get_modality_at(0) == TokenModality.TEXT
        assert sm.get_modality_at(5) == TokenModality.TEXT
        assert sm.get_modality_at(65) == TokenModality.TEXT

        # Image positions
        assert sm.get_modality_at(10) == TokenModality.IMAGE
        assert sm.get_modality_at(30) == TokenModality.IMAGE
        assert sm.get_modality_at(59) == TokenModality.IMAGE


class TestMultimodalBlockAllocator:
    """Tests for MultimodalBlockAllocator."""

    def test_init(self):
        alloc = MultimodalBlockAllocator(num_blocks=100, block_size=16)
        assert alloc.num_blocks == 100
        assert alloc.block_size == 16
        assert alloc.num_free == 100
        assert alloc.num_allocated == 0

    def test_allocate_text_block(self):
        alloc = MultimodalBlockAllocator(num_blocks=10)
        idx = alloc.allocate(modality=TokenModality.TEXT)

        assert idx is not None
        assert alloc.blocks[idx].modality == TokenModality.TEXT
        assert alloc.blocks[idx].is_free is False
        assert alloc.text_blocks_allocated == 1
        assert alloc.image_blocks_allocated == 0

    def test_allocate_image_block(self):
        alloc = MultimodalBlockAllocator(num_blocks=10)
        idx = alloc.allocate(
            modality=TokenModality.IMAGE,
            content_hash="test_hash",
        )

        assert idx is not None
        assert alloc.blocks[idx].modality == TokenModality.IMAGE
        assert alloc.blocks[idx].content_hash == "test_hash"
        assert alloc.image_blocks_allocated == 1
        assert alloc.text_blocks_allocated == 0

    def test_allocate_until_full(self):
        alloc = MultimodalBlockAllocator(num_blocks=5)
        indices = [alloc.allocate() for _ in range(5)]

        assert all(idx is not None for idx in indices)
        assert alloc.num_free == 0
        assert alloc.allocate() is None

    def test_free_block(self):
        alloc = MultimodalBlockAllocator(num_blocks=10)
        idx = alloc.allocate(modality=TokenModality.IMAGE)

        assert alloc.image_blocks_allocated == 1
        alloc.free(idx)

        assert alloc.blocks[idx].is_free is True
        assert alloc.blocks[idx].modality == TokenModality.TEXT  # Reset to default
        assert alloc.image_blocks_allocated == 0

    def test_allocate_image_blocks_fresh(self):
        alloc = MultimodalBlockAllocator(num_blocks=100, block_size=16)
        # 50 image tokens = ceil(50/16) = 4 blocks
        indices = alloc.allocate_image_blocks(num_tokens=50, image_hash="img1")

        assert indices is not None
        assert len(indices) == 4
        assert alloc.image_blocks_allocated == 4

        # Check valid_end for each block
        assert alloc.blocks[indices[0]].valid_end == 16
        assert alloc.blocks[indices[1]].valid_end == 16
        assert alloc.blocks[indices[2]].valid_end == 16
        assert alloc.blocks[indices[3]].valid_end == 2  # 50 - 48 = 2

    def test_prefix_cache_hit(self):
        alloc = MultimodalBlockAllocator(num_blocks=100, block_size=16)

        # First allocation
        indices1 = alloc.allocate_image_blocks(num_tokens=32, image_hash="shared_img")
        assert indices1 is not None
        initial_allocated = alloc.num_allocated

        # Second allocation with same hash should hit prefix cache
        indices2 = alloc.allocate_image_blocks(num_tokens=32, image_hash="shared_img")
        assert indices2 is not None

        # Should be the same blocks (prefix cache hit)
        assert indices1 == indices2
        assert alloc.prefix_cache_hits == 1

        # Refcounts should be incremented
        for idx in indices1:
            assert alloc.blocks[idx].ref_count == 2

        # No new blocks allocated
        assert alloc.num_allocated == initial_allocated

    def test_prefix_cache_different_images(self):
        alloc = MultimodalBlockAllocator(num_blocks=100, block_size=16)

        indices1 = alloc.allocate_image_blocks(num_tokens=16, image_hash="img_a")
        indices2 = alloc.allocate_image_blocks(num_tokens=16, image_hash="img_b")

        # Different images = different blocks
        assert indices1 != indices2
        assert alloc.prefix_cache_hits == 0

    def test_copy_on_write_preserves_modality(self):
        alloc = MultimodalBlockAllocator(num_blocks=100, block_size=16)

        # Allocate and share via prefix cache
        indices = alloc.allocate_image_blocks(num_tokens=16, image_hash="cow_test")
        _ = alloc.allocate_image_blocks(num_tokens=16, image_hash="cow_test")

        # Block has refcount 2
        assert alloc.blocks[indices[0]].ref_count == 2

        # COW should allocate new block with same modality
        new_idx = alloc.copy_on_write(indices[0])

        assert new_idx != indices[0]
        assert alloc.blocks[new_idx].modality == TokenModality.IMAGE
        assert alloc.blocks[indices[0]].ref_count == 1

    def test_sequence_modality_tracking(self):
        alloc = MultimodalBlockAllocator(num_blocks=100)

        # Register sequence
        alloc.register_sequence(seq_id=1)
        assert alloc.get_sequence_modality(1) is not None

        # Add modality info
        alloc.add_text_range(1, 0, 10)
        alloc.add_image_region(1, 10, 100, "img_hash", [0, 1, 2])
        alloc.add_text_range(1, 110, 150)

        sm = alloc.get_sequence_modality(1)
        assert sm.total_text_tokens == 50
        assert sm.total_image_tokens == 100

        # Unregister
        alloc.unregister_sequence(1)
        assert alloc.get_sequence_modality(1) is None

    def test_get_modality_mask(self):
        alloc = MultimodalBlockAllocator(num_blocks=100)
        alloc.register_sequence(1)
        alloc.add_image_region(1, 5, 10, "img", [])

        mask = alloc.get_modality_mask(1, total_len=20)

        assert len(mask) == 20
        assert all(m == TokenModality.TEXT for m in mask[:5])
        assert all(m == TokenModality.IMAGE for m in mask[5:15])
        assert all(m == TokenModality.TEXT for m in mask[15:])

    def test_get_stats(self):
        alloc = MultimodalBlockAllocator(num_blocks=100, block_size=16)

        alloc.allocate_image_blocks(32, "img1")
        alloc.allocate(modality=TokenModality.TEXT)
        alloc.allocate_image_blocks(32, "img1")  # Cache hit

        stats = alloc.get_stats()

        assert stats["num_blocks"] == 100
        assert stats["image_blocks"] == 2  # 32 tokens = 2 blocks
        assert stats["text_blocks"] == 1
        assert stats["prefix_cache_hits"] == 1
        assert stats["prefix_cache_entries"] == 1


class TestVisionEncoderCache:
    """Tests for VisionEncoderCache."""

    @pytest.fixture
    def mlx_available(self):
        """Check if MLX is available."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_init(self):
        cache = VisionEncoderCache(max_entries=50, max_memory_bytes=512 * 1024 * 1024)
        assert cache.max_entries == 50
        assert cache.max_memory_bytes == 512 * 1024 * 1024
        assert len(cache) == 0

    def test_compute_image_hash(self):
        hash1 = VisionEncoderCache.compute_image_hash(b"image data 1")
        hash2 = VisionEncoderCache.compute_image_hash(b"image data 1")
        hash3 = VisionEncoderCache.compute_image_hash(b"image data 2")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 64  # SHA-256 hex length

    def test_put_and_get(self, mlx_available):
        mx = mlx_available
        cache = VisionEncoderCache(max_entries=10, max_memory_bytes=10 * 1024 * 1024)

        tensor = mx.zeros((576, 4096))  # Typical vision encoder output
        image_hash = "test_hash_123"

        # Put
        assert cache.put(image_hash, tensor, num_tokens=576)
        assert image_hash in cache
        assert len(cache) == 1

        # Get
        result = cache.get(image_hash)
        assert result is not None
        retrieved_tensor, num_tokens = result
        assert num_tokens == 576
        assert retrieved_tensor.shape == tensor.shape

    def test_get_miss(self):
        cache = VisionEncoderCache()
        result = cache.get("nonexistent_hash")
        assert result is None

    def test_lru_eviction_by_count(self, mlx_available):
        mx = mlx_available
        cache = VisionEncoderCache(max_entries=3, max_memory_bytes=1024 * 1024 * 1024)

        # Add 3 entries
        for i in range(3):
            tensor = mx.zeros((10, 10))
            cache.put(f"hash_{i}", tensor, num_tokens=10)

        assert len(cache) == 3
        assert "hash_0" in cache

        # Access hash_1 and hash_2 to make hash_0 the LRU
        cache.get("hash_1")
        cache.get("hash_2")

        # Add 4th entry - should evict hash_0
        tensor = mx.zeros((10, 10))
        cache.put("hash_3", tensor, num_tokens=10)

        assert len(cache) == 3
        assert "hash_0" not in cache
        assert "hash_3" in cache

    def test_lru_eviction_by_memory(self, mlx_available):
        mx = mlx_available
        # 1MB limit
        cache = VisionEncoderCache(max_entries=100, max_memory_bytes=1024 * 1024)

        # Add entries until memory is exceeded
        # Each float32 tensor of shape (100, 100) = 40KB
        for i in range(30):  # ~1.2MB total
            tensor = mx.zeros((100, 100))
            cache.put(f"hash_{i}", tensor, num_tokens=100)

        # Should have evicted some entries
        assert len(cache) < 30

        stats = cache.get_stats()
        assert stats["memory_used_bytes"] <= cache.max_memory_bytes

    def test_remove(self, mlx_available):
        mx = mlx_available
        cache = VisionEncoderCache()

        tensor = mx.zeros((10, 10))
        cache.put("to_remove", tensor, num_tokens=10)
        assert "to_remove" in cache

        result = cache.remove("to_remove")
        assert result is True
        assert "to_remove" not in cache

        # Remove non-existent
        result = cache.remove("nonexistent")
        assert result is False

    def test_clear(self, mlx_available):
        mx = mlx_available
        cache = VisionEncoderCache()

        for i in range(5):
            tensor = mx.zeros((10, 10))
            cache.put(f"hash_{i}", tensor, num_tokens=10)

        assert len(cache) == 5

        cache.clear()
        assert len(cache) == 0

        stats = cache.get_stats()
        assert stats["memory_used_bytes"] == 0

    def test_get_stats(self, mlx_available):
        mx = mlx_available
        cache = VisionEncoderCache(max_entries=100, max_memory_bytes=10 * 1024 * 1024)

        tensor = mx.zeros((100, 100))
        cache.put("hash_1", tensor, num_tokens=100)
        cache.get("hash_1")
        cache.get("hash_1")

        stats = cache.get_stats()

        assert stats["num_entries"] == 1
        assert stats["max_entries"] == 100
        assert stats["memory_used_bytes"] > 0
        assert stats["total_accesses"] == 3  # 1 put + 2 gets

    def test_tensor_too_large(self, mlx_available):
        mx = mlx_available
        # Very small memory limit
        cache = VisionEncoderCache(max_entries=100, max_memory_bytes=100)

        # Try to cache a tensor larger than the limit
        tensor = mx.zeros((1000, 1000))  # ~4MB
        result = cache.put("too_large", tensor, num_tokens=100)

        assert result is False
        assert "too_large" not in cache


class TestMultimodalIntegration:
    """Integration tests for multimodal KV cache workflow."""

    @pytest.fixture
    def mlx_available(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_vlm_workflow(self, mlx_available):
        """Simulate a typical VLM inference workflow."""
        mx = mlx_available

        # Setup
        block_alloc = MultimodalBlockAllocator(num_blocks=1000, block_size=16)
        encoder_cache = VisionEncoderCache(max_entries=10)

        # Simulate processing: [TEXT] [IMAGE] [TEXT]
        seq_id = 1
        block_alloc.register_sequence(seq_id)

        # 1. Process system prompt text (5 tokens)
        [block_alloc.allocate(TokenModality.TEXT)]
        block_alloc.add_text_range(seq_id, 0, 5)

        # 2. Process image
        image_bytes = b"fake image data"
        image_hash = encoder_cache.compute_image_hash(image_bytes)

        # Check encoder cache (miss on first time)
        encoder_result = encoder_cache.get(image_hash)
        assert encoder_result is None

        # "Run" vision encoder and cache result
        encoder_output = mx.zeros((576, 4096))  # 576 image tokens
        encoder_cache.put(image_hash, encoder_output, num_tokens=576)

        # Allocate KV cache blocks for image tokens
        image_blocks = block_alloc.allocate_image_blocks(576, image_hash)
        assert image_blocks is not None
        block_alloc.add_image_region(seq_id, 5, 576, image_hash, image_blocks)

        # 3. Process user question (20 tokens)
        [
            block_alloc.allocate(TokenModality.TEXT) for _ in range(2)
        ]
        block_alloc.add_text_range(seq_id, 581, 601)

        # Verify modality tracking
        sm = block_alloc.get_sequence_modality(seq_id)
        assert sm.total_image_tokens == 576
        assert sm.total_text_tokens == 25

        # 4. Simulate second query with SAME image
        seq_id_2 = 2
        block_alloc.register_sequence(seq_id_2)

        # Encoder cache hit
        encoder_result = encoder_cache.get(image_hash)
        assert encoder_result is not None

        # Prefix cache hit for KV blocks
        image_blocks_2 = block_alloc.allocate_image_blocks(576, image_hash)
        assert image_blocks_2 == image_blocks  # Same blocks reused
        assert block_alloc.prefix_cache_hits == 1

        # Different question
        block_alloc.add_text_range(seq_id_2, 581, 610)  # 29 different tokens

        # Stats
        stats = block_alloc.get_stats()
        assert stats["prefix_cache_hits"] == 1
        assert stats["image_blocks"] == len(image_blocks)  # Shared, not doubled

    def test_multiple_images_per_sequence(self):
        """Test sequence with multiple interleaved images."""
        alloc = MultimodalBlockAllocator(num_blocks=500, block_size=16)
        alloc.register_sequence(1)

        # [TEXT][IMG1][TEXT][IMG2][TEXT]
        alloc.add_text_range(1, 0, 10)

        img1_blocks = alloc.allocate_image_blocks(100, "img_hash_1")
        alloc.add_image_region(1, 10, 100, "img_hash_1", img1_blocks)

        alloc.add_text_range(1, 110, 130)

        img2_blocks = alloc.allocate_image_blocks(200, "img_hash_2")
        alloc.add_image_region(1, 130, 200, "img_hash_2", img2_blocks)

        alloc.add_text_range(1, 330, 400)

        # Verify
        sm = alloc.get_sequence_modality(1)
        assert len(sm.image_regions) == 2
        assert sm.total_image_tokens == 300

        # Check modality mask
        mask = alloc.get_modality_mask(1, 400)
        assert mask[0] == TokenModality.TEXT
        assert mask[50] == TokenModality.IMAGE  # In first image
        assert mask[120] == TokenModality.TEXT  # Between images
        assert mask[200] == TokenModality.IMAGE  # In second image
        assert mask[350] == TokenModality.TEXT  # After images
