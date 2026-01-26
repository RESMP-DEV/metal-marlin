"""Tests for MLA (Multi-Head Latent Attention) paged KV cache.

MLA stores compressed latent representations instead of full K, V tensors,
providing ~8x memory reduction for long context inference.

Tests cover:
- MLACacheConfig memory calculations
- MLABlock latent storage and decompression
- MLABlockAllocator pool management and prefix caching
- mla_attention correctness
- Memory comparison utilities
"""

import pytest

from metal_marlin._compat import HAS_MLX
from metal_marlin.paged.mla_cache import (
    MLABlock,
    MLABlockAllocator,
    MLACacheConfig,
    compare_memory_usage,
)

if HAS_MLX:
    import mlx.core as mx

    from metal_marlin.paged.mla_cache import mla_attention

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="Requires MLX (Apple Silicon only)")


# =============================================================================
# MLACacheConfig Tests (no MLX required)
# =============================================================================


class TestMLACacheConfig:
    """Test configuration and memory calculations."""

    def test_defaults(self):
        cfg = MLACacheConfig()
        assert cfg.block_size == 16
        assert cfg.kv_lora_rank == 512
        assert cfg.num_kv_heads == 8
        assert cfg.head_dim == 128
        assert cfg.quantize_mode == "none"

    def test_latent_bytes_per_token_full_precision(self):
        cfg = MLACacheConfig(kv_lora_rank=512)
        # 512 * 2 bytes (fp16/bf16) = 1024 bytes
        assert cfg.latent_bytes_per_token == 1024

    def test_latent_bytes_per_token_fp8(self):
        cfg = MLACacheConfig(kv_lora_rank=512, quantize_mode="fp8")
        # 512 * 1 byte = 512 bytes
        assert cfg.latent_bytes_per_token == 512

    def test_latent_bytes_per_token_fp4(self):
        cfg = MLACacheConfig(kv_lora_rank=512, quantize_mode="fp4")
        # 512 * 0.5 bytes = 256 bytes
        assert cfg.latent_bytes_per_token == 256

    def test_standard_bytes_per_token(self):
        cfg = MLACacheConfig(num_kv_heads=8, head_dim=128)
        # 2 * 8 * 128 * 2 bytes = 4096 bytes
        assert cfg.standard_bytes_per_token == 4096

    def test_memory_savings_ratio(self):
        cfg = MLACacheConfig(kv_lora_rank=512, num_kv_heads=8, head_dim=128)
        # 4096 / 1024 = 4.0x
        assert cfg.memory_savings_ratio == 4.0

    def test_memory_savings_ratio_fp8(self):
        cfg = MLACacheConfig(
            kv_lora_rank=512, num_kv_heads=8, head_dim=128, quantize_mode="fp8"
        )
        # 4096 / 512 = 8.0x
        assert cfg.memory_savings_ratio == 8.0

    def test_memory_bytes(self):
        cfg = MLACacheConfig(block_size=16, kv_lora_rank=512)
        # 16 tokens * 1024 bytes/token = 16384 bytes
        assert cfg.memory_bytes == 16384

    def test_latent_shape(self):
        cfg = MLACacheConfig(block_size=8, kv_lora_rank=256)
        assert cfg.latent_shape == (8, 256)

    def test_decompressed_shape(self):
        cfg = MLACacheConfig(block_size=8, num_kv_heads=4, head_dim=64)
        assert cfg.decompressed_shape == (2, 8, 4, 64)

    def test_frozen(self):
        cfg = MLACacheConfig()
        with pytest.raises(Exception):
            cfg.block_size = 32  # type: ignore[misc]


# =============================================================================
# Memory Comparison Tests (no MLX required)
# =============================================================================


class TestCompareMemoryUsage:
    """Test memory comparison utility."""

    def test_basic_comparison(self):
        stats = compare_memory_usage(
            seq_len=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            kv_lora_rank=512,
        )

        # Standard: 2 * 4096 * 8 * 128 * 2 * 32 = 536,870,912 bytes = 512 MB
        assert pytest.approx(stats["standard_total_mb"], rel=0.01) == 512.0

        # MLA: 4096 * 512 * 2 * 32 = 134,217,728 bytes = 128 MB
        assert pytest.approx(stats["mla_total_mb"], rel=0.01) == 128.0

        # Savings: 512 / 128 = 4.0x
        assert pytest.approx(stats["savings_ratio"], rel=0.01) == 4.0

    def test_savings_per_layer(self):
        stats = compare_memory_usage(
            seq_len=4096,
            num_layers=1,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            kv_lora_rank=512,
        )

        # Standard per layer: 2 * 4096 * 32 * 128 * 2 = 67,108,864 = 64 MB
        assert pytest.approx(stats["standard_per_layer_mb"], rel=0.01) == 64.0

        # MLA per layer: 4096 * 512 * 2 = 4,194,304 = 4 MB
        assert pytest.approx(stats["mla_per_layer_mb"], rel=0.01) == 4.0

    def test_bytes_saved(self):
        stats = compare_memory_usage(
            seq_len=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            kv_lora_rank=512,
        )

        expected_saved = stats["standard_total_mb"] - stats["mla_total_mb"]
        assert pytest.approx(stats["bytes_saved_total_mb"], rel=0.01) == expected_saved


# =============================================================================
# MLABlock Tests (requires MLX)
# =============================================================================


@requires_mlx
class TestMLABlockLifecycle:
    """Test block lifecycle: allocation, reset, copy."""

    def test_init_unallocated(self):
        block = MLABlock()
        assert block.latents is None
        assert block.token_count == 0
        assert block.is_empty
        assert not block.is_full

    def test_allocate(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64)
        block = MLABlock(config=cfg)
        block.allocate()
        assert block.latents is not None
        assert block.latents.shape == (4, 64)

    def test_allocate_fp8(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64, quantize_mode="fp8")
        block = MLABlock(config=cfg)
        block.allocate()
        assert block.latents is not None
        assert block.latents.shape == (4, 64)
        assert block.latents.dtype == mx.uint8

    def test_allocate_fp4(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64, quantize_mode="fp4")
        block = MLABlock(config=cfg)
        block.allocate()
        assert block.latents is not None
        # Packed: 64 / 8 = 8 uint32s
        assert block.latents.shape == (4, 8)
        assert block.latents.dtype == mx.uint32

    def test_reset(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64)
        block = MLABlock(config=cfg)
        block.allocate()
        block.acquire()
        latent = mx.ones((64,), dtype=mx.bfloat16)
        block.append_latent(latent)
        block.set_prefix_hash(12345)

        block.reset()
        assert block.token_count == 0
        assert block.ref_count == 0
        assert block.prefix_hash is None
        assert block.latents is not None  # Still allocated


@requires_mlx
class TestMLABlockAppend:
    """Test latent append operations."""

    @pytest.fixture
    def small_block(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64)
        block = MLABlock(config=cfg)
        block.allocate()
        return block

    def test_append_single(self, small_block: MLABlock):
        latent = mx.ones((64,), dtype=mx.bfloat16)
        remaining = small_block.append_latent(latent)
        assert remaining == 3
        assert small_block.token_count == 1
        assert not small_block.is_full

    def test_append_until_full(self, small_block: MLABlock):
        for i in range(4):
            latent = mx.full((64,), float(i), dtype=mx.bfloat16)
            remaining = small_block.append_latent(latent)
            assert remaining == 3 - i
        assert small_block.is_full
        assert small_block.remaining == 0

    def test_append_overflow_raises(self, small_block: MLABlock):
        for _ in range(4):
            latent = mx.ones((64,), dtype=mx.bfloat16)
            small_block.append_latent(latent)
        with pytest.raises(RuntimeError, match="Block is full"):
            small_block.append_latent(mx.ones((64,), dtype=mx.bfloat16))

    def test_append_unallocated_raises(self):
        block = MLABlock()
        with pytest.raises(RuntimeError, match="not allocated"):
            block.append_latent(mx.ones((64,), dtype=mx.bfloat16))

    def test_append_preserves_values(self, small_block: MLABlock):
        l0 = mx.full((64,), 1.0, dtype=mx.bfloat16)
        l1 = mx.full((64,), 2.0, dtype=mx.bfloat16)

        small_block.append_latent(l0)
        small_block.append_latent(l1)

        latents = small_block.get_latents()
        mx.eval(latents)

        assert latents.shape == (2, 64)
        assert mx.allclose(latents[0], l0, atol=1e-3).item()
        assert mx.allclose(latents[1], l1, atol=1e-3).item()


@requires_mlx
class TestMLABlockBatchAppend:
    """Test batch latent append operations."""

    @pytest.fixture
    def block(self):
        cfg = MLACacheConfig(block_size=8, kv_lora_rank=64)
        block = MLABlock(config=cfg)
        block.allocate()
        return block

    def test_batch_append(self, block: MLABlock):
        latents = mx.ones((3, 64), dtype=mx.bfloat16)
        remaining = block.append_latent_batch(latents)
        assert remaining == 5
        assert block.token_count == 3

    def test_batch_overflow_raises(self, block: MLABlock):
        latents = mx.ones((9, 64), dtype=mx.bfloat16)
        with pytest.raises(RuntimeError, match="exceeds remaining"):
            block.append_latent_batch(latents)

    def test_batch_preserves_values(self, block: MLABlock):
        latents = mx.arange(3 * 64, dtype=mx.bfloat16).reshape(3, 64)

        block.append_latent_batch(latents)
        got = block.get_latents()
        mx.eval(got)

        assert got.shape == (3, 64)
        assert mx.allclose(got, latents, atol=1e-2).item()


@requires_mlx
class TestMLABlockDecompression:
    """Test latent decompression to K, V."""

    def test_decompress_basic(self):
        cfg = MLACacheConfig(
            block_size=4,
            kv_lora_rank=64,
            num_kv_heads=2,
            head_dim=32,
        )
        block = MLABlock(config=cfg)
        block.allocate()

        # Add some latents
        latent = mx.ones((64,), dtype=mx.bfloat16)
        block.append_latent(latent)
        block.append_latent(latent * 2)

        # Create a simple projection matrix
        # kv_b_proj: [2 * num_kv_heads * head_dim, kv_lora_rank]
        # = [2 * 2 * 32, 64] = [128, 64]
        kv_b_proj = mx.eye(64, dtype=mx.bfloat16)
        # Extend to [128, 64] by repeating
        kv_b_proj = mx.concatenate([kv_b_proj, kv_b_proj], axis=0)

        keys, values = block.decompress(kv_b_proj)
        mx.eval(keys, values)

        assert keys.shape == (2, 2, 32)
        assert values.shape == (2, 2, 32)

    def test_decompress_unallocated_raises(self):
        cfg = MLACacheConfig(kv_lora_rank=64, num_kv_heads=2, head_dim=32)
        block = MLABlock(config=cfg)
        kv_b_proj = mx.eye(64, dtype=mx.bfloat16)
        kv_b_proj = mx.concatenate([kv_b_proj, kv_b_proj], axis=0)

        with pytest.raises(RuntimeError, match="not allocated"):
            block.decompress(kv_b_proj)


@requires_mlx
class TestMLABlockQuantization:
    """Test FP8 and FP4 quantization modes."""

    def test_fp8_round_trip(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64, quantize_mode="fp8")
        block = MLABlock(config=cfg)
        block.allocate()

        # Random latent
        latent = mx.random.normal((64,), dtype=mx.float16) * 10
        block.append_latent(latent)

        recovered = block.get_latents()
        mx.eval(recovered)

        # FP8 has ~1% relative error for typical values
        rel_error = mx.abs(recovered[0] - latent) / (mx.abs(latent) + 1e-6)
        assert mx.mean(rel_error).item() < 0.05  # 5% mean relative error

    def test_fp4_round_trip(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64, quantize_mode="fp4")
        block = MLABlock(config=cfg)
        block.allocate()

        # Random latent (FP4 has limited range, keep values moderate)
        latent = mx.random.normal((64,), dtype=mx.float16) * 2
        block.append_latent(latent)

        recovered = block.get_latents()
        mx.eval(recovered)

        # FP4 has higher quantization error
        rel_error = mx.abs(recovered[0] - latent) / (mx.abs(latent) + 1e-6)
        assert mx.mean(rel_error).item() < 0.3  # 30% mean relative error (FP4 is coarse)


# =============================================================================
# MLABlockAllocator Tests (requires MLX)
# =============================================================================


@requires_mlx
class TestMLABlockAllocator:
    """Test block allocator pool management."""

    def test_allocate_free(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=10, config=config)

        assert allocator.num_free == 10
        assert allocator.num_allocated == 0

        idx = allocator.allocate()
        assert idx is not None
        assert allocator.num_free == 9
        assert allocator.num_allocated == 1

        allocator.free(idx)
        assert allocator.num_free == 10
        assert allocator.num_allocated == 0

    def test_allocate_exhaustion(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=3, config=config)

        indices = []
        for _ in range(3):
            idx = allocator.allocate()
            assert idx is not None
            indices.append(idx)

        # Pool exhausted
        assert allocator.allocate() is None

        # Free one
        allocator.free(indices[0])
        assert allocator.allocate() is not None

    def test_get_block(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=5, config=config)

        idx = allocator.allocate()
        block = allocator.get_block(idx)
        assert block is not None
        assert isinstance(block, MLABlock)
        assert block.latents is not None

    def test_copy_on_write_exclusive(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=5, config=config)

        idx = allocator.allocate()
        # Block has ref_count=1, already exclusive
        new_idx = allocator.copy_on_write(idx)
        assert new_idx == idx

    def test_copy_on_write_shared(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=5, config=config)

        idx = allocator.allocate()
        # Simulate sharing by incrementing ref count
        allocator.blocks[idx].ref_count = 2

        new_idx = allocator.copy_on_write(idx)
        assert new_idx != idx
        assert allocator.blocks[new_idx].ref_count == 1
        assert allocator.blocks[idx].ref_count == 1  # Decremented from 2


@requires_mlx
class TestMLABlockAllocatorPrefixCache:
    """Test prefix caching functionality."""

    def test_register_and_lookup_prefix(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=5, config=config)

        idx = allocator.allocate()
        prefix_hash = hash("system prompt prefix")

        allocator.register_prefix(idx, prefix_hash)
        assert allocator.num_cached_prefixes == 1

        # Lookup should return the same block and increment ref count
        found_idx = allocator.lookup_prefix(prefix_hash)
        assert found_idx == idx
        assert allocator.blocks[idx].ref_count == 2

    def test_lookup_nonexistent(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=5, config=config)

        assert allocator.lookup_prefix(12345) is None

    def test_prefix_invalidation_on_free(self):
        config = MLACacheConfig(block_size=4, kv_lora_rank=64)
        allocator = MLABlockAllocator(num_blocks=5, config=config)

        idx = allocator.allocate()
        prefix_hash = 12345
        allocator.register_prefix(idx, prefix_hash)

        # Free the block
        allocator.free(idx)

        # Prefix should no longer be found
        assert allocator.lookup_prefix(prefix_hash) is None

    def test_memory_usage_stats(self):
        config = MLACacheConfig(
            block_size=16,
            kv_lora_rank=512,
            num_kv_heads=8,
            head_dim=128,
        )
        allocator = MLABlockAllocator(num_blocks=10, config=config)

        # Allocate 5 blocks
        for _ in range(5):
            allocator.allocate()

        stats = allocator.memory_usage_stats()
        assert stats["allocated_blocks"] == 5
        assert stats["free_blocks"] == 5
        assert stats["memory_savings_ratio"] == config.memory_savings_ratio
        assert stats["mla_cache_mb"] < stats["standard_cache_mb"]


# =============================================================================
# MLA Attention Tests (requires MLX)
# =============================================================================


@requires_mlx
class TestMLAAttention:
    """Test MLA attention computation."""

    def test_mla_attention_basic(self):
        num_seqs = 2
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        kv_lora_rank = 64
        block_size = 4
        max_blocks = 4
        max_blocks * block_size

        # Create query
        query = mx.random.normal(
            (num_seqs, num_heads, 1, head_dim), dtype=mx.float16
        )

        # Create latent pool
        num_blocks = 10
        latent_pool = mx.random.normal(
            (num_blocks, block_size, kv_lora_rank), dtype=mx.float16
        )

        # Block tables
        block_tables = mx.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=mx.int32)

        # Context lengths
        context_lens = mx.array([12, 8], dtype=mx.int32)

        # Decompression projection
        kv_b_proj = mx.random.normal(
            (2 * num_kv_heads * head_dim, kv_lora_rank), dtype=mx.float16
        )

        output = mla_attention(
            query=query,
            latent_pool=latent_pool,
            block_tables=block_tables,
            context_lens=context_lens,
            kv_b_proj=kv_b_proj,
            scale=head_dim ** -0.5,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
        )

        mx.eval(output)
        assert output.shape == (num_seqs, num_heads, 1, head_dim)

    def test_mla_attention_prefill(self):
        """Test MLA attention with multiple query positions (prefill)."""
        num_seqs = 1
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        kv_lora_rank = 64
        block_size = 4
        seq_len = 8  # Prefill 8 tokens

        # Create query
        query = mx.random.normal(
            (num_seqs, num_heads, seq_len, head_dim), dtype=mx.float16
        )

        # Create latent pool
        num_blocks = 10
        latent_pool = mx.random.normal(
            (num_blocks, block_size, kv_lora_rank), dtype=mx.float16
        )

        # Block tables (need enough blocks for seq_len)
        block_tables = mx.array([[0, 1, 2, 3]], dtype=mx.int32)

        # Context length equals seq_len for prefill
        context_lens = mx.array([seq_len], dtype=mx.int32)

        # Decompression projection
        kv_b_proj = mx.random.normal(
            (2 * num_kv_heads * head_dim, kv_lora_rank), dtype=mx.float16
        )

        output = mla_attention(
            query=query,
            latent_pool=latent_pool,
            block_tables=block_tables,
            context_lens=context_lens,
            kv_b_proj=kv_b_proj,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
        )

        mx.eval(output)
        assert output.shape == (num_seqs, num_heads, seq_len, head_dim)


@requires_mlx
class TestMLABlockRepr:
    """Test block string representation."""

    def test_repr_unallocated(self):
        block = MLABlock()
        r = repr(block)
        assert "allocated=no" in r
        assert "tokens=0/16" in r

    def test_repr_allocated(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64)
        block = MLABlock(config=cfg)
        block.allocate()
        r = repr(block)
        assert "allocated=yes" in r
        assert "tokens=0/4" in r
        assert "rank=64" in r

    def test_repr_quantized(self):
        cfg = MLACacheConfig(block_size=4, kv_lora_rank=64, quantize_mode="fp8")
        block = MLABlock(config=cfg)
        r = repr(block)
        assert "mode=fp8" in r
