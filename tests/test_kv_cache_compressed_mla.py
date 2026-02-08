"""Tests for CompressedKVCacheMLA - optimized compressed KV cache for MLA.

Tests cover:
- Basic cache operations (update, get, decompress)
- Memory savings and compression ratio
- Block-sparse layout and paging
- Threadgroup cache functionality
- Prefetch optimization
- Quantization support (FP8/FP4)
- Performance statistics tracking
- Integration with Trellis attention
"""

from __future__ import annotations

import math

import pytest

from metal_marlin._compat import HAS_TORCH, torch

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")

if HAS_TORCH:
    from metal_marlin.trellis.config import TrellisModelConfig
    from metal_marlin.trellis.kv_cache_compressed import CompressedKVCacheMLA
    from metal_marlin.trellis.kv_cache_compressed_integration import (
        create_compressed_kv_cache,
        decompress_kv_optimized,
        estimate_memory_savings,
        get_cache_optimization_stats,
        get_optimized_kv_from_cache,
        print_cache_stats,
        reset_cache_all,
        reset_cache_for_batch,
        update_with_prefetch_and_cache,
    )


@requires_torch
class TestCompressedKVCacheMLABasic:
    """Test basic cache operations."""

    @staticmethod
    def _device() -> torch.device:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def test_init_defaults(self):
        """Test cache initialization with default parameters."""
        config = TrellisModelConfig(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_kv_heads=4,
            kv_lora_rank=256,
            qk_nope_head_dim=128,
            qk_rope_head_dim=32,
            v_head_dim=128,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=128,
            device=self._device(),
            dtype=torch.float16,
        )

        assert cache.num_layers == 2
        assert cache.max_batch_size == 1
        assert cache.max_seq_len == 128
        assert cache.kv_lora_rank == 256
        assert cache.cache_dim == 288  # 256 + 32

    def test_update_and_get_compressed_kv(self):
        """Test updating cache and retrieving compressed KV."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_kv_heads=4,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
            block_size=16,
        )

        # Update cache with compressed KV
        compressed_kv = torch.randn(1, 10, 160, device=self._device(), dtype=torch.float16)
        result = cache.update(layer_idx=0, compressed_kv=compressed_kv)

        assert result.shape == compressed_kv.shape
        assert torch.equal(result, compressed_kv)

        # Retrieve compressed KV
        retrieved = cache.get_compressed_kv(layer_idx=0)
        assert retrieved.shape == (1, 10, 160)
        torch.testing.assert_close(retrieved, compressed_kv, rtol=1e-3, atol=1e-3)

    def test_decompress_kv(self):
        """Test on-the-fly decompression of KV."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
            block_size=16,
        )

        # Create compressed KV and add to cache
        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 8, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Create decompression weight
        # Output: num_kv_heads * (qk_nope + v) = 2 * (64 + 64) = 256
        kv_b_proj_weight = torch.randn(256, 128, device=self._device(), dtype=torch.float16)

        # Decompress KV
        k, v = cache.decompress_kv(
            layer_idx=0,
            kv_b_proj_weight=kv_b_proj_weight,
            kv_a_layernorm=None,
        )

        assert k.shape == (1, 8, 2, 96)  # batch, seq, num_kv_heads, qk_nope+qk_rope
        assert v.shape == (1, 8, 2, 64)  # batch, seq, num_kv_heads, v_head_dim


@requires_torch
class TestCompressedKVCacheMLAOptimizations:
    """Test optimization features: block-sparse, prefetch, threadgroup cache."""

    @staticmethod
    def _device() -> torch.device:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def test_block_sparse_layout(self):
        """Test block-sparse layout for long sequences."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        block_size = 16
        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=256,
            device=self._device(),
            dtype=torch.float16,
            block_size=block_size,
        )

        # Add sequence longer than block_size
        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 100, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Verify blocks are allocated
        stats = cache.get_block_sparse_stats()
        # 100 tokens / 16 block_size = 7 blocks (rounded up)
        assert stats["used_blocks"] == 7

    def test_memory_pooling_efficiency(self):
        """Test that memory pooling reduces allocations."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
            block_size=16,
        )

        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim

        # Multiple updates should reuse memory pool
        for i in range(5):
            compressed_kv = torch.randn(1, 5, cache_dim, device=self._device(), dtype=torch.float16)
            cache.update(layer_idx=0, compressed_kv=compressed_kv)

        perf_stats = cache.get_performance_stats()
        # Should have exactly 2 allocations (5 tokens * 5 updates = 25, block_size=16 -> 2 blocks)
        assert perf_stats["total_allocations"] == 2

    def test_threadgroup_cache(self):
        """Test threadgroup cache for decompressed tiles."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
            block_size=16,
            threadgroup_cache_size=4,
        )

        # Update cache
        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 32, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Update threadgroup cache
        cache._update_threadgroup_cache(layer_idx=0, block_idx=0)

        # Check cache hit
        cached = cache.get_cached_block(layer_idx=0, block_idx=0)
        assert cached is not None
        assert cached.shape == (16, cache_dim)

        # Check cache miss
        not_cached = cache.get_cached_block(layer_idx=0, block_idx=5)
        assert not_cached is None

    def test_prefetch_optimization(self):
        """Test async prefetching of next layer's blocks."""
        config = TrellisModelConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
            block_size=16,
            prefetch_enabled=True,
        )

        # Update cache for layer 0
        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 16, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Prefetch layer 1
        cache.prefetch_layer_async(layer_idx=1)

        # Check prefetch queue
        perf_stats = cache.get_performance_stats()
        assert perf_stats["prefetch_count"] == 1


@requires_torch
class TestCompressedKVCacheMLAQuantization:
    """Test FP8/FP4 quantization support."""

    @staticmethod
    def _device() -> torch.device:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def test_fp8_quantization(self):
        """Test FP8 quantization of compressed KV."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
            quantize_mode="fp8",
        )

        assert cache.quantize_mode == "fp8"
        assert cache.kv_cache_pool.dtype == torch.uint8

    def test_fp4_quantization(self):
        """Test FP4 quantization of compressed KV."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
            quantize_mode="fp4",
        )

        assert cache.quantize_mode == "fp4"
        assert cache.kv_cache_pool.dtype == torch.uint8

    def test_memory_savings_with_quantization(self):
        """Test memory savings from quantization."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache_none = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=1024,
            device=self._device(),
            dtype=torch.float16,
            quantize_mode="none",
        )

        cache_fp8 = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=1024,
            device=self._device(),
            dtype=torch.float16,
            quantize_mode="fp8",
        )

        cache_fp4 = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=1024,
            device=self._device(),
            dtype=torch.float16,
            quantize_mode="fp4",
        )

        memory_none = cache_none.memory_usage_mb()
        memory_fp8 = cache_fp8.memory_usage_mb()
        memory_fp4 = cache_fp4.memory_usage_mb()

        # FP8 should use ~50% of standard
        assert abs(memory_fp8 / memory_none - 0.5) < 0.1

        # FP4 should use ~25% of standard
        assert abs(memory_fp4 / memory_none - 0.25) < 0.1


@requires_torch
class TestCompressedKVCacheMLAStatistics:
    """Test statistics and monitoring."""

    @staticmethod
    def _device() -> torch.device:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def test_block_sparse_stats(self):
        """Test block-sparse statistics."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=128,
            device=self._device(),
            dtype=torch.float16,
            block_size=16,
        )

        # Update cache
        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 50, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        stats = cache.get_block_sparse_stats()
        assert "total_blocks" in stats
        assert "used_blocks" in stats
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] > 1.0

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
            dtype=torch.float16,
        )

        # Perform operations
        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 10, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        kv_b_proj_weight = torch.randn(256, 128, device=self._device(), dtype=torch.float16)
        cache.decompress_kv(
            layer_idx=0,
            kv_b_proj_weight=kv_b_proj_weight,
            kv_a_layernorm=None,
        )

        cache.prefetch_layer_async(layer_idx=1)

        perf_stats = cache.get_performance_stats()
        assert perf_stats["total_allocations"] > 0
        assert perf_stats["decompression_count"] > 0
        assert perf_stats["prefetch_count"] > 0


@requires_torch
class TestCompressedKVCacheMLAIntegration:
    """Test integration utilities."""

    @staticmethod
    def _device() -> torch.device:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def test_create_compressed_kv_cache(self):
        """Test factory function for cache creation."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = create_compressed_kv_cache(
            config=config,
            max_batch_size=1,
            max_seq_len=512,
            device=self._device(),
            quantize_mode="fp8",
        )

        assert cache.quantize_mode == "fp8"
        assert cache.prefetch_enabled

    def test_update_with_prefetch_and_cache(self):
        """Test update wrapper with prefetch."""
        config = TrellisModelConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = create_compressed_kv_cache(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
        )

        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 8, cache_dim, device=self._device(), dtype=torch.float16)

        update_with_prefetch_and_cache(
            cache,
            layer_idx=0,
            compressed_kv=compressed_kv,
            trigger_prefetch=True,
        )

        perf_stats = cache.get_performance_stats()
        assert perf_stats["prefetch_count"] > 0

    def test_reset_cache(self):
        """Test cache reset functionality."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = create_compressed_kv_cache(
            config=config,
            max_batch_size=2,
            max_seq_len=64,
            device=self._device(),
        )

        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 8, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Reset specific batch
        reset_cache_for_batch(cache, batch_idx=0)

        # Reset all
        reset_cache_all(cache)

        perf_stats = cache.get_performance_stats()
        # After reset all, stats should be cleared
        assert perf_stats["total_allocations"] == 0

    def test_estimate_memory_savings(self):
        """Test memory savings estimation."""
        config = TrellisModelConfig(
            num_hidden_layers=32,
            num_attention_heads=20,
            num_kv_heads=20,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
        )

        stats = estimate_memory_savings(
            config,
            max_seq_len=8192,
            num_layers=32,
            num_kv_heads=20,
            head_dim=448,  # qk_nope + qk_rope + v
        )

        assert "standard_kv_mb" in stats
        assert "compressed_kv_mb" in stats
        assert "savings_ratio" in stats

        # Should have significant savings
        assert stats["savings_ratio"] > 4.0

    def test_print_cache_stats(self):
        """Test cache statistics printing (no exception)."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        cache = create_compressed_kv_cache(
            config=config,
            max_batch_size=1,
            max_seq_len=64,
            device=self._device(),
        )

        # Should not raise exception
        print_cache_stats(cache)


@requires_torch
class TestCompressedKVCacheMLALongContext:
    """Test long context handling (>8K tokens)."""

    @staticmethod
    def _device() -> torch.device:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def test_long_context_efficient_paging(self):
        """Test efficient paging for long sequences."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        block_size = 64
        cache = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=8192,
            device=self._device(),
            dtype=torch.float16,
            block_size=block_size,
        )

        # Add long sequence
        cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        compressed_kv = torch.randn(1, 8192, cache_dim, device=self._device(), dtype=torch.float16)
        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Verify block allocation
        stats = cache.get_block_sparse_stats()
        expected_blocks = math.ceil(8192 / block_size)
        assert stats["used_blocks"] == expected_blocks

    def test_memory_usage_scaling(self):
        """Test that memory scales with block size, not sequence length."""
        config = TrellisModelConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        # Cache for 4K tokens
        cache_4k = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=4096,
            device=self._device(),
            dtype=torch.float16,
            block_size=64,
        )

        # Cache for 8K tokens
        cache_8k = CompressedKVCacheMLA(
            config=config,
            max_batch_size=1,
            max_seq_len=8192,
            device=self._device(),
            dtype=torch.float16,
            block_size=64,
        )

        # Memory should double (roughly)
        memory_4k = cache_4k.memory_usage_mb()
        memory_8k = cache_8k.memory_usage_mb()

        assert abs((memory_8k / memory_4k) - 2.0) < 0.1
