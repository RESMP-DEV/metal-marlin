"""Integration tests for paged attention with MMFP4MLA layer.

This module tests the integration between MMFP4MLA layer and PagedKVCache,
validating that paged attention works correctly with the FP4-quantized MLA
attention implementation for GLM-style models.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch
from metal_marlin.paged_kv_cache import PagedKVCache


def _get_device() -> str:
    """Get the appropriate test device."""
    if HAS_MPS:
        return "mps"
    return "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestPagedKVCacheBasics:
    """Basic tests for PagedKVCache functionality."""

    def test_paged_kv_cache_creation(self):
        """Test basic PagedKVCache creation."""
        cache = PagedKVCache(
            num_blocks=64,
            num_kv_heads=20,
            head_dim=256,
            dtype="fp16",
        )
        
        assert cache.num_blocks == 64
        assert cache.num_kv_heads == 20
        assert cache.head_dim == 256
        assert cache.dtype == "fp16"
        assert cache.block_size == 16  # Fixed block size
        
    def test_paged_kv_cache_fp8_creation(self):
        """Test PagedKVCache creation with FP8 dtype."""
        cache = PagedKVCache(
            num_blocks=32,
            num_kv_heads=8,
            head_dim=128,
            dtype="fp8",
        )
        
        assert cache.dtype == "fp8"
        assert cache.k_blocks.dtype == np.dtype(np.uint8)
        assert cache.k_scales is not None
        
    def test_paged_kv_cache_invalid_block_size(self):
        """Test that invalid block size raises error."""
        with pytest.raises(ValueError, match="block_size"):
            PagedKVCache(
                num_blocks=64,
                num_kv_heads=20,
                head_dim=256,
                dtype="fp16",
                block_size=32,  # Should be 16
            )
            
    def test_paged_kv_cache_free_blocks(self):
        """Test free block tracking."""
        cache = PagedKVCache(
            num_blocks=64,
            num_kv_heads=20,
            head_dim=256,
            dtype="fp16",
        )
        
        assert cache.num_free_blocks == 64
        assert cache.num_allocated_blocks == 0
        
        # Allocate some blocks
        cache.allocate_blocks(32, seq_id=0)  # 2 blocks
        
        assert cache.num_free_blocks == 62
        assert cache.num_allocated_blocks == 2

    def test_paged_kv_cache_allocation(self):
        """Test PagedKVCache block allocation."""
        cache = PagedKVCache(
            num_blocks=64,
            num_kv_heads=20,
            head_dim=256,
            dtype="fp16",
        )
        
        # Allocate blocks for a sequence
        num_tokens = 32
        seq_id = 0
        allocated = cache.allocate_blocks(num_tokens, seq_id)
        
        # Should allocate 2 blocks (32 tokens / 16 tokens per block = 2)
        expected_blocks = (num_tokens + cache.block_size - 1) // cache.block_size
        assert len(allocated) == expected_blocks, f"Expected {expected_blocks} blocks, got {len(allocated)}"
        
        # Verify context length
        assert cache._context_lens[seq_id] == num_tokens, f"Context length should be {num_tokens}"
        
        # Verify block table
        block_tables = cache.get_block_tables()
        assert block_tables.shape[0] == 1, "Should have 1 sequence"
        assert block_tables.shape[1] >= expected_blocks, "Should have at least expected_blocks columns"

    def test_paged_kv_cache_quantize_fp8(self):
        """Test FP8 quantization in PagedKVCache."""
        cache = PagedKVCache(
            num_blocks=16,
            num_kv_heads=8,
            head_dim=128,
            dtype="fp8",
        )
        
        # Allocate and store some KV values
        num_tokens = 16
        cache.allocate_blocks(num_tokens, seq_id=0)
        
        # Create test K/V tensors
        k = np.random.randn(num_tokens, cache.num_kv_heads, cache.head_dim).astype(np.float16)
        v = np.random.randn(num_tokens, cache.num_kv_heads, cache.head_dim).astype(np.float16)
        slot_mapping = np.arange(num_tokens, dtype=np.int32)
        
        # Quantize and store
        cache.quantize_kv(k, v, slot_mapping)
        
        # Retrieve and verify
        k_retrieved, v_retrieved = cache.get_kv(seq_id=0)
        
        assert k_retrieved.shape == k.shape, f"K shape mismatch: {k_retrieved.shape} vs {k.shape}"
        assert v_retrieved.shape == v.shape, f"V shape mismatch: {v_retrieved.shape} vs {v.shape}"
        assert k_retrieved.dtype == np.float16, f"K dtype should be float16, got {k_retrieved.dtype}"


@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
class TestPagedAttentionMMFP4Integration:
    """Integration tests for paged attention with MMFP4MLA."""

    def test_mmfp4_mla_creation_with_paged_attention(self):
        """Test MMFP4MLA layer creation with paged attention enabled."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        device = _get_device()
        
        # Valid GQA configuration
        layer = MMFP4MLA(
            hidden_size=2048,
            num_heads=32,
            num_kv_heads=8,  # 32 / 8 = 4, valid GQA
            q_lora_rank=384,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            use_paged_attention=True,
            use_fused_qkv=False,  # Required for paged attention path
        ).to(device)
        
        assert layer.use_paged_attention is True
        assert layer.num_heads == 32
        assert layer.num_kv_heads == 8
        
    def test_paged_adapter_lazy_initialization(self):
        """Test that paged attention adapter is created lazily."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        
        device = _get_device()
        
        layer = MMFP4MLA(
            hidden_size=2048,
            num_heads=32,
            num_kv_heads=8,
            q_lora_rank=384,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            use_paged_attention=True,
            use_fused_qkv=False,
        ).to(device)
        
        # Adapter should be None initially (lazy initialization)
        assert layer._paged_adapter is None
        
        # Getting the adapter should create it
        adapter = layer._get_or_create_paged_adapter()
        assert adapter is not None
        assert layer._paged_adapter is not None
        
    def test_mmfp4_paged_attention_adapter_forward(self):
        """Test MMFP4PagedAttention adapter forward pass."""
        from metal_marlin.layers.mmfp4_mla import MMFP4MLA
        from metal_marlin.paged.mmfp4_paged_adapter import MMFP4PagedAttention
        
        device = _get_device()
        
        # Create MLA layer
        mla_layer = MMFP4MLA(
            hidden_size=2048,
            num_heads=32,
            num_kv_heads=8,
            q_lora_rank=384,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            use_paged_attention=True,
            use_fused_qkv=False,
        ).to(device)
        
        # Create paged attention adapter
        adapter = MMFP4PagedAttention(
            mla_layer=mla_layer,
            max_batch_size=1,
            max_seq_len=1024,
            num_layers=1,
        ).to(device)
        
        # Pre-populate the cache with some data to avoid empty cache issues
        # head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
        head_dim = 512 + 64
        test_kv = torch.randn(1, 16, head_dim, device=device, dtype=torch.float16)
        adapter.cache.update_compressed(0, test_kv)
        
        # Test forward with dummy inputs
        batch_size = 1
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        
        q_nope = torch.randn(batch_size, num_heads, qk_nope_head_dim, device=device, dtype=torch.float16)
        q_rope = torch.randn(batch_size, num_heads, qk_rope_head_dim, device=device, dtype=torch.float16)
        
        # Forward pass with explicit context length
        context_lens = torch.tensor([16], dtype=torch.int32, device=device)
        output = adapter.forward(q_nope, q_rope, layer_idx=0, context_lens=context_lens)
        
        # Validate output
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert output.shape == (batch_size, num_heads, 128), f"Expected shape {(batch_size, num_heads, 128)}, got {output.shape}"


@pytest.mark.smoke
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
def test_paged_attention_glm_style():
    """Smoke test for GLM-style paged attention integration.
    
    Validates the exact configuration mentioned in the task with valid GQA.
    """
    from metal_marlin.layers.mmfp4_mla import MMFP4MLA
    
    device = _get_device()
    
    # GLM-style dimensions with valid GQA (32 % 8 == 0)
    layer = MMFP4MLA(
        hidden_size=2048,
        num_heads=32,
        num_kv_heads=8,
        q_lora_rank=384,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_paged_attention=True,
        use_fused_qkv=False,
    ).to(device)
    
    cache = PagedKVCache(
        num_blocks=64,
        num_kv_heads=8,
        head_dim=256,
        dtype="fp16",
    )
    
    # Verify components exist and are properly configured
    assert layer.use_paged_attention is True
    assert cache.num_blocks == 64
    assert cache.num_kv_heads == 8
    assert cache.head_dim == 256
    
    # Test cache allocation
    cache.allocate_blocks(32, seq_id=0)
    assert cache.num_allocated_blocks == 2  # 32 tokens / 16 per block


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
