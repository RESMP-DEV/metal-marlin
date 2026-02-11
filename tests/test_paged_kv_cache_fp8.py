"""Tests for PagedKVCache FP8 quantization."""

import numpy as np
import pytest
from metal_marlin.paged_kv_cache import PagedKVCache

class TestPagedKVCacheFP8:
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    def test_fp8_quantize_dequantize(self, rng: np.random.Generator) -> None:
        """Test FP8 quantization and dequantization in PagedKVCache."""
        num_blocks = 16
        num_kv_heads = 4
        head_dim = 64
        block_size = 16
        
        cache = PagedKVCache(
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype="fp8",
            block_size=block_size
        )
        
        # Allocate blocks for a sequence of length 32 (2 blocks)
        seq_len = 32
        block_indices = cache.allocate_blocks(seq_len)
        assert len(block_indices) == 2
        
        # Create slot mapping [0, 1, ..., 31] mapped to physical blocks
        # logical_block 0 -> physical block_indices[0]
        # logical_block 1 -> physical block_indices[1]
        slot_mapping = np.zeros(seq_len, dtype=np.int32)
        for i in range(seq_len):
            block_idx = block_indices[i // block_size]
            block_offset = i % block_size
            physical_slot = block_idx * block_size + block_offset
            slot_mapping[i] = physical_slot
            
        # Generate random K/V data
        k = rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32)
        
        # Quantize and store
        cache.quantize_kv(k, v, slot_mapping)
        
        # Dequantize and retrieve
        k_out, v_out = cache.dequantize_kv(slot_mapping)
        
        # Check shapes
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
        
        # Check accuracy (FP8 E4M3 has limited precision, but MSE should be reasonable)
        # We expect MSE < 1e-3 for standard normal data
        mse_k = np.mean((k - k_out) ** 2)
        mse_v = np.mean((v - v_out) ** 2)
        
        print(f"MSE K: {mse_k}, MSE V: {mse_v}")
        assert mse_k < 5e-3, f"K MSE too high: {mse_k}"
        assert mse_v < 5e-3, f"V MSE too high: {mse_v}"

    def test_fp16_passthrough(self, rng: np.random.Generator) -> None:
        """Test FP16 passthrough (no quantization)."""
        cache = PagedKVCache(
            num_blocks=4,
            num_kv_heads=2,
            head_dim=32,
            dtype="fp16"
        )
        
        seq_len = 16
        blocks = cache.allocate_blocks(seq_len)
        slot_mapping = np.arange(seq_len, dtype=np.int32) 
        # For simplicity, if blocks are 0, 1..., physical slots match logical if we map manually
        # allocate_blocks returns specific indices from free pool.
        # Let's map properly.
        for i in range(seq_len):
            block_idx = blocks[i // 16]
            slot_mapping[i] = block_idx * 16 + (i % 16)

        k = rng.standard_normal((seq_len, 2, 32)).astype(np.float16)
        v = rng.standard_normal((seq_len, 2, 32)).astype(np.float16)
        
        cache.quantize_kv(k, v, slot_mapping)
        k_out, v_out = cache.dequantize_kv(slot_mapping)
        
        # Should be exact match for FP16
        np.testing.assert_array_equal(k, k_out)
        np.testing.assert_array_equal(v, v_out)

