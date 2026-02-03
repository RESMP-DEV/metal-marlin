"""Tests for TrellisKVCache - MLA compressed KV cache."""

from __future__ import annotations

import pytest
import torch

from metal_marlin._compat import HAS_MPS, HAS_TORCH
from metal_marlin.trellis.kv_cache import TrellisKVCache

# Skip entire module if PyTorch unavailable
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")


def _get_device() -> str:
    """Get appropriate device for tests."""
    return "mps" if HAS_MPS else "cpu"


class TestTrellisKVCache:
    """Test suite for TrellisKVCache MLA implementation."""

    @pytest.fixture
    def cache_params(self):
        """Default cache parameters."""
        return {
            "num_layers": 4,
            "batch_size": 2,
            "max_seq_len": 128,
            "kv_lora_rank": 64,
            "qk_rope_head_dim": 64,
            "num_kv_heads": 8,
            "head_dim": 128,
        }

    @pytest.fixture
    def cache(self, cache_params):
        """Create a TrellisKVCache instance."""
        device = _get_device()
        return TrellisKVCache(
            num_layers=cache_params["num_layers"],
            batch_size=cache_params["batch_size"],
            max_seq_len=cache_params["max_seq_len"],
            kv_lora_rank=cache_params["kv_lora_rank"],
            qk_rope_head_dim=cache_params["qk_rope_head_dim"],
            device=device,
            dtype=torch.float16,
        )

    def test_init(self, cache, cache_params):
        """Test cache initialization."""
        assert cache.num_layers == cache_params["num_layers"]
        assert cache.batch_size == cache_params["batch_size"]
        assert cache.max_seq_len == cache_params["max_seq_len"]
        assert cache.kv_lora_rank == cache_params["kv_lora_rank"]
        assert cache.qk_rope_head_dim == cache_params["qk_rope_head_dim"]

        # Check cache_dim calculation: kv_lora_rank + qk_rope_head_dim
        expected_cache_dim = cache_params["kv_lora_rank"] + cache_params["qk_rope_head_dim"]
        assert cache.cache_dim == expected_cache_dim

        # Check tensor shapes
        assert cache.kv_cache.shape == (
            cache_params["num_layers"],
            cache_params["batch_size"],
            cache_params["max_seq_len"],
            expected_cache_dim,
        )

        # Check data types
        assert cache.kv_cache.dtype == torch.float16

        # Check sequence lengths initialized to 0
        assert cache.get_seq_len() == 0
        assert torch.all(cache.seq_lens == 0)

    def test_memory_layout_mps_optimal(self, cache):
        """Verify MPS-optimal memory layout (contiguous, half precision)."""
        # Check tensors are contiguous
        assert cache.kv_cache.is_contiguous()

        # Check half precision for MPS
        if HAS_MPS:
            assert cache.kv_cache.dtype == torch.float16

    def test_update_single_token(self, cache, cache_params):
        """Test updating cache with a single token."""
        device = _get_device()
        cache_dim = cache_params["kv_lora_rank"] + cache_params["qk_rope_head_dim"]

        # Create single token compressed KV
        compressed_kv = torch.randn(
            cache_params["batch_size"], 1, cache_dim, dtype=torch.float16, device=device
        )

        # Update layer 0
        full_kv = cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Check output shape
        assert full_kv.shape == (cache_params["batch_size"], 1, cache_dim)

        # Check seq_len not yet advanced (only advances on last layer)
        assert cache.get_seq_len() == 0

    def test_update_incremental_generation(self, cache, cache_params):
        """Test incremental generation with position_ids handling."""
        device = _get_device()
        cache_dim = cache_params["kv_lora_rank"] + cache_params["qk_rope_head_dim"]

        # Simulate prefill with 10 tokens
        prefill_len = 10
        compressed_kv_prefill = torch.randn(
            cache_params["batch_size"], prefill_len, cache_dim, dtype=torch.float16, device=device
        )

        # Update all layers with prefill
        for layer_idx in range(cache_params["num_layers"]):
            full_kv = cache.update(layer_idx=layer_idx, compressed_kv=compressed_kv_prefill)
            assert full_kv.shape == (cache_params["batch_size"], prefill_len, cache_dim)

        # After last layer, seq_len should be updated
        assert cache.get_seq_len() == prefill_len

        # Simulate generation of 3 more tokens (one at a time)
        for i in range(3):
            compressed_kv_new = torch.randn(
                cache_params["batch_size"], 1, cache_dim, dtype=torch.float16, device=device
            )

            # Update all layers
            for layer_idx in range(cache_params["num_layers"]):
                full_kv = cache.update(layer_idx=layer_idx, compressed_kv=compressed_kv_new)
                # Should return prefill + generated tokens so far
                expected_len = prefill_len + i + 1
                assert full_kv.shape == (cache_params["batch_size"], expected_len, cache_dim)

            assert cache.get_seq_len() == prefill_len + i + 1

    def test_update_returns_correct_format(self, cache, cache_params):
        """Test that update returns cached states in correct format for decompression."""
        device = _get_device()
        cache_dim = cache.cache_dim

        # Create compressed KV with known values
        seq_len = 5
        compressed_kv = torch.randn(
            cache_params["batch_size"], seq_len, cache_dim, dtype=torch.float16, device=device
        )

        # Update cache
        full_kv = cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Verify output format: [c_kv | k_pe] concatenated
        c_kv_part = full_kv[..., : cache_params["kv_lora_rank"]]
        k_pe_part = full_kv[..., cache_params["kv_lora_rank"] :]

        assert c_kv_part.shape[-1] == cache_params["kv_lora_rank"]
        assert k_pe_part.shape[-1] == cache_params["qk_rope_head_dim"]

        # Verify values match what was stored
        expected_c_kv = compressed_kv[..., : cache_params["kv_lora_rank"]]
        expected_k_pe = compressed_kv[..., cache_params["kv_lora_rank"] :]

        torch.testing.assert_close(c_kv_part, expected_c_kv, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_pe_part, expected_k_pe, rtol=1e-3, atol=1e-3)

    def test_get_method(self, cache, cache_params):
        """Test the get method returns cached states correctly."""
        device = _get_device()
        cache_dim = cache.cache_dim

        # Initially should return None
        assert cache.get(layer_idx=0) is None

        # Store some data
        seq_len = 5
        compressed_kv = torch.randn(
            cache_params["batch_size"], seq_len, cache_dim, dtype=torch.float16, device=device
        )

        # Update all layers to set seq_len
        for layer_idx in range(cache_params["num_layers"]):
            cache.update(layer_idx=layer_idx, compressed_kv=compressed_kv)

        # Now get should return the cached data
        cached = cache.get(layer_idx=0)
        assert cached is not None
        assert cached.shape == (cache_params["batch_size"], seq_len, cache_dim)

        # Check format
        assert cached.shape[-1] == cache.cache_dim

    def test_reset(self, cache, cache_params):
        """Test cache reset functionality."""
        device = _get_device()
        cache_dim = cache.cache_dim

        # Store some data
        compressed_kv = torch.randn(
            cache_params["batch_size"], 5, cache_dim, dtype=torch.float16, device=device
        )

        for layer_idx in range(cache_params["num_layers"]):
            cache.update(layer_idx=layer_idx, compressed_kv=compressed_kv)

        assert cache.get_seq_len() == 5

        # Reset
        cache.reset()

        # Seq len should be 0
        assert cache.get_seq_len() == 0
        assert torch.all(cache.seq_lens == 0)

        # Get should return None
        assert cache.get(layer_idx=0) is None

    def test_memory_usage_calculation(self, cache, cache_params):
        """Test memory usage calculation."""
        # Calculate expected memory
        bytes_per_element = 2  # float16

        # c_kv elements: [num_layers, batch, max_seq, kv_lora_rank]
        c_kv_elements = (
            cache_params["num_layers"]
            * cache_params["batch_size"]
            * cache_params["max_seq_len"]
            * cache_params["kv_lora_rank"]
        )

        # k_pe elements: [num_layers, batch, max_seq, qk_rope_head_dim]
        k_pe_elements = (
            cache_params["num_layers"]
            * cache_params["batch_size"]
            * cache_params["max_seq_len"]
            * cache_params["qk_rope_head_dim"]
        )

        expected_mb = (c_kv_elements + k_pe_elements) * bytes_per_element / 1024 / 1024

        assert abs(cache.memory_usage_mb() - expected_mb) < 0.01

    def test_dimension_mismatch_error(self, cache):
        """Test that wrong input dimension raises error."""
        device = _get_device()

        # Create compressed KV with wrong dimension
        wrong_dim = cache.cache_dim + 10
        compressed_kv = torch.randn(1, 1, wrong_dim, dtype=torch.float16, device=device)

        with pytest.raises(ValueError, match="Input dimension"):
            cache.update(layer_idx=0, compressed_kv=compressed_kv)

    def test_exceed_max_seq_len(self, cache, cache_params):
        """Test that exceeding max_seq_len raises error."""
        device = _get_device()
        cache_dim = cache.cache_dim

        # Try to store more than max_seq_len
        too_long = cache_params["max_seq_len"] + 10
        compressed_kv = torch.randn(
            cache_params["batch_size"], too_long, cache_dim, dtype=torch.float16, device=device
        )

        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            cache.update(layer_idx=0, compressed_kv=compressed_kv)

    def test_batch_item_independence(self, cache, cache_params):
        """Test that different batch items are stored independently."""
        device = _get_device()
        cache_dim = cache.cache_dim

        # Create different values for each batch item
        batch_size = cache_params["batch_size"]
        seq_len = 3

        compressed_kv = torch.stack(
            [
                torch.full((seq_len, cache_dim), float(i), dtype=torch.float16, device=device)
                for i in range(batch_size)
            ]
        )

        # Update all layers to properly set seq_len
        for layer_idx in range(cache_params["num_layers"]):
            cache.update(layer_idx=layer_idx, compressed_kv=compressed_kv)

        # Verify each batch item has correct values
        cached = cache.get(layer_idx=0)
        for i in range(batch_size):
            expected = torch.full(
                (seq_len, cache_dim), float(i), dtype=torch.float16, device=device
            )
            torch.testing.assert_close(cached[i], expected, rtol=1e-3, atol=1e-3)

    def test_layer_independence(self, cache, cache_params):
        """Test that different layers store data independently."""
        device = _get_device()
        cache_dim = cache.cache_dim

        seq_len = 3

        # Store different values in different layers
        for layer_idx in range(cache_params["num_layers"]):
            compressed_kv = torch.full(
                (cache_params["batch_size"], seq_len, cache_dim),
                float(layer_idx),
                dtype=torch.float16,
                device=device,
            )
            cache.update(layer_idx=layer_idx, compressed_kv=compressed_kv)

        # Reset and check each layer has correct values
        cache.reset()

        for layer_idx in range(cache_params["num_layers"]):
            compressed_kv = torch.full(
                (cache_params["batch_size"], seq_len, cache_dim),
                float(layer_idx),
                dtype=torch.float16,
                device=device,
            )
            for li in range(cache_params["num_layers"]):
                cache.update(layer_idx=li, compressed_kv=compressed_kv)

            cached = cache.get(layer_idx=layer_idx)
            expected = torch.full(
                (cache_params["batch_size"], seq_len, cache_dim),
                float(layer_idx),
                dtype=torch.float16,
                device=device,
            )
            torch.testing.assert_close(cached, expected, rtol=1e-3, atol=1e-3)
            cache.reset()

    def test_output_contiguous(self, cache, cache_params):
        """Test that output tensors are contiguous for MPS efficiency."""
        device = _get_device()
        cache_dim = cache.cache_dim

        compressed_kv = torch.randn(
            cache_params["batch_size"], 5, cache_dim, dtype=torch.float16, device=device
        )

        # Make non-contiguous input
        compressed_kv = compressed_kv.transpose(1, 2).transpose(1, 2)

        full_kv = cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Output should be contiguous
        assert full_kv.is_contiguous()

        # Get output should also be contiguous
        for li in range(cache_params["num_layers"]):
            cache.update(layer_idx=li, compressed_kv=compressed_kv)

        cached = cache.get(layer_idx=0)
        assert cached.is_contiguous()


class TestTrellisKVCacheQuantization:
    """Test suite for TrellisKVCache int8 quantization."""

    @pytest.fixture
    def cache_params(self):
        """Default cache parameters."""
        return {
            "num_layers": 2,
            "batch_size": 2,
            "max_seq_len": 64,
            "kv_lora_rank": 64,
            "qk_rope_head_dim": 64,
        }

    @pytest.fixture
    def cache(self, cache_params):
        """Create a quantized TrellisKVCache instance."""
        device = _get_device()
        return TrellisKVCache(
            num_layers=cache_params["num_layers"],
            batch_size=cache_params["batch_size"],
            max_seq_len=cache_params["max_seq_len"],
            kv_lora_rank=cache_params["kv_lora_rank"],
            qk_rope_head_dim=cache_params["qk_rope_head_dim"],
            device=device,
            dtype=torch.float16,
            quantize=True,
        )

    def test_init_quantized(self, cache, cache_params):
        """Test quantized cache initialization."""
        assert cache.quantize is True
        assert cache.c_kv.dtype == torch.int8
        assert cache.k_pe.dtype == torch.int8
        assert cache.c_kv_scales is not None
        assert cache.k_pe_scales is not None

        # Check scale shapes (per-token: [num_layers, batch, max_seq_len, 1])
        assert cache.c_kv_scales.shape == (
            cache_params["num_layers"],
            cache_params["batch_size"],
            cache_params["max_seq_len"],
            1,
        )
        assert cache.k_pe_scales.shape == (
            cache_params["num_layers"],
            cache_params["batch_size"],
            cache_params["max_seq_len"],
            1,
        )

    def test_quantization_accuracy(self, cache, cache_params):
        """Test that quantization preserves values with reasonable accuracy."""
        device = _get_device()
        cache_dim = cache.cache_dim

        # Create random data in [-1, 1] range
        compressed_kv = torch.randn(
            cache_params["batch_size"], 5, cache_dim, dtype=torch.float16, device=device
        )

        # Update cache
        full_kv = cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Check that retrieved values are close to original
        # Int8 quantization error bound roughly 1/127 of max value
        # We allow a bit more due to float conversions
        torch.testing.assert_close(full_kv, compressed_kv, rtol=0.05, atol=0.05)

    def test_memory_usage_reduction(self, cache_params):
        """Test that quantized cache uses less memory."""
        device = _get_device()

        # Create unquantized cache
        cache_fp16 = TrellisKVCache(
            **cache_params, device=device, dtype=torch.float16, quantize=False
        )

        # Create quantized cache
        cache_int8 = TrellisKVCache(
            **cache_params, device=device, dtype=torch.float16, quantize=True
        )

        mb_fp16 = cache_fp16.memory_usage_mb()
        mb_int8 = cache_int8.memory_usage_mb()

        # Int8 should be roughly half size of FP16 (plus scales overhead)
        # 1 byte vs 2 bytes. Scales add negligible overhead.
        assert mb_int8 < mb_fp16 * 0.6  # Expect significant reduction

    def test_incremental_update_scales(self, cache, cache_params):
        """Test that scales are updated incrementally for each token."""
        device = _get_device()
        cache_dim = cache.cache_dim

        # Token 1: small values
        t1 = torch.full((cache_params["batch_size"], 1, cache_dim), 0.1, dtype=torch.float16, device=device)
        cache.update(layer_idx=0, compressed_kv=t1)

        # Check scale for t1 (approx 0.1 / 127 * 127 = 0.1)
        # Actually scale = max_val / 127. So scale approx 0.1/127.
        scale_t1 = cache.c_kv_scales[0, :, 0, 0]
        assert torch.all(scale_t1 > 0)
        assert torch.all(scale_t1 < 1.0)

        # Token 2: large values
        t2 = torch.full((cache_params["batch_size"], 1, cache_dim), 10.0, dtype=torch.float16, device=device)
        cache.update(layer_idx=0, compressed_kv=t2)

        # Check scale for t2 (approx 10.0 / 127)
        scale_t2 = cache.c_kv_scales[0, :, 1, 0]
        assert torch.all(scale_t2 > scale_t1)  # Scale should adapt to magnitude

    def test_get_layer_slices_and_attention_quantized(self, cache, cache_params):
        """Test get_layer_slices and get_layer_for_attention with quantization."""
        device = _get_device()
        cache_dim = cache.cache_dim

        compressed_kv = torch.randn(
            cache_params["batch_size"], 5, cache_dim, dtype=torch.float16, device=device
        )

        cache.update(layer_idx=0, compressed_kv=compressed_kv)

        # Test get_layer_slices
        c_kv_slice, k_pe_slice = cache.get_layer_slices(layer_idx=0)
        assert c_kv_slice.shape == (cache_params["batch_size"], 5, cache_params["kv_lora_rank"])
        assert k_pe_slice.shape == (cache_params["batch_size"], 5, cache_params["qk_rope_head_dim"])

        # Test get_layer_for_attention (BHSD format)
        kv_attn = cache.get_layer_for_attention(layer_idx=0)
        # Expected shape: [batch, cache_dim, seq_len]
        assert kv_attn.shape == (cache_params["batch_size"], cache_dim, 5)

        # Verify values are reasonable (dequantized)
        expected_bhsd = compressed_kv.permute(0, 2, 1)
        torch.testing.assert_close(kv_attn, expected_bhsd, rtol=0.05, atol=0.05)
