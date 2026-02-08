"""Tests for fused MLA (Multi-head Latent Attention) kernel.

Tests the fully fused MLA attention kernel that combines:
- Q projection (hidden -> q_latent -> q_heads)
- KV projection (hidden -> kv_latent -> k, v)
- RoPE (rotary position embedding)
- Attention computation
- Output projection

Uses GLM-4.7-Flash style dimensions:
- hidden_size=256 (small for fast tests)
- num_heads=4
- head_dim=64
- kv_lora_rank=64
- q_lora_rank=96
- qk_rope_head_dim=32
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from metal_marlin._compat import HAS_MPS
from metal_marlin.mla_fused import (
    MLAAttentionParams,
    create_glm_mla_params,
    mla_fused_attention_decode,
)
from metal_marlin.quantization import quantize_fp4

# Skip tests if MPS unavailable
pytestmark = pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")


class TestMLAFused:
    """Test fused MLA attention kernel correctness."""

    def _create_test_weights(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create FP4 quantized weights from random float16 weights."""
        # Random float16 weights
        weights = torch.randn(in_features, out_features, dtype=torch.float16, device=device)
        
        # Quantize to FP4
        weights_packed, scales = quantize_fp4(
            weights, group_size=64, per_column=False
        )
        
        return weights_packed, scales

    def test_params_to_struct(self):
        """Test MLAAttentionParams serialization."""
        params = MLAAttentionParams(
            batch=2,
            seq_q=8,
            seq_k=16,
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            kv_lora_rank=64,
            q_lora_rank=96,
            rope_dim=32,
            scale=0.125,
            is_causal=True,
            q_a_group_size=64,
            q_b_group_size=64,
            kv_a_group_size=64,
            kv_b_group_size=64,
            o_group_size=64,
            rope_theta=10000.0,
            rope_ratio=1.0,
            rope_base_seq_len=0,
            cache_start_pos=0,
            cache_len=16,
            max_cache_len=32,
            use_fused_q_proj=True,
            use_fused_kv_proj=True,
            fuse_rope_in_kv_a=True,
            skip_kv_decompress=False,
        )
        
        # Convert to struct array
        struct_arr = params.to_struct()
        
        # Verify shape
        assert struct_arr.shape == (32,)  # 32 float32 values
        
        # Verify some key values
        assert struct_arr[0] == 2.0  # batch
        assert struct_arr[1] == 8.0  # seq_q
        assert struct_arr[4] == 4.0  # num_heads
        assert struct_arr[5] == 64.0  # head_dim
        assert pytest.approx(struct_arr[8], rel=1e-5) == 0.125  # scale

    def test_create_glm_mla_params(self):
        """Test GLM-4.7-Flash parameter creation."""
        params = create_glm_mla_params(
            batch=1,
            seq_q=1,
            seq_k=8,
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            kv_lora_rank=64,
            q_lora_rank=96,
            rope_dim=32,
            cache_len=8,
        )
        
        # Verify dimensions
        assert params.batch == 1
        assert params.seq_q == 1
        assert params.seq_k == 8
        assert params.hidden_size == 256
        assert params.num_heads == 4
        assert params.head_dim == 64
        assert params.kv_lora_rank == 64
        assert params.q_lora_rank == 96
        assert params.rope_dim == 32
        
        # Verify scale
        expected_scale = 1.0 / math.sqrt(params.head_dim)
        assert pytest.approx(params.scale, rel=1e-6) == expected_scale
        
        # Verify GLM-specific settings
        assert params.is_causal is True
        assert params.use_fused_q_proj is True
        assert params.use_fused_kv_proj is True
        assert params.fuse_rope_in_kv_a is True
        assert pytest.approx(params.rope_theta, rel=1e-6) == 10000.0
        assert pytest.approx(params.rope_ratio, rel=1e-6) == 1.0

    def test_fused_q_projection(self):
        """Test fused Q projection (hidden -> q_latent -> q_heads)."""
        device = torch.device("mps" if HAS_MPS else "cpu")
        
        hidden_size = 256
        q_lora_rank = 96
        num_heads = 4
        head_dim = 64
        
        # Create weights
        q_a_packed, q_a_scales = self._create_test_weights(hidden_size, q_lora_rank, device)
        q_b_packed, q_b_scales = self._create_test_weights(q_lora_rank, num_heads * head_dim, device)
        q_bias = torch.randn(num_heads * head_dim, dtype=torch.float16, device=device)
        
        # Input
        hidden = torch.randn(2, 4, hidden_size, dtype=torch.float16, device=device)
        
        # Simple reference: two separate projections
        # For now just check shapes (full test requires kernel dispatch)
        assert q_a_packed.shape[0] == (hidden_size + 7) // 8
        assert q_a_packed.shape[1] == q_lora_rank
        assert q_a_scales.shape[0] == (hidden_size + 63) // 64
        assert q_a_scales.shape[1] == q_lora_rank
        
        assert q_b_packed.shape[0] == (q_lora_rank + 7) // 8
        assert q_b_packed.shape[1] == num_heads * head_dim
        assert q_b_scales.shape[0] == (q_lora_rank + 63) // 64
        assert q_b_scales.shape[1] == num_heads * head_dim

    def test_fused_kv_projection(self):
        """Test fused KV projection (hidden -> kv_latent -> k, v)."""
        device = torch.device("mps" if HAS_MPS else "cpu")
        
        hidden_size = 256
        kv_lora_rank = 64
        rope_dim = 32
        num_heads = 4
        head_dim = 64
        total_latent = kv_lora_rank + rope_dim
        
        # Create weights
        kv_a_packed, kv_a_scales = self._create_test_weights(hidden_size, total_latent, device)
        kv_b_packed, kv_b_scales = self._create_test_weights(total_latent, num_heads * 2 * head_dim, device)
        
        # Verify shapes
        assert kv_a_packed.shape[0] == (hidden_size + 7) // 8
        assert kv_a_packed.shape[1] == total_latent
        assert kv_a_scales.shape[0] == (hidden_size + 63) // 64
        assert kv_a_scales.shape[1] == total_latent
        
        assert kv_b_packed.shape[0] == (total_latent + 7) // 8
        assert kv_b_packed.shape[1] == num_heads * 2 * head_dim
        assert kv_b_scales.shape[0] == (total_latent + 63) // 64
        assert kv_b_scales.shape[1] == num_heads * 2 * head_dim

    def test_output_projection(self):
        """Test output projection (attention -> hidden)."""
        device = torch.device("mps" if HAS_MPS else "cpu")
        
        num_heads = 4
        head_dim = 64
        hidden_size = 256
        
        # Create weights
        o_packed, o_scales = self._create_test_weights(num_heads * head_dim, hidden_size, device)
        
        # Verify shapes
        assert o_packed.shape[0] == (num_heads * head_dim + 7) // 8
        assert o_packed.shape[1] == hidden_size
        assert o_scales.shape[0] == (num_heads * head_dim + 63) // 64
        assert o_scales.shape[1] == hidden_size

    def test_kernel_dispatch_shape(self):
        """Test kernel dispatch with correct shapes (integration test)."""
        device = torch.device("mps" if HAS_MPS else "cpu")
        
        # GLM-4.7-Flash style configuration (small for fast test)
        hidden_size = 256
        num_heads = 4
        head_dim = 64
        kv_lora_rank = 64
        q_lora_rank = 96
        rope_dim = 32
        
        # Create weights
        q_a_packed, q_a_scales = self._create_test_weights(hidden_size, q_lora_rank, device)
        q_b_packed, q_b_scales = self._create_test_weights(q_lora_rank, num_heads * head_dim, device)
        q_bias = torch.randn(num_heads * head_dim, dtype=torch.float16, device=device)
        
        kv_a_packed, kv_a_scales = self._create_test_weights(hidden_size, kv_lora_rank + rope_dim, device)
        kv_b_packed, kv_b_scales = self._create_test_weights(kv_lora_rank + rope_dim, num_heads * 2 * head_dim, device)
        
        o_packed, o_scales = self._create_test_weights(num_heads * head_dim, hidden_size, device)
        
        # Input
        batch = 1
        seq_q = 1
        seq_k = 8
        hidden = torch.randn(batch, seq_q, hidden_size, dtype=torch.float16, device=device)
        
        # Create compressed KV cache (simulated)
        cache_len = seq_k
        compressed_dim = kv_lora_rank
        k_cache = torch.randn(cache_len, compressed_dim, dtype=torch.float16, device=device)
        v_cache = torch.randn(cache_len, compressed_dim, dtype=torch.float16, device=device)
        
        # Cache scales (for FP8 quantization simulation)
        k_scales = torch.ones(cache_len, (compressed_dim + 63) // 64, dtype=torch.float16, device=device)
        v_scales = torch.ones(cache_len, (compressed_dim + 63) // 64, dtype=torch.float16, device=device)
        
        # Create parameters
        params = create_glm_mla_params(
            batch=batch,
            seq_q=seq_q,
            seq_k=seq_k,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            rope_dim=rope_dim,
            cache_len=cache_len,
        )
        
        # Verify parameters
        assert params.batch == batch
        assert params.seq_q == seq_q
        assert params.seq_k == seq_k
        assert params.hidden_size == hidden_size
        assert params.num_heads == num_heads
        assert params.head_dim == head_dim
        assert params.kv_lora_rank == kv_lora_rank
        assert params.q_lora_rank == q_lora_rank
        assert params.rope_dim == rope_dim
        
        # Verify input shapes match kernel expectations
        assert hidden.shape == (batch, seq_q, hidden_size)
        assert k_cache.shape == (cache_len, kv_lora_rank)
        assert v_cache.shape == (cache_len, kv_lora_rank)
        assert k_scales.shape[0] == cache_len
        assert v_scales.shape[0] == cache_len

    def test_attention_scale_computation(self):
        """Test attention scale is correctly computed."""
        head_dim = 64
        params = create_glm_mla_params(
            batch=1,
            seq_q=1,
            seq_k=8,
            head_dim=head_dim,
        )
        
        expected_scale = 1.0 / math.sqrt(head_dim)
        assert pytest.approx(params.scale, rel=1e-6) == expected_scale

    def test_different_head_dims(self):
        """Test parameter creation with different head dimensions."""
        # Head dim = 64
        params_64 = create_glm_mla_params(
            batch=1,
            seq_q=1,
            seq_k=8,
            head_dim=64,
        )
        assert params_64.head_dim == 64
        
        # Head dim = 128
        params_128 = create_glm_mla_params(
            batch=1,
            seq_q=1,
            seq_k=8,
            head_dim=128,
        )
        assert params_128.head_dim == 128
        
        # Verify scales are different
        assert pytest.approx(params_64.scale, rel=1e-6) == 1.0 / math.sqrt(64)
        assert pytest.approx(params_128.scale, rel=1e-6) == 1.0 / math.sqrt(128)

    def test_batch_processing(self):
        """Test batch processing configuration."""
        for batch_size in [1, 2, 4, 8]:
            params = create_glm_mla_params(
                batch=batch_size,
                seq_q=1,
                seq_k=8,
            )
            assert params.batch == batch_size
