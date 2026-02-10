"""Tests for MLAPagedAdapter.

Verifies that MLAPagedAdapter correctly connects TrellisKVCache to 
paged attention kernels and produces accurate results compared to 
standard TrellisMLAttention.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np

from metal_marlin.trellis.attention import TrellisMLAttention, TrellisMLAConfig
from metal_marlin.kv_cache import TrellisKVCache
from metal_marlin.paged.mla_paged_adapter import MLAPagedAdapter
from metal_marlin._compat import HAS_MPS, HAS_TORCH

# Skip if MPS is not available
pytestmark = pytest.mark.skipif(not HAS_MPS, reason="Requires PyTorch MPS")

class MockTrellisLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
            
    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

@pytest.fixture
def mla_setup():
    """Setup a small MLA attention layer and cache."""
    batch_size = 1
    num_heads = 4
    qk_nope_dim = 32
    qk_rope_dim = 16
    v_dim = 32
    kv_lora_rank = 64
    hidden_size = 128
    
    config = TrellisMLAConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_heads,
        qk_nope_head_dim=qk_nope_dim,
        qk_rope_head_dim=qk_rope_dim,
        v_head_dim=v_dim,
        kv_lora_rank=kv_lora_rank,
    )
    
    # Create mock projections
    q_a_proj = MockTrellisLinear(hidden_size, 64)
    q_b_proj = MockTrellisLinear(64, num_heads * (qk_nope_dim + qk_rope_dim))
    kv_a_proj = MockTrellisLinear(hidden_size, kv_lora_rank + qk_rope_dim)
    
    # kv_b_proj maps rank -> heads * (nope + v)
    kv_b_output_dim = num_heads * (qk_nope_dim + v_dim)
    kv_b_proj = MockTrellisLinear(kv_lora_rank, kv_b_output_dim)
    
    o_proj = MockTrellisLinear(num_heads * v_dim, hidden_size)
    
    attn_layer = TrellisMLAttention(
        config=config,
        q_a_proj=q_a_proj,
        q_b_proj=q_b_proj,
        kv_a_proj=kv_a_proj,
        kv_b_proj=kv_b_proj,
        o_proj=o_proj,
    ).to("mps", dtype=torch.float16)
    
    kv_cache = TrellisKVCache(
        num_layers=1,
        batch_size=batch_size,
        max_seq_len=128,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_dim,
        device="mps",
        dtype=torch.float16,
    )
    
    return attn_layer, kv_cache, config

def test_adapter_initialization(mla_setup):
    attn_layer, kv_cache, _ = mla_setup
    adapter = MLAPagedAdapter(kv_cache, attn_layer)
    assert adapter.kv_cache == kv_cache
    assert adapter.attn == attn_layer
    assert adapter.head_dim == 64 + 16 # kv_lora_rank + qk_rope_dim

def test_paged_attention_accuracy(mla_setup):
    attn_layer, kv_cache, config = mla_setup
    adapter = MLAPagedAdapter(kv_cache, attn_layer)
    
    batch_size = 1
    seq_len = 1
    hidden_size = 128
    
    # 1. Fill cache with some data
    hidden_states = torch.randn(batch_size, 16, hidden_size, device="mps", dtype=torch.float16)
    # We need to call attn_layer.forward to fill the cache properly if we were testing full flow,
    # but here we can just update the cache directly for unit testing the adapter.
    
    compressed_kv = torch.randn(batch_size, 16, 64 + 16, device="mps", dtype=torch.float16)
    kv_cache.update(0, compressed_kv=compressed_kv)
    
    # 2. Prepare query for a new token
    q_nope = torch.randn(batch_size, config.num_attention_heads, config.qk_nope_head_dim, device="mps", dtype=torch.float16)
    q_rope = torch.randn(batch_size, config.num_attention_heads, config.qk_rope_head_dim, device="mps", dtype=torch.float16)
    
    # 3. Compute attention using adapter
    output_paged = adapter.attention(q_nope, q_rope, layer_idx=0)
    
    assert output_paged.shape == (batch_size, config.num_attention_heads, config.v_head_dim)
    
    # 4. Compute reference (manually decompress and use standard SDPA)
    # Get full cache
    compressed_full = kv_cache.kv_cache[0, :, :16] # [batch, seq, dim]
    c_kv, k_pe = torch.split(compressed_full, [64, 16], dim=-1)
    
    # Decompress
    kv_decomp = attn_layer.kv_b_proj(c_kv) # [batch, seq, heads*(nope+v)]
    kv_decomp = kv_decomp.view(batch_size, 16, config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim).transpose(1, 2)
    k_nope, v = torch.split(kv_decomp, [config.qk_nope_head_dim, config.v_head_dim], dim=-1)
    
    # Concat K
    k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, config.num_attention_heads, -1).transpose(1, 2)
    k = torch.cat([k_nope, k_pe_expanded], dim=-1) # [batch, heads, seq, head_dim]
    
    # Q
    q = torch.cat([q_nope.unsqueeze(2), q_rope.unsqueeze(2)], dim=-1) # [batch, heads, 1, head_dim]
    
    # Attention
    scale = config.qk_head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output_ref = torch.matmul(attn_weights, v).squeeze(2)
    
    # Compare
    torch.testing.assert_close(output_paged, output_ref, atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("quant_mode", ["fp8", "fp4", "int4"])
def test_paged_attention_quantized(mla_setup, quant_mode):
    attn_layer, _, config = mla_setup
    
    # Create quantized cache
    # Note: MLAKVCache might need some fixes to support fp4 if it's not implemented yet,
    # but the adapter assumes it exists or uses CompressedKVCacheMLA.
    
    try:
        kv_cache = TrellisKVCache(
            num_layers=1,
            batch_size=1,
            max_seq_len=64,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            device="mps",
            dtype=torch.float16,
            quantize_mode=quant_mode,
        )
    except (ValueError, NotImplementedError):
        pytest.skip(f"Quantization mode {quant_mode} not supported by TrellisKVCache")

    adapter = MLAPagedAdapter(kv_cache, attn_layer)
    
    compressed_kv = torch.randn(1, 16, 64 + 16, device="mps", dtype=torch.float16)
    kv_cache.update(0, compressed_kv=compressed_kv)
    
    q_nope = torch.randn(1, config.num_attention_heads, config.qk_nope_head_dim, device="mps", dtype=torch.float16)
    q_rope = torch.randn(1, config.num_attention_heads, config.qk_rope_head_dim, device="mps", dtype=torch.float16)
    
    output = adapter.attention(q_nope, q_rope, layer_idx=0)
    assert output.shape == (1, config.num_attention_heads, config.v_head_dim)
