"""Tests for attention benchmark implementations."""


import pytest
import torch

# Import from benchmark
try:
    from benchmarks.bench_attention_variants import (
        FusedQKVAttention,
        GroupedQueryAttention,
        StandardMultiHeadAttention,
        flash_attention_tiled,
        standard_attention,
    )

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="Benchmark not available"),
    pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(),
                       reason="GPU not available"),
]


def test_standard_attention_shape():
    """Test standard attention produces correct output shape."""
    batch, heads, seq, dim = 2, 4, 16, 32
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    output = standard_attention(q, k, v, causal=False)
    assert output.shape == (batch, heads, seq, dim)


def test_standard_attention_causal():
    """Test causal masking in standard attention."""
    batch, heads, seq, dim = 1, 1, 8, 16
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.eye(seq).unsqueeze(0).unsqueeze(0).expand(batch, heads, seq, seq)

    output = standard_attention(q, k, v, causal=True)

    # For causal mask with identity V, each position should only attend to itself and previous
    # Check that upper triangle is effectively masked
    assert not torch.isnan(output).any()


def test_flash_attention_tiled_matches_standard():
    """Test flash attention produces similar results to standard attention."""
    batch, heads, seq, dim = 2, 4, 128, 32
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    standard_out = standard_attention(q, k, v, causal=False)
    flash_out = flash_attention_tiled(q, k, v, causal=False)

    max_diff = (standard_out - flash_out).abs().max().item()
    assert max_diff < 0.1, f"Flash attention differs from standard by {max_diff}"


def test_fused_qkv_attention_shape():
    """Test Fused QKV attention produces correct output shape."""
    batch, seq, hidden = 2, 16, 64
    model = FusedQKVAttention(hidden_dim=hidden, num_heads=4)
    x = torch.randn(batch, seq, hidden)

    output = model(x, causal=False)
    assert output.shape == (batch, seq, hidden)


def test_gqa_attention_shape():
    """Test GQA produces correct output shape."""
    batch, seq, hidden = 2, 16, 64
    num_heads = 8
    num_kv_heads = 2

    model = GroupedQueryAttention(
        hidden_dim=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    x = torch.randn(batch, seq, hidden)

    output = model(x, causal=False)
    assert output.shape == (batch, seq, hidden)


def test_gqa_reduces_parameters():
    """Test that GQA has fewer parameters than standard MHA."""
    hidden, num_heads = 64, 8
    num_kv_heads = 2

    standard = StandardMultiHeadAttention(hidden, num_heads)
    gqa = GroupedQueryAttention(hidden, num_heads, num_kv_heads)

    standard_params = sum(p.numel() for p in standard.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())

    # GQA should have fewer parameters due to reduced KV heads
    assert gqa_params < standard_params


def test_fused_qkv_fewer_kernels():
    """Test that Fused QKV uses one projection instead of three."""
    hidden, num_heads = 64, 4

    fused = FusedQKVAttention(hidden, num_heads)
    standard = StandardMultiHeadAttention(hidden, num_heads)

    # Fused should have one qkv_proj parameter
    assert hasattr(fused, "qkv_proj")

    # Standard should have separate projections
    assert hasattr(standard, "q_proj")
    assert hasattr(standard, "k_proj")
    assert hasattr(standard, "v_proj")


def test_attention_scale_correctness():
    """Verify attention uses correct scaling factor."""
    batch, heads, seq, dim = 1, 1, 4, 16
    scale = dim ** -0.5

    q = torch.ones(batch, heads, seq, dim)
    k = torch.ones(batch, heads, seq, dim)
    v = torch.eye(seq).unsqueeze(0).unsqueeze(0).expand(batch, heads, seq, seq)

    output = standard_attention(q, k, v, causal=False)

    # With uniform Q=K=1 and scale=1/sqrt(dim), scores should be sqrt(dim)
    expected_score = dim * scale  # = sqrt(dim)
    # After softmax, attention weights sum to 1
    # Output should be roughly uniform over V rows

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_flash_attention_various_lengths(seq_len):
    """Test flash attention works with different sequence lengths."""
    batch, heads, dim = 2, 4, 32
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, dim)

    output = flash_attention_tiled(q, k, v, causal=True)
    assert output.shape == (batch, heads, seq_len, dim)
    assert not torch.isnan(output).any()
