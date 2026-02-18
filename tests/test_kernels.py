"""Tests for kernel functions in metal_marlin.kernels.

This module tests the dispatch mechanisms and optimizations for MMFP4 kernels,
including the fused QKV projection kernel.
"""

from __future__ import annotations

import pytest
import torch

from metal_marlin.kernels import (
    HAS_METAL,
    HAS_MPS,
    dequantize_mmfp4,
    mmfp4_fused_qkv,
    mmfp4_gemm,
    pack_fp4_weights,
    reusable_command_buffer,
    submit_batch,
)

requires_mps = pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
requires_metal = pytest.mark.skipif(not HAS_METAL, reason="Requires Metal/PyObjC")


def test_mmfp4_fused_qkv_import():
    """Verify mmfp4_fused_qkv function is properly exported."""
    assert callable(mmfp4_fused_qkv)
    
    if HAS_METAL and HAS_MPS:
        # Real implementation should have proper docstring
        assert mmfp4_fused_qkv.__doc__ is not None
        assert "Fused MMFP4" in mmfp4_fused_qkv.__doc__ or "mmfp4" in mmfp4_fused_qkv.__doc__.lower()
    else:
        # Stub function in non-Metal environment
        assert mmfp4_fused_qkv.__doc__ is not None


def test_mmfp4_fused_qkv_signature():
    """Verify mmfp4_fused_qkv has the correct function signature."""
    import inspect
    sig = inspect.signature(mmfp4_fused_qkv)
    params = list(sig.parameters.keys())
    
    if HAS_METAL and HAS_MPS:
        # Real implementation has named parameters
        assert "A" in params
        assert "Wq_packed" in params
        assert "Wq_scales" in params
        assert "Wk_packed" in params
        assert "Wk_scales" in params
        assert "Wv_packed" in params
        assert "Wv_scales" in params
        assert "group_size" in params
        # Check group_size has default value
        assert sig.parameters["group_size"].default == 128
    else:
        # Stub function accepts *args, **kwargs
        assert "args" in params or any(p for p in params if p in ["A", "args"])


def test_mmfp4_fused_qkv_stub_raises():
    """Test that stub function raises ImportError when Metal is unavailable."""
    if HAS_METAL and HAS_MPS:
        pytest.skip("Metal is available, skipping stub test")
    
    # Stub should raise ImportError
    with pytest.raises(ImportError):
        mmfp4_fused_qkv(None, None, None, None, None, None, None)


@requires_mps
@requires_metal
def test_mmfp4_fused_qkv_basic_dispatch():
    """Test basic dispatch of mmfp4_fused_qkv with M=1 (decode phase)."""
    device = torch.device("mps")
    torch.manual_seed(42)
    
    # Dimensions for decode phase (M=1)
    M = 1
    K = 512
    Nq = 128
    Nk = 64
    Nv = 64
    group_size = 32
    
    # Create input activation
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    
    # Create packed weights and scales for Q, K, V
    Wq = torch.randn((K, Nq), dtype=torch.float16, device=device)
    Wk = torch.randn((K, Nk), dtype=torch.float16, device=device)
    Wv = torch.randn((K, Nv), dtype=torch.float16, device=device)
    
    Wq_packed, Wq_scales = pack_fp4_weights(Wq.T.contiguous(), group_size=group_size)
    Wk_packed, Wk_scales = pack_fp4_weights(Wk.T.contiguous(), group_size=group_size)
    Wv_packed, Wv_scales = pack_fp4_weights(Wv.T.contiguous(), group_size=group_size)
    
    # Call fused QKV
    Q, K_out, V = mmfp4_fused_qkv(
        A,
        Wq_packed, Wq_scales,
        Wk_packed, Wk_scales,
        Wv_packed, Wv_scales,
        group_size=group_size
    )
    
    # Verify output shapes
    assert Q.shape == (M, Nq), f"Expected Q shape {(M, Nq)}, got {Q.shape}"
    assert K_out.shape == (M, Nk), f"Expected K shape {(M, Nk)}, got {K_out.shape}"
    assert V.shape == (M, Nv), f"Expected V shape {(M, Nv)}, got {V.shape}"
    
    # Verify outputs are finite
    assert torch.isfinite(Q).all(), "Q contains non-finite values"
    assert torch.isfinite(K_out).all(), "K contains non-finite values"
    assert torch.isfinite(V).all(), "V contains non-finite values"


@requires_mps
@requires_metal
def test_mmfp4_fused_qkv_batched_dispatch():
    """Test mmfp4_fused_qkv with M > 1 (falls back to separate GEMMs)."""
    device = torch.device("mps")
    torch.manual_seed(43)
    
    # Dimensions for batched (M > 1)
    M = 4
    K = 512
    Nq = 128
    Nk = 64
    Nv = 64
    group_size = 32
    
    # Create input activation
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    
    # Create packed weights and scales for Q, K, V
    Wq = torch.randn((K, Nq), dtype=torch.float16, device=device)
    Wk = torch.randn((K, Nk), dtype=torch.float16, device=device)
    Wv = torch.randn((K, Nv), dtype=torch.float16, device=device)
    
    Wq_packed, Wq_scales = pack_fp4_weights(Wq.T.contiguous(), group_size=group_size)
    Wk_packed, Wk_scales = pack_fp4_weights(Wk.T.contiguous(), group_size=group_size)
    Wv_packed, Wv_scales = pack_fp4_weights(Wv.T.contiguous(), group_size=group_size)
    
    # Call fused QKV
    Q, K_out, V = mmfp4_fused_qkv(
        A,
        Wq_packed, Wq_scales,
        Wk_packed, Wk_scales,
        Wv_packed, Wv_scales,
        group_size=group_size
    )
    
    # Verify output shapes
    assert Q.shape == (M, Nq), f"Expected Q shape {(M, Nq)}, got {Q.shape}"
    assert K_out.shape == (M, Nk), f"Expected K shape {(M, Nk)}, got {K_out.shape}"
    assert V.shape == (M, Nv), f"Expected V shape {(M, Nv)}, got {V.shape}"
    
    # Verify outputs are finite
    assert torch.isfinite(Q).all(), "Q contains non-finite values"
    assert torch.isfinite(K_out).all(), "K contains non-finite values"
    assert torch.isfinite(V).all(), "V contains non-finite values"


@requires_mps
@requires_metal
def test_mmfp4_fused_qkv_matches_separate_gemms():
    """Verify fused QKV produces same results as separate GEMM calls."""
    device = torch.device("mps")
    torch.manual_seed(44)
    
    # Use M > 1 to test fallback path
    M = 2
    K = 512
    Nq = 128
    Nk = 64
    Nv = 64
    group_size = 32
    
    # Create input activation
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    
    # Create packed weights and scales for Q, K, V
    Wq = torch.randn((K, Nq), dtype=torch.float16, device=device)
    Wk = torch.randn((K, Nk), dtype=torch.float16, device=device)
    Wv = torch.randn((K, Nv), dtype=torch.float16, device=device)
    
    Wq_packed, Wq_scales = pack_fp4_weights(Wq.T.contiguous(), group_size=group_size)
    Wk_packed, Wk_scales = pack_fp4_weights(Wk.T.contiguous(), group_size=group_size)
    Wv_packed, Wv_scales = pack_fp4_weights(Wv.T.contiguous(), group_size=group_size)
    
    # Call fused QKV
    Q_fused, K_fused, V_fused = mmfp4_fused_qkv(
        A,
        Wq_packed, Wq_scales,
        Wk_packed, Wk_scales,
        Wv_packed, Wv_scales,
        group_size=group_size
    )
    
    # Compute reference with separate GEMMs
    Q_ref = mmfp4_gemm(A, Wq_packed, Wq_scales, group_size=group_size)
    K_ref = mmfp4_gemm(A, Wk_packed, Wk_scales, group_size=group_size)
    V_ref = mmfp4_gemm(A, Wv_packed, Wv_scales, group_size=group_size)
    
    # Verify outputs match (with tolerance for FP4 quantization)
    torch.testing.assert_close(Q_fused.float(), Q_ref.float(), rtol=0.1, atol=0.5)
    torch.testing.assert_close(K_fused.float(), K_ref.float(), rtol=0.1, atol=0.5)
    torch.testing.assert_close(V_fused.float(), V_ref.float(), rtol=0.1, atol=0.5)


@requires_mps
@requires_metal
def test_mmfp4_fused_qkv_input_validation():
    """Test input validation for mmfp4_fused_qkv."""
    device = torch.device("mps")
    
    # Test with non-2D input
    A_invalid = torch.randn((1, 2, 3), dtype=torch.float16, device=device)
    W_packed = torch.randint(0, 100, (64, 128), dtype=torch.uint32, device=device)
    W_scales = torch.randn((16, 128), dtype=torch.float16, device=device)
    
    with pytest.raises(ValueError, match="A must be 2D"):
        mmfp4_fused_qkv(
            A_invalid,
            W_packed, W_scales,
            W_packed, W_scales,
            W_packed, W_scales,
            group_size=32
        )


@requires_mps
@requires_metal
def test_mmfp4_fused_qkv_different_group_sizes():
    """Test mmfp4_fused_qkv with different group sizes."""
    device = torch.device("mps")
    torch.manual_seed(45)
    
    M = 1
    K = 1024
    Nq = 128
    Nk = 64
    Nv = 64
    
    for group_size in [32, 64, 128]:
        # Skip if K is not divisible by group_size
        if K % group_size != 0:
            continue
            
        A = torch.randn((M, K), dtype=torch.float16, device=device)
        
        Wq = torch.randn((K, Nq), dtype=torch.float16, device=device)
        Wk = torch.randn((K, Nk), dtype=torch.float16, device=device)
        Wv = torch.randn((K, Nv), dtype=torch.float16, device=device)
        
        Wq_packed, Wq_scales = pack_fp4_weights(Wq.T.contiguous(), group_size=group_size)
        Wk_packed, Wk_scales = pack_fp4_weights(Wk.T.contiguous(), group_size=group_size)
        Wv_packed, Wv_scales = pack_fp4_weights(Wv.T.contiguous(), group_size=group_size)
        
        Q, K_out, V = mmfp4_fused_qkv(
            A,
            Wq_packed, Wq_scales,
            Wk_packed, Wk_scales,
            Wv_packed, Wv_scales,
            group_size=group_size
        )
        
        assert Q.shape == (M, Nq)
        assert K_out.shape == (M, Nk)
        assert V.shape == (M, Nv)
        assert torch.isfinite(Q).all()
        assert torch.isfinite(K_out).all()
        assert torch.isfinite(V).all()


@requires_mps
@requires_metal
def test_mmfp4_fused_qkv_large_dimensions():
    """Test mmfp4_fused_qkv with larger dimensions typical of real models."""
    device = torch.device("mps")
    torch.manual_seed(46)
    
    # Typical LLM dimensions
    M = 1
    K = 4096
    Nq = 4096  # Q dim typically larger for GQA
    Nk = 1024  # K dim smaller with GQA
    Nv = 1024  # V dim same as K
    group_size = 128
    
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    
    Wq = torch.randn((K, Nq), dtype=torch.float16, device=device)
    Wk = torch.randn((K, Nk), dtype=torch.float16, device=device)
    Wv = torch.randn((K, Nv), dtype=torch.float16, device=device)
    
    Wq_packed, Wq_scales = pack_fp4_weights(Wq.T.contiguous(), group_size=group_size)
    Wk_packed, Wk_scales = pack_fp4_weights(Wk.T.contiguous(), group_size=group_size)
    Wv_packed, Wv_scales = pack_fp4_weights(Wv.T.contiguous(), group_size=group_size)
    
    Q, K_out, V = mmfp4_fused_qkv(
        A,
        Wq_packed, Wq_scales,
        Wk_packed, Wk_scales,
        Wv_packed, Wv_scales,
        group_size=group_size
    )
    
    assert Q.shape == (M, Nq)
    assert K_out.shape == (M, Nk)
    assert V.shape == (M, Nv)
    assert torch.isfinite(Q).all()
    assert torch.isfinite(K_out).all()
    assert torch.isfinite(V).all()


@requires_mps
@requires_metal
def test_mmfp4_reusable_command_buffer_dispatch():
    """Test dispatching multiple kernels within a reusable_command_buffer context."""
    device = torch.device("mps")
    torch.manual_seed(47)

    M = 4
    K = 128
    N = 64
    group_size = 32

    A = torch.randn((M, K), dtype=torch.float16, device=device)
    W = torch.randn((K, N), dtype=torch.float16, device=device)
    W_packed, W_scales = pack_fp4_weights(W.T.contiguous(), group_size=group_size)

    # Run multiple dispatches in one command buffer
    # Note: mmfp4_gemm uses get_default_library() internally
    with reusable_command_buffer():
        C1 = mmfp4_gemm(A, W_packed, W_scales, group_size=group_size)
        C2 = mmfp4_gemm(A, W_packed, W_scales, group_size=group_size)

    assert C1.shape == (M, N)
    assert C2.shape == (M, N)
    assert torch.isfinite(C1).all()
    assert torch.isfinite(C2).all()
    
    # Verify outputs match
    torch.testing.assert_close(C1, C2)


@requires_mps
@requires_metal
def test_mmfp4_submit_batch():
    """Test submit_batch functionality."""
    device = torch.device("mps")
    torch.manual_seed(48)

    M = 4
    K = 128
    N = 64
    group_size = 32

    A = torch.randn((M, K), dtype=torch.float16, device=device)
    W = torch.randn((K, N), dtype=torch.float16, device=device)
    W_packed, W_scales = pack_fp4_weights(W.T.contiguous(), group_size=group_size)

    results = []

    def run_kernel():
        res = mmfp4_gemm(A, W_packed, W_scales, group_size=group_size)
        results.append(res)

    submit_batch([run_kernel, run_kernel])

    assert len(results) == 2
    C1, C2 = results[0], results[1]

    assert C1.shape == (M, N)
    assert C2.shape == (M, N)
    assert torch.isfinite(C1).all()
    assert torch.isfinite(C2).all()
    
    # Verify outputs match
    torch.testing.assert_close(C1, C2)

