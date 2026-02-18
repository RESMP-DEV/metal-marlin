
import pytest
import torch
from metal_marlin.kernels import mmfp4_gemm, HAS_MPS, HAS_METAL

@pytest.mark.skipif(not (HAS_MPS and HAS_METAL), reason="Requires MPS and Metal")
def test_mmfp4_gemm_validation_m_zero():
    """Test that mmfp4_gemm asserts M > 0."""
    device = torch.device("mps")
    M = 0
    K = 32
    N = 32
    group_size = 32
    
    A = torch.zeros((M, K), dtype=torch.float16, device=device)
    # B packed: [K/8, N]
    B_packed = torch.zeros((K // 8, N), dtype=torch.int32, device=device) # uint32 via int32 in torch
    B_scales = torch.zeros((K // group_size, N), dtype=torch.float16, device=device)
    
    with pytest.raises(AssertionError, match="M must be > 0"):
        mmfp4_gemm(A, B_packed, B_scales, group_size)

@pytest.mark.skipif(not (HAS_MPS and HAS_METAL), reason="Requires MPS and Metal")
def test_mmfp4_gemm_validation_n_zero():
    """Test that mmfp4_gemm asserts N > 0."""
    device = torch.device("mps")
    M = 1
    K = 32
    N = 0
    group_size = 32
    
    A = torch.zeros((M, K), dtype=torch.float16, device=device)
    # B packed: [K/8, N] -> [4, 0]
    B_packed = torch.zeros((K // 8, N), dtype=torch.int32, device=device)
    B_scales = torch.zeros((K // group_size, N), dtype=torch.float16, device=device)
    
    with pytest.raises(AssertionError, match="N must be > 0"):
        mmfp4_gemm(A, B_packed, B_scales, group_size)
