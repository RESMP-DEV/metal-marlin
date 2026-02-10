"""Accuracy tests for MMFP4 GEMM kernel."""
import pytest
import torch

from metal_marlin.kernels import mmfp4_gemm
from metal_marlin.quantize_fp4 import quantize_fp4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
class TestMMFP4GEMMAccuracy:
    """Test MMFP4 GEMM produces non-zero, correct results."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 2048, 2048),      # Decode: single token
        (128, 2048, 2048),    # Prefill: small batch
        (512, 2048, 8192),   # Prefill: up projection
    ])
    def test_gemm_nonzero(self, M, K, N):
        """Verify GEMM output is non-zero."""
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        W = torch.randn(K, N, dtype=torch.float16, device="cpu")
        
        # Quantize weights
        W_packed, scales = quantize_fp4(W.t(), group_size=128)
        W_packed = W_packed.to("mps")
        scales = scales.to("mps")
        
        # Run kernel
        output = mmfp4_gemm(A, W_packed, scales, group_size=128)
        
        assert output.shape == (M, N)
        assert output.abs().sum() > 0, "GEMM output is all zeros!"
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_gemm_vs_torch(self):
        """Compare MMFP4 GEMM against torch.mm reference."""
        M, K, N = 1, 2048, 2048
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        W = torch.randn(K, N, dtype=torch.float16, device="cpu")
        
        # Reference: fp16 matmul
        ref = torch.mm(A.cpu(), W).to("mps")
        
        # Quantized
        W_packed, scales = quantize_fp4(W.t(), group_size=128)
        W_packed = W_packed.to("mps")
        scales = scales.to("mps")
        output = mmfp4_gemm(A, W_packed, scales, group_size=128)
        
        # FP4 has ~0.5-1 bit precision loss, allow larger tolerance
        rel_error = (output - ref).abs().mean() / ref.abs().mean()
        assert rel_error < 0.2, f"Relative error {rel_error:.4f} too high"
