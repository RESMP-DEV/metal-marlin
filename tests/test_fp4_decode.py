"""
Tests for FP4 M=1 decode kernels (dequant_fp4_fast.metal).

These tests verify the specialized decode path for single-token inference,
which uses LUT-based dequantization and optimized GEMV kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

# Optional imports - tests skip if Metal implementation not available
try:
    import torch
    from metal_marlin.kernels import dequant_fp4_decode_gemv
    from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS

    HAS_FP4_DECODE = HAS_METAL and HAS_MPS
except ImportError:
    HAS_FP4_DECODE = False
    torch = None

pytestmark = [
    pytest.mark.skipif(not HAS_FP4_DECODE, reason="FP4 decode requires Metal/MPS"),
]


class TestFP4DecodeGEMV:
    """Test M=1 FP4 decode GEMV kernel."""

    def test_decode_basic(self):
        """Basic decode GEMV test with small dimensions."""
        K, N = 128, 256
        group_size = 32

        # Create random input
        A = torch.randn(K, dtype=torch.float16, device="mps")

        # Create packed FP4 weights (K/8, N) uint32
        B_packed = torch.randint(
            0, 2**32, (K // 8, N), dtype=torch.int32, device="mps"
        )

        # Create scales (K/group_size, N) float16
        scales = torch.randn(
            K // group_size, N, dtype=torch.float16, device="mps"
        ).abs()

        # Run decode GEMV
        C = dequant_fp4_decode_gemv(A, B_packed, scales, K, N, group_size)

        # Verify output shape
        assert C.shape == (N,)
        assert C.dtype == torch.float16
        assert C.device.type == "mps"

    def test_decode_matches_reference(self):
        """Verify decode kernel matches CPU reference implementation."""
        K, N = 64, 128
        group_size = 32

        # Create test input
        A = torch.randn(K, dtype=torch.float32, device="cpu")

        # Create packed FP4 weights - use simple pattern for verification
        # Pack 8 FP4 values (0-15) into each uint32
        fp4_values = torch.randint(0, 16, (K, N), dtype=torch.int32, device="cpu")
        B_packed = torch.zeros((K // 8, N), dtype=torch.int32, device="cpu")
        for i in range(8):
            B_packed += fp4_values[i::8, :] << (i * 4)
        B_packed = B_packed.to("mps")

        # Create scales
        scales = torch.ones(K // group_size, N, dtype=torch.float32, device="cpu")
        scales_mps = scales.half().to("mps")

        # E2M1 FP4 lookup table
        e2m1_table = torch.tensor([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ], dtype=torch.float32)

        # CPU reference: dequantize and multiply
        # Dequantize weights
        W_dequant = e2m1_table[fp4_values]  # (K, N)

        # Apply scales
        scale_expanded = scales.repeat_interleave(group_size, dim=0)[:K, :]
        W_dequant = W_dequant * scale_expanded

        # GEMV: A @ W
        C_ref = A @ W_dequant  # (N,)

        # Run Metal kernel
        A_mps = A.half().to("mps")
        C_metal = dequant_fp4_decode_gemv(A_mps, B_packed, scales_mps, K, N, group_size)
        C_metal_cpu = C_metal.float().cpu()

        # Compare (allow some tolerance due to FP16 vs FP32)
        torch.testing.assert_close(C_metal_cpu, C_ref, rtol=1e-2, atol=1e-2)

    def test_decode_various_dimensions(self):
        """Test decode with various K and N dimensions."""
        test_cases = [
            (64, 64, 32),
            (128, 256, 32),
            (256, 512, 64),
            (512, 1024, 128),
        ]

        for K, N, group_size in test_cases:
            A = torch.randn(K, dtype=torch.float16, device="mps")
            B_packed = torch.randint(
                0, 2**32, (K // 8, N), dtype=torch.int32, device="mps"
            )
            scales = torch.randn(
                K // group_size, N, dtype=torch.float16, device="mps"
            ).abs()

            C = dequant_fp4_decode_gemv(A, B_packed, scales, K, N, group_size)

            assert C.shape == (N,), f"Failed for K={K}, N={N}, gs={group_size}"

    def test_decode_non_divisible_n(self):
        """Test decode where N is not a multiple of tile size (512)."""
        K, N = 128, 300  # 300 < 512, only one threadgroup needed
        group_size = 32

        A = torch.randn(K, dtype=torch.float16, device="mps")
        B_packed = torch.randint(
            0, 2**32, (K // 8, N), dtype=torch.int32, device="mps"
        )
        scales = torch.randn(
            K // group_size, N, dtype=torch.float16, device="mps"
        ).abs()

        C = dequant_fp4_decode_gemv(A, B_packed, scales, K, N, group_size)

        assert C.shape == (N,)

    def test_decode_large_n(self):
        """Test decode with large N requiring multiple threadgroups."""
        K, N = 128, 2048  # Requires ceil(2048/512) = 4 threadgroups
        group_size = 32

        A = torch.randn(K, dtype=torch.float16, device="mps")
        B_packed = torch.randint(
            0, 2**32, (K // 8, N), dtype=torch.int32, device="mps"
        )
        scales = torch.randn(
            K // group_size, N, dtype=torch.float16, device="mps"
        ).abs()

        C = dequant_fp4_decode_gemv(A, B_packed, scales, K, N, group_size)

        assert C.shape == (N,)

    def test_decode_all_zeros(self):
        """Test decode with all-zero weights (FP4 code 0)."""
        K, N = 128, 256
        group_size = 32

        A = torch.randn(K, dtype=torch.float16, device="mps")
        # All zeros in FP4 = 0.0
        B_packed = torch.zeros((K // 8, N), dtype=torch.int32, device="mps")
        scales = torch.ones(K // group_size, N, dtype=torch.float16, device="mps")

        C = dequant_fp4_decode_gemv(A, B_packed, scales, K, N, group_size)

        # With zero weights, output should be zero
        assert torch.allclose(C, torch.zeros_like(C), atol=1e-3)

    def test_decode_max_values(self):
        """Test decode with maximum FP4 weights (code 7 = +6.0, code 15 = -6.0)."""
        K, N = 64, 64
        group_size = 32

        A = torch.ones(K, dtype=torch.float16, device="mps")
        # All 7s in FP4 = +6.0 (0x77777777 = 8 nibbles of 7)
        B_packed = torch.full(
            (K // 8, N), 0x77777777, dtype=torch.int32, device="mps"
        )
        scales = torch.ones(K // group_size, N, dtype=torch.float16, device="mps")

        C = dequant_fp4_decode_gemv(A, B_packed, scales, K, N, group_size)

        # Each output should be K * 6.0 = 64 * 6.0 = 384.0
        # (each K contributes 6.0, and all K are 6.0 after dequant)
        expected = torch.full((N,), K * 6.0, dtype=torch.float16, device="mps")
        assert torch.allclose(C, expected, rtol=1e-2)


class TestFP4DecodeKernelAvailability:
    """Test kernel availability checks."""

    def test_kernel_compiles(self):
        """Verify the kernel can be loaded from source."""
        from metal_marlin.metal_dispatch import MetalKernelLibrary, get_shader_source

        lib = MetalKernelLibrary.from_source_dir()
        source = get_shader_source("dequant_fp4_fast")

        # Should compile without error
        lib.compile_source("dequant_fp4_fast", source)

        # Should be able to get the kernel function
        try:
            pipeline = lib.get_pipeline("dequant_fp4_decode_gemv")
            assert pipeline is not None
        except KeyError:
            pytest.fail("dequant_fp4_decode_gemv kernel not found after compilation")
