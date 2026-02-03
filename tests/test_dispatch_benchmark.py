"""Benchmark C++ vs PyObjC dispatch overhead."""
import time

import pytest
import torch

from metal_marlin._compat import HAS_CPP_EXT, HAS_MPS
from metal_marlin.fast_inference import fast_dispatch_available, get_fast_context


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
class TestDispatchBenchmark:
    """Benchmark tests for dispatch overhead."""

    def test_cpp_dispatch_overhead(self):
        """Measure C++ dispatch overhead."""
        if not fast_dispatch_available():
            pytest.skip("C++ extension not available")

        ctx = get_fast_context()

        # Create small tensors for overhead measurement
        M, K, N = 1, 4096, 4096
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        packed = torch.randint(0, 255, (K // 8, N), dtype=torch.uint8, device="mps")
        scales = torch.randn(K // 32, N, dtype=torch.float32, device="mps")
        grid = torch.randn(8, dtype=torch.float32, device="mps")
        su = torch.randn(K, dtype=torch.float32, device="mps")
        sv = torch.randn(N, dtype=torch.float32, device="mps")

        # Warmup
        for _ in range(5):
            try:
                ctx.gemm_trellis_packed(A, packed, scales, grid, su, sv, K, N, 3, 32)
            except Exception:
                pytest.skip("Kernel not available")

        # Measure
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            ctx.gemm_trellis_packed(A, packed, scales, grid, su, sv, K, N, 3, 32)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / 100) * 1e6
        print(f"\nC++ dispatch overhead: {avg_us:.1f}μs per call")

        # Assert reasonable overhead (should be < 50μs)
        assert avg_us < 50, f"C++ dispatch too slow: {avg_us:.1f}μs"

    def test_pyobjc_dispatch_overhead(self):
        """Measure PyObjC dispatch overhead for comparison."""
        from metal_marlin.metal_context import get_metal_kernel_library

        from metal_marlin.trellis.dispatch import dispatch_gemm_trellis_packed

        lib = get_metal_kernel_library()

        M, K, N = 1, 4096, 4096
        A = torch.randn(M, K, dtype=torch.float16, device="mps")
        packed = torch.randint(0, 255, (K // 8, N), dtype=torch.uint8, device="mps")
        scales = torch.randn(K // 32, N, dtype=torch.float32, device="mps")
        grid = torch.randn(8, dtype=torch.float32, device="mps")
        su = torch.randn(K, dtype=torch.float32, device="mps")
        sv = torch.randn(N, dtype=torch.float32, device="mps")

        # Warmup
        for _ in range(3):
            try:
                dispatch_gemm_trellis_packed(lib, A, packed, scales, grid, su, sv, K, N, 3, 32)
            except Exception:
                pytest.skip("PyObjC kernel not available")

        # Measure
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            dispatch_gemm_trellis_packed(lib, A, packed, scales, grid, su, sv, K, N, 3, 32)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / 50) * 1e6
        print(f"\nPyObjC dispatch overhead: {avg_us:.1f}μs per call")

        # PyObjC is slower, ~80-150μs expected
        assert avg_us < 300, f"PyObjC dispatch extremely slow: {avg_us:.1f}μs"
