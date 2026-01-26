#!/usr/bin/env python
"""Test fused Metal kernel availability and performance."""

try:
    from metal_marlin.kernels import HAS_METAL, HAS_MPS, marlin_gemm_fp4

    print(f"HAS_METAL: {HAS_METAL}")
    print(f"HAS_MPS: {HAS_MPS}")

    if HAS_METAL and HAS_MPS:
        print("Fused kernels available!")
        import time

        import torch

        M, K, N = 1, 2560, 10240  # Single token, Qwen3-4B MLP up_proj
        A = torch.randn(M, K, dtype=torch.float16, device="mps")

        # For FP4, packed is [K//8, N], scales is [K//128, N]
        B_packed = torch.randint(0, 255, (K // 8, N), dtype=torch.uint8, device="mps")
        scales = torch.randn(K // 128, N, dtype=torch.float16, device="mps")

        # Warmup
        for _ in range(3):
            C = marlin_gemm_fp4(A, B_packed, scales, group_size=128)
            torch.mps.synchronize()

        # Benchmark
        iters = 100
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            C = marlin_gemm_fp4(A, B_packed, scales, group_size=128)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        print(f"marlin_gemm_fp4 ({M}x{K} @ {K}x{N})")
        print(f"  {elapsed / iters * 1e6:.1f} us per call")
        print(f"  {iters / elapsed:.0f} calls/sec")
    else:
        print("Fused kernels NOT available")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"Error: {e}")
