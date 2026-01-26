import time

import torch

from metal_marlin.kernels import HAS_METAL, HAS_MPS, marlin_gemm_fp4


def main() -> None:
    assert HAS_METAL and HAS_MPS, "Metal not available"

    # Test dimensions matching Qwen3-4B
    m, k, n = 1, 2560, 10240
    a = torch.randn(m, k, dtype=torch.float16, device="mps")
    packed = torch.randint(0, 2**32 - 1, (k // 8, n), dtype=torch.int32, device="mps")
    scales = torch.randn(k // 128, n, dtype=torch.float16, device="mps")

    # Run fused kernel
    out = marlin_gemm_fp4(a, packed, scales, group_size=128)
    print(f"Output: {out.shape}, {out.dtype}")

    # Benchmark
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        out = marlin_gemm_fp4(a, packed, scales, group_size=128)
    torch.mps.synchronize()
    elapsed = (time.perf_counter() - t0) / 100
    print(f"Fused FP4 GEMM: {elapsed * 1e6:.1f} us")


if __name__ == "__main__":
    main()
