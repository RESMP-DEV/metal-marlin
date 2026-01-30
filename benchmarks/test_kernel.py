#!/usr/bin/env python3
"""Test dequant kernel performance."""

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.metal_dispatch import MetalKernelLibrary
from metal_marlin.quantization.trellis_codebook import TrellisCodebook
from metal_marlin.trellis.dispatch import dispatch_trellis_dequant_packed


def main():
    bits = 3
    K, N = 2048, 5632
    tiles_k = K // 16
    tiles_n = N // 16
    packed_bytes = {2: 64, 3: 96, 4: 128, 5: 160, 6: 192, 8: 256}[bits]

    # Get Metal library
    lib = MetalKernelLibrary.from_source_dir()

    packed = torch.randint(
        0, 256, (tiles_k, tiles_n, packed_bytes), dtype=torch.uint8, device="mps"
    )
    scales = torch.randn(K // 32, N, dtype=torch.float32, device="mps")
    codebook = TrellisCodebook(bits)
    grid = torch.from_numpy(codebook.get_grid()).to("mps")
    n_levels = codebook.get_n_levels()
    su = torch.sign(torch.randn(K, device="mps"))
    sv = torch.sign(torch.randn(N, device="mps"))

    # Warmup
    out = dispatch_trellis_dequant_packed(lib, packed, scales, grid, su, sv, K, N, bits)
    torch.mps.synchronize()

    # Benchmark
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = dispatch_trellis_dequant_packed(lib, packed, scales, grid, su, sv, K, N, bits)
    torch.mps.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / 10 * 1000
    gelem = K * N * 10 / (t1 - t0) / 1e9
    print(f"Dequant kernel [{K}x{N}] {bits}bit: {ms:.2f} ms")
    print(f"Element throughput: {gelem:.2f} GElem/s")


if __name__ == "__main__":
    main()
