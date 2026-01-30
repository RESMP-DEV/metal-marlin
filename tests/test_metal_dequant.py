"""Test Metal dequantization shaders produce correct output."""

import numpy as np
import pytest
import torch


@pytest.fixture
def small_indices_weight():
    """Create small indices weight for testing."""
    K, N = 32, 32  # 2x2 tiles
    bits = 3
    tiles_k, tiles_n = 2, 2

    # Random indices (int16)
    n_levels = 2**bits
    indices = torch.randint(0, n_levels, (tiles_k, tiles_n, 256), dtype=torch.int16)
    scales = torch.randn(1, N, dtype=torch.float32)
    su = torch.randn(K, dtype=torch.float32)
    sv = torch.randn(N, dtype=torch.float32)

    return indices, scales, su, sv, K, N, bits


def cpu_dequant_reference(indices, scales, grid, su, sv, K, N):
    """CPU reference implementation for verification."""
    tiles_k = (K + 15) // 16
    tiles_n = (N + 15) // 16
    n_levels = grid.shape[0]

    # Dequantize using grid
    output = np.zeros((K, N), dtype=np.float32)
    for row in range(K):
        for col in range(N):
            tk, tn = row // 16, col // 16
            local_idx = (row % 16) * 16 + (col % 16)
            cb_idx = indices[tk, tn, local_idx].item()
            base = grid[cb_idx].item()
            group_idx = row // 128
            scale = scales[group_idx, col].item()
            output[row, col] = base * scale * su[row].item() * sv[col].item()

    return torch.from_numpy(output).half()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_metal_matches_cpu(small_indices_weight):
    """Verify Metal output matches CPU reference."""
    from metal_marlin.metal_dispatch import MetalKernelLibrary
    from metal_marlin.quantization.trellis_codebook import TrellisCodebook
    from metal_marlin.trellis.dispatch import dispatch_trellis_dequant_fused

    indices, scales, su, sv, K, N, bits = small_indices_weight
    codebook = TrellisCodebook(bits=bits)
    grid = torch.from_numpy(codebook.get_grid()).float()

    # CPU reference
    cpu_output = cpu_dequant_reference(indices, scales, grid, su, sv, K, N)

    # Metal output
    lib = MetalKernelLibrary.from_source_dir()
    metal_output = dispatch_trellis_dequant_fused(
        lib,
        indices.to("mps"),
        scales.to("mps"),
        grid.to("mps"),
        su.to("mps"),
        sv.to("mps"),
        K,
        N,
        group_size=128,
    )

    # Compare
    torch.testing.assert_close(metal_output.cpu(), cpu_output, rtol=1e-2, atol=1e-2)
