"""Test GPTQ dequantization kernel."""

import numpy as np
import pytest
import torch

from metal_marlin._compat import HAS_MPS


@pytest.mark.skipif(not HAS_MPS, reason="MPS required")
def test_gptq_dequant_kernel():
    """Verify GPTQ dequantization kernel matches CPU reference."""
    from metal_marlin.metal_dispatch import (
        MetalKernelLibrary,
        _copy_buffer_to_tensor,
        _CopyBackBuffer,
        mps_tensor_to_metal_buffer,
    )

    # Dimensions
    K, N = 256, 256
    group_size = 128
    n_groups = K // group_size

    # 1. Prepare Inputs
    # Scales: [n_groups, N] half
    scales = torch.randn(n_groups, N, dtype=torch.float16, device="mps")

    # Zeros: [n_groups, N] half (unpacked)
    # Use values in [0, 15] range to simulate 4-bit zeros
    zeros = torch.randint(0, 15, (n_groups, N), dtype=torch.float32).to(torch.float16).to("mps")

    # Weights: Packed [K/8, N] int32
    # Create random 4-bit weights first (0-15)
    weights_u4 = torch.randint(0, 15, (K, N), dtype=torch.int32, device="cpu")

    # Pack weights: 8 weights per int32
    # Layout: each int32 at (k_block, n) holds weights for k_block*8 ... k_block*8+7
    # Packing: weights[0] | (weights[1] << 4) ...
    weights_packed = torch.zeros((K // 8, N), dtype=torch.int32)
    for i in range(8):
        # Extract rows k_block*8 + i
        rows = weights_u4[i::8, :]
        weights_packed |= (rows << (4 * i))

    weights_packed = weights_packed.to("mps")

    # 2. CPU Reference
    # w = (q - z) * s
    # Expand scales and zeros to [K, N]
    scales_expanded = scales.repeat_interleave(group_size, dim=0).cpu()
    zeros_expanded = zeros.repeat_interleave(group_size, dim=0).cpu()
    weights_cpu = weights_u4.float()

    # Dequant
    w_ref = (weights_cpu - zeros_expanded.float()) * scales_expanded.float()

    # 3. Run Metal Kernel
    lib = MetalKernelLibrary.from_source_dir()
    # The library name comes from the file stem: dequant_gptq
    kernel = lib.get_kernel("dequant_gptq", "dequant_gptq_kernel")

    output = torch.empty((K, N), dtype=torch.float16, device="mps")

    # Get buffers
    weights_buf = mps_tensor_to_metal_buffer(weights_packed, lib.device)
    scales_buf = mps_tensor_to_metal_buffer(scales, lib.device)
    zeros_buf = mps_tensor_to_metal_buffer(zeros, lib.device)

    # Output buffer with copy back support
    output_wrapper = mps_tensor_to_metal_buffer(output, lib.device, copy_back=True)
    output_buf = output_wrapper.buffer if isinstance(output_wrapper, _CopyBackBuffer) else output_wrapper

    # Dispatch
    # Grid size in threadgroups
    # Each thread handles 1 column (n_idx) and 1 K-block (k_block)
    # Total work items: (N, K//8)
    # Threadgroup size: (32, 8, 1) -> 256 threads
    # Grid dimensions: (ceil(N/32), ceil((K/8)/8), 1)

    threads_per_tg = (32, 8, 1)
    grid_dims = ((N + 31) // 32, (K // 8 + 7) // 8, 1)

    # Use internal _dispatch
    lib._dispatch(
        kernel,
        grid_dims,
        threads_per_tg,
        weights_buf,
        scales_buf,
        zeros_buf,
        output_buf,
        K, N, group_size
    )

    # Copy back result
    if isinstance(output_wrapper, _CopyBackBuffer):
        _copy_buffer_to_tensor(output_wrapper.buffer, output)

    # 4. Compare
    # Allow some tolerance for fp16 precision
    torch.testing.assert_close(output.cpu(), w_ref.to(torch.float16), rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_gptq_dequant_kernel()
