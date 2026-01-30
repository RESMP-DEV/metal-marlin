#!/usr/bin/env python3
"""Debug FP4 GEMM copy-back mechanism."""

import numpy as np
import torch

from metal_marlin.metal_dispatch import (
    TILE_M,
    TILE_N,
    _CopyBackBuffer,  # type: ignore[attr-defined]
    _private_buffer_from_bytes,  # type: ignore[attr-defined]
    _private_buffer_from_tensor,  # type: ignore[attr-defined]
    dispatch_kernel,
    get_default_library,
    get_gpu_family,
    mps_tensor_to_metal_buffer,
)


def main() -> None:
    lib = get_default_library()
    device = lib.device
    family = get_gpu_family(device)

    M, K, N = 8, 128, 128
    group_size = 32

    # Sync MPS before creating tensors
    torch.mps.synchronize()

    A = torch.randn(M, K, dtype=torch.float16, device="mps").contiguous()
    packed = torch.randint(0, 2**32 - 1, (K // 8, N), dtype=torch.uint32, device="mps").contiguous()
    scales = torch.randn(K // group_size, N, dtype=torch.float16, device="mps").contiguous()
    C = torch.empty((M, N), dtype=torch.float16, device="mps")

    # Sync again after tensor creation
    torch.mps.synchronize()

    print(f"C before: {C[0, :4].tolist()}")

    # Convert to buffers
    A_buf = _private_buffer_from_tensor(A, lib, device, cache=False)
    B_buf = _private_buffer_from_tensor(packed, lib, device, cache=True)
    S_buf = _private_buffer_from_tensor(scales, lib, device, cache=True)
    C_buf = mps_tensor_to_metal_buffer(C, device, copy_back=True)

    print(f"C_buf type: {type(C_buf).__name__}")
    print(f"Is CopyBackBuffer: {isinstance(C_buf, _CopyBackBuffer)}")

    # Check A buffer has valid data
    a_contents = A_buf.contents()
    a_len = A_buf.length()
    a_data = np.frombuffer(a_contents.as_buffer(a_len), dtype=np.float16)
    print(f"A_buf some values: {a_data[:4].tolist()}")
    print(f"A_buf non-zeros: {(a_data != 0).sum()} / {len(a_data)}")

    if isinstance(C_buf, _CopyBackBuffer):
        print(f"  buffer length: {C_buf.buffer.length()}")
        print(f"  tensor shape: {C_buf.tensor.shape}")

    # Create separate param buffers (kernel expects buffers at indices 4, 5, 6, 7)
    M_buf = _private_buffer_from_bytes(lib, device, np.array([M], dtype=np.uint32).tobytes())
    N_buf = _private_buffer_from_bytes(lib, device, np.array([N], dtype=np.uint32).tobytes())
    K_buf = _private_buffer_from_bytes(lib, device, np.array([K], dtype=np.uint32).tobytes())
    gs_buf = _private_buffer_from_bytes(
        lib, device, np.array([group_size], dtype=np.uint32).tobytes()
    )

    kernel_name = "marlin_gemm_fused_fp4" if family >= 9 else "marlin_gemm_fp4"
    print(f"Kernel: {kernel_name}")

    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    print(f"Grid: ({grid_n}, {grid_m}, 1) - {grid_n * grid_m} threadgroups")
    print(f"TILE_M={TILE_M}, TILE_N={TILE_N}")

    dispatch_kernel(
        lib,
        function_name=kernel_name,
        grid=(grid_n, grid_m, 1),
        threadgroup=(128, 1, 1),
        buffers=[A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf],
        wait=True,
    )

    print(f"C after dispatch: {C[0, :4].tolist()}")
    print(f"C non-zeros: {(C != 0).sum().item()} / {C.numel()}")

    # Manually read buffer contents
    if isinstance(C_buf, _CopyBackBuffer):
        buf = C_buf.buffer
    else:
        buf = C_buf

    contents = buf.contents()
    length = buf.length()
    raw = np.frombuffer(contents.as_buffer(length), dtype=np.float16)
    print(f"Raw buffer first 4: {raw[:4].tolist()}")
    print(f"Raw buffer non-zeros: {(raw != 0).sum()} / {len(raw)}")


if __name__ == "__main__":
    main()
