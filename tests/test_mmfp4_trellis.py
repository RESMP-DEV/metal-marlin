
import torch
import pytest
import time
import os
from metal_marlin.layers.mmfp4_linear import MMFP4Linear

# Helper to run test
def run_test_mmfp4_trellis():
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping test")
        return

    device = torch.device("mps")
    
    # Dimensions
    M = 128
    K = 4096
    N = 4096
    group_size = 128
    
    print(f"Testing MMFP4 Trellis integration with M={M}, K={K}, N={N}")

    # Random packed weights [N, K//8]
    # Use int32 range then cast to uint32 to avoid overflow issues in random generation
    packed = torch.randint(-2**31, 2**31-1, (N, K//8), dtype=torch.int32, device=device).view(torch.uint32)
    scales = torch.randn((K // group_size, N), dtype=torch.float16, device=device)
    
    # Init layer
    layer = MMFP4Linear(
        packed_weights=packed,
        scales=scales,
        group_size=group_size,
        use_trellis_kernel=False
    ).to(device)
    
    x = torch.randn((M, K), dtype=torch.float16, device=device)
    
    # Reference (MMFP4 native or fallback)
    print("Running MMFP4 path...")
    for _ in range(5):
        out_ref = layer(x)
    torch.mps.synchronize()
    
    start = time.time()
    n_iters = 50
    for _ in range(n_iters):
        out_ref = layer(x)
    torch.mps.synchronize()
    time_ref = (time.time() - start) / n_iters * 1000
    print(f"MMFP4 time: {time_ref:.3f} ms")
    
    # Enable Trellis
    print("Running Trellis path...")
    layer.use_trellis_kernel = True
    
    # Warm up (and trigger layout conversion)
    # This might take longer due to layout conversion on first run
    start_warm = time.time()
    for _ in range(5):
        out_trellis = layer(x)
    torch.mps.synchronize()
    print(f"Trellis warmup (inc conversion): {(time.time() - start_warm)*1000:.3f} ms")
    
    start = time.time()
    for _ in range(n_iters):
        out_trellis = layer(x)
    torch.mps.synchronize()
    time_trellis = (time.time() - start) / n_iters * 1000
    print(f"Trellis time: {time_trellis:.3f} ms")
    print(f"Speedup: {time_ref / time_trellis:.2f}x")

    # Verify
    diff = (out_ref - out_trellis).abs().max().item()
    print(f"Max absolute diff: {diff}")
    
    # Check for NaN
    if torch.isnan(out_trellis).any():
        print("Error: Trellis output contains NaNs!")
        return

    # Tolerances
    # FP16 matrix mult can accumulate error. 
    # MMFP4 kernel vs Trellis kernel might have different accumulation order or rounding.
    atol = 1e-2
    rtol = 1e-2
    
    if diff > atol:
        # Check relative error
        rel_diff = (out_ref - out_trellis).abs() / (out_ref.abs() + 1e-6)
        max_rel = rel_diff.max().item()
        print(f"Max relative diff: {max_rel}")
        if max_rel > rtol:
            print("Mismatch!")
        else:
            print("Match (within relative tolerance)")
    else:
        print("Match (within absolute tolerance)")

if __name__ == "__main__":
    run_test_mmfp4_trellis()
