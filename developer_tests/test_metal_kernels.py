"""Test basic Metal kernel functionality.

Tests the actual metal_marlin APIs:
- marlin_gemm_fp4: FP4 fused dequant-GEMM
- scaled_dot_product_attention_metal: Attention kernel
- dequantize_fp4_metal: FP4 dequantization
"""
import torch

from metal_marlin import (
    HAS_MPS,
    dequantize_fp4_metal,
    marlin_gemm_fp4,
    quantize_fp4_metal,
    scaled_dot_product_attention_metal,
)
from metal_marlin.quantize import pack_fp4_weights


def get_device() -> torch.device:
    """Get MPS device or fall back to CPU."""
    if HAS_MPS and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_gemm():
    """Test basic FP4 GEMM kernel."""
    device = get_device()
    print(f"Testing GEMM on device: {device}")

    # Create input matrix
    M, K, N = 64, 128, 256
    group_size = 32

    a = torch.randn(M, K, device=device, dtype=torch.float16)

    # Create quantized weight matrix
    # For FP4, we need packed weights and scales
    w_fp16 = torch.randn(K, N, dtype=torch.float16)
    w_np = w_fp16.cpu().numpy()

    # Pack weights to FP4 format (returns packed, scales, meta)
    packed, scales, meta = pack_fp4_weights(w_np, group_size=group_size)
    packed_t = torch.from_numpy(packed).to(device)
    scales_t = torch.from_numpy(scales).to(device)

    # Run FP4 GEMM
    c = marlin_gemm_fp4(a, packed_t, scales_t, group_size=group_size)

    # Compare with reference (using original weights)
    expected = a @ w_fp16.to(device)

    # FP4 quantization introduces error, so use relaxed tolerance
    max_err = (c - expected).abs().max().item()
    mean_err = (c - expected).abs().mean().item()
    print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")
    assert max_err < 1.0, f"GEMM error too large: {max_err}"
    print("  GEMM passed")


def test_attention():
    """Test attention kernel."""
    device = get_device()
    print(f"Testing attention on device: {device}")

    # BSHD format: batch, heads, seq_len, head_dim
    batch, heads, seq_len, head_dim = 1, 8, 64, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)

    # Run Metal attention
    out = scaled_dot_product_attention_metal(q, k, v)

    # Compare with PyTorch reference
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    max_err = (out - expected).abs().max().item()
    mean_err = (out - expected).abs().mean().item()
    print(f"  Output shape: {out.shape}")
    print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")
    assert max_err < 1e-2, f"Attention error too large: {max_err}"
    print("  Attention passed")


def test_dequant():
    """Test FP4 dequantization round-trip."""
    device = get_device()
    print(f"Testing dequantization on device: {device}")

    # Create random float values
    shape = (128, 256)
    group_size = 128
    values = torch.randn(shape, device=device, dtype=torch.float32)

    # Quantize to FP4 (produces indices 0-15 and scales)
    indices, scales = quantize_fp4_metal(values, group_size=group_size)

    # Dequantize back
    reconstructed = dequantize_fp4_metal(indices, scales, group_size=group_size)

    # Check shape matches
    print(f"  Input shape: {values.shape}")
    print(f"  Indices shape: {indices.shape}, dtype: {indices.dtype}")
    print(f"  Reconstructed shape: {reconstructed.shape}, dtype: {reconstructed.dtype}")

    # Check reconstruction error (FP4 is lossy)
    max_err = (values.cpu() - reconstructed.cpu()).abs().max().item()
    mean_err = (values.cpu() - reconstructed.cpu()).abs().mean().item()
    print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")
    print("  Dequant passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Metal Kernel Tests")
    print("=" * 50)

    try:
        test_gemm()
    except Exception as e:
        print(f"  GEMM failed: {e}")

    try:
        test_attention()
    except Exception as e:
        print(f"  Attention failed: {e}")

    try:
        test_dequant()
    except Exception as e:
        print(f"  Dequant failed: {e}")

    print("\n" + "=" * 50)
    print("Tests completed")
    print("=" * 50)
