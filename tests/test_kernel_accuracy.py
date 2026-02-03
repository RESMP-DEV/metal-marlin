import numpy as np
import pytest
import torch

try:
    from metal_marlin.kernels import (
        dequant_fp4,
        marlin_gemm_fp4,
        marlin_gemm_int4,
        pack_fp4_weights,
        pack_int2_weights,
        pack_int3_weights,
    )
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

# Skip all tests if MPS is not available
if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

@pytest.fixture
def device():
    return torch.device("mps")

def get_fp4_e2m1_table(device):
    """Returns the E2M1 lookup table for FP4 dequantization."""
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device, dtype=torch.float16
    )

def dequant_fp4_torch(packed_weights, scales, group_size, K, N):
    """
    Reference PyTorch implementation of FP4 dequantization.
    packed_weights: [K/8, N] uint32
    scales: [K/group_size, N] float16
    """
    device = packed_weights.device
    fp4_table = get_fp4_e2m1_table(device)

    # Expand scales
    # scales: [K/group_size, N] -> [K, N]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)
    if scales_expanded.shape[0] != K:
         scales_expanded = scales_expanded[:K, :]

    # Output buffer
    out = torch.zeros((K, N), dtype=torch.float16, device=device)

    for i in range(8):
        # Extract nibbles (4 bits)
        # packed: [K/8, N]
        nibbles = ((packed_weights >> (i * 4)) & 0xF).long()

        # Lookup values
        vals = fp4_table[nibbles] # [K/8, N]

        # Place in output
        # Rows corresponding to this bit position
        rows = torch.arange(i, K, 8, device=device)

        # Adjust scales to match the subset of rows
        scale_subset = scales_expanded[rows, :]

        out[rows, :] = vals * scale_subset

    return out

def check_accuracy(output, reference, atol=1e-3, cos_sim_thr=0.9999):
    """Validates output against reference."""
    # Ensure both are float32 for comparison to avoid half precision artifacts in error calc
    out_f32 = output.float()
    ref_f32 = reference.float()

    diff = (out_f32 - ref_f32).abs()
    max_diff = diff.max().item()

    out_flat = out_f32.flatten()
    ref_flat = ref_f32.flatten()

    # Handle zero vectors
    if out_flat.norm() < 1e-6 and ref_flat.norm() < 1e-6:
        cos_sim = 1.0
    else:
        cos_sim = torch.nn.functional.cosine_similarity(out_flat, ref_flat, dim=0).item()

    print(f"Max Diff: {max_diff:.6f}, Cos Sim: {cos_sim:.6f}")

    assert max_diff < atol, f"Max absolute error {max_diff} exceeds {atol}"
    assert cos_sim > cos_sim_thr, f"Cosine similarity {cos_sim} < {cos_sim_thr}"

@pytest.mark.parametrize("K", [1024, 2048]) # Large K to trigger kernel
@pytest.mark.parametrize("N", [64, 256])
@pytest.mark.parametrize("group_size", [32, 64])
def test_dequant_fp4_accuracy(K, N, group_size, device):
    """Test standalone FP4 dequantization kernel accuracy."""
    torch.manual_seed(42)

    # Generate random FP16 weights to pack
    # Range -2.0 to 2.0
    weights_ref = torch.randn(K, N, device=device, dtype=torch.float16) * 2.0

    # Pack
    # pack_fp4_weights expects [N, K] (out_features, in_features)
    # So transpose weights_ref [K, N] -> [N, K]
    packed, scales = pack_fp4_weights(weights_ref.T, group_size=group_size)

    # 1. Reference Dequantization
    weights_dequant_ref = dequant_fp4_torch(packed, scales, group_size, K, N)

    # 2. Kernel Dequantization
    weights_dequant_kernel = dequant_fp4(packed, scales, K, N, group_size=group_size)

    # Compare
    # Tolerance is tight because it should be exact same math (lookup + scale)
    check_accuracy(weights_dequant_kernel, weights_dequant_ref, atol=1e-3, cos_sim_thr=0.99999)

@pytest.mark.parametrize("M", [1, 16, 33, 64])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("K", [512, 1024]) # 512 is fallback, 1024 is kernel
@pytest.mark.parametrize("group_size", [32, 64])
def test_marlin_gemm_fp4_accuracy(M, N, K, group_size, device):
    """Test fused FP4 GEMM kernel accuracy."""
    torch.manual_seed(42)

    # Inputs
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    weights_raw = torch.randn(K, N, device=device, dtype=torch.float16)

    # Pack weights
    # pack_fp4_weights expects [N, K], so transpose
    packed, scales = pack_fp4_weights(weights_raw.T, group_size=group_size)

    # Reference: Dequantize then Matmul
    # Note: marlin_gemm logic uses A @ B
    B_dequant = dequant_fp4_torch(packed, scales, group_size, K, N)
    C_ref = torch.matmul(A, B_dequant)

    # Kernel
    C_kernel = marlin_gemm_fp4(A, packed, scales, group_size=group_size)

    # Compare
    # GEMM accumulation error can be larger than elementwise dequant error
    # Especially for half precision accumulation
    check_accuracy(C_kernel, C_ref, atol=1e-3, cos_sim_thr=0.999)

@pytest.mark.parametrize("batch_dims", [(1,), (4,), (2, 8)])
def test_marlin_gemm_fp4_shapes(batch_dims, device):
    """Test FP4 GEMM with various batch dimensions."""
    M = int(np.prod(batch_dims))
    N, K = 64, 1024
    group_size = 32

    A = torch.randn(*batch_dims, K, device=device, dtype=torch.float16)
    weights = torch.randn(K, N, device=device, dtype=torch.float16)

    # Pack weights
    packed, scales = pack_fp4_weights(weights.T, group_size)

    # Reference
    B_dequant = dequant_fp4_torch(packed, scales, group_size, K, N)
    C_ref = torch.matmul(A, B_dequant)

    # Kernel
    C_kernel = marlin_gemm_fp4(A, packed, scales, group_size)

    assert C_kernel.shape == (*batch_dims, N)
    check_accuracy(C_kernel, C_ref, atol=1e-3, cos_sim_thr=0.999)
