"""Compare MoE fast path buffers against slow path reference.

Also verifies sign buffer (su, sv) layout matches kernel expectations.
The Metal kernel uses flat indexing:
    su[expert_id * K_dim + global_k]
    sv[expert_id * N_dim + global_n]

If tensors are 2D [num_experts, dim], this works only if contiguous (row-major).
"""

import torch
import torch.nn.functional as F

from metal_marlin.trellis.testing import create_mock_moe_mlp


def compare_expert_forward(moe, x, expert_id):
    """Compare single expert computation."""
    expert = moe.experts[expert_id]

    # Slow path
    with torch.inference_mode():
        gate_slow = expert.gate_proj(x)
        up_slow = expert.up_proj(x)
        swiglu_slow = F.silu(gate_slow) * up_slow
        down_slow = expert.down_proj(swiglu_slow)

    print(f"Expert {expert_id} slow path:")
    print(f"  Gate: NaN={gate_slow.isnan().any()}, range=[{gate_slow.min():.4f}, {gate_slow.max():.4f}]")
    print(f"  Up: NaN={up_slow.isnan().any()}, range=[{up_slow.min():.4f}, {up_slow.max():.4f}]")
    print(f"  SwiGLU: NaN={swiglu_slow.isnan().any()}, range=[{swiglu_slow.min():.4f}, {swiglu_slow.max():.4f}]")
    print(f"  Down: NaN={down_slow.isnan().any()}, range=[{down_slow.min():.4f}, {down_slow.max():.4f}]")

    return {
        'gate': gate_slow, 'up': up_slow,
        'swiglu': swiglu_slow, 'down': down_slow
    }


def verify_sign_layout(moe) -> dict[str, bool]:
    """Verify su/sv buffer layout matches kernel expectations.

    The kernel uses flat indexing: su[expert_id * dim + offset]
    PyTorch 2D row-major tensor[e, d] -> offset e * D + d
    These match IFF tensor is contiguous.

    Args:
        moe: TrellisMoEMLP instance with stacked expert weights.

    Returns:
        Dictionary with verification results for each buffer.
    """
    results = {}

    gate_su = moe.gate_su_stacked
    gate_sv = moe.gate_sv_stacked
    up_su = moe.up_su_stacked
    up_sv = moe.up_sv_stacked
    down_su = moe.down_su_stacked
    down_sv = moe.down_sv_stacked

    num_experts = len(moe.experts)
    hidden_dim = moe.hidden_dim
    intermediate_dim = moe.intermediate_dim

    print("\n=== Sign Buffer Layout Analysis ===")
    print(f"num_experts: {num_experts}")
    print(f"hidden_dim (K for gate/up): {hidden_dim}")
    print(f"intermediate_dim (N for gate/up, K for down): {intermediate_dim}")

    buffers = {
        "gate_su": (gate_su, hidden_dim, "Gate row signs (K=hidden)"),
        "gate_sv": (gate_sv, intermediate_dim, "Gate col signs (N=intermediate)"),
        "up_su": (up_su, hidden_dim, "Up row signs (K=hidden)"),
        "up_sv": (up_sv, intermediate_dim, "Up col signs (N=intermediate)"),
        "down_su": (down_su, intermediate_dim, "Down row signs (K=intermediate)"),
        "down_sv": (down_sv, hidden_dim, "Down col signs (N=hidden)"),
    }

    print("\n--- Buffer Shapes ---")
    for name, (buf, expected_dim, desc) in buffers.items():
        print(f"{name}: shape={tuple(buf.shape)}, dtype={buf.dtype}, "
              f"contiguous={buf.is_contiguous()}")

        # Expected shapes
        expected_shape = (num_experts, expected_dim)
        shape_ok = tuple(buf.shape) == expected_shape

        if not shape_ok:
            print(f"  WARNING: Expected {expected_shape}, got {tuple(buf.shape)}")

        # Verify contiguity for correct flat indexing
        if not buf.is_contiguous():
            print(f"  ERROR: {name} is NOT contiguous! Kernel indexing will be wrong!")
            results[name] = False
        else:
            results[name] = True

    print("\n--- Metal Kernel Indexing Verification ---")
    print("Kernel expects flat indexing: su[expert_id * dim + offset]")
    print("PyTorch row-major 2D tensor [E, D]: tensor[e, d] -> offset e * D + d")
    print("These are EQUIVALENT if tensor is contiguous (row-major).")

    # Numerical verification: check that flat indexing matches 2D indexing
    print("\n--- Numerical Spot Check ---")

    # Pick a random expert and position to verify
    test_expert = min(3, num_experts - 1)

    for name, (buf, dim, desc) in buffers.items():
        test_idx = min(42, dim - 1)  # Pick an index to test

        # 2D indexing (what we have)
        val_2d = buf[test_expert, test_idx].item()

        # Flat indexing (what kernel uses)
        flat_buf = buf.flatten()
        flat_idx = test_expert * dim + test_idx
        val_flat = flat_buf[flat_idx].item()

        match = abs(val_2d - val_flat) < 1e-6
        status = "OK" if match else "MISMATCH"
        print(f"{name}[{test_expert}, {test_idx}] = {val_2d:.6f}, "
              f"flat[{flat_idx}] = {val_flat:.6f} -> {status}")

        if not match:
            results[name] = False

    return results


def check_weight_buffer_shapes(moe):
    """Verify stacked weight buffer shapes match kernel expectations."""
    print("\nWeight buffer shapes:")
    print(f"  gate_weights_stacked: {moe.gate_weights_stacked.shape}")
    print(f"  gate_scales_stacked: {moe.gate_scales_stacked.shape}")
    print(f"  gate_su_stacked: {moe.gate_su_stacked.shape}")
    print(f"  gate_sv_stacked: {moe.gate_sv_stacked.shape}")
    print(f"  down_weights_stacked: {moe.down_weights_stacked.shape}")
    print(f"  down_scales_stacked: {moe.down_scales_stacked.shape}")

    # Check for expected layout: [num_experts, ...]
    num_experts = len(moe.experts)
    assert moe.gate_weights_stacked.shape[0] == num_experts

    # Check hidden/intermediate dims
    hidden_dim = moe.hidden_dim
    intermediate_dim = moe.intermediate_dim
    print(f"\nDimensions: hidden={hidden_dim}, intermediate={intermediate_dim}")


def verify_weight_strides(moe) -> None:
    """Verify weight tensor strides are correct for Metal dispatch."""
    print("\n=== Weight Tensor Stride Analysis ===")

    weights = {
        "gate_weights": moe.gate_weights_stacked,
        "gate_scales": moe.gate_scales_stacked,
        "up_weights": moe.up_weights_stacked,
        "up_scales": moe.up_scales_stacked,
        "down_weights": moe.down_weights_stacked,
        "down_scales": moe.down_scales_stacked,
    }

    for name, tensor in weights.items():
        print(f"{name}:")
        print(f"  shape: {tuple(tensor.shape)}")
        print(f"  strides: {tensor.stride()}")
        print(f"  contiguous: {tensor.is_contiguous()}")
        print(f"  dtype: {tensor.dtype}")

        if not tensor.is_contiguous():
            print("  WARNING: Non-contiguous! This may cause incorrect Metal access.")


def verify_scale_layout(moe):
    """Verify scale buffer layout matches kernel expectations.

    The kernel expects scales indexed as (gemm_trellis_moe.metal:118):
        half scale = scales[expert_id * N_dim * n_groups + group_idx * N_dim + global_n];

    This means the expected layout is [num_experts, n_groups, N_dim] in row-major order.

    CRITICAL BUG CHECK:
    The kernel computes n_groups = (K_dim + group_size - 1) / group_size
    But moe_dispatch.py passes group_size=32 while scales are stored with group_size=128!
    """
    print("\n" + "=" * 70)
    print("SCALE BUFFER LAYOUT VERIFICATION")
    print("=" * 70)

    gate_scales = moe.gate_scales_stacked
    print(f"\nGate scales shape: {gate_scales.shape}")

    # For gate: K=hidden_dim, N=intermediate_dim
    hidden_dim = moe.hidden_dim
    intermediate_dim = moe.intermediate_dim
    num_experts = len(moe.experts)

    # Compute expected n_groups with different group_sizes
    group_size_128 = 128  # What scales are stored with
    group_size_32 = 32    # What moe_dispatch.py currently passes!
    n_groups_128 = (hidden_dim + group_size_128 - 1) // group_size_128
    n_groups_32 = (hidden_dim + group_size_32 - 1) // group_size_32

    print("\nDimension analysis:")
    print(f"  hidden_dim (K):      {hidden_dim}")
    print(f"  intermediate_dim (N): {intermediate_dim}")
    print(f"  num_experts:         {num_experts}")

    print("\nGroup size analysis:")
    print(f"  n_groups with group_size=128: {n_groups_128}")
    print(f"  n_groups with group_size=32:  {n_groups_32}")

    print("\nExpected shape [num_experts, n_groups, N]:")
    print(f"  With group_size=128: [{num_experts}, {n_groups_128}, {intermediate_dim}]")
    print(f"  With group_size=32:  [{num_experts}, {n_groups_32}, {intermediate_dim}]")

    # Check actual shape
    actual_shape = tuple(gate_scales.shape)
    expected_128 = (num_experts, n_groups_128, intermediate_dim)
    expected_32 = (num_experts, n_groups_32, intermediate_dim)

    print(f"\nActual shape: {actual_shape}")

    if actual_shape == expected_128:
        print("  ✓ Matches group_size=128 layout (CORRECT for scale storage)")
    elif actual_shape == expected_32:
        print("  ✗ Matches group_size=32 layout (UNEXPECTED)")
    else:
        print("  ✗ Shape mismatch!")

    # Check single expert scales vs stacked
    expert_0_scales = moe.experts[0].gate_proj.scales
    print(f"\nSingle expert gate scales shape: {expert_0_scales.shape}")

    # Compare first expert from stacked vs individual
    stacked_exp0 = gate_scales[0]
    print(f"Stacked expert 0 shape: {stacked_exp0.shape}")

    if torch.allclose(stacked_exp0.float(), expert_0_scales.float()):
        print("  ✓ Stacked scales match individual expert exactly")
    else:
        max_diff = (stacked_exp0.float() - expert_0_scales.float()).abs().max()
        print("  ✗ WARNING: Stacked scales don't match individual expert!")
        print(f"    Max diff: {max_diff}")

    # Critical bug detection
    print("\n" + "=" * 70)
    print("CRITICAL BUG CHECK")
    print("=" * 70)
    print("\nKernel scale indexing (gemm_trellis_moe.metal:116-118):")
    print("  uint n_groups = (K_dim + group_size - 1) / group_size;")
    print("  uint group_idx = global_k / group_size;")
    print("  half scale = scales[expert_id * N_dim * n_groups + group_idx * N_dim + global_n];")

    print("\nmoe_dispatch.py line 305 passes: group_size=32")
    print("But scales are stored with: group_size=128")

    if n_groups_32 != n_groups_128:
        print("\n❌ CRITICAL BUG DETECTED!")
        print(f"   Kernel will compute n_groups={n_groups_32} (from group_size=32)")
        print(f"   But scale buffer has n_groups={n_groups_128} (from group_size=128)")
        print("   This causes INCORRECT scale indexing!")
        print("\n   FIX: Change moe_dispatch.py line 305 from:")
        print("        32,  # group_size (fixed for Trellis)")
        print("   To:")
        print("        128,  # group_size (128 for Trellis scales)")
    else:
        print("\n✓ n_groups match (coincidentally) - but group_size should still be corrected")

    # Buffer contiguity check
    print("\n" + "-" * 40)
    print("Buffer contiguity:")
    print(f"  gate_scales.is_contiguous(): {gate_scales.is_contiguous()}")
    print(f"  up_scales.is_contiguous():   {moe.up_scales_stacked.is_contiguous()}")
    print(f"  down_scales.is_contiguous(): {moe.down_scales_stacked.is_contiguous()}")


def check_nan_sources(moe, x: torch.Tensor) -> None:
    """Trace potential NaN sources through MoE forward pass.

    Args:
        moe: TrellisMoEMLP instance.
        x: Sample input tensor [batch, hidden_dim].
    """
    print("\n=== NaN Source Tracing ===")

    # Check input
    print(f"Input: shape={tuple(x.shape)}, has_nan={x.isnan().any().item()}, "
          f"has_inf={x.isinf().any().item()}")

    # Check router
    router_out = moe.router(x.to(moe.router.weight.dtype))
    print(f"Router output: has_nan={router_out.isnan().any().item()}, "
          f"has_inf={router_out.isinf().any().item()}")

    # Check grid codebook values
    grid = moe.experts[0].gate_proj.grid
    print(f"Codebook grid: min={grid.min().item():.4f}, max={grid.max().item():.4f}, "
          f"has_nan={grid.isnan().any().item()}")

    # Check sign flip buffers for NaN/extreme values
    for name in ["gate_su", "gate_sv", "up_su", "up_sv", "down_su", "down_sv"]:
        buf = getattr(moe, f"{name}_stacked")
        has_nan = buf.isnan().any().item()
        has_inf = buf.isinf().any().item()
        nonzero = buf.ne(0).float().mean().item()
        print(f"{name}: min={buf.min().item():.4f}, max={buf.max().item():.4f}, "
              f"nan={has_nan}, inf={has_inf}, nonzero_ratio={nonzero:.4f}")

    # Check scales for extreme values
    for name in ["gate_scales", "up_scales", "down_scales"]:
        buf = getattr(moe, f"{name}_stacked")
        has_nan = buf.isnan().any().item()
        has_inf = buf.isinf().any().item()
        print(f"{name}: min={buf.min().item():.6f}, max={buf.max().item():.6f}, "
              f"nan={has_nan}, inf={has_inf}")


if __name__ == '__main__':
    torch.manual_seed(42)
    moe = create_mock_moe_mlp(device='mps')
    x = torch.randn(1, moe.hidden_dim, dtype=torch.float16, device='mps')

    check_weight_buffer_shapes(moe)
    verify_scale_layout(moe)  # NEW: Check scale buffer layout and group_size bug
    verify_sign_layout(moe)
    verify_weight_strides(moe)
    check_nan_sources(moe, x)
    compare_expert_forward(moe, x, expert_id=0)
