#!/usr/bin/env python3
"""Simple direct test of MoE kernels.

Tests:
1. Router dispatch - should return non-zero expert IDs and probabilities
2. Check if kernel functions are actually compiled
"""

import sys
from pathlib import Path

# Add contrib/metal_marlin to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch

print("=== Metal MoE Kernel Direct Test ===\n")

# Check Metal availability
try:
    from metal_marlin.metal_dispatch import HAS_METAL, get_default_library
    print(f"Metal available: {HAS_METAL}")
    if HAS_METAL:
        lib = get_default_library()
        print(f"Library loaded: {lib}")
        print(f"Device: {lib.device}")

        # Check what kernels are available
        print("\nAvailable kernels:")
        for name in sorted(dir(lib)):
            if not name.startswith('_') and callable(getattr(lib, name)):
                if 'moe' in name.lower() or 'router' in name.lower():
                    print(f"  - {name}")
    else:
        print("ERROR: Metal not available - cannot test kernels")
        sys.exit(1)
except Exception as e:
    print(f"ERROR importing metal_marlin: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 1: Router dispatch
print("\n" + "="*60)
print("TEST 1: Router Dispatch (matmul + softmax + top-k)")
print("="*60)

try:
    from metal_marlin.trellis.moe_dispatch import dispatch_moe_router_fused

    # Create minimal test inputs
    batch_size = 4
    hidden_dim = 64
    num_experts = 16
    top_k = 4

    hidden = torch.randn(batch_size, hidden_dim, device='mps', dtype=torch.float16)
    router_weights = torch.randn(num_experts, hidden_dim, device='mps', dtype=torch.float16)

    print(f"Input shape: {hidden.shape}")
    print(f"Router weights shape: {router_weights.shape}")
    print(f"Config: {num_experts} experts, top_k={top_k}")

    print(f"Hidden sample: {hidden[0, :5].cpu()}")
    print(f"Router weights sample: {router_weights[0, :5].cpu()}")

    # Dispatch router
    print("\nCalling dispatch_moe_router_fused...")
    expert_ids, expert_probs = dispatch_moe_router_fused(
        lib,
        hidden=hidden,
        router_weights=router_weights,
        num_experts=num_experts,
        top_k=top_k,
    )

    print(f"Output expert_ids shape: {expert_ids.shape}, dtype: {expert_ids.dtype}")
    print(f"Output expert_probs shape: {expert_probs.shape}, dtype: {expert_probs.dtype}")

    print(f"\nExpert IDs:\n{expert_ids.cpu()}")
    print(f"\nExpert probs:\n{expert_probs.cpu()}")

    # Sanity checks
    assert expert_ids.shape == (batch_size, top_k), f"Wrong expert_ids shape: {expert_ids.shape}"
    assert expert_probs.shape == (batch_size, top_k), f"Wrong expert_probs shape: {expert_probs.shape}"

    # Check expert IDs are in valid range
    ids_cpu = expert_ids.cpu().long()  # Convert to long to avoid promotion error
    assert ids_cpu.min() >= 0, f"Negative expert ID: {ids_cpu.min()}"
    assert ids_cpu.max() < num_experts, f"Expert ID out of range: {ids_cpu.max()}"

    # Check probabilities sum to ~1 (within tolerance due to fp16)
    prob_sums = expert_probs.sum(dim=-1).cpu()
    print(f"\nProbability sums (should be ~1.0): {prob_sums}")

    # Check if router actually worked (not all zeros)
    if expert_probs.abs().sum() == 0:
        print("\n❌ Router kernel FAILED - returned all zeros")
        print("This indicates a Metal kernel issue.")
    else:
        print("\n✓ Router dispatch test PASSED")
        if torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2):
            print("  - Probabilities sum to ~1")
        else:
            print(f"  - Warning: Probability sums deviate from 1.0: {prob_sums}")

except Exception as e:
    print(f"\n✗ Router dispatch test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Compare with PyTorch routing
print("\n" + "="*60)
print("TEST 2: Compare Router Output with PyTorch")
print("="*60)

try:
    import torch.nn.functional as F

    # Run same routing on CPU with PyTorch
    hidden_cpu = hidden.float().cpu()
    router_weights_cpu = router_weights.float().cpu()

    # PyTorch routing
    router_logits = hidden_cpu @ router_weights_cpu.T
    top_logits, top_ids = torch.topk(router_logits, k=top_k, dim=-1)
    pytorch_probs = F.softmax(top_logits, dim=-1)

    print(f"PyTorch expert IDs:\n{top_ids}")
    print(f"PyTorch expert probs:\n{pytorch_probs}")

    # Convert Metal output to same types
    metal_ids = expert_ids.cpu().long()
    metal_probs = expert_probs.cpu().float()

    print(f"\nMetal expert IDs:\n{metal_ids}")
    print(f"Metal expert probs:\n{metal_probs}")

    # Check if they're reasonably different (router should be random-ish)
    print(f"\nMetal vs PyTorch IDs equal: {(metal_ids == top_ids).all()}")
    print(f"Metal max prob: {metal_probs.max():.4f}, PyTorch max prob: {pytorch_probs.max():.4f}")

    if metal_probs.abs().sum() > 0:
        print("\n✓ Router produces non-trivial output")
    else:
        print("\n❌ Router still returns zeros - kernel issue")

except Exception as e:
    print(f"\n✗ Comparison test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check if softmax_topk kernels are available
print("\n" + "="*60)
print("TEST 3: Check softmax_topk kernels")
print("="*60)

try:
    from metal_marlin.trellis.softmax_topk import SoftmaxTopKDispatcher

    # Create dispatcher
    dispatcher = SoftmaxTopKDispatcher(lib, num_experts=16, top_k=4)

    # Create router logits directly
    router_logits = torch.randn(batch_size, num_experts, device='mps', dtype=torch.float16)

    print(f"Router logits shape: {router_logits.shape}")

    # Dispatch
    print("\nCalling softmax_topk dispatcher...")
    selected_experts, routing_weights = dispatcher.dispatch(router_logits)

    print(f"Selected experts shape: {selected_experts.shape}, dtype: {selected_experts.dtype}")
    print(f"Routing weights shape: {routing_weights.shape}, dtype: {routing_weights.dtype}")

    print(f"\nSelected experts:\n{selected_experts.cpu()}")
    print(f"\nRouting weights:\n{routing_weights.cpu()}")

    # Check output
    if routing_weights.abs().sum() > 0:
        print("\n✓ SoftmaxTopK dispatcher test PASSED")
    else:
        print("\n❌ SoftmaxTopK dispatcher FAILED - returned all zeros")

except Exception as e:
    print(f"\n✗ SoftmaxTopK test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("Tests completed. Review above for any FAILED results.")
