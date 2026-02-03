#!/usr/bin/env python3
"""Simple test to debug the router kernel returning all zeros."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

print("=== Router Kernel Debug Test ===\n")

# Check Metal
try:
    from metal_marlin.metal_dispatch import HAS_METAL, get_default_library
    print(f"Metal available: {HAS_METAL}")
    lib = get_default_library()
    print(f"Device: {lib.device}\n")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Test router
from metal_marlin.trellis.moe_dispatch import dispatch_moe_router_fused

# Create test inputs
batch_size = 1
hidden_dim = 64
num_experts = 4
top_k = 2

# Use non-zero input
hidden = torch.ones(batch_size, hidden_dim, device='mps', dtype=torch.float16)

# Use non-zero router weights - small positive values to ensure routing is deterministic
router_weights = torch.ones(num_experts, hidden_dim, device='mps', dtype=torch.float16) * 0.1

# Add variation so different experts have different weights
for i in range(num_experts):
    router_weights[i, i % hidden_dim] = 0.5

print(f"Input hidden[0][:10]: {hidden[0][:10].cpu()}")
print(f"Router weights[0][:10]: {router_weights[0][:10].cpu()}")
print(f"Router weights[1][:10]: {router_weights[1][:10].cpu()}")
print(f"Router weights[2][:10]: {router_weights[2][:10].cpu()}")
print(f"Router weights[3][:10]: {router_weights[3][:10].cpu()}")

print("\nCalling dispatch_moe_router_fused with:")
print(f"  batch_size={batch_size}, hidden_dim={hidden_dim}")
print(f"  num_experts={num_experts}, top_k={top_k}")

try:
    expert_ids, expert_probs = dispatch_moe_router_fused(
        lib,
        hidden=hidden,
        router_weights=router_weights,
        num_experts=num_experts,
        top_k=top_k,
    )

    print("\nResults:")
    print(f"  expert_ids shape: {expert_ids.shape}, dtype: {expert_ids.dtype}")
    print(f"  expert_probs shape: {expert_probs.shape}, dtype: {expert_probs.dtype}")
    print(f"\n  expert_ids: {expert_ids.cpu().flatten().tolist()}")
    print(f"  expert_probs: {expert_probs.cpu().flatten().tolist()}")

    # Check if all zeros
    if expert_ids.sum() == 0:
        print("\n✗ ERROR: Router kernel returned all zeros!")
        print("This indicates a kernel issue.")
    else:
        print("\n✓ Router kernel returned non-zero expert_ids")
        print(f"  sum(expert_ids) = {expert_ids.sum().item()}")

    # Check if probs are zero
    if expert_probs.abs().sum() == 0:
        print("✗ ERROR: Router kernel returned zero probabilities!")
    else:
        print("✓ Router kernel returned non-zero probabilities")
        print(f"  sum(expert_probs) = {expert_probs.abs().sum().item()}")

except Exception as e:
    print(f"\n✗ ERROR in dispatch: {e}")
    import traceback
    traceback.print_exc()

# Compare with PyTorch
print("\n" + "="*60)
print("PyTorch reference implementation:")
print("="*60)

# Router: compute logits = hidden @ W_router.T
# hidden: [batch, hidden] = [1, 64]
# router_weights: [num_experts, hidden] = [4, 64]
# We want logits: [batch, num_experts] = [1, 4]
# Standard matmul: [1, 64] @ [64, 4] = [1, 4]
# So we need router_weights.T

logits = hidden @ router_weights.T
print(f"Logits: {logits.cpu().flatten().tolist()}")

# Softmax
probs_pt = torch.softmax(logits.float(), dim=-1)
print(f"PyTorch probs: {probs_pt.cpu().flatten().tolist()}")

# Top-k
topk_probs_pt, topk_ids_pt = torch.topk(probs_pt, k=top_k, dim=-1)
print(f"\nPyTorch top-k IDs: {topk_ids_pt.cpu().flatten().tolist()}")
print(f"PyTorch top-k probs: {topk_probs_pt.cpu().flatten().tolist()}")

print(f"\nExpected expert_ids: {topk_ids_pt.cpu().flatten().tolist()}")
print(f"Got expert_ids: {expert_ids.cpu().flatten().tolist()}")
