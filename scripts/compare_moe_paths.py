#!/usr/bin/env python3
"""Compare slow vs fast MoE path outputs."""

import torch

from metal_marlin.trellis.testing import create_mock_moe_mlp

# Test slow path to verify it's correct
moe = create_mock_moe_mlp(
    hidden_dim=2048,
    intermediate_dim=1536,
    num_experts=64,
    num_experts_per_tok=8,
    bits=3,
    device="mps",
)

torch.manual_seed(42)
x = torch.randn(4, 2048, dtype=torch.float16, device="mps")

# Test slow path directly
moe._use_fast_moe = False
with torch.inference_mode():
    slow_out = moe(x)

print(f"Slow path: NaN={slow_out.isnan().any().item()}")
print(f"Slow output range: [{slow_out.min().item():.4f}, {slow_out.max().item():.4f}]")

# Test fast path
moe._use_fast_moe = True
torch.manual_seed(42)  # Same seed for same routing
x = torch.randn(4, 2048, dtype=torch.float16, device="mps")

with torch.inference_mode():
    fast_out = moe(x)

print(f"Fast path: NaN={fast_out.isnan().any().item()}")
if not fast_out.isnan().any().item():
    print(f"Fast output range: [{fast_out.min().item():.4f}, {fast_out.max().item():.4f}]")

    # Compare outputs
    diff = (slow_out - fast_out).abs()
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
else:
    # Find which rows have NaN
    nan_rows = fast_out.isnan().any(dim=1)
    print(f"NaN rows: {nan_rows.nonzero().squeeze().tolist()}")
