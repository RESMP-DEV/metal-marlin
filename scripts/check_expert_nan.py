#!/usr/bin/env python3
"""Check if individual experts produce NaN."""

import torch
import torch.nn.functional as F

from metal_marlin.trellis.testing import create_mock_moe_mlp

# Check individual experts for NaN
torch.manual_seed(0)
moe = create_mock_moe_mlp(
    hidden_dim=2048,
    intermediate_dim=1536,
    num_experts=64,
    num_experts_per_tok=8,
    bits=3,
    device="mps",
)

torch.manual_seed(100)
x = torch.randn(1, 2048, dtype=torch.float16, device="mps")

# Test each expert individually (slow path)
nan_experts = []
for exp_id in range(64):
    expert = moe.experts[exp_id]
    with torch.inference_mode():
        gate = expert.gate_proj(x)
        up = expert.up_proj(x)
        swiglu = F.silu(gate) * up
        down = expert.down_proj(swiglu)

    if down.isnan().any().item():
        nan_experts.append(exp_id)
        gate_nan = gate.isnan().any().item()
        up_nan = up.isnan().any().item()
        swiglu_nan = swiglu.isnan().any().item()
        down_nan = down.isnan().any().item()
        print(
            f"Expert {exp_id}: gate={gate_nan}, up={up_nan}, swiglu={swiglu_nan}, down={down_nan}"
        )

if not nan_experts:
    print("No individual experts produce NaN!")
else:
    print(f"NaN experts: {nan_experts}")
