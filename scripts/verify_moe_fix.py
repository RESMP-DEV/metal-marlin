#!/usr/bin/env python3
import torch
import torch.nn as nn

from metal_marlin.trellis.layer import TrellisDenseMLP
from metal_marlin.trellis.linear import TrellisLinear
from metal_marlin.trellis.model import TrellisMoEMLP

# Create minimal MoE
hidden_dim = 4096
intermediate_dim = 13696
num_experts = 64

# Mock experts
experts = []
for _ in range(num_experts):
    gate = TrellisLinear(hidden_dim, intermediate_dim, bits=3, bias=False)
    up = TrellisLinear(hidden_dim, intermediate_dim, bits=3, bias=False)
    down = TrellisLinear(intermediate_dim, hidden_dim, bits=3, bias=False)
    expert = TrellisDenseMLP(gate, up, down)
    experts.append(expert)

router = nn.Linear(hidden_dim, num_experts, bias=False)
shared_expert = TrellisDenseMLP(
    TrellisLinear(hidden_dim, intermediate_dim, bits=3, bias=False),
    TrellisLinear(hidden_dim, intermediate_dim, bits=3, bias=False),
    TrellisLinear(intermediate_dim, hidden_dim, bits=3, bias=False),
)

moe = TrellisMoEMLP(router, experts, shared_expert, num_experts_per_tok=8).to('mps').eval()

# Warm up
x = torch.randn(1, 1, hidden_dim, device='mps', dtype=torch.float16)
with torch.no_grad():
    for _ in range(5):
        _ = moe(x)

# Time 47 layers (simulate full model)
import time

torch.mps.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(47):
        x = moe(x)
torch.mps.synchronize()
elapsed_ms = (time.time() - t0) * 1000

ms_per_layer = elapsed_ms / 47
print(f"MoE latency: {ms_per_layer:.2f} ms per layer")
print(f"Total (47 layers): {elapsed_ms:.2f} ms")

if ms_per_layer < 20:
    print("✓ PASS: MoE is fast (< 20ms/layer), sync points removed")
else:
    print(f"✗ FAIL: MoE is slow ({ms_per_layer:.2f}ms/layer), sync points remain")
