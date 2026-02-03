#!/usr/bin/env python3
import time

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

x = torch.randint(0, 1000, (1, 128)).to('mps')

# Process through first layer only (dense layer)
print("Testing Layer 0 (dense layer)...")
hidden_states = model.model.embed_tokens(x)

t0 = time.perf_counter()
with torch.no_grad():
    hidden_states = model.model.layers[0](hidden_states)
torch.mps.synchronize()
layer0_time = time.perf_counter() - t0

print(f"Layer 0: {layer0_time:.3f}s")

# Process through second layer (first MoE layer)
print("\nTesting Layer 1 (first MoE layer)...")
t0 = time.perf_counter()
with torch.no_grad():
    hidden_states = model.model.layers[1](hidden_states)
torch.mps.synchronize()
layer1_time = time.perf_counter() - t0

print(f"Layer 1: {layer1_time:.3f}s")

print(f"\nRatio (MoE/Dense): {layer1_time/layer0_time:.1f}x")

if layer1_time > 1.0:
    print("⚠️  MoE layer is VERY slow - likely the bottleneck")
elif layer1_time > layer0_time * 5:
    print("⚠️  MoE layer is slower than expected")
else:
    print("✓ MoE layer performance reasonable")
