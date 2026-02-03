#!/usr/bin/env python3
import time

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

print("Testing Different Top-K Values:")
print("=" * 60)

model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

# Try to modify top-k if possible
for layer_idx in [1, 2]:
    layer = model.model.layers[layer_idx]
    if hasattr(layer.mlp, 'num_experts_per_tok'):
        print(f"Layer {layer_idx} top-k: {layer.mlp.num_experts_per_tok}")
    elif hasattr(layer.mlp, 'topk'):
        print(f"Layer {layer_idx} top-k: {layer.mlp.topk}")

x = torch.randint(0, 1000, (1, 128)).to('mps')

# Test current configuration
print("\nTesting current top-k configuration...")
t0 = time.perf_counter()
with torch.no_grad():
    out = model(x)
torch.mps.synchronize()
current_time = time.perf_counter() - t0

print(f"Time: {current_time:.2f}s ({128/current_time:.1f} tok/s)")

# If we can modify top-k, test with top-1
# (This requires finding how to set it)
