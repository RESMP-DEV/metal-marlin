#!/usr/bin/env python3
from pathlib import Path

from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained(
    Path(__file__).parent / 'models' / 'GLM-4.7-Flash-Trellis-3bpw',
    device='mps'
)

print("Layer 0 type:", type(model.model.layers[0]))
print("Layer 1 type:", type(model.model.layers[1]))
print()

# Check layer 1 structure
layer1 = model.model.layers[1]
for name, module in layer1.named_children():
    print(f"  {name}: {type(module)}")

# Check if it has mlp with moe
if hasattr(layer1, 'mlp'):
    mlp = layer1.mlp
    print("\nMLP type:", type(mlp))
    for name, module in mlp.named_children():
        print(f"  MLP.{name}: {type(module)}")
        if hasattr(module, 'experts'):
            print(f"    has {len(module.experts)} experts")
