#!/usr/bin/env python3
import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

moe_layer = model.model.layers[1]

if hasattr(moe_layer.mlp, 'experts'):
    print("Expert Weight Analysis:")
    print("=" * 60)

    expert = moe_layer.mlp.experts[0]

    for name, param in expert.named_parameters():
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Dtype: {param.dtype}")
        print(f"  Device: {param.device}")
        print(f"  Requires grad: {param.requires_grad}")
        print(f"  Is quantized: {hasattr(param, 'qscheme')}")

        if hasattr(param, 'qscheme'):
            print(f"  Quantization scheme: {param.qscheme()}")

    # Check for Trellis-specific attributes
    for name in dir(expert):
        if 'weight' in name.lower() or 'scale' in name.lower():
            attr = getattr(expert, name)
            if torch.is_tensor(attr):
                print(f"\n{name}: {attr.shape}, {attr.dtype}, {attr.device}")
else:
    print("Cannot access experts")
