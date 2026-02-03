#!/usr/bin/env python3
import torch

# Test FP4 dequant correctness
print("Testing FP4 Dequantization:")
print("=" * 60)

# Create test packed FP4 data
packed_data = torch.randint(0, 255, (1024,), dtype=torch.uint8, device='mps')
scales = torch.randn(128, device='mps', dtype=torch.float16)

# Try to call dequant kernel (need to find the actual function)
# This depends on the implementation

try:
    from metal_marlin.trellis import quantization
    print("Quantization module found")

    # List available functions
    funcs = [x for x in dir(quantization) if 'dequant' in x.lower() or 'fp4' in x.lower()]
    print(f"FP4 functions: {funcs}")

except ImportError:
    print("Cannot import quantization module")

# Alternative: Check if weights are pre-dequantized
from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

expert = model.model.layers[1].mlp.experts[0]
for name, param in expert.named_parameters():
    if 'weight' in name.lower():
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Dtype: {param.dtype}")
        print(f"  Min/Max: {param.min().item():.3f} / {param.max().item():.3f}")
        print(f"  Mean: {param.mean().item():.3f}")
