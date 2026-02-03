#!/usr/bin/env python3
import time

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

print("Loading model...")
model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')
print("Model loaded\n")

# Patch layers to add timing
layer_times = {}

def make_timed_forward(name, original_forward):
    def timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_forward(*args, **kwargs)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        layer_times[name] = elapsed
        return result
    return timed_forward

# Patch first few layers
for i in range(min(5, len(model.model.layers))):
    layer = model.model.layers[i]
    layer.forward = make_timed_forward(f"layer_{i}", layer.forward)

# Run test
x = torch.randint(0, 1000, (1, 128)).to('mps')

print("Running 128-token forward pass...")
t0 = time.perf_counter()
with torch.no_grad():
    out = model(x)
torch.mps.synchronize()
total_time = time.perf_counter() - t0

print(f"\nTotal time: {total_time:.2f}s")
print(f"Throughput: {128/total_time:.1f} tok/s\n")

print("Per-layer timing (first 5 layers):")
for name, t in sorted(layer_times.items()):
    print(f"  {name}: {t:.3f}s ({t/total_time*100:.1f}%)")

# Determine if Metal is working
if total_time > 10:
    print("\n⚠️  WARNING: Very slow, likely CPU fallback or Metal validation blocking")
elif total_time > 2:
    print("\n⚠️  Slower than expected, possible Metal inefficiency")
else:
    print("\n✓ Performance reasonable, Metal likely working")
