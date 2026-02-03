#!/usr/bin/env python3
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore')

# Add contrib path
sys.path.insert(0, str(Path(__file__).parent))

from metal_marlin.trellis.lm import TrellisForCausalLM

model_path = Path(__file__).parent / "models" / "GLM-4.7-Flash-Trellis-3bpw"
print(f"Loading model from: {model_path}")

model = TrellisForCausalLM.from_pretrained(str(model_path), device='mps')

x = torch.randint(0, 1000, (1, 64)).to('mps')
hidden_states = model.model.embed_tokens(x)

print("\nLayer-by-Layer Timing:")
print("=" * 60)

layer_times = []

for i, layer in enumerate(model.model.layers):
    t0 = time.perf_counter()
    with torch.no_grad():
        result = layer(hidden_states)
        # Handle both tuple and single return values
        if isinstance(result, tuple):
            hidden_states = result[0]
        else:
            hidden_states = result
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    layer_times.append((i, elapsed))
    print(f"Layer {i:2d}: {elapsed*1000:6.1f}ms")

    if elapsed > 1.0:
        print("  ⚠️  SLOW LAYER")

# Statistics
times = [t for _, t in layer_times]
print("\nStatistics:")
print(f"  Mean: {sum(times)/len(times)*1000:.1f}ms")
print(f"  Min: {min(times)*1000:.1f}ms")
print(f"  Max: {max(times)*1000:.1f}ms")
print(f"  Total: {sum(times):.2f}s")

# Find outliers
mean = sum(times) / len(times)
outliers = [(i, t) for i, t in layer_times if t > mean * 2]
if outliers:
    print("\nOutliers (>2x mean):")
    for i, t in outliers:
        print(f"  Layer {i}: {t*1000:.1f}ms ({t/mean:.1f}x mean)")

# Show distribution
print("\nTime Distribution:")
times_sorted = sorted(times)
for p in [10, 25, 50, 75, 90]:
    idx = int(len(times_sorted) * p / 100)
    print(f"  {p:3d}th percentile: {times_sorted[idx]*1000:6.1f}ms")
