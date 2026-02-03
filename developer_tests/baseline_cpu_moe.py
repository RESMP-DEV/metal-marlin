#!/usr/bin/env python3
import time

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

print("CPU vs MPS MoE Comparison:")
print("=" * 60)

# Test on CPU
print("\n1. Loading model on CPU...")
model_cpu = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='cpu')

x_cpu = torch.randint(0, 1000, (1, 64))  # Smaller for CPU

print("Running CPU forward...")
t0 = time.perf_counter()
with torch.no_grad():
    out_cpu = model_cpu(x_cpu)
cpu_time = time.perf_counter() - t0
print(f"CPU time (64 tokens): {cpu_time:.2f}s ({64/cpu_time:.2f} tok/s)")

# Test on MPS
print("\n2. Loading model on MPS...")
model_mps = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

x_mps = torch.randint(0, 1000, (1, 64)).to('mps')

print("Running MPS forward...")
t0 = time.perf_counter()
with torch.no_grad():
    out_mps = model_mps(x_mps)
torch.mps.synchronize()
mps_time = time.perf_counter() - t0
print(f"MPS time (64 tokens): {mps_time:.2f}s ({64/mps_time:.2f} tok/s)")

print(f"\nSpeedup: {cpu_time/mps_time:.2f}x")

if mps_time >= cpu_time:
    print("⚠️  MPS is SLOWER than CPU - Metal not being used!")
elif cpu_time/mps_time < 2:
    print("⚠️  MPS only slightly faster - possible CPU fallback")
else:
    print("✓ MPS is faster - Metal acceleration working")
