#!/usr/bin/env python3
import time

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

print("Loading model...")
model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', device='mps')

# Single prefill test with timeout
seq_len = 128
x = torch.randint(0, 1000, (1, seq_len)).to('mps')

print(f"Testing single prefill (seq_len={seq_len})...")
t0 = time.perf_counter()

with torch.no_grad():
    out = model(x)

torch.mps.synchronize()
elapsed = time.perf_counter() - t0

print(f"Prefill completed in {elapsed:.2f}s")
print(f"Throughput: {seq_len/elapsed:.1f} tok/s")
print(f"Output shape: {out.logits.shape}")

# Expected: >10 tok/s for Metal, <1 tok/s for CPU fallback
if elapsed > 20:
    print("WARNING: Very slow, likely CPU fallback")
else:
    print("SUCCESS: Metal acceleration working")
