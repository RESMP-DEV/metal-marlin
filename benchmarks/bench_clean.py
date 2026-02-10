#!/usr/bin/env python3
"""Profile without per-call syncs to get accurate timing."""
from transformers import AutoTokenizer
from metal_marlin.trellis.model import TrellisForCausalLM
import time
import warnings

import torch

warnings.filterwarnings('ignore')

print('Loading model...')

model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', device='mps')
tokenizer = AutoTokenizer.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', trust_remote_code=True)

tokens = tokenizer('Hello', return_tensors='pt').input_ids.to('mps')

# Warmup
print('Warming up...')
for _ in range(3):
    with torch.no_grad():
        _ = model(tokens)
torch.mps.synchronize()

# Benchmark 5 runs
print('Benchmarking...')
times = []
for i in range(5):
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model(tokens)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f'  Run {i+1}: {elapsed*1000:.0f}ms ({1/elapsed:.1f} tok/s)')

avg = sum(times) / len(times)
best = min(times)

print('\n=== RESULTS ===')
print(f'Average: {avg*1000:.0f}ms ({1/avg:.1f} tok/s)')
print(f'Best:    {best*1000:.0f}ms ({1/best:.1f} tok/s)')
print(f'\n47 layers @ ~{avg/47*1000:.1f}ms/layer')

# Check if compile available
print(f'\ntorch.compile available: {hasattr(torch, "compile")}')
print(f'MPS device: {torch.backends.mps.is_available()}')
