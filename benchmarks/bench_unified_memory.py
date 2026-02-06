#!/usr/bin/env python3
"""Benchmark unified memory optimizations."""
import time
import torch
from metal_marlin.trellis.model import TrellisForCausalLM
from transformers import AutoTokenizer

print('Loading model...')
model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Trellis-MM', device='mps')
tokenizer = AutoTokenizer.from_pretrained(
    'models/GLM-4.7-Flash-Trellis-MM', trust_remote_code=True)

tokens = tokenizer('Hello', return_tensors='pt').input_ids.to('mps')

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model(tokens)
torch.mps.synchronize()

# Benchmark
times = []
for _ in range(10):
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model(tokens)
    torch.mps.synchronize()
    times.append(time.perf_counter() - start)

avg = sum(times) / len(times)
best = min(times)
print(f'Average: {avg*1000:.0f}ms ({1/avg:.1f} tok/s)')
print(f'Best:    {best*1000:.0f}ms ({1/best:.1f} tok/s)')
print(f'Baseline was: 0.1 tok/s')
print(f'Previous:     0.3 tok/s')
print(f'Target:       15-30 tok/s')
