#!/usr/bin/env python3
"""Profile forward pass to find bottlenecks."""
import sys
import time
import warnings

import torch

warnings.filterwarnings('ignore')

print('Loading model...')
from metal_marlin.trellis.model import TrellisForCausalLM
from transformers import AutoTokenizer

model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Trellis-MM', device='mps')
tokenizer = AutoTokenizer.from_pretrained(
    'models/GLM-4.7-Flash-Trellis-MM', trust_remote_code=True)

tokens = tokenizer('Hi', return_tensors='pt').input_ids.to('mps')

# Patch to profile MoE dispatch
dispatch_times = []
orig_fallback = None

for i, layer in enumerate(model.model.layers):
    if hasattr(layer.mlp, '_forward_grouped_fallback'):
        orig = layer.mlp._forward_grouped_fallback
        def make_wrapper(layer_idx, fn):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                dispatch_times.append((layer_idx, elapsed))
                return result
            return wrapper
        layer.mlp._forward_grouped_fallback = make_wrapper(i, orig)

print('Running profiled forward pass...')
with torch.no_grad():
    start = time.perf_counter()
    out = model(tokens)
    torch.mps.synchronize()
    total = time.perf_counter() - start

print(f'\nTotal forward time: {total:.2f}s')
print(f'\nMoE dispatch times (top 10 slowest):')
dispatch_times.sort(key=lambda x: -x[1])
for layer_idx, elapsed in dispatch_times[:10]:
    print(f'  Layer {layer_idx}: {elapsed*1000:.1f}ms')

total_moe = sum(t for _, t in dispatch_times)
print(f'\nTotal MoE time: {total_moe:.2f}s ({total_moe/total*100:.0f}% of forward)')
print(f'Average per layer: {total_moe/len(dispatch_times)*1000:.1f}ms')
