#!/usr/bin/env python3
"""Quick benchmark for GLM-4.7-Flash-Trellis-MM decode speed."""
import time
import warnings

import torch
from metal_marlin.trellis.model import TrellisForCausalLM
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


print('Loading model...')
model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Trellis-MM', device='mps')
tokenizer = AutoTokenizer.from_pretrained(
    'models/GLM-4.7-Flash-Trellis-MM', trust_remote_code=True)

prompt = 'The meaning of life is'
tokens = tokenizer(prompt, return_tensors='pt').input_ids.to('mps')

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = model(tokens)
torch.mps.synchronize()

print('Benchmarking decode (50 tokens)...')
total_tokens = 50
times = []
for run in range(3):
    input_ids = tokens.clone()
    start = time.perf_counter()
    for _ in range(total_tokens):
        with torch.no_grad():
            out = model(input_ids)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f'  Run {run+1}: {total_tokens/elapsed:.1f} tok/s')

avg = sum(times) / len(times)
print(f'\nAverage: {total_tokens/avg:.1f} tok/s')
print('Target: 15-30 tok/s | Baseline: 0.1 tok/s')
