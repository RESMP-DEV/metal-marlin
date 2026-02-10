#!/usr/bin/env python3
"""Debug forward pass for GLM-4.7-Flash-Marlin-MMFP4."""
from transformers import AutoTokenizer
from metal_marlin.trellis.model import TrellisForCausalLM
import sys
import time
import traceback
import warnings

import torch

warnings.filterwarnings('ignore')

print('Step 1: Import model class...')

print('OK')

print('Step 2: Import tokenizer...')

print('OK')

print('Step 3: Load model...')
model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', device='mps')
print('OK')

print('Step 4: Load tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', trust_remote_code=True)
print('OK')

print('Step 5: Tokenize input...')
prompt = 'Hello'
tokens = tokenizer(prompt, return_tensors='pt').input_ids.to('mps')
print(f'OK - tokens shape: {tokens.shape}')

print('Step 6: First forward pass...')
try:
    with torch.no_grad():
        start = time.perf_counter()
        out = model(tokens)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
    print(f'OK - logits shape: {out.logits.shape}, time: {elapsed:.2f}s')
except Exception as e:
    print(f'FAILED: {e}')
    traceback.print_exc()
    sys.exit(1)

print('Step 7: Second forward pass...')
try:
    with torch.no_grad():
        start = time.perf_counter()
        out = model(tokens)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
    print(f'OK - time: {elapsed:.2f}s ({1/elapsed:.2f} tok/s)')
except Exception as e:
    print(f'FAILED: {e}')
    traceback.print_exc()
    sys.exit(1)

print('\n=== SUCCESS ===')
print(f'Estimated throughput: ~{1/elapsed:.1f} tok/s')
