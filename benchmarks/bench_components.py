#!/usr/bin/env python3
"""Profile all components to find true bottleneck."""
from transformers import AutoTokenizer
from metal_marlin.trellis.model import TrellisForCausalLM
import logging
import time
import warnings

import torch


logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

print('Loading model...')

model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', device='mps')
tokenizer = AutoTokenizer.from_pretrained(
    'models/GLM-4.7-Flash-Marlin-MMFP4', trust_remote_code=True)

tokens = tokenizer('Hi', return_tensors='pt').input_ids.to('mps')

# Profile helper


class Timer:
    def __init__(self, name):
        logger.debug("initializing %s with name=%s", type(self).__name__, name)
        self.name = name
        self.times = []

    def start(self):
        logger.debug("start called")
        torch.mps.synchronize()
        self._start = time.perf_counter()

    def stop(self):
        logger.debug("stop called")
        torch.mps.synchronize()
        self.times.append(time.perf_counter() - self._start)


timers = {
    'self_attn': [],
    'mlp': [],
    'layer_norm': [],
    'other': [],
}

# Patch model layers
for i, layer in enumerate(model.model.layers):
    orig_attn = layer.self_attn.forward
    orig_mlp = layer.mlp.forward

    def make_attn_wrapper(orig):
        logger.debug("make_attn_wrapper called with orig=%s", orig)
        def wrapper(*args, **kwargs):
            logger.debug("wrapper called")
            torch.mps.synchronize()
            start = time.perf_counter()
            result = orig(*args, **kwargs)
            torch.mps.synchronize()
            timers['self_attn'].append(time.perf_counter() - start)
            return result
        return wrapper

    def make_mlp_wrapper(orig):
        logger.debug("make_mlp_wrapper called with orig=%s", orig)
        def wrapper(*args, **kwargs):
            logger.debug("wrapper called")
            torch.mps.synchronize()
            start = time.perf_counter()
            result = orig(*args, **kwargs)
            torch.mps.synchronize()
            timers['mlp'].append(time.perf_counter() - start)
            return result
        return wrapper

    layer.self_attn.forward = make_attn_wrapper(orig_attn)
    layer.mlp.forward = make_mlp_wrapper(orig_mlp)

print('Running profiled forward pass...')
with torch.no_grad():
    start = time.perf_counter()
    out = model(tokens)
    torch.mps.synchronize()
    total = time.perf_counter() - start

print('\n=== PROFILE RESULTS ===')
print(f'Total forward time: {total:.2f}s')
print('\nPer-component breakdown:')

attn_total = sum(timers['self_attn'])
mlp_total = sum(timers['mlp'])
other = total - attn_total - mlp_total

print(
    f'  Attention: {attn_total:.2f}s ({attn_total/total*100:.0f}%) | {attn_total/len(timers["self_attn"])*1000:.1f}ms/layer')
print(
    f'  MLP (MoE): {mlp_total:.2f}s ({mlp_total/total*100:.0f}%) | {mlp_total/len(timers["mlp"])*1000:.1f}ms/layer')
print(f'  Other:     {other:.2f}s ({other/total*100:.0f}%)')
print(f'\nLayers: {len(timers["self_attn"])}')
