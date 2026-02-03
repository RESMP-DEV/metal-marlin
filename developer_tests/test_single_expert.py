#!/usr/bin/env python3
"""Test: Single expert vs full MoE to isolate overhead."""
import time
from pathlib import Path

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained(
    Path(__file__).parent / 'models' / 'GLM-4.7-Flash-Trellis-3bpw',
    device='mps'
)

x = torch.randint(0, 1000, (1, 128)).to('mps')

# Warmup
hidden_states = model.model.embed_tokens(x)
_ = model.model.layers[1](hidden_states)

# Test: Force only expert 0
layer = model.model.layers[1]
num_tokens = hidden_states.shape[0]

# Time single expert via direct call
t0 = time.perf_counter()
with torch.no_grad():
    single_out = layer.experts[0](hidden_states)
torch.mps.synchronize()
single_time = time.perf_counter() - t0

print(f"Single expert (3 GEMMs: gate+up+down): {single_time:.3f}s")
print(f"Estimated active experts per MoE forward: {5.0/single_time:.0f}")
