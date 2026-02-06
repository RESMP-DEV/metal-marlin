#!/usr/bin/env python3
"""Debug dynamic dispatch setup."""
from metal_marlin.trellis.model import TrellisForCausalLM
import sys

import torch

sys.path.insert(0, '.')

print('Loading model...')
model = TrellisForCausalLM.from_pretrained(
    'models/GLM-4.7-Flash-Trellis-MM', device='mps')

# Find first MoE layer
for i, layer in enumerate(model.model.layers):
    if hasattr(layer.mlp, '_is_mixed_precision'):
        print(f'Layer {i} is MoE:')
        mlp = layer.mlp
        print(f'  _is_mixed_precision: {mlp._is_mixed_precision}')
        print(f'  _dynamic_dispatcher: {mlp._dynamic_dispatcher is not None}')
        print(
            f'  _cached_weight_buffers: {mlp._cached_weight_buffers is not None}')
        if mlp._cached_weight_buffers:
            cached = mlp._cached_weight_buffers
            tw = cached._torch_gate_weights
            print(
                f'  torch_gate_weights: type={type(tw)}, is_tensor={isinstance(tw, torch.Tensor)}')
            if isinstance(tw, torch.Tensor):
                print(f'    shape={tw.shape}, device={tw.device}')
        break
