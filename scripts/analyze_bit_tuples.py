#!/usr/bin/env python3
"""Analyze bit tuple distribution for uniform bits percentage."""

import json
from collections import defaultdict
from pathlib import Path

path = Path('models/GLM-4.7-Flash-Trellis-MM/quantization_index.json')
data = json.load(open(path))

# Build expert bit tuples from layers array
# Each entry has name like "model.layers.X.mlp.experts.Y.gate_proj.weight"
# (layer_idx, expert_idx) -> {gate: bits, up: bits, down: bits}
expert_bits = defaultdict(dict)

for entry in data['layers']:
    name = entry['name']
    bits = entry['bits']

    if '.mlp.experts.' not in name:
        continue

    # Parse: model.layers.X.mlp.experts.Y.PROJ.weight
    parts = name.split('.')
    layer_idx = int(parts[2])
    expert_idx = int(parts[5])
    proj = parts[6]  # gate_proj, up_proj, or down_proj

    key = (layer_idx, expert_idx)
    if 'gate' in proj:
        expert_bits[key]['gate'] = bits
    elif 'up' in proj:
        expert_bits[key]['up'] = bits
    elif 'down' in proj:
        expert_bits[key]['down'] = bits

# Analyze first 3 layers
for layer_idx in range(3):
    layer_experts = {k: v for k, v in expert_bits.items() if k[0] == layer_idx}

    tuples = defaultdict(int)
    uniform_count = 0

    for key, bits_dict in layer_experts.items():
        if 'gate' in bits_dict and 'up' in bits_dict and 'down' in bits_dict:
            t = (bits_dict['gate'], bits_dict['up'], bits_dict['down'])
            tuples[t] += 1
            if t[0] == t[1] == t[2]:
                uniform_count += 1

    print(f'Layer {layer_idx}:')
    total = sum(tuples.values())
    for t, c in sorted(tuples.items(), key=lambda x: -x[1])[:5]:
        uniform = 'UNIFORM' if t[0] == t[1] == t[2] else ''
        print(f'  {t}: {c} experts ({100*c/total:.1f}%) {uniform}')
    print(
        f'  Uniform bits: {uniform_count}/{total} ({100*uniform_count/total if total else 0:.1f}%)')
    print()
