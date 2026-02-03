#!/usr/bin/env python3
"""
Minimal reproduction of GLM-4 MoE slowness.

This isolates the exact operation that's slow.
"""
import time

import torch

print("Minimal MoE Slow-down Reproduction:")
print("=" * 60)

# Step 1: Test basic Metal operations
print("\n1. Basic Metal matmul:")
a = torch.randn(128, 4096, device='mps', dtype=torch.float16)
b = torch.randn(4096, 14336, device='mps', dtype=torch.float16)

t0 = time.perf_counter()
c = torch.matmul(a, b)
torch.mps.synchronize()
matmul_time = time.perf_counter() - t0
print(f"   {matmul_time*1000:.1f}ms")

# Step 2: Test with expert loop
print("\n2. Expert loop (3 iterations):")
experts = [torch.randn(4096, 14336, device='mps', dtype=torch.float16) for _ in range(3)]

t0 = time.perf_counter()
results = []
for expert in experts:
    results.append(torch.matmul(a, expert))
torch.mps.synchronize()
loop_time = time.perf_counter() - t0
print(f"   {loop_time*1000:.1f}ms ({loop_time/matmul_time:.1f}x)")

# Step 3: Test with indexing
print("\n3. Expert selection with indexing:")
topk_indices = torch.randint(0, 64, (128, 3), device='mps')
expert_weights = torch.randn(64, 4096, 14336, device='mps', dtype=torch.float16)

t0 = time.perf_counter()
# Simplified dispatch
selected = expert_weights[topk_indices[:, 0]]  # [128, 4096, 14336]
output = torch.bmm(a.unsqueeze(1), selected)
torch.mps.synchronize()
indexed_time = time.perf_counter() - t0
print(f"   {indexed_time*1000:.1f}ms")

# Step 4: Test actual model layer
print("\n4. Actual model MoE layer:")
from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

x = torch.randint(0, 1000, (1, 64)).to('mps')
hidden_states = model.model.embed_tokens(x)
hidden_states = model.model.layers[0](hidden_states)[0]

t0 = time.perf_counter()
hidden_states = model.model.layers[1](hidden_states)[0]
torch.mps.synchronize()
model_time = time.perf_counter() - t0
print(f"   {model_time*1000:.1f}ms")

# Analysis
print("\nAnalysis:")
if model_time > indexed_time * 10:
    print("  ⚠️  Model layer is MUCH slower than expected")
    print("  Likely cause: Quantization overhead or kernel dispatch issue")
elif model_time > indexed_time * 3:
    print("  ⚠️  Model layer is slower than expected")
    print("  Possible cause: Additional operations or inefficient kernels")
else:
    print("  ✓ Model layer performance reasonable")
