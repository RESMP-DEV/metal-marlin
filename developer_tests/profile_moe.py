#!/usr/bin/env python3
"""Profile MoE layer to identify bottlenecks."""
import time

import torch

from metal_marlin.trellis.lm import TrellisForCausalLM

model = TrellisForCausalLM.from_pretrained('models/GLM-4.7-Flash-Trellis-3bpw', device='mps')

x = torch.randint(0, 1000, (1, 128)).to('mps')
hidden_states = model.model.embed_tokens(x)
hidden_states = model.model.layers[0](hidden_states)  # Through dense layer

mlp = model.model.layers[1].mlp
print(f"MoE MLP type: {type(mlp).__name__}")
print(f"Num experts: {len(mlp.experts)}")
print(f"Experts per token: {mlp.num_experts_per_tok}")
print(f"Fast MoE enabled: {mlp._use_fast_moe}")
print(f"Fast MoE disabled: {mlp._fast_moe_permanently_disabled}")
print(f"Input device: {hidden_states.device}")
print()

# Warmup
print("Warming up...")
_ = mlp(hidden_states)
torch.mps.synchronize()

# Profile with timing breakdown
print("\n=== Profiling MoE Forward Pass ===")

# Time the full MoE forward
t0 = time.perf_counter()
with torch.no_grad():
    output = mlp(hidden_states)
torch.mps.synchronize()
full_time = time.perf_counter() - t0
print(f"Full MoE forward: {full_time*1000:.1f}ms")

print(f"\nInput shape: {hidden_states.shape}")
print(f"Output shape: {output.shape}")
print(f"Is MPS: {hidden_states.is_mps}")

# Check if falling back to slow path
if mlp._fast_moe_failure_count > 0:
    print(f"⚠️  Fast MoE failures: {mlp._fast_moe_failure_count}")
if mlp._fast_moe_permanently_disabled:
    print("⚠️  Fast MoE PERMANENTLY DISABLED - using CPU fallback!")
    print("    This explains the 11.5x slowdown!")
