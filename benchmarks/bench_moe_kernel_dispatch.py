#!/usr/bin/env python3
"""Micro-benchmark for dispatch_moe_trellis_swiglu with cached buffers."""
import time

import torch
from metal_marlin.trellis.model import TrellisForCausalLM
from metal_marlin.trellis.moe_dispatch import dispatch_moe_trellis_swiglu

print("Loading model...")
model = TrellisForCausalLM.from_pretrained(
    "models/GLM-4.7-Flash-Trellis-MM", device="mps"
)

# Find first MoE layer
moe = None
for layer in model.model.layers:
    if hasattr(layer.mlp, "_is_mixed_precision"):
        moe = layer.mlp
        break

if moe is None:
    raise RuntimeError("No MoE layer found")

cached = moe._cached_weight_buffers
if cached is None:
    raise RuntimeError("Cached weight buffers not initialized")

batch = 1
hidden_dim = moe.hidden_dim
intermediate_dim = moe.intermediate_dim
num_experts = len(moe.experts)
top_k = moe.num_experts_per_tok
bits = moe.bits

x = torch.randn(batch, hidden_dim, device="mps", dtype=torch.float16)
expert_ids = torch.zeros(batch, top_k, device="mps", dtype=torch.long)
expert_probs = torch.ones(batch, top_k, device="mps", dtype=torch.float32)

# Warmup
for _ in range(5):
    with torch.no_grad():
        out = dispatch_moe_trellis_swiglu(
            lib=moe._get_lib(),
            activations=x,
            gate_weights=None,
            gate_scales=None,
            up_weights=None,
            up_scales=None,
            down_weights=None,
            down_scales=None,
            gate_su=None,
            gate_sv=None,
            up_su=None,
            up_sv=None,
            down_su=None,
            down_sv=None,
            grid=None,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            bits=bits,
            cached_buffers=cached,
            buffer_pool=moe._get_buffer_pool(),
            use_fp32_acc=hidden_dim >= 1024,
        )
    out = out + 0

torch.mps.synchronize()

# Benchmark
iters = 50
times = []
for _ in range(iters):
    with torch.no_grad():
        torch.mps.synchronize()
        start = time.perf_counter()
        out = dispatch_moe_trellis_swiglu(
            lib=moe._get_lib(),
            activations=x,
            gate_weights=None,
            gate_scales=None,
            up_weights=None,
            up_scales=None,
            down_weights=None,
            down_scales=None,
            gate_su=None,
            gate_sv=None,
            up_su=None,
            up_sv=None,
            down_su=None,
            down_sv=None,
            grid=None,
            expert_ids=expert_ids,
            expert_probs=expert_probs,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            bits=bits,
            cached_buffers=cached,
            buffer_pool=moe._get_buffer_pool(),
            use_fp32_acc=hidden_dim >= 1024,
        )
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)

avg = sum(times) / len(times)
print(f"Kernel dispatch avg: {avg * 1000:.2f} ms")
print(f"Approx tok/s (batch=1): {1.0 / avg:.2f}")
