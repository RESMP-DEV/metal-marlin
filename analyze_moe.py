#!/usr/bin/env python
"""Analyze GLM-4.7-Flash MoE architecture for Metal kernel design."""

from transformers import AutoConfig

config = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=False)

print("=== GLM-4.7-Flash MoE Architecture ===")
print(f"Hidden size: {config.hidden_size}")
print(f"Intermediate size: {config.intermediate_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num experts: {config.n_routed_experts}")
print(f"Num shared experts: {config.n_shared_experts}")
print(f"Top-k experts: {config.num_experts_per_tok}")
print(f"MoE intermediate size: {config.moe_intermediate_size}")

# Calculate expert dimensions
expert_in = config.hidden_size
expert_intermediate = config.moe_intermediate_size
print("\n=== Expert Dimensions ===")
print(f"Expert input: {expert_in}")
print(f"Expert intermediate: {expert_intermediate}")
print(f"gate_proj per expert: [{expert_in}, {expert_intermediate}]")
print(f"up_proj per expert: [{expert_in}, {expert_intermediate}]")
print(f"down_proj per expert: [{expert_intermediate}, {expert_in}]")

# Memory per expert (FP16)
gate_up_params = expert_in * expert_intermediate * 2  # gate + up fused
down_params = expert_intermediate * expert_in
expert_params = gate_up_params + down_params
print(f"\nParams per expert: {expert_params:,} ({expert_params * 2 / 1e6:.1f} MB FP16)")
print(f"Total expert params: {expert_params * config.n_routed_experts:,}")
print(f"FP16 size: {expert_params * config.n_routed_experts * 2 / 1e9:.2f} GB")
print(f"4-bit quantized: {expert_params * config.n_routed_experts / 2 / 1e9:.2f} GB")

print("\n=== Kernel Requirements ===")
print("1. MoE router: softmax over expert logits, top-k selection")
print("2. Grouped GEMM: batch tokens to selected experts")
print("3. Gate/Up projection: [B, hidden] -> [B, intermediate] (fused SiLU)")
print("4. Down projection: [B, intermediate] -> [B, hidden]")
print("5. Expert output combination: weighted sum of expert outputs")
