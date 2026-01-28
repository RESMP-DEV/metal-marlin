#!/usr/bin/env python
"""Test GLM-4.7-Flash full quantization (attention + MoE experts)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metal_marlin.layer_replacement import (
    find_linear_layers,
    find_moe_layers,
    replace_linear_layers,
    replace_moe_layers,
)

print("Loading GLM-4.7-Flash...")
model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-4.7-Flash",
    torch_dtype=torch.bfloat16,
    device_map="mps",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=False)

# Count layers
linears = find_linear_layers(model)
moe_layers = find_moe_layers(model)
print(f"Total Linear layers: {len(linears)}")
print(f"Total MoE blocks: {len(moe_layers)}")
print("Sample linear layers:")
for i, name in enumerate(list(linears.keys())[:5]):
    shape = linears[name].weight.shape
    print(f"  {name}: {shape}")
print("MoE blocks found:")
for name in list(moe_layers.keys())[:3]:
    print(f"  {name}")

# Step 1: Quantize MoE experts (the bulk of the model: ~28B params)
print("\n=== Phase 1: Quantizing MoE experts ===")
moe_stats = replace_moe_layers(
    model,
    bits=4,
    group_size=128,
    format="fp4",
)
print(f"MoE replaced: {moe_stats['replaced_count']}, Skipped: {moe_stats['skipped_count']}")
print(f"MoE params quantized: {moe_stats['total_params_quantized']:,}")

# Step 2: Quantize attention projections and other Linear layers
print("\n=== Phase 2: Quantizing attention layers ===")
# Skip embeddings, norms, and MoE related (already handled)
skip_patterns = [
    "embed_tokens",
    "lm_head",
    "shared_head",
    "layernorm",
    "norm",
    "mlp.gate",  # Router gate (small, keep FP16)
]
linear_stats = replace_linear_layers(
    model,
    bits=4,
    group_size=128,
    format="fp4",
    skip_patterns=skip_patterns,
)
print(
    f"Linear replaced: {linear_stats['replaced_count']}, Skipped: {linear_stats['skipped_count']}"
)
print(f"Linear params quantized: {linear_stats['total_params_quantized']:,}")

# Summary
total_quantized = moe_stats["total_params_quantized"] + linear_stats["total_params_quantized"]
print(f"\n=== TOTAL: {total_quantized:,} params quantized ===")

# Test generation
print("\nTesting generation...")
model.eval()
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
