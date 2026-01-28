#!/usr/bin/env python
"""Quick test of MoE quantization with fast RTN (no MR-GPTQ)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
print("Loading GLM-4.7-Flash...")
model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-4.7-Flash",
    torch_dtype=torch.bfloat16,
    device_map="mps",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=False)


from metal_marlin.layer_replacement import MetalQuantizedMoE, find_moe_layers
from metal_marlin.quantize_fp4 import quantize_fp4

# Find MoE layers
moe_layers = find_moe_layers(model)
print(f"Found {len(moe_layers)} MoE blocks")

# Quantize just the first MoE layer as a test
moe_name = list(moe_layers.keys())[0]
moe = moe_layers[moe_name]
print(f"Testing with: {moe_name}")
print(f"experts type: {type(moe.experts).__name__}")

# Get expert weights
experts = moe.experts
gate_up = experts.gate_up_proj  # [64, 3072, 2048] or similar
down = experts.down_proj  # [64, 2048, 3072]
print(f"gate_up_proj shape: {gate_up.shape}")
print(f"down_proj shape: {down.shape}")

# Quick RTN quantization
group_size = 128
num_experts = gate_up.shape[0]


def quantize_expert_stack(weights, group_size):
    """Quantize [num_experts, out, in] -> packed + scales."""
    weights_np = weights.detach().float().cpu().numpy()
    packed_list = []
    scales_list = []
    for e in range(weights_np.shape[0]):
        # quantize_fp4 expects [out, in], marlin_layout transposes to [in, out]
        packed, scales = quantize_fp4(weights_np[e], group_size=group_size, marlin_layout=True)
        packed_list.append(torch.from_numpy(packed))
        scales_list.append(torch.from_numpy(scales))
    return torch.stack(packed_list).to("mps"), torch.stack(scales_list).to("mps")


print("Quantizing experts (RTN)...")
gate_up_packed, gate_up_scales = quantize_expert_stack(gate_up, group_size)
down_packed, down_scales = quantize_expert_stack(down, group_size)
print(f"gate_up_packed: {gate_up_packed.shape}, scales: {gate_up_scales.shape}")
print(f"down_packed: {down_packed.shape}, scales: {down_scales.shape}")

# Create MetalQuantizedMoE
quantized_experts = MetalQuantizedMoE(
    gate_up_weight_packed=gate_up_packed,
    gate_up_scales=gate_up_scales,
    down_weight_packed=down_packed,
    down_scales=down_scales,
    bits=4,
    group_size=group_size,
    format="fp4",
)

# Replace experts
print("Replacing experts...")
moe._metal_marlin_experts_fp16 = experts
moe.experts = quantized_experts

# Test forward
print("\nTesting forward pass...")
test_hidden = torch.randn(4, 2048, dtype=torch.bfloat16, device="mps")
with torch.no_grad():
    # The MoE block handles routing
    output = moe(test_hidden)
print(f"Forward output shape: {output.shape}")
print("Forward pass PASSED!")

# Test generation
print("\nTesting generation...")
model.eval()
prompt = "Hello, I am"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\nSingle MoE layer quantization test PASSED!")
