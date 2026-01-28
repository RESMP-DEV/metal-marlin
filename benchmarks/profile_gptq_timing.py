#!/usr/bin/env python3
"""Profile GPTQ timing breakdown."""

import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from metal_marlin.eval_perplexity import load_wikitext2
from metal_marlin.layer_replacement import (
    collect_moe_expert_hessians,
    find_moe_layers,
    quantize_moe_experts,
)


def main():
    # Load 2 layers
    config = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
    config.num_hidden_layers = 2

    model = AutoModelForCausalLM.from_pretrained(
        "zai-org/GLM-4.7-Flash",
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)

    # Load calibration data
    texts = load_wikitext2(max_samples=10)
    calibration_inputs = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to("mps")
        for text in texts
    ]

    # Time Hessian collection
    start = time.perf_counter()
    hessians = collect_moe_expert_hessians(model, calibration_inputs, device="mps")
    elapsed_hessian = time.perf_counter() - start
    print(f"Hessian collection: {elapsed_hessian:.2f}s")

    # Time quantization
    moe_layers = find_moe_layers(model)
    layer_name = list(moe_layers.keys())[0]
    moe = moe_layers[layer_name]

    layer_hessians = hessians.get(layer_name)
    print(f"Layer {layer_name} has {len(layer_hessians) if layer_hessians else 0} expert hessians")

    start = time.perf_counter()
    quantized = quantize_moe_experts(
        moe,
        bits=4,
        group_size=128,
        format="fp4",
        hessians=layer_hessians,
        use_gptq=True,
    )
    elapsed_quant = time.perf_counter() - start
    print(f"Quantization: {elapsed_quant:.2f}s")

    print(f"Total: {elapsed_hessian + elapsed_quant:.2f}s")


if __name__ == "__main__":
    main()
