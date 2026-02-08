#!/usr/bin/env python3
"""Debug Hessian collection for GPTQ quantization."""

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from metal_marlin.layer_replacement import collect_moe_expert_hessians


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
    from metal_marlin.eval import load_wikitext2

    texts = load_wikitext2(max_samples=5)
    calibration_inputs = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to("mps")
        for text in texts
    ]

    print(f"Collecting Hessians from {len(calibration_inputs)} samples...")
    hessians = collect_moe_expert_hessians(model, calibration_inputs, device="mps")
    print(f"Collected hessians for {len(hessians)} layers")

    for layer_name, expert_hessians in hessians.items():
        print(f"{layer_name}: {len(expert_hessians)} experts")
        for idx, H in list(expert_hessians.items())[:2]:
            print(f"  Expert {idx}: Hessian shape {H.shape}, dtype {H.dtype}")
            print(f"    diag mean: {np.mean(np.diag(H)):.4f}, max: {H.max():.4f}")

    # Test quantizing ONE expert to see if GPTQ works
    print("\nTesting GPTQ on single expert...")
    from metal_marlin.layer_replacement import find_moe_layers
    from metal_marlin.mr_gptq import MRGPTQQuantizer, QuantizationFormat

    moe_layers = find_moe_layers(model)
    for name, moe in moe_layers.items():
        experts = getattr(moe, "experts", None)
        gate_up = getattr(experts, "gate_up_proj", None)
        if gate_up is None:
            continue

        # Get single expert weight
        if hasattr(gate_up, "weight"):
            weight = gate_up.weight
        elif hasattr(gate_up, "data"):
            weight = gate_up.data
        else:
            continue

        weight_np = weight[0].detach().float().cpu().numpy()  # Expert 0
        print(f"Expert 0 weight shape: {weight_np.shape}")

        # Get Hessian for expert 0
        expert_hessians = hessians.get(name, {})
        H = expert_hessians.get(0)
        if H is None:
            print("No Hessian found for expert 0")
            continue

        print(f"Hessian shape: {H.shape}")

        # Try GPTQ quantization
        quantizer = MRGPTQQuantizer(
            bits=4,
            format=QuantizationFormat.FP4,
            group_size=128,
            use_hadamard=True,
        )

        try:
            packed, scales, meta = quantizer.quantize_layer(
                weight_np,
                hessian=H,
                layer_name="test_expert_0",
                use_hadamard=True,
            )
            print("GPTQ SUCCESS!")
            print(f"  packed shape: {packed.shape}")
            print(f"  scales shape: {scales.shape}")
            print(f"  error: {meta.get('error', {})}")
        except Exception as e:
            print(f"GPTQ FAILED: {e}")
            import traceback

import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

            traceback.print_exc()

        break


if __name__ == "__main__":
    main()
