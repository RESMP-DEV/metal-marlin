"""Extract non-quantized weights from base model for trellis inference."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def extract_base_weights(model_name: str, output_path: Path):
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Patterns for non-quantized weights
    keep_patterns = [
        "embed_tokens",
        "layernorm",
        "ln_",
        "norm",
        "lm_head",
    ]

    weights = {}
    for name, param in model.named_parameters():
        if any(p in name.lower() for p in keep_patterns):
            weights[name] = param.data.clone()
            print(f"  Extracted: {name} {param.shape}")

    print(f"\nExtracted {len(weights)} weights")
    save_file(weights, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract_base_weights(args.model, args.output)
