"""Extract MoE router weights from base model for trellis inference."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def extract_router_weights(model_name: str, output_path: Path):
    """Extract router gate weights from MoE model."""
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Don't need GPU
    )

    router_weights = {}
    for name, param in model.named_parameters():
        if "router" in name.lower() or "gate" in name.lower():
            if "experts" not in name:  # Skip expert gates
                router_weights[name] = param.data
                print(f"  Extracted: {name} {param.shape}")

    print(f"\nFound {len(router_weights)} router weights")
    save_file(router_weights, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract_router_weights(args.model, args.output)
