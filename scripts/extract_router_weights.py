"""Extract MoE router weights from base model for trellis inference."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file


def extract_router_weights(model_name: str, output_path: Path):
    """Extract router gate weights from MoE model using safetensors directly."""
    print(f"Scanning {model_name} for safetensors files...")

    # Find all safetensors files in the repo
    files = list_repo_files(model_name)
    safetensor_files = [f for f in files if f.endswith(".safetensors")]
    print(f"Found {len(safetensor_files)} safetensors files")

    router_weights = {}
    for sf_file in safetensor_files:
        print(f"  Downloading {sf_file}...")
        local_path = hf_hub_download(model_name, sf_file)

        with safe_open(local_path, framework="pt") as f:
            for key in f.keys():
                # Look for router/gate weights (not expert gate_proj)
                if "router" in key.lower() or ".mlp.gate.weight" in key.lower():
                    if "expert" not in key.lower():
                        tensor = f.get_tensor(key)
                        router_weights[key] = tensor
                        print(f"    Found: {key} {tensor.shape}")

    print(f"\nFound {len(router_weights)} router weights")
    if router_weights:
        save_file(router_weights, output_path)
        print(f"Saved to {output_path}")
    else:
        print("No router weights found! Model may not be MoE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract_router_weights(args.model, args.output)
