#!/usr/bin/env python3
"""Fix missing base_weights.safetensors in MMFP4 checkpoint.

The MMFP4 quant pipeline missed copying model.norm.weight from the original
checkpoint. This script creates base_weights.safetensors with:
- model.embed_tokens.weight (from MMFP4 shard 48)
- model.norm.weight (from original GLM-4.7-Flash shard 47)
- lm_head.weight (from MMFP4 shard 48)
"""

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    models_dir = Path(__file__).parent.parent / "models"

    # Load original norm weight from HF checkpoint
    orig_shard = models_dir / "GLM-4.7-Flash" / "model-00047-of-00048.safetensors"
    if not orig_shard.exists():
        raise FileNotFoundError(f"Original model not found: {orig_shard}")

    with safe_open(orig_shard, framework="pt") as f:
        norm_weight = f.get_tensor("model.norm.weight")
        print(
            f"Loaded model.norm.weight: {norm_weight.shape}, {norm_weight.dtype}")

    # Load embed_tokens and lm_head from MMFP4 (these exist)
    mmfp4_dir = models_dir / "GLM-4.7-Flash-Marlin-MMFP4"
    mmfp4_shard = mmfp4_dir / "model-00048-of-00048.safetensors"

    with safe_open(mmfp4_shard, framework="pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight")
        lm_head = f.get_tensor("lm_head.weight")
        print(f"Loaded embed_tokens: {embed.shape}, {embed.dtype}")
        print(f"Loaded lm_head: {lm_head.shape}, {lm_head.dtype}")

    # Create base_weights.safetensors
    base_weights = {
        "model.embed_tokens.weight": embed,
        "model.norm.weight": norm_weight.to(embed.dtype),  # Match dtype
        "lm_head.weight": lm_head,
    }

    output = mmfp4_dir / "base_weights.safetensors"
    save_file(base_weights, output)
    print(f"\nSaved: {output}")

    # Verify
    print("\nVerification:")
    with safe_open(output, framework="pt") as f:
        for k in sorted(f.keys()):
            t = f.get_tensor(k)
            print(f"  {k}: {t.shape}, {t.dtype}")


if __name__ == "__main__":
    main()
