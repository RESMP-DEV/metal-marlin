#!/usr/bin/env python3
"""Repair corrupted MMFP4 checkpoint by copying missing weights from original model.

This script:
1. Scans the MMFP4 checkpoint to identify weights claimed by index but missing from shards
2. Loads those weights from the original FP16 model
3. Saves them to new shard files in the checkpoint
4. Rebuilds the index to accurately reflect actual shard contents

Usage:
    python scripts/repair_mmfp4_checkpoint.py \
        --mmfp4 models/glm47-flash-mmfp4 \
        --original ~/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def find_missing_weights(mmfp4_path: Path, num_hidden_layers: int = 47) -> dict[str, str]:
    """Find weights that are in index but missing from actual shard files.

    Only reports missing weights for layers 0 to num_hidden_layers-1.
    Extra layers in checkpoint (like layer 47 in a 47-layer model) are NOT
    part of the model architecture and should be ignored.
    """
    index_path = mmfp4_path / "model.safetensors.index.json"
    if not index_path.exists():
        print("No index file found - single shard model")
        return {}

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    missing: dict[str, str] = {}  # key -> expected shard

    # Check each weight in index
    for key, shard_name in weight_map.items():
        # Skip layer indices >= num_hidden_layers (not used by model)
        import re
        layer_match = re.search(r"model\.layers\.(\d+)\.", key)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            if layer_idx >= num_hidden_layers:
                continue  # Skip - not part of model architecture

        shard_path = mmfp4_path / shard_name
        if not shard_path.exists():
            missing[key] = shard_name
            continue

        with safe_open(shard_path, framework="pt") as sf:
            if key not in sf.keys():
                missing[key] = shard_name

    return missing


def find_missing_nonquant_weights(
    mmfp4_path: Path,
    original_path: Path,
    num_hidden_layers: int = 47
) -> set[str]:
    """Find non-quantized weights that exist in original but are missing from MMFP4.

    These are weights that should be copied directly from FP16 checkpoint:
    - e_score_correction_bias (MoE routing biases)
    - layernorm weights (already handled by load_prequantized but good to verify)
    - gate.weight (MoE router weights)
    """
    # Patterns for non-quantized weights that must be preserved
    NONQUANT_PATTERNS = [
        r"\.e_score_correction_bias$",  # MoE routing bias
        r"\.gate\.weight$",  # MoE router weights
        r"_layernorm\.weight$",  # Various layernorms
        r"input_layernorm\.weight$",
        r"post_attention_layernorm\.weight$",
        r"model\.norm\.weight$",
    ]
    import re
    patterns = [re.compile(p) for p in NONQUANT_PATTERNS]

    # Get all keys from MMFP4 checkpoint
    mmfp4_keys: set[str] = set()
    index_path = mmfp4_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            mmfp4_keys = set(json.load(f).get("weight_map", {}).keys())
    else:
        for sf_path in mmfp4_path.glob("*.safetensors"):
            with safe_open(sf_path, framework="pt") as sf:
                mmfp4_keys.update(sf.keys())

    # Get all keys from original FP16 checkpoint
    original_keys: set[str] = set()
    orig_index = original_path / "model.safetensors.index.json"
    if orig_index.exists():
        with open(orig_index) as f:
            original_keys = set(json.load(f).get("weight_map", {}).keys())
    else:
        for sf_path in original_path.glob("*.safetensors"):
            with safe_open(sf_path, framework="pt") as sf:
                original_keys.update(sf.keys())

    # Find missing non-quantized weights
    missing: set[str] = set()
    for key in original_keys:
        # Skip layers >= num_hidden_layers
        layer_match = re.search(r"model\.layers\.(\d+)\.", key)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            if layer_idx >= num_hidden_layers:
                continue

        # Check if this is a non-quantized weight pattern
        if any(p.search(key) for p in patterns):
            if key not in mmfp4_keys:
                missing.add(key)

    return missing


def load_weights_from_original(
    original_path: Path,
    keys_to_load: set[str],
) -> dict[str, torch.Tensor]:
    """Load specified weights from original FP16 model."""
    loaded: dict[str, torch.Tensor] = {}

    # Find all safetensors files
    st_files = sorted(original_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {original_path}")

    for st_file in st_files:
        with safe_open(st_file, framework="pt") as sf:
            for key in sf.keys():
                if key in keys_to_load:
                    loaded[key] = sf.get_tensor(key)
                    print(f"  Loaded {key} from {st_file.name}")

    return loaded


def rebuild_index(mmfp4_path: Path) -> dict[str, str]:
    """Rebuild weight_map by scanning actual shard contents."""
    weight_map: dict[str, str] = {}

    for sf_path in sorted(mmfp4_path.glob("model-*.safetensors")):
        with safe_open(sf_path, framework="pt") as sf:
            for key in sf.keys():
                weight_map[key] = sf_path.name

    # Also check for single-file model
    single_file = mmfp4_path / "model.safetensors"
    if single_file.exists():
        with safe_open(single_file, framework="pt") as sf:
            for key in sf.keys():
                weight_map[key] = "model.safetensors"

    return weight_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair corrupted MMFP4 checkpoint",
    )
    parser.add_argument(
        "--mmfp4",
        type=Path,
        required=True,
        help="Path to MMFP4 checkpoint directory",
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Path to original FP16 model directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report issues, don't fix",
    )
    args = parser.parse_args()

    mmfp4_path = args.mmfp4.resolve()
    original_path = args.original.resolve()

    if not mmfp4_path.exists():
        print(f"ERROR: MMFP4 checkpoint not found: {mmfp4_path}")
        return 1
    if not original_path.exists():
        print(f"ERROR: Original model not found: {original_path}")
        return 1

    print("=" * 60)
    print("MMFP4 Checkpoint Repair")
    print("=" * 60)
    print(f"MMFP4 checkpoint: {mmfp4_path}")
    print(f"Original model: {original_path}")
    print()

    # Find missing weights (in index but not in shards)
    print("Scanning for missing weights (index vs shards)...")
    missing = find_missing_weights(mmfp4_path)
    
    # Find missing non-quantized weights (should exist but aren't in checkpoint at all)
    print("Scanning for missing non-quantized weights...")
    missing_nonquant = find_missing_nonquant_weights(mmfp4_path, original_path)
    
    # Combine for repair
    all_missing_keys = set(missing.keys()) | missing_nonquant
    
    if not all_missing_keys:
        print("No missing weights found - checkpoint appears complete!")
        print("\nRebuilding index to ensure accuracy...")
        if not args.dry_run:
            weight_map = rebuild_index(mmfp4_path)
            index_path = mmfp4_path / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump({
                    "metadata": {"format": "mmfp4"},
                    "weight_map": weight_map,
                }, f, indent=2)
            print(f"Index rebuilt with {len(weight_map)} entries")
        return 0

    print(f"\nFound {len(missing)} weights in index but missing from shards")
    print(f"Found {len(missing_nonquant)} non-quantized weights missing entirely")
    print(f"Total to repair: {len(all_missing_keys)}")
    
    # Show some examples
    if missing:
        print("\nIndex/shard mismatches:")
        for key, shard in list(missing.items())[:10]:
            print(f"  {key} (expected in {shard})")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    
    if missing_nonquant:
        print("\nMissing non-quantized weights:")
        for key in sorted(missing_nonquant)[:10]:
            print(f"  {key}")
        if len(missing_nonquant) > 10:
            print(f"  ... and {len(missing_nonquant) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] Would copy these weights from original model")
        return 0

    # Load missing weights from original
    print("\nLoading missing weights from original model...")
    loaded = load_weights_from_original(original_path, all_missing_keys)

    not_found = all_missing_keys - set(loaded.keys())
    if not_found:
        print(f"\nWARNING: {len(not_found)} weights not found in original:")
        for key in list(not_found)[:10]:
            print(f"  {key}")

    if not loaded:
        print("No weights to add")
        return 0

    # Group loaded weights by target shard
    # For index/shard mismatches, use the originally expected shard
    # For missing non-quantized weights, put them in appropriate shard based on layer
    import re
    by_shard: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in loaded.items():
        if key in missing:
            # Index claimed this shard
            target_shard = missing[key]
        else:
            # Determine shard from layer number
            layer_match = re.search(r"model\.layers\.(\d+)\.", key)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                # Shards are 1-indexed and each layer gets its own shard
                target_shard = f"model-{layer_idx + 1:05d}-of-00048.safetensors"
            else:
                # Put in last shard for non-layer weights
                target_shard = "model-00047-of-00048.safetensors"
        by_shard[target_shard][key] = tensor

    # Save weights to appropriate shards
    print("\nSaving weights to shards...")
    for shard_name, tensors in by_shard.items():
        shard_path = mmfp4_path / shard_name

        # Load existing shard content if it exists
        existing: dict[str, torch.Tensor] = {}
        if shard_path.exists():
            with safe_open(shard_path, framework="pt") as sf:
                for key in sf.keys():
                    existing[key] = sf.get_tensor(key)

        # Merge with new tensors
        merged = {**existing, **tensors}

        # Save merged shard
        save_file(merged, shard_path)
        print(f"  {shard_name}: added {len(tensors)} weights ({len(merged)} total)")

    # Rebuild index
    print("\nRebuilding index...")
    weight_map = rebuild_index(mmfp4_path)

    # Compute total size
    total_size = 0
    for sf_path in mmfp4_path.glob("*.safetensors"):
        total_size += sf_path.stat().st_size

    index_path = mmfp4_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump({
            "metadata": {"format": "mmfp4", "total_size": total_size},
            "weight_map": weight_map,
        }, f, indent=2)

    print(f"Index rebuilt with {len(weight_map)} entries")

    # Clean up any monkey-patch files
    missing_norm = mmfp4_path / "missing_norm.safetensors"
    if missing_norm.exists():
        missing_norm.unlink()
        print("Removed monkey-patch file: missing_norm.safetensors")

    print("\n" + "=" * 60)
    print("Repair complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
