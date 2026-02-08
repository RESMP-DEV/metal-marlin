#!/usr/bin/env python3
"""Finish incomplete quantization using streaming RTN.

Loads ONLY the remaining layers directly from safetensors.
Does NOT use transformers or load the full model.
Uses RTN (no calibration) for remaining layers.

Usage:
    uv run python scripts/finish_quantization.py models/GLM-4.7-Flash-Marlin-MMFP4-CUDA

Memory: ~2GB peak (single layer at a time)
"""

from __future__ import annotations
from metal_marlin.mr_gptq import MRGPTQQuantizer, QuantizationFormat

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finish incomplete quantization (streaming RTN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to incomplete quantization output directory",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def load_checkpoint(output_dir: Path) -> dict:
    """Load checkpoint state."""
    checkpoint_file = output_dir / "checkpoints" / "state.json"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")
    with open(checkpoint_file, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(output_dir: Path, state: dict) -> None:
    """Save checkpoint state."""
    checkpoint_file = output_dir / "checkpoints" / "state.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_source_index(model_path: Path) -> dict[str, str]:
    """Get mapping of tensor names to source files."""
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            return json.load(f).get("weight_map", {})

    # Single file fallback
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    index = {}
    for f in safetensor_files:
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                index[key] = f.name
    return index


def get_all_tensor_names(model_path: Path) -> list[str]:
    """Get ordered list of all tensor names from source model."""
    index = get_source_index(model_path)
    return sorted(index.keys())


def load_tensor(model_path: Path, name: str, index: dict[str, str]) -> torch.Tensor:
    """Load a single tensor from source model."""
    filename = index[name]
    file_path = model_path / filename
    with safe_open(file_path, framework="pt") as f:
        return f.get_tensor(name)


def is_quantizable(name: str, tensor: torch.Tensor) -> bool:
    """Check if a tensor should be quantized (linear weight, 2D)."""
    if not name.endswith(".weight"):
        return False
    if tensor.dim() != 2:
        return False
    # Skip embeddings, norms, gates
    skip = ["embed", "norm", "ln_", "layernorm",
            "e_score_correction", "gate.weight"]
    lower = name.lower()
    for pattern in skip:
        if pattern in lower:
            return False
    # Must be large enough
    if tensor.shape[0] < 64 or tensor.shape[1] < 64:
        return False
    return True


def quantize_rtn_fp4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """RTN FP4 quantization (no Hessian, simple round-to-nearest).

    Returns:
        packed: Packed FP4 weights as uint32
        scales: Per-group scales as float16 [out_features, n_groups]
    """
    weight_np = weight.float().cpu().numpy()
    out_features, in_features = weight_np.shape

    # Pad in_features to multiple of group_size
    if in_features % group_size != 0:
        pad_size = group_size - (in_features % group_size)
        weight_np = np.pad(weight_np, ((0, 0), (0, pad_size)), mode='constant')
        in_features = weight_np.shape[1]

    n_groups = in_features // group_size

    # Reshape for per-group processing [out, n_groups, group_size]
    weight_grouped = weight_np.reshape(out_features, n_groups, group_size)

    # Compute per-group scales (max abs value)
    scales = np.abs(weight_grouped).max(axis=2, keepdims=True)
    scales = np.maximum(scales, 1e-8)  # Avoid division by zero

    # Normalize weights to [-1, 1]
    weight_norm = weight_grouped / scales

    # FP4 E2M1 quantization bins: -6, -4, -2, -1, 0, 1, 2, 4, 6 (scaled to [-1,1])
    # Map to 4-bit codes 0-15
    fp4_bins = np.array([-0.857, -0.571, -0.286, -0.143,
                        0.0, 0.143, 0.286, 0.571, 0.857])

    # Quantize to nearest bin
    quantized = np.zeros_like(weight_norm, dtype=np.uint8)
    for i, bin_val in enumerate(fp4_bins):
        bin_quantized = np.uint8(i + 1)  # Codes 1-9
        mask = (weight_norm >= (bin_val - 0.07)
                ) & (weight_norm < (bin_val + 0.07))
        quantized[mask] = bin_quantized

    # Handle extremes
    quantized[weight_norm < -0.857] = 0
    quantized[weight_norm >= 0.857] = 9

    # Pack pairs of 4-bit values into uint8
    quantized_flat = quantized.reshape(out_features, -1)  # [out, in_features]

    # Pack into uint32 (8 nibbles per uint32)
    n_uint32 = (quantized_flat.shape[1] + 7) // 8
    packed = np.zeros((out_features, n_uint32), dtype=np.uint32)

    for i in range(min(8, quantized_flat.shape[1])):
        if i < quantized_flat.shape[1]:
            packed |= quantized_flat[:, i::8].astype(np.uint32) << (i * 4)

    # Scales: [out_features, n_groups]
    # Scale for FP4 range
    scales_out = (scales.squeeze(-1) / 6.0).astype(np.float16)

    return packed, scales_out


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    verbose = args.verbose

    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        return 1

    # Load checkpoint
    state = load_checkpoint(output_dir)
    model_path = Path(state["model_path"])
    completed = set(state["completed_layers"])
    current_layer = state.get("current_layer", "")

    if verbose:
        print(f"Output dir: {output_dir}")
        print(f"Model path: {model_path}")
        print(f"Completed layers: {len(completed)}")
        print(f"Current layer: {current_layer}")

    # Get all tensor names from source
    source_index = get_source_index(model_path)
    all_tensors = sorted(source_index.keys())
    remaining = [t for t in all_tensors if t not in completed]

    if not remaining:
        print("All layers already quantized!")
        return 0

    print(f"Remaining tensors to process: {len(remaining)}")

    # Process remaining layers one at a time
    quantizer = MRGPTQQuantizer(
        bits=4,
        format=QuantizationFormat.FP4,
        group_size=args.group_size,
        use_hadamard=False,  # No Hadamard for RTN
        actorder=False,
    )

    # Find next shard to write
    existing_shards = sorted(output_dir.glob("model-*.safetensors"))
    next_shard_idx = len(existing_shards) + 1

    # Load existing weight map if present
    index_file = output_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            output_index = json.load(f)
    else:
        output_index = {"metadata": {"format": "mmfp4"}, "weight_map": {}}

    # Process remaining layers
    current_shard_tensors = {}
    current_shard_size = 0
    MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024  # 5GB per shard

    for i, name in enumerate(remaining):
        if verbose:
            print(f"[{i+1}/{len(remaining)}] Processing: {name}")

        # Load tensor from source
        tensor = load_tensor(model_path, name, source_index)

        if is_quantizable(name, tensor):
            # Quantize
            packed, scales = quantize_rtn_fp4(tensor, args.group_size)

            # Add to current shard
            weight_name = name
            scale_name = name.replace(".weight", ".scales")

            current_shard_tensors[weight_name] = torch.from_numpy(packed)
            current_shard_tensors[scale_name] = torch.from_numpy(scales)
            current_shard_size += packed.nbytes + scales.nbytes

            if verbose:
                print(
                    f"  Quantized: {tensor.shape} -> packed {packed.shape}, scales {scales.shape}")
        else:
            # Copy non-quantizable tensor as-is
            current_shard_tensors[name] = tensor.to(torch.float16)
            current_shard_size += tensor.numel() * 2

            if verbose:
                print(f"  Copied: {tensor.shape}")

        # Mark as completed
        completed.add(name)

        # Save shard if large enough or last tensor
        if current_shard_size >= MAX_SHARD_SIZE or i == len(remaining) - 1:
            shard_name = f"model-{next_shard_idx:05d}-of-00048.safetensors"
            shard_path = output_dir / shard_name

            if verbose:
                print(
                    f"  Saving shard: {shard_name} ({len(current_shard_tensors)} tensors, {current_shard_size/(1024**3):.2f} GB)")

            save_file(current_shard_tensors, shard_path)

            # Update weight map
            for tensor_name in current_shard_tensors:
                output_index["weight_map"][tensor_name] = shard_name

            # Save index
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(output_index, f, indent=2)

            # Save checkpoint
            state["completed_layers"] = list(completed)
            state["current_layer"] = name
            save_checkpoint(output_dir, state)

            # Reset for next shard
            current_shard_tensors = {}
            current_shard_size = 0
            next_shard_idx += 1

        # Free memory
        del tensor
        gc.collect()

    print(f"Done! Completed all {len(completed)} layers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
