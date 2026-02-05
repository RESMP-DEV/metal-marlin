#!/usr/bin/env python3
"""Migrate Trellis v2 models to v3 (sharded HuggingFace-compatible) format.

Usage:
    uv run python scripts/migrate_trellis_v2_to_v3.py \
        --input models/GLM-4.7-Flash-Trellis-3bpw \
        --output models/GLM-4.7-Flash-Trellis-3bpw-v3

The v2 format stores each layer in a separate directory:
    layer_XXXX/
        index.json
        tensor_*.safetensors (or batch_*.safetensors)

The v3 format uses HuggingFace-compatible sharded safetensors:
    config.json
    model.safetensors.index.json
    model-00001-of-XXXXX.safetensors
    tokenizer.json, tokenizer_config.json, etc.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm


@dataclass
class TensorInfo:
    """Information about a tensor in the model."""

    name: str  # Original name (e.g., "model.layers.0.mlp.down_proj.weight")
    key: str  # Safe key (e.g., "model__layers__0__mlp__down_proj__weight")
    shape: tuple[int, ...]
    bits: int
    mse: float | None
    component: str  # "indices", "scales", "su", "sv"


class ShardWriter:
    """Writes tensors to sharded safetensors files.

    Follows HuggingFace's sharded format with:
    - model-XXXXX-of-XXXXX.safetensors files
    - model.safetensors.index.json for mapping
    """

    def __init__(
        self,
        output_dir: Path,
        shard_size_bytes: int = 5 * 1024 * 1024 * 1024,  # 5GB default
        prefix: str = "model",
    ):
        self.output_dir = Path(output_dir)
        self.shard_size_bytes = shard_size_bytes
        self.prefix = prefix

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._current_shard: dict[str, torch.Tensor] = {}
        self._current_shard_size = 0
        self._shard_count = 0
        self._weight_map: dict[str, str] = {}
        self._tensor_metadata: dict[str, dict[str, Any]] = {}

    def _get_shard_filename(self, shard_num: int, total_shards: int) -> str:
        """Generate shard filename following HF conventions."""
        return f"{self.prefix}-{shard_num:05d}-of-{total_shards:05d}.safetensors"

    def _write_current_shard(self) -> None:
        """Write the current shard to disk."""
        if not self._current_shard:
            return

        self._shard_count += 1
        # Use temporary filename, will rename after all shards are written
        shard_file = self.output_dir / f"{self.prefix}_{self._shard_count:05d}.safetensors"

        save_file(self._current_shard, str(shard_file))

        # Track weight map
        for tensor_name in self._current_shard.keys():
            self._weight_map[tensor_name] = shard_file.name

        self._current_shard = {}
        self._current_shard_size = 0

    def add_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        bits: int | None = None,
        mse: float | None = None,
    ) -> None:
        """Add a tensor to the current shard.

        Args:
            name: Full tensor name (with dots, e.g., "model.layers.0.mlp.down_proj.weight")
            tensor: The tensor to save
            bits: Optional bits info for metadata
            mse: Optional MSE for metadata
        """
        tensor_size = tensor.numel() * tensor.element_size()

        # Check if we need to start a new shard
        if (
            self._current_shard_size > 0
            and self._current_shard_size + tensor_size > self.shard_size_bytes
        ):
            self._write_current_shard()

        # Store the tensor (convert numpy to torch if needed)
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        self._current_shard[name] = tensor
        self._current_shard_size += tensor_size

        # Store metadata
        self._tensor_metadata[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        if bits is not None:
            self._tensor_metadata[name]["bits"] = bits
        if mse is not None:
            self._tensor_metadata[name]["mse"] = mse

    def finalize(self) -> None:
        """Finalize all shards and write the index file."""
        # Write the last shard
        if self._current_shard:
            self._write_current_shard()

        # Rename shards to proper format
        temp_files = sorted(self.output_dir.glob(f"{self.prefix}_*.safetensors"))
        total_shards = len(temp_files)

        for i, temp_file in enumerate(temp_files, 1):
            new_name = self._get_shard_filename(i, total_shards)
            temp_file.rename(self.output_dir / new_name)

            # Update weight map with new filenames
            for tensor_name, old_filename in list(self._weight_map.items()):
                if old_filename == temp_file.name:
                    self._weight_map[tensor_name] = new_name

        # Write index file
        index = {
            "metadata": {
                "format": "trellis_v3",
                "total_size": sum(
                    m["shape"][0] * m.get("shape", [1])[0] if len(m["shape"]) > 0 else 1
                    for m in self._tensor_metadata.values()
                ),
            },
            "weight_map": self._weight_map,
        }

        index_file = self.output_dir / f"{self.prefix}.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

        print(f"\nWrote {total_shards} shard(s) to {self.output_dir}")
        print(f"  Index: {index_file.name}")


def load_v2_layer(
    layer_dir: Path,
) -> tuple[list[TensorInfo], dict[str, torch.Tensor]]:
    """Load a v2 format layer directory.

    Returns:
        Tuple of (tensor_infos, tensor_data)
    """
    from safetensors.torch import load_file

    # Load index.json
    index_path = layer_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Layer index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    # Find all safetensors files
    tensor_files = list(layer_dir.glob("tensor_*.safetensors"))
    if not tensor_files:
        tensor_files = list(layer_dir.glob("batch_*.safetensors"))

    if not tensor_files:
        raise FileNotFoundError(f"No tensor files found in {layer_dir}")

    # Load all tensors
    all_tensors: dict[str, torch.Tensor] = {}
    for tensor_file in tensor_files:
        tensors = load_file(tensor_file)
        all_tensors.update(tensors)

    # Parse tensor metadata
    tensor_infos = []
    for tensor_meta in index.get("tensors", []):
        name = tensor_meta["name"]
        bits = tensor_meta.get("bits", 4)
        shape = tuple(tensor_meta.get("shape", []))
        mse = tensor_meta.get("mse")

        # Build safe key
        safe_key = name.replace(".", "__")

        # Each quantized weight has 4 components
        for component in ["indices", "scales", "su", "sv"]:
            component_key = f"{safe_key}__{component}"
            if component_key in all_tensors:
                tensor_infos.append(
                    TensorInfo(
                        name=name,
                        key=safe_key,
                        shape=shape,
                        bits=bits,
                        mse=mse,
                        component=component,
                    )
                )

    return tensor_infos, all_tensors


def migrate_model(
    input_dir: Path,
    output_dir: Path,
    shard_size_gb: float = 5.0,
    verify: bool = True,
) -> None:
    """Migrate a v2 Trellis model to v3 format.

    Args:
        input_dir: Path to v2 model directory
        output_dir: Path to write v3 model
        shard_size_gb: Target size per shard in GB
        verify: Whether to verify the migration
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"Migrating v2 model from: {input_dir}")
    print(f"Output v3 model to: {output_dir}")

    # Discover layers
    layer_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("layer_")])
    if not layer_dirs:
        raise FileNotFoundError(f"No layer directories found in {input_dir}")

    layer_indices = [int(d.name.split("_")[1]) for d in layer_dirs]
    print(f"\nFound {len(layer_dirs)} layers (indices: {min(layer_indices)}-{max(layer_indices)})")

    # Copy config files
    print("\nCopying config files...")
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = input_dir / config_file
        if src.exists():
            dst = output_dir / config_file
            shutil.copy2(src, dst)
            print(f"  Copied: {config_file}")

    # Also copy router weights and base weights if they exist
    for special_file in ["router_weights.safetensors", "base_weights.safetensors"]:
        src = input_dir / special_file
        if src.exists():
            dst = output_dir / special_file
            shutil.copy2(src, dst)
            print(f"  Copied: {special_file}")

    # Create shard writer
    shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)
    writer = ShardWriter(output_dir, shard_size_bytes=shard_size_bytes)

    # Track all tensor shapes for verification
    v2_shapes: dict[str, tuple[int, ...]] = {}

    # Process each layer
    print("\nProcessing layers...")
    for layer_dir in tqdm(layer_dirs, desc="Layers"):
        layer_idx = int(layer_dir.name.split("_")[1])

        try:
            tensor_infos, tensor_data = load_v2_layer(layer_dir)
        except FileNotFoundError as e:
            tqdm.write(f"  Warning: Skipping {layer_dir.name}: {e}")
            continue

        # Group by weight name
        weight_groups: dict[str, list[TensorInfo]] = {}
        for info in tensor_infos:
            if info.key not in weight_groups:
                weight_groups[info.key] = []
            weight_groups[info.key].append(info)

        # Add tensors to shards
        for weight_key, components in weight_groups.items():
            # Get original name from first component
            weight_name = components[0].name
            bits = components[0].bits
            mse = components[0].mse

            for comp_info in components:
                component_key = f"{weight_key}__{comp_info.component}"
                if component_key in tensor_data:
                    tensor = tensor_data[component_key]

                    # Store shape for verification
                    v2_shapes[f"{weight_name}.{comp_info.component}"] = tuple(tensor.shape)

                    # Convert to torch if needed
                    if isinstance(tensor, np.ndarray):
                        tensor = torch.from_numpy(tensor)

                    # Write with v3 naming convention (dots instead of underscores)
                    v3_name = f"{weight_name}.{comp_info.component}"
                    writer.add_tensor(v3_name, tensor, bits=bits, mse=mse)

    # Finalize shards
    writer.finalize()

    # Verification
    if verify:
        print("\nVerifying round-trip...")
        verify_migration(output_dir, v2_shapes)

    print(f"\n✓ Migration complete: {output_dir}")


def verify_migration(
    output_dir: Path,
    v2_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Verify the v3 model by loading and checking tensor shapes.

    Args:
        output_dir: Path to v3 model
        v2_shapes: Dictionary of tensor names to shapes from v2
    """
    from safetensors import safe_open

    # Load the index
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Check that all v2 tensors are present in v3
    v3_shapes: dict[str, tuple[int, ...]] = {}

    # Group by shard for efficient loading
    shard_to_tensors: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        if shard_name not in shard_to_tensors:
            shard_to_tensors[shard_name] = []
        shard_to_tensors[shard_name].append(tensor_name)

    # Load shapes from each shard
    for shard_name, tensor_names in shard_to_tensors.items():
        shard_path = output_dir / shard_name
        with safe_open(str(shard_path), framework="pt") as f:
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name)
                v3_shapes[tensor_name] = tuple(tensor.shape)

    # Compare shapes
    print(f"  V2 tensors: {len(v2_shapes)}")
    print(f"  V3 tensors: {len(v3_shapes)}")

    mismatches = []
    missing = []

    for name, v2_shape in v2_shapes.items():
        # Convert v2 name format to v3
        # v2 keys in our dict are already dot-separated: model.layers.0.mlp.down_proj.weight.indices
        v3_name = name.replace("__", ".")

        if v3_name not in v3_shapes:
            missing.append(name)
            continue

        v3_shape = v3_shapes[v3_name]
        if v2_shape != v3_shape:
            mismatches.append((name, v2_shape, v3_shape))

    if missing:
        print(f"  ⚠️  Missing tensors: {len(missing)}")
        for name in missing[:5]:
            print(f"    - {name}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    if mismatches:
        print(f"  ⚠️  Shape mismatches: {len(mismatches)}")
        for name, v2, v3 in mismatches[:5]:
            print(f"    - {name}: v2={v2}, v3={v3}")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches) - 5} more")

    if not missing and not mismatches:
        print("  ✓ All tensor shapes match!")
    else:
        raise RuntimeError("Verification failed: tensor mismatches found")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Trellis v2 models to v3 format"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input v2 model directory (containing layer_XXXX/ subdirs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output v3 model directory",
    )
    parser.add_argument(
        "--shard-size",
        "-s",
        type=float,
        default=5.0,
        help="Target shard size in GB (default: 5.0)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        migrate_model(
            input_dir=args.input,
            output_dir=args.output,
            shard_size_gb=args.shard_size,
            verify=not args.no_verify,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
