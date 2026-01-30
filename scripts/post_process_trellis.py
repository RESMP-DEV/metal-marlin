#!/usr/bin/env python3
"""Post-process Trellis quantized models to pack indices.

Converts existing models with unpacked int16 indices to packed uint8 format.
This reduces storage by 3-5x depending on quantization bits:
- 3-bit: 16 bits -> 3 bits = 5.33x reduction
- 4-bit: 16 bits -> 4 bits = 4x reduction
- 2-bit: 16 bits -> 2 bits = 8x reduction

The script processes in-place with backup, or creates a new output directory.

Usage:
    cd contrib/metal_marlin
    uv run python scripts/post_process_trellis.py models/GLM-4.7-Flash-EXL3-3bpw

    # Create new output (preserves original)
    uv run python scripts/post_process_trellis.py models/original --output models/packed

    # Dry run to see savings without modifying
    uv run python scripts/post_process_trellis.py models/mymodel --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent.parent))

from metal_marlin.trellis.packing import pack_indices_vectorized


def discover_layers(model_path: Path) -> list[int]:
    """Find all layer directories in the model."""
    layers = []
    for item in model_path.iterdir():
        if item.is_dir() and item.name.startswith("layer_"):
            try:
                idx = int(item.name.split("_")[1])
                layers.append(idx)
            except (IndexError, ValueError):
                continue
    return sorted(layers)


def process_layer(
    layer_dir: Path,
    output_dir: Path | None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Process a single layer directory.

    Returns:
        Tuple of (tensors_processed, original_bytes, packed_bytes)
    """
    # Load metadata
    index_path = layer_dir / "index.json"
    if not index_path.exists():
        print(f"  Warning: No index.json in {layer_dir}, skipping")
        return 0, 0, 0

    with open(index_path) as f:
        metadata = json.load(f)

    tensor_infos = metadata.get("tensors", [])
    if not tensor_infos:
        print(f"  Warning: No tensors in {layer_dir}, skipping")
        return 0, 0, 0

    # Build mapping from tensor name to bits
    name_to_bits = {t["name"]: t["bits"] for t in tensor_infos}

    # Find and process safetensor files (tensor_*.safetensors or batch_*.safetensors)
    tensor_files = sorted(layer_dir.glob("tensor_*.safetensors"))
    if not tensor_files:
        # Fall back to batch_*.safetensors (from parallel quantizer)
        tensor_files = sorted(layer_dir.glob("batch_*.safetensors"))
    if not tensor_files:
        print(f"  Warning: No tensor files in {layer_dir}, skipping")
        return 0, 0, 0

    tensors_processed = 0
    original_bytes = 0
    packed_bytes = 0

    # Determine output directory
    if output_dir is not None:
        out_layer_dir = output_dir / layer_dir.name
        out_layer_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_layer_dir = layer_dir

    for tensor_file in tensor_files:
        tensors = load_file(str(tensor_file))

        # Separate indices from other arrays
        packed_tensors: dict[str, np.ndarray] = {}
        indices_metadata: dict[str, dict] = {}

        for key, arr in tensors.items():
            if key.endswith("__indices"):
                # Extract base tensor name
                base_name = key[:-9].replace("__", ".")  # Remove __indices, convert to dot notation

                # Get bits from metadata
                bits = name_to_bits.get(base_name, 4)  # Default to 4-bit if not found

                original_bytes += arr.nbytes

                if dry_run:
                    # Just calculate savings
                    n_indices = int(np.prod(arr.shape))
                    estimated_packed = 1 + (n_indices * bits + 7) // 8
                    packed_bytes += estimated_packed
                else:
                    # Actually pack the indices
                    packed = pack_indices_vectorized(arr, bits)
                    packed_bytes += packed.nbytes

                    # Store packed array
                    packed_tensors[key] = packed

                    # Store metadata for unpacking
                    indices_metadata[key] = {
                        "shape": list(arr.shape),
                        "n_indices": int(np.prod(arr.shape)),
                        "bits": bits,
                    }

                tensors_processed += 1
            else:
                # Keep other arrays as-is (scales, su, sv)
                packed_tensors[key] = arr
                original_bytes += arr.nbytes
                packed_bytes += arr.nbytes

        if not dry_run:
            # Save the packed tensor file
            out_tensor_file = out_layer_dir / tensor_file.name
            save_file(packed_tensors, str(out_tensor_file))

        # Cleanup
        del tensors
        del packed_tensors
        gc.collect()

    # Update metadata with packing info
    if not dry_run:
        # Add packing format to metadata
        metadata["format"] = "trellis_v2"
        metadata["packing"] = {
            "indices_format": "packed_uint8",
            "header_byte": True,
        }

        # Update index
        out_index_path = out_layer_dir / "index.json"
        with open(out_index_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return tensors_processed, original_bytes, packed_bytes


def update_model_index(model_path: Path, output_path: Path | None) -> None:
    """Update the main quantization_index.json with format info."""
    src_index = model_path / "quantization_index.json"
    if not src_index.exists():
        return

    target_path = output_path if output_path else model_path
    target_index = target_path / "quantization_index.json"

    with open(src_index) as f:
        index = json.load(f)

    # Update quantization format
    if "quantization" in index:
        index["quantization"]["format"] = "trellis_v2"
        index["quantization"]["indices_packed"] = True

    with open(target_index, "w") as f:
        json.dump(index, f, indent=2)


def rename_model_dir(model_path: Path) -> Path | None:
    """Rename model directory from EXL3 to Trellis if needed."""
    name = model_path.name
    if "EXL3" in name:
        new_name = name.replace("EXL3", "Trellis")
        new_path = model_path.parent / new_name
        if new_path.exists():
            print(f"  Warning: {new_path} already exists, not renaming")
            return None
        return new_path
    return None


def process_model(
    model_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False,
    rename: bool = True,
) -> dict:
    """Process entire model.

    Args:
        model_path: Path to input model directory
        output_path: Optional output directory (if None, modifies in-place)
        dry_run: If True, only calculate savings without modifying
        rename: If True, rename directory from EXL3 to Trellis

    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'=' * 60}")
    print(f"Post-processing Trellis model: {model_path}")
    print(f"{'=' * 60}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    layers = discover_layers(model_path)
    if not layers:
        raise ValueError(f"No layer directories found in {model_path}")

    print(f"Found {len(layers)} layers")
    print(f"Mode: {'DRY RUN' if dry_run else 'PROCESSING'}")
    if output_path:
        print(f"Output: {output_path}")
    else:
        print("Output: in-place modification")
    print()

    # Process each layer
    total_tensors = 0
    total_original = 0
    total_packed = 0

    for layer_idx in layers:
        layer_dir = model_path / f"layer_{layer_idx:04d}"
        print(f"Layer {layer_idx}: ", end="", flush=True)

        tensors, orig, packed = process_layer(layer_dir, output_path, dry_run)

        compression = orig / packed if packed > 0 else 0
        print(
            f"{tensors} tensors, {orig / 1024 / 1024:.1f} MB -> "
            f"{packed / 1024 / 1024:.1f} MB ({compression:.2f}x)"
        )

        total_tensors += tensors
        total_original += orig
        total_packed += packed

    # Update model index
    if not dry_run:
        update_model_index(model_path, output_path)

    # Calculate savings
    savings_bytes = total_original - total_packed
    overall_compression = total_original / total_packed if total_packed > 0 else 0

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Tensors processed: {total_tensors}")
    print(f"Original size: {total_original / 1024 / 1024 / 1024:.2f} GB")
    print(f"Packed size: {total_packed / 1024 / 1024 / 1024:.2f} GB")
    print(f"Savings: {savings_bytes / 1024 / 1024 / 1024:.2f} GB ({overall_compression:.2f}x)")

    # Rename directory if requested
    final_path = output_path if output_path else model_path
    if rename and not dry_run:
        new_path = rename_model_dir(final_path)
        if new_path:
            print(f"\nRenaming: {final_path} -> {new_path}")
            shutil.move(str(final_path), str(new_path))
            final_path = new_path

    return {
        "tensors": total_tensors,
        "original_bytes": total_original,
        "packed_bytes": total_packed,
        "savings_bytes": savings_bytes,
        "compression_ratio": overall_compression,
        "final_path": str(final_path),
    }


def copy_unchanged_files(model_path: Path, output_path: Path) -> None:
    """Copy files that don't need modification to output directory."""
    output_path.mkdir(parents=True, exist_ok=True)

    for item in model_path.iterdir():
        if item.is_file():
            # Copy config files, etc.
            if item.suffix in [".json", ".txt", ".md"]:
                shutil.copy(item, output_path / item.name)
        elif item.is_dir() and not item.name.startswith("layer_"):
            # Copy non-layer directories (like tokenizer files)
            shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process Trellis models to pack indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pack model in-place
    uv run python scripts/post_process_trellis.py models/GLM-4.7-Flash-EXL3-3bpw

    # Pack to new directory (preserves original)
    uv run python scripts/post_process_trellis.py models/unpacked --output models/packed

    # Dry run: see savings without modifying
    uv run python scripts/post_process_trellis.py models/mymodel --dry-run

    # Don't rename EXL3 -> Trellis
    uv run python scripts/post_process_trellis.py models/mymodel --no-rename
        """,
    )
    parser.add_argument("model_path", type=Path, help="Path to model directory")
    parser.add_argument("--output", "-o", type=Path, help="Output directory (default: in-place)")
    parser.add_argument("--dry-run", action="store_true", help="Show savings without modifying")
    parser.add_argument(
        "--no-rename", action="store_true", help="Don't rename EXL3 -> Trellis in directory name"
    )

    args = parser.parse_args()

    # Copy unchanged files if creating new output
    if args.output and not args.dry_run:
        print(f"Copying unchanged files to {args.output}...")
        copy_unchanged_files(args.model_path, args.output)

    stats = process_model(
        model_path=args.model_path,
        output_path=args.output,
        dry_run=args.dry_run,
        rename=not args.no_rename,
    )

    if not args.dry_run:
        print(f"\nDone! Model saved to: {stats['final_path']}")
    else:
        print(f"\n[DRY RUN] Would save {stats['savings_bytes'] / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
