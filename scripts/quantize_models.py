#!/usr/bin/env python3
"""Quantize safetensors models to Marlin FP4/INT4 format.

Processes all safetensors files in a model directory, quantizing weight matrices
to the specified format while preserving other tensors.

Usage:
    python scripts/quantize_models.py models/Qwen3-4B --quant-type fp4
    python scripts/quantize_models.py models/Qwen3-32B --quant-type int4 --group-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

# Add parent to path for metal_marlin imports
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402


def quantize_to_fp4(
    weight: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize weight to FP4 E2M1 format.

    Returns (packed, scales, mins) where packed is uint8 with 2 values per byte.
    """
    from metal_marlin.quantize import pack_fp4_weights

    return pack_fp4_weights(weight, group_size)


def quantize_to_int4(
    weight: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize weight to INT4 asymmetric format.

    Returns (packed, scales, zeros).
    """
    K, N = weight.shape

    # Pad K to multiple of group_size
    pad_K = (group_size - K % group_size) % group_size
    if pad_K > 0:
        weight = np.pad(weight, ((0, pad_K), (0, 0)), mode="constant")

    K_padded = weight.shape[0]
    num_groups = K_padded // group_size

    # Reshape for per-group quantization
    weight_groups = weight.reshape(num_groups, group_size, N)

    # Compute min/max per group
    mins = weight_groups.min(axis=1)  # [num_groups, N]
    maxs = weight_groups.max(axis=1)

    # Compute scales and zeros
    scales = (maxs - mins) / 15.0
    scales = np.maximum(scales, 1e-8).astype(np.float16)
    zeros = mins.astype(np.float16)

    # Quantize to 0-15
    scales_expanded = np.repeat(scales, group_size, axis=0)
    zeros_expanded = np.repeat(zeros, group_size, axis=0)
    quantized = np.clip(
        np.round((weight - zeros_expanded) / (scales_expanded + 1e-8)), 0, 15
    ).astype(np.uint8)

    # Pack two 4-bit values per byte
    packed = (quantized[:, 0::2] & 0xF) | ((quantized[:, 1::2] & 0xF) << 4)

    return packed.astype(np.uint8), scales, zeros


def quantize_safetensors_file(
    input_path: Path,
    output_path: Path,
    quant_type: str = "fp4",
    group_size: int = 128,
    verbose: bool = True,
) -> dict[str, int]:
    """Quantize a single safetensors file.

    Returns dict with stats: {original_bytes, quantized_bytes, num_weights_quantized}
    """

    stats = {"original_bytes": 0, "quantized_bytes": 0, "num_quantized": 0}
    output_tensors: dict[str, np.ndarray] = {}

    # Use torch framework to handle bfloat16 properly
    with safe_open(str(input_path), framework="pt") as f:
        keys = list(f.keys())
        if verbose:
            print(f"  Processing {len(keys)} tensors...")

        for name in keys:
            raw_tensor = f.get_tensor(name)
            # Convert to float32 numpy
            tensor = raw_tensor.float().numpy()

            # Only quantize 2D weight matrices, skip biases/norms
            if tensor.ndim == 2 and tensor.shape[0] >= 64 and tensor.shape[1] >= 64:
                if "weight" in name and "norm" not in name.lower():
                    orig_bytes = tensor.nbytes
                    stats["original_bytes"] += orig_bytes

                    # Convert to float32 for quantization
                    weight = tensor.astype(np.float32)

                    if quant_type == "fp4":
                        packed, scales, meta = quantize_to_fp4(weight, group_size)
                        output_tensors[f"{name}.packed"] = packed
                        output_tensors[f"{name}.scales"] = scales
                        # Store metadata as separate tensors for safetensors compatibility
                        output_tensors[f"{name}.meta.orig_K"] = np.array(
                            [meta["orig_K"]], dtype=np.int64
                        )
                        output_tensors[f"{name}.meta.orig_N"] = np.array(
                            [meta["orig_N"]], dtype=np.int64
                        )
                    elif quant_type == "int4":
                        packed, scales, zeros = quantize_to_int4(weight, group_size)
                        output_tensors[f"{name}.packed"] = packed
                        output_tensors[f"{name}.scales"] = scales
                        output_tensors[f"{name}.zeros"] = zeros
                    else:
                        raise ValueError(f"Unknown quant_type: {quant_type}")

                    quant_bytes = packed.nbytes + scales.nbytes
                    stats["quantized_bytes"] += quant_bytes
                    stats["num_quantized"] += 1

                    if verbose:
                        ratio = orig_bytes / quant_bytes if quant_bytes > 0 else 0
                        print(f"    {name}: {tensor.shape} -> {packed.shape} ({ratio:.1f}x)")
                else:
                    # Keep as-is
                    output_tensors[name] = tensor
            else:
                # Keep non-weight tensors as-is
                output_tensors[name] = tensor

    # Save quantized file
    save_file(output_tensors, str(output_path))
    return stats


def quantize_model(
    model_dir: Path,
    output_dir: Path | None = None,
    quant_type: str = "fp4",
    group_size: int = 128,
    verbose: bool = True,
    resume: bool = False,
) -> None:
    """Quantize all safetensors files in a model directory.

    Args:
        model_dir: Input model directory
        output_dir: Output directory (default: <model_dir>-<QUANT_TYPE>)
        quant_type: 'fp4' or 'int4'
        group_size: Quantization group size
        verbose: Print detailed output
        resume: Skip already-quantized files
    """

    if output_dir is None:
        output_dir = model_dir.parent / f"{model_dir.name}-{quant_type.upper()}"

    output_dir.mkdir(parents=True, exist_ok=True)

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"No safetensors files found in {model_dir}")
        return

    # Check which files need processing
    files_to_process = []
    skipped_files = []
    for f in safetensor_files:
        output_file = output_dir / f.name
        if resume and output_file.exists() and output_file.stat().st_size > 0:
            skipped_files.append(f.name)
        else:
            files_to_process.append(f)

    print(f"Quantizing {model_dir.name} to {quant_type.upper()} (group_size={group_size})")
    print(f"Output: {output_dir}")
    print(f"Found {len(safetensor_files)} safetensors files")
    if resume and skipped_files:
        print(f"Resuming: skipping {len(skipped_files)} already-quantized files")
    if not files_to_process:
        print("All files already quantized!")
        return
    print(f"Processing: {len(files_to_process)} files")
    print()

    total_stats = {"original_bytes": 0, "quantized_bytes": 0, "num_quantized": 0}
    start_time = perf_counter()

    for i, input_file in enumerate(files_to_process, 1):
        output_file = output_dir / input_file.name
        print(f"[{i}/{len(files_to_process)}] {input_file.name}")

        file_start = perf_counter()
        stats = quantize_safetensors_file(input_file, output_file, quant_type, group_size, verbose)
        file_time = perf_counter() - file_start

        for k, v in stats.items():
            total_stats[k] += v

        print(f"    Done in {file_time:.1f}s")
        print()

    # Copy non-safetensors files (config, tokenizer, etc.)
    for f in model_dir.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            import shutil

            shutil.copy2(f, output_dir / f.name)

    total_time = perf_counter() - start_time

    orig_gb = total_stats["original_bytes"] / (1024**3)
    quant_gb = total_stats["quantized_bytes"] / (1024**3)
    ratio = (
        total_stats["original_bytes"] / total_stats["quantized_bytes"]
        if total_stats["quantized_bytes"] > 0
        else 0
    )

    print("=" * 60)
    print("Quantization complete!")
    print(f"  Weights quantized: {total_stats['num_quantized']}")
    print(f"  Original size:     {orig_gb:.2f} GB")
    print(f"  Quantized size:    {quant_gb:.2f} GB")
    print(f"  Compression:       {ratio:.1f}x")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Output:            {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize safetensors models to Marlin FP4/INT4 format"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model directory containing safetensors files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: <model_dir>-<QUANT_TYPE>)",
    )
    parser.add_argument(
        "--quant-type",
        "-q",
        choices=["fp4", "int4"],
        default="fp4",
        help="Quantization type (default: fp4)",
    )
    parser.add_argument(
        "--group-size",
        "-g",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Skip files that already exist in output directory",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    quantize_model(
        model_dir=args.model_dir,
        output_dir=args.output,
        quant_type=args.quant_type,
        group_size=args.group_size,
        verbose=not args.quiet,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
