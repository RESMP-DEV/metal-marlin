#!/usr/bin/env python3
"""
Quantize a HuggingFace model to Marlin FP4 format and verify accuracy.

End-to-end workflow:
  1. Download model weights from HuggingFace (safetensors format)
  2. Quantize all linear layers to FP4 (E2M1) with per-group scales
  3. Save the quantized model as .marlin.safetensors
  4. Reload from disk and verify numerical accuracy vs. the original

Usage:
    python quantize_hf_model.py                              # default: microsoft/phi-2
    python quantize_hf_model.py --model microsoft/phi-2
    python quantize_hf_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python quantize_hf_model.py --group-size 64 --output ./phi2_fp4.marlin.safetensors

Requirements:
    pip install torch safetensors transformers huggingface_hub numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def download_safetensors(model_id: str, cache_dir: Path | None = None) -> list[Path]:
    """Download safetensors files for a HuggingFace model.

    Returns list of local paths to the downloaded .safetensors files.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    local_path = Path(local_dir)
    safetensor_files = sorted(local_path.glob("*.safetensors"))
    if not safetensor_files:
        print(f"ERROR: No .safetensors files found in {local_path}", file=sys.stderr)
        sys.exit(1)
    return safetensor_files


def quantize_safetensors_to_marlin(
    input_paths: list[Path],
    output_path: Path,
    group_size: int = 128,
    skip_patterns: set[str] | None = None,
) -> dict[str, int]:
    """Quantize safetensors model files to Marlin FP4 format.

    Processes each tensor:
      - 2D weight matrices: quantized to FP4 with per-group scales
      - Everything else (embeddings, norms, biases): preserved as-is

    Args:
        input_paths: One or more .safetensors files comprising the model.
        output_path: Where to write the .marlin.safetensors output.
        group_size: Elements per quantization group along the K dimension.
        skip_patterns: Layer name patterns to skip quantization for
            (e.g., {"lm_head", "embed_tokens"}).

    Returns:
        Stats dict with counts of quantized/skipped/passthrough tensors.
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    from metal_marlin.quantize import pack_fp4_weights

    skip_patterns = skip_patterns or {"lm_head", "embed_tokens", "wte", "wpe"}

    output_tensors: dict[str, np.ndarray] = {}
    stats = {"quantized": 0, "skipped": 0, "passthrough": 0}

    for sf_path in input_paths:
        print(f"  Processing {sf_path.name}...")
        with safe_open(str(sf_path), framework="numpy") as f:
            for name in f.keys():
                tensor_np = f.get_tensor(name)
                tensor = torch.from_numpy(tensor_np)

                # Decide whether to quantize this tensor
                is_weight = "weight" in name and tensor.ndim == 2
                is_skipped = any(pat in name for pat in skip_patterns)

                if is_weight and not is_skipped:
                    K, N = tensor.shape
                    # Verify dimensions are compatible with packing
                    if N % 8 != 0 or K % group_size != 0:
                        # Dimensions not aligned; pass through unquantized
                        output_tensors[name] = tensor_np.astype(np.float16)
                        stats["passthrough"] += 1
                        continue

                    # Quantize: tensor is [K, N] here (safetensors stores row-major)
                    # pack_fp4_weights expects [out_features, in_features] and
                    # transposes internally, so pass tensor.T
                    packed, scales, _meta = pack_fp4_weights(
                        tensor, group_size=group_size, output_backend="numpy"
                    )

                    output_tensors[f"{name}.packed"] = packed
                    output_tensors[f"{name}.scales"] = scales
                    stats["quantized"] += 1
                else:
                    output_tensors[name] = tensor_np
                    if is_skipped:
                        stats["skipped"] += 1
                    else:
                        stats["passthrough"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, str(output_path))
    return stats


def load_marlin_safetensors(
    path: Path,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """Load a .marlin.safetensors file, reconstructing packed weight pairs.

    Returns a dict where:
      - Quantized layers: key -> (packed_weights, scales)
      - Other tensors: key -> torch.Tensor
    """
    from safetensors import safe_open

    raw: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="numpy") as f:
        for name in f.keys():
            raw[name] = torch.from_numpy(f.get_tensor(name))

    result: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    seen_packed: set[str] = set()

    for name in sorted(raw.keys()):
        if name.endswith(".packed"):
            base = name.removesuffix(".packed")
            scales_key = f"{base}.scales"
            if scales_key in raw:
                result[base] = (raw[name], raw[scales_key])
                seen_packed.add(name)
                seen_packed.add(scales_key)
        elif name not in seen_packed:
            result[name] = raw[name]

    return result


def verify_accuracy(
    original_paths: list[Path],
    marlin_path: Path,
    group_size: int = 128,
    num_layers_to_check: int = 5,
) -> dict[str, float]:
    """Verify quantization accuracy by comparing original vs. dequantized weights.

    For a sample of quantized layers, dequantizes the FP4 weights back to FP16
    and computes error metrics against the original FP16 weights.

    Returns:
        Dict with mean/max absolute error and mean relative error across
        the sampled layers.
    """
    from safetensors import safe_open

    from metal_marlin.quantize import unpack_fp4_weights

    # Load the Marlin file
    marlin_tensors = load_marlin_safetensors(marlin_path)

    # Find quantized layers (those stored as tuples)
    quantized_names = [
        name for name, val in marlin_tensors.items()
        if isinstance(val, tuple)
    ]

    if not quantized_names:
        print("  No quantized layers found to verify!")
        return {}

    # Sample a subset for verification
    check_names = quantized_names[:num_layers_to_check]
    print(f"  Checking {len(check_names)} of {len(quantized_names)} quantized layers...")

    # Load original weights for comparison
    originals: dict[str, np.ndarray] = {}
    for sf_path in original_paths:
        with safe_open(str(sf_path), framework="numpy") as f:
            for name in check_names:
                if name in f.keys():
                    originals[name] = f.get_tensor(name)

    errors: list[dict[str, float]] = []

    for name in check_names:
        if name not in originals:
            continue

        original = originals[name].astype(np.float32)
        packed, scales = marlin_tensors[name]

        # Reconstruct meta for unpack
        packed_np = packed.numpy()
        K_padded, N_packed = packed_np.shape
        N_padded = N_packed * 8
        meta = {
            "orig_K": original.shape[0],
            "orig_N": original.shape[1],
            "padded_K": K_padded,
            "padded_N": N_padded,
            "group_size": group_size,
        }

        dequantized = unpack_fp4_weights(packed, scales, meta, output_backend="numpy")
        deq_np = dequantized.astype(np.float32)

        # Compute error metrics
        abs_err = np.abs(original - deq_np)
        rel_err = abs_err / (np.abs(original) + 1e-8)

        layer_errors = {
            "mean_abs_error": float(abs_err.mean()),
            "max_abs_error": float(abs_err.max()),
            "mean_rel_error": float(rel_err.mean()),
        }
        errors.append(layer_errors)

        print(f"    {name}:")
        print(f"      shape: {original.shape}")
        print(f"      mean |error|: {layer_errors['mean_abs_error']:.6f}")
        print(f"      max  |error|: {layer_errors['max_abs_error']:.6f}")
        print(f"      mean  rel err: {layer_errors['mean_rel_error']:.4%}")

    # Aggregate
    agg = {
        "mean_abs_error": float(np.mean([e["mean_abs_error"] for e in errors])),
        "max_abs_error": float(np.max([e["max_abs_error"] for e in errors])),
        "mean_rel_error": float(np.mean([e["mean_rel_error"] for e in errors])),
    }
    return agg


def compute_size_reduction(
    original_paths: list[Path],
    marlin_path: Path,
) -> dict[str, float]:
    """Compare file sizes between original and quantized models."""
    original_bytes = sum(p.stat().st_size for p in original_paths)
    marlin_bytes = marlin_path.stat().st_size

    return {
        "original_mb": original_bytes / (1024 * 1024),
        "quantized_mb": marlin_bytes / (1024 * 1024),
        "compression_ratio": original_bytes / max(marlin_bytes, 1),
        "size_reduction_pct": (1 - marlin_bytes / original_bytes) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace model to Marlin FP4 format"
    )
    parser.add_argument(
        "--model",
        default="microsoft/phi-2",
        help="HuggingFace model ID (default: microsoft/phi-2)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .marlin.safetensors (default: auto from model name)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for downloaded models",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip accuracy verification step",
    )
    parser.add_argument(
        "--verify-layers",
        type=int,
        default=5,
        help="Number of layers to verify accuracy on (default: 5)",
    )
    args = parser.parse_args()

    model_id: str = args.model
    group_size: int = args.group_size
    model_slug = model_id.replace("/", "_")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"{model_slug}.marlin.safetensors")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Step 1: Download model
    print(f"\n[1/4] Downloading {model_id} from HuggingFace...")
    t0 = time.perf_counter()
    safetensor_files = download_safetensors(model_id, cache_dir=cache_dir)
    print(f"  Found {len(safetensor_files)} safetensors file(s)")
    for f in safetensor_files:
        print(f"    {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Download completed in {time.perf_counter() - t0:.1f}s")

    # Step 2: Quantize to FP4
    print(f"\n[2/4] Quantizing to FP4 (group_size={group_size})...")
    t0 = time.perf_counter()
    stats = quantize_safetensors_to_marlin(
        safetensor_files, output_path, group_size=group_size
    )
    elapsed = time.perf_counter() - t0
    print(f"  Quantized: {stats['quantized']} layers")
    print(f"  Skipped:   {stats['skipped']} layers (embeddings/lm_head)")
    print(f"  Passthrough: {stats['passthrough']} tensors (norms/biases)")
    print(f"  Completed in {elapsed:.1f}s")

    # Step 3: Save (already done in quantize step)
    print(f"\n[3/4] Saved to {output_path}")
    sizes = compute_size_reduction(safetensor_files, output_path)
    print(f"  Original:   {sizes['original_mb']:.1f} MB")
    print(f"  Quantized:  {sizes['quantized_mb']:.1f} MB")
    print(f"  Compression: {sizes['compression_ratio']:.2f}x")
    print(f"  Size reduction: {sizes['size_reduction_pct']:.1f}%")

    # Step 4: Reload and verify accuracy
    if not args.skip_verify:
        print(f"\n[4/4] Verifying accuracy (checking {args.verify_layers} layers)...")
        t0 = time.perf_counter()
        agg_errors = verify_accuracy(
            safetensor_files,
            output_path,
            group_size=group_size,
            num_layers_to_check=args.verify_layers,
        )
        elapsed = time.perf_counter() - t0

        if agg_errors:
            print("\n  Aggregate error metrics:")
            print(f"    Mean |error|:   {agg_errors['mean_abs_error']:.6f}")
            print(f"    Max  |error|:   {agg_errors['max_abs_error']:.6f}")
            print(f"    Mean rel error: {agg_errors['mean_rel_error']:.4%}")
            print(f"  Verification completed in {elapsed:.1f}s")

            # FP4 E2M1 has only 16 representable values; expect ~10-20% relative error
            if agg_errors["mean_rel_error"] < 0.5:
                print("\n  PASS: Quantization error within expected FP4 bounds.")
            else:
                print("\n  WARNING: Quantization error higher than expected.")
                print("  This may indicate a packing/unpacking bug.")
    else:
        print("\n[4/4] Skipping accuracy verification (--skip-verify)")

    print(f"\nDone. Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
