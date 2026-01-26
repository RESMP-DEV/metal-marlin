#!/usr/bin/env python3
"""Comprehensive FP4 quantization quality benchmark across all models."""

import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file

_ROOT = Path(__file__).parent.parent
MODELS_DIR = _ROOT / "models"
sys.path.insert(0, str(_ROOT))

from metal_marlin.quantize_fp4 import unpack_fp4


def test_model(model_name: str, max_layers: int = 10) -> dict:
    """Test a single model's quantization quality."""
    orig_dir = MODELS_DIR / model_name
    fp4_dir = MODELS_DIR / f"{model_name}-FP4"

    if not fp4_dir.exists():
        return {"model": model_name, "status": "not_found"}

    # Find safetensor files
    orig_files = sorted(orig_dir.glob("model-*.safetensors"))
    fp4_files = sorted(fp4_dir.glob("model-*.safetensors"))

    if not orig_files or not fp4_files:
        return {"model": model_name, "status": "no_files"}

    results = []
    layers_tested = 0

    for orig_file, fp4_file in zip(orig_files, fp4_files):
        # Load original with torch (handles bfloat16)
        orig_tensors = load_file(str(orig_file))

        with safe_open(str(fp4_file), framework="numpy") as f:
            keys = [k for k in f.keys() if k.endswith(".packed")]

            for packed_key in keys:
                if layers_tested >= max_layers:
                    break

                base_name = packed_key[:-7]  # Remove ".packed"

                if base_name not in orig_tensors:
                    continue

                # Get tensors
                orig = orig_tensors[base_name].float().numpy()
                packed = f.get_tensor(packed_key)
                scales = f.get_tensor(base_name + ".scales")

                # Reconstruct
                try:
                    recon = unpack_fp4(packed, scales, group_size=128).astype(np.float32)

                    # Verify shapes match
                    if orig.shape != recon.shape:
                        print(
                            f"  Shape mismatch {base_name}: orig {orig.shape} vs recon {recon.shape}"
                        )
                        continue
                except Exception as e:
                    print(f"  Error unpacking {base_name}: {e}")
                    continue

                # Compute metrics
                mse = float(np.mean((orig - recon) ** 2))
                cos = float(
                    np.dot(orig.flatten(), recon.flatten())
                    / (np.linalg.norm(orig) * np.linalg.norm(recon))
                )

                results.append(
                    {
                        "name": base_name,
                        "mse": mse,
                        "cos": cos,
                        "shape": orig.shape,
                    }
                )
                layers_tested += 1

        if layers_tested >= max_layers:
            break

    if not results:
        return {"model": model_name, "status": "no_results"}

    avg_mse = np.mean([r["mse"] for r in results])
    avg_cos = np.mean([r["cos"] for r in results])
    min_cos = min(r["cos"] for r in results)
    max_cos = max(r["cos"] for r in results)

    return {
        "model": model_name,
        "status": "ok",
        "layers_tested": len(results),
        "avg_mse": avg_mse,
        "avg_cos": avg_cos,
        "min_cos": min_cos,
        "max_cos": max_cos,
    }


def main():
    print("=" * 70)
    print("FP4 Quantization Quality Benchmark - Real Model Weights")
    print("=" * 70)

    models = [
        "Qwen3-4B",
        "GLM-4.7-Flash",
        "Qwen3-32B",
        "Qwen3-30B-A3B",
        "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ]

    results = []

    for model in models:
        print(f"\nTesting {model}...", end=" ", flush=True)
        start = time.time()
        result = test_model(model, max_layers=15)
        elapsed = time.time() - start

        if result["status"] == "ok":
            print(f"done ({elapsed:.1f}s)")
            results.append(result)
        else:
            print(f"skipped: {result['status']}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Layers':>7} {'Avg MSE':>12} {'Avg Cos':>10} {'Min Cos':>10}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['model']:<35} {r['layers_tested']:>7} {r['avg_mse']:>12.8f} {r['avg_cos']:>10.6f} {r['min_cos']:>10.6f}"
        )

    # Overall statistics
    if results:
        overall_cos = np.mean([r["avg_cos"] for r in results])
        overall_mse = np.mean([r["avg_mse"] for r in results])

        print("-" * 70)
        print(
            f"{'OVERALL':<35} {sum(r['layers_tested'] for r in results):>7} {overall_mse:>12.8f} {overall_cos:>10.6f}"
        )

        # Quality assessment
        print("\n" + "=" * 70)
        if overall_cos >= 0.99:
            print("QUALITY ASSESSMENT: EXCELLENT (>99% cosine similarity)")
            print("FP4 quantization preserves model quality very well.")
        elif overall_cos >= 0.98:
            print("QUALITY ASSESSMENT: GOOD (>98% cosine similarity)")
            print("FP4 quantization has acceptable quality loss.")
        else:
            print("QUALITY ASSESSMENT: ACCEPTABLE")
            print("FP4 quantization has noticeable quality loss.")
        print("=" * 70)


if __name__ == "__main__":
    main()
