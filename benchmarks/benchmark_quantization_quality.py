#!/usr/bin/env python3
"""
Quantization Quality Benchmark - Pure NumPy (no MLX required)

Measures quantization error (MSE, RMSE, max error) across all FP4 models
by loading original BF16 weights and comparing against dequantized FP4.

Usage:
    cd metal_marlin
    uv run python scripts/benchmark_quantization_quality.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin.quantize_fp4 import quantize_fp4, unpack_fp4


def benchmark_single_tensor(
    original: np.ndarray,
    group_size: int = 128,
) -> dict[str, float]:
    """Benchmark quantization quality for a single tensor."""
    # Ensure 2D and float32
    if original.ndim == 1:
        original = original.reshape(1, -1)

    # Convert to float32
    if str(original.dtype).startswith(">"):  # big-endian (bfloat16)
        original_u16 = original.view(np.uint16).astype(np.uint32) << 16
        original_f32 = original_u16.view(np.float32)
    else:
        original_f32 = original.astype(np.float32)

    # Ensure dimensions are compatible with group_size
    out_feat, in_feat = original_f32.shape
    if in_feat % 8 != 0 or in_feat % group_size != 0:
        # Pad to next multiple
        new_in = ((in_feat + group_size - 1) // group_size) * group_size
        original_padded = np.zeros((out_feat, new_in), dtype=np.float32)
        original_padded[:, :in_feat] = original_f32
        original_f32 = original_padded

    # Quantize and dequantize
    packed, scales = quantize_fp4(original_f32, group_size=group_size)
    reconstructed = unpack_fp4(packed, scales, group_size=group_size)

    # Crop back to original size if padded
    if in_feat % group_size != 0:
        reconstructed = reconstructed[:, :in_feat]
        original_f32 = original_f32[:, :in_feat]

    # Compute metrics
    error = original_f32 - reconstructed.astype(np.float32)
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(np.abs(error)))
    mean_abs_error = float(np.mean(np.abs(error)))

    # Relative error (avoiding division by zero)
    abs_original = np.abs(original_f32)
    mask = abs_original > 1e-6
    if np.any(mask):
        rel_error = np.abs(error[mask]) / abs_original[mask]
        mean_rel_error = float(np.mean(rel_error))
        max_rel_error = float(np.max(rel_error))
    else:
        mean_rel_error = float("nan")
        max_rel_error = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "max_error": max_error,
        "mean_abs_error": mean_abs_error,
        "mean_rel_error": mean_rel_error,
        "max_rel_error": max_rel_error,
    }


def load_safetensors_header(path: Path) -> dict:
    """Load safetensors header to get tensor metadata."""
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header_bytes = f.read(header_size)
        return json.loads(header_bytes.decode("utf-8"))


def load_tensor_from_safetensors(path: Path, tensor_name: str) -> np.ndarray | None:
    """Load a single tensor from safetensors file."""
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))

        if tensor_name not in header:
            return None

        info = header[tensor_name]
        dtype_str = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]

        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.dtype(">f2"),  # bfloat16 as big-endian float16
            "I32": np.int32,
            "I64": np.int64,
        }
        dtype = dtype_map.get(dtype_str, np.float32)

        f.seek(8 + header_size + start)
        data = np.frombuffer(f.read(end - start), dtype=dtype)

        if dtype_str == "BF16":
            # Properly convert bfloat16
            data_u16 = data.view(np.uint16).astype(np.uint32) << 16
            data = data_u16.view(np.float32).astype(np.float16)

        return data.reshape(shape)


def benchmark_model(
    source_dir: Path,
    fp4_dir: Path,
    num_samples: int = 10,
    group_size: int = 128,
) -> dict:
    """Benchmark quantization quality for a model."""
    print(f"\nBenchmarking: {source_dir.name}")
    print(f"  FP4 output: {fp4_dir.name}")

    # Find safetensors files
    source_files = sorted(source_dir.glob("model*.safetensors"))
    if not source_files:
        source_files = sorted(source_dir.glob("*.safetensors"))

    if not source_files:
        print("  ERROR: No safetensors files found")
        return {}

    # Sample tensors from first and last shard
    all_metrics: list[dict] = []
    total_tensors = 0
    sampled_tensors = 0

    for shard_path in [source_files[0], source_files[-1]]:
        header = load_safetensors_header(shard_path)
        # Filter for weight tensors (skip embeddings, norms)
        weight_names = [
            k
            for k in header.keys()
            if k != "__metadata__"
            and ("weight" in k or "proj" in k)
            and header[k].get("dtype") in ("F16", "BF16", "F32")
            and len(header[k].get("shape", [])) == 2  # 2D weight matrices
        ]
        total_tensors += len(weight_names)

        # Sample up to num_samples/2 tensors per shard
        samples_per_shard = min(len(weight_names), num_samples // 2)
        rng = np.random.default_rng(42)
        sampled_names = rng.choice(weight_names, size=samples_per_shard, replace=False)

        for name in sampled_names:
            tensor = load_tensor_from_safetensors(shard_path, name)
            if tensor is None or tensor.size < 256:  # Skip tiny tensors
                continue

            metrics = benchmark_single_tensor(tensor, group_size=group_size)
            metrics["tensor_name"] = name
            metrics["shape"] = list(tensor.shape)
            metrics["elements"] = int(tensor.size)
            all_metrics.append(metrics)
            sampled_tensors += 1

    # Aggregate metrics
    if not all_metrics:
        print("  ERROR: No valid tensors found")
        return {}

    avg_mse = np.mean([m["mse"] for m in all_metrics])
    avg_rmse = np.mean([m["rmse"] for m in all_metrics])
    avg_max_error = np.mean([m["max_error"] for m in all_metrics])
    avg_mean_rel_error = np.nanmean([m["mean_rel_error"] for m in all_metrics])
    max_max_error = np.max([m["max_error"] for m in all_metrics])

    print(f"  Sampled: {sampled_tensors}/{total_tensors} weight tensors")
    print(f"  Avg MSE:        {avg_mse:.6f}")
    print(f"  Avg RMSE:       {avg_rmse:.6f}")
    print(f"  Avg Max Error:  {avg_max_error:.4f}")
    print(f"  Max Max Error:  {max_max_error:.4f}")
    print(f"  Avg Rel Error:  {avg_mean_rel_error:.4%}")

    return {
        "model": source_dir.name,
        "fp4_model": fp4_dir.name,
        "sampled_tensors": sampled_tensors,
        "total_weight_tensors": total_tensors,
        "avg_mse": float(avg_mse),
        "avg_rmse": float(avg_rmse),
        "avg_max_error": float(avg_max_error),
        "max_max_error": float(max_max_error),
        "avg_mean_rel_error": float(avg_mean_rel_error),
        "metrics": all_metrics,
    }


def main():
    models_dir = _ROOT / "models"

    # Model pairs: (source_name, fp4_name)
    model_pairs = [
        ("Qwen3-4B", "Qwen3-4B-FP4"),
        ("GLM-4.7-Flash", "GLM-4.7-Flash-FP4"),
        ("Qwen3-32B", "Qwen3-32B-FP4"),
        ("Qwen3-30B-A3B", "Qwen3-30B-A3B-FP4"),
        ("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-FP4"),
    ]

    print("=" * 60)
    print("FP4 Quantization Quality Benchmark")
    print("=" * 60)
    print("Group size: 128")
    print("Quantization: FP4 E2M1 (NVFP4-style)")

    results: list[dict] = []
    start_time = time.perf_counter()

    for source_name, fp4_name in model_pairs:
        source_dir = models_dir / source_name
        fp4_dir = models_dir / fp4_name

        if not source_dir.exists():
            print(f"\nSkipping {source_name} (not found)")
            continue
        if not fp4_dir.exists():
            print(f"\nSkipping {source_name} (FP4 not found)")
            continue

        result = benchmark_model(source_dir, fp4_dir, num_samples=20)
        if result:
            results.append(result)

    elapsed = time.perf_counter() - start_time

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<40} {'MSE':>10} {'RMSE':>10} {'RelErr':>10}")
    print("-" * 72)

    for r in results:
        model = r["model"][:38]
        print(
            f"{model:<40} {r['avg_mse']:>10.6f} {r['avg_rmse']:>10.6f} {r['avg_mean_rel_error']:>9.2%}"
        )

    print("-" * 72)
    print(f"Total time: {elapsed:.1f}s")

    # Quality assessment
    avg_all_mse = np.mean([r["avg_mse"] for r in results])
    print(f"\nOverall Avg MSE: {avg_all_mse:.6f}")
    if avg_all_mse < 0.01:
        print("Quality: EXCELLENT - negligible quantization error")
    elif avg_all_mse < 0.05:
        print("Quality: GOOD - minor quantization error")
    elif avg_all_mse < 0.1:
        print("Quality: ACCEPTABLE - noticeable but usable")
    else:
        print("Quality: POOR - significant quantization error")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
