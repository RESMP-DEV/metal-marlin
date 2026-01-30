#!/usr/bin/env python3
"""Subprocess-based quantization coordinator with resume support.

Spawns a fresh Python process for EACH layer to ensure memory is fully released.
This is the only reliable way to prevent memory accumulation in numpy/PyTorch.

Usage:
    # Start fresh
    uv run python scripts/quantize_subprocess.py --model zai-org/GLM-4.7-Flash --output /tmp/glm47_quant

    # Resume from crash
    uv run python scripts/quantize_subprocess.py --model zai-org/GLM-4.7-Flash --output /tmp/glm47_quant --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def get_completed_layers(output_path: Path) -> set[int]:
    """Check which layers have been successfully quantized."""
    completed = set()
    if not output_path.exists():
        return completed

    for layer_dir in output_path.glob("layer_*"):
        if not layer_dir.is_dir():
            continue
        index_file = layer_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    data = json.load(f)
                    if data.get("total_tensors", 0) > 0:
                        layer_idx = int(layer_dir.name.split("_")[1])
                        completed.add(layer_idx)
            except Exception:
                pass
    return completed


def get_num_layers(model_id: str) -> int:
    """Get number of layers from model config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(config, "num_hidden_layers", 32)


def quantize_single_layer(
    model_id: str,
    layer_idx: int,
    output_path: Path,
    expert_workers: int = 12,
    calibration_samples: int = 64,
) -> bool:
    """Spawn subprocess to quantize a single layer.

    Returns True if successful, False if failed.
    """
    # Build command - use the worker script
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "quantize_single_layer.py"),
        "--model",
        model_id,
        "--layer-idx",
        str(layer_idx),
        "--output",
        str(output_path),
        "--expert-workers",
        str(expert_workers),
        "--calibration-samples",
        str(calibration_samples),
    ]

    print(f"\n{'=' * 60}")
    print(f"LAYER {layer_idx} - Spawning subprocess")
    print(f"{'=' * 60}")

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output go to terminal
            timeout=600,  # 10 minute timeout per layer
        )
        elapsed = time.perf_counter() - start
        print(f"Layer {layer_idx} completed in {elapsed:.1f}s")
        return True
    except subprocess.TimeoutExpired:
        print(f"Layer {layer_idx} TIMEOUT after 600s")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Layer {layer_idx} FAILED: return code {e.returncode}")
        return False
    except Exception as e:
        print(f"Layer {layer_idx} ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Subprocess-based quantization with resume support"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--expert-workers", type=int, default=12)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previously completed layers"
    )
    parser.add_argument("--start-layer", type=int, default=0, help="Skip layers before this index")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get model info
    print(f"Loading model config: {args.model}")
    num_layers = get_num_layers(args.model)
    if args.max_layers:
        num_layers = min(num_layers, args.max_layers)
    print(f"Total layers: {num_layers}")

    # Check for resume
    if args.resume:
        completed = get_completed_layers(output_path)
        print(f"Already completed: {sorted(completed)}")
    else:
        completed = set()

    # Process layers
    total_start = time.perf_counter()
    failed_layers = []

    for layer_idx in range(args.start_layer, num_layers):
        if layer_idx in completed:
            print(f"Skipping layer {layer_idx} (already done)")
            continue

        success = quantize_single_layer(
            model_id=args.model,
            layer_idx=layer_idx,
            output_path=output_path,
            expert_workers=args.expert_workers,
            calibration_samples=args.calibration_samples,
        )

        if not success:
            failed_layers.append(layer_idx)
            print(f"\nWARNING: Layer {layer_idx} failed. Continuing with next layer...")
            # Could add retry logic here

    total_time = time.perf_counter() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("QUANTIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Layers processed: {num_layers - len(completed) - len(failed_layers)}")
    if failed_layers:
        print(f"Failed layers: {failed_layers}")
        print("To retry failed layers, run with --resume")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
