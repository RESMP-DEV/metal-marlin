#!/usr/bin/env python3
"""Single-layer quantization worker - spawned by quantize_subprocess.py.

This script quantizes exactly ONE layer and exits, ensuring all memory is released.
Do NOT import this as a module - it's designed to be run as a subprocess.

Memory strategy:
- Load only ONE layer's weights at a time
- Compute Hessians → quantize → save → delete, repeat for each tensor
- Exit after layer is done (OS reclaims all memory)
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Quantize single layer")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--expert-workers", type=int, default=12)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    layer_idx = args.layer_idx
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[Layer {layer_idx}] Starting quantization")
    layer_start = time.perf_counter()

    # Add parent dir to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from huggingface_hub import snapshot_download

    from metal_marlin.calibration import CalibrationDataset
    from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_hessian_compute
    from metal_marlin.quantization.exl3_quantizer import EXL3Quantizer

    # Load model config
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_path = Path(snapshot_download(args.model))

    # Load weight map
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f).get("weight_map", {})

    hidden_size = getattr(config, "hidden_size", 2048)

    # Load calibration
    print(f"[Layer {layer_idx}] Loading calibration...")
    calib_data = CalibrationDataset.v3(max_samples=args.calibration_samples)
    calib_tokens = []
    for sample in calib_data.samples:
        tokens = tokenizer.encode(
            sample,
            add_special_tokens=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        if isinstance(tokens, torch.Tensor):
            calib_tokens.append(tokens.squeeze(0))
        else:
            calib_tokens.append(torch.tensor(tokens))

    # Initialize Metal
    print(f"[Layer {layer_idx}] Initializing Metal...")
    metal_lib = MetalKernelLibrary.from_source_dir()

    # Create quantizers for each bit width
    quantizers = {}
    for bits in range(2, 9):
        quantizers[bits] = EXL3Quantizer(
            bits=bits, group_size=args.group_size, max_workers=1, use_metal=False
        )

    # Find tensors for this layer
    prefix = f"model.layers.{layer_idx}."
    layer_tensors = {}
    shards_needed: dict[str, list[str]] = {}

    for tensor_name, shard_file in weight_map.items():
        if tensor_name.startswith(prefix) and tensor_name.endswith(".weight"):
            layer_tensors[tensor_name] = shard_file
            if shard_file not in shards_needed:
                shards_needed[shard_file] = []
            shards_needed[shard_file].append(tensor_name)

    print(f"[Layer {layer_idx}] Found {len(layer_tensors)} tensors in {len(shards_needed)} shards")

    # Sensitivity patterns
    SENSITIVE_PATTERNS = {"router": 8, "gate": 6, "embed": 8, "lm_head": 8, "norm": 8}

    def determine_bits(name: str, weight: torch.Tensor, H: np.ndarray) -> int:
        name_lower = name.lower()
        for pattern, bits in SENSITIVE_PATTERNS.items():
            if pattern in name_lower:
                return bits

        # Compute sensitivity
        w_abs = np.abs(weight.numpy())
        p99, p50 = np.percentile(w_abs, 99), np.percentile(w_abs, 50)
        outlier_ratio = np.log1p(min(p99 / (p50 + 1e-8), 100)) / np.log1p(100)

        try:
            H_trace = np.trace(H)
            H_frob = np.linalg.norm(H, "fro")
            condition_proxy = min(H_trace / (H_frob + 1e-8), 10) / 10
        except Exception:
            condition_proxy = 0.5

        sensitivity = outlier_ratio * 2 + condition_proxy * 1.5

        is_attention = any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "o_proj"])
        is_expert = "expert" in name_lower

        if sensitivity > 7.5:
            bits = 8
        elif sensitivity > 6.0:
            bits = 6
        elif sensitivity > 4.5:
            bits = 5
        elif sensitivity > 3.0:
            bits = 4
        elif sensitivity > 1.5:
            bits = 3
        else:
            bits = 2

        if is_attention:
            bits = max(bits, 4)
        elif is_expert:
            bits = max(bits, 2)

        return max(2, min(8, bits))

    # Generate activations for this layer
    print(f"[Layer {layer_idx}] Generating activations...")
    torch.manual_seed(42 + layer_idx)
    total_tokens = args.calibration_samples * 512

    # Use calibration diversity to shape activations
    unique_tokens = set()
    for t in calib_tokens[: args.calibration_samples]:
        unique_tokens.update(t.tolist())
    diversity = len(unique_tokens) / max(
        1, sum(len(t) for t in calib_tokens[: args.calibration_samples])
    )

    num_layers = getattr(config, "num_hidden_layers", 32)
    depth_factor = 1.0 - 0.3 * (layer_idx / max(num_layers - 1, 1))
    variance = 0.5 * diversity + 0.3 * depth_factor

    X = torch.randn(total_tokens, hidden_size, device="mps") * np.sqrt(variance)

    # Process tensors in batches, save each batch immediately
    batch_size = 12
    batch_num = 0
    temp_dir = output_path / f".tmp_layer_{layer_idx:04d}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    total_bytes = 0
    bit_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 0}

    tensor_names = list(layer_tensors.keys())
    print(f"[Layer {layer_idx}] Processing {len(tensor_names)} tensors...")

    # Load weights from shards one at a time
    for shard_file, tensor_names_in_shard in shards_needed.items():
        shard_path = model_path / shard_file
        with safe_open(str(shard_path), framework="pt") as f:
            available = set(f.keys())
            for tensor_name in tensor_names_in_shard:
                if tensor_name not in available:
                    continue

                # Load weight
                weight = f.get_tensor(tensor_name).float().cpu().clone()
                if weight.dim() != 2:
                    continue

                # Compute Hessian
                out_feat, in_feat = weight.shape
                if X.shape[1] != in_feat:
                    X_adj = torch.randn(X.shape[0], in_feat, device="mps") * 0.5
                else:
                    X_adj = X

                H = dispatch_hessian_compute(metal_lib, X_adj, sigma_reg=0.01)
                torch.mps.synchronize()
                H_np = H.cpu().numpy().astype(np.float64)

                del H
                if X_adj is not X:
                    del X_adj
                torch.mps.empty_cache()

                # Determine bits
                bits = determine_bits(tensor_name, weight, H_np)
                bit_counts[bits] = bit_counts.get(bits, 0) + 1

                # Quantize
                quantizer = quantizers[bits]
                short_name = tensor_name.split(".")[-2]
                result = quantizer.quantize_layer(weight, H_np, layer_name=short_name)

                # Prepare for saving (pack indices to save ~5x space)
                safe_key = tensor_name.replace(".", "__")
                packed_indices = pack_indices_vectorized(result.trellis_indices, bits)
                batch_tensors = {
                    f"{safe_key}__indices": packed_indices,
                    f"{safe_key}__scales": result.scales.astype(np.float32),
                    f"{safe_key}__su": result.su.astype(np.float32),
                    f"{safe_key}__sv": result.sv.astype(np.float32),
                }

                bytes_this = sum(arr.nbytes for arr in batch_tensors.values())
                total_bytes += bytes_this

                all_metadata.append(
                    {
                        "name": tensor_name,
                        "bits": bits,
                        "shape": list(weight.shape),
                        "mse": result.reconstruction_mse,
                    }
                )

                # Save immediately (one tensor per file for minimal memory)
                tensor_file = temp_dir / f"tensor_{len(all_metadata):04d}.safetensors"
                save_file(batch_tensors, str(tensor_file))

                # Clear everything
                del weight, H_np, result, batch_tensors
                gc.collect()

                # Progress
                if len(all_metadata) % 12 == 0:
                    bits_str = " ".join(f"{b}b:{c}" for b, c in sorted(bit_counts.items()) if c > 0)
                    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
                    print(
                        f"[Layer {layer_idx}] {len(all_metadata)}/{len(tensor_names)} "
                        f"[{bits_str}] Memory: {mem_mb:.0f} MB"
                    )

    # Free GPU memory
    del X
    torch.mps.empty_cache()
    gc.collect()

    # Finalize: rename temp dir to permanent location
    final_dir = output_path / f"layer_{layer_idx:04d}"
    if final_dir.exists():
        import shutil

        shutil.rmtree(final_dir)
    temp_dir.rename(final_dir)

    # Save index
    index_file = final_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(
            {
                "format": "trellis_v2",
                "packing": {
                    "indices_format": "packed_uint8",
                    "header_byte": True,
                },
                "layer_idx": layer_idx,
                "total_tensors": len(all_metadata),
                "total_bytes": total_bytes,
                "tensors": all_metadata,
            },
            f,
            indent=2,
        )

    elapsed = time.perf_counter() - layer_start
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    print(
        f"[Layer {layer_idx}] DONE: {len(all_metadata)} tensors, "
        f"{total_bytes / 1024 / 1024:.1f} MB, {elapsed:.1f}s, peak {mem_mb:.0f} MB"
    )


if __name__ == "__main__":
    main()
