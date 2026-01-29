#!/usr/bin/env python3
"""Parallelized layer-by-layer EXL3 quantization with prefetching.

Optimizations over quantize_layerwise_metal.py:
1. Prefetch next layer weights while quantizing current (uses idle RAM)
2. Parallel expert quantization within MoE layers (ThreadPoolExecutor)
3. True mixed-precision: actually quantize at recommended bits, not uniform
4. Special handling for routing/embedding layers (higher bits)

Usage:
    cd contrib/metal_marlin
    uv run python scripts/quantize_layerwise_parallel.py --model zai-org/GLM-4.7-Flash
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_hessian_compute
from metal_marlin.quantization.exl3_quantizer import EXL3Quantizer

# Layers that need higher precision
SENSITIVE_PATTERNS = {
    "router": 8,  # MoE routing - critical for expert selection
    "gate": 6,  # Gating layers
    "embed": 8,  # Embedding layers
    "lm_head": 8,  # Output projection
    "norm": 8,  # Normalization (but we skip these anyway)
}

# Minimum bits for attention vs MLP
ATTENTION_MIN_BITS = 4
EXPERT_MIN_BITS = 2  # Experts can go very low


@dataclass
class LayerResult:
    """Result from quantizing a single tensor."""

    name: str
    shape: tuple[int, ...]
    mse: float
    sensitivity: float
    actual_bits: int  # Bits actually used (not just recommended)
    time_sec: float
    success: bool
    error: str | None = None


@dataclass
class PrefetchedLayer:
    """Prefetched layer data ready for quantization."""

    layer_idx: int
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    shard_handles: dict[str, Any] = field(default_factory=dict)


class WeightPrefetcher:
    """Background thread that prefetches next layer's weights."""

    def __init__(self, model_path: Path, weight_map: dict[str, str], prefetch_ahead: int = 5):
        self.model_path = model_path
        self.weight_map = weight_map
        self.prefetch_ahead = prefetch_ahead
        self.queue: Queue[PrefetchedLayer | None] = Queue(maxsize=prefetch_ahead)
        self.thread: Thread | None = None
        self._stop = False

        # Cache open shard handles to avoid reopening
        self._shard_cache: dict[str, Any] = {}

    def start(self, layer_indices: list[int]):
        """Start prefetching layers in background."""
        self._stop = False
        self.thread = Thread(target=self._prefetch_loop, args=(layer_indices,), daemon=True)
        self.thread.start()

    def stop(self):
        """Stop prefetching."""
        self._stop = True
        if self.thread:
            self.thread.join(timeout=5)

    def get_layer(self, timeout: float = 60.0) -> PrefetchedLayer | None:
        """Get next prefetched layer (blocking)."""
        return self.queue.get(timeout=timeout)

    def _prefetch_loop(self, layer_indices: list[int]):
        """Background loop that prefetches layers."""
        for layer_idx in layer_indices:
            if self._stop:
                break

            layer = self._load_layer(layer_idx)
            self.queue.put(layer)

        # Signal completion
        self.queue.put(None)

    def _load_layer(self, layer_idx: int) -> PrefetchedLayer:
        """Load all tensors for a layer."""
        prefix = f"model.layers.{layer_idx}."
        layer = PrefetchedLayer(layer_idx=layer_idx)

        for tensor_name, shard_file in self.weight_map.items():
            if not tensor_name.startswith(prefix):
                continue
            if not tensor_name.endswith(".weight"):
                continue

            # Open shard if not cached
            shard_path = self.model_path / shard_file
            if shard_file not in self._shard_cache:
                self._shard_cache[shard_file] = safe_open(shard_path, framework="pt")

            handle = self._shard_cache[shard_file]
            tensor = handle.get_tensor(tensor_name).float()

            # Only keep 2D tensors (linear layers)
            if tensor.dim() == 2:
                layer.tensors[tensor_name] = tensor

        return layer


class ParallelLayerwiseQuantizer:
    """High-performance layerwise quantization with parallelization."""

    def __init__(
        self,
        model_id: str,
        min_bits: int = 2,
        max_bits: int = 8,
        group_size: int = 128,
        sigma_reg: float = 0.01,
        expert_workers: int = 48,  # Parallel expert quantization (use available RAM)
    ):
        self.model_id = model_id
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.group_size = group_size
        self.sigma_reg = sigma_reg
        self.expert_workers = expert_workers

        # Initialize Metal kernel library
        print("Initializing Metal kernel library...")
        self.metal_lib = MetalKernelLibrary.from_source_dir()
        print(f"  Loaded {len(self.metal_lib._pipelines)} cached pipelines")

        # Lock for Metal operations (not thread-safe)
        self._metal_lock = Lock()

        # Create quantizers for each bit width - one per worker to avoid contention
        # Using ThreadLocal would be cleaner but this works
        self.quantizers: dict[int, list[EXL3Quantizer]] = {}
        for bits in range(min_bits, max_bits + 1):
            self.quantizers[bits] = [
                EXL3Quantizer(bits=bits, group_size=group_size, max_workers=1)
                for _ in range(expert_workers)
            ]

        # Model info
        self.config: Any = None
        self.tokenizer: Any = None
        self.model_path: Path | None = None
        self.weight_map: dict[str, str] = {}

        # Prefetcher
        self.prefetcher: WeightPrefetcher | None = None

    def initialize(self) -> None:
        """Download model metadata and build layer index."""
        from huggingface_hub import snapshot_download

        print(f"\nInitializing model: {self.model_id}")

        self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        self.model_path = Path(snapshot_download(self.model_id))
        print(f"  Model path: {self.model_path}")

        index_file = self.model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            self.weight_map = index.get("weight_map", {})
            print(f"  Weight map: {len(self.weight_map)} tensors")

        # Model info
        hidden_size = getattr(self.config, "hidden_size", 4096)
        num_layers = getattr(self.config, "num_hidden_layers", 32)
        model_type = getattr(self.config, "model_type", "unknown")

        print("\nModel architecture:")
        print(f"  Type: {model_type}")
        print(f"  Hidden: {hidden_size}, Layers: {num_layers}")

        if hasattr(self.config, "num_experts"):
            print(
                f"  Experts: {self.config.num_experts} ({getattr(self.config, 'num_experts_per_tok', 4)} active)"
            )

    def determine_bits(self, tensor_name: str, weight: torch.Tensor, H: np.ndarray) -> int:
        """Determine optimal bit width based on sensitivity and layer type."""

        # Check for sensitive layer patterns
        name_lower = tensor_name.lower()
        for pattern, bits in SENSITIVE_PATTERNS.items():
            if pattern in name_lower:
                return bits

        # Compute sensitivity
        sensitivity = self._compute_sensitivity(weight.numpy(), H)

        # Check if it's an attention or expert layer
        is_attention = any(
            p in name_lower for p in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"]
        )
        is_expert = "expert" in name_lower

        # Map sensitivity to bits
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

        # Apply minimums based on layer type
        if is_attention:
            bits = max(bits, ATTENTION_MIN_BITS)
        elif is_expert:
            bits = max(bits, EXPERT_MIN_BITS)

        return max(self.min_bits, min(self.max_bits, bits))

    def _compute_sensitivity(self, weight: np.ndarray, H: np.ndarray) -> float:
        """Compute layer sensitivity score."""
        w_abs = np.abs(weight)
        p99 = np.percentile(w_abs, 99)
        p50 = np.percentile(w_abs, 50)

        outlier_ratio = np.log1p(min(p99 / (p50 + 1e-8), 100)) / np.log1p(100)

        try:
            H_trace = np.trace(H)
            H_frob = np.linalg.norm(H, "fro")
            condition_proxy = min(H_trace / (H_frob + 1e-8), 10) / 10
        except:
            condition_proxy = 0.5

        return outlier_ratio * 2 + condition_proxy * 1.5

    def compute_hessian(
        self,
        tensor_name: str,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> tuple[np.ndarray, int, float]:
        """Compute Hessian for a tensor (MUST run on main thread - Metal not thread-safe).

        Returns: (H_np, optimal_bits, sensitivity)
        """
        out_feat, in_feat = weight.shape

        # Adjust activations if needed
        if activations.shape[1] != in_feat:
            activations = torch.randn(activations.shape[0], in_feat, device="mps") * 0.5

        # Compute Hessian on GPU
        H = dispatch_hessian_compute(self.metal_lib, activations, sigma_reg=self.sigma_reg)
        torch.mps.synchronize()
        H_np = H.cpu().numpy().astype(np.float64)

        # Determine optimal bits
        bits = self.determine_bits(tensor_name, weight, H_np)
        sensitivity = self._compute_sensitivity(weight.numpy(), H_np)

        return H_np, bits, sensitivity

    def quantize_tensor_cpu(
        self,
        tensor_name: str,
        weight: torch.Tensor,
        H_np: np.ndarray,
        bits: int,
        sensitivity: float,
        worker_id: int = 0,
    ) -> LayerResult:
        """Quantize a tensor (CPU-only, thread-safe)."""
        start = time.perf_counter()

        try:
            # Quantize at determined bit width (parallel-safe, CPU-bound)
            quantizer = self.quantizers[bits][worker_id % len(self.quantizers[bits])]
            result = quantizer.quantize_layer(
                weight.cpu(),
                H_np,
                layer_name=tensor_name.split(".")[-2],
            )

            elapsed = time.perf_counter() - start

            return LayerResult(
                name=tensor_name,
                shape=tuple(weight.shape),
                mse=result.reconstruction_mse,
                sensitivity=sensitivity,
                actual_bits=bits,
                time_sec=elapsed,
                success=True,
            )

        except Exception as e:
            return LayerResult(
                name=tensor_name,
                shape=tuple(weight.shape) if weight is not None else (0, 0),
                mse=float("inf"),
                sensitivity=0,
                actual_bits=bits,
                time_sec=time.perf_counter() - start,
                success=False,
                error=str(e),
            )

    def quantize_layer_parallel(
        self,
        layer: PrefetchedLayer,
        activations: torch.Tensor,
    ) -> list[LayerResult]:
        """Quantize ALL tensors in parallel (not just experts)."""
        all_tensors = layer.tensors
        total = len(all_tensors)

        print(f"    Processing {total} tensors with {self.expert_workers} workers...")

        results = []
        with ThreadPoolExecutor(max_workers=self.expert_workers) as executor:
            futures = {}
            for idx, (name, weight) in enumerate(all_tensors.items()):
                future = executor.submit(self.quantize_tensor, name, weight, activations, idx)
                futures[future] = name

            done_count = 0
            bit_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 0}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                done_count += 1

                if result.success and result.actual_bits in bit_counts:
                    bit_counts[result.actual_bits] += 1

                # Progress every 10% or at end
                if done_count % max(1, total // 10) == 0 or done_count == total:
                    bits_str = " ".join(f"{b}b:{c}" for b, c in sorted(bit_counts.items()) if c > 0)
                    print(f"      {done_count}/{total} done [{bits_str}]", flush=True)

        return results

    def quantize_model(
        self,
        max_layers: int | None = None,
        calibration_samples: int = 64,
        max_seq_len: int = 512,
    ) -> list[LayerResult]:
        """Quantize entire model with prefetching and parallelization."""
        self.initialize()

        num_layers = getattr(self.config, "num_hidden_layers", 32)
        hidden_size = getattr(self.config, "hidden_size", 2048)

        if max_layers is not None:
            num_layers = min(num_layers, max_layers)

        layer_indices = list(range(num_layers))

        # Start prefetcher - use more layers ahead to saturate RAM
        prefetch = min(5, num_layers)
        print(f"\nStarting prefetcher ({prefetch} layers ahead)...")
        self.prefetcher = WeightPrefetcher(
            self.model_path, self.weight_map, prefetch_ahead=prefetch
        )
        self.prefetcher.start(layer_indices)

        all_results = []
        total_start = time.perf_counter()

        print(f"\n{'=' * 60}")
        print(f"Quantizing {num_layers} layers (mixed-precision, parallel experts)")
        print(f"{'=' * 60}")

        try:
            layer_num = 0
            while True:
                layer = self.prefetcher.get_layer(timeout=120)
                if layer is None:
                    break

                layer_num += 1
                print(
                    f"\n[Layer {layer_num}/{num_layers}] (Layer idx {layer.layer_idx}, {len(layer.tensors)} tensors)"
                )

                # Generate activations
                torch.manual_seed(42 + layer.layer_idx)
                total_tokens = calibration_samples * max_seq_len
                progress = layer.layer_idx / max(num_layers - 1, 1)
                variance = 0.5 + 0.5 * np.sin(progress * np.pi)
                X = torch.randn(total_tokens, hidden_size, device="mps") * np.sqrt(variance)

                # Quantize layer
                layer_start = time.perf_counter()
                results = self.quantize_layer_parallel(layer, X)
                layer_time = time.perf_counter() - layer_start

                all_results.extend(results)

                # Stats
                successful = [r for r in results if r.success]
                if successful:
                    avg_bits = np.mean([r.actual_bits for r in successful])
                    avg_mse = np.mean([r.mse for r in successful])
                    print(
                        f"  Layer summary: {len(successful)} tensors, avg {avg_bits:.1f}b, MSE={avg_mse:.6f}, {layer_time:.1f}s"
                    )

                # Free memory
                del X, layer
                gc.collect()
                torch.mps.empty_cache()

        finally:
            self.prefetcher.stop()

        total_time = time.perf_counter() - total_start
        self._print_summary(all_results, total_time)

        return all_results

    def _print_summary(self, results: list[LayerResult], total_time: float) -> None:
        """Print quantization summary with bit allocation breakdown."""
        from collections import Counter

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n{'=' * 60}")
        print("QUANTIZATION SUMMARY")
        print(f"{'=' * 60}")

        if not successful:
            print("No successful quantizations!")
            return

        # Basic stats
        avg_mse = np.mean([r.mse for r in successful])
        avg_rmse = np.sqrt(avg_mse)

        print(f"\nTensors: {len(successful)} successful, {len(failed)} failed")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"Throughput: {len(successful) / total_time:.1f} tensors/sec")

        # Bit allocation breakdown
        bit_counts = Counter(r.actual_bits for r in successful)
        total_params = sum(r.shape[0] * r.shape[1] for r in successful)

        print("\nMixed-precision bit allocation:")
        print(f"  {'Bits':<6} {'Tensors':<10} {'Params':<15} {'%Params':<10}")
        print(f"  {'-' * 45}")

        weighted_bits = 0
        for bits in sorted(bit_counts.keys()):
            count = bit_counts[bits]
            tensors_at_bits = [r for r in successful if r.actual_bits == bits]
            params_at_bits = sum(r.shape[0] * r.shape[1] for r in tensors_at_bits)
            pct = 100 * params_at_bits / total_params
            weighted_bits += bits * params_at_bits
            print(f"  {bits}b     {count:<10} {params_at_bits:>12,}   {pct:>6.1f}%")

        effective_bits = weighted_bits / total_params
        compression = 16.0 / effective_bits

        print(f"\n  Effective bits/weight: {effective_bits:.2f}b")
        print(f"  Compression vs FP16: {compression:.1f}x")
        print(f"  Total params: {total_params:,}")

        # Most/least compressed
        by_bits = sorted(successful, key=lambda r: r.actual_bits)

        print("\n  High precision (8b):")
        for r in [x for x in by_bits if x.actual_bits == 8][:3]:
            short = ".".join(r.name.split(".")[-3:]).replace(".weight", "")
            print(f"    {short}: sens={r.sensitivity:.2f}")

        print("\n  Low precision (2-3b):")
        for r in [x for x in by_bits if x.actual_bits <= 3][:3]:
            short = ".".join(r.name.split(".")[-3:]).replace(".weight", "")
            print(f"    {short}: {r.actual_bits}b MSE={r.mse:.6f}")

        if failed:
            print(f"\nFailed tensors: {len(failed)}")
            for r in failed[:5]:
                print(f"  {r.name}: {r.error}")


def main():
    parser = argparse.ArgumentParser(description="Parallel mixed-precision EXL3 quantization")
    parser.add_argument("--model", type=str, default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--min-bits", type=int, default=2, help="Minimum bits")
    parser.add_argument("--max-bits", type=int, default=8, help="Maximum bits")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument(
        "--expert-workers", type=int, default=48, help="Parallel workers (use available RAM)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output safetensors path")
    args = parser.parse_args()

    print("=" * 60)
    print("PARALLEL MIXED-PRECISION EXL3 QUANTIZATION")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Bits: {args.min_bits}-{args.max_bits} (mixed)")
    print(f"  Expert workers: {args.expert_workers}")
    print(f"  Calibration: {args.calibration_samples} samples")
    if args.max_layers:
        print(f"  Max layers: {args.max_layers}")

    quantizer = ParallelLayerwiseQuantizer(
        model_id=args.model,
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        group_size=args.group_size,
        expert_workers=args.expert_workers,
    )

    results = quantizer.quantize_model(
        max_layers=args.max_layers,
        calibration_samples=args.calibration_samples,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
