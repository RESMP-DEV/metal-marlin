#!/usr/bin/env python3
"""Parallelized layer-by-layer EXL3 quantization with deep pipelining.

Architecture:
1. Weight Prefetcher: Loads weights for layers N+1, N+2, ... in background
2. Hessian Prefetcher: Computes Hessians for layer N+1 while quantizing layer N
3. Worker threads: CPU trellis quantization (parallel, use_metal=False for thread safety)
4. Double-buffered pipeline: While quantizing layer N, prepare Hessians for layer N+1

Pipeline visualization:
  Layer N:   [Weights] → [Hessians] → [Quantize........]
  Layer N+1:             [Weights] → [Hessians] → [Quantize........]
                                     ↑ overlap    ↑ overlap

Features:
- Bartowski v3 calibration (multi-domain: code, chat, math, reasoning)
- Deep GPU/CPU pipelining for ~30-40% speedup on MoE layers
- Mixed-precision: 2-8 bits based on sensitivity analysis

Performance notes:
- Optimal worker count is ~24 on Apple Silicon (M4 Max tested)
- Higher worker counts (e.g., 48) do NOT improve speed and may be slower
  due to thread contention in numpy LAPACK calls and memory bandwidth limits
- The bottleneck shifts from CPU parallelism to memory/cache at ~24 workers

Usage:
    cd contrib/metal_marlin
    uv run python scripts/quantize_layerwise_parallel.py --model zai-org/GLM-4.7-Flash --expert-workers 24
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from metal_marlin.calibration import CalibrationDataset
from metal_marlin.metal_dispatch import MetalKernelLibrary, dispatch_hessian_compute
from metal_marlin.quantization.exl3_quantizer import EXL3Quantizer
from metal_marlin.trellis_packing import pack_indices_vectorized

# Layers that need higher precision
SENSITIVE_PATTERNS = {
    "router": 8,
    "gate": 6,
    "embed": 8,
    "lm_head": 8,
    "norm": 8,
}

ATTENTION_MIN_BITS = 4
EXPERT_MIN_BITS = 2


@dataclass
class LayerResult:
    """Metadata for a quantized tensor (lightweight, safe to accumulate)."""

    name: str
    shape: tuple[int, ...]
    mse: float
    sensitivity: float
    actual_bits: int
    time_sec: float
    success: bool
    error: str | None = None


@dataclass
class QuantizedTensor:
    """Full quantization result including data to save to disk."""

    name: str
    trellis_indices: np.ndarray  # [tiles_n, tiles_k, 256] int16
    scales: np.ndarray  # [n_groups, out_features] float32
    su: np.ndarray  # Input sign flips
    sv: np.ndarray  # Output sign flips
    bits: int
    shape: tuple[int, ...]
    mse: float


@dataclass
class PrefetchedLayer:
    layer_idx: int
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class HessianData:
    """Pre-computed Hessian data ready for quantization."""

    name: str
    weight: torch.Tensor
    hessian: np.ndarray
    bits: int
    sensitivity: float


@dataclass
class PreparedLayer:
    """Layer with all Hessians pre-computed, ready for quantization."""

    layer_idx: int
    hessian_data: list[HessianData]
    hessian_time: float


class WeightPrefetcher:
    """Load weights layer-by-layer without memory mapping.

    Uses direct tensor loading instead of safetensors mmap to avoid
    keeping file handles open and accumulating mmap'd pages in RSS.
    """

    def __init__(self, model_path: Path, weight_map: dict[str, str], prefetch_ahead: int = 1):
        self.model_path = model_path
        self.weight_map = weight_map
        self.prefetch_ahead = prefetch_ahead
        self.queue: Queue[PrefetchedLayer | None] = Queue(maxsize=prefetch_ahead)
        self.thread: Thread | None = None
        self._stop = False
        # Group tensors by shard file for efficient loading
        self._tensors_by_shard: dict[str, list[str]] = {}
        for tensor_name, shard_file in weight_map.items():
            if shard_file not in self._tensors_by_shard:
                self._tensors_by_shard[shard_file] = []
            self._tensors_by_shard[shard_file].append(tensor_name)

    def start(self, layer_indices: list[int]):
        self._stop = False
        self.thread = Thread(target=self._prefetch_loop, args=(layer_indices,), daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
        if self.thread:
            self.thread.join(timeout=5)

    def get_layer(self, timeout: float = 60.0) -> PrefetchedLayer | None:
        return self.queue.get(timeout=timeout)

    def _prefetch_loop(self, layer_indices: list[int]):
        for layer_idx in layer_indices:
            if self._stop:
                break
            layer = self._load_layer(layer_idx)
            self.queue.put(layer)
        self.queue.put(None)

    def _load_layer(self, layer_idx: int) -> PrefetchedLayer:
        """Load a single layer's weights using selective tensor access.

        Uses safe_open to selectively load only the tensors we need,
        avoiding loading entire multi-GB shard files into memory.
        The file handle is closed after extraction, releasing mmap pages.
        """
        from safetensors import safe_open

        prefix = f"model.layers.{layer_idx}."
        layer = PrefetchedLayer(layer_idx=layer_idx)

        # Find which shards contain tensors for this layer
        shards_needed: dict[str, list[str]] = {}
        for tensor_name, shard_file in self.weight_map.items():
            if not tensor_name.startswith(prefix):
                continue
            if not tensor_name.endswith(".weight"):
                continue
            if shard_file not in shards_needed:
                shards_needed[shard_file] = []
            shards_needed[shard_file].append(tensor_name)

        # Load ONLY needed tensors from each shard (not the whole file)
        for shard_file, tensor_names in shards_needed.items():
            shard_path = self.model_path / shard_file
            # safe_open with context manager: mmap opens, we read specific tensors, mmap closes
            with safe_open(str(shard_path), framework="pt") as f:
                available_keys = set(f.keys())
                for tensor_name in tensor_names:
                    if tensor_name in available_keys:
                        # Get tensor and immediately convert to float32 on CPU
                        tensor = f.get_tensor(tensor_name)
                        if tensor.dim() == 2:
                            # .clone() ensures we own the data, not referencing mmap
                            layer.tensors[tensor_name] = tensor.float().cpu().clone()
                        del tensor
            # File handle closed here - mmap pages can be reclaimed by OS

        gc.collect()
        return layer

    def release_layer(self, layer_idx: int) -> None:
        """No-op - we don't cache anything anymore."""
        pass


class HessianPrefetcher:
    """Prefetch Hessian computations for the next layer while quantizing current.

    This creates a double-buffered pipeline:
    - While CPU workers quantize layer N
    - GPU computes Hessians for layer N+1

    Memory-optimized: Only keeps 1 layer in queue to limit peak memory.
    For MoE layers with 201 tensors, each layer's Hessian data is ~5GB.
    """

    def __init__(
        self,
        quantizer: ParallelLayerwiseQuantizer,
        weight_prefetcher: WeightPrefetcher,
        calibration_samples: int,
        max_seq_len: int,
    ):
        self.quantizer = quantizer
        self.weight_prefetcher = weight_prefetcher
        self.calibration_samples = calibration_samples
        self.max_seq_len = max_seq_len
        # CRITICAL: maxsize=1 to limit memory to one layer's worth of Hessians
        self.queue: Queue[PreparedLayer | None] = Queue(maxsize=1)
        self.thread: Thread | None = None
        self._stop = False
        self._current_layer: PrefetchedLayer | None = None

    def start(self):
        """Start the Hessian prefetcher thread."""
        self._stop = False
        self.thread = Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the prefetcher thread."""
        self._stop = True
        if self.thread:
            self.thread.join(timeout=10)

    def get_prepared_layer(self, timeout: float = 180.0) -> PreparedLayer | None:
        """Get the next layer with Hessians pre-computed."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def _prefetch_loop(self):
        """Background thread that computes Hessians ahead of quantization.

        Memory-optimized: Clears each tensor after computing its Hessian,
        rather than keeping all tensors in memory until the layer is done.
        """
        hidden_size = getattr(self.quantizer.config, "hidden_size", 2048)

        while not self._stop:
            # Get next layer from weight prefetcher
            layer = self.weight_prefetcher.get_layer(timeout=120)
            if layer is None:
                self.queue.put(None)
                break

            # Generate activations for this layer
            X = self.quantizer._generate_activations(
                layer_idx=layer.layer_idx,
                in_features=hidden_size,
                calibration_samples=self.calibration_samples,
                max_seq_len=self.max_seq_len,
            )

            # Compute Hessians one at a time, clearing weights as we go
            hessian_start = time.perf_counter()
            hessian_data: list[HessianData] = []
            tensor_names = list(layer.tensors.keys())

            for name in tensor_names:
                if self._stop:
                    break

                weight = layer.tensors.pop(name)  # Remove from dict immediately

                try:
                    H_np, bits, sensitivity = self.quantizer.compute_hessian(name, weight, X)
                except Exception:
                    # Fallback on error
                    H_np = np.eye(weight.shape[1], dtype=np.float64) * 0.01
                    bits = 4
                    sensitivity = 0.5

                hessian_data.append(
                    HessianData(
                        name=name,
                        weight=weight.cpu(),  # Move to CPU, release GPU memory
                        hessian=H_np,
                        bits=bits,
                        sensitivity=sensitivity,
                    )
                )

                # Clear GPU memory periodically
                if len(hessian_data) % 50 == 0:
                    torch.mps.empty_cache()

            hessian_time = time.perf_counter() - hessian_start

            # Free GPU memory
            del X
            layer.tensors.clear()
            torch.mps.empty_cache()
            gc.collect()

            # Put prepared layer in queue (blocks if queue is full = good backpressure)
            prepared = PreparedLayer(
                layer_idx=layer.layer_idx,
                hessian_data=hessian_data,
                hessian_time=hessian_time,
            )
            self.queue.put(prepared)


class ParallelLayerwiseQuantizer:
    def __init__(
        self,
        model_id: str,
        min_bits: int = 2,
        max_bits: int = 8,
        group_size: int = 128,
        sigma_reg: float = 0.01,
        expert_workers: int = 8,
        calibration_source: str = "bartowski-v3",
    ):
        self.model_id = model_id
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.group_size = group_size
        self.sigma_reg = sigma_reg
        self.expert_workers = expert_workers
        self.calibration_source = calibration_source

        print("Initializing Metal kernel library...")
        self.metal_lib = MetalKernelLibrary.from_source_dir()
        print(f"  Loaded {len(self.metal_lib._pipelines)} cached pipelines")

        # Create quantizers with use_metal=False for thread safety
        self.quantizers: dict[int, list[EXL3Quantizer]] = {}
        for bits in range(min_bits, max_bits + 1):
            self.quantizers[bits] = [
                EXL3Quantizer(bits=bits, group_size=group_size, max_workers=1, use_metal=False)
                for _ in range(expert_workers)
            ]

        self.config: Any = None
        self.tokenizer: Any = None
        self.model_path: Path | None = None
        self.weight_map: dict[str, str] = {}
        self.prefetcher: WeightPrefetcher | None = None
        self.calibration_data: CalibrationDataset | None = None
        self.calibration_tokens: list[torch.Tensor] | None = None

    def initialize(self, calibration_samples: int | None = None) -> None:
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

        hidden_size = getattr(self.config, "hidden_size", 4096)
        num_layers = getattr(self.config, "num_hidden_layers", 32)
        model_type = getattr(self.config, "model_type", "unknown")

        print("\nModel architecture:")
        print(f"  Type: {model_type}")
        print(f"  Hidden: {hidden_size}, Layers: {num_layers}")

        if hasattr(self.config, "num_experts"):
            print(f"  Experts: {self.config.num_experts}")

        # Load calibration data
        print(f"\nLoading calibration data: {self.calibration_source}")
        if self.calibration_source == "bartowski-v3":
            self.calibration_data = CalibrationDataset.v3(max_samples=calibration_samples)
        elif self.calibration_source == "synthetic":
            # Use synthetic data (for testing/debugging)
            self.calibration_data = None
            print("  Using synthetic activations (for testing only)")
        else:
            # Try loading as local file
            try:
                self.calibration_data = CalibrationDataset.from_local(self.calibration_source)
            except FileNotFoundError:
                print(f"  Warning: Could not load {self.calibration_source}, using synthetic")
                self.calibration_data = None

        if self.calibration_data is not None:
            print(f"  Loaded {len(self.calibration_data)} samples")
            coverage = self.calibration_data.domain_coverage
            print(f"  Domain coverage: {coverage}")

            # Pre-tokenize calibration data
            print("  Tokenizing calibration data...")
            self.calibration_tokens = []
            for sample in self.calibration_data.samples:
                tokens = self.tokenizer.encode(
                    sample,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                )
                if isinstance(tokens, torch.Tensor):
                    self.calibration_tokens.append(tokens.squeeze(0))
                else:
                    self.calibration_tokens.append(torch.tensor(tokens))
            print(f"  Tokenized {len(self.calibration_tokens)} samples")

    def determine_bits(self, tensor_name: str, weight: torch.Tensor, H: np.ndarray) -> int:
        name_lower = tensor_name.lower()
        for pattern, bits in SENSITIVE_PATTERNS.items():
            if pattern in name_lower:
                return bits

        sensitivity = self._compute_sensitivity(weight.numpy(), H)

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
            bits = max(bits, ATTENTION_MIN_BITS)
        elif is_expert:
            bits = max(bits, EXPERT_MIN_BITS)

        return max(self.min_bits, min(self.max_bits, bits))

    def _compute_sensitivity(self, weight: np.ndarray, H: np.ndarray) -> float:
        w_abs = np.abs(weight)
        p99 = np.percentile(w_abs, 99)
        p50 = np.percentile(w_abs, 50)
        outlier_ratio = np.log1p(min(p99 / (p50 + 1e-8), 100)) / np.log1p(100)

        try:
            H_trace = np.trace(H)
            H_frob = np.linalg.norm(H, "fro")
            condition_proxy = min(H_trace / (H_frob + 1e-8), 10) / 10
        except Exception:
            condition_proxy = 0.5

        return outlier_ratio * 2 + condition_proxy * 1.5

    def compute_hessian(
        self, tensor_name: str, weight: torch.Tensor, activations: torch.Tensor
    ) -> tuple[np.ndarray, int, float]:
        out_feat, in_feat = weight.shape

        if activations.shape[1] != in_feat:
            activations = torch.randn(activations.shape[0], in_feat, device="mps") * 0.5

        H = dispatch_hessian_compute(self.metal_lib, activations, sigma_reg=self.sigma_reg)
        torch.mps.synchronize()
        H_np = H.cpu().numpy().astype(np.float64)

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
    ) -> tuple[LayerResult, QuantizedTensor | None]:
        """Quantize a single tensor and return both metadata and quantized data.

        Returns:
            Tuple of (LayerResult metadata, QuantizedTensor data or None on failure)
        """
        short_name = tensor_name.split(".")[-2]
        start = time.perf_counter()

        try:
            quantizer = self.quantizers[bits][worker_id % len(self.quantizers[bits])]
            result = quantizer.quantize_layer(weight.cpu(), H_np, layer_name=short_name)
            elapsed = time.perf_counter() - start

            # Create the quantized tensor data for saving
            quant_data = QuantizedTensor(
                name=tensor_name,
                trellis_indices=result.trellis_indices,
                scales=result.scales,
                su=result.su,
                sv=result.sv,
                bits=bits,
                shape=tuple(weight.shape),
                mse=result.reconstruction_mse,
            )

            meta = LayerResult(
                name=tensor_name,
                shape=tuple(weight.shape),
                mse=result.reconstruction_mse,
                sensitivity=sensitivity,
                actual_bits=bits,
                time_sec=elapsed,
                success=True,
            )

            return meta, quant_data

        except Exception as e:
            meta = LayerResult(
                name=tensor_name,
                shape=tuple(weight.shape) if weight is not None else (0, 0),
                mse=float("inf"),
                sensitivity=0,
                actual_bits=bits,
                time_sec=time.perf_counter() - start,
                success=False,
                error=str(e),
            )
            return meta, None

    def quantize_layer_parallel(
        self, layer: PrefetchedLayer, activations: torch.Tensor
    ) -> list[LayerResult]:
        """Quantize layer with GPU/CPU pipelining.

        Pipeline architecture:
        - Main thread computes Hessians on GPU (serial, fast)
        - Worker threads quantize on CPU (parallel) AS SOON AS Hessian is ready
        - This overlaps GPU and CPU work for ~10% speedup on MoE layers
        """
        all_tensors = layer.tensors
        total = len(all_tensors)

        print(
            f"    Computing Hessians + quantizing (pipelined, {self.expert_workers} CPU threads)..."
        )

        layer_start = time.perf_counter()
        results: list[LayerResult] = []
        bit_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 0}
        hessian_count = 0
        quant_count = 0

        with ThreadPoolExecutor(max_workers=self.expert_workers) as executor:
            futures: dict[Future, str] = {}

            # Main thread computes Hessians, immediately submits to workers
            for idx, (name, weight) in enumerate(all_tensors.items()):
                try:
                    H_np, bits, sensitivity = self.compute_hessian(name, weight, activations)
                except Exception as e:
                    print(f"      Warning: Hessian failed for {name}: {e}")
                    H_np = np.eye(weight.shape[1], dtype=np.float64) * 0.01
                    bits = 4
                    sensitivity = 0.5

                hessian_count += 1

                # Immediately submit quantization to thread pool (GPU/CPU overlap)
                future = executor.submit(
                    self.quantize_tensor_cpu, name, weight, H_np, bits, sensitivity, idx
                )
                futures[future] = name

                # Progress: show Hessian progress
                if hessian_count % max(1, total // 5) == 0:
                    print(f"      Hessians: {hessian_count}/{total}", flush=True)

                # Check for completed quantizations while computing Hessians
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    if future in futures:
                        result = future.result()
                        results.append(result)
                        del futures[future]
                        quant_count += 1
                        if result.success and result.actual_bits in bit_counts:
                            bit_counts[result.actual_bits] += 1

            print(f"      Hessians: {total}/{total} (all done)", flush=True)

            # Wait for remaining quantizations
            # Better progress interval: report at ~10 evenly spaced points
            report_interval = max(1, (total + 9) // 10)

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                quant_count += 1

                if result.success and result.actual_bits in bit_counts:
                    bit_counts[result.actual_bits] += 1

                # Report at intervals AND at completion
                if quant_count % report_interval == 0 or quant_count == total:
                    bits_str = " ".join(f"{b}b:{c}" for b, c in sorted(bit_counts.items()) if c > 0)
                    print(f"      Quantized: {quant_count}/{total} [{bits_str}]", flush=True)

        elapsed = time.perf_counter() - layer_start
        print(f"      Pipelined completion: {elapsed:.1f}s", flush=True)

        return results

    def quantize_prepared_layer(
        self, prepared: PreparedLayer, executor: ThreadPoolExecutor, output_path: Path | None = None
    ) -> list[LayerResult]:
        """Quantize a layer with pre-computed Hessians (CPU-only, no GPU blocking).

        Processes tensors in batches. Each batch saves to disk immediately
        to avoid accumulating 10GB+ of quantized tensors in RAM.

        Returns:
            List of LayerResult metadata (tensors saved directly to disk)
        """
        total = len(prepared.hessian_data)
        batch_size = min(self.expert_workers, 24)

        print(
            f"    Quantizing {total} tensors (CPU, {self.expert_workers} threads, batch={batch_size})..."
        )

        layer_start = time.perf_counter()
        results: list[LayerResult] = []
        all_metadata: list[dict] = []  # Only metadata, not tensor data
        bit_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 0}
        quant_count = 0
        batch_num = 0
        total_bytes = 0

        # Create temp directory for batch shards
        if output_path is not None:
            output_path = Path(output_path)
            temp_dir = output_path / f".tmp_layer_{prepared.layer_idx:04d}"
            temp_dir.mkdir(parents=True, exist_ok=True)

        # Process in batches to control memory
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = prepared.hessian_data[batch_start:batch_end]

            # Submit batch
            futures: dict[Future, int] = {}
            for idx, hd in enumerate(batch):
                weight_cpu = hd.weight.cpu().clone()
                hessian_copy = hd.hessian.copy()
                hd.weight = None  # type: ignore
                hd.hessian = None  # type: ignore

                future = executor.submit(
                    self.quantize_tensor_cpu,
                    hd.name,
                    weight_cpu,
                    hessian_copy,
                    hd.bits,
                    hd.sensitivity,
                    batch_start + idx,
                )
                futures[future] = batch_start + idx

            # Collect batch results and save immediately
            batch_tensors: dict[str, np.ndarray] = {}
            batch_metadata: list[dict] = []

            for future in as_completed(futures):
                meta, quant_data = future.result()
                results.append(meta)
                quant_count += 1

                if meta.success and meta.actual_bits in bit_counts:
                    bit_counts[meta.actual_bits] += 1

                # Add to batch save dict (pack indices for ~5x space savings)
                if quant_data is not None and output_path is not None:
                    safe_key = quant_data.name.replace(".", "__")
                    packed_indices = pack_indices_vectorized(
                        quant_data.trellis_indices, quant_data.bits
                    )
                    batch_tensors[f"{safe_key}__indices"] = packed_indices
                    batch_tensors[f"{safe_key}__scales"] = quant_data.scales.astype(np.float32)
                    batch_tensors[f"{safe_key}__su"] = quant_data.su.astype(np.float32)
                    batch_tensors[f"{safe_key}__sv"] = quant_data.sv.astype(np.float32)

                    bytes_this = (
                        packed_indices.nbytes
                        + quant_data.scales.nbytes
                        + quant_data.su.nbytes
                        + quant_data.sv.nbytes
                    )
                    total_bytes += bytes_this

                    batch_metadata.append(
                        {
                            "name": quant_data.name,
                            "bits": quant_data.bits,
                            "shape": list(quant_data.shape),
                            "mse": quant_data.mse,
                        }
                    )

                    # Clear quant_data immediately
                    quant_data.trellis_indices = None  # type: ignore
                    quant_data.scales = None  # type: ignore
                    quant_data.su = None  # type: ignore
                    quant_data.sv = None  # type: ignore
                    del quant_data

            # Save batch shard immediately to disk
            if batch_tensors and output_path is not None:
                batch_file = temp_dir / f"batch_{batch_num:04d}.safetensors"
                save_file(batch_tensors, str(batch_file))
                all_metadata.extend(batch_metadata)
                batch_num += 1

            # Progress report
            bits_str = " ".join(f"{b}b:{c}" for b, c in sorted(bit_counts.items()) if c > 0)
            print(f"      Quantized: {quant_count}/{total} [{bits_str}]", flush=True)

            # Aggressive cleanup
            batch_tensors.clear()
            batch_metadata.clear()
            del futures
            for hd in batch:
                del hd
            gc.collect()

        elapsed = time.perf_counter() - layer_start
        print(f"      CPU quantization: {elapsed:.1f}s", flush=True)

        # Merge batch shards into final layer shard
        if output_path is not None and batch_num > 0:
            self._merge_batch_shards(
                prepared.layer_idx, temp_dir, output_path, all_metadata, total_bytes
            )

        # Clear the full hessian_data list
        prepared.hessian_data.clear()

        return results

    def _merge_batch_shards(
        self,
        layer_idx: int,
        temp_dir: Path,
        output_path: Path,
        metadata: list[dict],
        total_bytes: int,
    ) -> None:
        """Finalize batch shards into layer output.

        Instead of merging all batches into one giant file (which requires loading
        everything back into RAM), we keep batch files and create an index.
        This allows streaming reads without holding entire layer in memory.
        """
        import shutil

        # Rename temp dir to permanent location
        final_dir = output_path / f"layer_{layer_idx:04d}"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)

        # Create index file listing all batch shards
        batch_files = sorted(final_dir.glob("batch_*.safetensors"))
        shard_info = [{"file": f.name, "tensors": []} for f in batch_files]

        # Map tensors to their shards (batch_size = 12)
        for i, meta in enumerate(metadata):
            batch_num = i // 12
            if batch_num < len(shard_info):
                shard_info[batch_num]["tensors"].append(meta["name"])

        # Save layer index with shard mapping
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
                    "total_tensors": len(metadata),
                    "total_bytes": total_bytes,
                    "shards": shard_info,
                    "tensors": metadata,
                },
                f,
                indent=2,
            )

        print(
            f"    Saved layer {layer_idx}: {len(metadata)} tensors, "
            f"{total_bytes / 1024 / 1024:.1f} MB ({len(batch_files)} shards)"
        )
        gc.collect()

    def save_layer_to_disk(
        self, layer_idx: int, quant_tensors: list[QuantizedTensor], output_path: Path
    ) -> None:
        """Save quantized layer data to disk incrementally.

        Uses safetensors format for efficient storage and loading.
        Each layer is saved as a separate shard file.
        Clears each tensor's data immediately after adding to save dict.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare tensors for saving - use views/casts that avoid copies where possible
        tensors_to_save: dict[str, np.ndarray] = {}
        metadata_entries = []
        total_bytes = 0

        for qt in quant_tensors:
            # Create safe key from tensor name
            safe_key = qt.name.replace(".", "__")

            # Pack indices for efficient storage (~5x savings)
            packed_indices = pack_indices_vectorized(qt.trellis_indices, qt.bits)
            scales = qt.scales if qt.scales.dtype == np.float32 else qt.scales.astype(np.float32)
            su = qt.su.astype(np.float32)  # Always float64 -> float32
            sv = qt.sv.astype(np.float32)

            tensors_to_save[f"{safe_key}__indices"] = packed_indices
            tensors_to_save[f"{safe_key}__scales"] = scales
            tensors_to_save[f"{safe_key}__su"] = su
            tensors_to_save[f"{safe_key}__sv"] = sv

            total_bytes += packed_indices.nbytes + scales.nbytes + su.nbytes + sv.nbytes

            metadata_entries.append(
                {
                    "name": qt.name,
                    "bits": qt.bits,
                    "shape": list(qt.shape),
                    "mse": qt.mse,
                }
            )

            # Clear original data immediately after copying to save dict
            qt.trellis_indices = None  # type: ignore
            qt.scales = None  # type: ignore
            qt.su = None  # type: ignore
            qt.sv = None  # type: ignore

        # Save tensors to shard file
        shard_file = output_path / f"layer_{layer_idx:04d}.safetensors"
        save_file(tensors_to_save, str(shard_file))

        # Save metadata for this layer
        meta_file = output_path / f"layer_{layer_idx:04d}.json"
        with open(meta_file, "w") as f:
            json.dump({"layer_idx": layer_idx, "tensors": metadata_entries}, f, indent=2)

        # Report
        print(
            f"    Saved layer {layer_idx}: {len(quant_tensors)} tensors, {total_bytes / 1024 / 1024:.1f} MB"
        )

        # Clear everything
        tensors_to_save.clear()
        metadata_entries.clear()
        gc.collect()

    def _generate_activations(
        self,
        layer_idx: int,
        in_features: int,
        calibration_samples: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Generate activations for Hessian computation.

        If calibration data is available, use it to create synthetic activations
        that match the calibration distribution. Otherwise, use Gaussian noise.
        """
        hidden_size = getattr(self.config, "hidden_size", 2048)

        if self.calibration_tokens is not None and len(self.calibration_tokens) > 0:
            # Use calibration token statistics to create realistic activations
            # For each token, we simulate the embedding + position flow
            torch.manual_seed(42 + layer_idx)

            # Sample from calibration tokens
            total_tokens = calibration_samples * max_seq_len
            tokens_collected = []

            for i in range(calibration_samples):
                idx = (layer_idx * calibration_samples + i) % len(self.calibration_tokens)
                tokens = self.calibration_tokens[idx]
                tokens_collected.append(tokens[:max_seq_len])

            # Create embeddings based on token diversity
            # More diverse tokens -> higher variance activations
            unique_tokens = set()
            for t in tokens_collected:
                unique_tokens.update(t.tolist())
            diversity = len(unique_tokens) / max(1, sum(len(t) for t in tokens_collected))

            # Layer-dependent activation pattern (deeper = more processed)
            num_layers = getattr(self.config, "num_hidden_layers", 32)
            depth_factor = 1.0 - 0.3 * (layer_idx / max(num_layers - 1, 1))

            # Variance based on calibration diversity and depth
            variance = 0.5 * diversity + 0.3 * depth_factor

            # Generate activations with calibration-informed distribution
            X = torch.randn(total_tokens, in_features, device="mps") * np.sqrt(variance)

            # Add some structure based on calibration (not purely random)
            # This helps with expert routing in MoE models
            if layer_idx > 0:
                structure_factor = 0.1 * (1 - depth_factor)
                structure = torch.randn(in_features, device="mps") * structure_factor
                X = X + structure

        else:
            # Fallback: synthetic activations (original behavior)
            torch.manual_seed(42 + layer_idx)
            total_tokens = calibration_samples * max_seq_len
            num_layers = getattr(self.config, "num_hidden_layers", 32)
            progress = layer_idx / max(num_layers - 1, 1)
            variance = 0.5 + 0.5 * np.sin(progress * np.pi)
            X = torch.randn(total_tokens, in_features, device="mps") * np.sqrt(variance)

        return X

    def quantize_model(
        self,
        output_path: Path | str | None = None,
        max_layers: int | None = None,
        calibration_samples: int = 64,
        max_seq_len: int = 512,
    ) -> list[LayerResult]:
        """Quantize model with incremental saving and memory management.

        Args:
            output_path: Directory to save quantized weights (required for large models)
            max_layers: Limit number of layers (for testing)
            calibration_samples: Number of calibration samples
            max_seq_len: Maximum sequence length for calibration

        Returns:
            List of LayerResult metadata (quantized data is saved to disk)
        """
        self.initialize(calibration_samples=calibration_samples)

        num_layers = getattr(self.config, "num_hidden_layers", 32)
        hidden_size = getattr(self.config, "hidden_size", 2048)

        if max_layers is not None:
            num_layers = min(num_layers, max_layers)

        layer_indices = list(range(num_layers))

        # Setup output directory
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"\nOutput directory: {output_path}")
        else:
            print("\nWARNING: No output path specified. Running in dry-run mode (no saves).")

        # Start weight prefetcher - only 1 layer ahead to limit memory
        # MoE layers have 201 tensors × ~10MB = ~2GB per layer
        weight_prefetch = 1
        print(f"\nStarting weight prefetcher ({weight_prefetch} layer ahead)...")
        self.prefetcher = WeightPrefetcher(
            self.model_path, self.weight_map, prefetch_ahead=weight_prefetch
        )
        self.prefetcher.start(layer_indices)

        # Start Hessian prefetcher (computes Hessians while quantizing previous layer)
        # Queue size is 1 to limit memory to one layer's worth of Hessians
        print("Starting Hessian prefetcher (1 layer ahead, memory-optimized)...")
        hessian_prefetcher = HessianPrefetcher(
            quantizer=self,
            weight_prefetcher=self.prefetcher,
            calibration_samples=calibration_samples,
            max_seq_len=max_seq_len,
        )
        hessian_prefetcher.start()

        all_results = []
        total_start = time.perf_counter()

        print(f"\n{'=' * 60}")
        print(f"Quantizing {num_layers} layers (mixed-precision, deep pipeline)")
        print(f"{'=' * 60}")

        # Use a single ThreadPoolExecutor for all layers (avoids thread creation overhead)
        with ThreadPoolExecutor(max_workers=self.expert_workers) as executor:
            try:
                layer_num = 0
                while True:
                    # Get layer with pre-computed Hessians
                    prepared = hessian_prefetcher.get_prepared_layer(timeout=180)
                    if prepared is None:
                        break

                    layer_num += 1
                    print(
                        f"\n[Layer {layer_num}/{num_layers}] (idx {prepared.layer_idx}, "
                        f"{len(prepared.hessian_data)} tensors, Hessians: {prepared.hessian_time:.1f}s)"
                    )

                    # Quantize using pre-computed Hessians (CPU only - GPU is free for next layer)
                    # Saves each batch to disk immediately to avoid 10GB+ RAM accumulation
                    layer_start = time.perf_counter()
                    results = self.quantize_prepared_layer(prepared, executor, output_path)
                    layer_time = time.perf_counter() - layer_start

                    all_results.extend(results)

                    successful = [r for r in results if r.success]
                    if successful:
                        avg_bits = np.mean([r.actual_bits for r in successful])
                        avg_mse = np.mean([r.mse for r in successful])
                        print(
                            f"  Layer summary: {len(successful)} tensors, avg {avg_bits:.1f}b, "
                            f"MSE={avg_mse:.6f}, total {layer_time:.1f}s"
                        )

                    # Aggressive memory cleanup after each layer
                    del prepared
                    del results
                    del successful

                    # Release shard handles every layer for MoE models
                    self.prefetcher.release_layer(layer_num)

                    # Force garbage collection
                    gc.collect()
                    torch.mps.empty_cache()

                    # Log memory usage every layer during debugging
                    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
                    print(f"  Memory: {mem_mb:.0f} MB (peak RSS)")

            finally:
                hessian_prefetcher.stop()
                self.prefetcher.stop()

        total_time = time.perf_counter() - total_start

        # Save final index if output path specified
        if output_path is not None:
            self._save_model_index(all_results, output_path, total_time)

        self._print_summary(all_results, total_time)

        return all_results

    def _save_model_index(
        self, results: list[LayerResult], output_path: Path, total_time: float
    ) -> None:
        """Save model index and configuration after quantization completes."""
        from collections import Counter

        successful = [r for r in results if r.success]
        bit_counts = Counter(r.actual_bits for r in successful)
        total_params = sum(r.shape[0] * r.shape[1] for r in successful)

        # Calculate effective bits
        weighted_bits = sum(r.actual_bits * r.shape[0] * r.shape[1] for r in successful)
        effective_bits = weighted_bits / total_params if total_params > 0 else 0

        index = {
            "model_id": self.model_id,
            "quantization": {
                "method": "exl3_mixed",
                "min_bits": self.min_bits,
                "max_bits": self.max_bits,
                "group_size": self.group_size,
                "effective_bits": effective_bits,
            },
            "statistics": {
                "total_tensors": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "total_params": total_params,
                "avg_mse": float(np.mean([r.mse for r in successful])) if successful else 0,
                "total_time_sec": total_time,
                "bit_distribution": {f"{b}b": c for b, c in sorted(bit_counts.items())},
            },
            "layers": [
                {
                    "name": r.name,
                    "shape": list(r.shape),
                    "bits": r.actual_bits,
                    "mse": r.mse,
                    "sensitivity": r.sensitivity,
                }
                for r in successful
            ],
        }

        index_path = output_path / "quantization_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"\nSaved model index to {index_path}")

    def _print_summary(self, results: list[LayerResult], total_time: float) -> None:
        from collections import Counter

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n{'=' * 60}")
        print("QUANTIZATION SUMMARY")
        print(f"{'=' * 60}")

        if not successful:
            print("No successful quantizations!")
            return

        avg_mse = np.mean([r.mse for r in successful])
        avg_rmse = np.sqrt(avg_mse)

        print(f"\nTensors: {len(successful)} successful, {len(failed)} failed")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"Throughput: {len(successful) / total_time:.1f} tensors/sec")

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

        if failed:
            print(f"\nFailed tensors: {len(failed)}")
            for r in failed[:5]:
                print(f"  {r.name}: {r.error}")


def main():
    parser = argparse.ArgumentParser(description="Parallel mixed-precision EXL3 quantization")
    parser.add_argument("--model", type=str, default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--min-bits", type=int, default=2)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=64,
        help="Number of calibration samples (default: 64, use None for all)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="bartowski-v3",
        choices=["bartowski-v3", "synthetic"],
        help="Calibration dataset: bartowski-v3 (recommended) or synthetic",
    )
    # NOTE: Optimal is ~24 workers on Apple Silicon. Higher (e.g., 48) is NOT faster
    # due to thread contention in numpy LAPACK and memory bandwidth limits.
    parser.add_argument(
        "--expert-workers",
        type=int,
        default=24,
        help="CPU threads for quantization (optimal: ~24, higher is NOT faster)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for quantized weights (auto-generated from model name if not specified)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving (for testing - will OOM on large models)",
    )
    args = parser.parse_args()

    # Determine output path
    if args.dry_run:
        output_path = None
    elif args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output path from model name
        safe_model_name = args.model.replace("/", "_").replace(":", "_")
        output_path = Path(f"quantized_{safe_model_name}_exl3_mixed")
        print(f"Auto-generated output path: {output_path}")

    print("=" * 60)
    print("PARALLEL MIXED-PRECISION EXL3 QUANTIZATION")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Bits: {args.min_bits}-{args.max_bits} (mixed)")
    print(f"  CPU workers: {args.expert_workers}")
    print(f"  Calibration: {args.calibration} ({args.calibration_samples} samples)")
    print(f"  Output: {output_path if output_path else '(dry-run, no saves)'}")
    if args.max_layers:
        print(f"  Max layers: {args.max_layers}")

    quantizer = ParallelLayerwiseQuantizer(
        model_id=args.model,
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        group_size=args.group_size,
        expert_workers=args.expert_workers,
        calibration_source=args.calibration,
    )

    results = quantizer.quantize_model(
        output_path=output_path,
        max_layers=args.max_layers,
        calibration_samples=args.calibration_samples,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
