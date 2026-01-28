"""Distributed GPTQ Quantization via AlphaHENG.

This module enables distributed GPTQ quantization across multiple machines
using AlphaHENG's orchestration and ZeroMQ infrastructure.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Coordinator (Mac)                           │
    │   - Loads model weights (safetensors)                          │
    │   - Distributes layers to workers                              │
    │   - Aggregates results and saves output                        │
    └─────────────────────────────────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  Worker 1     │   │  Worker 2     │   │  Worker 3     │
    │  (CUDA A100)  │   │  (CUDA H100)  │   │  (Mac M4)     │
    │  - Hessians   │   │  - Hessians   │   │  - Hessians   │
    │  - GPTQ quant │   │  - GPTQ quant │   │  - GPTQ quant │
    └───────────────┘   └───────────────┘   └───────────────┘

Workflow:
    1. Coordinator loads model and calibration data
    2. Coordinator runs calibration forward passes, streaming Hessians
    3. Hessians are distributed to workers
    4. Workers quantize layers in parallel
    5. Coordinator collects results and saves model

For a 30B MoE model with 200+ layers:
    - Single M4 Max: ~12 hours
    - Distributed (1 Mac + 2 CUDA servers): ~30 minutes

Usage:
    # Coordinator (on Mac):
    from metal_marlin.distributed_gptq import DistributedQuantizer

    quantizer = DistributedQuantizer(
        workers=["cuda-server-1:5556", "cuda-server-2:5556"],
        local_workers=2,  # Also use local MPS
    )

    report = await quantizer.quantize_model(
        model_path="path/to/model",
        calibration=calibration_dataset,
        output_path="path/to/output",
    )

    # Workers (on CUDA servers):
    python -m metal_marlin.gptq_accelerated server --port 5556
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .calibration import CalibrationDataset

from .gptq_accelerated import (
    Backend,
    GPTQAccelerated,
    GPTQConfig,
    GPTQLayerResult,
    RemoteGPTQClient,
    detect_best_backend,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Distributed Work Queue
# =============================================================================


@dataclass
class LayerTask:
    """A layer quantization task for distributed processing."""

    layer_name: str
    weight_shape: tuple[int, int]
    hessian_shape: tuple[int, int]
    weight_file: str  # Safetensors file containing weights
    hessian_file: str  # NPZ file containing Hessian
    group_size: int = 128
    actorder: bool = True
    damp: float = 0.01


@dataclass
class LayerResult:
    """Result of distributed layer quantization."""

    layer_name: str
    packed_weights: NDArray[np.uint32]
    scales: NDArray[np.float16]
    error: float
    time_total: float
    worker_id: str


@dataclass
class DistributedQuantizationReport:
    """Report for distributed quantization job."""

    model_path: str
    output_path: str
    total_layers: int
    quantized_layers: int
    total_time: float
    mean_layer_time: float
    workers_used: list[str]
    layer_distribution: dict[str, int]  # worker -> count
    total_error: float
    mean_error: float


# =============================================================================
# Worker Pool
# =============================================================================


class WorkerConnection:
    """Connection to a remote or local worker."""

    def __init__(
        self,
        worker_id: str,
        backend: Backend,
        address: str | None = None,
    ):
        self._worker_id = worker_id
        self._backend = backend
        self._address = address
        self._client: RemoteGPTQClient | GPTQAccelerated | None = None
        self._busy = False
        self._tasks_completed = 0
        self._total_time = 0.0

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def is_remote(self) -> bool:
        return self._backend == Backend.REMOTE_CUDA

    @property
    def is_busy(self) -> bool:
        return self._busy

    def connect(self) -> None:
        """Establish connection to worker."""
        if self._backend == Backend.REMOTE_CUDA and self._address:
            self._client = RemoteGPTQClient(self._address)
            self._client.connect()
        else:
            self._client = GPTQAccelerated.create(backend=self._backend)

    def disconnect(self) -> None:
        """Close connection."""
        if isinstance(self._client, RemoteGPTQClient):
            self._client.close()
        self._client = None

    async def quantize_layer(
        self,
        weights: NDArray[np.floating],
        hessian: NDArray[np.floating],
        config: GPTQConfig,
    ) -> GPTQLayerResult:
        """Quantize a layer on this worker."""
        self._busy = True
        t_start = time.perf_counter()

        try:
            if isinstance(self._client, RemoteGPTQClient):
                # Remote worker: send weights and get activations-based result
                # Note: For remote, we send Hessian directly since activations
                # were already processed locally during calibration
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._quantize_remote,
                    weights,
                    hessian,
                    config,
                )
            else:
                # Local worker
                assert self._client is not None
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._quantize_local,
                    weights,
                    hessian,
                    config,
                )

            self._tasks_completed += 1
            self._total_time += time.perf_counter() - t_start

            return result
        finally:
            self._busy = False

    def _quantize_remote(
        self,
        weights: NDArray[np.floating],
        hessian: NDArray[np.floating],
        config: GPTQConfig,
    ) -> GPTQLayerResult:
        """Quantize on remote worker."""
        assert isinstance(self._client, RemoteGPTQClient)

        # For remote, we use the Hessian directly since calibration was local
        # The remote worker computes H_inv and quantizes
        return self._client.quantize_layer(
            weights=weights,
            activations=hessian,  # Actually Hessian, not activations
            bits=config.bits if hasattr(config, "bits") else 4,
            group_size=config.group_size,
            actorder=config.actorder,
            damp=config.damp,
        )

    def _quantize_local(
        self,
        weights: NDArray[np.floating],
        hessian: NDArray[np.floating],
        config: GPTQConfig,
    ) -> GPTQLayerResult:
        """Quantize on local backend."""
        assert isinstance(self._client, GPTQAccelerated)

        # Store Hessian temporarily
        self._client._hessians["_temp"] = (hessian.astype(np.float64), 1)

        # Quantize
        result = self._client.quantize_layer("_temp", weights)

        # Cleanup
        self._client.clear_hessians()

        return result


class WorkerPool:
    """Pool of distributed workers for quantization."""

    def __init__(self):
        self._workers: list[WorkerConnection] = []
        self._worker_semaphores: dict[str, asyncio.Semaphore] = {}

    def add_remote_worker(self, address: str) -> None:
        """Add a remote CUDA worker."""
        worker_id = f"remote:{address}"
        worker = WorkerConnection(
            worker_id=worker_id,
            backend=Backend.REMOTE_CUDA,
            address=address,
        )
        self._workers.append(worker)
        self._worker_semaphores[worker_id] = asyncio.Semaphore(1)

    def add_local_worker(self, backend: Backend = Backend.AUTO) -> None:
        """Add a local worker."""
        if backend == Backend.AUTO:
            backend = detect_best_backend()

        worker_id = f"local:{backend.name}:{len(self._workers)}"
        worker = WorkerConnection(
            worker_id=worker_id,
            backend=backend,
        )
        self._workers.append(worker)
        self._worker_semaphores[worker_id] = asyncio.Semaphore(1)

    def connect_all(self) -> None:
        """Connect all workers."""
        for worker in self._workers:
            try:
                worker.connect()
                logger.info(f"Connected to worker: {worker.worker_id}")
            except Exception as e:
                logger.warning(f"Failed to connect to {worker.worker_id}: {e}")

    def disconnect_all(self) -> None:
        """Disconnect all workers."""
        for worker in self._workers:
            worker.disconnect()

    async def get_available_worker(self) -> WorkerConnection:
        """Get the next available worker (round-robin with semaphore)."""
        while True:
            for worker in self._workers:
                sem = self._worker_semaphores[worker.worker_id]
                if sem.locked():
                    continue
                await sem.acquire()
                return worker

            # All busy, wait a bit
            await asyncio.sleep(0.1)

    def release_worker(self, worker: WorkerConnection) -> None:
        """Release a worker back to the pool."""
        sem = self._worker_semaphores[worker.worker_id]
        sem.release()

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    @property
    def worker_ids(self) -> list[str]:
        return [w.worker_id for w in self._workers]


# =============================================================================
# Distributed Quantizer
# =============================================================================


@dataclass
class HessianCacheConfig:
    """Configuration for Hessian caching."""

    cache_dir: Path
    max_cache_size_gb: float = 50.0
    compression: bool = True


class DistributedQuantizer:
    """Distributed GPTQ quantization orchestrator.

    Coordinates quantization across multiple workers (local and remote)
    for maximum throughput on large models.

    Example:
        quantizer = DistributedQuantizer(
            workers=["server1:5556", "server2:5556"],
            local_workers=2,  # Use 2 local MPS workers
        )

        report = await quantizer.quantize_model(
            model_path="path/to/model",
            calibration=calibration_dataset,
            output_path="path/to/output",
        )
    """

    def __init__(
        self,
        workers: list[str] | None = None,
        local_workers: int = 1,
        local_backend: Backend = Backend.AUTO,
        config: GPTQConfig | None = None,
        hessian_cache: HessianCacheConfig | None = None,
    ):
        """Initialize distributed quantizer.

        Args:
            workers: List of remote worker addresses (host:port)
            local_workers: Number of local workers to spawn
            local_backend: Backend for local workers (AUTO, MPS, NUMPY)
            config: GPTQ configuration
            hessian_cache: Optional Hessian caching configuration
        """
        self._pool = WorkerPool()
        self._config = config or GPTQConfig()
        self._hessian_cache = hessian_cache

        # Add remote workers
        for address in workers or []:
            self._pool.add_remote_worker(address)

        # Add local workers
        for _ in range(local_workers):
            self._pool.add_local_worker(local_backend)

    async def quantize_model(
        self,
        model_path: str | Path,
        calibration: CalibrationDataset,
        output_path: str | Path,
        tokenizer: Any = None,
        num_calibration_batches: int = 128,
        batch_size: int = 4,
        max_seq_len: int = 2048,
        verbose: bool = True,
    ) -> DistributedQuantizationReport:
        """Quantize model using distributed workers.

        Pipeline:
        1. Load model for calibration forward passes
        2. Run calibration and collect Hessians (locally)
        3. Distribute layers to workers for quantization
        4. Aggregate results and save model

        Args:
            model_path: Path to HuggingFace model
            calibration: Calibration dataset
            output_path: Output directory for quantized model
            tokenizer: HuggingFace tokenizer (loaded if not provided)
            num_calibration_batches: Number of calibration batches
            batch_size: Batch size for calibration
            max_seq_len: Maximum sequence length
            verbose: Print progress

        Returns:
            DistributedQuantizationReport
        """
        import torch

        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()

        if verbose:
            print("=" * 60)
            print("Distributed GPTQ Quantization")
            print("=" * 60)
            print(f"Model: {model_path}")
            print(f"Output: {output_path}")
            print(f"Workers: {self._pool.num_workers}")
            for w_id in self._pool.worker_ids:
                print(f"  - {w_id}")
            print()

        # Connect workers
        if verbose:
            print("Connecting to workers...")
        self._pool.connect_all()

        try:
            # Step 1: Load model and tokenizer
            if verbose:
                print("\nStep 1: Loading model for calibration...")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            model.eval()

            # Step 2: Collect Hessians
            if verbose:
                print("\nStep 2: Collecting Hessians from calibration...")

            hessians = await self._collect_hessians(
                model,
                calibration,
                tokenizer,
                num_calibration_batches,
                batch_size,
                max_seq_len,
                verbose,
            )

            # Release model memory
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if verbose:
                print(f"  Collected Hessians for {len(hessians)} layers")

            # Step 3: Load weights and distribute quantization
            if verbose:
                print("\nStep 3: Distributed quantization...")

            results = await self._distribute_quantization(
                model_path,
                hessians,
                verbose,
            )

            # Step 4: Save quantized model
            if verbose:
                print("\nStep 4: Saving quantized model...")

            self._save_quantized_model(
                model_path,
                output_path,
                results,
                verbose,
            )

            # Compute statistics
            total_time = time.perf_counter() - start_time
            mean_time = total_time / len(results) if results else 0.0
            total_error = sum(r.quantization_error for r in results.values())
            mean_error = total_error / len(results) if results else 0.0

            # Worker distribution
            layer_dist: dict[str, int] = {}
            for result in results.values():
                worker = result.backend
                layer_dist[worker] = layer_dist.get(worker, 0) + 1

            report = DistributedQuantizationReport(
                model_path=str(model_path),
                output_path=str(output_path),
                total_layers=len(hessians),
                quantized_layers=len(results),
                total_time=total_time,
                mean_layer_time=mean_time,
                workers_used=self._pool.worker_ids,
                layer_distribution=layer_dist,
                total_error=total_error,
                mean_error=mean_error,
            )

            if verbose:
                print()
                print("=" * 60)
                print("Distributed Quantization Complete!")
                print("=" * 60)
                print(f"  Layers quantized: {report.quantized_layers}")
                print(f"  Total time: {report.total_time:.1f}s ({report.total_time / 60:.1f}m)")
                print(f"  Mean layer time: {report.mean_layer_time:.2f}s")
                print(f"  Mean error: {report.mean_error:.6f}")
                print("  Worker distribution:")
                for worker, count in report.layer_distribution.items():
                    print(f"    {worker}: {count} layers")

            return report

        finally:
            self._pool.disconnect_all()

    async def _collect_hessians(
        self,
        model: Any,
        calibration: CalibrationDataset,
        tokenizer: Any,
        num_batches: int,
        batch_size: int,
        max_seq_len: int,
        verbose: bool,
    ) -> dict[str, NDArray[np.float32]]:
        """Collect Hessians from calibration forward passes."""
        import torch

        # Streaming Hessian accumulators
        hessian_accum: dict[str, tuple[NDArray[np.float64], int]] = {}
        hooks: list = []

        # Register hooks
        import torch.nn as nn

        def make_hook(name: str, in_features: int):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input

                if not isinstance(x, torch.Tensor):
                    return

                with torch.no_grad():
                    x_flat = x.view(-1, x.shape[-1]).float()
                    x_np = x_flat.cpu().numpy().astype(np.float64)

                # Accumulate H += 2 * X^T @ X
                H_contrib = 2.0 * (x_np.T @ x_np)

                if name in hessian_accum:
                    H_sum, count = hessian_accum[name]
                    hessian_accum[name] = (H_sum + H_contrib, count + x_np.shape[0])
                else:
                    hessian_accum[name] = (H_contrib, x_np.shape[0])

            return hook

        # Register hooks on all linear layers
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Skip non-quantizable layers
            name_lower = name.lower()
            if any(p in name_lower for p in ["embed", "norm", "lm_head", "output", "router"]):
                continue

            in_features = module.in_features
            h = module.register_forward_hook(make_hook(name, in_features))
            hooks.append(h)

        if verbose:
            print(f"  Registered hooks on {len(hooks)} layers")

        # Run calibration
        samples = list(calibration.samples)
        n_samples = min(len(samples), num_batches * batch_size)

        with torch.no_grad():
            for batch_idx in range(0, n_samples, batch_size):
                batch_end = min(batch_idx + batch_size, n_samples)
                batch_texts = samples[batch_idx:batch_end]

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                    padding=True,
                )

                model(**inputs)

                if verbose and (batch_idx // batch_size + 1) % 20 == 0:
                    batches_done = batch_idx // batch_size + 1
                    total_batches = (n_samples + batch_size - 1) // batch_size
                    print(f"  Batch {batches_done}/{total_batches}")

        # Remove hooks
        for h in hooks:
            h.remove()

        # Normalize Hessians
        hessians: dict[str, NDArray[np.float32]] = {}
        for name, (H_sum, count) in hessian_accum.items():
            H = H_sum / count

            # Add damping
            diag_mean = np.mean(np.diag(H))
            H[np.diag_indices_from(H)] += self._config.damp * diag_mean

            hessians[name] = H.astype(np.float32)

        return hessians

    async def _distribute_quantization(
        self,
        model_path: Path,
        hessians: dict[str, NDArray[np.float32]],
        verbose: bool,
    ) -> dict[str, GPTQLayerResult]:
        """Distribute layer quantization to workers."""
        from safetensors import safe_open

        # Find safetensors files
        st_files = sorted(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors in {model_path}")

        # Build layer manifest
        layer_manifest: list[tuple[Path, str]] = []
        for st_file in st_files:
            with safe_open(str(st_file), framework="numpy") as f:
                for name in f.keys():
                    if name.replace(".weight", "") in hessians:
                        layer_manifest.append((st_file, name))

        if verbose:
            print(f"  {len(layer_manifest)} layers to quantize")
            print(f"  {self._pool.num_workers} workers available")

        # Create tasks
        tasks: list[asyncio.Task] = []
        results: dict[str, GPTQLayerResult] = {}
        progress = {"done": 0, "total": len(layer_manifest)}

        async def process_layer(st_file: Path, tensor_name: str) -> tuple[str, GPTQLayerResult]:
            # Load weight
            with safe_open(str(st_file), framework="numpy") as f:
                weights = f.get_tensor(tensor_name)

            # Get Hessian
            hessian_name = tensor_name.replace(".weight", "")
            hessian = hessians[hessian_name]

            # Get worker
            worker = await self._pool.get_available_worker()

            try:
                result = await worker.quantize_layer(
                    weights.astype(np.float32),
                    hessian,
                    self._config,
                )

                progress["done"] += 1
                if verbose and progress["done"] % 10 == 0:
                    pct = progress["done"] / progress["total"] * 100
                    print(
                        f"  [{pct:5.1f}%] {tensor_name}: "
                        f"error={result.quantization_error:.4f} "
                        f"({worker.worker_id})"
                    )

                return tensor_name, result
            finally:
                self._pool.release_worker(worker)

        # Launch all tasks
        for st_file, tensor_name in layer_manifest:
            task = asyncio.create_task(process_layer(st_file, tensor_name))
            tasks.append(task)

        # Wait for all tasks
        task_results = await asyncio.gather(*tasks)

        for tensor_name, result in task_results:
            results[tensor_name] = result

        return results

    def _save_quantized_model(
        self,
        model_path: Path,
        output_path: Path,
        results: dict[str, GPTQLayerResult],
        verbose: bool,
    ) -> None:
        """Save quantized model to output directory."""
        import shutil

        from safetensors import safe_open
        from safetensors.numpy import save_file

        # Copy config files
        for fname in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ]:
            src = model_path / fname
            if src.exists():
                shutil.copy(src, output_path / fname)

        # Load and save tensors
        st_files = sorted(model_path.glob("*.safetensors"))

        output_tensors: dict[str, np.ndarray] = {}

        for st_file in st_files:
            with safe_open(str(st_file), framework="numpy") as f:
                for name in f.keys():
                    if name in results:
                        result = results[name]

                        # Pack to FP4
                        packed = self._pack_fp4(result.indices)

                        output_tensors[name] = packed
                        output_tensors[f"{name}.scales"] = result.scales
                        output_tensors[f"{name}.group_size"] = np.array(
                            [self._config.group_size], dtype=np.int32
                        )
                    else:
                        # Keep original
                        output_tensors[name] = f.get_tensor(name)

        # Save
        output_file = output_path / "model.safetensors"
        save_file(output_tensors, str(output_file))

        if verbose:
            print(f"  Saved to {output_file}")

    def _pack_fp4(self, indices: NDArray[np.int32]) -> NDArray[np.uint32]:
        """Pack FP4 indices to uint32."""
        out_feat, in_feat = indices.shape
        if in_feat % 8 != 0:
            raise ValueError(f"in_features must be divisible by 8, got {in_feat}")

        packed = np.zeros((out_feat, in_feat // 8), dtype=np.uint32)
        for i in range(8):
            packed |= (indices[:, i::8].astype(np.uint32) & 0xF) << (i * 4)

        return packed


# =============================================================================
# CLI for running workers
# =============================================================================


async def distributed_quantize_cli(
    model_path: str,
    output_path: str,
    workers: list[str] | None = None,
    local_workers: int = 2,
    num_calibration: int = 128,
    verbose: bool = True,
) -> None:
    """CLI wrapper for distributed quantization."""
    from .calibration import CalibrationDatasetLoader

    if verbose:
        print("Loading calibration dataset (v3)...")
    calibration = CalibrationDatasetLoader.v3()

    quantizer = DistributedQuantizer(
        workers=workers or [],
        local_workers=local_workers,
    )

    report = await quantizer.quantize_model(
        model_path=model_path,
        calibration=calibration,
        output_path=output_path,
        num_calibration_batches=num_calibration,
        verbose=verbose,
    )

    # Save report
    report_path = Path(output_path) / "distributed_quantization_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "model_path": report.model_path,
                "output_path": report.output_path,
                "total_layers": report.total_layers,
                "quantized_layers": report.quantized_layers,
                "total_time": report.total_time,
                "mean_layer_time": report.mean_layer_time,
                "workers_used": report.workers_used,
                "layer_distribution": report.layer_distribution,
                "total_error": report.total_error,
                "mean_error": report.mean_error,
            },
            f,
            indent=2,
        )

    print(f"\nReport saved to: {report_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Distributed GPTQ Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quantize using local workers only (2 MPS)
    python -m metal_marlin.distributed_gptq model/ output/
    
    # Quantize using remote CUDA servers
    python -m metal_marlin.distributed_gptq model/ output/ \\
        --workers cuda-server-1:5556 cuda-server-2:5556
    
    # Start a CUDA worker server
    python -m metal_marlin.gptq_accelerated server --port 5556
""",
    )

    parser.add_argument("model_path", help="Path to HuggingFace model")
    parser.add_argument("output_path", help="Output directory")
    parser.add_argument(
        "--workers",
        nargs="*",
        help="Remote worker addresses (host:port)",
    )
    parser.add_argument(
        "--local-workers",
        type=int,
        default=2,
        help="Number of local workers (default: 2)",
    )
    parser.add_argument(
        "--calibration",
        type=int,
        default=128,
        help="Number of calibration batches (default: 128)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    asyncio.run(
        distributed_quantize_cli(
            model_path=args.model_path,
            output_path=args.output_path,
            workers=args.workers,
            local_workers=args.local_workers,
            num_calibration=args.calibration,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
