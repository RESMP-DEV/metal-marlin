#!/usr/bin/env python3
"""
Multiprocess parallelized comparison: RTN vs MR-GPTQ on Qwen3-30B-A3B.

Uses ProcessPoolExecutor to bypass Python GIL for true CPU parallelization.
Each worker process gets its own numpy context for full core utilization.

Optimizations:
- ProcessPoolExecutor for true parallel numpy computation
- Chunked processing to reduce IPC overhead
- Memory-mapped tensors where possible
- Batch submission for better scheduling
"""

import gc
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Must be at top level for pickling
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class LayerResult:
    name: str
    shape: tuple[int, int]
    layer_type: str
    rtn_rmse: float
    rtn_mre: float
    mr_rmse: float
    mr_mre: float
    rtn_time: float
    mr_time: float
    params: int

    @property
    def rmse_improvement(self) -> float:
        if self.rtn_rmse == 0:
            return 0.0
        return (self.rtn_rmse - self.mr_rmse) / self.rtn_rmse * 100


def classify_layer(name: str) -> str:
    if "self_attn" in name:
        return "attention"
    if "shared_expert" in name:
        return "shared"
    if "experts" in name:
        return "expert"
    if "mlp" in name:
        return "mlp"
    return "other"


def is_quantizable(name: str, shape: tuple) -> bool:
    if "norm" in name.lower() or "embed" in name.lower():
        return False
    if len(shape) != 2:
        return False
    if shape[0] < 64 or shape[1] < 64:
        return False
    if shape[1] % 128 != 0 or shape[1] % 8 != 0:
        return False
    return True


def process_chunk(chunk: list[tuple[str, np.ndarray]]) -> list[LayerResult]:
    """Process a chunk of layers in a worker process."""
    # Import inside worker to avoid pickling issues
    from metal_marlin.mr_gptq import MRGPTQQuantizer

    # Create quantizers in this process
    rtn_q = MRGPTQQuantizer(
        bits=4,
        format="fp4",
        group_size=128,
        use_hadamard=False,
        actorder=False,
    )
    mr_q = MRGPTQQuantizer(
        bits=4,
        format="fp4",
        group_size=128,
        use_hadamard=True,
        hadamard_block_size=64,
        actorder=True,
    )

    results = []
    for name, tensor in chunk:
        layer_type = classify_layer(name)
        params = tensor.shape[0] * tensor.shape[1]

        # RTN quantization
        t0 = time.perf_counter()
        _, _, m_rtn = rtn_q.quantize_layer(tensor, None, name)
        rtn_time = time.perf_counter() - t0

        # MR-GPTQ quantization
        t0 = time.perf_counter()
        _, _, m_mr = mr_q.quantize_layer(tensor, None, name)
        mr_time = time.perf_counter() - t0

        results.append(
            LayerResult(
                name=name,
                shape=tensor.shape,
                layer_type=layer_type,
                rtn_rmse=m_rtn["error"]["rmse"],
                rtn_mre=m_rtn["error"]["mean_relative_error"],
                mr_rmse=m_mr["error"]["rmse"],
                mr_mre=m_mr["error"]["mean_relative_error"],
                rtn_time=rtn_time,
                mr_time=mr_time,
                params=params,
            )
        )

    return results


def load_tensors_from_shard(st_file: Path) -> list[tuple[str, np.ndarray]]:
    """Load all quantizable tensors from a safetensors file."""
    from safetensors import safe_open

    tensors = []
    with safe_open(str(st_file), framework="pt") as f:
        for name in f.keys():
            if "weight" in name and "norm" not in name.lower() and "embed" not in name.lower():
                tensor = f.get_tensor(name).float().numpy()
                if is_quantizable(name, tensor.shape):
                    tensors.append((name, tensor))
    return tensors


def run_multiprocess_comparison(
    model_dir: Path,
    num_shards: int = 2,
    num_workers: int | None = None,
    chunk_size: int = 16,
) -> list[LayerResult]:
    """Run multiprocess quantization comparison."""

    if num_workers is None:
        num_workers = max(4, mp.cpu_count() - 2)

    st_files = sorted(model_dir.glob("model-*.safetensors"))[:num_shards]

    print(f"Workers: {num_workers} (multiprocessing)")
    print(f"Chunk size: {chunk_size} layers per task")
    print(f"Shards: {len(st_files)}")
    print()

    # Phase 1: Load tensors
    print("Phase 1: Loading tensors...")
    load_start = time.perf_counter()

    all_tensors = []
    for st_file in st_files:
        print(f"  Loading {st_file.name}...")
        tensors = load_tensors_from_shard(st_file)
        all_tensors.extend(tensors)
        print(f"    {len(tensors)} quantizable layers")

    load_time = time.perf_counter() - load_start
    total_bytes = sum(t.nbytes for _, t in all_tensors)
    print(f"\n  Total: {len(all_tensors)} tensors ({total_bytes / 1e9:.2f} GB) in {load_time:.1f}s")
    print(f"  I/O throughput: {total_bytes / load_time / 1e9:.2f} GB/s")
    print()

    # Phase 2: Create chunks for parallel processing
    chunks = []
    for i in range(0, len(all_tensors), chunk_size):
        chunks.append(all_tensors[i : i + chunk_size])

    print(f"Phase 2: Parallel quantization ({len(chunks)} chunks)...")
    quant_start = time.perf_counter()

    results: list[LayerResult] = []
    completed_chunks = 0

    # Use spawn context for macOS compatibility
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}

        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
                completed_chunks += 1

                if completed_chunks % 10 == 0 or completed_chunks == len(chunks):
                    elapsed = time.perf_counter() - quant_start
                    layers_done = len(results)
                    rate = layers_done / elapsed
                    eta = (len(all_tensors) - layers_done) / rate if rate > 0 else 0
                    pct = layers_done / len(all_tensors) * 100
                    print(
                        f"  Progress: {layers_done}/{len(all_tensors)} ({pct:.0f}%) "
                        f"- {rate:.1f} layers/s - ETA: {eta:.0f}s"
                    )
            except Exception as e:
                print(f"  Chunk error: {e}")

    quant_time = time.perf_counter() - quant_start
    total_params = sum(r.params for r in results)

    print(f"\n  Quantized {len(results)} layers in {quant_time:.1f}s")
    print(
        f"  Throughput: {total_params / quant_time / 1e6:.1f}M params/s ({len(results) / quant_time:.1f} layers/s)"
    )

    # Cleanup
    del all_tensors
    gc.collect()

    return results


def print_analysis(results: list[LayerResult], total_time: float):
    """Print comprehensive analysis."""

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    total_params = sum(r.params for r in results)
    total_rtn_time = sum(r.rtn_time for r in results)
    total_mr_time = sum(r.mr_time for r in results)

    avg_rtn_rmse = np.mean([r.rtn_rmse for r in results])
    avg_mr_rmse = np.mean([r.mr_rmse for r in results])
    avg_rtn_mre = np.mean([r.rtn_mre for r in results])
    avg_mr_mre = np.mean([r.mr_mre for r in results])

    weighted_rtn_rmse = sum(r.rtn_rmse * r.params for r in results) / total_params
    weighted_mr_rmse = sum(r.mr_rmse * r.params for r in results) / total_params
    weighted_rtn_mre = sum(r.rtn_mre * r.params for r in results) / total_params
    weighted_mr_mre = sum(r.mr_mre * r.params for r in results) / total_params

    print(
        f"\nLayers: {len(results)} | Params: {total_params:,} ({total_params / 1e9:.2f}B) | Wall time: {total_time:.1f}s"
    )
    print()

    print("QUALITY COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<25} {'RTN FP4':>15} {'MR-GPTQ FP4':>15} {'Improvement':>15}")
    print("-" * 80)

    rmse_imp = (avg_rtn_rmse - avg_mr_rmse) / avg_rtn_rmse * 100
    mre_imp = (avg_rtn_mre - avg_mr_mre) / avg_rtn_mre * 100
    w_rmse_imp = (weighted_rtn_rmse - weighted_mr_rmse) / weighted_rtn_rmse * 100
    w_mre_imp = (weighted_rtn_mre - weighted_mr_mre) / weighted_rtn_mre * 100

    print(f"{'Avg RMSE':<25} {avg_rtn_rmse:>15.6f} {avg_mr_rmse:>15.6f} {rmse_imp:>+14.1f}%")
    print(f"{'Avg MRE':<25} {avg_rtn_mre:>14.2%} {avg_mr_mre:>14.2%} {mre_imp:>+14.1f}%")
    print(
        f"{'Weighted RMSE':<25} {weighted_rtn_rmse:>15.6f} {weighted_mr_rmse:>15.6f} {w_rmse_imp:>+14.1f}%"
    )
    print(
        f"{'Weighted MRE':<25} {weighted_rtn_mre:>14.2%} {weighted_mr_mre:>14.2%} {w_mre_imp:>+14.1f}%"
    )
    print()

    # By layer type
    print("BY LAYER TYPE")
    print("-" * 80)
    print(
        f"{'Type':<12} {'Count':>8} {'Params':>12} {'RTN RMSE':>12} {'MR RMSE':>12} {'RMSE Δ':>10}"
    )
    print("-" * 80)

    for ltype in ["attention", "shared", "expert", "mlp", "other"]:
        layer_results = [r for r in results if r.layer_type == ltype]
        if not layer_results:
            continue

        type_params = sum(r.params for r in layer_results)
        avg_rtn = np.mean([r.rtn_rmse for r in layer_results])
        avg_mr = np.mean([r.mr_rmse for r in layer_results])
        imp = (avg_rtn - avg_mr) / avg_rtn * 100

        print(
            f"{ltype:<12} {len(layer_results):>8} {type_params / 1e6:>10.1f}M {avg_rtn:>12.6f} {avg_mr:>12.6f} {imp:>+9.1f}%"
        )

    print()

    # Throughput
    sequential_time = total_rtn_time + total_mr_time
    speedup = sequential_time / total_time
    overhead = (total_mr_time - total_rtn_time) / total_rtn_time * 100

    print("THROUGHPUT")
    print("-" * 80)
    print(f"Parallel speedup: {speedup:.1f}x")
    print(f"Effective throughput: {total_params / total_time / 1e6:.1f}M params/s")
    print(f"MR-GPTQ overhead vs RTN: {overhead:+.1f}%")
    print()

    # Top improvements
    print("TOP 10 IMPROVEMENTS")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda r: r.rmse_improvement, reverse=True)
    for r in sorted_results[:10]:
        print(f"  {r.name}: RTN={r.rtn_rmse:.6f} → MR={r.mr_rmse:.6f} ({r.rmse_improvement:+.1f}%)")

    print()
    print("=" * 80)
    print(f"CONCLUSION: MR-GPTQ provides {w_rmse_imp:+.1f}% weighted RMSE improvement over RTN FP4")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multiprocess RTN vs MR-GPTQ comparison")
    parser.add_argument("--shards", type=int, default=2, help="Model shards to process")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument("--chunk-size", type=int, default=16, help="Layers per chunk")
    parser.add_argument("--model-dir", type=str, default=None)
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path(__file__).parent.parent / "models" / "Qwen3-30B-A3B"

    print("=" * 80)
    print("MULTIPROCESS COMPARISON: RTN vs MR-GPTQ")
    print(f"Model: {model_dir.name}")
    print("=" * 80)
    print()

    start_time = time.perf_counter()
    results = run_multiprocess_comparison(
        model_dir,
        num_shards=args.shards,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
    )
    total_time = time.perf_counter() - start_time

    print_analysis(results, total_time)


if __name__ == "__main__":
    main()
