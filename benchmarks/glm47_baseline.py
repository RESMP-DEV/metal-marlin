#!/usr/bin/env python3
"""
GLM-4.7-Flash Comprehensive Baseline Benchmark

Establishes baseline metrics for:
1. Decode throughput (tokens/second)
2. Prefill throughput (tokens/second) 
3. Perplexity on WikiText-2 (llama.cpp compatible)
4. Memory usage

These baselines serve as optimization targets for mixed-precision MoE improvements.

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/glm47_baseline.py

    # Quick test (fewer samples)
    uv run python benchmarks/glm47_baseline.py --quick

    # Save as official baseline
    uv run python benchmarks/glm47_baseline.py --save-baseline
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from metal_marlin.eval.perplexity import load_tokenizer, load_wikitext2, log_softmax
from metal_marlin.trellis.lm import TrellisForCausalLM

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

import numpy as np
import torch

# Ensure project is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@dataclass
class ThroughputMetrics:
    """Throughput benchmark results."""
    decode_tok_s: float
    decode_ms_per_token: float
    decode_p50_ms: float
    decode_p99_ms: float
    prefill_128_tok_s: float
    prefill_512_tok_s: float
    prefill_2048_tok_s: float | None = None
    latencies_ms: list[float] = field(default_factory=list)


@dataclass
class PerplexityMetrics:
    """Perplexity benchmark results."""
    ppl: float
    n_tokens: int
    context_length: int
    n_samples: int
    eval_time_s: float


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    model_size_gb: float
    peak_gb: float
    allocated_gb: float


@dataclass
class BaselineResult:
    """Complete baseline benchmark result."""
    model_name: str
    model_path: str
    timestamp: str
    throughput: ThroughputMetrics
    perplexity: PerplexityMetrics | None
    memory: MemoryMetrics
    hardware_info: dict[str, Any]
    notes: str = ""


def _mps_sync() -> None:
    """Synchronize MPS device."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _get_memory_gb() -> float:
    """Get current MPS memory allocation in GB."""
    if hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def _percentile(values: list[float], quantile: float) -> float:
    """Compute percentile of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * quantile), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _get_hardware_info() -> dict[str, Any]:
    """Get hardware information."""
    info: dict[str, Any] = {
        "platform": sys.platform,
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }

    try:
        from metal_marlin.profiling.occupancy import detect_gpu
        gpu = detect_gpu()
        info["gpu_name"] = gpu.name.replace("_", " ").title()
        info["gpu_cores"] = gpu.gpu_cores
        info["memory_bandwidth_gbs"] = gpu.memory_bandwidth_gbs
    except Exception:
        info["gpu_name"] = "Unknown Apple Silicon"
        info["gpu_cores"] = 0
        info["memory_bandwidth_gbs"] = 0.0

    return info


def load_model(model_path: str) -> tuple[TrellisForCausalLM, MemoryMetrics, float]:
    """Load model and measure memory/time."""
    gc.collect()
    torch.mps.empty_cache()

    initial_mem = _get_memory_gb()

    print(f"Loading model from {model_path}...")
    start = time.perf_counter()
    model = TrellisForCausalLM.from_pretrained(model_path, device="mps")
    _mps_sync()
    load_time = time.perf_counter() - start

    model.eval()

    # Warm up with a single forward pass
    with torch.no_grad():
        dummy = torch.randint(0, 1000, (1, 1), device="mps")
        _ = model(dummy)
    _mps_sync()

    allocated = _get_memory_gb()
    model_size = allocated - initial_mem
    peak = allocated  # After warmup

    memory = MemoryMetrics(
        model_size_gb=model_size,
        peak_gb=peak,
        allocated_gb=allocated,
    )

    print(f"  Loaded in {load_time:.1f}s, memory: {allocated:.2f} GB")
    return model, memory, load_time


def benchmark_throughput(
    model: TrellisForCausalLM,
    warmup: int = 5,
    decode_tokens: int = 100,
    decode_runs: int = 3,
    prefill_lengths: list[int] | None = None,
) -> ThroughputMetrics:
    """Benchmark decode and prefill throughput."""

    if prefill_lengths is None:
        prefill_lengths = [128, 512]

    print("\n[Throughput Benchmark]")

    # === Decode benchmark (single token at a time) ===
    print(f"  Decode: {decode_tokens} tokens x {decode_runs} runs...")

    input_ids = torch.randint(0, 1000, (1, 1), device="mps")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
            _mps_sync()

    all_latencies: list[float] = []
    run_times: list[float] = []

    with torch.no_grad():
        for run in range(decode_runs):
            run_start = time.perf_counter()

            for _ in range(decode_tokens):
                _mps_sync()
                step_start = time.perf_counter()
                _ = model(input_ids)
                _mps_sync()
                step_ms = (time.perf_counter() - step_start) * 1000
                all_latencies.append(step_ms)

            run_time = time.perf_counter() - run_start
            run_times.append(run_time)

            avg_ms = statistics.mean(all_latencies[-decode_tokens:])
            print(
                f"    Run {run + 1}: {1000 / avg_ms:.1f} tok/s ({avg_ms:.2f} ms/tok)")

    decode_tok_s = decode_tokens / statistics.mean(run_times)
    decode_ms_per_token = statistics.mean(all_latencies)
    decode_p50 = _percentile(all_latencies, 0.50)
    decode_p99 = _percentile(all_latencies, 0.99)

    print(
        f"  => Decode: {decode_tok_s:.1f} tok/s, P50={decode_p50:.2f}ms, P99={decode_p99:.2f}ms")

    # === Prefill benchmark ===
    prefill_results: dict[int, float] = {}

    for seq_len in prefill_lengths:
        print(f"  Prefill {seq_len} tokens...")
        input_ids = torch.randint(0, 1000, (1, seq_len), device="mps")

        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_ids)
                _mps_sync()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(3):
                _mps_sync()
                start = time.perf_counter()
                _ = model(input_ids)
                _mps_sync()
                times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times)
        throughput = seq_len / avg_time
        prefill_results[seq_len] = throughput
        print(f"    => {throughput:.1f} tok/s ({avg_time * 1000:.1f} ms)")

    return ThroughputMetrics(
        decode_tok_s=decode_tok_s,
        decode_ms_per_token=decode_ms_per_token,
        decode_p50_ms=decode_p50,
        decode_p99_ms=decode_p99,
        prefill_128_tok_s=prefill_results.get(128, 0.0),
        prefill_512_tok_s=prefill_results.get(512, 0.0),
        prefill_2048_tok_s=prefill_results.get(2048),
        latencies_ms=all_latencies,
    )


def benchmark_perplexity(
    model: TrellisForCausalLM,
    tokenizer: Any,
    max_samples: int = 50,
    context_length: int = 512,
    stride: int | None = None,
) -> PerplexityMetrics:
    """Benchmark perplexity on WikiText-2."""

    if stride is None:
        stride = context_length // 2

    print("\n[Perplexity Benchmark]")
    print(f"  Loading WikiText-2 ({max_samples} samples)...")

    texts = load_wikitext2(max_samples)
    full_text = "\n\n".join(texts)

    print(f"  Tokenizing ({len(full_text)} chars)...")
    tokens = tokenizer.encode(full_text)

    # Prepend BOS if needed
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None and (len(tokens) == 0 or tokens[0] != bos_token_id):
        tokens = [bos_token_id] + tokens

    tokens = np.array(tokens, dtype=np.int64)
    n_tokens = len(tokens)

    print(
        f"  Evaluating {n_tokens} tokens (context={context_length}, stride={stride})...")

    start_time = time.perf_counter()

    total_nll = 0.0
    total_tokens_scored = 0
    n_windows = 0

    pos = 0
    while pos < n_tokens - 1:
        end = min(pos + context_length, n_tokens)
        window_tokens = tokens[pos:end]

        # Get logits
        input_ids = torch.tensor(window_tokens[:-1], device="mps").unsqueeze(0)
        targets = window_tokens[1:]

        with torch.no_grad():
            logits = model(input_ids)
            _mps_sync()

        logits_np = logits.squeeze(0).float().cpu().numpy()
        log_probs = log_softmax(logits_np, axis=-1)

        # Score non-overlapping portion
        if pos == 0:
            score_start = 0
        else:
            score_start = context_length - stride

        score_end = len(targets)

        if score_start < score_end:
            scored_targets = targets[score_start:score_end]
            scored_log_probs = log_probs[score_start:score_end]

            token_log_probs = scored_log_probs[np.arange(
                len(scored_targets)), scored_targets]
            window_nll = -np.sum(token_log_probs)

            total_nll += window_nll
            total_tokens_scored += len(scored_targets)

        n_windows += 1
        if n_windows % 20 == 0:
            ppl_so_far = np.exp(
                total_nll / total_tokens_scored) if total_tokens_scored > 0 else float("inf")
            print(
                f"    Window {n_windows}: {total_tokens_scored} tokens scored, PPL={ppl_so_far:.2f}")

        pos += stride
        if end >= n_tokens:
            break

    eval_time = time.perf_counter() - start_time
    ppl = float(np.exp(total_nll / total_tokens_scored))

    print(
        f"  => Perplexity: {ppl:.4f} on {total_tokens_scored} tokens ({eval_time:.1f}s)")

    return PerplexityMetrics(
        ppl=ppl,
        n_tokens=total_tokens_scored,
        context_length=context_length,
        n_samples=len(texts),
        eval_time_s=eval_time,
    )


def save_baseline(result: BaselineResult, output_path: Path) -> None:
    """Save baseline results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_name": result.model_name,
        "model_path": result.model_path,
        "timestamp": result.timestamp,
        "notes": result.notes,
        "hardware": result.hardware_info,
        "memory": {
            "model_size_gb": result.memory.model_size_gb,
            "peak_gb": result.memory.peak_gb,
            "allocated_gb": result.memory.allocated_gb,
        },
        "throughput": {
            "decode_tok_s": result.throughput.decode_tok_s,
            "decode_ms_per_token": result.throughput.decode_ms_per_token,
            "decode_p50_ms": result.throughput.decode_p50_ms,
            "decode_p99_ms": result.throughput.decode_p99_ms,
            "prefill_128_tok_s": result.throughput.prefill_128_tok_s,
            "prefill_512_tok_s": result.throughput.prefill_512_tok_s,
            "prefill_2048_tok_s": result.throughput.prefill_2048_tok_s,
        },
    }

    if result.perplexity:
        data["perplexity"] = {
            "ppl": result.perplexity.ppl,
            "n_tokens": result.perplexity.n_tokens,
            "context_length": result.perplexity.context_length,
            "n_samples": result.perplexity.n_samples,
            "eval_time_s": result.perplexity.eval_time_s,
        }

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"\nBaseline saved to {output_path}")


def print_summary(result: BaselineResult) -> None:
    """Print summary of baseline results."""
    print("\n" + "=" * 70)
    print("GLM-4.7-Flash Baseline Summary")
    print("=" * 70)

    print(f"\nModel: {result.model_name}")
    print(f"Hardware: {result.hardware_info.get('gpu_name', 'Unknown')}")
    print(f"Timestamp: {result.timestamp}")

    print("\n--- Memory ---")
    print(f"  Model size: {result.memory.model_size_gb:.2f} GB")
    print(f"  Peak usage: {result.memory.peak_gb:.2f} GB")

    print("\n--- Throughput ---")
    t = result.throughput
    print(
        f"  Decode:      {t.decode_tok_s:.1f} tok/s ({t.decode_ms_per_token:.2f} ms/tok)")
    print(f"  Prefill 128: {t.prefill_128_tok_s:.1f} tok/s")
    print(f"  Prefill 512: {t.prefill_512_tok_s:.1f} tok/s")
    if t.prefill_2048_tok_s:
        print(f"  Prefill 2048: {t.prefill_2048_tok_s:.1f} tok/s")

    if result.perplexity:
        print("\n--- Perplexity (WikiText-2) ---")
        p = result.perplexity
        print(f"  PPL: {p.ppl:.4f}")
        print(f"  Tokens: {p.n_tokens}")
        print(f"  Context: {p.context_length}")

    print("\n" + "=" * 70)
    print("Baseline established. Use these metrics as optimization targets.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="GLM-4.7-Flash Baseline Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=str(_ROOT / "models" / "GLM-4.7-Flash-Trellis-MM"),
        help="Path to quantized model",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer samples, faster)",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity evaluation (faster)",
    )
    parser.add_argument(
        "--ppl-samples",
        type=int,
        default=50,
        help="Number of WikiText-2 samples for perplexity",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context window for perplexity",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=100,
        help="Tokens for decode benchmark",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save as official baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output path for results",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.ppl_samples = 10
        args.decode_tokens = 20
        args.context_length = 256

    print("=" * 70)
    print("GLM-4.7-Flash Baseline Benchmark")
    print("=" * 70)

    # Load model
    model, memory, load_time = load_model(args.model)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(args.model)

    # Get hardware info
    hardware_info = _get_hardware_info()

    # Throughput benchmark
    throughput = benchmark_throughput(
        model,
        decode_tokens=args.decode_tokens,
        decode_runs=3 if not args.quick else 2,
        prefill_lengths=[128, 512] if not args.quick else [128],
    )

    # Perplexity benchmark
    perplexity = None
    if not args.skip_perplexity:
        perplexity = benchmark_perplexity(
            model,
            tokenizer,
            max_samples=args.ppl_samples,
            context_length=args.context_length,
        )

    # Compile results
    result = BaselineResult(
        model_name=Path(args.model).name,
        model_path=args.model,
        timestamp=datetime.now().isoformat(),
        throughput=throughput,
        perplexity=perplexity,
        memory=memory,
        hardware_info=hardware_info,
        notes="Baseline after slow path removal, before mixed-precision optimization",
    )

    # Print summary
    print_summary(result)

    # Save results
    if args.output:
        save_baseline(result, args.output)

    if args.save_baseline:
        baseline_path = Path(__file__).parent / \
            "results" / "glm47_baseline.json"
        save_baseline(result, baseline_path)


if __name__ == "__main__":
    main()
