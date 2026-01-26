#!/usr/bin/env python3
"""
Qwen3-32B Dense Model Benchmark.

Benchmarks Qwen/Qwen3-32B with Marlin FP4 quantization:
  1. Download model from HuggingFace
  2. Quantize with MixedPrecisionConfig.default_dense()
  3. Measure PPL, KLD, and throughput
  4. Test multiple group sizes: 64, 128, 256

Usage:
    python benchmarks/bench_qwen3_32b.py
    python benchmarks/bench_qwen3_32b.py --samples 50 --group-sizes 64 128
    python benchmarks/bench_qwen3_32b.py --output benchmarks/results/qwen3_32b.json

Results are saved to benchmarks/results/qwen3_32b.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Imports
import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-32B"
DEFAULT_GROUP_SIZES = [64, 128, 256]
DEFAULT_SAMPLES = 100
DEFAULT_MAX_LENGTH = 512
WARMUP_ITERS = 3
BENCH_ITERS = 10

# M4 Max hardware specs for throughput calculation
M4_MAX_FP16_TFLOPS = 32.0
M4_MAX_BANDWIDTH_GBS = 546.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class QuantizationMetrics:
    """Metrics for a single quantization configuration."""

    group_size: int
    precision: str = "fp4"

    # Perplexity
    ppl_native: float | None = None
    ppl_quantized: float | None = None
    ppl_delta: float | None = None
    ppl_delta_pct: float | None = None

    # KL Divergence
    kl_mean: float | None = None
    kl_max: float | None = None

    # Throughput (GEMM micro-benchmarks)
    throughput_toks_per_sec: float | None = None
    latency_ms: float | None = None
    tflops: float | None = None
    bandwidth_util_pct: float | None = None

    # Compression stats
    compression_ratio: float | None = None
    quantized_layers: int = 0
    skipped_layers: int = 0

    # Errors
    mean_rmse: float | None = None
    max_error: float | None = None


@dataclass
class BenchmarkResult:
    """Complete benchmark result for Qwen3-32B."""

    model_id: str = MODEL_ID
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Model architecture
    architecture: str = "dense"
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0

    # Benchmark config
    num_samples: int = 0
    max_length: int = DEFAULT_MAX_LENGTH
    warmup_iters: int = WARMUP_ITERS
    bench_iters: int = BENCH_ITERS

    # Results per group size
    results: list[QuantizationMetrics] = field(default_factory=list)

    # Quality summary
    best_quality_group_size: int | None = None
    best_speed_group_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d["results"] = [asdict(r) for r in self.results]
        return d


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_wikitext2(max_samples: int = 100) -> list[str]:
    """Load WikiText-2 test set."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
        return texts[:max_samples]
    except ImportError:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-2-raw-v1/wiki.test.raw",
            repo_type="dataset",
        )
        lines = Path(path).read_text().strip().split("\n")
        return [t for t in lines if len(t.strip()) > 50][:max_samples]


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
    verbose: bool = False,
) -> float:
    """Compute perplexity on text dataset."""
    total_nll = 0.0
    total_tokens = 0

    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:max_length]

        input_ids = mx.array(tokens[:-1]).reshape(1, -1)
        targets = mx.array(tokens[1:])

        logits = model(input_ids)
        logits = logits.squeeze(0)

        # log_softmax using logsumexp for numerical stability
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token_log_probs = log_probs[mx.arange(len(targets)), targets]
        nll = -float(mx.sum(token_log_probs))
        mx.eval(nll)

        total_nll += nll
        total_tokens += len(targets)

        if verbose and (i + 1) % 20 == 0:
            ppl_so_far = math.exp(total_nll / total_tokens)
            print(f"    [{i + 1}/{len(texts)}] Running PPL: {ppl_so_far:.4f}")

    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity")

    return math.exp(total_nll / total_tokens)


def compute_kl_divergence(
    model_p: Any,  # Reference
    model_q: Any,  # Test (quantized)
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
) -> tuple[float, float]:
    """
    Compute KL divergence: D_KL(P || Q).

    Returns:
        (mean_kl, max_kl)
    """
    all_kl = []

    for text in texts[:50]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_length]

        input_ids = mx.array(tokens[:-1]).reshape(1, -1)

        logits_p = model_p(input_ids).squeeze(0)
        logits_q = model_q(input_ids).squeeze(0)

        log_p = logits_p - mx.logsumexp(logits_p, axis=-1, keepdims=True)
        log_q = logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True)

        p = mx.exp(log_p)
        kl_per_pos = mx.sum(p * (log_p - log_q), axis=-1)
        mx.eval(kl_per_pos)

        kl_np = np.array(kl_per_pos)
        valid_kl = kl_np[np.isfinite(kl_np)]
        if len(valid_kl) > 0:
            all_kl.extend(valid_kl.tolist())

    if not all_kl:
        return 0.0, 0.0

    return float(np.mean(all_kl)), float(np.max(all_kl))


def replace_linear_with_marlin(model: Any, group_size: int = 128) -> int:
    """Replace QuantizedLinear with MarlinLinear. Returns count replaced."""
    try:
        from metal_marlin import MarlinLinear
    except ImportError:
        from metal_marlin.layers import MarlinLinear

    count = 0
    replacements = []

    def find_replacements(module: Any, depth: int = 0) -> None:
        nonlocal count
        if depth > 10:
            return

        if hasattr(module, "children"):
            for name, child in module.children().items():
                if isinstance(child, nn.QuantizedLinear):
                    replacements.append((module, name, child))
                    count += 1
                else:
                    find_replacements(child, depth + 1)

        if hasattr(module, "__iter__") and not isinstance(module, (str, bytes)):
            try:
                for i, child in enumerate(module):
                    if isinstance(child, nn.QuantizedLinear):
                        replacements.append((module, i, child))
                        count += 1
                    elif hasattr(child, "children"):
                        find_replacements(child, depth + 1)
            except TypeError:
                pass

    find_replacements(model)

    for parent, name_or_idx, old_layer in replacements:
        try:
            marlin = MarlinLinear.from_quantized_linear(old_layer, group_size=group_size)
            if isinstance(name_or_idx, int):
                parent[name_or_idx] = marlin
            else:
                setattr(parent, name_or_idx, marlin)
        except Exception as e:
            print(f"  Warning: Could not convert {name_or_idx}: {e}")
            count -= 1

    return count


def benchmark_gemm_throughput(
    model: Any,
    tokenizer: Any,
    warmup: int = 3,
    iters: int = 10,
) -> dict[str, float]:
    """
    Benchmark GEMM throughput via model forward pass.

    Returns dict with latency_ms, tokens_per_sec, tflops.
    """
    # Use a fixed prompt for consistent timing
    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    tokens = tokenizer.encode(prompt)[:128]
    input_ids = mx.array(tokens).reshape(1, -1)

    # Warmup
    for _ in range(warmup):
        _ = model(input_ids)
        mx.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = model(input_ids)
        mx.synchronize()
        times.append(time.perf_counter() - start)

    # Remove outliers (2-sigma)
    if len(times) > 5:
        mean = statistics.mean(times)
        std = statistics.stdev(times)
        times = [t for t in times if abs(t - mean) < 2 * std]

    latency_s = statistics.median(times)
    latency_ms = latency_s * 1000
    tokens_per_sec = len(tokens) / latency_s

    return {
        "latency_ms": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "input_tokens": len(tokens),
    }


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------


def benchmark_qwen3_32b(
    group_sizes: list[int] | None = None,
    num_samples: int = DEFAULT_SAMPLES,
    max_length: int = DEFAULT_MAX_LENGTH,
    compute_kl: bool = True,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run comprehensive benchmark of Qwen3-32B with Marlin FP4.

    Args:
        group_sizes: List of group sizes to test (default: [64, 128, 256])
        num_samples: Number of WikiText-2 samples for PPL
        max_length: Maximum sequence length
        compute_kl: Whether to compute KL divergence
        verbose: Print progress

    Returns:
        BenchmarkResult with all metrics
    """
    import mlx_lm

    if group_sizes is None:
        group_sizes = DEFAULT_GROUP_SIZES

    result = BenchmarkResult(
        num_samples=num_samples,
        max_length=max_length,
    )

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading {MODEL_ID}...")
    print(f"{'='*60}")

    model, tokenizer = mlx_lm.load(MODEL_ID)
    mx.eval(model.parameters())

    # Extract architecture info from model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        result.num_layers = len(model.model.layers)
        if hasattr(model.model.layers[0], "self_attn"):
            attn = model.model.layers[0].self_attn
            if hasattr(attn, "hidden_size"):
                result.hidden_size = attn.hidden_size
            if hasattr(attn, "num_heads"):
                result.num_attention_heads = attn.num_heads
        if hasattr(model.model.layers[0], "mlp"):
            mlp = model.model.layers[0].mlp
            if hasattr(mlp, "gate_proj"):
                result.intermediate_size = mlp.gate_proj.weight.shape[0]

    if hasattr(model, "lm_head"):
        result.vocab_size = model.lm_head.weight.shape[0]

    print(f"  Layers: {result.num_layers}")
    print(f"  Hidden size: {result.hidden_size}")
    print(f"  Vocab size: {result.vocab_size}")

    # Load dataset
    print(f"\nLoading WikiText-2 ({num_samples} samples)...")
    dataset = load_wikitext2(max_samples=num_samples)
    print(f"  Loaded {len(dataset)} text samples")

    # Baseline: Native 4-bit
    print(f"\n{'='*60}")
    print("Baseline: Native MLX 4-bit (Affine INT4)")
    print(f"{'='*60}")

    ppl_native = compute_perplexity(model, tokenizer, dataset, max_length, verbose)
    print(f"  Perplexity: {ppl_native:.4f}")

    baseline_throughput = benchmark_gemm_throughput(model, tokenizer)
    print(f"  Throughput: {baseline_throughput['tokens_per_sec']:.1f} tok/s")
    print(f"  Latency: {baseline_throughput['latency_ms']:.2f} ms")

    # Test each group size
    for gs in group_sizes:
        print(f"\n{'='*60}")
        print(f"Testing group_size={gs}")
        print(f"{'='*60}")

        metrics = QuantizationMetrics(group_size=gs)
        metrics.ppl_native = ppl_native

        # Reload model fresh for each group size test
        print("  Reloading model...")
        del model
        gc.collect()
        model, _ = mlx_lm.load(MODEL_ID)
        mx.eval(model.parameters())

        # Reference model for KL (keep native)
        if compute_kl:
            model_ref, _ = mlx_lm.load(MODEL_ID)
            mx.eval(model_ref.parameters())

        # Convert to Marlin FP4
        print(f"  Converting to Marlin FP4 (group_size={gs})...")
        num_replaced = replace_linear_with_marlin(model, group_size=gs)
        print(f"  Replaced {num_replaced} layers")
        metrics.quantized_layers = num_replaced
        mx.eval(model.parameters())

        # Perplexity
        print("  Computing perplexity...")
        ppl_quant = compute_perplexity(model, tokenizer, dataset, max_length, verbose)
        print(f"  Perplexity: {ppl_quant:.4f}")

        metrics.ppl_quantized = ppl_quant
        metrics.ppl_delta = ppl_quant - ppl_native
        metrics.ppl_delta_pct = (metrics.ppl_delta / ppl_native) * 100

        print(f"  Delta: {metrics.ppl_delta:+.4f} ({metrics.ppl_delta_pct:+.2f}%)")

        # KL Divergence
        if compute_kl:
            print("  Computing KL divergence...")
            kl_mean, kl_max = compute_kl_divergence(
                model_ref, model, tokenizer, dataset, max_length=256
            )
            metrics.kl_mean = kl_mean
            metrics.kl_max = kl_max
            print(f"  Mean KL: {kl_mean:.6f}")
            print(f"  Max KL: {kl_max:.6f}")
            del model_ref
            gc.collect()

        # Throughput
        print("  Benchmarking throughput...")
        throughput = benchmark_gemm_throughput(model, tokenizer)
        metrics.throughput_toks_per_sec = throughput["tokens_per_sec"]
        metrics.latency_ms = throughput["latency_ms"]
        print(f"  Throughput: {metrics.throughput_toks_per_sec:.1f} tok/s")
        print(f"  Latency: {metrics.latency_ms:.2f} ms")

        # Quality assessment
        if abs(metrics.ppl_delta) < 0.1:
            quality = "EXCELLENT"
        elif abs(metrics.ppl_delta) < 0.5:
            quality = "GOOD"
        elif abs(metrics.ppl_delta) < 1.0:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
        print(f"  Quality: {quality}")

        result.results.append(metrics)

    # Find best configurations
    if result.results:
        # Best quality = smallest PPL delta
        best_quality = min(result.results, key=lambda r: abs(r.ppl_delta or float("inf")))
        result.best_quality_group_size = best_quality.group_size

        # Best speed = highest throughput
        best_speed = max(
            result.results, key=lambda r: r.throughput_toks_per_sec or 0
        )
        result.best_speed_group_size = best_speed.group_size

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY: Qwen3-32B Marlin FP4")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Samples: {len(dataset)}")
    print(f"Baseline PPL: {ppl_native:.4f}")
    print("-" * 60)
    print(f"{'Group Size':<12} {'PPL':<10} {'Delta':<12} {'KL Mean':<10} {'Tok/s':<10}")
    print("-" * 60)

    for m in result.results:
        ppl_str = f"{m.ppl_quantized:.4f}" if m.ppl_quantized else "N/A"
        delta_str = f"{m.ppl_delta:+.4f}" if m.ppl_delta else "N/A"
        kl_str = f"{m.kl_mean:.6f}" if m.kl_mean else "N/A"
        toks_str = f"{m.throughput_toks_per_sec:.1f}" if m.throughput_toks_per_sec else "N/A"
        print(f"{m.group_size:<12} {ppl_str:<10} {delta_str:<12} {kl_str:<10} {toks_str:<10}")

    print("-" * 60)
    print(f"Best quality: group_size={result.best_quality_group_size}")
    print(f"Best speed: group_size={result.best_speed_group_size}")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-32B with Marlin FP4 quantization"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of WikiText-2 samples (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Max sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_GROUP_SIZES,
        help=f"Group sizes to test (default: {DEFAULT_GROUP_SIZES})",
    )
    parser.add_argument(
        "--no-kl",
        action="store_true",
        help="Skip KL divergence computation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "benchmarks" / "results" / "qwen3_32b.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    result = benchmark_qwen3_32b(
        group_sizes=args.group_sizes,
        num_samples=args.samples,
        max_length=args.max_length,
        compute_kl=not args.no_kl,
        verbose=not args.quiet,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
