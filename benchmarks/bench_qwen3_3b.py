#!/usr/bin/env python3
"""
Qwen3-3B Benchmark Suite: FP4 Quantization Validation

Benchmarks Qwen3-3B (dense 3B model) with Marlin FP4 quantization:
  - Download model from HuggingFace (Qwen/Qwen3-3B via mlx-lm)
  - Quantize to FP4/g128 using Marlin
  - Measure prefill and decode throughput
  - Compute perplexity on WikiText-2
  - Compare native 4-bit vs Marlin FP4

Output: benchmarks/results/qwen3_3b.json

Usage:
    cd metal_marlin
    uv run python benchmarks/bench_qwen3_3b.py
    uv run python benchmarks/bench_qwen3_3b.py --samples 50 --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "metal_marlin"))
sys.path.insert(0, str(_ROOT / "python"))

# Check MLX availability early for clear error message
try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

# Model configuration
MODEL_ID = "Qwen/Qwen3-3B"
MODEL_ID_4BIT = "mlx-community/Qwen3-3B-4bit"  # Pre-quantized MLX version


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    model_id: str = MODEL_ID_4BIT
    group_size: int = 128
    num_samples: int = 100  # WikiText-2 samples
    max_length: int = 512  # Max tokens per sample
    prompt_lengths: tuple[int, ...] = (32, 128, 512)
    gen_lengths: tuple[int, ...] = (32, 128, 256)
    warmup_iterations: int = 5
    benchmark_iterations: int = 20


@dataclass
class ThroughputMetrics:
    """Throughput benchmark results."""

    prompt_tokens: int
    generated_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    prefill_tok_s: float
    decode_tok_s: float


@dataclass
class QualityMetrics:
    """Quality evaluation results."""

    perplexity_native: float
    perplexity_marlin: float
    delta_perplexity: float
    delta_percent: float
    kl_divergence_mean: float | None
    kl_divergence_max: float | None


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    model: str
    model_type: str
    params_b: float
    group_size: int
    timestamp: str
    config: dict[str, Any]
    throughput: list[dict[str, Any]]
    quality: dict[str, float]
    summary: dict[str, Any]


def load_wikitext2(max_samples: int = 100) -> list[str]:
    """Load WikiText-2 test set for perplexity evaluation."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
        return texts[:max_samples]
    except ImportError:
        print("Warning: datasets package not found, using fallback")
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

        if verbose and (i + 1) % 10 == 0:
            ppl_so_far = math.exp(total_nll / total_tokens)
            print(f"  Sample {i + 1}/{len(texts)}, running PPL: {ppl_so_far:.4f}")

    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity")

    return math.exp(total_nll / total_tokens)


def compute_kl_divergence(
    model_ref: Any,
    model_test: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
    max_samples: int = 30,
) -> tuple[float, float]:
    """Compute KL divergence between reference and test model."""
    all_kl = []

    for text in texts[:max_samples]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_length]

        input_ids = mx.array(tokens[:-1]).reshape(1, -1)

        logits_ref = model_ref(input_ids).squeeze(0)
        logits_test = model_test(input_ids).squeeze(0)

        log_ref = logits_ref - mx.logsumexp(logits_ref, axis=-1, keepdims=True)
        log_test = logits_test - mx.logsumexp(logits_test, axis=-1, keepdims=True)

        p = mx.exp(log_ref)
        kl_per_pos = mx.sum(p * (log_ref - log_test), axis=-1)
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
        from layers import MarlinLinear

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
            marlin = MarlinLinear.from_quantized_linear(old_layer)
            if isinstance(name_or_idx, int):
                parent[name_or_idx] = marlin
            else:
                setattr(parent, name_or_idx, marlin)
        except Exception as e:
            print(f"Warning: Could not convert {name_or_idx}: {e}")
            count -= 1

    return count


def benchmark_throughput(
    model: Any,
    config: BenchmarkConfig,
) -> list[ThroughputMetrics]:
    """Benchmark prefill and decode throughput."""
    results = []

    for prompt_len in config.prompt_lengths:
        for gen_len in config.gen_lengths:
            input_ids = mx.ones((1, prompt_len), dtype=mx.int32)

            # Warmup
            for _ in range(config.warmup_iterations):
                _ = model(input_ids)
                mx.synchronize()

            prefill_times = []
            decode_times = []

            for _ in range(config.benchmark_iterations):
                # Prefill
                start = time.perf_counter()
                logits = model(input_ids)
                mx.eval(logits)
                mx.synchronize()
                prefill_times.append(time.perf_counter() - start)

                # Decode: sequential single-token passes
                start = time.perf_counter()
                for _ in range(gen_len):
                    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
                    logits = model(next_token)
                    mx.eval(logits)
                mx.synchronize()
                decode_times.append(time.perf_counter() - start)

            avg_prefill = sum(prefill_times) / len(prefill_times) * 1000
            avg_decode = sum(decode_times) / len(decode_times) * 1000

            metrics = ThroughputMetrics(
                prompt_tokens=prompt_len,
                generated_tokens=gen_len,
                prefill_time_ms=avg_prefill,
                decode_time_ms=avg_decode,
                prefill_tok_s=prompt_len / (avg_prefill / 1000),
                decode_tok_s=gen_len / (avg_decode / 1000),
            )
            results.append(metrics)

            print(
                f"  Prompt={prompt_len:>4}, Gen={gen_len:>4}: "
                f"Prefill {metrics.prefill_tok_s:>8.0f} tok/s, "
                f"Decode {metrics.decode_tok_s:>7.1f} tok/s"
            )

    return results


def run_benchmark(config: BenchmarkConfig, compute_kl: bool = True) -> BenchmarkResults:
    """Run complete benchmark suite."""
    import mlx_lm

    print("=" * 70)
    print("Qwen3-3B Benchmark: FP4 Quantization Validation")
    print("=" * 70)

    # Load model
    print(f"\n[1/5] Loading model: {config.model_id}")
    model, tokenizer = mlx_lm.load(config.model_id)
    mx.eval(model.parameters())

    # Count parameters
    total_params = sum(p.size for p in mx.utils.tree_flatten(model.parameters()))
    params_b = total_params / 1e9
    print(f"  Parameters: {params_b:.2f}B")

    # Load evaluation dataset
    print(f"\n[2/5] Loading WikiText-2 ({config.num_samples} samples)...")
    dataset = load_wikitext2(max_samples=config.num_samples)
    print(f"  Loaded {len(dataset)} text samples")

    # Baseline: Native MLX 4-bit
    print("\n[3/5] Evaluating native MLX 4-bit (Affine INT4)...")
    ppl_native = compute_perplexity(
        model, tokenizer, dataset, max_length=config.max_length, verbose=True
    )
    print(f"  Native 4-bit Perplexity: {ppl_native:.4f}")

    # Keep reference for KL divergence
    if compute_kl:
        model_ref, _ = mlx_lm.load(config.model_id)
        mx.eval(model_ref.parameters())

    # Convert to Marlin FP4
    print(f"\n[4/5] Converting to Marlin FP4 (group_size={config.group_size})...")
    num_replaced = replace_linear_with_marlin(model, group_size=config.group_size)
    print(f"  Replaced {num_replaced} QuantizedLinear layers")
    mx.eval(model.parameters())

    # Marlin FP4 perplexity
    print("\n[5/5] Evaluating Marlin FP4...")
    ppl_marlin = compute_perplexity(
        model, tokenizer, dataset, max_length=config.max_length, verbose=True
    )
    print(f"  Marlin FP4 Perplexity: {ppl_marlin:.4f}")

    delta = ppl_marlin - ppl_native
    delta_pct = delta / ppl_native * 100

    # KL Divergence
    kl_mean, kl_max = None, None
    if compute_kl:
        print("\nComputing KL divergence...")
        kl_mean, kl_max = compute_kl_divergence(model_ref, model, tokenizer, dataset)
        print(f"  Mean KL: {kl_mean:.6f}")
        print(f"  Max KL:  {kl_max:.6f}")
        del model_ref
        mx.synchronize()

    # Throughput benchmark
    print("\nBenchmarking throughput (Marlin FP4)...")
    throughput = benchmark_throughput(model, config)

    # Quality metrics
    quality = QualityMetrics(
        perplexity_native=ppl_native,
        perplexity_marlin=ppl_marlin,
        delta_perplexity=delta,
        delta_percent=delta_pct,
        kl_divergence_mean=kl_mean,
        kl_divergence_max=kl_max,
    )

    # Summary
    avg_prefill = sum(t.prefill_tok_s for t in throughput) / len(throughput)
    avg_decode = sum(t.decode_tok_s for t in throughput) / len(throughput)

    summary = {
        "avg_prefill_tok_s": avg_prefill,
        "avg_decode_tok_s": avg_decode,
        "perplexity_delta": delta,
        "perplexity_delta_pct": delta_pct,
        "quality_assessment": assess_quality(delta, kl_mean),
    }

    results = BenchmarkResults(
        model=config.model_id,
        model_type="dense",
        params_b=params_b,
        group_size=config.group_size,
        timestamp=datetime.now(UTC).isoformat(),
        config=asdict(config),
        throughput=[asdict(t) for t in throughput],
        quality=asdict(quality),
        summary=summary,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY: Qwen3-3B Marlin FP4")
    print("=" * 70)
    print(f"Model:              {config.model_id}")
    print(f"Parameters:         {params_b:.2f}B (dense)")
    print(f"Group Size:         {config.group_size}")
    print(f"Samples:            {len(dataset)}")
    print("-" * 70)
    print(f"Native 4-bit PPL:   {ppl_native:.4f}")
    print(f"Marlin FP4 PPL:     {ppl_marlin:.4f}")
    print(f"PPL Delta:          {delta:+.4f} ({delta_pct:+.2f}%)")
    if kl_mean is not None:
        print(f"Mean KL Divergence: {kl_mean:.6f}")
    print("-" * 70)
    print(f"Avg Prefill:        {avg_prefill:.0f} tok/s")
    print(f"Avg Decode:         {avg_decode:.1f} tok/s")
    print("-" * 70)
    print(f"Quality:            {summary['quality_assessment']}")
    print("=" * 70)

    return results


def assess_quality(delta_ppl: float, kl_mean: float | None) -> str:
    """Assess quantization quality."""
    if abs(delta_ppl) < 0.1:
        return "EXCELLENT - negligible degradation"
    elif abs(delta_ppl) < 0.5:
        return "GOOD - minor degradation"
    elif abs(delta_ppl) < 1.0:
        return "ACCEPTABLE - noticeable but usable"
    elif abs(delta_ppl) < 2.0:
        return "MARGINAL - significant degradation"
    else:
        return "POOR - severe degradation"


def main():
    if not HAS_MLX:
        print("ERROR: Benchmarks require MLX for Metal GPU access.")
        print("Install with: pip install mlx")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-3B with Marlin FP4 quantization"
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID_4BIT,
        help=f"HuggingFace model ID (default: {MODEL_ID_4BIT})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of WikiText-2 samples (default: 100)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Marlin FP4 group size (default: 128)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer samples, skip KL divergence",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "benchmarks" / "results" / "qwen3_3b.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--no-kl",
        action="store_true",
        help="Skip KL divergence computation",
    )

    args = parser.parse_args()

    # Configure benchmark
    if args.quick:
        config = BenchmarkConfig(
            model_id=args.model,
            group_size=args.group_size,
            num_samples=min(args.samples, 30),
            prompt_lengths=(32, 128),
            gen_lengths=(32, 64),
            warmup_iterations=2,
            benchmark_iterations=5,
        )
        compute_kl = False
    else:
        config = BenchmarkConfig(
            model_id=args.model,
            group_size=args.group_size,
            num_samples=args.samples,
        )
        compute_kl = not args.no_kl

    # Run benchmark
    results = run_benchmark(config, compute_kl=compute_kl)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
