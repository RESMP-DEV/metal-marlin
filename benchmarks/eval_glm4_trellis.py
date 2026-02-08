#!/usr/bin/env python3
"""GLM-4.7-Flash Trellis Serving Benchmark.

Comprehensive benchmark measuring perplexity and throughput across context lengths:
- Perplexity on WikiText-2 (llama.cpp compatible methodology)
- Encode (prefill) TPS at 1K, 2K, 4K, 8K, 16K, 30K+ contexts
- Decode TPS (autoregressive generation)
- Peak memory usage with accurate tracking

Uses the TrellisGenerator serving interface with fused Metal kernels
for efficient inference and accurate end-to-end measurements.

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/eval_glm4_trellis.py
    uv run python benchmarks/eval_glm4_trellis.py --context-lengths "1024,2048,4096,8192,16384,32768"
    uv run python benchmarks/eval_glm4_trellis.py --throughput-only --context-lengths "1024,2048,4096"
    uv run python benchmarks/eval_glm4_trellis.py --perplexity-only --ppl-samples 50
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# GLM-4.7-Flash tokenizer - CRITICAL: use zai-org, NOT THUDM (wrong vocab size)
GLM4_TOKENIZER_ID = "zai-org/GLM-4.7-Flash"


@dataclass
class ThroughputResult:
    """Throughput measurement for a single context length."""

    context_length: int
    encode_tps: float  # Prefill tokens per second
    decode_tps: float  # Decode tokens per second
    total_tps: float  # Combined throughput
    encode_latency_ms: float
    decode_latency_ms: float
    tokens_generated: int


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    model_id: str
    tokenizer_id: str
    timestamp: str
    device: str
    dtype: str

    # Perplexity
    perplexity: float | None = None
    ppl_tokens: int = 0
    ppl_context_length: int = 0

    # Memory
    model_size_gb: float = 0.0
    peak_memory_gb: float = 0.0
    memory_at_end_gb: float = 0.0

    # Throughput by context length
    throughput_results: list[ThroughputResult] = field(default_factory=list)

    # Summary stats
    avg_encode_tps: float = 0.0
    avg_decode_tps: float = 0.0
    max_context_tested: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "timestamp": self.timestamp,
            "device": self.device,
            "dtype": self.dtype,
            "perplexity": {
                "value": self.perplexity,
                "tokens": self.ppl_tokens,
                "context_length": self.ppl_context_length,
            },
            "memory": {
                "model_size_gb": self.model_size_gb,
                "peak_memory_gb": self.peak_memory_gb,
                "memory_at_end_gb": self.memory_at_end_gb,
            },
            "throughput": [
                {
                    "context_length": r.context_length,
                    "encode_tps": r.encode_tps,
                    "decode_tps": r.decode_tps,
                    "total_tps": r.total_tps,
                    "encode_latency_ms": r.encode_latency_ms,
                    "decode_latency_ms": r.decode_latency_ms,
                    "tokens_generated": r.tokens_generated,
                }
                for r in self.throughput_results
            ],
            "summary": {
                "avg_encode_tps": self.avg_encode_tps,
                "avg_decode_tps": self.avg_decode_tps,
                "max_context_tested": self.max_context_tested,
            },
        }


class MemoryTracker:
    """Track peak memory usage during benchmarking."""

    def __init__(self):
        self.peak_memory_gb = 0.0
        self._baseline_gb = 0.0

    def reset(self):
        """Reset and record baseline memory."""
        gc.collect()
        self._baseline_gb = self._get_memory_gb()
        self.peak_memory_gb = self._baseline_gb

    def update(self) -> float:
        """Update peak memory and return current usage in GB."""
        current = self._get_memory_gb()
        self.peak_memory_gb = max(self.peak_memory_gb, current)
        return current

    def _get_memory_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        try:
            import torch

            if torch.backends.mps.is_available():
                # MPS doesn't have direct memory query, use allocated
                return torch.mps.current_allocated_memory() / (1024**3)
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            pass
        return 0.0

    def get_peak(self) -> float:
        """Get peak memory usage in GB."""
        self.update()
        return self.peak_memory_gb


def sync_device():
    """Synchronize GPU for accurate timing."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def measure_perplexity(
    model,
    tokenizer,
    max_samples: int = 100,
    context_length: int = 2048,
    verbose: bool = True,
) -> tuple[float, int]:
    """Measure perplexity on WikiText-2 using metal_marlin.eval.perplexity.

    Args:
        model: TrellisForCausalLM model
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum WikiText-2 samples to use
        context_length: Context window size for sliding window
        verbose: Print progress

    Returns:
        (perplexity, n_tokens_scored)
    """
    from metal_marlin.eval.perplexity import compute_perplexity_wikitext

    def logits_fn(input_ids: np.ndarray) -> np.ndarray:
        """Wrapper to get logits from model."""
        import torch

        device = next(model.parameters()).device
        ids = torch.from_numpy(input_ids).to(device)
        with torch.no_grad():
            outputs = model(ids)
            # Return as numpy
            return outputs.float().cpu().numpy()

    result = compute_perplexity_wikitext(
        logits_fn=logits_fn,
        tokenizer=tokenizer,
        max_samples=max_samples,
        context_length=context_length,
        verbose=verbose,
    )

    return result["perplexity"], result["n_tokens"]


def measure_throughput_fused(
    generator,
    tokenizer,
    context_length: int,
    decode_tokens: int = 64,
    warmup: int = 1,
    iterations: int = 3,
    verbose: bool = True,
) -> ThroughputResult:
    """Measure encode and decode throughput using fused inference path.

    Uses TrellisGenerator with KV cache for realistic end-to-end measurements.

    Args:
        generator: TrellisGenerator instance
        tokenizer: HuggingFace tokenizer
        context_length: Input context length to test
        decode_tokens: Number of tokens to generate for decode measurement
        warmup: Warmup iterations
        iterations: Measurement iterations
        verbose: Print progress

    Returns:
        ThroughputResult with TPS measurements
    """
    import torch

    # Generate input text of approximately the right length
    # Use repeated common words to get predictable tokenization
    words = "the quick brown fox jumps over the lazy dog " * (context_length // 8)
    input_ids = tokenizer.encode(words, truncation=True, max_length=context_length)
    input_ids = input_ids[:context_length]  # Exact length
    actual_context = len(input_ids)

    prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    if verbose:
        print(f"    Context {actual_context}: ", end="", flush=True)

    from metal_marlin.trellis.generate import GenerationConfig

    config = GenerationConfig(
        max_new_tokens=decode_tokens,
        temperature=1.0,  # No temperature scaling for benchmarking
        do_sample=False,  # Greedy for deterministic benchmarking
    )

    # === ENCODE (PREFILL) BENCHMARK ===
    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = generator.generate(prompt, config=config)
        sync_device()

    # Measure prefill
    encode_times = []
    for _ in range(iterations):
        gc.collect()
        sync_device()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = generator.generate(prompt, config=config)
        sync_device()
        encode_times.append(time.perf_counter() - t0)

    avg_total_time = sum(encode_times) / len(encode_times)

    # To separate encode vs decode, we need a different approach
    # Measure just the prefill phase by generating 0 tokens
    config_prefill = GenerationConfig(
        max_new_tokens=1,  # Single token to measure decode time
        temperature=1.0,
        do_sample=False,
    )

    # Measure time for single token (mostly prefill + minimal decode)
    prefill_times = []
    decode_times = []

    for _ in range(iterations):
        gc.collect()
        sync_device()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = generator.generate(prompt, config=config_prefill)
        sync_device()
        single_token_time = time.perf_counter() - t0
        prefill_times.append(single_token_time)

    avg_prefill_with_decode = sum(prefill_times) / len(prefill_times)

    # Measure decode time separately by generating multiple tokens from short prompt
    short_prompt = "Hello"
    config_decode = GenerationConfig(
        max_new_tokens=decode_tokens,
        temperature=1.0,
        do_sample=False,
    )

    decode_iterations = []
    for _ in range(iterations):
        gc.collect()
        sync_device()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = generator.generate(short_prompt, config=config_decode)
        sync_device()
        decode_total_time = time.perf_counter() - t0
        decode_iterations.append(decode_total_time)

    avg_decode_total = sum(decode_iterations) / len(decode_iterations)

    # Estimate: prefill time for full context = total - (avg decode per token * 1)
    # This is approximate but gives reasonable separation
    avg_decode_per_token = avg_decode_total / decode_tokens
    avg_prefill = max(0, avg_prefill_with_decode - avg_decode_per_token)

    # Recalculate with full generation for more accurate total
    encode_tps = actual_context / avg_prefill if avg_prefill > 0 else float("inf")
    decode_tps = 1.0 / avg_decode_per_token if avg_decode_per_token > 0 else float("inf")

    # More accurate measurement using full generation
    full_times = []
    for _ in range(iterations):
        gc.collect()
        sync_device()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = generator.generate(prompt, config=config)
        sync_device()
        full_times.append(time.perf_counter() - t0)

    avg_full_time = sum(full_times) / len(full_times)

    # Adjust calculations: total time = prefill + (decode_tokens * decode_per_token)
    # So: decode_per_token = (avg_full_time - prefill) / decode_tokens
    # But we also measured avg_decode_total for short prompt

    # Use weighted average of estimates
    # From full generation: decode_time = (avg_full_time - prefill_time)
    # From short prompt: decode_time = avg_decode_total
    # We expect decode times to be similar

    # Better approach: measure prefill only with 1 token, then measure total
    # prefill_time = time_to_1_token - decode_time
    # total_time = prefill_time + decode_tokens * decode_time

    # Use the short prompt measurement as decode baseline
    estimated_decode_time = avg_decode_total / decode_tokens

    # Estimate prefill from single token generation
    estimated_prefill_time = max(0.001, avg_prefill_with_decode - estimated_decode_time)

    # Recalculate TPS
    encode_tps = actual_context / estimated_prefill_time
    decode_tps = 1.0 / estimated_decode_time if estimated_decode_time > 0 else 0.0

    # Total TPS: tokens processed / total time
    total_tokens = actual_context + decode_tokens
    total_tps = total_tokens / avg_full_time

    encode_ms = estimated_prefill_time * 1000
    decode_ms = estimated_decode_time * 1000

    if verbose:
        print(
            f"encode={encode_tps:.1f} tok/s, decode={decode_tps:.1f} tok/s, total={total_tps:.1f} tok/s"
        )

    return ThroughputResult(
        context_length=actual_context,
        encode_tps=encode_tps,
        decode_tps=decode_tps,
        total_tps=total_tps,
        encode_latency_ms=encode_ms,
        decode_latency_ms=decode_ms,
        tokens_generated=decode_tokens,
    )


def main() -> int:
    # Default to local Trellis 3bpw quantized model
    DEFAULT_MODEL_PATH = str(_ROOT / "models" / "GLM-4.7-Flash-Trellis-3bpw")

    parser = argparse.ArgumentParser(description="GLM-4.7-Flash Trellis 3bpw Serving Benchmark")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to Trellis quantized model (default: models/GLM-4.7-Flash-Trellis-3bpw)",
    )
    parser.add_argument(
        "--context-lengths",
        default="1024,2048,4096,8192,16384,32768",
        help="Comma-separated context lengths to test",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=64,
        help="Tokens to generate for decode benchmark",
    )
    parser.add_argument(
        "--ppl-samples",
        type=int,
        default=50,
        help="WikiText-2 samples for perplexity",
    )
    parser.add_argument(
        "--ppl-context",
        type=int,
        default=2048,
        help="Context length for perplexity sliding window",
    )
    parser.add_argument(
        "--perplexity-only",
        action="store_true",
        help="Only run perplexity benchmark",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Only run throughput benchmark",
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "benchmarks" / "results" / "glm4_trellis_serving.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations for throughput benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Measurement iterations for throughput benchmark",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer

        from metal_marlin.trellis.generate import TrellisGenerator
        from metal_marlin.trellis.lm import TrellisForCausalLM
        from metal_marlin.utils.memory import get_system_memory
    except ImportError as e:
        print(f"Error: Required dependencies not available: {e}")
        return 1

    # Resolve device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_path = Path(args.model_path)

    print("=" * 70)
    print("GLM-4.7-Flash Trellis 3bpw Serving Benchmark")
    print("=" * 70)
    print(f"Model: {model_path.name} (Trellis 3-bit quantized)")
    print(f"Tokenizer: {GLM4_TOKENIZER_ID}")
    print(f"Device: {device}")

    # Get system memory info
    mem_info = get_system_memory()
    print(f"System RAM: {mem_info.total_ram_gb:.1f} GB total, {mem_info.available_ram_gb:.1f} GB available")

    # Calculate model size from safetensors files
    model_size_gb = 0.0
    if model_path.exists():
        model_size_gb = sum(f.stat().st_size for f in model_path.rglob("*.safetensors")) / (1024**3)
        print(f"Model size on disk: {model_size_gb:.2f} GB")

    results = BenchmarkResults(
        model_id=str(model_path),
        tokenizer_id=GLM4_TOKENIZER_ID,
        timestamp=datetime.now().isoformat(),
        device=device,
        dtype="trellis-3bpw",
    )
    results.model_size_gb = model_size_gb

    # Initialize memory tracker
    memory_tracker = MemoryTracker()
    memory_tracker.reset()

    # Load Trellis quantized model with fused inference path
    print("\n[1/3] Loading Trellis quantized model (fused inference path)...")
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    try:
        model = TrellisForCausalLM.from_pretrained(str(model_path), device=device)
        model.eval()
        print("    Loaded TrellisForCausalLM (3-bit quantized, fused kernels)")
    except Exception as e:
        print(f"Error loading Trellis model: {e}")
        print("\nMake sure the model exists at the specified path.")
        print(f"Expected: {model_path}")
        print("\nTo quantize the model, run:")
        print("  uv run python -m metal_marlin.quantize --model zai-org/GLM-4.7-Flash --bits 3")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(GLM4_TOKENIZER_ID, trust_remote_code=True)

    # Create generator for efficient inference
    generator = TrellisGenerator(model, tokenizer)
    print("    Created TrellisGenerator (fused KV cache)")

    load_time = time.perf_counter() - t0
    print(f"    Loaded in {load_time:.1f}s")

    # Record memory after loading
    memory_tracker.update()
    results.peak_memory_gb = memory_tracker.get_peak()
    print(f"    Memory after load: {results.peak_memory_gb:.2f} GB")

    # Perplexity benchmark
    if not args.throughput_only:
        print("\n[2/3] Perplexity benchmark (WikiText-2)...")
        try:
            ppl, n_tokens = measure_perplexity(
                model,
                tokenizer,
                max_samples=args.ppl_samples,
                context_length=args.ppl_context,
                verbose=True,
            )
            results.perplexity = ppl
            results.ppl_tokens = n_tokens
            results.ppl_context_length = args.ppl_context
            print(f"    Perplexity: {ppl:.4f} ({n_tokens:,} tokens)")
        except Exception as e:
            print(f"    Perplexity error: {e}")

        # Update peak memory
        memory_tracker.update()
        results.peak_memory_gb = memory_tracker.get_peak()

    # Throughput benchmark
    if not args.perplexity_only:
        print("\n[3/3] Throughput benchmark (fused inference)...")
        context_lengths = [int(x) for x in args.context_lengths.split(",")]

        for ctx_len in context_lengths:
            try:
                # Clear cache before each test
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()

                result = measure_throughput_fused(
                    generator,
                    tokenizer,
                    context_length=ctx_len,
                    decode_tokens=args.decode_tokens,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    verbose=True,
                )
                results.throughput_results.append(result)
                results.max_context_tested = max(results.max_context_tested, result.context_length)

                # Update peak memory
                memory_tracker.update()

            except Exception as e:
                print(f"    Context {ctx_len}: ERROR - {e}")
                import traceback
import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

                traceback.print_exc()

        # Calculate averages
        if results.throughput_results:
            results.avg_encode_tps = sum(r.encode_tps for r in results.throughput_results) / len(
                results.throughput_results
            )
            results.avg_decode_tps = sum(r.decode_tps for r in results.throughput_results) / len(
                results.throughput_results
            )

    # Final memory measurements
    results.peak_memory_gb = memory_tracker.get_peak()
    results.memory_at_end_gb = memory_tracker._get_memory_gb()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_path.name} (Trellis 3bpw)")
    print(f"  Device: {device}")
    print(f"  Model size: {results.model_size_gb:.2f} GB")
    if results.perplexity:
        print(f"  Perplexity: {results.perplexity:.4f} ({results.ppl_tokens:,} tokens)")
    if results.throughput_results:
        print(f"  Avg Encode TPS: {results.avg_encode_tps:.1f}")
        print(f"  Avg Decode TPS: {results.avg_decode_tps:.1f}")
        print(f"  Max Context: {results.max_context_tested:,}")
    print(f"  Peak Memory: {results.peak_memory_gb:.2f} GB")
    print(f"  Memory at End: {results.memory_at_end_gb:.2f} GB")

    # Memory validation
    if results.peak_memory_gb > 20:
        print(f"\n  WARNING: Peak memory ({results.peak_memory_gb:.1f} GB) exceeds 20GB target!")
        print("  Expected ~15GB for GLM-4.7-Flash 3bpw quantized model.")
    else:
        print(f"\n  Memory usage: OK ({results.peak_memory_gb:.1f} GB < 20GB target)")

    # Decode TPS validation
    if results.throughput_results:
        min_decode_tps = min(r.decode_tps for r in results.throughput_results)
        if min_decode_tps < 5:
            print(f"\n  WARNING: Decode TPS ({min_decode_tps:.1f}) below 5 tok/s target!")
        else:
            print(f"\n  Decode TPS: OK ({min_decode_tps:.1f} tok/s > 5 tok/s target)")

    # Throughput table
    if results.throughput_results:
        print("\n  Context  |  Encode TPS  |  Decode TPS  |  Total TPS  |  Encode ms  |  Decode ms")
        print("  " + "-" * 75)
        for r in results.throughput_results:
            print(
                f"  {r.context_length:>7,}  |  {r.encode_tps:>10.1f}  |  {r.decode_tps:>10.1f}  |"
                f"  {r.total_tps:>9.1f}  |  {r.encode_latency_ms:>9.1f}  |  {r.decode_latency_ms:>9.2f}"
            )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
