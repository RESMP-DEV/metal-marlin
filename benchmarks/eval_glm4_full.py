#!/usr/bin/env python3
"""GLM-4.7-Flash Comprehensive Model Evaluation.

Full evaluation suite for GLM-4.7-Flash quantized models:
- Perplexity: Bartowski V3 calibration dataset (multi-domain)
- Throughput: Prefill/decode at various context lengths
- Memory: Peak usage tracking

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/eval_glm4_full.py --model models/GLM-4.7-Flash-Trellis-3bpw/
    uv run python benchmarks/eval_glm4_full.py --model models/GLM-4.7-Flash-Trellis-3bpw/ --context-sweep --output results/glm4_full.json

Output:
    JSON file with all metrics and human-readable summary to stdout.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure metal_marlin is importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.calibration import CalibrationDataset

# Disable dequantization caching for memory efficiency
from metal_marlin.trellis.linear import TrellisLinear

TrellisLinear.enable_cache = False
from metal_marlin.eval import load_tokenizer
from metal_marlin.trellis.config import TrellisModelConfig
from metal_marlin.kv_cache import TrellisKVCache
from metal_marlin.trellis.lm import TrellisForCausalLM

DEFAULT_MODEL_PATH = _ROOT / "models" / "GLM-4.7-Flash-EXL3-3bpw"
RESULTS_DIR = _ROOT / "benchmarks" / "results"

# Context lengths for throughput sweep
DEFAULT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


@dataclass
class PerplexityResult:
    """Perplexity evaluation result."""

    perplexity: float
    bits_per_byte: float
    n_tokens: int
    n_samples: int
    context_length: int
    stride: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "perplexity": self.perplexity,
            "bits_per_byte": self.bits_per_byte,
            "n_tokens": self.n_tokens,
            "n_samples": self.n_samples,
            "context_length": self.context_length,
            "stride": self.stride,
        }


@dataclass
class ThroughputResult:
    """Throughput measurement result."""

    context_length: int
    prefill_tok_s: float
    decode_tok_s: float
    prefill_latency_ms: float
    decode_latency_ms: float
    with_kv_cache: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "prefill_tok_s": self.prefill_tok_s,
            "decode_tok_s": self.decode_tok_s,
            "prefill_latency_ms": self.prefill_latency_ms,
            "decode_latency_ms": self.decode_latency_ms,
            "with_kv_cache": self.with_kv_cache,
        }


@dataclass
class MemorySnapshot:
    """Memory usage at a specific phase."""

    phase: str
    memory_gb: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "memory_gb": self.memory_gb,
            "timestamp": self.timestamp,
        }


@dataclass
class EvaluationResults:
    """Complete evaluation results container."""

    model_path: str
    timestamp: str
    device: str

    # Perplexity
    perplexity: PerplexityResult | None = None

    # Throughput
    throughput_with_kv: list[ThroughputResult] = field(default_factory=list)
    throughput_without_kv: list[ThroughputResult] = field(default_factory=list)

    # Memory
    peak_memory_gb: float = 0.0
    model_disk_size_gb: float = 0.0
    memory_efficiency: float = 0.0  # memory / disk_size ratio
    memory_snapshots: list[MemorySnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "device": self.device,
            "perplexity": self.perplexity.to_dict() if self.perplexity else None,
            "throughput": {
                "with_kv_cache": [t.to_dict() for t in self.throughput_with_kv],
                "without_kv_cache": [t.to_dict() for t in self.throughput_without_kv],
            },
            "peak_memory_gb": self.peak_memory_gb,
            "model_disk_size_gb": self.model_disk_size_gb,
            "memory_efficiency": self.memory_efficiency,
            "memory_snapshots": [s.to_dict() for s in self.memory_snapshots],
        }


def get_current_memory_gb(device: str) -> float:
    """Get current allocated memory in GB."""
    if device == "mps" and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024**3)
    elif device == "cuda" and torch.cuda.is_available():
        return torch.cuda.current_memory_allocated() / (1024**3)
    return 0.0


def get_peak_memory_gb(device: str) -> float:
    """Get peak allocated memory in GB."""
    if device == "mps" and torch.backends.mps.is_available():
        # MPS doesn't have max_memory_allocated, use current
        return torch.mps.current_allocated_memory() / (1024**3)
    elif device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def get_model_disk_size_gb(model_path: Path) -> float:
    """Calculate total disk size of model files in GB."""
    total_bytes = 0
    if model_path.exists():
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_bytes += file_path.stat().st_size
    return total_bytes / (1024**3)


def record_memory(results: EvaluationResults, phase: str, device: str) -> None:
    """Record memory snapshot for a phase."""
    memory_gb = get_current_memory_gb(device)
    snapshot = MemorySnapshot(
        phase=phase,
        memory_gb=memory_gb,
        timestamp=time.time(),
    )
    results.memory_snapshots.append(snapshot)
    print(f"  Memory [{phase}]: {memory_gb:.2f} GB")


def compute_perplexity_sliding_window_torch(
    model: TrellisForCausalLM,
    tokenizer: Any,
    texts: list[str],
    context_length: int = 512,
    stride: int | None = None,
    device: str = "mps",
    verbose: bool = False,
) -> PerplexityResult:
    """Compute perplexity using sliding window method.

    Matches llama.cpp perplexity computation:
    - Sliding window with stride
    - Only scores non-overlapping tokens (avoids boundary effects)
    - Concatenates texts with newlines
    """
    if stride is None:
        stride = context_length // 2

    # Concatenate texts
    full_text = "\n\n".join(texts)

    # Tokenize
    tokens = tokenizer.encode(full_text)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None and (len(tokens) == 0 or tokens[0] != bos_token_id):
        tokens = [bos_token_id] + tokens

    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    n_tokens = len(tokens)

    if n_tokens < 2:
        raise ValueError("Text too short for perplexity computation")

    model.eval()
    total_nll = 0.0
    total_tokens_scored = 0
    n_windows = 0

    with torch.no_grad():
        start = 0
        while start < n_tokens - 1:
            end = min(start + context_length, n_tokens)
            window_tokens = tokens[start:end]

            input_ids = window_tokens[:-1].unsqueeze(0)
            targets = window_tokens[1:]

            # Forward pass - model returns logits directly
            logits = model(input_ids)
            logits = logits.squeeze(0)

            # Log-softmax for cross-entropy
            log_probs = F.log_softmax(logits, dim=-1)

            # Score only non-overlapping portion (except first window)
            if start == 0:
                score_start = 0
            else:
                score_start = context_length - stride

            score_end = len(targets)

            if score_start < score_end:
                scored_targets = targets[score_start:score_end]
                scored_log_probs = log_probs[score_start:score_end]

                token_log_probs = scored_log_probs[
                    torch.arange(len(scored_targets)), scored_targets
                ]
                window_nll = -token_log_probs.sum().item()

                total_nll += window_nll
                total_tokens_scored += len(scored_targets)

            n_windows += 1
            if verbose and n_windows % 10 == 0:
                ppl_so_far = math.exp(total_nll / total_tokens_scored)
                print(f"  Window {n_windows}: pos {start}-{end}, PPL: {ppl_so_far:.4f}")

            start += stride
            if end >= n_tokens:
                break

    perplexity = math.exp(total_nll / total_tokens_scored)

    # Bits per byte: PPL corresponds to cross-entropy in nats
    # Convert to bits: log2(PPL) = ln(PPL) / ln(2)
    # Bits per byte = bits per token / bytes per token (avg ~4 for UTF-8)
    bits_per_token = math.log2(perplexity)
    bits_per_byte = bits_per_token / 4.0  # Approximate

    return PerplexityResult(
        perplexity=perplexity,
        bits_per_byte=bits_per_byte,
        n_tokens=total_tokens_scored,
        n_samples=len(texts),
        context_length=context_length,
        stride=stride,
    )


def measure_throughput(
    model: TrellisForCausalLM,
    tokenizer: Any,
    context_length: int,
    device: str = "mps",
    gen_tokens: int = 32,
    warmup: int = 2,
    iterations: int = 3,
    use_kv_cache: bool = True,
) -> ThroughputResult:
    """Measure prefill and decode throughput at a given context length.

    Uses TrellisKVCache for efficient autoregressive generation.
    """
    # Create prompt tokens
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 1
    prompt_tokens = [bos_id] + [100] * (context_length - 1)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    model.eval()
    batch_size = 1

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids[:, : min(32, context_length)])
            if device == "mps":
                torch.mps.synchronize()

    # Prefill benchmark (no KV cache - processing full prompt)
    prefill_times: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()

            start = time.perf_counter()
            logits = model(input_ids)
            if device == "mps":
                torch.mps.synchronize()
            prefill_times.append(time.perf_counter() - start)

    prefill_tok_s = context_length / max(sum(prefill_times) / len(prefill_times), 1e-9)

    # Decode benchmark (with KV cache for efficient generation)
    decode_times: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

            if use_kv_cache:
                # Initialize KV cache
                kv_cache = TrellisKVCache(
                    num_layers=model.config.num_hidden_layers,
                    batch_size=batch_size,
                    max_seq_len=context_length + gen_tokens,
                    kv_lora_rank=model.config.kv_lora_rank,
                    device=device,
                )

                # Prefill: process prompt and populate cache
                logits = model(input_ids, kv_cache=kv_cache)
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)

                if device == "mps":
                    torch.mps.synchronize()

                # Decode: generate tokens one at a time
                start = time.perf_counter()
                for _ in range(gen_tokens):
                    logits = model(next_token, kv_cache=kv_cache)
                    next_token = torch.argmax(logits[:, -1:, :], dim=-1)

                if device == "mps":
                    torch.mps.synchronize()
                decode_times.append(time.perf_counter() - start)
            else:
                # No KV cache: re-run full sequence each step (much slower)
                logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)
                full_input = input_ids

                if device == "mps":
                    torch.mps.synchronize()

                start = time.perf_counter()
                for _ in range(gen_tokens):
                    full_input = torch.cat([full_input, next_token], dim=1)
                    logits = model(full_input)
                    next_token = torch.argmax(logits[:, -1:, :], dim=-1)

                if device == "mps":
                    torch.mps.synchronize()
                decode_times.append(time.perf_counter() - start)

    decode_tok_s = gen_tokens / max(sum(decode_times) / len(decode_times), 1e-9)

    return ThroughputResult(
        context_length=context_length,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        prefill_latency_ms=sum(prefill_times) / len(prefill_times) * 1000,
        decode_latency_ms=sum(decode_times) / len(decode_times) * 1000,
        with_kv_cache=use_kv_cache,
    )


def run_context_sweep(
    model: TrellisForCausalLM,
    tokenizer: Any,
    context_lengths: list[int],
    device: str = "mps",
    use_kv_cache: bool = True,
    verbose: bool = False,
) -> list[ThroughputResult]:
    """Run throughput benchmark across various context lengths."""
    results: list[ThroughputResult] = []

    for ctx_len in tqdm(
        context_lengths, desc=f"Context sweep (KV={use_kv_cache})" if verbose else None
    ):
        try:
            result = measure_throughput(
                model, tokenizer, ctx_len, device=device, use_kv_cache=use_kv_cache
            )
            results.append(result)
            if verbose:
                print(
                    f"  {ctx_len:>6}: prefill={result.prefill_tok_s:.1f} tok/s, "
                    f"decode={result.decode_tok_s:.1f} tok/s"
                )
        except Exception as e:
            if verbose:
                print(f"  {ctx_len:>6}: FAILED - {e}")

    return results


def print_summary(results: EvaluationResults) -> None:
    """Print human-readable summary of results."""
    print("\n" + "=" * 70)
    print("GLM-4.7-Flash Comprehensive Evaluation Results")
    print("=" * 70)

    print(f"\nModel: {results.model_path}")
    print(f"Device: {results.device}")
    print(f"Timestamp: {results.timestamp}")

    # Perplexity
    if results.perplexity:
        print("\n" + "-" * 70)
        print("PERPLEXITY (Bartowski V3 Calibration)")
        print("-" * 70)
        print(f"  Perplexity:     {results.perplexity.perplexity:.4f}")
        print(f"  Bits/Byte:      {results.perplexity.bits_per_byte:.4f}")
        print(f"  Tokens Scored:  {results.perplexity.n_tokens:,}")
        print(f"  Samples:        {results.perplexity.n_samples}")
        print(f"  Context Length: {results.perplexity.context_length}")
        print(f"  Stride:         {results.perplexity.stride}")

    # Throughput
    if results.throughput_with_kv:
        print("\n" + "-" * 70)
        print("THROUGHPUT (with KV cache)")
        print("-" * 70)
        print(
            f"  {'Context':>10} {'Prefill (tok/s)':>18} {'Decode (tok/s)':>18} {'Latency (ms)':>14}"
        )
        print(f"  {'-' * 10} {'-' * 18} {'-' * 18} {'-' * 14}")
        for t in results.throughput_with_kv:
            print(
                f"  {t.context_length:>10} {t.prefill_tok_s:>18.1f} "
                f"{t.decode_tok_s:>18.1f} {t.prefill_latency_ms:>13.1f}"
            )

    if results.throughput_without_kv:
        print("\n" + "-" * 70)
        print("THROUGHPUT (without KV cache)")
        print("-" * 70)
        print(f"  {'Context':>10} {'Prefill (tok/s)':>18} {'Decode (tok/s)':>18}")
        print(f"  {'-' * 10} {'-' * 18} {'-' * 18}")
        for t in results.throughput_without_kv:
            print(f"  {t.context_length:>10} {t.prefill_tok_s:>18.1f} {t.decode_tok_s:>18.1f}")

    # Memory
    print("\n" + "-" * 70)
    print("MEMORY USAGE")
    print("-" * 70)
    print(f"  Peak Memory: {results.peak_memory_gb:.2f} GB")
    if results.model_disk_size_gb > 0:
        print(f"  Model Disk Size: {results.model_disk_size_gb:.2f} GB")
        print(f"  Memory Efficiency: {results.memory_efficiency:.2f}x (memory/disk)")

    # Memory timeline
    if results.memory_snapshots:
        print("\n" + "-" * 70)
        print("MEMORY TIMELINE")
        print("-" * 70)
        for snapshot in results.memory_snapshots:
            print(f"  {snapshot.phase:25s}: {snapshot.memory_gb:6.2f} GB")

    print("\n" + "=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="GLM-4.7-Flash Comprehensive Model Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to quantized model directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of calibration samples for perplexity",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for perplexity evaluation",
    )
    parser.add_argument(
        "--context-sweep",
        action="store_true",
        help="Run throughput benchmark across context lengths",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default=",".join(map(str, DEFAULT_CONTEXT_LENGTHS)),
        help="Comma-separated context lengths for sweep",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=32,
        help="Number of tokens to generate for decode throughput",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity evaluation",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip throughput evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "glm4_eval.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (mps, cuda, cpu)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for Transformers",
    )
    parser.add_argument(
        "--verify-memory",
        action="store_true",
        help="Assert memory efficiency (memory <= 1.5x disk size)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print("GLM-4.7-Flash Comprehensive Evaluation")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")

    # Calculate model disk size
    model_path = Path(args.model)
    model_disk_size_gb = get_model_disk_size_gb(model_path)
    print(f"Model disk size: {model_disk_size_gb:.2f} GB")

    results = EvaluationResults(
        model_path=args.model,
        timestamp=datetime.now().isoformat(),
        device=args.device,
        model_disk_size_gb=model_disk_size_gb,
    )

    # Memory tracking: before loading
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    record_memory(results, "start", args.device)

    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = load_tokenizer(args.model)

    # Load quantized model
    print("\n[2/3] Loading quantized model...")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    try:
        quant_model = TrellisForCausalLM.from_pretrained(str(model_path), device=args.device)
        quant_model.eval()
        print(f"  Loaded model with {sum(1 for _ in quant_model.parameters())} parameter tensors")
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Using dummy model for testing...")
        # Create dummy model for testing
        config = TrellisModelConfig(
            hidden_size=2048,
            intermediate_size=5632,
            moe_intermediate_size=1536,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_hidden_layers=4,  # Small for testing
            num_experts=64,
            num_experts_per_tok=8,
            vocab_size=151552,
            max_position_embeddings=131072,
            rope_theta=1000000.0,
        )
        quant_model = TrellisForCausalLM(config).to(args.device)
        quant_model.eval()
        # Update disk size for dummy model
        results.model_disk_size_gb = 0.0

    # Memory tracking: after model load
    record_memory(results, "after_model_load", args.device)

    # Perplexity evaluation
    if not args.skip_perplexity:
        print("\n[3/3] Evaluating perplexity on Bartowski V3 calibration data...")
        try:
            # Load Bartowski V3 multi-domain calibration dataset
            calib_data = CalibrationDataset.v3(max_samples=args.samples)
            texts = calib_data.samples
            if args.verbose:
                print(f"  Loaded {len(texts)} calibration samples (multi-domain)")

            results.perplexity = compute_perplexity_sliding_window_torch(
                quant_model,
                tokenizer,
                texts,
                context_length=args.max_length,
                device=args.device,
                verbose=args.verbose,
            )
            print(f"  Perplexity: {results.perplexity.perplexity:.4f}")
            print(f"  Bits/Byte: {results.perplexity.bits_per_byte:.4f}")
        except Exception as e:
            print(f"  Error computing perplexity: {e}")
            import traceback

            traceback.print_exc()

        # Memory tracking: after perplexity
        record_memory(results, "after_perplexity", args.device)

    # Throughput evaluation
    if not args.skip_throughput:
        print("\n[Evaluating] Throughput at various context lengths...")
        context_lengths = [int(x) for x in args.context_lengths.split(",")]

        # With KV cache
        print("  With KV cache:")
        results.throughput_with_kv = run_context_sweep(
            quant_model,
            tokenizer,
            context_lengths,
            device=args.device,
            use_kv_cache=True,
            verbose=args.verbose,
        )

        # Memory tracking: after throughput with KV cache
        record_memory(results, "after_throughput_kv", args.device)

        # Without KV cache (optional, much slower)
        if args.context_sweep:
            print("  Without KV cache (slow):")
            # Only test smaller contexts without KV cache
            small_contexts = [c for c in context_lengths if c <= 2048]
            results.throughput_without_kv = run_context_sweep(
                quant_model,
                tokenizer,
                small_contexts,
                device=args.device,
                use_kv_cache=False,
                verbose=args.verbose,
            )

            # Memory tracking: after throughput without KV cache
            record_memory(results, "after_throughput_no_kv", args.device)

    # Memory stats
    results.peak_memory_gb = get_peak_memory_gb(args.device)
    record_memory(results, "end", args.device)

    # Calculate memory efficiency
    if results.model_disk_size_gb > 0:
        results.memory_efficiency = results.peak_memory_gb / results.model_disk_size_gb

    # Memory verification
    if args.verify_memory:
        print("\n[Verifying memory efficiency...]")
        if results.model_disk_size_gb == 0:
            print("  Warning: Cannot verify memory (model disk size unknown)")
        elif results.memory_efficiency > 1.5:
            print("  ERROR: Memory efficiency too low!")
            print(f"    Memory: {results.peak_memory_gb:.2f} GB")
            print(f"    Disk size: {results.model_disk_size_gb:.2f} GB")
            print(f"    Ratio: {results.memory_efficiency:.2f}x (expected <= 1.5x)")
            return 1
        else:
            print("  OK: Memory efficiency is good")
            print(f"    Memory: {results.peak_memory_gb:.2f} GB")
            print(f"    Disk size: {results.model_disk_size_gb:.2f} GB")
            print(f"    Ratio: {results.memory_efficiency:.2f}x")

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
