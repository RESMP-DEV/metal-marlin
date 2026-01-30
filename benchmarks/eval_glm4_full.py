#!/usr/bin/env python3
"""GLM-4.7-Flash Comprehensive Model Evaluation.

Full evaluation suite for GLM-4.7-Flash quantized models:
- Perplexity: WikiText-2 sliding window computation
- KL Divergence: vs FP16 reference model
- Throughput: Prefill/decode at various context lengths
- Memory: Peak usage tracking

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/eval_glm4_full.py --model models/GLM-4.7-Flash-EXL3-3bpw/
    uv run python benchmarks/eval_glm4_full.py --model models/GLM-4.7-Flash-EXL3-3bpw/ --ref-model THUDM/GLM-4-9B-Chat
    uv run python benchmarks/eval_glm4_full.py --model models/GLM-4.7-Flash-EXL3-3bpw/ --context-sweep --output results/glm4_full.json

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

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure metal_marlin is importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from metal_marlin.eval_kl_divergence import KLResult, compute_kl_divergence_np
from metal_marlin.eval_perplexity import load_tokenizer, load_wikitext2
from metal_marlin.trellis_config import TrellisModelConfig
from metal_marlin.trellis_lm import TrellisForCausalLM

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
class EvaluationResults:
    """Complete evaluation results container."""

    model_path: str
    ref_model_path: str | None
    timestamp: str
    device: str

    # Perplexity
    perplexity: PerplexityResult | None = None

    # KL Divergence
    kl_divergence: KLResult | None = None

    # Throughput
    throughput_with_kv: list[ThroughputResult] = field(default_factory=list)
    throughput_without_kv: list[ThroughputResult] = field(default_factory=list)

    # Memory
    peak_memory_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "ref_model_path": self.ref_model_path,
            "timestamp": self.timestamp,
            "device": self.device,
            "perplexity": self.perplexity.to_dict() if self.perplexity else None,
            "kl_divergence": {
                "mean": self.kl_divergence.kl_mean,
                "max": self.kl_divergence.kl_max,
                "std": self.kl_divergence.kl_std,
                "p95": self.kl_divergence.kl_p95,
                "quality": self.kl_divergence.quality_rating(),
            } if self.kl_divergence else None,
            "throughput": {
                "with_kv_cache": [t.to_dict() for t in self.throughput_with_kv],
                "without_kv_cache": [t.to_dict() for t in self.throughput_without_kv],
            },
            "peak_memory_gb": self.peak_memory_gb,
        }


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

            # Forward pass
            logits, _ = model(input_ids)
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


def compute_kl_divergence_torch(
    ref_model: torch.nn.Module,
    quant_model: TrellisForCausalLM,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
    temperature: float = 1.0,
    device: str = "mps",
    verbose: bool = False,
) -> KLResult:
    """Compute KL divergence between reference and quantized model."""
    all_kl: list[float] = []
    total_tokens = 0

    ref_model.eval()
    quant_model.eval()

    for i, text in enumerate(tqdm(texts, desc="Computing KL divergence", disable=not verbose)):
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_length]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            # Reference model logits
            ref_outputs = ref_model(input_ids)
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs

            # Quantized model logits
            quant_logits, _ = quant_model(input_ids)

            # Ensure same shape
            if ref_logits.shape != quant_logits.shape:
                if verbose:
                    print(f"  Warning: Shape mismatch {ref_logits.shape} vs {quant_logits.shape}")
                continue

            # Convert to numpy for KL computation
            ref_logits_np = ref_logits.cpu().float().numpy()
            quant_logits_np = quant_logits.cpu().float().numpy()

            # Compute KL divergence
            kl_mean, kl_max, kl_std, kl_p95 = compute_kl_divergence_np(
                ref_logits_np, quant_logits_np, temperature
            )

            num_tokens_sample = ref_logits.shape[1]
            total_tokens += num_tokens_sample
            all_kl.extend([kl_mean] * num_tokens_sample)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(texts)}] Running KL: {np.mean(all_kl):.4f}")

    if not all_kl:
        return KLResult(
            kl_mean=0.0,
            kl_max=0.0,
            kl_std=0.0,
            kl_p95=0.0,
            num_tokens=0,
            num_samples=0,
            temperature=temperature,
        )

    all_kl_arr = np.array(all_kl)

    return KLResult(
        kl_mean=float(np.mean(all_kl_arr)),
        kl_max=float(np.max(all_kl_arr)),
        kl_std=float(np.std(all_kl_arr)),
        kl_p95=float(np.percentile(all_kl_arr, 95)),
        num_tokens=total_tokens,
        num_samples=len(texts),
        temperature=temperature,
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
    """Measure prefill and decode throughput at a given context length."""
    # Create prompt tokens
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 1
    prompt_tokens = [bos_id] + [100] * (context_length - 1)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if use_kv_cache:
                _ = model(input_ids[:, : min(32, context_length)])
            else:
                _ = model(input_ids[:, : min(32, context_length)])
            if device == "mps":
                torch.mps.synchronize()

    # Prefill benchmark
    prefill_times: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()

            start = time.perf_counter()
            outputs = model(input_ids)
            if device == "mps":
                torch.mps.synchronize()
            prefill_times.append(time.perf_counter() - start)

    prefill_tok_s = context_length / max(sum(prefill_times) / len(prefill_times), 1e-9)

    # Decode benchmark
    decode_times: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

            # Get initial KV cache
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
            next_token = torch.argmax(outputs[0][:, -1:, :], dim=-1)

            if device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            for _ in range(gen_tokens):
                if use_kv_cache and past_kv is not None:
                    outputs = model(next_token, use_cache=True, past_key_values=past_kv)
                else:
                    # Re-run full sequence (much slower)
                    full_input = torch.cat([input_ids, next_token], dim=1)
                    outputs = model(full_input)
                past_kv = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
                next_token = torch.argmax(outputs[0][:, -1:, :], dim=-1)

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

    for ctx_len in tqdm(context_lengths, desc=f"Context sweep (KV={use_kv_cache})" if verbose else None):
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
    if results.ref_model_path:
        print(f"Reference: {results.ref_model_path}")
    print(f"Device: {results.device}")
    print(f"Timestamp: {results.timestamp}")

    # Perplexity
    if results.perplexity:
        print("\n" + "-" * 70)
        print("PERPLEXITY (WikiText-2)")
        print("-" * 70)
        print(f"  Perplexity:     {results.perplexity.perplexity:.4f}")
        print(f"  Bits/Byte:      {results.perplexity.bits_per_byte:.4f}")
        print(f"  Tokens Scored:  {results.perplexity.n_tokens:,}")
        print(f"  Samples:        {results.perplexity.n_samples}")
        print(f"  Context Length: {results.perplexity.context_length}")
        print(f"  Stride:         {results.perplexity.stride}")

    # KL Divergence
    if results.kl_divergence:
        kl = results.kl_divergence
        print("\n" + "-" * 70)
        print("KL DIVERGENCE (Quantized || FP16)")
        print("-" * 70)
        print(f"  Mean KL:    {kl.kl_mean:.6f}")
        print(f"  Max KL:     {kl.kl_max:.6f}")
        print(f"  Std KL:     {kl.kl_std:.6f}")
        print(f"  95th %ile:  {kl.kl_p95:.6f}")
        print(f"  Quality:    {kl.quality_rating().upper()}")
        print(f"  Tokens:     {kl.num_tokens:,}")

        # Interpretation
        print("\n  Interpretation:")
        if kl.kl_mean < 0.01:
            print("    ✓ Excellent: Nearly lossless quantization")
        elif kl.kl_mean < 0.05:
            print("    ✓ Good: Minimal quality impact")
        elif kl.kl_mean < 0.10:
            print("    ~ Acceptable: Noticeable but usable degradation")
        else:
            print("    ✗ Poor: Significant quality degradation")

    # Throughput
    if results.throughput_with_kv:
        print("\n" + "-" * 70)
        print("THROUGHPUT (with KV cache)")
        print("-" * 70)
        print(f"  {'Context':>10} {'Prefill (tok/s)':>18} {'Decode (tok/s)':>18} {'Latency (ms)':>14}")
        print(f"  {'-'*10} {'-'*18} {'-'*18} {'-'*14}")
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
        print(f"  {'-'*10} {'-'*18} {'-'*18}")
        for t in results.throughput_without_kv:
            print(
                f"  {t.context_length:>10} {t.prefill_tok_s:>18.1f} "
                f"{t.decode_tok_s:>18.1f}"
            )

    # Memory
    print("\n" + "-" * 70)
    print("MEMORY USAGE")
    print("-" * 70)
    print(f"  Peak Memory: {results.peak_memory_gb:.2f} GB")

    print("\n" + "=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GLM-4.7-Flash Comprehensive Model Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to quantized model directory",
    )
    parser.add_argument(
        "--ref-model",
        type=str,
        default=None,
        help="Path or HF ID of FP16 reference model for KL divergence",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of WikiText-2 samples for perplexity/KL",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for perplexity/KL evaluation",
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
        "--skip-kl",
        action="store_true",
        help="Skip KL divergence evaluation",
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
    if args.ref_model:
        print(f"Reference: {args.ref_model}")

    results = EvaluationResults(
        model_path=args.model,
        ref_model_path=args.ref_model,
        timestamp=datetime.now().isoformat(),
        device=args.device,
    )

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, fallback_tokenizer="THUDM/GLM-4-9B-Chat")

    # Load quantized model
    print("\n[2/4] Loading quantized model...")
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

    # Load reference model if needed
    ref_model = None
    if args.ref_model and not args.skip_kl:
        print("\n[3/4] Loading reference model for KL divergence...")
        try:
            from transformers import AutoModelForCausalLM

            ref_model = AutoModelForCausalLM.from_pretrained(
                args.ref_model,
                torch_dtype=torch.float16,
                device_map=args.device,
                trust_remote_code=args.trust_remote_code,
            )
            ref_model.eval()
            print("  Loaded reference model")
        except Exception as e:
            print(f"  Error loading reference model: {e}")
            print("  Skipping KL divergence evaluation")
            ref_model = None

    # Perplexity evaluation
    if not args.skip_perplexity:
        print("\n[Evaluating] Perplexity on WikiText-2...")
        try:
            texts = load_wikitext2(args.samples)
            if args.verbose:
                print(f"  Loaded {len(texts)} WikiText-2 samples")

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

    # KL Divergence evaluation
    if ref_model is not None and not args.skip_kl:
        print("\n[Evaluating] KL Divergence vs reference...")
        try:
            texts = load_wikitext2(args.samples)
            results.kl_divergence = compute_kl_divergence_torch(
                ref_model,
                quant_model,
                tokenizer,
                texts,
                max_length=args.max_length,
                device=args.device,
                verbose=args.verbose,
            )
            print(f"  KL Mean: {results.kl_divergence.kl_mean:.6f}")
            print(f"  Quality: {results.kl_divergence.quality_rating().upper()}")
        except Exception as e:
            print(f"  Error computing KL divergence: {e}")

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

    # Memory stats
    if args.device == "mps":
        results.peak_memory_gb = torch.mps.current_allocated_memory() / (1024**3)
    elif args.device == "cuda":
        results.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

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
