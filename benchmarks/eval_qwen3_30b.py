#!/usr/bin/env python3
"""
Qwen3-30B-A3B MoE Quality Benchmark

Compares quantized model quality against BF16 reference using
Transformers integration (not MLX).

Usage:
    uv run python benchmarks/eval_qwen3_30b.py --samples 50
    uv run python benchmarks/eval_qwen3_30b.py --full  # 100 samples
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure metal_marlin is importable from the project layout.
_ROOT = Path(__file__).resolve().parents[1]
_METAL_MARLIN_ROOT = _ROOT / "contrib" / "metal_marlin"
sys.path.insert(0, str(_METAL_MARLIN_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from metal_marlin.benchmarks.quality import QualityMetrics, compare_models  # noqa: E402
from metal_marlin.layer_replacement import replace_linear_layers  # noqa: E402


def _sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _build_prompt_tokens(tokenizer, prompt_length: int, device: str) -> torch.Tensor:
    if prompt_length < 1:
        raise ValueError("prompt_length must be >= 1")
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 1
    prompt_tokens = [bos_id] + [100] * (prompt_length - 1)
    return torch.tensor([prompt_tokens], dtype=torch.long, device=device)


def measure_throughput(
    model,
    tokenizer,
    device: str,
    *,
    prompt_length: int = 512,
    gen_tokens: int = 128,
    warmup: int = 1,
    iterations: int = 3,
) -> tuple[float, float]:
    """Measure prefill and decode throughput (tokens/sec)."""
    input_ids = _build_prompt_tokens(tokenizer, prompt_length, device)

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, use_cache=True)
            _sync_device(device)

    # Prefill throughput
    prefill_times: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            _sync_device(device)
            start = time.perf_counter()
            _ = model(input_ids, use_cache=True)
            _sync_device(device)
            prefill_times.append(time.perf_counter() - start)

    prefill_tok_s = prompt_length / max(sum(prefill_times) / len(prefill_times), 1e-6)

    # Decode throughput
    decode_times: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            outputs = model(input_ids, use_cache=True)
            past = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            _sync_device(device)
            start = time.perf_counter()
            for _ in range(gen_tokens):
                outputs = model(next_token, use_cache=True, past_key_values=past)
                past = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            _sync_device(device)
            decode_times.append(time.perf_counter() - start)

    decode_tok_s = gen_tokens / max(sum(decode_times) / len(decode_times), 1e-6)

    return float(prefill_tok_s), float(decode_tok_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-30B-A3B quality benchmark.")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate.")
    parser.add_argument("--full", action="store_true", help="Use 100 samples (slower).")
    parser.add_argument("--max-length", type=int, default=256, help="Max length per sample.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model ID.",
    )
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits.")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size.")
    parser.add_argument("--format", type=str, default="fp4", help="Quantization format.")
    parser.add_argument("--prompt-length", type=int, default=512, help="Prompt length for throughput.")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Decode tokens for throughput.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "benchmarks" / "results" / "qwen3_30b_a3b_quality.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for Transformers.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend required for Metal Marlin quantized inference.")

    device = "mps"
    samples = 100 if args.full else args.samples

    print("=" * 70)
    print("Qwen3-30B-A3B MoE Quality Benchmark (Transformers)")
    print("=" * 70)

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

    print("\n[2/5] Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=args.trust_remote_code,
    )

    print("\n[3/5] Loading quantized model...")
    quant_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=args.trust_remote_code,
    )

    print("  Replacing linear layers with MetalQuantizedLinear...")
    stats = replace_linear_layers(
        quant_model,
        bits=args.bits,
        group_size=args.group_size,
        format=args.format,
    )
    print(f"  Replaced {stats['replaced_count']} layers (skipped {stats['skipped_count']}).")

    print("\n[4/5] Running quality comparison...")
    metrics: QualityMetrics = compare_models(
        ref_model,
        quant_model,
        tokenizer,
        num_samples=samples,
        max_length=args.max_length,
        verbose=True,
    )

    print("\n[5/5] Benchmarking throughput...")
    prefill_ref, decode_ref = measure_throughput(
        ref_model,
        tokenizer,
        device,
        prompt_length=args.prompt_length,
        gen_tokens=args.gen_tokens,
    )
    prefill_quant, decode_quant = measure_throughput(
        quant_model,
        tokenizer,
        device,
        prompt_length=args.prompt_length,
        gen_tokens=args.gen_tokens,
    )

    print("\nResults Summary")
    print("-" * 70)
    print(f"Perplexity: {metrics.perplexity_ref:.2f} -> {metrics.perplexity_quant:.2f}")
    print(f"  Delta: {metrics.perplexity_delta_pct:+.1f}%")
    print(f"KL Divergence: mean={metrics.kl_divergence_mean:.4f}")
    print(f"Mean RMSE: {metrics.mean_rmse:.4f}")
    print(
        f"Throughput (prefill): ref={prefill_ref:.1f} tok/s, quant={prefill_quant:.1f} tok/s"
    )
    print(
        f"Throughput (decode):  ref={decode_ref:.1f} tok/s, quant={decode_quant:.1f} tok/s"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model_id": args.model_id,
        "timestamp": datetime.now().isoformat(),
        "samples": samples,
        "max_length": args.max_length,
        "quantization": {
            "bits": args.bits,
            "group_size": args.group_size,
            "format": args.format,
            "replaced_count": stats.get("replaced_count", 0),
            "skipped_count": stats.get("skipped_count", 0),
        },
        "perplexity_ref": metrics.perplexity_ref,
        "perplexity_quant": metrics.perplexity_quant,
        "perplexity_delta_pct": metrics.perplexity_delta_pct,
        "kl_divergence": {
            "mean": metrics.kl_divergence_mean,
            "max": metrics.kl_divergence_max,
            "std": metrics.kl_divergence_std,
            "p95": metrics.kl_divergence_p95,
        },
        "rmse": {
            "mean": metrics.mean_rmse,
            "layers": metrics.layer_rmse,
        },
        "throughput": {
            "prefill_tok_s": {"ref": prefill_ref, "quant": prefill_quant},
            "decode_tok_s": {"ref": decode_ref, "quant": decode_quant},
            "prompt_length": args.prompt_length,
            "gen_tokens": args.gen_tokens,
        },
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
