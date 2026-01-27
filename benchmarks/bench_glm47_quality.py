#!/usr/bin/env python3
"""
GLM-4.7-Flash Quality Benchmark using Metal Marlin inference.

Measures perplexity, KL divergence, and RMSE for the quantized model.
Uses our MetalGLM47Model for actual quantized inference.

Usage:
    cd contrib/metal_marlin
    uv run python benchmarks/bench_glm47_quality.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from metal_marlin.eval_perplexity import (  # noqa: E402
    compute_perplexity_sliding_window,
    load_tokenizer,
    load_wikitext2,
)
from metal_marlin.inference_metal import MetalGLM47Model  # noqa: E402

# Check MPS availability
HAS_MPS = torch.backends.mps.is_available()
assert HAS_MPS, "MPS required for Metal Marlin inference"

# Model paths
MODEL_DIR = _ROOT / "benchmarks" / "results" / "glm47_sensitivity_fp8_int2"


@dataclass
class QualityResults:
    """Quality benchmark results."""

    model_path: str
    timestamp: str

    # Perplexity
    ppl_quantized: float
    ppl_tokens: int

    # KL divergence (vs BF16, if available)
    kl_mean: float
    kl_max: float
    kl_std: float
    kl_p95: float

    # RMSE (per-layer average from quantization)
    mean_rmse: float

    # Throughput
    prefill_tok_s: float
    decode_tok_s: float

    def to_json(self) -> dict:
        return asdict(self)


def measure_throughput(
    model: MetalGLM47Model,
    tokenizer,
    prompt_length: int = 512,
    gen_tokens: int = 128,
    warmup: int = 2,
    iterations: int = 3,
) -> tuple[float, float]:
    """Measure prefill and decode throughput.

    Returns:
        (prefill_tok_s, decode_tok_s)
    """
    # Create prompt
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 1
    prompt_tokens = [bos_id] + [100] * (prompt_length - 1)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device="mps")

    # Warmup
    print(f"  Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        kv_cache = model.create_kv_cache()
        _ = model(input_ids[:, :64], kv_cache=kv_cache)
        torch.mps.synchronize()

    gc.collect()
    torch.mps.empty_cache()

    # Measure prefill
    print("  Measuring prefill...")
    prefill_times = []
    for _ in range(iterations):
        gc.collect()
        kv_cache = model.create_kv_cache()
        start = time.perf_counter()
        logits = model(input_ids, kv_cache=kv_cache)
        torch.mps.synchronize()
        prefill_times.append(time.perf_counter() - start)

    avg_prefill = np.mean(prefill_times)
    prefill_tok_s = prompt_length / avg_prefill

    # Measure decode
    print(f"  Measuring decode ({gen_tokens} tokens)...")
    decode_times = []
    for _ in range(iterations):
        gc.collect()
        kv_cache = model.create_kv_cache()
        _ = model(input_ids, kv_cache=kv_cache)
        kv_cache.advance(prompt_length)

        start = time.perf_counter()
        next_token = torch.argmax(logits[:, -1:, :], dim=-1)
        for _ in range(gen_tokens - 1):
            logits = model(next_token, kv_cache=kv_cache)
            kv_cache.advance(1)
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
        torch.mps.synchronize()
        decode_times.append(time.perf_counter() - start)

    avg_decode = np.mean(decode_times)
    decode_tok_s = gen_tokens / avg_decode

    return float(prefill_tok_s), float(decode_tok_s)


def main():
    print("=" * 70)
    print("GLM-4.7-Flash Quality Benchmark (Metal Marlin)")
    print("=" * 70)

    # Check model exists
    if not MODEL_DIR.exists():
        print(f"ERROR: Model not found at {MODEL_DIR}")
        return 1

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(MODEL_DIR)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Load quantized model
    print("\n[2/5] Loading quantized model...")
    start = time.perf_counter()
    model = MetalGLM47Model.from_quantized(MODEL_DIR, bits=4)
    load_time = time.perf_counter() - start
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Layers: {model.num_layers}")
    print(f"  Hidden size: {model.hidden_size}")

    # Load WikiText-2
    print("\n[3/5] Loading WikiText-2...")
    texts = load_wikitext2(max_samples=100)
    print(f"  Loaded {len(texts)} samples")

    # Concatenate texts (llama.cpp style)
    full_text = "\n".join(texts)
    print(f"  Total chars: {len(full_text):,}")

    # Compute perplexity
    print("\n[4/5] Computing perplexity...")

    def logits_fn(input_ids_np):
        """Wrapper for perplexity computation."""
        gc.collect()
        torch.mps.empty_cache()
        input_ids = torch.tensor(input_ids_np, dtype=torch.long, device="mps")
        kv_cache = model.create_kv_cache()
        with torch.no_grad():
            logits = model(input_ids, kv_cache=kv_cache)
            result = logits.cpu().float().numpy()
        del logits, kv_cache, input_ids
        gc.collect()
        torch.mps.empty_cache()
        return result

    # Use smaller context for better memory management with MoE model
    ppl, ppl_tokens = compute_perplexity_sliding_window(
        logits_fn=logits_fn,
        tokenizer=tokenizer,
        text=full_text,
        context_length=512,  # Smaller context for MoE memory
        stride=256,
        verbose=True,
    )
    print(f"  Perplexity: {ppl:.4f} on {ppl_tokens} tokens")

    # Measure throughput
    print("\n[5/5] Measuring throughput...")
    prefill_tok_s, decode_tok_s = measure_throughput(model, tokenizer)
    print(f"  Prefill: {prefill_tok_s:.1f} tok/s")
    print(f"  Decode: {decode_tok_s:.1f} tok/s")

    # Compute RMSE from model metadata (if available)
    meta_file = MODEL_DIR / "quantization_meta.json"
    mean_rmse = 0.0
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        mean_rmse = meta.get("mean_rmse", 0.0)
        print(f"\n  RMSE (from quantization): {mean_rmse:.6f}")

    # Build results
    results = QualityResults(
        model_path=str(MODEL_DIR),
        timestamp=datetime.now().isoformat(),
        ppl_quantized=ppl,
        ppl_tokens=ppl_tokens,
        kl_mean=0.0,  # Would need BF16 reference to compute
        kl_max=0.0,
        kl_std=0.0,
        kl_p95=0.0,
        mean_rmse=mean_rmse,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"\n  Model: {MODEL_DIR.name}")
    print("\n  Quality:")
    print(f"    Perplexity: {ppl:.4f} on {ppl_tokens} tokens")
    if mean_rmse > 0:
        print(f"    Mean RMSE: {mean_rmse:.6f}")
    print("\n  Throughput:")
    print(f"    Prefill: {prefill_tok_s:.1f} tok/s")
    print(f"    Decode: {decode_tok_s:.1f} tok/s")

    # Save results
    results_file = MODEL_DIR / "quality_results.json"
    with open(results_file, "w") as f:
        json.dump(results.to_json(), f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
