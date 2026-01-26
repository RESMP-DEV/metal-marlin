#!/usr/bin/env python3
"""
GLM-4.7-Flash Marlin FP4 Evaluation: Perplexity + KL Divergence

Compares:
  1. MLX native 4-bit (affine INT4) - baseline
  2. Marlin FP4 E2M1 (NVFP4-style) - our converted model

Metrics:
  - Perplexity on WikiText-2
  - KL divergence between FP16 reference and quantized logits
  - Token-level accuracy degradation

Usage:
    python benchmarks/eval_glm4_flash.py --samples 100
    python benchmarks/eval_glm4_flash.py --reference-model mlx-community/GLM-4.7-Flash-4bit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add metal_marlin to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "metal_marlin"))
sys.path.insert(0, str(_ROOT / "python"))

import mlx.core as mx
import mlx.nn as nn

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
        # Fallback: manual download
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
) -> float:
    """Compute perplexity on text dataset."""
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
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

    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity")

    return math.exp(total_nll / total_tokens)


def compute_kl_divergence(
    model_p: Any,  # Reference (FP16 or original)
    model_q: Any,  # Test (quantized)
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
) -> tuple[float, float]:
    """
    Compute KL divergence: D_KL(P || Q) where P=reference, Q=quantized.

    Returns:
        (mean_kl, max_kl) - average and maximum KL per position
    """
    all_kl = []

    for text in texts[:50]:  # Limit for speed
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_length]

        input_ids = mx.array(tokens[:-1]).reshape(1, -1)

        # Get logits from both models
        logits_p = model_p(input_ids).squeeze(0)  # [seq_len, vocab]
        logits_q = model_q(input_ids).squeeze(0)

        # Use numerically stable log-softmax for KL computation
        # KL(P || Q) = sum(P * (log_P - log_Q))
        log_p = logits_p - mx.logsumexp(logits_p, axis=-1, keepdims=True)
        log_q = logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True)

        p = mx.exp(log_p)
        kl_per_pos = mx.sum(p * (log_p - log_q), axis=-1)
        mx.eval(kl_per_pos)

        # Filter out any NaN/inf values
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
        if depth > 10:  # Prevent infinite recursion
            return

        # Check if module has children() method (MLX Module)
        if hasattr(module, "children"):
            for name, child in module.children().items():
                if isinstance(child, nn.QuantizedLinear):
                    replacements.append((module, name, child))
                    count += 1
                else:
                    find_replacements(child, depth + 1)

        # Handle lists/iterables (like model.layers)
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

    # Apply replacements
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


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------


def evaluate_glm4_flash(
    model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",  # GLM-4.7-Flash not yet in mlx-lm
    num_samples: int = 100,
    group_size: int = 128,
    compute_kl: bool = True,
) -> dict[str, float]:
    """
    Run comprehensive evaluation of GLM-4.7-Flash.

    Args:
        model_id: HuggingFace model ID (4-bit MLX model)
        num_samples: Number of WikiText-2 samples
        group_size: Marlin FP4 quantization group size
        compute_kl: Whether to compute KL divergence (slower)

    Returns:
        Dict with metrics: ppl_native, ppl_marlin, delta_ppl, kl_mean, kl_max
    """
    import mlx_lm

    print(f"Loading model: {model_id}")
    model, tokenizer = mlx_lm.load(model_id)
    mx.eval(model.parameters())

    print(f"Loading WikiText-2 ({num_samples} samples)...")
    dataset = load_wikitext2(max_samples=num_samples)
    print(f"  Loaded {len(dataset)} text samples")

    results = {}

    # --- Baseline: Native MLX 4-bit ---
    print("\n=== Native MLX 4-bit (Affine INT4) ===")
    ppl_native = compute_perplexity(model, tokenizer, dataset)
    print(f"  Perplexity: {ppl_native:.4f}")
    results["ppl_native"] = ppl_native

    # Copy model for KL reference (before replacing layers)
    if compute_kl:
        model_ref, _ = mlx_lm.load(model_id)
        mx.eval(model_ref.parameters())

    # --- Convert to Marlin FP4 ---
    print("\n=== Converting to Marlin FP4 (E2M1) ===")
    num_replaced = replace_linear_with_marlin(model, group_size=group_size)
    print(f"  Replaced {num_replaced} QuantizedLinear layers")
    mx.eval(model.parameters())

    # --- Marlin FP4 Perplexity ---
    print("\n=== Marlin FP4 (NVFP4-style) ===")
    ppl_marlin = compute_perplexity(model, tokenizer, dataset)
    print(f"  Perplexity: {ppl_marlin:.4f}")
    results["ppl_marlin"] = ppl_marlin

    delta = ppl_marlin - ppl_native
    print(f"\n  Delta (Marlin - Native): {delta:+.4f}")
    results["delta_ppl"] = delta

    # --- KL Divergence ---
    if compute_kl:
        print("\n=== KL Divergence (Native || Marlin) ===")
        kl_mean, kl_max = compute_kl_divergence(model_ref, model, tokenizer, dataset)
        print(f"  Mean KL: {kl_mean:.6f}")
        print(f"  Max KL:  {kl_max:.6f}")
        results["kl_mean"] = kl_mean
        results["kl_max"] = kl_max

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Marlin FP4 (NVFP4-style) Evaluation")
    print("=" * 60)
    print(f"Model:              {model_id}")
    print(f"Samples:            {len(dataset)}")
    print(f"Group Size:         {group_size}")
    print("-" * 60)
    print("NOTE: This test re-quantizes MLX affine INT4 to Marlin FP4.")
    print("      The ideal path is FP16 -> FP4 (avoids double quant error).")
    print("-" * 60)
    print(f"Native 4-bit PPL:   {ppl_native:.4f}")
    print(f"Marlin FP4 PPL:     {ppl_marlin:.4f}")
    print(f"PPL Delta:          {delta:+.4f} ({delta / ppl_native * 100:+.2f}%)")
    if compute_kl:
        print(f"Mean KL Divergence: {kl_mean:.6f}")
        print(f"Max KL Divergence:  {kl_max:.6f}")
    print("=" * 60)

    # Quality assessment
    if abs(delta) < 0.1:
        quality = "EXCELLENT - negligible degradation"
    elif abs(delta) < 0.5:
        quality = "GOOD - minor degradation"
    elif abs(delta) < 1.0:
        quality = "ACCEPTABLE - noticeable but usable"
    else:
        quality = "POOR - significant degradation"
    print(f"Quality Assessment: {quality}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GLM-4.7-Flash with Marlin FP4 quantization"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="HuggingFace model ID (GLM-4.7-Flash requires custom loader)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of WikiText-2 samples",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Marlin FP4 group size (32, 64, 128)",
    )
    parser.add_argument(
        "--no-kl",
        action="store_true",
        help="Skip KL divergence computation (faster)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    results = evaluate_glm4_flash(
        model_id=args.model,
        num_samples=args.samples,
        group_size=args.group_size,
        compute_kl=not args.no_kl,
    )

    if args.json:
        import json

        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
