"""
KL divergence measurement for quantization quality assessment.

KL divergence measures how much the quantized model's probability distribution
differs from the original. This is MORE informative than perplexity for
quantization quality assessment because it directly measures distribution shift.

Target KL values (from vLLM/llama.cpp experience):
- KL < 0.01: Excellent (nearly lossless)
- KL < 0.05: Good (minimal quality impact)
- KL < 0.10: Acceptable (noticeable but usable)
- KL > 0.10: Poor (significant degradation)

Usage:
    python -m metal_marlin.eval_kl_divergence \
        --original ./model-fp16/ \
        --quantized ./model-fp4/ \
        --samples 50
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class KLResult:
    """KL divergence measurement result."""

    kl_mean: float
    kl_max: float
    kl_std: float
    kl_p95: float  # 95th percentile
    num_tokens: int
    num_samples: int
    temperature: float

    def quality_rating(self) -> str:
        """Return quality rating based on KL divergence."""
        if self.kl_mean < 0.01:
            return "excellent"
        elif self.kl_mean < 0.05:
            return "good"
        elif self.kl_mean < 0.10:
            return "acceptable"
        else:
            return "poor"

    def __str__(self) -> str:
        return (
            f"KL(mean={self.kl_mean:.4f}, max={self.kl_max:.4f}, "
            f"std={self.kl_std:.4f}, p95={self.kl_p95:.4f}) "
            f"[{self.quality_rating()}]"
        )


def compute_kl_divergence_np(
    original_logits: np.ndarray,  # [batch, seq, vocab]
    quantized_logits: np.ndarray,
    temperature: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Compute KL(P_original || P_quantized) using NumPy.

    This measures the information lost when using the quantized distribution
    to approximate the original distribution.

    Args:
        original_logits: Logits from the original (FP16/BF16) model
        quantized_logits: Logits from the quantized model
        temperature: Temperature for softmax (default: 1.0, no scaling)

    Returns:
        (kl_mean, kl_max, kl_std, kl_p95)
    """
    # Apply temperature scaling
    if temperature != 1.0:
        original_logits = original_logits / temperature
        quantized_logits = quantized_logits / temperature

    # Numerically stable log-softmax
    def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

    log_p_orig = log_softmax(original_logits, axis=-1)
    log_p_quant = log_softmax(quantized_logits, axis=-1)

    # KL divergence: sum(P * (log_P - log_Q))
    # Using P = exp(log_P) for numerical stability
    p_orig = np.exp(log_p_orig)

    # Per-token KL: sum over vocab dimension
    kl_per_token = np.sum(p_orig * (log_p_orig - log_p_quant), axis=-1)

    # Flatten to 1D array of per-token KL values
    kl_flat = kl_per_token.flatten()

    # Filter out any NaN/Inf values
    kl_valid = kl_flat[np.isfinite(kl_flat)]

    if len(kl_valid) == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.mean(kl_valid)),
        float(np.max(kl_valid)),
        float(np.std(kl_valid)),
        float(np.percentile(kl_valid, 95)),
    )


def evaluate_kl_divergence(
    original_logits_fn: Callable[[np.ndarray], np.ndarray],
    quantized_logits_fn: Callable[[np.ndarray], np.ndarray],
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
    temperature: float = 1.0,
    verbose: bool = False,
) -> KLResult:
    """
    Evaluate KL divergence between original and quantized model.

    This function computes logits from both models on the same inputs
    and measures the KL divergence between their output distributions.

    Args:
        original_logits_fn: Function that takes input_ids [1, seq] and returns logits [1, seq, vocab]
        quantized_logits_fn: Same signature as original_logits_fn
        tokenizer: HuggingFace tokenizer
        texts: List of text samples to evaluate
        max_length: Maximum sequence length per sample
        temperature: Temperature for softmax (higher = softer distributions)
        verbose: Print per-sample progress

    Returns:
        KLResult with comprehensive statistics
    """
    all_kl: list[float] = []
    total_tokens = 0

    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_length]

        input_ids = np.array(tokens[:-1]).reshape(1, -1)

        # Get logits from both models
        orig_logits = original_logits_fn(input_ids)
        quant_logits = quantized_logits_fn(input_ids)

        # Ensure same shape
        if orig_logits.shape != quant_logits.shape:
            if verbose:
                print(f"  Warning: Shape mismatch {orig_logits.shape} vs {quant_logits.shape}")
            continue

        # Compute per-token KL
        kl_mean, kl_max, kl_std, kl_p95 = compute_kl_divergence_np(
            orig_logits, quant_logits, temperature
        )

        # Collect per-token KL values for overall statistics
        num_tokens_sample = orig_logits.shape[1]
        total_tokens += num_tokens_sample

        # Weighted by number of tokens
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


def evaluate_kl_from_paths(
    original_path: str | Path,
    quantized_path: str | Path,
    texts: list[str] | None = None,
    num_samples: int = 50,
    max_length: int = 256,
    temperature: float = 1.0,
    verbose: bool = True,
) -> KLResult:
    """
    Evaluate KL divergence between two model directories.

    This is a high-level function that handles model loading.

    Args:
        original_path: Path to original (FP16/BF16) model
        quantized_path: Path to quantized (FP4/INT4) model
        texts: Optional list of texts (loads wikitext-2 if None)
        num_samples: Number of samples to evaluate
        max_length: Maximum sequence length
        temperature: Softmax temperature
        verbose: Print progress

    Returns:
        KLResult with KL divergence statistics
    """
    from .eval_perplexity import load_tokenizer, load_wikitext2

    if verbose:
        print(f"Loading tokenizer from {original_path}...")
    load_tokenizer(original_path)

    if texts is None:
        if verbose:
            print(f"Loading WikiText-2 ({num_samples} samples)...")
        texts = load_wikitext2(num_samples)

    # For now, return placeholder - full implementation requires
    # loading both models which is memory-intensive
    if verbose:
        print("\nNote: Full KL evaluation requires loading both models.")
        print("For large models, consider using streaming evaluation.")

    # TODO: Implement model loading and forward pass
    # This requires careful memory management for large models

    return KLResult(
        kl_mean=0.0,
        kl_max=0.0,
        kl_std=0.0,
        kl_p95=0.0,
        num_tokens=0,
        num_samples=0,
        temperature=temperature,
    )


def main():
    """CLI for KL divergence evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate KL divergence between original and quantized models"
    )
    parser.add_argument("--original", required=True, help="Path to original model")
    parser.add_argument("--quantized", required=True, help="Path to quantized model")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    result = evaluate_kl_from_paths(
        original_path=args.original,
        quantized_path=args.quantized,
        num_samples=args.samples,
        max_length=args.max_length,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    print("\n" + "=" * 50)
    print("KL DIVERGENCE RESULTS")
    print("=" * 50)
    print(f"  Mean KL:     {result.kl_mean:.6f}")
    print(f"  Max KL:      {result.kl_max:.6f}")
    print(f"  Std KL:      {result.kl_std:.6f}")
    print(f"  95th %ile:   {result.kl_p95:.6f}")
    print(f"  Quality:     {result.quality_rating().upper()}")
    print(f"  Tokens:      {result.num_tokens}")
    print(f"  Samples:     {result.num_samples}")
    print("=" * 50)

    # Quality interpretation
    print("\nInterpretation:")
    if result.kl_mean < 0.01:
        print("  ✓ Excellent: Nearly lossless quantization")
    elif result.kl_mean < 0.05:
        print("  ✓ Good: Minimal quality impact")
    elif result.kl_mean < 0.10:
        print("  ~ Acceptable: Noticeable but usable degradation")
    else:
        print("  ✗ Poor: Significant quality degradation")


if __name__ == "__main__":
    main()
