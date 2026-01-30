"""
Evaluation module for metal_marlin models.

This module provides comprehensive evaluation utilities for quantized language models:
- perplexity: PPL measurement with llama.cpp compatibility
- accuracy: Quality metrics including KL divergence

Example usage:
    from metal_marlin.eval import compute_perplexity_wikitext, evaluate_kl_divergence

    # Perplexity evaluation
    ppl_result = compute_perplexity_wikitext(logits_fn, tokenizer)

    # KL divergence (accuracy) evaluation
    kl_result = evaluate_kl_divergence(original_fn, quantized_fn, tokenizer, texts)
"""

# Re-export all public functions from submodules
from .accuracy import (
    KLResult,
    compute_kl_divergence_np,
    evaluate_kl_divergence,
    evaluate_kl_from_paths,
)
from .perplexity import (
    compute_kl_divergence,
    compute_perplexity_from_logits,
    compute_perplexity_sliding_window,
    compute_perplexity_wikitext,
    load_tokenizer,
    load_wikitext2,
    log_softmax,
    softmax,
)

__all__ = [
    # Tokenizer and data loading
    "load_tokenizer",
    "load_wikitext2",
    # Perplexity functions
    "compute_perplexity_from_logits",
    "compute_perplexity_sliding_window",
    "compute_perplexity_wikitext",
    "compute_kl_divergence",  # Simple KL from perplexity module
    # Accuracy/KL divergence functions
    "KLResult",
    "compute_kl_divergence_np",
    "evaluate_kl_divergence",
    "evaluate_kl_from_paths",
    # Utility functions
    "softmax",
    "log_softmax",
]
