"""
Standalone perplexity evaluation with llama.cpp compatibility.

Computes perplexity without MLX-LM dependency. Uses HuggingFace tokenizers
directly and our own Metal kernels for inference.

Provides two perplexity computation methods:
  1. compute_perplexity_from_logits() - Simple per-sample computation
  2. compute_perplexity_sliding_window() - llama.cpp-compatible sliding window

The sliding window method matches llama.cpp's perplexity command:
  - Concatenates all text with newlines
  - Uses sliding window with stride (default: context_length // 2)
  - Only scores tokens in non-overlapping portion (avoids boundary effects)
  - Handles BOS token prepending

For comparing against llama.cpp benchmarks, use compute_perplexity_wikitext()
which wraps the sliding window method with WikiText-2 loading.

Usage:
    python -m metal_marlin.eval_perplexity ./model-fp4/ --samples 100

Example (programmatic):
    from metal_marlin.eval_perplexity import compute_perplexity_wikitext
    result = compute_perplexity_wikitext(
        logits_fn=model.forward,
        tokenizer=tokenizer,
        context_length=2048,
        verbose=True,
    )
    print(f"PPL: {result['perplexity']:.2f} on {result['n_tokens']} tokens")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def load_tokenizer(model_path: str | Path) -> Any:
    """
    Load tokenizer from model directory.

    Supports:
    - tokenizer.json (fast tokenizer)
    - tokenizer.model (sentencepiece)
    """
    from transformers import AutoTokenizer

    model_path = Path(model_path)

    # Try to load from local path
    if model_path.exists():
        return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # Otherwise treat as HF model ID
    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def load_wikitext2(max_samples: int | None = None) -> list[str]:
    """Load WikiText-2 test set.

    For calibration, use ALL samples (max_samples=None) to get accurate
    activation ranges. For evaluation/perplexity, a subset is acceptable
    for faster results.

    Args:
        max_samples: Maximum samples to return, or None for ALL samples.

    Returns:
        List of text samples from WikiText-2 test set.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
        if max_samples is not None:
            texts = texts[:max_samples]
        return texts
    except ImportError:
        # Fallback: direct download
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-2-raw-v1/wiki.test.raw",
            repo_type="dataset",
        )
        lines = Path(path).read_text().strip().split("\n")
        texts = [t for t in lines if len(t.strip()) > 50]
        if max_samples is not None:
            texts = texts[:max_samples]
        return texts


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def compute_perplexity_from_logits(
    logits_fn,  # Callable[[np.ndarray], np.ndarray] - input_ids -> logits
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
    verbose: bool = False,
) -> float:
    """
    Compute perplexity given a function that produces logits.

    Args:
        logits_fn: Function that takes input_ids [1, seq_len] and returns logits [1, seq_len, vocab]
        tokenizer: HuggingFace tokenizer
        texts: List of text strings
        max_length: Maximum sequence length
        verbose: Print per-sample stats

    Returns:
        Perplexity (exp of mean cross-entropy)
    """
    total_nll = 0.0
    total_tokens = 0

    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:max_length]

        input_ids = np.array(tokens[:-1]).reshape(1, -1)
        targets = np.array(tokens[1:])

        # Get logits
        logits = logits_fn(input_ids)
        logits = logits.squeeze(0)  # [seq_len, vocab]

        # Cross-entropy via log-softmax
        log_probs = log_softmax(logits, axis=-1)

        # Gather log probs for target tokens
        token_log_probs = log_probs[np.arange(len(targets)), targets]
        nll = -np.sum(token_log_probs)

        total_nll += nll
        total_tokens += len(targets)

        if verbose and (i + 1) % 10 == 0:
            ppl_so_far = math.exp(total_nll / total_tokens)
            print(f"  [{i + 1}/{len(texts)}] Running PPL: {ppl_so_far:.4f}")

    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity computation")

    return math.exp(total_nll / total_tokens)


def compute_perplexity_sliding_window(
    logits_fn,  # Callable[[np.ndarray], np.ndarray] - input_ids -> logits
    tokenizer: Any,
    text: str,
    context_length: int = 512,
    stride: int | None = None,
    add_bos: bool = True,
    verbose: bool = False,
) -> tuple[float, int]:
    """
    Compute perplexity using llama.cpp-compatible sliding window methodology.

    This matches the perplexity measurement used by llama.cpp's `perplexity` command:
    - Sliding window with configurable stride (default: context_length // 2)
    - Only score tokens in the latter half of each window (avoids boundary effects)
    - BOS token handling matches llama.cpp convention

    Args:
        logits_fn: Function that takes input_ids [1, seq_len] and returns logits [1, seq_len, vocab]
        tokenizer: HuggingFace tokenizer
        text: Full text to evaluate (typically concatenated WikiText-2 or similar)
        context_length: Maximum context window size (model's max_position_embeddings)
        stride: Step size between windows (default: context_length // 2)
        add_bos: Whether to prepend BOS token (llama.cpp default: True)
        verbose: Print per-window stats

    Returns:
        (perplexity, n_tokens_scored): Perplexity and count of tokens scored

    Reference:
        llama.cpp/examples/perplexity/perplexity.cpp - hellaswag_score() and perplexity()
    """
    if stride is None:
        stride = context_length // 2
    if stride <= 0 or stride > context_length:
        raise ValueError(f"Stride must be in [1, {context_length}], got {stride}")

    # Tokenize full text
    tokens = tokenizer.encode(text)

    # Optionally prepend BOS
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if add_bos and bos_token_id is not None and (len(tokens) == 0 or tokens[0] != bos_token_id):
        tokens = [bos_token_id] + tokens

    tokens = np.array(tokens, dtype=np.int64)
    n_tokens = len(tokens)

    if n_tokens < 2:
        raise ValueError("Text too short for perplexity computation")

    total_nll = 0.0
    total_tokens_scored = 0
    n_windows = 0

    # Sliding window over the entire sequence
    start = 0
    while start < n_tokens - 1:
        end = min(start + context_length, n_tokens)
        window_tokens = tokens[start:end]

        # Get logits for this window
        input_ids = window_tokens[:-1].reshape(1, -1)
        targets = window_tokens[1:]

        logits = logits_fn(input_ids)
        logits = logits.squeeze(0)  # [seq_len, vocab]

        # Log-softmax for cross-entropy
        log_probs = log_softmax(logits, axis=-1)

        # llama.cpp only scores tokens in the latter part of the window
        # to avoid boundary effects. For the first window, score all tokens.
        # For subsequent windows, only score tokens beyond the overlap.
        if start == 0:
            # First window: score all tokens
            score_start = 0
        else:
            # Subsequent windows: only score the non-overlapping portion
            # This is tokens beyond (context_length - stride) from window start
            score_start = context_length - stride

        score_end = len(targets)

        if score_start < score_end:
            scored_targets = targets[score_start:score_end]
            scored_log_probs = log_probs[score_start:score_end]

            token_log_probs = scored_log_probs[np.arange(len(scored_targets)), scored_targets]
            window_nll = -np.sum(token_log_probs)

            total_nll += window_nll
            total_tokens_scored += len(scored_targets)

        n_windows += 1
        if verbose and n_windows % 10 == 0:
            ppl_so_far = math.exp(total_nll / total_tokens_scored) if total_tokens_scored > 0 else float("inf")
            print(f"  Window {n_windows}: pos {start}-{end}, scored {total_tokens_scored} tokens, PPL: {ppl_so_far:.4f}")

        # Move window by stride
        start += stride

        # Stop if we've processed all tokens
        if end >= n_tokens:
            break

    if total_tokens_scored == 0:
        raise ValueError("No tokens scored for perplexity computation")

    perplexity = math.exp(total_nll / total_tokens_scored)
    return perplexity, total_tokens_scored


def compute_perplexity_wikitext(
    logits_fn,  # Callable[[np.ndarray], np.ndarray]
    tokenizer: Any,
    max_samples: int | None = 100,
    context_length: int = 512,
    stride: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Compute perplexity on WikiText-2 using llama.cpp-compatible methodology.

    This is the recommended function for comparing against llama.cpp benchmarks.

    Args:
        logits_fn: Model's forward function (input_ids -> logits)
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum WikiText-2 samples to use (None for all)
        context_length: Context window size
        stride: Sliding window stride (default: context_length // 2)
        verbose: Print progress

    Returns:
        Dict with 'perplexity', 'n_tokens', 'context_length', 'stride'
    """
    texts = load_wikitext2(max_samples)

    # Concatenate texts with newlines (matches llama.cpp preprocessing)
    full_text = "\n\n".join(texts)

    if verbose:
        print(f"Loaded {len(texts)} WikiText-2 samples")
        print(f"Total characters: {len(full_text)}")
        print(f"Context length: {context_length}, Stride: {stride or context_length // 2}")

    ppl, n_tokens = compute_perplexity_sliding_window(
        logits_fn=logits_fn,
        tokenizer=tokenizer,
        text=full_text,
        context_length=context_length,
        stride=stride,
        add_bos=True,
        verbose=verbose,
    )

    return {
        "perplexity": ppl,
        "n_tokens": n_tokens,
        "context_length": context_length,
        "stride": stride or context_length // 2,
        "n_samples": len(texts),
    }


def compute_kl_divergence(
    logits_fn_p,  # Reference model
    logits_fn_q,  # Test model
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
) -> tuple[float, float]:
    """
    Compute KL divergence D_KL(P || Q).

    Returns:
        (mean_kl, max_kl) across all positions
    """
    all_kl = []

    for text in texts[:50]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_length]

        input_ids = np.array(tokens[:-1]).reshape(1, -1)

        logits_p = logits_fn_p(input_ids).squeeze(0)
        logits_q = logits_fn_q(input_ids).squeeze(0)

        # Use log-softmax for numerical stability
        log_p = log_softmax(logits_p, axis=-1)
        log_q = log_softmax(logits_q, axis=-1)

        # KL = sum(P * (log_P - log_Q))
        p = np.exp(log_p)
        kl_per_pos = np.sum(p * (log_p - log_q), axis=-1)

        valid_kl = kl_per_pos[np.isfinite(kl_per_pos)]
        if len(valid_kl) > 0:
            all_kl.extend(valid_kl.tolist())

    if not all_kl:
        return 0.0, 0.0

    return float(np.mean(all_kl)), float(np.max(all_kl))


# ============================================================================
# Model inference (numpy reference implementation)
# ============================================================================


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def rope_embed(x: np.ndarray, positions: np.ndarray, theta: float = 10000.0) -> np.ndarray:
    """Apply rotary position embeddings."""
    seq_len, head_dim = x.shape[-2], x.shape[-1]

    # Compute frequencies
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

    # Position-dependent angles
    angles = positions[:, None] * freqs[None, :]  # [seq_len, head_dim/2]

    # Split x into pairs and apply rotation
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = np.cos(angles)
    sin = np.sin(angles)

    # Rotate
    x_rot = np.empty_like(x)
    x_rot[..., ::2] = x1 * cos - x2 * sin
    x_rot[..., 1::2] = x1 * sin + x2 * cos

    return x_rot


def attention_forward(
    x: np.ndarray,  # [batch, seq_len, hidden]
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    o_weight: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
    positions: np.ndarray | None = None,
    rope_theta: float = 10000.0,
) -> np.ndarray:
    """Multi-head attention with RoPE."""
    batch, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    kv_head_dim = hidden // num_kv_heads

    # Project Q, K, V
    q = x @ q_weight.T  # [batch, seq_len, hidden]
    k = x @ k_weight.T  # [batch, seq_len, kv_hidden]
    v = x @ v_weight.T

    # Reshape to heads
    q = q.reshape(batch, seq_len, num_heads, head_dim)
    k = k.reshape(batch, seq_len, num_kv_heads, head_dim)
    v = v.reshape(batch, seq_len, num_kv_heads, head_dim)

    # Apply RoPE
    if positions is None:
        positions = np.arange(seq_len)

    for h in range(num_heads):
        q[:, :, h, :] = rope_embed(q[:, :, h, :], positions, rope_theta)
    for h in range(num_kv_heads):
        k[:, :, h, :] = rope_embed(k[:, :, h, :], positions, rope_theta)

    # GQA: repeat KV heads
    if num_kv_heads != num_heads:
        repeats = num_heads // num_kv_heads
        k = np.repeat(k, repeats, axis=2)
        v = np.repeat(v, repeats, axis=2)

    # Attention scores
    # [batch, heads, seq, seq]
    q_t = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
    k_t = k.transpose(0, 2, 3, 1)  # [batch, heads, head_dim, seq]

    scores = q_t @ k_t / np.sqrt(head_dim)

    # Causal mask
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    scores = scores + mask

    # Softmax
    attn = softmax(scores, axis=-1)

    # Apply to values
    v_t = v.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
    out = attn @ v_t  # [batch, heads, seq, head_dim]

    # Reshape back
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden)

    # Output projection
    return out @ o_weight.T


def mlp_forward(
    x: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    activation: str = "silu",
) -> np.ndarray:
    """SwiGLU MLP: down(silu(gate(x)) * up(x))"""
    gate = x @ gate_weight.T
    up = x @ up_weight.T

    if activation == "silu":
        gate = gate * (1 / (1 + np.exp(-gate)))  # silu = x * sigmoid(x)
    elif activation == "gelu":
        gate = 0.5 * gate * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3)))

    return (gate * up) @ down_weight.T


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--samples", type=int, default=100, help="WikiText-2 samples")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = load_tokenizer(args.model_path)

    print(f"Loading WikiText-2 ({args.samples} samples)...")
    texts = load_wikitext2(args.samples)
    print(f"  Loaded {len(texts)} samples")

    # For now, just demonstrate the API
    print("\nNote: Full inference requires Metal kernel integration.")
    print("This module provides the perplexity computation framework.")
    print("\nTo use with your model:")
    print("  1. Load quantized weights with hf_loader.load_quantized_weights()")
    print("  2. Implement forward pass using Metal Marlin kernels")
    print("  3. Pass logits function to compute_perplexity_from_logits()")


if __name__ == "__main__":
    main()
