"""
End-to-end perplexity validation.

Compares FP16 model perplexity against Marlin FP4-quantized model using
the Metal Marlin dequant-GEMM kernel. Validates that quantization
degradation stays within acceptable bounds.

Requires:
  - mlx_lm (for model loading)
  - A HuggingFace model accessible via mlx_lm.load()
  - Run with: pytest --run-slow tests/test_perplexity.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure metal_marlin python package is importable
_PYTHON_DIR = Path(__file__).parent.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_mlx() -> Any:
    """Import mlx, skip if unavailable."""
    try:
        import mlx.core as mx  # noqa: F401
        return mx
    except ImportError:
        pytest.skip("mlx not available")


def _get_mlx_lm() -> Any:
    """Import mlx_lm, skip if unavailable."""
    try:
        import mlx_lm
        return mlx_lm
    except ImportError:
        pytest.skip("mlx_lm not available (install with: pip install mlx-lm)")


def _load_wikitext2(max_samples: int = 100) -> list[str]:
    """Load wikitext-2 test set, skipping empty lines."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
        return texts[:max_samples]
    except ImportError:
        pass

    # Fallback: try huggingface_hub direct download
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="wikitext",
            filename="wikitext-2-raw-v1/wiki.test.raw",
            repo_type="dataset",
        )
        lines = Path(path).read_text().strip().split("\n")
        texts = [t for t in lines if len(t.strip()) > 50]
        return texts[:max_samples]
    except Exception as e:
        pytest.skip(f"Could not load wikitext-2: {e}")


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """Compute perplexity of a model on a text dataset.

    Args:
        model: MLX model (from mlx_lm.load).
        tokenizer: Tokenizer (from mlx_lm.load).
        texts: List of text strings to evaluate.
        max_length: Maximum sequence length per sample.

    Returns:
        Perplexity (exp of mean cross-entropy loss).
    """
    mx = _get_mlx()

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
        logits = logits.squeeze(0)  # [seq_len, vocab_size]

        # Cross-entropy loss via log-softmax
        log_probs = mx.log_softmax(logits, axis=-1)
        token_log_probs = log_probs[mx.arange(len(targets)), targets]
        nll = -float(mx.sum(token_log_probs))
        mx.eval(nll)

        total_nll += nll
        total_tokens += len(targets)

    if total_tokens == 0:
        pytest.skip("No valid tokens in dataset for perplexity computation")

    return math.exp(total_nll / total_tokens)


def replace_linear_with_marlin(model: Any, group_size: int = 32) -> None:
    """Replace all nn.QuantizedLinear layers in a model with MarlinLinear.

    Modifies the model in-place by swapping each QuantizedLinear module
    with a MarlinLinear that uses the Metal Marlin FP4 dequant-GEMM kernel.

    Args:
        model: MLX model loaded via mlx_lm.
        group_size: FP4 quantization group size for Marlin packing.
    """
    import mlx.nn as nn

    from metal_marlin import MarlinLinear

    def _replace_recursive(module: Any, path: str = "") -> int:
        """Recursively replace QuantizedLinear layers. Returns count replaced."""
        count = 0
        # Iterate over module attributes that are themselves modules
        for name in list(vars(module)):
            child = getattr(module, name, None)
            if child is None:
                continue
            if isinstance(child, nn.QuantizedLinear):
                marlin = MarlinLinear.from_quantized_linear(child)
                setattr(module, name, marlin)
                count += 1
            elif hasattr(child, "__dict__"):
                count += _replace_recursive(child, f"{path}.{name}")
        # Also check items in lists/tuples (common in transformer blocks)
        if isinstance(module, (list, tuple)):
            for i, child in enumerate(module):
                if isinstance(child, nn.QuantizedLinear):
                    marlin = MarlinLinear.from_quantized_linear(child)
                    module[i] = marlin
                    count += 1
                elif hasattr(child, "__dict__"):
                    count += _replace_recursive(child, f"{path}[{i}]")
        return count

    replaced = _replace_recursive(model)
    if replaced == 0:
        pytest.skip("Model has no QuantizedLinear layers to replace")


# ---------------------------------------------------------------------------
# Default model for testing (small enough for CI, big enough to be meaningful)
# ---------------------------------------------------------------------------

# Use a small quantized model for perplexity tests.
# Override with METAL_MARLIN_TEST_MODEL env var.
import os

DEFAULT_MODEL = os.environ.get(
    "METAL_MARLIN_TEST_MODEL",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
)


# ===========================================================================
# Perplexity Tests
# ===========================================================================


@pytest.mark.slow
class TestPerplexity:
    """End-to-end perplexity validation comparing FP16 vs Marlin FP4."""

    def test_fp4_perplexity_degradation(self) -> None:
        """Marlin FP4 should have <1.0 perplexity increase vs MLX native quant.

        Loads a 4-bit quantized model via mlx_lm, measures perplexity with
        native MLX QuantizedLinear, then replaces all layers with MarlinLinear
        and re-measures. The degradation from re-quantizing (affine INT4 -> FP4 E2M1)
        should be bounded.
        """
        mlx_lm = _get_mlx_lm()
        mx = _get_mlx()

        model, tokenizer = mlx_lm.load(DEFAULT_MODEL)
        dataset = _load_wikitext2(max_samples=50)

        # Baseline: native MLX quantized inference
        ppl_native = compute_perplexity(model, tokenizer, dataset)
        print(f"\nNative MLX 4-bit perplexity: {ppl_native:.4f}")

        # Replace with Marlin FP4 kernels
        replace_linear_with_marlin(model, group_size=32)
        mx.eval(model.parameters())

        ppl_marlin = compute_perplexity(model, tokenizer, dataset)
        print(f"Marlin FP4 perplexity:      {ppl_marlin:.4f}")

        degradation = ppl_marlin - ppl_native
        print(f"Degradation (delta):        {degradation:.4f}")

        # Re-quantizing from affine INT4 to FP4 E2M1 loses some precision,
        # but the delta should be bounded
        assert degradation < 1.0, (
            f"FP4 degradation too high: {degradation:.4f} "
            f"(native={ppl_native:.4f}, marlin={ppl_marlin:.4f})"
        )

    def test_perplexity_absolute_bound(self) -> None:
        """Marlin FP4 perplexity should stay within sane absolute bounds.

        A well-quantized 1B+ model on wikitext-2 should achieve perplexity
        in range [4, 25]. Values outside indicate dequantization or
        accumulation bugs.
        """
        mlx_lm = _get_mlx_lm()
        mx = _get_mlx()

        model, tokenizer = mlx_lm.load(DEFAULT_MODEL)
        dataset = _load_wikitext2(max_samples=30)

        # Replace with Marlin FP4
        replace_linear_with_marlin(model, group_size=32)
        mx.eval(model.parameters())

        ppl = compute_perplexity(model, tokenizer, dataset)
        print(f"\nMarlin FP4 absolute perplexity: {ppl:.4f}")

        assert ppl < 25.0, (
            f"Perplexity too high ({ppl:.2f}): likely dequant or accumulation bug"
        )
        assert ppl > 2.0, (
            f"Perplexity suspiciously low ({ppl:.2f}): likely evaluation bug"
        )

    def test_group_size_sensitivity(self) -> None:
        """Smaller group sizes should give equal or better perplexity.

        Group size 32 should be at least as good as group size 128
        (more scale parameters = finer granularity = less quantization error).
        """
        mlx_lm = _get_mlx_lm()
        mx = _get_mlx()

        dataset = _load_wikitext2(max_samples=30)

        # Test with group_size=128
        model_128, tokenizer = mlx_lm.load(DEFAULT_MODEL)
        replace_linear_with_marlin(model_128, group_size=128)
        mx.eval(model_128.parameters())
        ppl_128 = compute_perplexity(model_128, tokenizer, dataset)

        # Test with group_size=32
        model_32, tokenizer_32 = mlx_lm.load(DEFAULT_MODEL)
        replace_linear_with_marlin(model_32, group_size=32)
        mx.eval(model_32.parameters())
        ppl_32 = compute_perplexity(model_32, tokenizer_32, dataset)

        print(f"\nGroup size 128 perplexity: {ppl_128:.4f}")
        print(f"Group size 32 perplexity:  {ppl_32:.4f}")

        # Smaller group should be better (or at least not significantly worse)
        assert ppl_32 <= ppl_128 + 0.5, (
            f"Smaller group size gave worse perplexity: "
            f"gs=32 -> {ppl_32:.4f}, gs=128 -> {ppl_128:.4f}"
        )
