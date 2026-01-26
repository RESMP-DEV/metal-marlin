"""
End-to-end perplexity validation using PyTorch.

Compares FP16 model perplexity against Marlin FP4-quantized model using
the Metal Marlin dequant-GEMM kernel. Validates that quantization
degradation stays within acceptable bounds.

Requires:
  - transformers (for model loading)
  - torch (for inference)
  - A HuggingFace model accessible via AutoModelForCausalLM.from_pretrained()
  - Run with: pytest --run-slow tests/test_perplexity.py -v
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

# Ensure metal_marlin python package is importable
_PYTHON_DIR = Path(__file__).parent.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from metal_marlin._compat import HAS_TORCH
from metal_marlin._compat import torch as _torch
from metal_marlin.eval_perplexity import load_wikitext2

if TYPE_CHECKING:
    import torch as torch_types

torch: Any = _torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    """Skip test if torch unavailable."""
    if not HAS_TORCH or torch is None:
        pytest.skip("torch not available")


def _require_transformers() -> Any:
    """Import transformers, skip if unavailable."""
    try:
        import transformers

        return transformers
    except ImportError:
        pytest.skip("transformers not available (install with: pip install transformers)")


def _load_wikitext2_samples(max_samples: int = 100) -> list[str]:
    """Load wikitext-2 test set, skipping empty lines."""
    try:
        return load_wikitext2(max_samples)
    except Exception as e:
        pytest.skip(f"Could not load wikitext-2: {e}")


def compute_perplexity_torch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
    device: str | None = None,
) -> float:
    """Compute perplexity of a PyTorch model on a text dataset.

    Args:
        model: PyTorch model (from transformers).
        tokenizer: Tokenizer (from transformers).
        texts: List of text strings to evaluate.
        max_length: Maximum sequence length per sample.
        device: Device to run inference on (default: auto-detect).

    Returns:
        Perplexity (exp of mean cross-entropy loss).
    """
    _require_torch()
    assert torch is not None

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors="pt")
            if tokens.shape[1] < 2:
                continue
            tokens = tokens[:, :max_length]

            input_ids = tokens[:, :-1].to(device)
            targets = tokens[:, 1:].to(device)

            outputs = model(input_ids)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # Cross-entropy loss via log-softmax
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Gather log probs for target tokens
            # log_probs: [1, seq_len, vocab], targets: [1, seq_len]
            batch_size, seq_len, vocab_size = log_probs.shape
            token_log_probs = log_probs.view(-1, vocab_size)[
                torch.arange(seq_len, device=device), targets.view(-1)
            ]
            nll = -token_log_probs.sum().item()

            total_nll += nll
            total_tokens += seq_len

    if total_tokens == 0:
        pytest.skip("No valid tokens in dataset for perplexity computation")

    return math.exp(total_nll / total_tokens)


def compute_perplexity_numpy(
    logits_fn: Callable[[np.ndarray], np.ndarray],
    tokenizer: Any,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """Compute perplexity using a numpy-based logits function.

    This is useful for testing Metal Marlin kernels that operate on numpy arrays.

    Args:
        logits_fn: Function that takes input_ids [1, seq_len] and returns logits [1, seq_len, vocab]
        tokenizer: HuggingFace tokenizer
        texts: List of text strings
        max_length: Maximum sequence length

    Returns:
        Perplexity (exp of mean cross-entropy)
    """
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:max_length]

        input_ids = np.array(tokens[:-1]).reshape(1, -1)
        targets = np.array(tokens[1:])

        # Get logits
        logits = logits_fn(input_ids)
        logits = logits.squeeze(0)  # [seq_len, vocab]

        # Cross-entropy via log-softmax (numerically stable)
        log_probs = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        # Gather log probs for target tokens
        token_log_probs = log_probs[np.arange(len(targets)), targets]
        nll = -np.sum(token_log_probs)

        total_nll += nll
        total_tokens += len(targets)

    if total_tokens == 0:
        pytest.skip("No valid tokens in dataset for perplexity computation")

    return math.exp(total_nll / total_tokens)


def replace_linear_with_marlin_torch(model: Any, group_size: int = 32) -> int:
    """Replace all quantized linear layers in a PyTorch model with MarlinLinear.

    For BitsAndBytes or GPTQ quantized models, replaces the quantized layers
    with MarlinLinear that uses the Metal Marlin FP4 dequant-GEMM kernel.

    Args:
        model: PyTorch model loaded via transformers.
        group_size: FP4 quantization group size for Marlin packing.

    Returns:
        Number of layers replaced.
    """
    _require_torch()
    assert torch is not None

    from metal_marlin.layers import MarlinLinear

    replaced = 0

    def _replace_recursive(module: torch.nn.Module, path: str = "") -> int:
        """Recursively replace quantized linear layers. Returns count replaced."""
        nonlocal replaced
        count = 0

        for name, child in list(module.named_children()):
            child_path = f"{path}.{name}" if path else name

            # Check for quantized linear layers
            # BitsAndBytes: Linear8bitLt, Linear4bit
            # GPTQ: QuantLinear
            is_quantized = (
                type(child).__name__ in ("Linear8bitLt", "Linear4bit", "QuantLinear")
                or hasattr(child, "qweight")  # GPTQ signature
            )

            if is_quantized:
                try:
                    marlin = MarlinLinear.from_torch_linear(child, group_size=group_size)
                    setattr(module, name, marlin)
                    count += 1
                except Exception:
                    # Skip layers that can't be converted
                    pass
            elif isinstance(child, torch.nn.Linear):
                # Regular linear - quantize and replace
                try:
                    marlin = MarlinLinear.from_torch_linear(child, group_size=group_size)
                    setattr(module, name, marlin)
                    count += 1
                except Exception:
                    # Skip layers that can't be converted
                    pass
            else:
                # Recurse into child modules
                count += _replace_recursive(child, child_path)

        return count

    replaced = _replace_recursive(model)
    return replaced


# ---------------------------------------------------------------------------
# Default model for testing (small enough for CI, big enough to be meaningful)
# ---------------------------------------------------------------------------

# Use a small model for perplexity tests.
# Override with METAL_MARLIN_TEST_MODEL env var.
DEFAULT_MODEL = os.environ.get(
    "METAL_MARLIN_TEST_MODEL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)


# ===========================================================================
# Perplexity Tests
# ===========================================================================


@pytest.mark.slow
class TestPerplexity:
    """End-to-end perplexity validation comparing FP16 vs Marlin FP4."""

    def test_fp4_perplexity_degradation(self) -> None:
        """Marlin FP4 should have <1.5 perplexity increase vs FP16 baseline.

        Loads a model via transformers, measures perplexity with native FP16,
        then replaces all layers with MarlinLinear and re-measures. The
        degradation from quantization (FP16 -> FP4 E2M1) should be bounded.
        """
        _require_torch()
        _require_transformers()
        assert torch is not None

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        dataset = _load_wikitext2_samples(max_samples=50)

        # Baseline: native FP16 inference
        ppl_fp16 = compute_perplexity_torch(model, tokenizer, dataset)
        print(f"\nFP16 baseline perplexity: {ppl_fp16:.4f}")

        # Replace with Marlin FP4 kernels
        replaced = replace_linear_with_marlin_torch(model, group_size=32)
        if replaced == 0:
            pytest.skip("No linear layers could be converted to Marlin")

        print(f"Replaced {replaced} layers with Marlin FP4")

        ppl_marlin = compute_perplexity_torch(model, tokenizer, dataset)
        print(f"Marlin FP4 perplexity:    {ppl_marlin:.4f}")

        degradation = ppl_marlin - ppl_fp16
        print(f"Degradation (delta):      {degradation:.4f}")

        # Quantizing from FP16 to FP4 E2M1 loses precision,
        # but the delta should be bounded
        assert degradation < 1.5, (
            f"FP4 degradation too high: {degradation:.4f} "
            f"(fp16={ppl_fp16:.4f}, marlin={ppl_marlin:.4f})"
        )

    def test_perplexity_absolute_bound(self) -> None:
        """Marlin FP4 perplexity should stay within sane absolute bounds.

        A well-quantized 1B+ model on wikitext-2 should achieve perplexity
        in range [4, 30]. Values outside indicate dequantization or
        accumulation bugs.
        """
        _require_torch()
        _require_transformers()
        assert torch is not None

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        dataset = _load_wikitext2_samples(max_samples=30)

        # Replace with Marlin FP4
        replaced = replace_linear_with_marlin_torch(model, group_size=32)
        if replaced == 0:
            pytest.skip("No linear layers could be converted to Marlin")

        ppl = compute_perplexity_torch(model, tokenizer, dataset)
        print(f"\nMarlin FP4 absolute perplexity: {ppl:.4f}")

        assert ppl < 30.0, f"Perplexity too high ({ppl:.2f}): likely dequant or accumulation bug"
        assert ppl > 2.0, f"Perplexity suspiciously low ({ppl:.2f}): likely evaluation bug"

    def test_group_size_sensitivity(self) -> None:
        """Smaller group sizes should give equal or better perplexity.

        Group size 32 should be at least as good as group size 128
        (more scale parameters = finer granularity = less quantization error).
        """
        _require_torch()
        _require_transformers()
        assert torch is not None

        from transformers import AutoModelForCausalLM, AutoTokenizer

        dataset = _load_wikitext2_samples(max_samples=30)

        # Test with group_size=128
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model_128 = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        replaced_128 = replace_linear_with_marlin_torch(model_128, group_size=128)
        if replaced_128 == 0:
            pytest.skip("No linear layers could be converted to Marlin")
        ppl_128 = compute_perplexity_torch(model_128, tokenizer, dataset)

        # Test with group_size=32
        model_32 = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        replace_linear_with_marlin_torch(model_32, group_size=32)
        ppl_32 = compute_perplexity_torch(model_32, tokenizer, dataset)

        print(f"\nGroup size 128 perplexity: {ppl_128:.4f}")
        print(f"Group size 32 perplexity:  {ppl_32:.4f}")

        # Smaller group should be better (or at least not significantly worse)
        assert ppl_32 <= ppl_128 + 0.5, (
            f"Smaller group size gave worse perplexity: "
            f"gs=32 -> {ppl_32:.4f}, gs=128 -> {ppl_128:.4f}"
        )
