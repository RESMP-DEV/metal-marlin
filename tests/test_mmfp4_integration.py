"""MMFP4 integration tests for GLM-4.7-Flash-Marlin-MMFP4.

Covers:
1. Full model loading from models/GLM-4.7-Flash-Marlin-MMFP4
2. Single forward pass correctness
3. Generation quality via perplexity benchmark
4. Long-context handling (4K, 8K tokens)
5. Batch inference

Verify:
  cd contrib/metal_marlin && uv run pytest tests/test_mmfp4_integration.py -v --collect-only
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "GLM-4.7-Flash-Marlin-MMFP4"
PERPLEXITY_TEXTS = [
    "The capital of France is Paris and it is known for art and history.",
    "Quantum computing uses qubits and interference to solve specific problems efficiently.",
    "AlphaHENG coordinates many agents in parallel to execute independent coding tasks.",
]
MAX_ACCEPTABLE_PERPLEXITY = 300.0

if not HAS_TORCH or torch is None:
    _TORCH_MISSING = True
else:
    _TORCH_MISSING = False

_MPS_MISSING = not HAS_MPS
_MODEL_MISSING = not MODEL_DIR.exists()

from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline

pytestmark = [
    pytest.mark.requires_mps,
    pytest.mark.slow,
    pytest.mark.skipif(_TORCH_MISSING, reason="PyTorch is required for MMFP4 integration tests."),
    pytest.mark.skipif(_MPS_MISSING, reason="MPS backend required for MMFP4 integration tests."),
    pytest.mark.skipif(_MODEL_MISSING, reason=f"Model not found: {MODEL_DIR}"),
]


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "insufficient memory" in message


def _build_long_input_ids(tokenizer: Any, seq_len: int, device: str) -> Any:
    seed_text = "Long context handling for MMFP4 integration tests."
    token_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not token_ids:
        token_ids = [1]
    repeats = (seq_len // len(token_ids)) + 1
    token_ids = (token_ids * repeats)[:seq_len]
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def _compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: str,
    max_length: int = 512,
) -> float:
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if input_ids.shape[1] < 2:
            continue

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[:, :-1, :]

        targets = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(targets.numel())

    if total_tokens == 0:
        raise AssertionError("No valid tokens available for perplexity computation.")

    return math.exp(total_nll / total_tokens)


@pytest.fixture(scope="module")
def mmfp4_pipeline() -> MMFP4Pipeline:
    pytest.importorskip("transformers")
    return MMFP4Pipeline.from_pretrained(str(MODEL_DIR), device="mps")


@pytest.fixture(scope="module")
def model(mmfp4_pipeline: MMFP4Pipeline) -> Any:
    return mmfp4_pipeline.model


@pytest.fixture(scope="module")
def tokenizer(mmfp4_pipeline: MMFP4Pipeline) -> Any:
    tokenizer = mmfp4_pipeline.tokenizer
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def model_device(model: Any) -> str:
    return str(next(model.parameters()).device)


class TestMMFP4Integration:
    def test_01_full_model_loading(self, mmfp4_pipeline: MMFP4Pipeline) -> None:
        """Test 1: Full model loading from models/GLM-4.7-Flash-Marlin-MMFP4."""
        assert mmfp4_pipeline.model is not None
        assert mmfp4_pipeline.tokenizer is not None
        assert hasattr(mmfp4_pipeline.model, "generate")
        assert hasattr(mmfp4_pipeline.model, "config")

        assert getattr(mmfp4_pipeline.model.config, "model_type", None) == "glm4_moe_lite"

        index_path = MODEL_DIR / "model.safetensors.index.json"
        assert index_path.exists(), f"Missing shard index: {index_path}"
        index_data = json.loads(index_path.read_text())
        assert index_data.get("metadata", {}).get("format") == "mmfp4"

    def test_02_single_forward_pass_correctness(
        self,
        model: Any,
        tokenizer: Any,
        model_device: str,
    ) -> None:
        """Test 2: Single forward pass correctness."""
        encoded = tokenizer("The capital of France is", return_tensors="pt")
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        logits = outputs.logits
        assert logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
        assert torch.isfinite(logits).all(), "Forward pass produced non-finite logits."

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
            )
        expected_next = int(logits[0, -1].argmax().item())
        generated_next = int(generated[0, -1].item())
        assert generated_next == expected_next

    def test_03_generation_quality_perplexity_benchmark(
        self,
        model: Any,
        tokenizer: Any,
        model_device: str,
    ) -> None:
        """Test 3: Generation quality (perplexity benchmark)."""
        perplexity = _compute_perplexity(
            model,
            tokenizer,
            PERPLEXITY_TEXTS,
            device=model_device,
            max_length=512,
        )

        assert math.isfinite(perplexity)
        assert perplexity > 1.0
        assert perplexity < MAX_ACCEPTABLE_PERPLEXITY, (
            f"Perplexity {perplexity:.2f} exceeded threshold {MAX_ACCEPTABLE_PERPLEXITY:.2f}"
        )

    @pytest.mark.parametrize("seq_len", [4096, 8192], ids=["4k", "8k"])
    def test_04_long_context_handling(
        self,
        model: Any,
        tokenizer: Any,
        model_device: str,
        seq_len: int,
    ) -> None:
        """Test 4: Long context handling (4K, 8K tokens)."""
        input_ids = _build_long_input_ids(tokenizer, seq_len, model_device)
        attention_mask = torch.ones_like(input_ids)

        try:
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                )
        except RuntimeError as exc:
            if _is_oom_error(exc):
                if model_device.startswith("mps"):
                    torch.mps.empty_cache()
                pytest.skip(f"Insufficient memory for seq_len={seq_len}: {exc}")
            raise

        assert generated.shape[0] == 1
        assert generated.shape[1] == seq_len + 1

    def test_05_batch_inference(
        self,
        model: Any,
        tokenizer: Any,
        model_device: str,
    ) -> None:
        """Test 5: Batch inference."""
        prompts = [
            "Explain what a cache is.",
            "What is the capital of Japan?",
            "Write one sentence about machine learning.",
            "Summarize the purpose of integration tests.",
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        logits = outputs.logits
        assert logits.shape[:2] == input_ids.shape
        assert logits.shape[2] == model.config.vocab_size
        assert torch.isfinite(logits[:, -1, :]).all(), "Batch inference produced non-finite logits."
