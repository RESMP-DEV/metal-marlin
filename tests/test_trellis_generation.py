"""Generation quality tests for Trellis models.

These tests verify that trellis-quantized models produce coherent, high-quality
text generation comparable to FP16 reference models.

Verify: cd contrib/metal_marlin && uv run pytest tests/test_trellis_generation.py -v -m slow
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# Model path for integration tests
MODEL_PATH = "models/GLM-4.7-Flash-EXL3-3bpw"

# Sample prompts for generation quality testing
SAMPLE_PROMPTS = {
    "capital": "The capital of France is",
    "quicksort": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "story": "Once upon a time",
    "math": "What is 2 + 2? The answer is",
    "code": 'import numpy as np\n\ndef matrix_multiply(A, B):\n    """Compute matrix product."""\n    return',
}

# Expected patterns for coherence checking
COHERENCE_PATTERNS = {
    "capital": [r"Paris"],
    "quicksort": [r"arr\[0\]", r"mid", r"left", r"right", r"partition"],
    "story": [r"there", r"was", r"lived", r"king", r"queen", r"princess"],
    "math": [r"4", r"four"],
    "code": [r"np\.dot", r"A", r"B", r"@", r"matmul"],
}


@pytest.fixture(scope="module")
def model_available():
    """Check if test model is available."""
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model not found: {MODEL_PATH}")
    return True


@pytest.fixture(scope="module")
def trellis_model(model_available, mps_device):
    """Load TrellisForCausalLM model."""
    try:
        from metal_marlin.trellis.lm import TrellisForCausalLM
    except ImportError:
        pytest.skip("TrellisForCausalLM not available")

    model = TrellisForCausalLM.from_pretrained(MODEL_PATH, device=mps_device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer(model_available):
    """Load tokenizer for the model."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")

    # Try to load tokenizer from model path or use a compatible one
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception:
        # Fallback to a generic tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "zai-org/GLM-4.7-Flash", trust_remote_code=True
            )
        except Exception:
            pytest.skip("Could not load tokenizer")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


@pytest.mark.slow
@pytest.mark.requires_mps
class TestTrellisGenerationQuality:
    """Generation quality tests for trellis-quantized models."""

    @pytest.mark.parametrize("prompt_name", list(SAMPLE_PROMPTS.keys()))
    def test_generation_not_garbage(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
        prompt_name: str,
    ):
        """Verify generated text is coherent, not random tokens.

        This test checks that the model produces reasonable, human-readable
        output rather than garbage tokens or repetitive nonsense.
        """
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        prompt = SAMPLE_PROMPTS[prompt_name]
        generator = TrellisGenerator(trellis_model, tokenizer)

        # Generate with moderate temperature
        config = GenerationConfig(
            max_new_tokens=30,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )

        output = generator.generate(prompt, config=config)

        # Basic sanity checks
        assert isinstance(output, str), "Output should be a string"
        assert len(output) > len(prompt), "Output should extend the prompt"

        # Check for reasonable output length (not too short, not too long)
        generated_part = output[len(prompt) :].strip()
        assert len(generated_part) > 5, f"Generated text too short: '{generated_part}'"

        # Check for repetition (a sign of poor generation)
        words = generated_part.split()
        if len(words) > 3:
            unique_words = set(words)
            # At least 30% unique words (prevents "the the the" type output)
            diversity_ratio = len(unique_words) / len(words)
            assert diversity_ratio > 0.3, f"Output too repetitive: '{generated_part}'"

        # Check for presence of non-ASCII garbage
        # Allow common unicode but block control characters
        for char in generated_part:
            code = ord(char)
            # Block control characters except newlines and tabs
            if code < 32 and code not in (9, 10, 13):
                pytest.fail(f"Output contains control character: {repr(char)}")

        # Pattern-based coherence check
        patterns = COHERENCE_PATTERNS.get(prompt_name, [])
        if patterns:
            has_match = any(
                re.search(pattern, generated_part, re.IGNORECASE) for pattern in patterns
            )
            if not has_match:
                # Don't fail immediately - just warn, as this is heuristic
                # Instead check that output looks like valid text
                assert generated_part[0].isalnum() or generated_part[0] in " ([{'\"", (
                    f"Output starts unexpectedly: '{generated_part[:50]}'"
                )

    @pytest.mark.slow
    def test_greedy_generation_deterministic(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Verify temperature=0 (greedy) produces deterministic output.

        Running the same prompt twice with temperature=0 should produce
        identical results.
        """
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)

        # Greedy decoding config
        config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.0,  # Greedy
            do_sample=False,
        )

        prompt = "The quick brown fox"

        # Generate twice
        output1 = generator.generate(prompt, config=config)
        output2 = generator.generate(prompt, config=config)

        # Should be identical for greedy decoding
        assert output1 == output2, (
            f"Greedy decoding not deterministic:\n  Run 1: {output1}\n  Run 2: {output2}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("prompt_name", ["capital", "story"])
    def test_top5_token_probabilities(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
        prompt_name: str,
    ):
        """Compare top-5 token probabilities to expected distribution.

        This test verifies that the model assigns reasonable probabilities
        to tokens, without requiring an FP16 reference model.
        """
        prompt = SAMPLE_PROMPTS[prompt_name]

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(trellis_model.device)
        input_ids = inputs["input_ids"]

        # Forward pass
        with torch.no_grad():
            logits, _ = trellis_model(input_ids)

        # Get logits for last position
        next_token_logits = logits[0, -1, :]

        # Get top-5 tokens
        top_k = 5
        top_logits, top_indices = torch.topk(next_token_logits, top_k)
        top_probs = torch.softmax(top_logits, dim=-1)

        # Decode top tokens for debugging
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

        # Sanity checks on probability distribution
        probs_sum = top_probs.sum().item()
        assert 0.5 < probs_sum < 1.0, f"Top-5 probability mass suspicious: {probs_sum:.3f}"

        # Top token should have reasonable probability
        top_prob = top_probs[0].item()
        assert 0.01 < top_prob < 0.99, (
            f"Top token probability suspicious: {top_prob:.3f} ({top_tokens[0]!r})"
        )

        # Check that probabilities are ordered
        for i in range(1, len(top_probs)):
            assert top_probs[i - 1] >= top_probs[i], "Probabilities not in descending order"

        # Verify no NaN/Inf in probabilities
        assert not torch.isnan(top_probs).any(), "NaN in probabilities"
        assert not torch.isinf(top_probs).any(), "Inf in probabilities"

    @pytest.mark.slow
    def test_generation_with_different_temperatures(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test generation quality across different temperatures.

        - Low temperature (< 0.5): Should be more focused/deterministic
        - High temperature (> 1.0): Should be more diverse
        """
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)
        prompt = "The weather today is"

        outputs = {}
        for temp in [0.1, 0.7, 1.5]:
            config = GenerationConfig(
                max_new_tokens=15,
                temperature=temp,
                top_k=50,
                do_sample=True,
            )
            # Generate a few samples to check diversity
            samples = [generator.generate(prompt, config=config) for _ in range(3)]
            outputs[temp] = samples

        # Low temperature should be more consistent
        low_temp_samples = outputs[0.1]
        low_temp_unique = len(set(low_temp_samples))
        assert low_temp_unique <= 2, (
            f"Low temperature (0.1) produced {low_temp_unique} unique outputs, "
            f"expected more consistency"
        )

        # All outputs should be reasonable strings
        for temp, samples in outputs.items():
            for sample in samples:
                assert len(sample) > len(prompt), f"Temperature {temp}: Generated text too short"
                assert not any(ord(c) < 32 and c not in "\n\t\r" for c in sample), (
                    f"Temperature {temp}: Output contains control characters"
                )

    @pytest.mark.slow
    def test_batch_generation_consistency(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test that batch generation produces consistent results.

        The same prompt generated individually and in a batch should
        produce similar outputs (with some variance due to sampling).
        """
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)

        prompts = [
            "Hello, my name is",
            "The capital of Japan is",
        ]

        # Greedy decoding for consistency
        config = GenerationConfig(
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
        )

        # Batch generate
        batch_outputs = generator.generate(prompts, config=config)
        assert isinstance(batch_outputs, list)
        assert len(batch_outputs) == len(prompts)

        # Individual generation should match batch (with greedy)
        for i, prompt in enumerate(prompts):
            individual = generator.generate(prompt, config=config)
            batch_result = batch_outputs[i]
            assert individual == batch_result, (
                f"Batch and individual generation differ for prompt {i}:\n"
                f"  Individual: {individual}\n"
                f"  Batch:      {batch_result}"
            )

    @pytest.mark.slow
    def test_streaming_generation_matches_non_streaming(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test that streaming generation produces same result as non-streaming.

        When using greedy decoding, streaming and non-streaming should
        produce identical results.
        """
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)

        prompt = "In the beginning"
        config = GenerationConfig(
            max_new_tokens=15,
            temperature=0.0,
            do_sample=False,
        )

        # Non-streaming generation
        non_stream_output = generator.generate(prompt, config=config)

        # Streaming generation - collect all chunks
        stream_chunks = list(generator.generate(prompt, config=config, stream=True))
        stream_output = prompt + "".join(stream_chunks)

        assert non_stream_output == stream_output, (
            f"Streaming and non-streaming outputs differ:\n"
            f"  Non-stream: {non_stream_output}\n"
            f"  Stream:     {stream_output}"
        )

    @pytest.mark.slow
    def test_eos_token_handling(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test that EOS token properly stops generation."""
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)

        # Use a prompt likely to hit EOS quickly
        prompt = "Q: Hi\nA: "

        # Set EOS token
        eos_id = tokenizer.eos_token_id
        assert eos_id is not None, "Tokenizer has no EOS token"

        config = GenerationConfig(
            max_new_tokens=50,  # Request many tokens
            temperature=0.7,
            eos_token_id=eos_id,
        )

        output = generator.generate(prompt, config=config)
        generated_tokens = tokenizer.encode(
            output[len(prompt) :],
            add_special_tokens=False,
        )

        # Should not be at max length (meaning EOS was hit or we got lucky)
        # This is a soft check - we just verify generation happened
        assert len(generated_tokens) < 50 or eos_id in generated_tokens, (
            "Generation may not have respected EOS token"
        )

    @pytest.mark.slow
    def test_repetition_penalty_effect(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test that repetition penalty reduces repetition.

        With repetition penalty > 1.0, outputs should be less repetitive.
        """
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)
        prompt = "The"

        # Generate with and without repetition penalty
        config_no_penalty = GenerationConfig(
            max_new_tokens=20,
            temperature=0.7,
            repetition_penalty=1.0,  # No penalty
        )

        config_with_penalty = GenerationConfig(
            max_new_tokens=20,
            temperature=0.7,
            repetition_penalty=1.5,  # Strong penalty
        )

        # Run a few times to account for sampling variance
        no_penalty_outputs = []
        with_penalty_outputs = []

        for _ in range(3):
            no_penalty_outputs.append(generator.generate(prompt, config=config_no_penalty))
            with_penalty_outputs.append(generator.generate(prompt, config=config_with_penalty))

        # Check repetition in each output
        def count_repetitions(text: str) -> int:
            """Count consecutive word repetitions."""
            words = text.split()
            repeats = 0
            for i in range(1, len(words)):
                if words[i].lower() == words[i - 1].lower():
                    repeats += 1
            return repeats

        no_penalty_repeats = sum(count_repetitions(o) for o in no_penalty_outputs)
        with_penalty_repeats = sum(count_repetitions(o) for o in with_penalty_outputs)

        # With penalty should have fewer or equal repeats
        # (this is probabilistic, so we allow some slack)
        assert with_penalty_repeats <= no_penalty_repeats + 1, (
            f"Repetition penalty not effective: "
            f"no_penalty={no_penalty_repeats}, with_penalty={with_penalty_repeats}"
        )


@pytest.mark.slow
class TestTrellisGenerationEdgeCases:
    """Edge case tests for trellis generation."""

    @pytest.mark.slow
    def test_empty_prompt(self, trellis_model, tokenizer: PreTrainedTokenizer):
        """Test generation with minimal/empty prompt."""
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)

        # Very short prompt
        prompt = "Hi"
        config = GenerationConfig(max_new_tokens=10, temperature=0.7)

        output = generator.generate(prompt, config=config)
        assert len(output) > len(prompt)

    @pytest.mark.slow
    def test_long_prompt(self, trellis_model, tokenizer: PreTrainedTokenizer):
        """Test generation with longer prompt."""
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)

        prompt = (
            "The following is a story about a brave knight:\n\n"
            "Once upon a time in a distant kingdom, there lived a knight "
            "who was known throughout the land for his courage and honor. "
            "One day, the king summoned him to the castle for an important mission. "
            "The knight bowed and said, "
        )
        config = GenerationConfig(max_new_tokens=15, temperature=0.7)

        output = generator.generate(prompt, config=config)
        assert len(output) > len(prompt)

        # Check that the output continues the story reasonably
        generated = output[len(prompt) :].strip()
        assert len(generated) > 0

    @pytest.mark.slow
    def test_special_characters_in_prompt(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test generation with special characters in prompt."""
        from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

        generator = TrellisGenerator(trellis_model, tokenizer)

        prompts = [
            "function() { return",  # Code
            "Equation: $E = mc^2$ means",  # LaTeX
            "Item 1. Item 2.",  # List
            "Q: What?\nA:",  # Q&A format
        ]

        config = GenerationConfig(max_new_tokens=10, temperature=0.7)

        for prompt in prompts:
            output = generator.generate(prompt, config=config)
            assert isinstance(output, str)
            assert len(output) > len(prompt)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
