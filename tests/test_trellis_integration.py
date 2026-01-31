"""Comprehensive integration test for Trellis model inference.

Tests the complete inference pipeline including:
1. Model loading from GLM-4.7-Flash-Trellis-3bpw
2. Memory usage validation (under 8GB total)
3. Numerical stability (no NaN in logits)
4. Token decoding produces readable text
5. Decode phase TPS performance (> 10 TPS)

Target hardware: Apple M1 Pro (32GB) as minimum spec.

Verify: cd contrib/metal_marlin && uv run pytest tests/test_trellis_integration.py -v
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from metal_marlin._compat import HAS_MPS

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Model path for integration tests
MODEL_PATH = "models/GLM-4.7-Flash-Trellis-3bpw"

# Memory limits (GB)
MAX_TOTAL_MEMORY_GB = 8.0  # 4GB model + 4GB KV cache
MAX_MODEL_MEMORY_GB = 5.0  # Model alone (with overhead)

# Performance thresholds
MIN_DECODE_TPS = 10.0  # Minimum tokens per second for decode phase

# Test configuration
NUM_TOKENS_TO_GENERATE = 10
TEST_PROMPT = "The capital of France is"


def get_mps_memory_gb() -> float:
    """Get current MPS allocated memory in GB."""
    if not HAS_MPS:
        return 0.0
    return torch.mps.current_allocated_memory() / (1024**3)


def get_mps_driver_memory_gb() -> float:
    """Get MPS driver allocated memory in GB."""
    if not HAS_MPS:
        return 0.0
    return torch.mps.driver_allocated_memory() / (1024**3)


def clear_mps_memory() -> None:
    """Clear MPS memory cache and run garbage collection."""
    gc.collect()
    if HAS_MPS:
        torch.mps.empty_cache()
        torch.mps.synchronize()


@pytest.fixture(scope="module")
def model_available() -> bool:
    """Check if test model is available."""
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model not found: {MODEL_PATH}")
    return True


@pytest.fixture(scope="module")
def mps_available() -> bool:
    """Check if MPS is available."""
    if not HAS_MPS:
        pytest.skip("MPS not available (requires Apple Silicon)")
    return True


@pytest.fixture(scope="module")
def trellis_model(model_available: bool, mps_available: bool):
    """Load TrellisForCausalLM model for testing.

    This fixture is module-scoped to avoid reloading for each test.
    """
    try:
        from metal_marlin.trellis.lm import TrellisForCausalLM
    except ImportError:
        pytest.skip("TrellisForCausalLM not available")

    clear_mps_memory()

    model = TrellisForCausalLM.from_pretrained(MODEL_PATH, device="mps")
    model.eval()

    yield model

    # Cleanup
    del model
    clear_mps_memory()


@pytest.fixture(scope="module")
def tokenizer(model_available: bool):
    """Load tokenizer for the model."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")

    try:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(
                "zai-org/GLM-4.7-Flash", trust_remote_code=True
            )
        except Exception:
            pytest.skip("Could not load tokenizer")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


@pytest.fixture(scope="module")
def generator(trellis_model, tokenizer):
    """Create TrellisGenerator instance."""
    try:
        from metal_marlin.trellis.generate import TrellisGenerator
    except ImportError:
        pytest.skip("TrellisGenerator not available")

    return TrellisGenerator(trellis_model, tokenizer)


@pytest.mark.requires_mps
@pytest.mark.slow
class TestTrellisIntegration:
    """Comprehensive integration tests for Trellis model inference."""

    def test_model_loads_successfully(
        self,
        model_available: bool,
        mps_available: bool,
    ):
        """Test 1: Model loads from GLM-4.7-Flash-Trellis-3bpw successfully."""
        try:
            from metal_marlin.trellis.lm import TrellisForCausalLM
        except ImportError:
            pytest.skip("TrellisForCausalLM not available")

        clear_mps_memory()
        baseline_memory = get_mps_memory_gb()

        model = TrellisForCausalLM.from_pretrained(MODEL_PATH, device="mps")

        # Verify model loaded
        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")
        assert hasattr(model, "config")

        # Verify layer count
        assert len(model.model.layers) == 47, (
            f"Expected 47 layers, got {len(model.model.layers)}"
        )

        # Verify model is on MPS
        assert next(model.parameters()).device.type == "mps"

        # Cleanup
        del model
        clear_mps_memory()

    def test_memory_under_8gb(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test 2: Memory stays under 8GB (4GB model + 4GB KV cache)."""
        try:
            from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator
        except ImportError:
            pytest.skip("TrellisGenerator not available")

        clear_mps_memory()
        torch.mps.synchronize()

        # Measure memory after model is loaded
        model_memory_gb = get_mps_memory_gb()
        driver_memory_gb = get_mps_driver_memory_gb()

        print(f"\nModel memory: {model_memory_gb:.2f} GB allocated")
        print(f"Driver memory: {driver_memory_gb:.2f} GB reserved")

        # Model alone should be under 5GB
        assert model_memory_gb < MAX_MODEL_MEMORY_GB, (
            f"Model memory {model_memory_gb:.2f} GB exceeds {MAX_MODEL_MEMORY_GB} GB limit"
        )

        # Now generate tokens to create KV cache
        generator = TrellisGenerator(trellis_model, tokenizer)
        config = GenerationConfig(
            max_new_tokens=NUM_TOKENS_TO_GENERATE,
            temperature=0.7,
            do_sample=True,
        )

        _ = generator.generate(TEST_PROMPT, config=config)
        torch.mps.synchronize()

        # Measure memory after generation (includes KV cache)
        total_memory_gb = get_mps_memory_gb()
        total_driver_gb = get_mps_driver_memory_gb()

        print(f"After generation: {total_memory_gb:.2f} GB allocated")
        print(f"Total driver memory: {total_driver_gb:.2f} GB reserved")

        # Total should stay under 8GB
        assert total_memory_gb < MAX_TOTAL_MEMORY_GB, (
            f"Total memory {total_memory_gb:.2f} GB exceeds {MAX_TOTAL_MEMORY_GB} GB limit"
        )

    def test_no_nan_in_logits(
        self,
        trellis_model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test 3: Verify no NaN in logits during forward pass."""
        # Tokenize test prompt
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("mps")
        input_ids = inputs["input_ids"]

        # Forward pass
        with torch.no_grad():
            logits = trellis_model(input_ids)

        # Check for NaN/Inf
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

        # Check logit magnitudes are reasonable
        max_logit = logits.abs().max().item()
        assert max_logit < 1000, f"Logit magnitude too high: {max_logit}"

        # Check that logits have reasonable distribution
        logit_std = logits.float().std().item()
        assert 0.1 < logit_std < 100, f"Logit std out of expected range: {logit_std}"

        print(f"\nLogits shape: {logits.shape}")
        print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        print(f"Logits std: {logit_std:.2f}")

    def test_tokens_decode_to_readable_text(
        self,
        generator,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test 4: Verify tokens decode to readable text."""
        try:
            from metal_marlin.trellis.generate import GenerationConfig
        except ImportError:
            pytest.skip("GenerationConfig not available")

        config = GenerationConfig(
            max_new_tokens=NUM_TOKENS_TO_GENERATE,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )

        output = generator.generate(TEST_PROMPT, config=config)

        # Basic sanity checks
        assert isinstance(output, str), "Output should be a string"
        assert len(output) > len(TEST_PROMPT), "Output should extend the prompt"

        # Extract generated text
        generated_part = output[len(TEST_PROMPT):].strip()
        assert len(generated_part) > 0, "No text was generated"

        print(f"\nPrompt: {TEST_PROMPT!r}")
        print(f"Generated: {generated_part!r}")
        print(f"Full output: {output!r}")

        # Check for reasonable output
        # Should contain mostly printable characters
        printable_ratio = sum(c.isprintable() or c.isspace() for c in generated_part) / len(generated_part)
        assert printable_ratio > 0.8, (
            f"Output has too many non-printable characters: {printable_ratio:.1%}"
        )

        # Check for control characters (except newlines/tabs)
        for char in generated_part:
            code = ord(char)
            if code < 32 and code not in (9, 10, 13):  # Allow \t, \n, \r
                pytest.fail(f"Output contains control character: {repr(char)}")

        # Check for repetition (sign of degenerate generation)
        words = generated_part.split()
        if len(words) >= 3:
            unique_ratio = len(set(words)) / len(words)
            assert unique_ratio > 0.3, f"Output too repetitive: {unique_ratio:.1%} unique words"

    def test_decode_tps_above_threshold(
        self,
        generator,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test 5: Verify TPS > 10 for decode phase."""
        try:
            from metal_marlin.trellis.generate import GenerationConfig
        except ImportError:
            pytest.skip("GenerationConfig not available")

        # Use greedy decoding for consistent measurement
        config = GenerationConfig(
            max_new_tokens=NUM_TOKENS_TO_GENERATE,
            temperature=0.0,
            do_sample=False,
        )

        # Warm-up run
        _ = generator.generate(TEST_PROMPT, config=config)
        torch.mps.synchronize()

        # Clear memory before timing
        clear_mps_memory()

        # Timed run
        start_time = time.perf_counter()
        output = generator.generate(TEST_PROMPT, config=config)
        torch.mps.synchronize()
        end_time = time.perf_counter()

        elapsed_seconds = end_time - start_time

        # Count tokens generated
        prompt_tokens = len(tokenizer.encode(TEST_PROMPT))
        total_tokens = len(tokenizer.encode(output))
        tokens_generated = total_tokens - prompt_tokens

        # Calculate TPS (decode phase only, excluding prefill)
        # For a fair comparison, we measure only decode tokens
        decode_tps = tokens_generated / elapsed_seconds if elapsed_seconds > 0 else 0

        print(f"\nTokens generated: {tokens_generated}")
        print(f"Time elapsed: {elapsed_seconds:.3f}s")
        print(f"Decode TPS: {decode_tps:.1f}")

        assert decode_tps > MIN_DECODE_TPS, (
            f"Decode TPS {decode_tps:.1f} below threshold of {MIN_DECODE_TPS}"
        )


@pytest.mark.requires_mps
@pytest.mark.slow
class TestTrellisIntegrationEdgeCases:
    """Edge case tests for Trellis integration."""

    def test_multiple_generations_stable_memory(
        self,
        generator,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test that multiple generations don't leak memory."""
        try:
            from metal_marlin.trellis.generate import GenerationConfig
        except ImportError:
            pytest.skip("GenerationConfig not available")

        config = GenerationConfig(
            max_new_tokens=5,
            temperature=0.7,
            do_sample=True,
        )

        # Clear and measure baseline
        clear_mps_memory()
        baseline_memory = get_mps_memory_gb()

        # Run multiple generations
        for i in range(5):
            _ = generator.generate(TEST_PROMPT, config=config)
            torch.mps.synchronize()

        # Measure memory after multiple generations
        final_memory = get_mps_memory_gb()
        memory_growth = final_memory - baseline_memory

        print(f"\nBaseline memory: {baseline_memory:.2f} GB")
        print(f"Final memory: {final_memory:.2f} GB")
        print(f"Memory growth: {memory_growth:.2f} GB")

        # Memory growth should be minimal (allow 1GB for temporary allocations)
        assert memory_growth < 1.0, (
            f"Memory grew by {memory_growth:.2f} GB after multiple generations"
        )

    def test_longer_generation_stable(
        self,
        generator,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test longer generation (50 tokens) for stability."""
        try:
            from metal_marlin.trellis.generate import GenerationConfig
        except ImportError:
            pytest.skip("GenerationConfig not available")

        config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            do_sample=True,
        )

        output = generator.generate(TEST_PROMPT, config=config)

        # Should generate substantial text
        generated_part = output[len(TEST_PROMPT):]
        assert len(generated_part) > 20, f"Expected >20 chars, got {len(generated_part)}"

        # No NaN/degradation after longer generation
        inputs = tokenizer(output, return_tensors="pt").to("mps")
        with torch.no_grad():
            logits = generator.model(inputs["input_ids"])

        assert not torch.isnan(logits).any(), "NaN detected after long generation"

    def test_greedy_deterministic(
        self,
        generator,
        tokenizer: PreTrainedTokenizer,
    ):
        """Test greedy decoding produces deterministic output."""
        try:
            from metal_marlin.trellis.generate import GenerationConfig
        except ImportError:
            pytest.skip("GenerationConfig not available")

        config = GenerationConfig(
            max_new_tokens=NUM_TOKENS_TO_GENERATE,
            temperature=0.0,
            do_sample=False,
        )

        output1 = generator.generate(TEST_PROMPT, config=config)
        output2 = generator.generate(TEST_PROMPT, config=config)

        assert output1 == output2, (
            f"Greedy decoding not deterministic:\n"
            f"  Run 1: {output1!r}\n"
            f"  Run 2: {output2!r}"
        )


@pytest.mark.requires_mps
@pytest.mark.slow
class TestTrellisModelConfig:
    """Tests for model configuration validation."""

    def test_config_matches_expected(
        self,
        model_available: bool,
        mps_available: bool,
    ):
        """Verify model config matches GLM-4.7-Flash specifications."""
        try:
            from metal_marlin.trellis.config import TrellisModelConfig
        except ImportError:
            pytest.skip("TrellisModelConfig not available")

        config = TrellisModelConfig.from_pretrained(MODEL_PATH)

        # Verify key config values for GLM-4.7-Flash
        assert config.num_hidden_layers == 47, (
            f"Expected 47 layers, got {config.num_hidden_layers}"
        )
        assert config.hidden_size == 2048, (
            f"Expected hidden_size=2048, got {config.hidden_size}"
        )
        assert config.num_attention_heads == 32, (
            f"Expected 32 attention heads, got {config.num_attention_heads}"
        )
        assert config.num_experts == 64, (
            f"Expected 64 experts, got {config.num_experts}"
        )
        assert config.num_experts_per_tok == 8, (
            f"Expected top-8 experts, got {config.num_experts_per_tok}"
        )

    def test_moe_layer_detection(
        self,
        model_available: bool,
        mps_available: bool,
    ):
        """Verify MoE layer detection is correct."""
        try:
            from metal_marlin.trellis.config import TrellisModelConfig
        except ImportError:
            pytest.skip("TrellisModelConfig not available")

        config = TrellisModelConfig.from_pretrained(MODEL_PATH)

        # Layer 0 should be dense
        assert not config.is_moe_layer(0), "Layer 0 should be dense"

        # Layers 1-46 should be MoE
        for i in range(1, 47):
            assert config.is_moe_layer(i), f"Layer {i} should be MoE"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-slow"])
