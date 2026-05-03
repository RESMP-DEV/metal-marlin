"""Tests for trellis streaming generation.

Verify: Test streaming output with a simple prompt
"""

from collections.abc import Iterator
import logging

import pytest

# Import at module level to catch import errors
try:
    from metal_marlin.trellis.generate import GenerationConfig, TrellisGenerator

    HAS_TRELLIS_GENERATE = True
except ImportError:
    HAS_TRELLIS_GENERATE = False



logger = logging.getLogger(__name__)

@pytest.mark.skipif(not HAS_TRELLIS_GENERATE, reason="trellis_generate not available")
class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_values(self):
        """Test GenerationConfig default values."""
        logger.info("running test_default_values")
        config = GenerationConfig()
        assert config.max_new_tokens == 256
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.0
        assert config.do_sample is True
        assert config.eos_token_id is None
        assert config.pad_token_id is None

    def test_custom_values(self):
        """Test GenerationConfig accepts custom values."""
        logger.info("running test_custom_values")
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.95,
            top_k=10,
            eos_token_id=2,
            pad_token_id=0,
        )
        assert config.max_new_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.top_k == 10
        assert config.eos_token_id == 2
        assert config.pad_token_id == 0


@pytest.mark.skipif(not HAS_TRELLIS_GENERATE, reason="trellis_generate not available")
class TestTrellisGenerator:
    """Test TrellisGenerator class."""

    def test_class_exists(self):
        """Test TrellisGenerator class is importable."""
        logger.info("running test_class_exists")
        assert TrellisGenerator is not None

    def test_has_stream_generate(self):
        """Test stream_generate method exists."""
        logger.info("running test_has_stream_generate")
        assert hasattr(TrellisGenerator, "stream_generate")

    def test_has_stream_generate_tokens(self):
        """Test stream_generate_tokens method exists."""
        logger.info("running test_has_stream_generate_tokens")
        assert hasattr(TrellisGenerator, "stream_generate_tokens")

    def test_has_generate(self):
        """Test generate method exists."""
        logger.info("running test_has_generate")
        assert hasattr(TrellisGenerator, "generate")

    def test_stream_generate_is_iterator(self):
        """Test stream_generate returns an Iterator type hint."""
        logger.info("running test_stream_generate_is_iterator")
        import inspect

        sig = inspect.signature(TrellisGenerator.stream_generate)
        # Check return annotation (works with string annotations)
        annotation = sig.return_annotation
        assert "Iterator" in str(annotation) and "str" in str(annotation)

    def test_stream_generate_tokens_is_iterator(self):
        """Test stream_generate_tokens returns an Iterator type hint."""
        logger.info("running test_stream_generate_tokens_is_iterator")
        import inspect

        sig = inspect.signature(TrellisGenerator.stream_generate_tokens)
        # Check return annotation contains Iterator and tuple
        annotation = str(sig.return_annotation)
        assert "Iterator" in annotation and "tuple" in annotation


def test_module_exports():
    """Test module exports."""
    logger.info("running test_module_exports")
    from metal_marlin.trellis import GenerationConfig, TrellisGenerator
    assert TrellisGenerator is not None
    assert GenerationConfig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
