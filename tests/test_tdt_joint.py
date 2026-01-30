"""Tests for TDT (Transducer Dynamic Temperature) ASR components.

Tests for the TDT joint network that combines encoder and predictor outputs
in RNN-T/Transducer models for speech recognition.
"""

import pytest

from metal_marlin._compat import HAS_TORCH

# Import PyTorch modules only after skip check
if HAS_TORCH:
    import torch

    from metal_marlin.asr.config import TDTConfig
    from metal_marlin.asr.tdt_joint import TDTJoint

# Skip entire module if PyTorch unavailable
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")


class TestTDTJoint:
    """Test cases for TDTJoint network."""

    def test_init_with_default_config(self) -> None:
        """Test TDTJoint initialization with default configuration."""
        config = TDTConfig()
        model = TDTJoint(config)

        assert model.encoder_proj.in_features == config.encoder_hidden_size
        assert model.encoder_proj.out_features == config.joint_hidden_size
        assert model.predictor_proj.in_features == config.predictor_hidden_size
        assert model.predictor_proj.out_features == config.joint_hidden_size
        assert model.output_linear.in_features == config.joint_hidden_size
        assert model.output_linear.out_features == config.vocab_size
        assert model.duration_head.in_features == config.joint_hidden_size
        assert model.duration_head.out_features == config.max_duration + 1

    def test_init_with_custom_config(self) -> None:
        """Test TDTJoint initialization with custom configuration."""
        config = TDTConfig(
            vocab_size=500,
            encoder_hidden_size=256,
            predictor_hidden_size=128,
            joint_hidden_size=192,
            max_duration=75,
        )
        model = TDTJoint(config)

        assert model.encoder_proj.in_features == 256
        assert model.encoder_proj.out_features == 192
        assert model.predictor_proj.in_features == 128
        assert model.predictor_proj.out_features == 192
        assert model.output_linear.in_features == 192
        assert model.output_linear.out_features == 500
        assert model.duration_head.in_features == 192
        assert model.duration_head.out_features == 76  # max_duration + 1

    def test_forward_pass_shapes(self) -> None:
        """Test forward pass output shapes."""
        config = TDTConfig(vocab_size=100, max_duration=20)
        model = TDTJoint(config)

        batch_size, seq_len, pred_len = 2, 15, 8
        encoder_out = torch.randn(batch_size, seq_len, config.encoder_hidden_size)
        predictor_out = torch.randn(batch_size, pred_len, config.predictor_hidden_size)

        logits, durations = model(encoder_out, predictor_out)

        # Expected shapes: [B, T, U, vocab_size] and [B, T, U, max_duration+1]
        expected_logits_shape = (batch_size, seq_len, pred_len, config.vocab_size)
        expected_durations_shape = (batch_size, seq_len, pred_len, config.max_duration + 1)

        assert logits.shape == expected_logits_shape
        assert durations.shape == expected_durations_shape

    def test_forward_pass_values(self) -> None:
        """Test forward pass produces reasonable values."""
        config = TDTConfig(vocab_size=10, max_duration=5)
        model = TDTJoint(config)

        encoder_out = torch.zeros(1, 1, config.encoder_hidden_size)
        predictor_out = torch.zeros(1, 1, config.predictor_hidden_size)

        logits, durations = model(encoder_out, predictor_out)

        # Check that outputs are finite and not all zeros
        assert torch.isfinite(logits).all()
        assert torch.isfinite(durations).all()
        assert not (logits == 0).all()
        assert not (durations == 0).all()

    def test_backward_pass(self) -> None:
        """Test that gradients flow correctly through the network."""
        config = TDTConfig(vocab_size=50, max_duration=10)
        model = TDTJoint(config)

        encoder_out = torch.randn(2, 5, config.encoder_hidden_size, requires_grad=True)
        predictor_out = torch.randn(2, 3, config.predictor_hidden_size, requires_grad=True)

        logits, durations = model(encoder_out, predictor_out)

        # Create dummy loss
        logits_loss = logits.sum()
        durations_loss = durations.sum()
        total_loss = logits_loss + durations_loss

        total_loss.backward()

        # Check that gradients exist for inputs
        assert encoder_out.grad is not None
        assert predictor_out.grad is not None
        assert not (encoder_out.grad == 0).all()
        assert not (predictor_out.grad == 0).all()
