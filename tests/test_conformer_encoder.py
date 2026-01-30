"""Tests for ConformerEncoder module.

Tests the ConformerEncoder implementation including:
- Forward pass shape validation
- MPS device compatibility
- Subsampling behavior
"""

import pytest
import torch

from metal_marlin.asr.conformer_config import ConformerConfig
from metal_marlin.asr.conformer_encoder import ConformerEncoder


class TestConformerEncoder:
    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        config = ConformerConfig(num_layers=2)
        encoder = ConformerEncoder(config)

        # Input mel spectrogram: 2 batches, 1000 frames, 80 mel channels
        mel = torch.randn(2, 1000, 80)  # ~10s of audio at 100Hz frame rate
        lengths = torch.tensor([1000, 800])  # Different sequence lengths

        # Forward pass
        out, out_lengths = encoder(mel, lengths)

        # Check output shape
        # Expected: (batch_size, seq_len//4, hidden_size)
        expected_seq_len = 1000 // 4  # 4x subsampling
        assert out.shape == (2, expected_seq_len, config.hidden_size)

        # Check output lengths (should be subsampled by 4)
        expected_lengths = torch.tensor([250, 200])
        assert torch.equal(out_lengths, expected_lengths)

    def test_mps_inference(self):
        """Test that encoder works on MPS device (Apple Silicon GPU)."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        config = ConformerConfig(num_layers=2)
        encoder = ConformerEncoder(config)

        # Move encoder to MPS
        encoder = encoder.to("mps")

        # Create input tensors and move to MPS
        mel = torch.randn(2, 1000, 80).to("mps")
        lengths = torch.tensor([1000, 800]).to("mps")

        # Forward pass on MPS
        out, out_lengths = encoder(mel, lengths)

        # Check output shapes
        expected_seq_len = 1000 // 4
        assert out.shape == (2, expected_seq_len, config.hidden_size)

        # Verify output is on MPS device
        assert out.device.type == "mps"
        assert out_lengths.device.type == "mps"

        # Check output lengths
        expected_lengths = torch.tensor([250, 200]).to("mps")
        assert torch.equal(out_lengths, expected_lengths)

    def test_single_sequence(self):
        """Test encoder with single sequence."""
        config = ConformerConfig(num_layers=2)
        encoder = ConformerEncoder(config)

        # Single sequence input
        mel = torch.randn(1, 500, 80)  # 5s of audio
        lengths = torch.tensor([500])

        out, out_lengths = encoder(mel, lengths)

        # Check output shape
        expected_seq_len = 500 // 4
        assert out.shape == (1, expected_seq_len, config.hidden_size)
        assert out_lengths.shape == (1,)
        assert out_lengths.item() == 125

    def test_different_hidden_sizes(self):
        """Test encoder with different hidden sizes."""
        for hidden_size in [256, 512, 1024]:
            config = ConformerConfig(num_layers=2, hidden_size=hidden_size)
            encoder = ConformerEncoder(config)

            mel = torch.randn(1, 400, 80)
            lengths = torch.tensor([400])

            out, out_lengths = encoder(mel, lengths)

            expected_seq_len = 400 // 4
            assert out.shape == (1, expected_seq_len, hidden_size)

    def test_different_num_layers(self):
        """Test encoder with different number of layers."""
        for num_layers in [1, 3, 5]:
            config = ConformerConfig(num_layers=num_layers)
            encoder = ConformerEncoder(config)

            # Count actual layers
            assert len(encoder.layers) == num_layers

            # Test forward pass still works
            mel = torch.randn(2, 800, 80)
            lengths = torch.tensor([800, 600])

            out, out_lengths = encoder(mel, lengths)

            expected_seq_len = 800 // 4
            assert out.shape == (2, expected_seq_len, config.hidden_size)
