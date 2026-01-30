"""End-to-end inference tests for ParakeetTDT."""

import pytest
import torch

from metal_marlin.asr import ConformerConfig, ParakeetTDT
from metal_marlin.asr.tdt_config import TDTConfig


@pytest.fixture
def model():
    """Create small ParakeetTDT for testing."""
    conformer_cfg = ConformerConfig(
        num_layers=2,  # Small for testing
        hidden_size=64,
        num_attention_heads=2,
        ffn_intermediate_size=128,
        conv_kernel_size=7,
        dropout=0.0,
        n_mels=80,
        subsampling_factor=4,
    )
    tdt_cfg = TDTConfig(
        vocab_size=128,
        predictor_hidden_size=32,
        predictor_num_layers=1,
        encoder_hidden_size=64,
        joint_hidden_size=64,
    )
    return ParakeetTDT(conformer_cfg, tdt_cfg)


class TestParakeetInference:
    """Test ParakeetTDT inference paths."""

    def test_encode_basic(self, model):
        """Test basic encode() call."""
        mel = torch.randn(1, 100, 80)
        lengths = torch.tensor([100])

        out, out_lens = model.encode(mel, lengths)

        assert out.dim() == 3
        assert out.size(0) == 1
        assert out.size(2) == 64  # hidden_size

    def test_forward_equals_encode(self, model):
        """Test forward() is alias to encode()."""
        model.eval()  # Ensure deterministic behavior
        mel = torch.randn(1, 100, 80)
        lengths = torch.tensor([100])

        with torch.no_grad():
            out1, lens1 = model(mel, lengths)
            out2, lens2 = model.encode(mel, lengths)

        torch.testing.assert_close(out1, out2)
        torch.testing.assert_close(lens1, lens2)

    def test_batch_inference(self, model):
        """Test batched inference."""
        mel = torch.randn(4, 100, 80)
        lengths = torch.tensor([100, 90, 80, 70])

        out, out_lens = model.encode(mel, lengths)

        assert out.size(0) == 4

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_inference(self, model):
        """Test inference on MPS device."""
        model = model.to("mps")
        mel = torch.randn(1, 100, 80, device="mps")
        lengths = torch.tensor([100], device="mps")

        out, out_lens = model.encode(mel, lengths)

        assert out.device.type == "mps"

    def test_transcribe(self, model):
        """Test transcribe() decoding."""
        mel = torch.randn(1, 100, 80)
        lengths = torch.tensor([100])

        tokens = model.transcribe(mel, lengths)

        assert isinstance(tokens, list)
        assert len(tokens) == 1
        assert isinstance(tokens[0], list)
