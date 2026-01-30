"""Parakeet-TDT model implementation for ASR.

This module implements the complete Parakeet-TDT (Transducer-Duration-Transducer) model
that combines a Conformer encoder with a TDT decoder for speech recognition.
The TDT decoder includes duration predictions to skip multiple frames at once,
enabling more efficient decoding compared to traditional RNN-T models.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .conformer_config import ConformerConfig
from .conformer_encoder import ConformerEncoder
from .tdt_config import TDTConfig
from .tdt_joint import TDTJoint
from .tdt_predictor import TDTPredictor


class TDTDecoder(nn.Module):
    """TDT (Transducer-Duration-Transducer) decoder.

    The TDT decoder combines an autoregressive predictor with a joint network
    to generate token predictions and duration predictions. The duration
    predictions enable skipping multiple frames during decoding.

    Args:
        config: TDTConfig containing decoder architecture parameters.

    Attributes:
        predictor: Autoregressive LSTM-based predictor.
        joint: Joint network combining encoder and predictor outputs.
    """

    def __init__(self, config: TDTConfig):
        super().__init__()
        self.config = config
        self.predictor = TDTPredictor(config)
        self.joint = TDTJoint(config)

    def forward(
        self, encoder_out: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of TDT decoder.

        Args:
            encoder_out: Encoder output tensor [B, T, enc_dim]
            targets: Target token indices [B, U] for training

        Returns:
            Tuple containing:
            - logits: Vocabulary logits [B, T, U, vocab_size]
            - durations: Duration predictions [B, T, U, max_duration+1]
        """
        batch_size, seq_len, enc_dim = encoder_out.shape

        # Shift targets right by 1 for teacher forcing
        # Add start token (assuming 0 is blank token)
        shifted_targets = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=targets.dtype, device=targets.device),
                targets[:, :-1],
            ],
            dim=1,
        )

        # Get predictor output
        predictor_out, _ = self.predictor(shifted_targets)

        # Get joint output
        logits, durations = self.joint(encoder_out, predictor_out)

        return logits, durations

    def decode_step(
        self,
        encoder_frame: torch.Tensor,
        prev_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single decoding step for greedy decoding.

        Args:
            encoder_frame: Single encoder frame [1, 1, enc_dim]
            prev_tokens: Previous tokens [1, U]
            state: Optional LSTM state tuple

        Returns:
            Tuple containing:
            - joint_out: Joint network output [1, 1, vocab_size]
            - new_state: Updated LSTM state
        """
        # Get predictor output
        predictor_out, new_state = self.predictor(prev_tokens, state)

        # Get joint output for the last token
        joint_out, _ = self.joint(encoder_frame, predictor_out[:, -1:, :])

        return joint_out.squeeze(1), new_state  # [1, vocab_size], state


class ParakeetTDT(nn.Module):
    """Full Parakeet-TDT model for ASR.

    This model combines a Conformer encoder with a TDT decoder for
    efficient speech recognition. The TDT architecture includes duration
    predictions that allow the decoder to skip multiple frames during
    inference, improving decoding speed.

    Args:
        conformer_config: ConformerConfig for encoder architecture
        tdt_config: TDTConfig for decoder architecture

    Attributes:
        encoder: Conformer encoder processing mel spectrograms
        decoder: TDT decoder with predictor and joint networks
    """

    def __init__(self, conformer_config: ConformerConfig, tdt_config: TDTConfig):
        super().__init__()

        # Ensure configs are compatible
        tdt_config.encoder_hidden_size = conformer_config.hidden_size

        self.encoder = ConformerEncoder(conformer_config)
        self.decoder = TDTDecoder(tdt_config)

        # Store configs for convenience
        self.conformer_config = conformer_config
        self.tdt_config = tdt_config

    def forward(
        self, mel: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            mel: Input mel spectrogram [B, T, n_mels]
            lengths: Original sequence lengths [B]
            targets: Target token indices [B, U]

        Returns:
            Tuple containing:
            - logits: Vocabulary logits [B, T, U, vocab_size]
            - durations: Duration predictions [B, T, U, max_duration+1]
            - enc_lengths: Encoder output lengths after subsampling [B]
        """
        # Encode mel spectrogram
        encoder_out, enc_lengths = self.encoder(mel, lengths)

        # Decode with TDT
        logits, durations = self.decoder(encoder_out, targets)

        return logits, durations, enc_lengths

    @torch.no_grad()
    def transcribe(self, mel: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """Transcribe audio using greedy TDT decoding.

        Args:
            mel: Input mel spectrogram [B, T, n_mels]
            lengths: Original sequence lengths [B]

        Returns:
            List of decoded token sequences, one per batch element
        """
        self.eval()

        # Encode mel spectrogram
        encoder_out, _ = self.encoder(mel, lengths)

        # Greedy decode each sample
        batch_size = encoder_out.size(0)
        results = []

        for i in range(batch_size):
            single_encoder = encoder_out[i : i + 1, :, :]  # [1, T, enc_dim]
            tokens = self._decode_single(single_encoder)
            results.append(tokens)

        return results

    def _decode_single(self, encoder_out: torch.Tensor) -> list[int]:
        """Decode a single audio sample with greedy search.

        Args:
            encoder_out: Encoder output [1, T, enc_dim]

        Returns:
            List of decoded token IDs
        """
        # Start with blank token
        tokens = [0]  # Assuming 0 is blank
        state = None
        t, T = 0, encoder_out.size(1)

        while t < T:
            # Get current encoder frame
            enc_frame = encoder_out[:, t : t + 1, :]  # [1, 1, enc_dim]

            # Prepare tokens tensor
            tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=encoder_out.device)

            # Get predictor output
            predictor_out, state = self.decoder.predictor(tokens_tensor, state)

            # Get joint output
            joint_out, duration_out = self.decoder.joint(enc_frame, predictor_out[:, -1:, :])

            # Get most likely token
            log_probs = torch.log_softmax(joint_out, dim=-1)
            max_id = torch.argmax(log_probs, dim=-1).item()

            if max_id == 0:  # Blank token
                # Predict duration and skip frames
                duration_probs = torch.softmax(duration_out.squeeze(), dim=-1)
                duration_values = torch.arange(
                    1, len(duration_probs) + 1, dtype=torch.float32, device=encoder_out.device
                )
                duration = torch.sum(duration_probs * duration_values).item()
                duration = max(1, int(duration))  # At least 1 frame

                t += duration
            else:
                # Non-blank token: emit and continue at same time step
                tokens.append(max_id)

                # Limit sequence length to prevent infinite loops
                if len(tokens) > 1000:  # Arbitrary limit
                    break

        return tokens[1:] if tokens[0] == 0 else tokens  # Remove initial blank

    @classmethod
    def from_pretrained(cls, path: str | Path) -> ParakeetTDT:
        """Load model from converted checkpoint.

        Args:
            path: Path to checkpoint directory or file

        Returns:
            Loaded ParakeetTDT model
        """
        import json
        from pathlib import Path

        checkpoint_path = Path(path)

        # Load configs
        if (checkpoint_path / "config.json").exists():
            config_path = checkpoint_path / "config.json"
        elif checkpoint_path.suffix == ".json":
            config_path = checkpoint_path
            checkpoint_path = checkpoint_path.parent
        else:
            raise FileNotFoundError(f"Config file not found in {path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Create configs
        conformer_config = ConformerConfig(**config_dict.get("conformer", {}))
        tdt_config = TDTConfig(**config_dict.get("tdt", {}))

        # Initialize model
        model = cls(conformer_config, tdt_config)

        # Load weights
        if (checkpoint_path / "model.safetensors").exists():
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path / "model.safetensors")
        elif (checkpoint_path / "model.pt").exists():
            state_dict = torch.load(checkpoint_path / "model.pt", map_location="cpu")
        elif checkpoint_path.suffix in [".safetensors", ".pt", ".pth"]:
            if checkpoint_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {path}")

        model.load_state_dict(state_dict, strict=False)
        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save model to checkpoint.

        Args:
            path: Path to save checkpoint directory
        """
        import json
        from pathlib import Path

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save configs
        config_dict = {"conformer": self.conformer_config.__dict__, "tdt": self.tdt_config.__dict__}

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save weights
        torch.save(self.state_dict(), save_path / "model.pt")


def create_parakeet_tdt_small() -> ParakeetTDT:
    """Create a small Parakeet-TDT model for testing.

    Returns:
        Small ParakeetTDT model
    """
    conformer_config = ConformerConfig(
        num_layers=6,
        hidden_size=256,
        num_attention_heads=4,
        ffn_intermediate_size=1024,
        conv_kernel_size=31,
        dropout=0.1,
        n_mels=80,
        sample_rate=16000,
        subsampling_factor=4,
    )

    tdt_config = TDTConfig(
        vocab_size=1024,
        joint_hidden_size=512,
        predictor_hidden_size=512,
        predictor_num_layers=2,
        max_duration=10,
        encoder_hidden_size=256,
    )

    return ParakeetTDT(conformer_config, tdt_config)
