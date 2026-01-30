"""Export Conformer encoder to CoreML for ANE execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import coremltools as ct

    HAS_COREMLTOOLS = True
except ImportError:
    ct = None
    HAS_COREMLTOOLS = False


class CoreMLEncoderWrapper(nn.Module):
    """Wrapper to make encoder CoreML-compatible.

    CoreML needs fixed shapes for some ops, so we wrap the encoder
    to handle dynamic sequence lengths.
    """

    def __init__(self, encoder: nn.Module, max_seq_len: int = 3000):
        super().__init__()
        self.encoder = encoder
        self.max_seq_len = max_seq_len

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward with implicit lengths."""
        batch_size, seq_len, _ = mel.shape
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=mel.device)
        out, _ = self.encoder(mel, lengths)
        return out


def export_encoder_to_coreml(
    encoder: nn.Module,
    output_path: Path | str,
    n_mels: int = 80,
    max_seq_len: int = 3000,
    compute_units: str = "CPU_AND_NE",
) -> Any:
    """Export Conformer encoder to CoreML.

    Args:
        encoder: ConformerEncoder module
        output_path: Path to save .mlpackage
        n_mels: Number of mel channels
        max_seq_len: Maximum sequence length
        compute_units: CoreML compute units ("CPU_AND_NE", "ALL", "CPU_ONLY")

    Returns:
        Compiled CoreML model
    """
    if not HAS_COREMLTOOLS:
        raise ImportError("coremltools required: pip install coremltools")

    output_path = Path(output_path)

    # Wrap encoder for CoreML compatibility
    wrapper = CoreMLEncoderWrapper(encoder, max_seq_len)
    wrapper.eval()

    # Create example input
    example_mel = torch.randn(1, 500, n_mels)

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_mel)

    # Define input shape with flexible sequence length
    input_shape = ct.Shape(shape=(1, ct.RangeDim(1, max_seq_len), n_mels))

    # Convert to CoreML
    compute_unit_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ANE_ONLY": ct.ComputeUnit.CPU_AND_NE,  # No pure ANE option
    }

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=input_shape)],
        outputs=[ct.TensorType(name="encoder_output")],
        compute_units=compute_unit_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE),
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Saved CoreML model to {output_path}")

    return mlmodel


class ANEEncoder:
    """Encoder that runs on ANE via CoreML."""

    def __init__(self, mlmodel_path: Path | str):
        """Load CoreML model.

        Args:
            mlmodel_path: Path to .mlpackage
        """
        if not HAS_COREMLTOOLS:
            raise ImportError("coremltools required")

        self.model = ct.models.MLModel(str(mlmodel_path))
        self._path = Path(mlmodel_path)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Run inference on ANE.

        Args:
            mel: Input mel spectrogram [B, T, n_mels]

        Returns:
            Encoder output [B, T', hidden_size]
        """
        # CoreML needs numpy input
        mel_np = mel.cpu().numpy()

        # Run inference
        result = self.model.predict({"mel": mel_np})

        # Convert back to torch
        output = torch.from_numpy(result["encoder_output"])

        # Move to original device if needed
        if mel.is_mps:
            output = output.to("mps")

        return output

    @property
    def path(self) -> Path:
        return self._path
