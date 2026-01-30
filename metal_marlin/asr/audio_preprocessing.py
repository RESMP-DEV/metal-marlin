"""Audio preprocessing for ASR inference.

Provides mel spectrogram extraction compatible with Parakeet models.
"""

from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T


class MelSpectrogramExtractor:
    """Extract mel spectrograms from audio waveforms.

    Compatible with Parakeet/NeMo preprocessing settings.

    Args:
        sample_rate: Audio sample rate (default: 16000)
        n_mels: Number of mel filterbank channels (default: 80)
        n_fft: FFT window size (default: 512)
        hop_length: Hop length in samples (default: 160 = 10ms at 16kHz)
        win_length: Window length (default: 400 = 25ms at 16kHz)
        f_min: Minimum frequency (default: 0)
        f_max: Maximum frequency (default: None = sr/2)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from waveform.

        Args:
            waveform: Audio waveform [B, T] or [T]

        Returns:
            Mel spectrogram [B, T', n_mels] (T' = frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Get mel spectrogram: [B, n_mels, T']
        mel = self.mel_transform(waveform)

        # Apply log compression (log mel)
        mel = torch.log(mel.clamp(min=1e-10))

        # Transpose to [B, T', n_mels] for model input
        mel = mel.transpose(1, 2)

        return mel

    def to(self, device: torch.device | str) -> MelSpectrogramExtractor:
        """Move transform to device."""
        self.mel_transform = self.mel_transform.to(device)
        return self


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 16000)

    Returns:
        Waveform tensor [T]
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)
