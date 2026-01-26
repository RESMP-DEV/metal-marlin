"""Multi-domain calibration dataset loader for LLM quantization.

The v3 calibration dataset is a multi-domain dataset (code, chat, reasoning, math)
that provides better activation ranges than WikiText-2 for LLM quantization.

The dataset contains diverse samples covering:
- Scientific/technical papers
- Code snippets (C++, Python, etc.)
- Mathematical reasoning (proofs, Q&A)
- Literary text for natural language diversity

Usage:
    from metal_marlin.calibration import CalibrationDataset, compute_activation_ranges

    # Load calibration dataset
    dataset = CalibrationDataset.v3(max_samples=512)

    # Compute activation ranges for quantization
    ranges = compute_activation_ranges("path/to/model", dataset)
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Multi-domain calibration v3 URL
_CALIBRATION_V3_URL = (
    "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/"
    "raw/2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt"
)
_CALIBRATION_V3_SHA256 = None  # Will be computed on first download

# Default cache directory
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "metal_marlin" / "calibration"


@dataclass
class CalibrationDataset:
    """Container for calibration samples.

    Attributes:
        samples: List of text samples for calibration.
        name: Dataset name/identifier.
        version: Dataset version string.
        metadata: Optional metadata dict.
    """

    samples: list[str]
    name: str
    version: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    @property
    def total_chars(self) -> int:
        """Total character count across all samples."""
        return sum(len(s) for s in self.samples)

    @property
    def avg_sample_length(self) -> float:
        """Average sample length in characters."""
        if not self.samples:
            return 0.0
        return self.total_chars / len(self.samples)

    def filter(
        self, predicate: Callable[[str], bool], in_place: bool = False
    ) -> CalibrationDataset:
        """Filter samples by predicate.

        Args:
            predicate: Function returning True to keep sample.
            in_place: If True, modify this dataset; else return new one.

        Returns:
            Filtered dataset (self if in_place, new instance otherwise).
        """
        filtered = [s for s in self.samples if predicate(s)]
        if in_place:
            self.samples = filtered
            return self
        return CalibrationDataset(
            samples=filtered,
            name=self.name,
            version=self.version,
            metadata=dict(self.metadata),
        )

    def truncate(self, max_tokens: int, tokenizer) -> CalibrationDataset:
        """Truncate samples to max_tokens using provided tokenizer.

        Args:
            max_tokens: Maximum tokens per sample.
            tokenizer: HuggingFace tokenizer with encode/decode methods.

        Returns:
            New dataset with truncated samples.
        """
        truncated = []
        for sample in self.samples:
            ids = tokenizer.encode(sample, add_special_tokens=False)
            if len(ids) > max_tokens:
                ids = ids[:max_tokens]
            truncated.append(tokenizer.decode(ids))
        return CalibrationDataset(
            samples=truncated,
            name=self.name,
            version=self.version,
            metadata={**self.metadata, "max_tokens": max_tokens},
        )


class CalibrationDatasetLoader:
    """Loader for multi-domain calibration datasets."""

    @classmethod
    def v3(
        cls,
        max_samples: int | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        min_sample_length: int = 100,
    ) -> CalibrationDataset:
        """Load calibration v3 from Gist.

        The v3 dataset is a multi-domain text file containing scientific papers,
        code snippets, math problems, and literary passages. Samples are separated
        by blank lines.

        IMPORTANT: For accurate quantization, use ALL samples (max_samples=None).
        The v3 dataset contains 800+ carefully curated multi-domain samples.
        Limiting samples reduces calibration accuracy and may hurt quantization
        quality.

        Args:
            max_samples: Maximum number of samples to return, or None for ALL.
                Default is None (use full dataset). Only limit for quick testing.
            cache_dir: Directory to cache downloaded file.
            force_download: If True, re-download even if cached.
            min_sample_length: Minimum characters per sample (filters noise).

        Returns:
            CalibrationDataset with loaded samples.
        """
        cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / "calibration_v3.txt"

        # Download if needed
        if force_download or not cache_file.exists():
            cls._download_file(_CALIBRATION_V3_URL, cache_file)

        # Parse the file
        raw_text = cache_file.read_text(encoding="utf-8")
        samples = cls._parse_v3_text(raw_text, min_sample_length)

        # Only limit samples if explicitly requested
        total_samples = len(samples)
        if max_samples is not None and len(samples) > max_samples:
            samples = samples[:max_samples]

        return CalibrationDataset(
            samples=samples,
            name="calibration_dataset",
            version="v3",
            metadata={
                "source": _CALIBRATION_V3_URL,
                "total_raw_samples": len(cls._parse_v3_text(raw_text, 0)),
                "min_sample_length": min_sample_length,
                "samples_used": len(samples),
                "total_available": total_samples,
            },
        )

    @classmethod
    def from_local(cls, path: str | Path) -> CalibrationDataset:
        """Load calibration data from local JSON/JSONL/TXT file.

        Supports:
        - JSON: List of strings or list of dicts with "text" key
        - JSONL: One JSON object per line with "text" key
        - TXT: Plain text, blank-line separated samples

        Args:
            path: Path to local calibration file.

        Returns:
            CalibrationDataset with loaded samples.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")

        if suffix == ".json":
            data = json.loads(text)
            samples = cls._parse_json_data(data)
        elif suffix == ".jsonl":
            samples = []
            for line in text.strip().split("\n"):
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        samples.append(obj)
                    elif isinstance(obj, dict) and "text" in obj:
                        samples.append(obj["text"])
        else:
            # Plain text format
            samples = cls._parse_v3_text(text, min_length=50)

        return CalibrationDataset(
            samples=samples,
            name=path.stem,
            version="local",
            metadata={"source_path": str(path)},
        )

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        """Download file from URL with progress."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "metal_marlin/1.0"},
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()

        dest.write_bytes(content)

    @staticmethod
    def _parse_v3_text(text: str, min_length: int) -> list[str]:
        """Parse v3 format: blank-line separated paragraphs.

        Consecutive non-blank lines are joined into samples.
        """
        samples = []
        current = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                # Blank line: end of sample
                if current:
                    sample = "\n".join(current)
                    if len(sample) >= min_length:
                        samples.append(sample)
                    current = []
            else:
                current.append(line)

        # Handle last sample without trailing blank
        if current:
            sample = "\n".join(current)
            if len(sample) >= min_length:
                samples.append(sample)

        return samples

    @staticmethod
    def _parse_json_data(data) -> list[str]:
        """Parse JSON data into list of strings."""
        if isinstance(data, list):
            samples = []
            for item in data:
                if isinstance(item, str):
                    samples.append(item)
                elif isinstance(item, dict):
                    # Try common keys
                    for key in ["text", "content", "sample", "prompt"]:
                        if key in item:
                            samples.append(item[key])
                            break
            return samples
        elif isinstance(data, dict):
            # Try to find samples in dict
            for key in ["samples", "data", "texts", "calibration"]:
                if key in data and isinstance(data[key], list):
                    return CalibrationDatasetLoader._parse_json_data(data[key])
        return []


def compute_activation_ranges(
    model_path: str,
    calibration: CalibrationDataset,
    layers: list[str] | None = None,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    verbose: bool = True,
) -> dict[str, tuple[float, float]]:
    """Run calibration samples through model, track activation min/max.

    Requires transformers library for tokenization and model loading.
    Uses numpy for computation (no MLX dependency).

    Args:
        model_path: Path to HuggingFace model or model ID.
        calibration: CalibrationDataset to run through model.
        layers: Specific layers to track. If None, tracks all linear layers.
        batch_size: Batch size for inference.
        max_seq_len: Maximum sequence length for tokenization.
        verbose: Print progress.

    Returns:
        Dict mapping layer names to (min, max) activation ranges.
        Used for setting optimal quantization scales.

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers library required for compute_activation_ranges. "
            "Install with: pip install transformers"
        ) from e

    if verbose:
        print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if verbose:
        print(f"Loading model from {model_path}...")
    # Load in float32 for accurate activation tracking
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="float32",
        device_map="cpu",  # CPU for numpy compatibility
    )
    model.eval()

    # Set up activation hooks
    activation_ranges: dict[str, tuple[float, float]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            # Get input activations (what feeds into the layer)
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input

            # Convert to numpy
            if hasattr(x, "detach"):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = np.array(x)

            x_flat = x_np.flatten().astype(np.float64)
            batch_min = float(np.min(x_flat))
            batch_max = float(np.max(x_flat))

            if name in activation_ranges:
                cur_min, cur_max = activation_ranges[name]
                activation_ranges[name] = (
                    min(cur_min, batch_min),
                    max(cur_max, batch_max),
                )
            else:
                activation_ranges[name] = (batch_min, batch_max)

        return hook

    # Register hooks on linear layers
    import torch.nn as nn

    for name, module in model.named_modules():
        if layers is not None:
            if not any(name.startswith(p) or p in name for p in layers):
                continue
        elif not isinstance(module, nn.Linear):
            continue

        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)

    if verbose:
        print(f"Registered hooks on {len(hooks)} layers")

    # Run calibration samples
    import torch

    num_samples = len(calibration)
    for i, sample in enumerate(calibration):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing sample {i + 1}/{num_samples}...")

        inputs = tokenizer(
            sample,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

        with torch.no_grad():
            model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if verbose:
        print(f"Computed activation ranges for {len(activation_ranges)} layers")

    return activation_ranges


def ranges_to_scales(
    ranges: dict[str, tuple[float, float]],
    quant_type: str = "fp4",
) -> dict[str, float]:
    """Convert activation ranges to quantization scales.

    Args:
        ranges: Dict of layer_name -> (min, max) from compute_activation_ranges.
        quant_type: Quantization type: "fp4", "int4_sym", "int4_asym", "int8_sym".

    Returns:
        Dict mapping layer names to optimal scale values.
    """
    scales = {}

    for name, (rmin, rmax) in ranges.items():
        absmax = max(abs(rmin), abs(rmax))

        if quant_type == "fp4":
            # FP4 E2M1: representable magnitudes [0, 0.5, 1, 1.5, 2, 3, 4, 6]
            scale = absmax / 6.0
        elif quant_type == "int4_sym":
            # INT4 symmetric: [-8, 7] -> use 7 as max
            scale = absmax / 7.0
        elif quant_type == "int4_asym":
            # INT4 asymmetric: [0, 15]
            scale = (rmax - rmin) / 15.0
        elif quant_type == "int8_sym":
            # INT8 symmetric: [-128, 127]
            scale = absmax / 127.0
        else:
            raise ValueError(
                f"Unknown quant_type={quant_type!r}. "
                f"Expected: 'fp4', 'int4_sym', 'int4_asym', 'int8_sym'"
            )

        # Prevent zero scales
        scales[name] = max(scale, 1e-10)

    return scales


def save_ranges(
    ranges: dict[str, tuple[float, float]],
    path: str | Path,
) -> None:
    """Save activation ranges to JSON file."""
    path = Path(path)
    data = {name: {"min": rmin, "max": rmax} for name, (rmin, rmax) in ranges.items()}
    path.write_text(json.dumps(data, indent=2))


def load_ranges(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load activation ranges from JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return {name: (v["min"], v["max"]) for name, v in data.items()}


# Backwards compatibility alias
BartowskiCalibration = CalibrationDatasetLoader
