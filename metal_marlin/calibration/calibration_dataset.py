"""Multi-domain calibration dataset loader for LLM quantization.

The v3 calibration dataset is a multi-domain dataset containing ~800 samples across:
- Scientific/medical text
- Code snippets (C++, Python, etc.)
- Conversational/chat
- Mathematical reasoning

This multi-domain coverage produces better Hessian estimates than single-domain
datasets like WikiText-2, resulting in improved quantization quality.

Usage:
    from metal_marlin.calibration import CalibrationDataset

    # Load calibration v3 from Gist
    dataset = CalibrationDataset.v3(max_samples=512)

    # Tokenize for model forward passes
    token_arrays = dataset.tokenize(tokenizer, max_length=2048)

    # Iterate in batches
    for batch in dataset.get_batches(batch_size=4):
        # batch is list[str] of text samples
        ...
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

# Multi-domain calibration v3 URL (external gist)
_CALIBRATION_V3_URL = (
    "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/"
    "raw/2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt"
)

# Default cache directory
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "metal_marlin" / "calibration"


@dataclass
class CalibrationDataset:
    """Multi-domain calibration dataset for LLM quantization.

    Contains text samples across multiple domains for computing activation
    statistics during LLM quantization. Multi-domain coverage ensures better
    Hessian estimates than single-domain datasets.

    Attributes:
        samples: List of text samples for calibration.
        name: Dataset name/identifier.
        version: Dataset version string.
        metadata: Additional metadata about the dataset.
    """

    samples: list[str]
    name: str = "calibration_dataset"
    version: str = "v3"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.samples)

    def filter(self, predicate: Any, in_place: bool = False) -> CalibrationDataset:
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

    @classmethod
    def v3(
        cls,
        max_samples: int | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        min_sample_length: int = 100,
    ) -> CalibrationDataset:
        """Load calibration v3 from examples folder or download from Gist.

        The v3 dataset is a multi-domain text file containing scientific papers,
        code snippets, math problems, and literary passages. Samples are separated
        by blank lines.

        IMPORTANT: For accurate quantization, use ALL samples (max_samples=None).
        The v3 dataset contains ~800 carefully curated multi-domain samples.
        Limiting samples reduces calibration accuracy.

        Args:
            max_samples: Maximum number of samples to return, or None for ALL.
                Default is None (use full dataset). Only limit for quick testing.
            cache_dir: Directory to cache downloaded file.
            force_download: If True, re-download even if cached.
            min_sample_length: Minimum characters per sample (filters noise).

        Returns:
            CalibrationDataset instance with loaded samples.
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

        # Track total before limiting
        total_samples = len(samples)

        # Only limit samples if explicitly requested
        if max_samples is not None and len(samples) > max_samples:
            samples = samples[:max_samples]

        return cls(
            samples=samples,
            name="calibration_dataset",
            version="v3",
            metadata={
                "source": _CALIBRATION_V3_URL,
                "min_sample_length": min_sample_length,
                "samples_used": len(samples),
                "total_available": total_samples,
            },
        )

    @classmethod
    def from_local(cls, path: str | Path, min_sample_length: int = 50) -> CalibrationDataset:
        """Load custom calibration file.

        Supports multiple formats:
        - Plain text (.txt): Samples separated by blank lines
        - JSON (.json): Array of strings or array of objects with "text" key
        - JSONL (.jsonl): One JSON object per line with "text" key

        Args:
            path: Path to local calibration file.
            min_sample_length: Minimum characters per sample (filters noise).

        Returns:
            CalibrationDataset instance with loaded samples.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix == ".json":
            # JSON format: array of strings or objects with "text" key
            data = json.loads(raw_text)
            if isinstance(data, list):
                samples = []
                for item in data:
                    if isinstance(item, str):
                        if len(item) >= min_sample_length:
                            samples.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        if len(text) >= min_sample_length:
                            samples.append(text)
            else:
                samples = []

        elif suffix == ".jsonl":
            # JSONL format: one JSON object per line with "text" key
            samples = []
            for line in raw_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "text" in obj:
                        text = obj["text"]
                        if len(text) >= min_sample_length:
                            samples.append(text)
                    elif isinstance(obj, str):
                        if len(obj) >= min_sample_length:
                            samples.append(obj)
                except json.JSONDecodeError:
                    continue

        else:
            # Plain text format: blank-line separated
            samples = cls._parse_v3_text(raw_text, min_length=min_sample_length)

        return cls(
            samples=samples,
            name=path.stem,
            version="local",
            metadata={
                "source_path": str(path),
                "total_samples": len(samples),
            },
        )

    def tokenize(self, tokenizer: Any, max_length: int = 2048) -> list[np.ndarray]:
        """Tokenize samples for model forward passes.

        Uses a HuggingFace-compatible tokenizer to convert text samples into
        token ID arrays suitable for model forward passes during calibration.

        Args:
            tokenizer: A HuggingFace tokenizer with encode() method.
            max_length: Maximum sequence length. Samples longer than this
                are truncated.

        Returns:
            List of numpy arrays containing token IDs for each sample.
            Each array has shape [seq_len] where seq_len <= max_length.
        """
        tokenized: list[np.ndarray] = []

        for sample in self.samples:
            # Handle tokenizers with different interfaces
            if hasattr(tokenizer, "encode"):
                # HuggingFace tokenizer
                ids = tokenizer.encode(
                    sample,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )
                if isinstance(ids, list):
                    ids = np.array(ids, dtype=np.int32)
                elif hasattr(ids, "numpy"):
                    ids = ids.numpy().astype(np.int32)
                else:
                    ids = np.asarray(ids, dtype=np.int32)
            elif hasattr(tokenizer, "__call__"):
                # Tokenizer with __call__ returning dict
                encoded = tokenizer(
                    sample,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np",
                )
                ids = encoded["input_ids"].squeeze().astype(np.int32)
            else:
                raise TypeError(
                    f"Unsupported tokenizer type: {type(tokenizer)}. "
                    "Expected HuggingFace tokenizer with encode() method."
                )

            tokenized.append(ids)

        return tokenized

    def get_batches(self, batch_size: int = 4) -> Iterator[list[str]]:
        """Yield batches for calibration.

        Iterates through samples in chunks of batch_size. The final batch
        may be smaller if the total sample count is not evenly divisible.

        Args:
            batch_size: Number of samples per batch.

        Yields:
            Lists of text samples, each containing up to batch_size samples.
        """
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i : i + batch_size]

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

    @property
    def domain_coverage(self) -> dict[str, int]:
        """Estimate domain coverage by detecting content patterns.

        Returns approximate counts of samples in each domain category.
        """
        domains = {
            "code": 0,
            "scientific": 0,
            "conversational": 0,
            "mathematical": 0,
            "other": 0,
        }

        for sample in self.samples:
            lower = sample.lower()

            # Code detection
            if any(
                kw in sample
                for kw in ["def ", "class ", "import ", "function ", "return ", "//", "/*"]
            ):
                domains["code"] += 1
            # Scientific/medical detection
            elif any(
                kw in lower
                for kw in ["research", "study", "experiment", "hypothesis", "conclusion"]
            ):
                domains["scientific"] += 1
            # Conversational detection
            elif any(kw in sample for kw in ["User:", "Assistant:", "Human:", "AI:", "Q:", "A:"]):
                domains["conversational"] += 1
            # Mathematical detection
            elif any(
                kw in lower for kw in ["equation", "theorem", "proof", "calculate", "formula"]
            ) or any(c in sample for c in ["∑", "∫", "√", "±", "≈"]):
                domains["mathematical"] += 1
            else:
                domains["other"] += 1

        return domains

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        """Download file from URL."""
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
        current: list[str] = []

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
    def _parse_json_data(data: Any, min_length: int = 0) -> list[str]:
        """Parse JSON data in various formats into a list of text samples.

        Supports:
        - List of strings: ["text1", "text2"]
        - List of dicts with "text" key: [{"text": "sample1"}, {"text": "sample2"}]
        - Dict with "samples" key: {"samples": ["text1", "text2"]}
        - Dict with "samples" containing dicts: {"samples": [{"text": "..."}]}

        Args:
            data: Parsed JSON data (dict or list)
            min_length: Minimum sample length to include

        Returns:
            List of text samples
        """
        samples: list[str] = []

        # Handle dict with 'samples' key
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]

        # Handle list
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    if len(item) >= min_length:
                        samples.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    if isinstance(text, str) and len(text) >= min_length:
                        samples.append(text)

        return samples


# Backwards compatibility alias
BartowskiCalibration = CalibrationDataset
