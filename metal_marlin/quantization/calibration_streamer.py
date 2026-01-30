"""RAM-aware calibration data streaming for quantization.

This module provides utilities for streaming calibration data in batches
that fit within available system memory, enabling calibration of large
models without OOM errors.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch


class CalibrationDataset(Protocol):
    """Protocol for calibration datasets."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> str:
        """Return a text sample at the given index."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over text samples."""
        ...


@dataclass
class CalibrationBatch:
    """Single batch of calibration activations.

    Attributes:
        input_ids: Tokenized input IDs with shape [batch_size, seq_len]
        batch_idx: Index of this batch (0-indexed)
        total_batches: Total number of batches in the dataset
    """

    input_ids: torch.Tensor
    batch_idx: int
    total_batches: int


class CalibrationStreamer:
    """Stream calibration data in RAM-aware batches.

    Automatically sizes batches based on available memory to prevent OOM
    during calibration of large language models.

    Args:
        dataset: Calibration dataset providing text samples
        tokenizer: Tokenizer with a __call__ method for encoding
        max_seq_len: Maximum sequence length for tokenization
        target_memory_gb: Target memory usage in GB (default: 8.0)

    Example:
        >>> streamer = CalibrationStreamer(
        ...     dataset=calib_data,
        ...     tokenizer=tokenizer,
        ...     max_seq_len=2048,
        ...     target_memory_gb=16.0,
        ... )
        >>> for batch in streamer.iter_batches(hidden_dim=4096):
        ...     process_batch(batch)
    """

    def __init__(
        self,
        dataset: CalibrationDataset,
        tokenizer: Any,
        max_seq_len: int = 2048,
        target_memory_gb: float = 8.0,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.target_memory_gb = target_memory_gb

    def estimate_batch_size(self, hidden_dim: int) -> int:
        """Calculate batch size that fits in target memory.

        Estimates the maximum number of samples that can be processed
        in a single batch without exceeding the target memory. Accounts
        for activation storage and Hessian matrix accumulation used in
        quantization algorithms like GPTQ.

        Memory estimation:
            - Per sample: seq_len * hidden_dim * 4 bytes (float32 activations)
            - Hessian buffer: hidden_dim * hidden_dim * 8 bytes (float64)

        Args:
            hidden_dim: Hidden dimension of the model

        Returns:
            Maximum batch size (at least 1)
        """
        bytes_per_sample = self.max_seq_len * hidden_dim * 4
        hessian_bytes = hidden_dim * hidden_dim * 8  # float64
        available = int(self.target_memory_gb * 1e9) - hessian_bytes
        return max(1, available // bytes_per_sample)

    def iter_batches(self, hidden_dim: int) -> Iterator[CalibrationBatch]:
        """Yield calibration batches sized for available RAM.

        Tokenizes samples on-the-fly to minimize memory usage. Batch
        size is automatically calculated based on the hidden dimension
        and target memory.

        Args:
            hidden_dim: Hidden dimension of the model being calibrated

        Yields:
            CalibrationBatch containing tokenized input IDs and metadata

        Raises:
            ValueError: If the dataset is empty
        """
        batch_size = self.estimate_batch_size(hidden_dim)
        samples = list(self.dataset)

        if len(samples) == 0:
            raise ValueError("Calibration dataset is empty")

        total_batches = (len(samples) + batch_size - 1) // batch_size

        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i : i + batch_size]
            # Tokenize on-the-fly to save memory
            encoded = self.tokenizer(
                batch_samples,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            yield CalibrationBatch(
                input_ids=encoded["input_ids"],
                batch_idx=i // batch_size,
                total_batches=total_batches,
            )
