"""Shard writer for Trellis v3 flat shard format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file as save_numpy_file


class ShardWriter:
    """Writes quantized tensors to sharded safetensors files."""

    def __init__(
        self,
        output_path: Path,
        max_shard_size_gb: float = 4.0,
    ):
        self.output_path = Path(output_path)
        self.max_shard_size = int(max_shard_size_gb * 1024 * 1024 * 1024)
        self.current_shard: dict[str, np.ndarray] = {}
        self.current_shard_size = 0
        self.shard_count = 0
        self.tensor_metadata: list[dict] = []

    def add_tensor(
        self,
        name: str,
        indices: np.ndarray,
        scales: np.ndarray,
        su: np.ndarray,
        sv: np.ndarray,
        bits: int,
        shape: tuple[int, ...],
        mse: float,
    ) -> None:
        """Add a quantized tensor to the current shard."""
        safe_key = name.replace(".", "__")

        indices_size = indices.nbytes
        scales_size = scales.nbytes
        su_size = su.nbytes
        sv_size = sv.nbytes
        total_size = indices_size + scales_size + su_size + sv_size

        if self.current_shard and (self.current_shard_size + total_size > self.max_shard_size):
            self._flush_shard()

        self.current_shard[f"{safe_key}__indices"] = indices
        self.current_shard[f"{safe_key}__scales"] = scales.astype(np.float32)
        self.current_shard[f"{safe_key}__su"] = su.astype(np.float32)
        self.current_shard[f"{safe_key}__sv"] = sv.astype(np.float32)

        self.current_shard_size += total_size

        self.tensor_metadata.append({
            "name": name,
            "bits": bits,
            "shape": list(shape),
            "mse": float(mse),
            "shard": self.shard_count,
        })

    def _flush_shard(self) -> None:
        """Write current shard to disk."""
        if not self.current_shard:
            return

        shard_path = self.output_path / f"trellis_shard_{self.shard_count:05d}.safetensors"
        save_numpy_file(self.current_shard, str(shard_path))

        self.current_shard = {}
        self.current_shard_size = 0
        self.shard_count += 1

    def finalize(self) -> None:
        """Flush remaining tensors and save metadata index."""
        self._flush_shard()

        index_path = self.output_path / "trellis_index.json"
        with open(index_path, "w") as f:
            json.dump({
                "tensors": self.tensor_metadata,
                "num_shards": self.shard_count,
            }, f, indent=2)
