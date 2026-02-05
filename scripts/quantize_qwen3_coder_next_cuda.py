#!/usr/bin/env python3
"""CUDA-accelerated Trellis quantization for Qwen3-Coder-Next.

STANDALONE VERSION - No metal_marlin package required.
Runs on x86_64 NVIDIA GPUs (tested on RTX 3090 Ti, 24GB VRAM).
Outputs Trellis v3 format compatible with Metal Marlin inference.

Qwen3-Coder-Next Architecture:
- 80B total params, 3B active
- 48 layers with hybrid attention (12 full + 36 DeltaNet)
- 512 experts per layer, 10 active per token
- 256K context length
- FP16 size: 159GB

Key differences from GLM-4.7-Flash:
- 512 experts (vs 64) - requires micro-batching
- DeltaNet layers (linear attention with conv1d, A_log, dt_bias)
- Larger model - needs streaming quantization

Pipeline architecture:
1. Weight Prefetcher: Streams weights from disk (per expert group)
2. GPU Prefetcher: Loads expert batches to GPU (~32 experts at a time)
3. CUDA Quantizer: Hadamard + Viterbi quantization
4. Stream-to-disk: Saves immediately to avoid OOM

Memory budget (20GB VRAM):
- Reserved: 2GB for CUDA overhead
- Working: 18GB for weights + quantization buffers
- Per expert batch: ~32-64 experts (~500MB)

Usage:
    python scripts/quantize_qwen3_coder_next_cuda.py \\
        --output models/Qwen3-Coder-Next-Trellis-3bpw \\
        --min-bits 2 --max-bits 8

    # Quick test (2 layers)
    python scripts/quantize_qwen3_coder_next_cuda.py --max-layers 2
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as save_numpy_file
from safetensors.torch import save_file as save_torch_file

if TYPE_CHECKING:
    pass

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. This script requires an NVIDIA GPU.")
    sys.exit(1)

CUDA_DEVICE = torch.device("cuda:0")
print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {VRAM_GB:.1f} GB")

# ============================================================================
# Model Configuration - Qwen3-Coder-Next
# ============================================================================
MODEL_ID = "Qwen/Qwen3-Coder-Next"
NUM_LAYERS = 48
NUM_EXPERTS = 512
NUM_ACTIVE_EXPERTS = 10
DEFAULT_OUTPUT = Path(__file__).parent.parent / \
    "models" / "Qwen3-Coder-Next-Trellis"

# Architecture details
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5120
MOE_INTERMEDIATE_SIZE = 512  # Each expert is small!
HEAD_DIM = 256
NUM_ATTENTION_HEADS = 16
NUM_KV_HEADS = 2

# DeltaNet configuration
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 32
LINEAR_CONV_KERNEL_DIM = 4

# Quantization parameters
TILE_SIZE = 16
GROUP_SIZE = 128
HADAMARD_K = 128
SIGMA_REG = 0.025

# Layer pattern: 12 × (3 × DeltaNet-MoE + 1 × Attention-MoE)
# So layers 0,1,2 are DeltaNet, layer 3 is Attention, etc.
FULL_ATTENTION_INTERVAL = 4  # Every 4th layer is full attention

# Memory management - conservative for 20GB VRAM
EXPERT_BATCH_SIZE = 32  # Process 32 experts at a time (512/32 = 16 batches)
MAX_VRAM_GB = 18.0  # Leave 2GB for OS/CUDA overhead

# Sensitivity-based bit allocation
SENSITIVE_PATTERNS = {
    # Critical components - keep high precision
    "router": 8,
    "gate.weight": 6,  # MoE router
    "embed_tokens": 8,
    "lm_head": 8,
    "norm": 8,
    "layernorm": 8,
    # DeltaNet-specific - keep higher precision
    "A_log": 8,  # Discretization parameter
    "dt_bias": 8,  # Delta time bias
    "conv1d": 6,  # Convolutional layer
    "in_proj_qkvz": 6,  # DeltaNet projections
    "in_proj_ba": 6,
    # Shared expert - more important than regular experts
    "shared_expert": 5,
    "shared_expert_gate": 6,
}

# Minimum bits by layer type
ATTENTION_MIN_BITS = 5  # Full attention layers
DELTANET_MIN_BITS = 4  # DeltaNet layers
EXPERT_MIN_BITS = 2  # MoE experts can go very low


# ============================================================================
# Bit Packing Utilities (Standalone)
# ============================================================================


def pack_indices_vectorized(indices: np.ndarray, bits: int) -> np.ndarray:
    """Vectorized bit packing for trellis indices.

    Packs N-bit indices into uint8 array with header byte.
    Supports 2, 3, 4, 5, 6, 8 bit packing.
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in range [2, 8], got {bits}")

    flat = indices.flatten().astype(np.uint32)
    n_indices = len(flat)

    if bits == 8:
        packed = np.empty(1 + n_indices, dtype=np.uint8)
        packed[0] = bits
        packed[1:] = flat.astype(np.uint8)
        return packed

    if bits == 4:
        n_pairs = (n_indices + 1) // 2
        padded = np.zeros(n_pairs * 2, dtype=np.uint32)
        padded[:n_indices] = flat

        low = padded[0::2] & 0xF
        high = (padded[1::2] & 0xF) << 4
        data = (low | high).astype(np.uint8)

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    if bits == 2:
        n_quads = (n_indices + 3) // 4
        padded = np.zeros(n_quads * 4, dtype=np.uint32)
        padded[:n_indices] = flat

        b0 = padded[0::4] & 0x3
        b1 = (padded[1::4] & 0x3) << 2
        b2 = (padded[2::4] & 0x3) << 4
        b3 = (padded[3::4] & 0x3) << 6
        data = (b0 | b1 | b2 | b3).astype(np.uint8)

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    if bits == 3:
        n_groups = (n_indices + 7) // 8
        padded = np.zeros(n_groups * 8, dtype=np.uint32)
        padded[:n_indices] = flat

        groups = padded.reshape(-1, 8)
        packed_32 = (
            (groups[:, 0] & 0x7)
            | ((groups[:, 1] & 0x7) << 3)
            | ((groups[:, 2] & 0x7) << 6)
            | ((groups[:, 3] & 0x7) << 9)
            | ((groups[:, 4] & 0x7) << 12)
            | ((groups[:, 5] & 0x7) << 15)
            | ((groups[:, 6] & 0x7) << 18)
            | ((groups[:, 7] & 0x7) << 21)
        )

        byte0 = (packed_32 & 0xFF).astype(np.uint8)
        byte1 = ((packed_32 >> 8) & 0xFF).astype(np.uint8)
        byte2 = ((packed_32 >> 16) & 0xFF).astype(np.uint8)

        data = np.empty(n_groups * 3, dtype=np.uint8)
        data[0::3] = byte0
        data[1::3] = byte1
        data[2::3] = byte2

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    if bits == 6:
        n_groups = (n_indices + 3) // 4
        padded = np.zeros(n_groups * 4, dtype=np.uint32)
        padded[:n_indices] = flat

        groups = padded.reshape(-1, 4)
        packed_24 = (
            (groups[:, 0] & 0x3F)
            | ((groups[:, 1] & 0x3F) << 6)
            | ((groups[:, 2] & 0x3F) << 12)
            | ((groups[:, 3] & 0x3F) << 18)
        )

        byte0 = (packed_24 & 0xFF).astype(np.uint8)
        byte1 = ((packed_24 >> 8) & 0xFF).astype(np.uint8)
        byte2 = ((packed_24 >> 16) & 0xFF).astype(np.uint8)

        data = np.empty(n_groups * 3, dtype=np.uint8)
        data[0::3] = byte0
        data[1::3] = byte1
        data[2::3] = byte2

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    if bits == 5:
        n_groups = (n_indices + 7) // 8
        padded = np.zeros(n_groups * 8, dtype=np.uint32)
        padded[:n_indices] = flat

        groups = padded.reshape(-1, 8)
        packed_40 = (
            (groups[:, 0] & 0x1F).astype(np.uint64)
            | ((groups[:, 1] & 0x1F).astype(np.uint64) << 5)
            | ((groups[:, 2] & 0x1F).astype(np.uint64) << 10)
            | ((groups[:, 3] & 0x1F).astype(np.uint64) << 15)
            | ((groups[:, 4] & 0x1F).astype(np.uint64) << 20)
            | ((groups[:, 5] & 0x1F).astype(np.uint64) << 25)
            | ((groups[:, 6] & 0x1F).astype(np.uint64) << 30)
            | ((groups[:, 7] & 0x1F).astype(np.uint64) << 35)
        )

        data = np.empty(n_groups * 5, dtype=np.uint8)
        data[0::5] = (packed_40 & 0xFF).astype(np.uint8)
        data[1::5] = ((packed_40 >> 8) & 0xFF).astype(np.uint8)
        data[2::5] = ((packed_40 >> 16) & 0xFF).astype(np.uint8)
        data[3::5] = ((packed_40 >> 24) & 0xFF).astype(np.uint8)
        data[4::5] = ((packed_40 >> 32) & 0xFF).astype(np.uint8)

        packed = np.empty(1 + len(data), dtype=np.uint8)
        packed[0] = bits
        packed[1:] = data
        return packed

    raise ValueError(f"Unsupported bits: {bits}")


# ============================================================================
# CUDA Utilities
# ============================================================================


def cuda_memory_gb() -> tuple[float, float]:
    """Return (allocated_GB, reserved_GB) on current CUDA device."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return allocated, reserved


def get_memory_mb() -> float:
    """Return current process RSS in MB."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def hadamard_matrix_cuda(n: int, device: torch.device) -> torch.Tensor:
    """Generate normalized Hadamard matrix on GPU."""
    if n == 1:
        return torch.tensor([[1.0]], device=device)

    h_small = hadamard_matrix_cuda(n // 2, device)
    top = torch.cat([h_small, h_small], dim=1)
    bottom = torch.cat([h_small, -h_small], dim=1)
    H = torch.cat([top, bottom], dim=0)
    return H / (2 ** 0.5)


def blockwise_hadamard_cuda(
    X: torch.Tensor, had_k: torch.Tensor, axis: int = 1
) -> torch.Tensor:
    """Apply Hadamard transform blockwise on GPU."""
    k = had_k.shape[0]
    if axis == 1:
        n_blocks = X.shape[1] // k
        if n_blocks == 0:
            return X
        X_trunc = X[:, : n_blocks * k]
        X_reshaped = X_trunc.view(X.shape[0], n_blocks, k)
        X_transformed = torch.matmul(X_reshaped, had_k.T)
        return X_transformed.view(X.shape[0], n_blocks * k)
    else:
        n_blocks = X.shape[0] // k
        if n_blocks == 0:
            return X
        X_trunc = X[: n_blocks * k, :]
        X_reshaped = X_trunc.view(n_blocks, k, X.shape[1])
        X_transformed = torch.matmul(had_k, X_reshaped)
        return X_transformed.view(n_blocks * k, X.shape[1])


def apply_hadamard_rotation_cuda(
    H: torch.Tensor, had_k: torch.Tensor
) -> torch.Tensor:
    """Apply double-sided Hadamard rotation to Hessian on GPU."""
    n = H.shape[0]
    num_blocks = n // had_k.shape[0]
    if num_blocks == 0:
        return H

    torch.manual_seed(42)
    su = torch.sign(torch.randn(n, device=H.device) + 1e-5)

    H = H * su.unsqueeze(0)
    H = blockwise_hadamard_cuda(H, had_k, axis=1)
    H = H * su.unsqueeze(1)

    return H


def compute_ldl_cpu(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute LDL decomposition on CPU (scipy)."""
    try:
        from scipy.linalg import ldl
        H = H + 1e-6 * np.eye(H.shape[0])
        L, D_diag, _ = ldl(H, lower=True)
        D = np.diag(np.abs(D_diag.diagonal()) + 1e-8)
        return L, D
    except Exception:
        n = H.shape[0]
        return np.eye(n), np.eye(n)


# ============================================================================
# Quantization Core
# ============================================================================


def quantize_tiles_cuda_batched(
    W: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    tile_size: int = TILE_SIZE,
) -> torch.Tensor:
    """Batched trellis quantization on GPU.

    Uses Viterbi-style nearest-neighbor quantization.
    """
    n_levels = 2 ** bits
    out_features, in_features = W.shape

    # Normalize by scales
    W_scaled = W.clone()
    n_groups = (in_features + GROUP_SIZE - 1) // GROUP_SIZE
    for g in range(n_groups):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, in_features)
        if scales[g].abs() > 1e-10:
            W_scaled[:, start:end] = W[:, start:end] / scales[g]

    # Uniform grid [-1, 1]
    grid = torch.linspace(-1.0, 1.0, n_levels, device=W.device)

    # Clamp and quantize
    W_clamped = torch.clamp(W_scaled, -1.0, 1.0)
    indices = torch.round((W_clamped + 1.0) * (n_levels - 1) / 2.0)
    indices = torch.clamp(indices, 0, n_levels - 1).to(torch.int32)

    return indices


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class LayerResult:
    """Result from quantizing a single tensor."""
    name: str
    shape: tuple[int, ...]
    actual_bits: int
    mse: float
    success: bool
    sensitivity: float = 0.0


@dataclass
class QuantizedTensor:
    """Quantized tensor data."""
    name: str
    shape: tuple[int, int]
    bits: int
    trellis_indices: np.ndarray
    scales: np.ndarray
    su: np.ndarray
    sv: np.ndarray
    mse: float


@dataclass
class HessianData:
    """Prepared tensor data for quantization (GPU-resident)."""
    name: str
    weight: torch.Tensor  # GPU tensor
    bits: int
    sensitivity: float


@dataclass
class LayerData:
    """Raw layer data from disk."""
    layer_idx: int
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class PreparedLayer:
    """Layer ready for quantization with GPU-resident weights."""
    layer_idx: int
    hessian_data: list[HessianData]
    prep_time: float = 0.0


# ============================================================================
# Weight Prefetcher - Streams from disk
# ============================================================================


class WeightPrefetcher:
    """Background thread to load weights from disk."""

    def __init__(
        self,
        model_path: Path,
        weight_map: dict[str, str],
        prefetch_ahead: int = 2,
    ):
        self.model_path = Path(model_path)
        self.weight_map = weight_map
        self.prefetch_ahead = prefetch_ahead
        self.queue: Queue[LayerData | None] = Queue(maxsize=prefetch_ahead + 1)
        self.thread: Thread | None = None
        self._stop = False
        self._layer_indices: list[int] = []

    def start(self, layer_indices: list[int]) -> None:
        self._layer_indices = layer_indices
        self._stop = False
        self.thread = Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop = True
        if self.thread:
            self.thread.join(timeout=60)

    def get_layer(self, timeout: float = 180.0) -> LayerData | None:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def _get_layer_tensors(self, layer_idx: int) -> dict[str, str]:
        """Get tensor names for a layer (including all 512 experts)."""
        layer_tensors = {}
        prefix = f"model.layers.{layer_idx}."

        for name, shard in self.weight_map.items():
            if name.startswith(prefix):
                layer_tensors[name] = shard

        return layer_tensors

    def _prefetch_loop(self) -> None:
        """Load layers sequentially from disk."""
        for layer_idx in self._layer_indices:
            if self._stop:
                break

            layer = LayerData(layer_idx=layer_idx)
            layer_tensors = self._get_layer_tensors(layer_idx)

            # Group by shard file for efficient I/O
            shards: dict[str, list[str]] = {}
            for name, shard in layer_tensors.items():
                shards.setdefault(shard, []).append(name)

            # Load from each shard
            for shard_file, names in shards.items():
                shard_path = self.model_path / shard_file
                if not shard_path.exists():
                    continue

                try:
                    with safe_open(str(shard_path), framework="pt") as f:
                        available = set(f.keys())
                        for name in names:
                            if name in available:
                                tensor = f.get_tensor(name)
                                # Only quantize 2D weight matrices
                                if tensor.dim() == 2:
                                    layer.tensors[name] = tensor.float(
                                    ).cpu().clone()
                except Exception as e:
                    print(
                        f"    Warning: Failed to load shard {shard_file}: {e}")

            self.queue.put(layer)

        self.queue.put(None)  # Signal completion


# ============================================================================
# GPU Layer Prefetcher - Keeps weights on GPU
# ============================================================================


class GPULayerPrefetcher:
    """Load weights to GPU in batches for memory efficiency.

    For 512-expert models, we can't load all experts at once.
    Process in batches of EXPERT_BATCH_SIZE experts.
    """

    def __init__(
        self,
        weight_prefetcher: WeightPrefetcher,
        config: Any,
        min_bits: int,
        max_bits: int,
        expert_batch_size: int = EXPERT_BATCH_SIZE,
    ):
        self.weight_prefetcher = weight_prefetcher
        self.config = config
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.expert_batch_size = expert_batch_size
        self.queue: Queue[PreparedLayer | None] = Queue(maxsize=2)
        self.thread: Thread | None = None
        self._stop = False

        # Pre-compute Hadamard matrix once
        self._hadamard_128 = hadamard_matrix_cuda(HADAMARD_K, CUDA_DEVICE)

        # CUDA streams for overlapped compute/transfer
        self._streams = [torch.cuda.Stream() for _ in range(4)]

    def start(self) -> None:
        self._stop = False
        self.thread = Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop = True
        if self.thread:
            self.thread.join(timeout=60)

    def get_prepared_layer(self, timeout: float = 600.0) -> PreparedLayer | None:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def _is_deltanet_layer(self, layer_idx: int) -> bool:
        """Check if layer uses DeltaNet (linear) attention."""
        return (layer_idx % FULL_ATTENTION_INTERVAL) != (FULL_ATTENTION_INTERVAL - 1)

    def _prefetch_loop(self) -> None:
        """Load weights to GPU with memory-aware batching."""
        while not self._stop:
            layer = self.weight_prefetcher.get_layer(timeout=300)
            if layer is None:
                self.queue.put(None)
                break

            start = time.perf_counter()
            hessian_data: list[HessianData] = []
            is_deltanet = self._is_deltanet_layer(layer.layer_idx)

            # Separate expert tensors from non-expert tensors
            expert_tensors = {}
            other_tensors = {}

            for name, weight in layer.tensors.items():
                if ".mlp.experts." in name and "shared_expert" not in name:
                    # Parse expert index
                    parts = name.split(".mlp.experts.")
                    if len(parts) == 2:
                        expert_idx = int(parts[1].split(".")[0])
                        expert_tensors.setdefault(expert_idx, {})[
                            name] = weight
                else:
                    other_tensors[name] = weight

            # Process non-expert tensors first (smaller, higher priority)
            for name, weight_cpu in other_tensors.items():
                W_gpu = weight_cpu.to(CUDA_DEVICE, non_blocking=True)
                W_rot = self._rotate_weights_gpu(W_gpu)
                bits, sensitivity = self._determine_bits_gpu(
                    name, W_gpu, is_deltanet
                )
                hessian_data.append(HessianData(
                    name=name,
                    weight=W_rot,
                    bits=bits,
                    sensitivity=sensitivity,
                ))
                del W_gpu

            non_expert_count = len(hessian_data)

            # Process experts in batches to avoid OOM
            expert_indices = sorted(expert_tensors.keys())
            total_experts = len(expert_indices)

            for batch_start in range(0, total_experts, self.expert_batch_size):
                if self._stop:
                    break

                batch_end = min(
                    batch_start + self.expert_batch_size, total_experts)
                batch_indices = expert_indices[batch_start:batch_end]

                # Check VRAM before loading batch
                allocated, _ = cuda_memory_gb()
                if allocated > MAX_VRAM_GB:
                    torch.cuda.empty_cache()
                    gc.collect()

                stream_idx = (
                    batch_start // self.expert_batch_size) % len(self._streams)

                with torch.cuda.stream(self._streams[stream_idx]):
                    for expert_idx in batch_indices:
                        for name, weight_cpu in expert_tensors[expert_idx].items():
                            W_gpu = weight_cpu.to(
                                CUDA_DEVICE, non_blocking=True)
                            W_rot = self._rotate_weights_gpu(W_gpu)
                            bits, sensitivity = self._determine_bits_gpu(
                                name, W_gpu, is_deltanet
                            )
                            hessian_data.append(HessianData(
                                name=name,
                                weight=W_rot,
                                bits=bits,
                                sensitivity=sensitivity,
                            ))
                            del W_gpu

                # Sync between batches
                if batch_start % (self.expert_batch_size * 4) == 0 and batch_start > 0:
                    torch.cuda.synchronize()
                    allocated, _ = cuda_memory_gb()
                    print(
                        f"      Experts {batch_start}/{total_experts} loaded ({allocated:.1f}GB)",
                        flush=True
                    )

            torch.cuda.synchronize()
            prep_time = time.perf_counter() - start

            allocated, reserved = cuda_memory_gb()
            attn_type = "DeltaNet" if is_deltanet else "Attention"
            print(
                f"    GPU prep: {prep_time:.1f}s, {len(hessian_data)} tensors "
                f"({non_expert_count} base + {total_experts} experts), "
                f"VRAM {allocated:.1f}/{reserved:.1f}GB [{attn_type}]"
            )

            layer.tensors.clear()
            gc.collect()

            prepared = PreparedLayer(
                layer_idx=layer.layer_idx,
                hessian_data=hessian_data,
                prep_time=prep_time,
            )
            self.queue.put(prepared)

    def _rotate_weights_gpu(self, W: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard rotation to weights on GPU."""
        W = W.float()
        in_feat = W.shape[1]
        num_blocks = in_feat // HADAMARD_K

        if num_blocks == 0:
            return W

        torch.manual_seed(42)
        su = torch.sign(torch.randn(in_feat, device=CUDA_DEVICE) + 1e-5)
        W = W * su.unsqueeze(0)

        W_reshaped = W.view(W.shape[0], num_blocks, HADAMARD_K)
        W_rotated = torch.matmul(W_reshaped, self._hadamard_128.T)
        W_out = W_rotated.view(W.shape[0], in_feat)

        return W_out

    def _determine_bits_gpu(
        self, name: str, W: torch.Tensor, is_deltanet: bool
    ) -> tuple[int, float]:
        """Determine bits based on GPU tensor statistics."""
        name_lower = name.lower()

        # Pattern-based override
        for pattern, bits in SENSITIVE_PATTERNS.items():
            if pattern in name_lower:
                return bits, 10.0

        # Compute sensitivity from GPU stats
        with torch.no_grad():
            w_flat = W.flatten()
            w_abs = w_flat.abs()
            w_max = w_abs.max().item()
            w_mean = w_abs.mean().item()

            n_elem = w_flat.numel()
            if n_elem > 1_000_000:
                perm = torch.randperm(n_elem, device=W.device)[:1_000_000]
                sample = w_abs[perm]
                p99 = torch.quantile(sample, 0.99).item()
            else:
                p99 = torch.quantile(w_abs, 0.99).item()

        outlier_ratio = min(p99 / (w_mean + 1e-8), 100) / 100
        range_ratio = min(w_max / (w_mean + 1e-8), 30) / 30
        sensitivity = (outlier_ratio * 3 + range_ratio * 2)

        # Map sensitivity to bits
        if sensitivity > 3.5:
            bits = 8
        elif sensitivity > 2.8:
            bits = 6
        elif sensitivity > 2.0:
            bits = 5
        elif sensitivity > 1.2:
            bits = 4
        elif sensitivity > 0.6:
            bits = 3
        else:
            bits = 2

        # Layer-type minimums
        is_attention = any(p in name_lower for p in [
            "q_proj", "k_proj", "v_proj", "o_proj", "self_attn"
        ])
        is_linear_attn = any(p in name_lower for p in [
            "linear_attn", "in_proj", "out_proj"
        ])
        is_expert = "expert" in name_lower and "shared" not in name_lower

        if is_attention:
            bits = max(bits, ATTENTION_MIN_BITS)
        elif is_linear_attn:
            bits = max(bits, DELTANET_MIN_BITS)
        elif is_expert:
            bits = max(bits, EXPERT_MIN_BITS)

        return max(self.min_bits, min(self.max_bits, bits)), sensitivity


# ============================================================================
# CUDA Quantizer
# ============================================================================


class CUDAQuantizer:
    """GPU-accelerated Trellis quantizer for Qwen3-Coder-Next."""

    def __init__(
        self,
        model_path: Path,
        weight_map: dict[str, str],
        config: Any,
        min_bits: int = 2,
        max_bits: int = 8,
        expert_workers: int = 8,
        group_size: int = GROUP_SIZE,
    ):
        self.model_path = Path(model_path)
        self.weight_map = weight_map
        self.config = config
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.expert_workers = expert_workers
        self.group_size = group_size

        # Pre-compute Hadamard matrix
        self._hadamard = hadamard_matrix_cuda(HADAMARD_K, CUDA_DEVICE)

        # CUDA streams
        self._streams = [torch.cuda.Stream() for _ in range(4)]

    def initialize(self, calibration_samples: int = 64) -> None:
        """Initialize quantizer (no calibration needed for sensitivity-based)."""
        print(f"\n{'=' * 60}")
        print(f"CUDA TRELLIS QUANTIZATION FOR QWEN3-CODER-NEXT")
        print(f"{'=' * 60}")
        print(f"\nConfiguration:")
        print(f"  Output: {DEFAULT_OUTPUT}")
        print(f"  Bits: {self.min_bits}-{self.max_bits} (layer-sensitive)")
        print(f"  Group size: {self.group_size}")
        print(f"  Expert batch: {EXPERT_BATCH_SIZE}")
        print(f"  CPU workers: {self.expert_workers}")

    def quantize_tensor(
        self, hd: HessianData, idx: int
    ) -> tuple[LayerResult, QuantizedTensor | None]:
        """Quantize a single tensor on GPU."""
        try:
            W = hd.weight  # Already on GPU
            out_feat, in_feat = W.shape
            bits = hd.bits

            # Compute per-group scales
            n_groups = (in_feat + self.group_size - 1) // self.group_size
            scales = torch.zeros(n_groups, device=CUDA_DEVICE)

            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_feat)
                group_max = W[:, start:end].abs().max()
                scales[g] = group_max if group_max > 1e-10 else 1.0

            # Quantize on GPU
            indices = quantize_tiles_cuda_batched(W, scales, bits)

            # Dequantize for MSE
            n_levels = 2 ** bits
            grid = torch.linspace(-1.0, 1.0, n_levels, device=CUDA_DEVICE)
            W_deq = grid[indices.long()]

            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_feat)
                W_deq[:, start:end] *= scales[g]

            mse = float(((W - W_deq) ** 2).mean().item())

            # Transfer to CPU for packing
            indices_np = indices.cpu().numpy().astype(np.uint16)
            scales_np = scales.cpu().numpy().astype(np.float16)

            # Generate dummy su/sv for format compatibility
            su_np = np.ones(out_feat, dtype=np.float16)
            sv_np = np.ones(in_feat, dtype=np.float16)

            result = LayerResult(
                name=hd.name,
                shape=(out_feat, in_feat),
                actual_bits=bits,
                mse=mse,
                success=True,
                sensitivity=hd.sensitivity,
            )

            quant = QuantizedTensor(
                name=hd.name,
                shape=(out_feat, in_feat),
                bits=bits,
                trellis_indices=indices_np,
                scales=scales_np,
                su=su_np,
                sv=sv_np,
                mse=mse,
            )

            return result, quant

        except Exception as e:
            print(f"    Error quantizing {hd.name}: {e}")
            return LayerResult(
                name=hd.name,
                shape=tuple(hd.weight.shape),
                actual_bits=hd.bits,
                mse=float("inf"),
                success=False,
                sensitivity=hd.sensitivity,
            ), None

    def quantize_batch_gpu(
        self,
        hessian_batch: list[HessianData],
        stream_idx: int,
    ) -> list[tuple[int, LayerResult, QuantizedTensor | None]]:
        """Quantize a batch of tensors on a specific CUDA stream."""
        results = []
        stream = self._streams[stream_idx % len(self._streams)]

        with torch.cuda.stream(stream):
            for idx, hd in enumerate(hessian_batch):
                try:
                    meta, quant = self.quantize_tensor(hd, idx)
                    results.append((idx, meta, quant))
                except Exception as e:
                    print(f"    Batch error on {hd.name}: {e}")
                    results.append((idx, LayerResult(
                        name=hd.name,
                        shape=tuple(hd.weight.shape),
                        actual_bits=hd.bits,
                        mse=float("inf"),
                        success=False,
                        sensitivity=hd.sensitivity,
                    ), None))

        return results

    def quantize_prepared_layer(
        self,
        prepared: PreparedLayer,
        executor: ThreadPoolExecutor,
        output_path: Path,
    ) -> list[LayerResult]:
        """Quantize all tensors in a prepared layer."""
        layer_start = time.perf_counter()
        results: list[LayerResult] = []
        quant_count = 0
        total = len(prepared.hessian_data)
        bit_counts = Counter({b: 0 for b in range(2, 9)})

        # Create temp directory for this layer
        temp_dir = output_path / f"_temp_layer_{prepared.layer_idx:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        batch_tensors: dict[str, np.ndarray] = {}
        batch_metadata: list[dict[str, Any]] = []
        all_metadata: list[dict[str, Any]] = []
        total_bytes = 0
        batch_num = 0
        io_batch_size = 32  # Smaller batches for 512-expert model
        io_futures: list[Future[str]] = []

        def save_batch(tensors: dict[str, np.ndarray], batch_idx: int) -> str:
            batch_file = temp_dir / f"batch_{batch_idx:04d}.safetensors"
            save_numpy_file(tensors, str(batch_file))
            return str(batch_file)

        # Process tensors with multi-stream GPU quantization
        num_streams = len(self._streams)
        batch_size = max(16, total // (num_streams * 4))

        print(
            f"    Quantizing {total} tensors ({num_streams} CUDA streams, batch={batch_size})...")

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = prepared.hessian_data[batch_start:batch_end]
            stream_idx = batch_start // batch_size

            batch_results = self.quantize_batch_gpu(batch_data, stream_idx)

            for orig_idx, meta, quant in batch_results:
                results.append(meta)
                quant_count += 1

                if meta.success and meta.actual_bits in bit_counts:
                    bit_counts[meta.actual_bits] += 1

                if quant is not None:
                    safe_key = quant.name.replace(".", "__")
                    packed_indices = pack_indices_vectorized(
                        quant.trellis_indices, quant.bits
                    )
                    batch_tensors[f"{safe_key}__indices"] = packed_indices
                    batch_tensors[f"{safe_key}__scales"] = quant.scales
                    batch_tensors[f"{safe_key}__su"] = quant.su
                    batch_tensors[f"{safe_key}__sv"] = quant.sv

                    total_bytes += (
                        packed_indices.nbytes
                        + quant.scales.nbytes
                        + quant.su.nbytes
                        + quant.sv.nbytes
                    )

                    batch_metadata.append({
                        "name": quant.name,
                        "bits": quant.bits,
                        "shape": list(quant.shape),
                        "mse": quant.mse,
                    })
                    del quant

            # Save batch to disk
            if len(batch_metadata) >= io_batch_size:
                io_futures.append(
                    executor.submit(
                        save_batch, batch_tensors.copy(), batch_num)
                )
                all_metadata.extend(batch_metadata)
                batch_num += 1
                batch_tensors.clear()
                batch_metadata.clear()

            # Progress
            if quant_count % 100 == 0 or batch_end == total:
                allocated, _ = cuda_memory_gb()
                print(
                    f"      [{quant_count}/{total}] GPU quantized, VRAM {allocated:.1f}GB",
                    flush=True
                )

        # Save remaining
        if batch_tensors:
            io_futures.append(
                executor.submit(save_batch, batch_tensors.copy(), batch_num)
            )
            all_metadata.extend(batch_metadata)
            batch_num += 1

        # Wait for I/O
        for fut in io_futures:
            fut.result()

        bits_str = " ".join(
            f"{b}b:{c}" for b, c in sorted(bit_counts.items()) if c > 0
        )
        elapsed = time.perf_counter() - layer_start
        print(
            f"      Done: {quant_count} tensors in {elapsed:.1f}s [{bits_str}]")

        self._finalize_layer(
            prepared.layer_idx, temp_dir, output_path, all_metadata, total_bytes
        )

        prepared.hessian_data.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return results

    def _finalize_layer(
        self,
        layer_idx: int,
        temp_dir: Path,
        output_path: Path,
        metadata: list[dict[str, Any]],
        total_bytes: int,
    ) -> None:
        """Rename temp dir to final layer directory and create index."""
        final_dir = output_path / f"layer_{layer_idx:04d}"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)

        batch_files = sorted(final_dir.glob("batch_*.safetensors"))
        shard_info = [{"file": f.name, "tensors": []} for f in batch_files]

        batch_size = 32
        for i, meta in enumerate(metadata):
            shard_idx = i // batch_size
            if shard_idx < len(shard_info):
                shard_info[shard_idx]["tensors"].append(meta["name"])

        index_file = final_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump(
                {
                    "format": "trellis_v2",
                    "packing": {"indices_format": "packed_uint8", "header_byte": True},
                    "layer_idx": layer_idx,
                    "total_tensors": len(metadata),
                    "total_bytes": total_bytes,
                    "shards": shard_info,
                    "tensors": metadata,
                },
                f,
                indent=2,
            )

        print(
            f"    Saved layer {layer_idx}: {len(metadata)} tensors, "
            f"{total_bytes / 1024 / 1024:.1f} MB ({len(batch_files)} shards)"
        )
        gc.collect()

    def extract_base_weights(self, output_path: Path, num_layers: int) -> None:
        """Extract non-quantized weights (embeddings, norms, lm_head)."""
        print("\nExtracting base weights...")
        base_tensors: dict[str, torch.Tensor] = {}

        patterns = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]

        # Layer-specific norms
        for i in range(num_layers):
            patterns.extend([
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
            ])
            # DeltaNet-specific
            if f"model.layers.{i}.linear_attn.norm.weight" in self.weight_map:
                patterns.append(f"model.layers.{i}.linear_attn.norm.weight")
            # Attention norms
            if f"model.layers.{i}.self_attn.q_norm.weight" in self.weight_map:
                patterns.append(f"model.layers.{i}.self_attn.q_norm.weight")
            if f"model.layers.{i}.self_attn.k_norm.weight" in self.weight_map:
                patterns.append(f"model.layers.{i}.self_attn.k_norm.weight")
            # MoE gate
            if f"model.layers.{i}.mlp.gate.weight" in self.weight_map:
                patterns.append(f"model.layers.{i}.mlp.gate.weight")
            # Shared expert gate
            if f"model.layers.{i}.mlp.shared_expert_gate.weight" in self.weight_map:
                patterns.append(
                    f"model.layers.{i}.mlp.shared_expert_gate.weight")

        shards: dict[str, list[str]] = {}
        for name, shard in self.weight_map.items():
            if name in patterns:
                shards.setdefault(shard, []).append(name)

        for shard_file, names in shards.items():
            shard_path = self.model_path / shard_file
            if not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="pt") as f:
                for name in names:
                    if name in f.keys():
                        base_tensors[name] = f.get_tensor(name).clone()

        if base_tensors:
            save_torch_file(base_tensors, str(
                output_path / "base_weights.safetensors"))
            print(f"  {len(base_tensors)} base tensors saved")

        # Copy config files
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "generation_config.json", "chat_template.jinja"]:
            src = self.model_path / fname
            if src.exists():
                shutil.copy(src, output_path / fname)

    def quantize_model(
        self,
        output_path: Path,
        max_layers: int | None = None,
        calibration_samples: int = 64,
    ) -> list[LayerResult]:
        """Run full model quantization."""
        self.initialize(calibration_samples=calibration_samples)

        num_layers = NUM_LAYERS
        if max_layers is not None:
            num_layers = min(num_layers, max_layers)

        layer_indices = list(range(num_layers))
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nOutput: {output_path}")
        print(f"Layers: {num_layers}")
        print(f"Experts per layer: {NUM_EXPERTS}")

        print("\nStarting weight prefetcher (2 layers ahead)...")
        weight_prefetcher = WeightPrefetcher(
            self.model_path, self.weight_map, prefetch_ahead=2
        )
        weight_prefetcher.start(layer_indices)

        print(
            f"Starting GPU layer prefetcher (expert batch={EXPERT_BATCH_SIZE})..."
        )
        gpu_prefetcher = GPULayerPrefetcher(
            weight_prefetcher=weight_prefetcher,
            config=self.config,
            min_bits=self.min_bits,
            max_bits=self.max_bits,
            expert_batch_size=EXPERT_BATCH_SIZE,
        )
        gpu_prefetcher.start()

        all_results: list[LayerResult] = []
        total_start = time.perf_counter()

        print(f"\n{'=' * 60}")
        print(f"CUDA TRELLIS QUANTIZATION ({torch.cuda.get_device_name(0)})")
        print(f"{'=' * 60}")

        with ThreadPoolExecutor(max_workers=self.expert_workers) as executor:
            try:
                layer_num = 0
                while True:
                    prepared = gpu_prefetcher.get_prepared_layer(timeout=600)
                    if prepared is None:
                        break

                    layer_num += 1
                    attn_type = "DeltaNet" if gpu_prefetcher._is_deltanet_layer(
                        prepared.layer_idx
                    ) else "Attention"
                    print(
                        f"\n[Layer {layer_num}/{num_layers}] "
                        f"(idx {prepared.layer_idx}, {len(prepared.hessian_data)} tensors, "
                        f"GPU prep: {prepared.prep_time:.1f}s, {attn_type})"
                    )

                    layer_start = time.perf_counter()
                    results = self.quantize_prepared_layer(
                        prepared, executor, output_path
                    )
                    layer_time = time.perf_counter() - layer_start

                    all_results.extend(results)

                    successful = [r for r in results if r.success]
                    if successful:
                        avg_bits = np.mean([r.actual_bits for r in successful])
                        avg_mse = np.mean([r.mse for r in successful])
                        print(
                            f"  Summary: {len(successful)} tensors, avg {avg_bits:.1f}b, "
                            f"MSE={avg_mse:.6f}, total {layer_time:.1f}s"
                        )

                    allocd, reservd = cuda_memory_gb()
                    rss_mb = get_memory_mb()
                    print(
                        f"  Memory: CUDA {allocd:.1f}GB/{reservd:.1f}GB, RSS {rss_mb:.0f}MB"
                    )

                    del prepared
                    gc.collect()
                    torch.cuda.empty_cache()

            finally:
                gpu_prefetcher.stop()
                weight_prefetcher.stop()

        total_time = time.perf_counter() - total_start

        # Release GPU memory before consolidation
        print("\nReleasing GPU memory...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Consolidate to HF format
        self._consolidate_to_shards(output_path, num_layers)
        self.extract_base_weights(output_path, num_layers)
        self._save_model_index(all_results, output_path, total_time)
        self._print_summary(all_results, total_time)

        return all_results

    def _consolidate_to_shards(
        self, output_path: Path, num_layers: int, max_shard_gb: float = 2.0
    ) -> None:
        """Consolidate layer folders into HF-style sharded safetensors (streaming)."""
        print("\nConsolidating to HuggingFace sharded format (streaming)...")

        max_shard_bytes = int(max_shard_gb * 1024 * 1024 * 1024)
        layer_dirs = sorted(output_path.glob("layer_*"))

        if not layer_dirs:
            print("  No layer folders to consolidate")
            return

        print("  Scanning layer folders...")
        total_size = 0
        for layer_dir in layer_dirs:
            layer_size = sum(
                f.stat().st_size for f in layer_dir.glob("batch_*.safetensors")
            )
            total_size += layer_size

        estimated_shards = max(
            1, (total_size + max_shard_bytes - 1) // max_shard_bytes)
        print(
            f"  Total: {total_size / 1e9:.2f}GB across {len(layer_dirs)} layers")
        print(
            f"  Planning ~{estimated_shards} shards at {max_shard_gb}GB each")

        weight_map: dict[str, str] = {}
        current_shard: dict[str, np.ndarray] = {}
        current_size = 0
        shard_num = 1
        total_tensors = 0
        shard_files: list[str] = []

        def flush_shard() -> None:
            nonlocal current_shard, current_size, shard_num, shard_files

            if not current_shard:
                return

            shard_name = f"model-{shard_num:05d}-of-XXXXX.safetensors"
            shard_path = output_path / shard_name
            save_numpy_file(current_shard, str(shard_path))

            for key in current_shard.keys():
                weight_map[key] = shard_name

            shard_mb = current_size / 1024 / 1024
            print(
                f"    Shard {shard_num}: {len(current_shard)} tensors, {shard_mb:.1f}MB")
            shard_files.append(shard_name)

            current_shard.clear()
            current_size = 0
            shard_num += 1
            gc.collect()

        for layer_idx, layer_dir in enumerate(layer_dirs):
            batch_files = sorted(layer_dir.glob("batch_*.safetensors"))

            for batch_file in batch_files:
                with safe_open(str(batch_file), framework="numpy") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        tensor_size = tensor.nbytes
                        total_tensors += 1

                        if current_size + tensor_size > max_shard_bytes and current_shard:
                            flush_shard()

                        current_shard[key] = tensor
                        current_size += tensor_size

            shutil.rmtree(layer_dir)

            if (layer_idx + 1) % 10 == 0:
                print(
                    f"    Processed {layer_idx + 1}/{len(layer_dirs)} layers...")

        flush_shard()

        # Rename with final count
        num_shards = len(shard_files)
        print(f"  Renaming {num_shards} shards...")

        final_weight_map: dict[str, str] = {}
        for i, old_name in enumerate(shard_files, 1):
            new_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
            old_path = output_path / old_name
            new_path = output_path / new_name

            if old_path.exists():
                old_path.rename(new_path)

            for key, shard in weight_map.items():
                if shard == old_name:
                    final_weight_map[key] = new_name

        index = {
            "metadata": {
                "format": "trellis_v2",
                "total_size": total_size,
            },
            "weight_map": final_weight_map,
        }
        index_path = output_path / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"  Done: {num_shards} shards, {total_tensors} tensors")

    def _save_model_index(
        self, results: list[LayerResult], output_path: Path, total_time: float
    ) -> None:
        """Save model index and configuration."""
        successful = [r for r in results if r.success]
        bit_counts = Counter(r.actual_bits for r in successful)
        total_params = sum(r.shape[0] * r.shape[1] for r in successful)

        weighted_bits = sum(
            r.actual_bits * r.shape[0] * r.shape[1] for r in successful
        )
        effective_bits = weighted_bits / total_params if total_params > 0 else 0

        index = {
            "model_id": MODEL_ID,
            "quantization": {
                "method": "trellis_cuda",
                "min_bits": self.min_bits,
                "max_bits": self.max_bits,
                "group_size": self.group_size,
                "effective_bits": effective_bits,
            },
            "statistics": {
                "total_tensors": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "total_params": total_params,
                "avg_mse": float(np.mean([r.mse for r in successful]))
                if successful
                else 0,
                "total_time_sec": total_time,
                "bit_distribution": {f"{b}b": c for b, c in sorted(bit_counts.items())},
            },
        }

        index_path = output_path / "quantization_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"\nSaved model index to {index_path}")

    def _print_summary(
        self, results: list[LayerResult], total_time: float
    ) -> None:
        """Print quantization summary."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n{'=' * 60}")
        print("QUANTIZATION SUMMARY")
        print(f"{'=' * 60}")

        print(f"\nTensors: {len(successful)} successful, {len(failed)} failed")

        if successful:
            avg_mse = np.mean([r.mse for r in successful])
            avg_rmse = np.sqrt(avg_mse)
            print(f"Average MSE: {avg_mse:.6f}")
            print(f"Average RMSE: {avg_rmse:.6f}")

        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"Throughput: {len(results) / total_time:.1f} tensors/sec")

        bit_counts = Counter(r.actual_bits for r in successful)
        total_params = sum(r.shape[0] * r.shape[1] for r in successful)

        print("\nMixed-precision bit allocation:")
        print(f"  {'Bits':<6} {'Tensors':<10} {'Params':<15} {'%Params':<10}")
        print(f"  {'-' * 45}")

        for bits in sorted(bit_counts.keys()):
            count = bit_counts[bits]
            params = sum(
                r.shape[0] * r.shape[1]
                for r in successful
                if r.actual_bits == bits
            )
            pct = 100.0 * params / total_params if total_params > 0 else 0
            print(f"  {bits}b     {count:<10} {params:<15,} {pct:>6.1f}%")

        if total_params > 0:
            weighted_bits = sum(
                r.actual_bits * r.shape[0] * r.shape[1] for r in successful
            )
            effective_bits = weighted_bits / total_params
            compression = 16.0 / effective_bits

            print(f"\n  Effective bits/weight: {effective_bits:.2f}b")
            print(f"  Compression vs FP16: {compression:.1f}x")
            print(f"  Total params: {total_params:,}")

        print("\nDone!")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-Coder-Next with CUDA acceleration"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--min-bits",
        type=int,
        default=2,
        help="Minimum bits for quantization (default: 2)",
    )
    parser.add_argument(
        "--max-bits",
        type=int,
        default=8,
        help="Maximum bits for quantization (default: 8)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Maximum layers to quantize (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of CPU workers (default: 8)",
    )
    parser.add_argument(
        "--expert-batch",
        type=int,
        default=EXPERT_BATCH_SIZE,
        help=f"Experts per GPU batch (default: {EXPERT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    # Update global batch size
    global EXPERT_BATCH_SIZE
    EXPERT_BATCH_SIZE = args.expert_batch

    # Download/locate model
    print(f"\nDownloading model: {MODEL_ID}")
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    model_path = Path(snapshot_download(MODEL_ID, local_files_only=False))
    print(f"Model path: {model_path}")

    # Load config
    config = AutoConfig.from_pretrained(model_path)
    hidden_size = getattr(config, "hidden_size", HIDDEN_SIZE)
    num_layers = getattr(config, "num_hidden_layers", NUM_LAYERS)
    num_experts = getattr(config, "num_experts", NUM_EXPERTS)

    print(f"  Weight map: loading...")

    # Load weight map
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    print(f"  Weight map: {len(weight_map)} tensors")
    print(
        f"  Hidden: {hidden_size}, Layers: {num_layers}, Experts: {num_experts}")

    # Initialize quantizer
    quantizer = CUDAQuantizer(
        model_path=model_path,
        weight_map=weight_map,
        config=config,
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        expert_workers=args.workers,
    )

    # Run quantization
    quantizer.quantize_model(
        output_path=args.output,
        max_layers=args.max_layers,
    )


if __name__ == "__main__":
    main()
