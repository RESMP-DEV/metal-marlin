#!/usr/bin/env python3
"""Benchmark KV cache memory layouts for decode performance.

Compares memory layouts for KV cache, including multi-layer consolidated layouts:

Per-layer layouts (list of tensors, one per layer):
1. BHSD [batch, num_heads, seq_len, head_dim] - current layout
   - Heads contiguous, good for head-parallel GEMM
   - Standard PyTorch/HuggingFace convention
   - K vectors at k_base + k_idx * head_dim are contiguous

2. BSHD [batch, seq_len, num_heads, head_dim] - better locality
   - Tokens contiguous, each position has all heads together
   - Better cache locality for single-token decode writes
   - Requires transpose for attention kernels

3. HBSD [num_heads, batch, seq_len, head_dim] - per-head access
   - Each head's data is fully contiguous
   - Optimal for per-head parallelism
   - No broadcast needed for GQA within a head

Multi-layer consolidated layouts (single tensor for all layers):
4. LHSD [num_layers, num_heads, seq_len, head_dim] - contiguous layout
   - All layers in single allocation (better memory management)
   - Same access pattern as BHSD within each layer
   - Layer prefetching possible

5. LSHD [num_layers, seq_len, num_heads, head_dim] - transposed layout
   - All layers in single allocation
   - Better locality for single-token writes across heads
   - May benefit from coalesced writes during decode

Tests measure:
- Decode update: single-token append performance
- Attention read: Q @ K^T access pattern (strided K vector reads)
- Write coalescing: memory write efficiency
- Multi-layer iteration: cost of sequential layer access

Run:
    cd contrib/metal_marlin
    uv run python scripts/benchmark_kv_layout.py
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Literal

import torch

# Warmup iterations to prime Metal shader caches
WARMUP_ITERS = 20
BENCHMARK_ITERS = 200


@dataclass
class CacheConfig:
    """Configuration matching typical LLM models."""

    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8  # GQA
    head_dim: int = 128
    max_seq_len: int = 4096
    batch_size: int = 1


LayoutType = Literal["BHSD", "BSHD", "HBSD"]
MultiLayerLayoutType = Literal["LHSD", "LSHD"]


def sync_mps() -> None:
    """Synchronize MPS operations."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def create_cache_tensor(
    layout: LayoutType, config: CacheConfig, device: str
) -> torch.Tensor:
    """Create cache tensor with specified layout."""
    B = config.batch_size
    H = config.num_kv_heads
    S = config.max_seq_len
    D = config.head_dim

    if layout == "BHSD":
        shape = (B, H, S, D)
    elif layout == "BSHD":
        shape = (B, S, H, D)
    elif layout == "HBSD":
        shape = (H, B, S, D)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return torch.zeros(shape, dtype=torch.float16, device=device)


def decode_update_bhsd(
    cache: torch.Tensor, token: torch.Tensor, pos: int
) -> torch.Tensor:
    """Update BHSD cache at position, return slice for attention."""
    # token: [B, H, 1, D]
    cache[:, :, pos : pos + 1, :] = token
    # Return slice [B, H, :pos+1, D] - already contiguous along D
    return cache[:, :, : pos + 1, :]


def decode_update_bshd(
    cache: torch.Tensor, token: torch.Tensor, pos: int
) -> torch.Tensor:
    """Update BSHD cache at position, return slice for attention."""
    # token: [B, H, 1, D] -> need [B, 1, H, D]
    token_t = token.permute(0, 2, 1, 3)
    cache[:, pos : pos + 1, :, :] = token_t
    # Return transposed back to [B, H, :pos+1, D] for attention
    return cache[:, : pos + 1, :, :].permute(0, 2, 1, 3)


def decode_update_hbsd(
    cache: torch.Tensor, token: torch.Tensor, pos: int
) -> torch.Tensor:
    """Update HBSD cache at position, return slice for attention."""
    # token: [B, H, 1, D] -> need [H, B, 1, D]
    token_t = token.permute(1, 0, 2, 3)
    cache[:, :, pos : pos + 1, :] = token_t
    # Return transposed back to [B, H, :pos+1, D] for attention
    return cache[:, :, : pos + 1, :].permute(1, 0, 2, 3)


def benchmark_decode_update(layout: LayoutType, config: CacheConfig) -> dict[str, float]:
    """Benchmark single-token decode updates across all layers.

    This measures the core decode loop: append one KV pair per layer,
    then read the full cache slice for attention.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Create caches for all layers
    caches = [create_cache_tensor(layout, config, device) for _ in range(config.num_layers)]

    # Prefill with some tokens
    prefill_len = 1024
    B = config.batch_size
    H = config.num_kv_heads
    D = config.head_dim

    prefill_token = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)

    # Select update function based on layout
    if layout == "BHSD":
        update_fn = decode_update_bhsd
    elif layout == "BSHD":
        update_fn = decode_update_bshd
    else:
        update_fn = decode_update_hbsd

    # Prefill all layers
    for pos in range(prefill_len):
        for cache in caches:
            update_fn(cache, prefill_token, pos)
    sync_mps()

    # Generate new token for decode
    new_token = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)

    # Warmup decode steps
    pos = prefill_len
    for _ in range(WARMUP_ITERS):
        for cache in caches:
            _ = update_fn(cache, new_token, pos)
        pos += 1
    sync_mps()

    # Reset position
    pos = prefill_len

    # Benchmark decode: update all layers, read cache slice
    decode_steps = 100
    start = time.perf_counter()
    for _ in range(decode_steps):
        for cache in caches:
            kv = update_fn(cache, new_token, pos)
            # Simulate attention read by accessing the tensor
            _ = kv.data_ptr()
        pos += 1
    sync_mps()
    elapsed = time.perf_counter() - start

    del caches
    gc.collect()

    return {
        "total_ms": elapsed * 1000,
        "per_token_ms": elapsed * 1000 / decode_steps,
        "per_layer_us": elapsed * 1e6 / (decode_steps * config.num_layers),
    }


def benchmark_attention_read_pattern(
    layout: LayoutType, config: CacheConfig, seq_len: int = 2048
) -> dict[str, float]:
    """Benchmark the actual memory access pattern during attention.

    During decode attention:
    - Q: [B, H, 1, D] (single query)
    - K: [B, H, seq, D] (full cache)
    - Compute Q @ K^T: need to read each K[..., i, :] vector

    The K vectors are accessed with stride seq_len in head_dim chunks.
    This benchmarks how efficiently we can read K vectors for dot products.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    B = config.batch_size
    H = config.num_kv_heads
    D = config.head_dim

    # Create and fill cache
    cache = create_cache_tensor(layout, config, device)

    # Fill with random data
    if layout == "BHSD":
        cache[:, :, :seq_len, :] = torch.randn(B, H, seq_len, D, dtype=torch.float16, device=device)
    elif layout == "BSHD":
        cache[:, :seq_len, :, :] = torch.randn(B, seq_len, H, D, dtype=torch.float16, device=device)
    else:  # HBSD
        cache[:, :, :seq_len, :] = torch.randn(H, B, seq_len, D, dtype=torch.float16, device=device)

    sync_mps()

    # Query vector
    q = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)

    # Get K in standard [B, H, seq, D] format for attention
    def get_k_for_attention() -> torch.Tensor:
        if layout == "BHSD":
            return cache[:, :, :seq_len, :]
        elif layout == "BSHD":
            return cache[:, :seq_len, :, :].permute(0, 2, 1, 3)
        else:  # HBSD
            return cache[:, :, :seq_len, :].permute(1, 0, 2, 3)

    # Warmup
    for _ in range(WARMUP_ITERS):
        k = get_k_for_attention()
        # Simulate Q @ K^T: need K transposed to [B, H, D, seq]
        k_t = k.transpose(-2, -1)
        scores = torch.matmul(q, k_t)  # [B, H, 1, seq]
        _ = scores.sum()
    sync_mps()

    # Benchmark
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        k = get_k_for_attention()
        k_t = k.transpose(-2, -1)
        scores = torch.matmul(q, k_t)
        _ = scores.sum()
    sync_mps()
    elapsed = time.perf_counter() - start

    del cache, q
    gc.collect()

    return {
        "total_ms": elapsed * 1000,
        "per_iter_ms": elapsed * 1000 / BENCHMARK_ITERS,
        "bandwidth_gb_s": (B * H * seq_len * D * 2 * BENCHMARK_ITERS) / elapsed / 1e9,
    }


def benchmark_strided_read(
    layout: LayoutType, config: CacheConfig, seq_len: int = 2048
) -> dict[str, float]:
    """Benchmark strided K vector reads (kernel access pattern).

    The attention kernel reads K vectors as:
        k_vec = K_base + k_idx * head_dim

    This tests how efficiently the GPU can read vectors at stride head_dim
    across the sequence dimension.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    B = config.batch_size
    H = config.num_kv_heads
    D = config.head_dim

    # Create cache in standard BHSD format (what kernel sees)
    # We'll test raw strided access
    cache = torch.randn(B, H, seq_len, D, dtype=torch.float16, device=device)
    sync_mps()

    # Simulate kernel's strided read pattern:
    # For each head, read all K vectors in sequence
    # This is what attention_qk_softmax does

    # Output accumulator
    output = torch.zeros(B, H, D, dtype=torch.float32, device=device)

    # Warmup: simulate reading all K vectors and accumulating
    for _ in range(WARMUP_ITERS // 10):
        for k_idx in range(0, seq_len, 32):  # Sample every 32nd
            k_vec = cache[:, :, k_idx, :]  # [B, H, D]
            output += k_vec.float()
    sync_mps()

    # Benchmark: sequential reads across seq dimension
    read_iters = 50
    start = time.perf_counter()
    for _ in range(read_iters):
        acc = torch.zeros(B, H, D, dtype=torch.float32, device=device)
        # Read every K vector (simulates full attention over seq_len)
        for k_idx in range(seq_len):
            k_vec = cache[:, :, k_idx, :]  # [B, H, D] strided read
            acc += k_vec.float()
        _ = acc.sum()
    sync_mps()
    elapsed = time.perf_counter() - start

    del cache, output
    gc.collect()

    return {
        "total_ms": elapsed * 1000,
        "per_iter_ms": elapsed * 1000 / read_iters,
        "reads_per_second": (read_iters * seq_len) / elapsed,
    }


def benchmark_write_coalescing(
    layout: LayoutType, config: CacheConfig
) -> dict[str, float]:
    """Benchmark memory write coalescing for different layouts.

    Tests how efficiently single-token writes are performed.
    BHSD: writes are strided (non-coalesced)
    BSHD: writes are contiguous per token (coalesced)
    HBSD: writes are strided per head
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    B = config.batch_size
    H = config.num_kv_heads
    D = config.head_dim

    cache = create_cache_tensor(layout, config, device)
    token = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)

    # Convert token to layout format
    if layout == "BSHD":
        token_layout = token.permute(0, 2, 1, 3)  # [B, 1, H, D]
    elif layout == "HBSD":
        token_layout = token.permute(1, 0, 2, 3)  # [H, B, 1, D]
    else:
        token_layout = token

    sync_mps()

    # Warmup
    for pos in range(WARMUP_ITERS):
        if layout == "BHSD":
            cache[:, :, pos : pos + 1, :] = token_layout
        elif layout == "BSHD":
            cache[:, pos : pos + 1, :, :] = token_layout
        else:  # HBSD
            cache[:, :, pos : pos + 1, :] = token_layout
    sync_mps()

    # Benchmark sequential writes
    write_count = 1000
    start = time.perf_counter()
    for pos in range(write_count):
        if layout == "BHSD":
            cache[:, :, pos : pos + 1, :] = token_layout
        elif layout == "BSHD":
            cache[:, pos : pos + 1, :, :] = token_layout
        else:  # HBSD
            cache[:, :, pos : pos + 1, :] = token_layout
    sync_mps()
    elapsed = time.perf_counter() - start

    del cache, token
    gc.collect()

    bytes_per_write = B * H * D * 2  # float16

    return {
        "total_ms": elapsed * 1000,
        "per_write_us": elapsed * 1e6 / write_count,
        "bandwidth_gb_s": (bytes_per_write * write_count) / elapsed / 1e9,
    }


def benchmark_contiguous_read(
    layout: LayoutType, config: CacheConfig, seq_len: int = 2048
) -> dict[str, float]:
    """Benchmark reading cache as contiguous tensor.

    Tests the cost of making non-contiguous views contiguous,
    which is required for some operations.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    B = config.batch_size
    H = config.num_kv_heads
    D = config.head_dim

    cache = create_cache_tensor(layout, config, device)

    # Fill cache
    if layout == "BHSD":
        cache[:, :, :seq_len, :] = torch.randn(
            B, H, seq_len, D, dtype=torch.float16, device=device
        )
    elif layout == "BSHD":
        cache[:, :seq_len, :, :] = torch.randn(
            B, seq_len, H, D, dtype=torch.float16, device=device
        )
    else:
        cache[:, :, :seq_len, :] = torch.randn(
            H, B, seq_len, D, dtype=torch.float16, device=device
        )

    sync_mps()

    def get_contiguous_bhsd() -> torch.Tensor:
        """Get cache as contiguous [B, H, seq, D] tensor."""
        if layout == "BHSD":
            return cache[:, :, :seq_len, :].contiguous()
        elif layout == "BSHD":
            return cache[:, :seq_len, :, :].permute(0, 2, 1, 3).contiguous()
        else:
            return cache[:, :, :seq_len, :].permute(1, 0, 2, 3).contiguous()

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = get_contiguous_bhsd()
    sync_mps()

    # Benchmark
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        k = get_contiguous_bhsd()
        _ = k.data_ptr()
    sync_mps()
    elapsed = time.perf_counter() - start

    del cache
    gc.collect()

    bytes_read = B * H * seq_len * D * 2

    return {
        "total_ms": elapsed * 1000,
        "per_iter_ms": elapsed * 1000 / BENCHMARK_ITERS,
        "bandwidth_gb_s": (bytes_read * BENCHMARK_ITERS) / elapsed / 1e9,
    }


# =============================================================================
# Multi-layer consolidated layout benchmarks
# =============================================================================


def create_multilayer_cache(
    layout: MultiLayerLayoutType, config: CacheConfig, device: str
) -> torch.Tensor:
    """Create a single tensor holding all layers' KV cache.

    LHSD: [num_layers, num_heads, seq_len, head_dim]
    LSHD: [num_layers, seq_len, num_heads, head_dim]
    """
    L = config.num_layers
    H = config.num_kv_heads
    S = config.max_seq_len
    D = config.head_dim

    if layout == "LHSD":
        shape = (L, H, S, D)
    elif layout == "LSHD":
        shape = (L, S, H, D)
    else:
        raise ValueError(f"Unknown multi-layer layout: {layout}")

    return torch.zeros(shape, dtype=torch.float16, device=device)


def benchmark_multilayer_decode_update(
    layout: MultiLayerLayoutType, config: CacheConfig
) -> dict[str, float]:
    """Benchmark decode updates across all layers using consolidated cache.

    This simulates the decode loop where we append one token per layer.
    The consolidated layout allows potential prefetching benefits.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    L = config.num_layers
    H = config.num_kv_heads
    D = config.head_dim

    # Create consolidated cache for all layers
    cache = create_multilayer_cache(layout, config, device)

    # Prefill with some tokens
    prefill_len = 1024
    if layout == "LHSD":
        # Fill [L, H, :prefill_len, D]
        cache[:, :, :prefill_len, :] = torch.randn(
            L, H, prefill_len, D, dtype=torch.float16, device=device
        )
    else:  # LSHD
        # Fill [L, :prefill_len, H, D]
        cache[:, :prefill_len, :, :] = torch.randn(
            L, prefill_len, H, D, dtype=torch.float16, device=device
        )

    sync_mps()

    # Token for update - standard [H, 1, D] for single-head update
    token = torch.randn(H, 1, D, dtype=torch.float16, device=device)

    # Warmup
    pos = prefill_len
    for _ in range(WARMUP_ITERS):
        for layer_idx in range(L):
            if layout == "LHSD":
                cache[layer_idx, :, pos : pos + 1, :] = token
            else:  # LSHD
                cache[layer_idx, pos : pos + 1, :, :] = token.permute(1, 0, 2)
        pos += 1
    sync_mps()

    # Reset position
    pos = prefill_len

    # Benchmark decode steps
    decode_steps = 100
    start = time.perf_counter()
    for _ in range(decode_steps):
        for layer_idx in range(L):
            if layout == "LHSD":
                cache[layer_idx, :, pos : pos + 1, :] = token
                # Return slice for attention (view, no copy)
                _ = cache[layer_idx, :, : pos + 1, :]
            else:  # LSHD
                cache[layer_idx, pos : pos + 1, :, :] = token.permute(1, 0, 2)
                # Return slice for attention (needs transpose)
                _ = cache[layer_idx, : pos + 1, :, :].permute(1, 0, 2)
        pos += 1
    sync_mps()
    elapsed = time.perf_counter() - start

    del cache
    gc.collect()

    return {
        "total_ms": elapsed * 1000,
        "per_token_ms": elapsed * 1000 / decode_steps,
        "per_layer_us": elapsed * 1e6 / (decode_steps * L),
    }


def benchmark_multilayer_write_coalescing(
    layout: MultiLayerLayoutType, config: CacheConfig
) -> dict[str, float]:
    """Benchmark write efficiency for multi-layer cache.

    LHSD: writes stride across seq dimension (non-coalesced within layer)
    LSHD: writes are contiguous within seq position (coalesced for all heads)
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    L = config.num_layers
    H = config.num_kv_heads
    D = config.head_dim

    cache = create_multilayer_cache(layout, config, device)

    # Token for single-position write
    if layout == "LHSD":
        token = torch.randn(H, 1, D, dtype=torch.float16, device=device)
    else:  # LSHD
        token = torch.randn(1, H, D, dtype=torch.float16, device=device)

    sync_mps()

    # Warmup
    for pos in range(WARMUP_ITERS):
        for layer_idx in range(L):
            if layout == "LHSD":
                cache[layer_idx, :, pos : pos + 1, :] = token
            else:  # LSHD
                cache[layer_idx, pos : pos + 1, :, :] = token
    sync_mps()

    # Benchmark
    write_count = 500  # Fewer since we iterate all layers
    start = time.perf_counter()
    for pos in range(write_count):
        for layer_idx in range(L):
            if layout == "LHSD":
                cache[layer_idx, :, pos : pos + 1, :] = token
            else:  # LSHD
                cache[layer_idx, pos : pos + 1, :, :] = token
    sync_mps()
    elapsed = time.perf_counter() - start

    del cache
    gc.collect()

    bytes_per_write = L * H * D * 2  # float16, all layers

    return {
        "total_ms": elapsed * 1000,
        "per_position_ms": elapsed * 1000 / write_count,
        "per_layer_us": elapsed * 1e6 / (write_count * L),
        "bandwidth_gb_s": (bytes_per_write * write_count) / elapsed / 1e9,
    }


def benchmark_multilayer_attention_read(
    layout: MultiLayerLayoutType, config: CacheConfig, seq_len: int = 2048
) -> dict[str, float]:
    """Benchmark attention read pattern for multi-layer cache.

    Simulates reading K cache for Q @ K^T during decode.
    Tests both the raw read and any transpose overhead.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    L = config.num_layers
    H = config.num_kv_heads
    D = config.head_dim

    cache = create_multilayer_cache(layout, config, device)

    # Fill cache
    if layout == "LHSD":
        cache[:, :, :seq_len, :] = torch.randn(
            L, H, seq_len, D, dtype=torch.float16, device=device
        )
    else:  # LSHD
        cache[:, :seq_len, :, :] = torch.randn(
            L, seq_len, H, D, dtype=torch.float16, device=device
        )

    sync_mps()

    # Query vector for decode: single token
    q = torch.randn(1, H, 1, D, dtype=torch.float16, device=device)

    def get_k_and_compute(layer_idx: int) -> torch.Tensor:
        """Get K for layer and compute Q @ K^T."""
        if layout == "LHSD":
            # Already [H, S, D] - add batch dim, transpose for matmul
            k = cache[layer_idx, :, :seq_len, :].unsqueeze(0)  # [1, H, S, D]
            k_t = k.transpose(-2, -1)  # [1, H, D, S]
        else:  # LSHD
            # [S, H, D] -> [1, H, S, D]
            k = cache[layer_idx, :seq_len, :, :].permute(1, 0, 2).unsqueeze(0)
            k_t = k.transpose(-2, -1)
        return torch.matmul(q, k_t)  # [1, H, 1, S]

    # Warmup
    for _ in range(WARMUP_ITERS):
        for layer_idx in range(L):
            _ = get_k_and_compute(layer_idx)
    sync_mps()

    # Benchmark
    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        for layer_idx in range(L):
            scores = get_k_and_compute(layer_idx)
            _ = scores.sum()
    sync_mps()
    elapsed = time.perf_counter() - start

    del cache, q
    gc.collect()

    bytes_per_layer = H * seq_len * D * 2  # float16

    return {
        "total_ms": elapsed * 1000,
        "per_forward_ms": elapsed * 1000 / iters,
        "per_layer_us": elapsed * 1e6 / (iters * L),
        "bandwidth_gb_s": (bytes_per_layer * L * iters) / elapsed / 1e9,
    }


def benchmark_multilayer_prefetch(
    layout: MultiLayerLayoutType, config: CacheConfig, seq_len: int = 2048
) -> dict[str, float]:
    """Benchmark layer prefetching benefit.

    Tests whether reading layer N+1 while computing layer N
    provides any performance benefit with the consolidated layout.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    L = config.num_layers
    H = config.num_kv_heads
    D = config.head_dim

    cache = create_multilayer_cache(layout, config, device)

    # Fill cache
    if layout == "LHSD":
        cache[:, :, :seq_len, :] = torch.randn(
            L, H, seq_len, D, dtype=torch.float16, device=device
        )
    else:
        cache[:, :seq_len, :, :] = torch.randn(
            L, seq_len, H, D, dtype=torch.float16, device=device
        )

    sync_mps()

    # Simulate layer iteration with prefetch
    def prefetch_layer(layer_idx: int) -> torch.Tensor:
        """Touch layer data to warm cache."""
        if layout == "LHSD":
            return cache[layer_idx, :, :seq_len, :].sum()
        else:
            return cache[layer_idx, :seq_len, :, :].sum()

    def compute_layer(layer_idx: int) -> torch.Tensor:
        """Simulate attention computation on layer."""
        if layout == "LHSD":
            k = cache[layer_idx, :, :seq_len, :]
        else:
            k = cache[layer_idx, :seq_len, :, :]
        # Simple reduction to simulate compute
        return k.sum()

    # Benchmark WITHOUT prefetch
    sync_mps()
    start_no_prefetch = time.perf_counter()
    for _ in range(50):
        for layer_idx in range(L):
            _ = compute_layer(layer_idx)
    sync_mps()
    elapsed_no_prefetch = time.perf_counter() - start_no_prefetch

    # Benchmark WITH prefetch
    sync_mps()
    start_prefetch = time.perf_counter()
    for _ in range(50):
        for layer_idx in range(L):
            # Prefetch next layer while computing current
            if layer_idx + 1 < L:
                _ = prefetch_layer(layer_idx + 1)
            _ = compute_layer(layer_idx)
    sync_mps()
    elapsed_prefetch = time.perf_counter() - start_prefetch

    del cache
    gc.collect()

    return {
        "no_prefetch_ms": elapsed_no_prefetch * 1000 / 50,
        "with_prefetch_ms": elapsed_prefetch * 1000 / 50,
        "prefetch_overhead_pct": (elapsed_prefetch - elapsed_no_prefetch)
        / elapsed_no_prefetch
        * 100,
    }


def run_benchmarks() -> None:
    """Run all benchmarks and report results."""
    print("=" * 80)
    print("KV Cache Memory Layout Benchmark")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "mps":
        # Get GPU info
        try:
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.split("\n"):
                if "Chip" in line or "Metal" in line:
                    print(f"  {line.strip()}")
        except Exception:
            pass

    config = CacheConfig()
    print("\nConfig:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads} (KV: {config.num_kv_heads})")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Max seq: {config.max_seq_len}")

    # Calculate memory per layout
    B, H, S, D = config.batch_size, config.num_kv_heads, config.max_seq_len, config.head_dim
    bytes_per_cache = B * H * S * D * 2 * 2 * config.num_layers  # K+V, float16
    print(f"  Cache memory: {bytes_per_cache / 1e9:.2f} GB (K+V across all layers)")

    layouts: list[LayoutType] = ["BHSD", "BSHD", "HBSD"]

    # ==========================================================================
    # Benchmark 1: Decode Update
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. DECODE UPDATE (single-token append across all layers)")
    print("   Measures: cache write + return slice for attention")
    print("=" * 80)

    results: dict[str, dict[str, float]] = {}
    for layout in layouts:
        r = benchmark_decode_update(layout, config)
        results[layout] = r
        print(f"\n  {layout} [{_layout_shape(layout)}]:")
        print(f"    Per-token: {r['per_token_ms']:.3f} ms")
        print(f"    Per-layer: {r['per_layer_us']:.1f} µs")

    _print_comparison(results, "per_token_ms", "ms/token")

    # ==========================================================================
    # Benchmark 2: Attention Read Pattern
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. ATTENTION READ PATTERN (Q @ K^T matmul)")
    print("   Measures: get K, transpose, matmul with Q")
    print("=" * 80)

    for seq_len in [512, 1024, 2048, 4096]:
        print(f"\n  Sequence length: {seq_len}")
        results = {}
        for layout in layouts:
            r = benchmark_attention_read_pattern(layout, config, seq_len)
            results[layout] = r
            print(f"    {layout}: {r['per_iter_ms']:.3f} ms ({r['bandwidth_gb_s']:.1f} GB/s)")

        _print_comparison(results, "per_iter_ms", "ms")

    # ==========================================================================
    # Benchmark 3: Write Coalescing
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. WRITE COALESCING (memory write efficiency)")
    print("   Measures: raw write performance per token")
    print("=" * 80)

    results = {}
    for layout in layouts:
        r = benchmark_write_coalescing(layout, config)
        results[layout] = r
        print(f"\n  {layout}:")
        print(f"    Per-write: {r['per_write_us']:.2f} µs")
        print(f"    Bandwidth: {r['bandwidth_gb_s']:.1f} GB/s")

    _print_comparison(results, "per_write_us", "µs")

    # ==========================================================================
    # Benchmark 4: Contiguous Read
    # ==========================================================================
    print("\n" + "=" * 80)
    print("4. CONTIGUOUS READ (transpose + copy cost)")
    print("   Measures: cost to get [B,H,S,D] contiguous tensor")
    print("=" * 80)

    for seq_len in [1024, 2048, 4096]:
        print(f"\n  Sequence length: {seq_len}")
        results = {}
        for layout in layouts:
            r = benchmark_contiguous_read(layout, config, seq_len)
            results[layout] = r
            print(f"    {layout}: {r['per_iter_ms']:.3f} ms ({r['bandwidth_gb_s']:.1f} GB/s)")

        _print_comparison(results, "per_iter_ms", "ms")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Memory Layout Analysis:

BHSD [batch, num_heads, seq_len, head_dim] - Current
  + Native format for attention kernels (no transpose)
  + Efficient for head-parallel GEMM
  + Standard convention (PyTorch, HuggingFace)
  - Strided writes (one head_dim chunk per head, spread across memory)

BSHD [batch, seq_len, num_heads, head_dim] - Better Locality
  + Coalesced writes (all heads for a token are contiguous)
  + Good cache locality for sequential token append
  - Requires transpose to [B,H,S,D] for attention kernels
  - Transpose overhead may negate write benefits

HBSD [num_heads, batch, seq_len, head_dim] - Per-Head Access
  + Each head's data is fully contiguous
  + No GQA broadcast overhead within a head
  - Requires transpose for standard attention format
  - Poor cache utilization for multi-head access

Recommendations:
  - For standard attention (MHA/GQA): Keep BHSD
    The attention kernels expect [B,H,S,D] and any transpose adds overhead.

  - For custom fused kernels: Consider BSHD if kernel can accept it natively
    Only beneficial if you can eliminate the transpose entirely.

  - For very long sequences: BHSD with chunked/tiled access
    Memory bandwidth dominates; layout matters less than access pattern.
""")


def _layout_shape(layout: LayoutType) -> str:
    """Return human-readable shape string."""
    if layout == "BHSD":
        return "B, H, S, D"
    elif layout == "BSHD":
        return "B, S, H, D"
    else:
        return "H, B, S, D"


def _print_comparison(
    results: dict[str, dict[str, float]], key: str, unit: str
) -> None:
    """Print comparison between layouts."""
    values = {k: v[key] for k, v in results.items()}
    best = min(values, key=lambda x: values[x])
    worst_val = max(values.values())
    best_val = values[best]

    print(f"\n    Winner: {best} ({best_val:.3f} {unit})")
    if worst_val > best_val * 1.01:  # More than 1% difference
        speedup = worst_val / best_val
        print(f"    Speedup: {speedup:.2f}x vs slowest")


if __name__ == "__main__":
    run_benchmarks()
