"""
Micro-benchmark for KV cache memory bandwidth.

Measures actual memory throughput for different access patterns:
- Sequential read/write (good cache utilization)
- Strided read/write (poor cache utilization)
- Random access (worst case)
"""

import time
from dataclasses import dataclass

import torch


import os
import sys

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

@dataclass
class BandwidthResult:
    """Result from bandwidth benchmark."""
    layout_name: str
    operation: str  # read or write
    seq_len: int
    bytes_transferred: int
    time_ms: float
    bandwidth_gbps: float


def benchmark_sequential_bandwidth(
    tensor: torch.Tensor,
    dim: int,
    num_iterations: int = 100,
) -> BandwidthResult:
    """Benchmark sequential access along a dimension."""
    device = tensor.device
    seq_len = tensor.shape[dim]

    # Warmup
    for i in range(min(10, seq_len)):
        if dim == 0:
            _ = tensor[i]
        elif dim == 1:
            _ = tensor[:, i]
        elif dim == 2:
            _ = tensor[:, :, i]
        elif dim == 3:
            _ = tensor[:, :, :, i]

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark sequential reads
    start = time.perf_counter()
    for _ in range(num_iterations):
        for i in range(seq_len):
            if dim == 0:
                _ = tensor[i]
            elif dim == 1:
                _ = tensor[:, i]
            elif dim == 2:
                _ = tensor[:, :, i]
            elif dim == 3:
                _ = tensor[:, :, :, i]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_per_slice = tensor.numel() // seq_len * tensor.element_size()
    total_bytes = bytes_per_slice * seq_len * num_iterations

    return BandwidthResult(
        layout_name=f"dim_{dim}",
        operation="sequential_read",
        seq_len=seq_len,
        bytes_transferred=total_bytes,
        time_ms=elapsed * 1000,
        bandwidth_gbps=(total_bytes / elapsed) / 1e9,
    )


def benchmark_contiguous_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_iterations: int = 100,
) -> BandwidthResult:
    """Benchmark contiguous memory copy (best case)."""
    device = src.device

    # Warmup
    dst.copy_(src)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        dst.copy_(src)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_bytes = src.numel() * src.element_size() * num_iterations * 2  # read + write

    return BandwidthResult(
        layout_name="contiguous",
        operation="copy",
        seq_len=src.shape[2] if len(src.shape) >= 3 else 0,
        bytes_transferred=total_bytes,
        time_ms=elapsed * 1000,
        bandwidth_gbps=(total_bytes / elapsed) / 1e9,
    )


def benchmark_strided_access(
    tensor: torch.Tensor,
    stride_dim: int,
    num_iterations: int = 100,
) -> BandwidthResult:
    """Benchmark strided access (worst case for cache)."""
    device = tensor.device
    shape = tensor.shape

    # Access pattern: iterate over stride_dim, read all other dims
    # This creates strided memory access pattern
    stride_size = shape[stride_dim]

    # Warmup
    for i in range(min(10, stride_size)):
        slices = [slice(None)] * len(shape)
        slices[stride_dim] = i
        _ = tensor[tuple(slices)]

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        for i in range(stride_size):
            slices = [slice(None)] * len(shape)
            slices[stride_dim] = i
            _ = tensor[tuple(slices)]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_per_access = tensor.numel() // stride_size * tensor.element_size()
    total_bytes = bytes_per_access * stride_size * num_iterations

    return BandwidthResult(
        layout_name=f"strided_dim_{stride_dim}",
        operation="strided_read",
        seq_len=stride_size,
        bytes_transferred=total_bytes,
        time_ms=elapsed * 1000,
        bandwidth_gbps=(total_bytes / elapsed) / 1e9,
    )


def benchmark_kv_cache_patterns(
    batch_size: int = 1,
    num_heads: int = 8,
    seq_len: int = 2048,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
):
    """Benchmark different access patterns for KV cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("KV Cache Memory Bandwidth Analysis")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Config: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}")
    print(f"{'='*70}\n")

    results = []

    # Layout 1: [B, H, S, D] - BHSD
    print("Layout: BHSD [batch, heads, seq, dim]")
    print("-" * 50)
    tensor_bhsd = torch.randn(batch_size, num_heads, seq_len, head_dim,
                              dtype=dtype, device=device)
    tensor_bhsd_dst = torch.empty_like(tensor_bhsd)

    # Sequential along seq dim (dim=2)
    r = benchmark_sequential_bandwidth(tensor_bhsd, dim=2)
    results.append(r)
    print(f"  Sequential read along seq:  {r.bandwidth_gbps:.2f} GB/s")

    # Contiguous copy
    r = benchmark_contiguous_copy(tensor_bhsd, tensor_bhsd_dst)
    results.append(r)
    print(f"  Contiguous copy:            {r.bandwidth_gbps:.2f} GB/s")

    # Layout 2: [B, S, H, D] - BSHD
    print("\nLayout: BSHD [batch, seq, heads, dim]")
    print("-" * 50)
    tensor_bshd = torch.randn(batch_size, seq_len, num_heads, head_dim,
                              dtype=dtype, device=device)
    tensor_bshd_dst = torch.empty_like(tensor_bshd)

    # Sequential along seq dim (dim=1)
    r = benchmark_sequential_bandwidth(tensor_bshd, dim=1)
    results.append(r)
    print(f"  Sequential read along seq:  {r.bandwidth_gbps:.2f} GB/s")

    # Contiguous copy
    r = benchmark_contiguous_copy(tensor_bshd, tensor_bshd_dst)
    results.append(r)
    print(f"  Contiguous copy:            {r.bandwidth_gbps:.2f} GB/s")

    # Simulate decode-phase access patterns
    print("\n" + "="*70)
    print("Decode-Phase Access Patterns")
    print("="*70)

    # Single token write (append to cache)
    print("\nSingle Token Write (append):")
    print("-" * 50)

    # BHSD: Write to [B, H, 1, D] at position seq_len//2
    pos = seq_len // 2
    k_new = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)

    # Warmup
    tensor_bhsd[:, :, pos:pos+1, :].copy_(k_new)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(1000):
        tensor_bhsd[:, :, pos:pos+1, :].copy_(k_new)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_written = k_new.numel() * k_new.element_size() * 1000
    bhsd_write_bw = (bytes_written / elapsed) / 1e9
    print(f"  BHSD single token write: {bhsd_write_bw:.2f} GB/s")

    # BSHD: Write to [B, 1, H, D] at position seq_len//2
    k_new_t = k_new.permute(0, 2, 1, 3)  # [B, 1, H, D]

    # Warmup
    tensor_bshd[:, pos:pos+1, :, :].copy_(k_new_t)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(1000):
        tensor_bshd[:, pos:pos+1, :, :].copy_(k_new_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bshd_write_bw = (bytes_written / elapsed) / 1e9
    print(f"  BSHD single token write: {bshd_write_bw:.2f} GB/s")

    # Full cache read (for attention)
    print("\nFull Cache Read (for attention):")
    print("-" * 50)

    # BHSD: Read [B, H, S, D]
    # Warmup
    _ = tensor_bhsd[:, :, :seq_len//2, :]
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = tensor_bhsd[:, :, :seq_len//2, :].clone()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_read = tensor_bhsd[:, :, :seq_len//2, :].numel() * tensor_bhsd.element_size() * 100
    bhsd_read_bw = (bytes_read / elapsed) / 1e9
    print(f"  BHSD partial read:       {bhsd_read_bw:.2f} GB/s")

    # BSHD: Read [B, S, H, D] and transpose
    # Warmup
    _ = tensor_bshd[:, :seq_len//2, :, :].permute(0, 2, 1, 3)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = tensor_bshd[:, :seq_len//2, :, :].permute(0, 2, 1, 3).clone()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_read = tensor_bshd[:, :seq_len//2, :, :].numel() * tensor_bshd.element_size() * 100
    bshd_read_bw = (bytes_read / elapsed) / 1e9
    print(f"  BSHD partial read+trans: {bshd_read_bw:.2f} GB/s")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    write_speedup = bshd_write_bw / bhsd_write_bw if bhsd_write_bw > 0 else 0
    read_speedup = bhsd_read_bw / bhsd_read_bw if bhsd_read_bw > 0 else 0

    print(f"\nWrite speedup (BSHD vs BHSD): {write_speedup:.2f}x")
    print(f"Read comparison (BSHD vs BHSD): {bshd_read_bw/bhsd_read_bw:.2f}x")

    print("\nRecommendations:")
    if write_speedup > 1.1:
        print(f"  ✓ BSHD provides significantly faster writes ({write_speedup:.1f}x)")
        print("    - Better for decode phase where single-token writes dominate")
    else:
        print("  → Write performance is similar between layouts")

    if bhsd_read_bw > bshd_read_bw * 1.05:
        print("  ✓ BHSD provides faster reads for attention")
        print("    - Better for prefill phase where full-cache reads dominate")
    elif bshd_read_bw > bhsd_read_bw * 1.05:
        print("  ✓ BSHD provides faster reads (unexpected!)")
    else:
        print("  → Read performance is similar between layouts")

    # Cleanup
    del tensor_bhsd, tensor_bhsd_dst, tensor_bshd, tensor_bshd_dst, k_new, k_new_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def benchmark_paged_attention_pattern(
    batch_size: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
    page_size: int = 16,
    num_pages: int = 256,  # 4096 tokens
    dtype: torch.Tensor = torch.float16,
):
    """Benchmark paged attention memory patterns."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("Paged Attention Memory Pattern Benchmark")
    print(f"{'='*70}")
    print(f"Page size: {page_size}, Num pages: {num_pages}")
    print(f"Total tokens: {page_size * num_pages}")

    # Paged layout: [num_pages, page_size, num_heads, head_dim]
    # vs contiguous: [batch, num_heads, seq_len, head_dim]

    seq_len = page_size * num_pages

    # Contiguous layout
    contiguous = torch.randn(batch_size, num_heads, seq_len, head_dim,
                            dtype=dtype, device=device)

    # Paged layout (simulate with flattened)
    paged = torch.randn(num_pages, page_size, num_heads, head_dim,
                       dtype=dtype, device=device)

    print("\nMemory overhead:")
    contiguous_bytes = contiguous.numel() * contiguous.element_size()
    paged_bytes = paged.numel() * paged.element_size()
    print(f"  Contiguous: {contiguous_bytes / 1024 / 1024:.2f} MB")
    print(f"  Paged:      {paged_bytes / 1024 / 1024:.2f} MB")

    # Random page access (simulating variable-length sequences)
    print("\nRandom page access pattern:")

    # Generate random page indices
    page_indices = torch.randint(0, num_pages, (batch_size, num_pages // batch_size),
                                device=device)

    # Warmup
    for b in range(batch_size):
        for p_idx in page_indices[b]:
            _ = paged[p_idx]

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark paged access
    start = time.perf_counter()
    for _ in range(10):
        for b in range(batch_size):
            for p_idx in page_indices[b]:
                _ = paged[p_idx].clone()
    if device.type == "cuda":
        torch.cuda.synchronize()
    paged_time = time.perf_counter() - start

    # Benchmark contiguous access
    start = time.perf_counter()
    for _ in range(10):
        for b in range(batch_size):
            start_idx = 0
            for p_idx in page_indices[b]:
                _ = contiguous[b, :, start_idx:start_idx+page_size, :].clone()
                start_idx += page_size
    if device.type == "cuda":
        torch.cuda.synchronize()
    contiguous_time = time.perf_counter() - start

    print(f"  Paged access:     {paged_time*1000:.2f} ms")
    print(f"  Contiguous access:{contiguous_time*1000:.2f} ms")
    print(f"  Overhead ratio:   {paged_time/contiguous_time:.2f}x")

    # Cleanup
    del contiguous, paged, page_indices
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    """Run all bandwidth benchmarks."""
    # Basic bandwidth test
    benchmark_kv_cache_patterns(
        batch_size=1,
        num_heads=8,
        seq_len=2048,
        head_dim=64,
    )

    # Larger config if GPU available
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("Running with larger configuration...")
        print("="*70)
        benchmark_kv_cache_patterns(
            batch_size=4,
            num_heads=32,
            seq_len=4096,
            head_dim=128,
        )

        # Paged attention benchmark
        benchmark_paged_attention_pattern(
            batch_size=4,
            num_heads=32,
            head_dim=128,
            page_size=16,
            num_pages=256,
        )


if __name__ == "__main__":
    main()
