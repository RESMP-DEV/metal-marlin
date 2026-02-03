"""
Benchmark KV cache memory layout for faster decode access.

Compares two layouts:
- Contiguous (BHSD): [num_layers, num_heads, seq_len, head_dim]
- Transposed (BSHD): [num_layers, seq_len, num_heads, head_dim]

For decode phase (single token at a time), memory access patterns differ:
- BHSD: Better for attention reads (heads contiguous, strided seq access)
- BSHD: Better for cache writes (seq contiguous, coalesced token writes)

This benchmark measures both read and write performance to determine optimal layout.
"""

import json
import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class LayoutConfig:
    """Configuration for KV cache layout benchmark."""
    name: str
    layout: Literal["BHSD", "BSHD", "LBHSD", "LBSHD"]  # L = layer dimension
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    batch_size: int
    dtype: torch.dtype = torch.float16


def create_cache_tensors(config: LayoutConfig, device: torch.device):
    """Create K and V cache tensors with specified layout."""
    # Full precision cache allocation
    if config.layout == "LBHSD":
        # [num_layers, batch, num_kv_heads, max_seq_len, head_dim]
        shape = (
            config.num_layers,
            config.batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )
    elif config.layout == "LBSHD":
        # [num_layers, batch, max_seq_len, num_kv_heads, head_dim]
        shape = (
            config.num_layers,
            config.batch_size,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        )
    elif config.layout == "BHSD":
        # [batch, num_kv_heads, max_seq_len, head_dim]
        shape = (
            config.batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )
    else:  # BSHD
        # [batch, max_seq_len, num_kv_heads, head_dim]
        shape = (
            config.batch_size,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        )

    # Create per-layer caches or single stacked tensor
    if config.layout in ("LBHSD", "LBSHD"):
        k_cache = torch.zeros(shape, dtype=config.dtype, device=device)
        v_cache = torch.zeros(shape, dtype=config.dtype, device=device)
        return k_cache, v_cache, True  # is_stacked=True
    else:
        k_cache = [
            torch.zeros(shape, dtype=config.dtype, device=device)
            for _ in range(config.num_layers)
        ]
        v_cache = [
            torch.zeros(shape, dtype=config.dtype, device=device)
            for _ in range(config.num_layers)
        ]
        return k_cache, v_cache, False  # is_stacked=False


def write_to_cache_bhsd(
    k_cache: list[torch.Tensor],
    v_cache: list[torch.Tensor],
    k_new: torch.Tensor,  # [batch, num_kv_heads, new_seq_len, head_dim]
    v_new: torch.Tensor,
    start_pos: int,
    new_seq_len: int,
):
    """Write to BHSD layout cache: [batch, heads, seq, dim]."""
    for layer_idx in range(len(k_cache)):
        k_cache[layer_idx].narrow(2, start_pos, new_seq_len).copy_(k_new)
        v_cache[layer_idx].narrow(2, start_pos, new_seq_len).copy_(v_new)


def write_to_cache_bshd(
    k_cache: list[torch.Tensor],
    v_cache: list[torch.Tensor],
    k_new: torch.Tensor,  # [batch, num_kv_heads, new_seq_len, head_dim]
    v_new: torch.Tensor,
    start_pos: int,
    new_seq_len: int,
):
    """Write to BSHD layout cache: [batch, seq, heads, dim].
    
    Input comes in BHSD format from attention computation, need to transpose.
    """
    # Transpose from [B, H, S, D] to [B, S, H, D]
    k_transposed = k_new.permute(0, 2, 1, 3)
    v_transposed = v_new.permute(0, 2, 1, 3)
    for layer_idx in range(len(k_cache)):
        k_cache[layer_idx].narrow(1, start_pos, new_seq_len).copy_(k_transposed)
        v_cache[layer_idx].narrow(1, start_pos, new_seq_len).copy_(v_transposed)


def write_to_cache_lbhsd(
    k_cache: torch.Tensor,  # [L, B, H, S, D]
    v_cache: torch.Tensor,
    k_new: torch.Tensor,  # [batch, num_kv_heads, new_seq_len, head_dim]
    v_new: torch.Tensor,
    start_pos: int,
    new_seq_len: int,
):
    """Write to stacked LBHSD layout cache."""
    num_layers = k_cache.shape[0]
    # Expand for all layers: [L, B, H, S, D]
    k_expanded = k_new.unsqueeze(0).expand(num_layers, -1, -1, -1, -1)
    v_expanded = v_new.unsqueeze(0).expand(num_layers, -1, -1, -1, -1)
    k_cache.narrow(3, start_pos, new_seq_len).copy_(k_expanded)
    v_cache.narrow(3, start_pos, new_seq_len).copy_(v_expanded)


def write_to_cache_lbshd(
    k_cache: torch.Tensor,  # [L, B, S, H, D]
    v_cache: torch.Tensor,
    k_new: torch.Tensor,  # [batch, num_kv_heads, new_seq_len, head_dim]
    v_new: torch.Tensor,
    start_pos: int,
    new_seq_len: int,
):
    """Write to stacked LBSHD layout cache."""
    num_layers = k_cache.shape[0]
    # Transpose and expand: [B, H, S, D] -> [B, S, H, D] -> [L, B, S, H, D]
    k_transposed = k_new.permute(0, 2, 1, 3)
    v_transposed = v_new.permute(0, 2, 1, 3)
    k_expanded = k_transposed.unsqueeze(0).expand(num_layers, -1, -1, -1, -1)
    v_expanded = v_transposed.unsqueeze(0).expand(num_layers, -1, -1, -1, -1)
    k_cache.narrow(2, start_pos, new_seq_len).copy_(k_expanded)
    v_cache.narrow(2, start_pos, new_seq_len).copy_(v_expanded)


def read_from_cache_bhsd(
    k_cache: list[torch.Tensor],
    v_cache: list[torch.Tensor],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read full cache up to seq_len from BHSD layout.
    
    Returns: [batch, num_kv_heads, seq_len, head_dim]
    """
    # Stack and slice
    k_stacked = torch.stack([k_cache[i][:, :, :seq_len, :] for i in range(len(k_cache))])
    v_stacked = torch.stack([v_cache[i][:, :, :seq_len, :] for i in range(len(v_cache))])
    return k_stacked, v_stacked


def read_from_cache_bshd(
    k_cache: list[torch.Tensor],
    v_cache: list[torch.Tensor],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read full cache up to seq_len from BSHD layout.
    
    Returns: [batch, num_kv_heads, seq_len, head_dim] (transposed back)
    """
    # Slice then transpose from [B, S, H, D] back to [B, H, S, D]
    k_stacked = torch.stack([
        k_cache[i][:, :seq_len, :, :].permute(0, 2, 1, 3)
        for i in range(len(k_cache))
    ])
    v_stacked = torch.stack([
        v_cache[i][:, :seq_len, :, :].permute(0, 2, 1, 3)
        for i in range(len(v_cache))
    ])
    return k_stacked, v_stacked


def read_from_cache_lbhsd(
    k_cache: torch.Tensor,  # [L, B, H, S, D]
    v_cache: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read from stacked LBHSD layout."""
    return k_cache[:, :, :, :seq_len, :], v_cache[:, :, :, :seq_len, :]


def read_from_cache_lbshd(
    k_cache: torch.Tensor,  # [L, B, S, H, D]
    v_cache: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read from stacked LBSHD layout."""
    # Slice to [L, B, seq_len, H, D] then transpose to [L, B, H, seq_len, D]
    k_sliced = k_cache[:, :, :seq_len, :, :].permute(0, 1, 3, 2, 4)
    v_sliced = v_cache[:, :, :seq_len, :, :].permute(0, 1, 3, 2, 4)
    return k_sliced, v_sliced


def benchmark_cache_write(
    config: LayoutConfig,
    device: torch.device,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> dict:
    """Benchmark cache write performance (single token decode)."""
    k_cache, v_cache, is_stacked = create_cache_tensors(config, device)

    # Single token input (decode phase)
    k_new = torch.randn(
        config.batch_size, config.num_kv_heads, 1, config.head_dim,
        dtype=config.dtype, device=device
    )
    v_new = torch.randn(
        config.batch_size, config.num_kv_heads, 1, config.head_dim,
        dtype=config.dtype, device=device
    )

    # Select write function
    if is_stacked:
        if config.layout == "LBHSD":
            write_fn = lambda pos: write_to_cache_lbhsd(k_cache, v_cache, k_new, v_new, pos, 1)
        else:
            write_fn = lambda pos: write_to_cache_lbshd(k_cache, v_cache, k_new, v_new, pos, 1)
    else:
        if config.layout == "BHSD":
            write_fn = lambda pos: write_to_cache_bhsd(k_cache, v_cache, k_new, v_new, pos, 1)
        else:
            write_fn = lambda pos: write_to_cache_bshd(k_cache, v_cache, k_new, v_new, pos, 1)

    # Warmup
    for pos in range(num_warmup):
        write_fn(pos % config.max_seq_len)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for pos in range(num_iterations):
        write_fn(pos % config.max_seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    time_per_write_us = (elapsed / num_iterations) * 1_000_000

    # Cleanup
    del k_cache, v_cache, k_new, v_new
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "time_per_write_us": time_per_write_us,
        "writes_per_sec": num_iterations / elapsed,
    }


def benchmark_cache_read(
    config: LayoutConfig,
    device: torch.device,
    seq_len: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> dict:
    """Benchmark cache read performance at given sequence length."""
    k_cache, v_cache, is_stacked = create_cache_tensors(config, device)

    # Pre-populate cache
    k_new = torch.randn(
        config.batch_size, config.num_kv_heads, 1, config.head_dim,
        dtype=config.dtype, device=device
    )
    v_new = torch.randn(
        config.batch_size, config.num_kv_heads, 1, config.head_dim,
        dtype=config.dtype, device=device
    )

    # Fill cache to seq_len
    for pos in range(min(seq_len, config.max_seq_len)):
        if is_stacked:
            if config.layout == "LBHSD":
                write_to_cache_lbhsd(k_cache, v_cache, k_new, v_new, pos, 1)
            else:
                write_to_cache_lbshd(k_cache, v_cache, k_new, v_new, pos, 1)
        else:
            if config.layout == "BHSD":
                write_to_cache_bhsd(k_cache, v_cache, k_new, v_new, pos, 1)
            else:
                write_to_cache_bshd(k_cache, v_cache, k_new, v_new, pos, 1)

    # Select read function
    if is_stacked:
        if config.layout == "LBHSD":
            read_fn = lambda: read_from_cache_lbhsd(k_cache, v_cache, seq_len)
        else:
            read_fn = lambda: read_from_cache_lbshd(k_cache, v_cache, seq_len)
    else:
        if config.layout == "BHSD":
            read_fn = lambda: read_from_cache_bhsd(k_cache, v_cache, seq_len)
        else:
            read_fn = lambda: read_from_cache_bshd(k_cache, v_cache, seq_len)

    # Warmup
    for _ in range(num_warmup):
        _ = read_fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = read_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    time_per_read_us = (elapsed / num_iterations) * 1_000_000
    bytes_read = (
        2  # K and V
        * config.num_layers
        * config.batch_size
        * config.num_kv_heads
        * seq_len
        * config.head_dim
        * (2 if config.dtype == torch.float16 else 4)  # bytes per element
    )
    bandwidth_gbps = (bytes_read * num_iterations / elapsed) / 1e9

    # Cleanup
    del k_cache, v_cache, k_new, v_new
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "time_per_read_us": time_per_read_us,
        "reads_per_sec": num_iterations / elapsed,
        "bytes_read": bytes_read,
        "bandwidth_gbps": bandwidth_gbps,
    }


def benchmark_attention_with_cache(
    config: LayoutConfig,
    device: torch.device,
    seq_len: int,
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> dict:
    """Benchmark end-to-end attention computation with cache reads."""
    k_cache, v_cache, is_stacked = create_cache_tensors(config, device)

    # Pre-populate cache
    k_new = torch.randn(
        config.batch_size, config.num_kv_heads, 1, config.head_dim,
        dtype=config.dtype, device=device
    )
    v_new = torch.randn(
        config.batch_size, config.num_kv_heads, 1, config.head_dim,
        dtype=config.dtype, device=device
    )

    for pos in range(min(seq_len, config.max_seq_len)):
        if is_stacked:
            if config.layout == "LBHSD":
                write_to_cache_lbhsd(k_cache, v_cache, k_new, v_new, pos, 1)
            else:
                write_to_cache_lbshd(k_cache, v_cache, k_new, v_new, pos, 1)
        else:
            if config.layout == "BHSD":
                write_to_cache_bhsd(k_cache, v_cache, k_new, v_new, pos, 1)
            else:
                write_to_cache_bshd(k_cache, v_cache, k_new, v_new, pos, 1)

    # Query tensor (single token)
    q = torch.randn(
        config.batch_size, config.num_heads, 1, config.head_dim,
        dtype=config.dtype, device=device
    )

    # Select read function
    if is_stacked:
        if config.layout == "LBHSD":
            read_fn = lambda: read_from_cache_lbhsd(k_cache, v_cache, seq_len)
        else:
            read_fn = lambda: read_from_cache_lbshd(k_cache, v_cache, seq_len)
    else:
        if config.layout == "BHSD":
            read_fn = lambda: read_from_cache_bhsd(k_cache, v_cache, seq_len)
        else:
            read_fn = lambda: read_from_cache_bshd(k_cache, v_cache, seq_len)

    num_groups = config.num_heads // config.num_kv_heads
    scale = config.head_dim ** -0.5

    def attention_fn():
        k_full, v_full = read_fn()
        # k_full: [L, B, H, S, D] or [L, B, Hkv, S, D]
        # Need to process per-layer
        outputs = []
        for layer_idx in range(config.num_layers):
            k_layer = k_full[layer_idx]  # [B, Hkv, S, D]
            v_layer = v_full[layer_idx]
            # Repeat for GQA
            if num_groups > 1:
                k_layer = k_layer.repeat_interleave(num_groups, dim=1)
                v_layer = v_layer.repeat_interleave(num_groups, dim=1)
            # Attention: [B, H, 1, D] @ [B, H, D, S] = [B, H, 1, S]
            scores = torch.matmul(q, k_layer.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            # [B, H, 1, S] @ [B, H, S, D] = [B, H, 1, D]
            output = torch.matmul(attn_weights, v_layer)
            outputs.append(output)
        return torch.stack(outputs)

    # Warmup
    for _ in range(num_warmup):
        _ = attention_fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = attention_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    time_per_layer_ms = (elapsed / num_iterations / config.num_layers) * 1000

    # Cleanup
    del k_cache, v_cache, k_new, v_new, q
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "time_per_layer_ms": time_per_layer_ms,
        "total_time_ms": (elapsed / num_iterations) * 1000,
        "layers_per_sec": (config.num_layers * num_iterations) / elapsed,
    }


def run_benchmark_suite(
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,  # GQA
    head_dim: int = 128,
    max_seq_len: int = 4096,
    batch_size: int = 1,
) -> dict:
    """Run complete benchmark suite comparing all layouts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"\n{'='*70}")
    print("KV Cache Memory Layout Benchmark")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model config: {num_layers} layers, {num_heads} heads, {num_kv_heads} KV heads")
    print(f"Head dim: {head_dim}, Max seq len: {max_seq_len}, Batch: {batch_size}")
    print(f"{'='*70}\n")

    configs = [
        LayoutConfig("BHSD (per-layer)", "BHSD", num_layers, num_heads, num_kv_heads,
                    head_dim, max_seq_len, batch_size, dtype),
        LayoutConfig("BSHD (per-layer)", "BSHD", num_layers, num_heads, num_kv_heads,
                    head_dim, max_seq_len, batch_size, dtype),
        LayoutConfig("LBHSD (stacked)", "LBHSD", num_layers, num_heads, num_kv_heads,
                    head_dim, max_seq_len, batch_size, dtype),
        LayoutConfig("LBSHD (stacked)", "LBSHD", num_layers, num_heads, num_kv_heads,
                    head_dim, max_seq_len, batch_size, dtype),
    ]

    results = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "config": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "dtype": "float16" if dtype == torch.float16 else "float32",
        },
        "write_benchmarks": {},
        "read_benchmarks": {},
        "attention_benchmarks": {},
    }

    # Write benchmarks (single token decode)
    print("Write Performance (single token decode):")
    print("-" * 70)
    print(f"{'Layout':<20} {'Time (μs)':>15} {'Writes/sec':>15}")
    print("-" * 70)
    for config in configs:
        result = benchmark_cache_write(config, device)
        results["write_benchmarks"][config.name] = result
        print(f"{config.name:<20} {result['time_per_write_us']:>15.2f} {result['writes_per_sec']:>15.1f}")
    print()

    # Read benchmarks at different sequence lengths
    seq_lengths = [128, 512, 1024, 2048, 4096]
    print("Read Performance (bandwidth at different seq lengths):")
    print("-" * 70)
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        print(f"{'Layout':<20} {'Time (μs)':>12} {'Bandwidth (GB/s)':>18}")
        print("-" * 70)
        for config in configs:
            result = benchmark_cache_read(config, device, seq_len)
            key = f"{config.name}_seq{seq_len}"
            results["read_benchmarks"][key] = result
            print(f"{config.name:<20} {result['time_per_read_us']:>12.2f} {result['bandwidth_gbps']:>18.2f}")
    print()

    # Attention benchmarks
    print("\nEnd-to-End Attention Performance:")
    print("-" * 70)
    for seq_len in [512, 2048]:
        print(f"\nSequence length: {seq_len}")
        print(f"{'Layout':<20} {'Time/layer (ms)':>18} {'Total (ms)':>12}")
        print("-" * 70)
        for config in configs:
            result = benchmark_attention_with_cache(config, device, seq_len)
            key = f"{config.name}_seq{seq_len}"
            results["attention_benchmarks"][key] = result
            print(f"{config.name:<20} {result['time_per_layer_ms']:>18.3f} {result['total_time_ms']:>12.3f}")
    print()

    # Memory usage analysis
    print("\nMemory Usage Analysis:")
    print("-" * 70)
    bytes_per_element = 2 if dtype == torch.float16 else 4
    for config in configs:
        if config.layout in ("LBHSD", "LBSHD"):
            # Stacked: [L, B, H/S, S/H, D]
            total_elements = (
                config.num_layers
                * config.batch_size
                * config.max_seq_len
                * config.num_kv_heads
                * config.head_dim
                * 2  # K and V
            )
        else:
            # Per-layer list
            total_elements = (
                config.num_layers
                * config.batch_size
                * config.num_kv_heads
                * config.max_seq_len
                * config.head_dim
                * 2  # K and V
            )
        total_mb = total_elements * bytes_per_element / 1024 / 1024
        print(f"{config.name:<20} {total_mb:>10.1f} MB")

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("Summary & Recommendations:")
    print("=" * 70)

    # Find best layout for writes
    write_times = {name: data["time_per_write_us"]
                   for name, data in results["write_benchmarks"].items()}
    best_write = min(write_times, key=write_times.get)

    # Find best layout for reads at longest sequence
    longest_seq = max(seq_lengths)
    read_times = {}
    for config in configs:
        key = f"{config.name}_seq{longest_seq}"
        read_times[config.name] = results["read_benchmarks"][key]["time_per_read_us"]
    best_read = min(read_times, key=read_times.get)

    print(f"\n1. Best for cache WRITES (decode): {best_write}")
    print(f"   - {write_times[best_write]:.2f} μs per token write")
    print(f"\n2. Best for cache READS (attention): {best_read}")
    print(f"   - {read_times[best_read]:.2f} μs read at seq_len={longest_seq}")

    print("\n3. Layout Analysis:")
    print("   - BHSD: Better for attention reads (contiguous heads)")
    print("   - BSHD: Better for decode writes (contiguous sequence)")
    print("   - Stacked (L*): Reduces Python overhead, better GPU utilization")
    print("   - Per-layer: More flexible, easier partial cache eviction")

    print("\n4. Recommendations:")
    print("   - For short sequences (< 1K): Use BHSD for simplicity")
    print("   - For long sequences (> 2K): Consider BSHD for faster writes")
    print("   - For batch size > 1: BHSD typically better (amortized writes)")
    print("   - Use stacked layout (LBHSD/LBSHD) for better GPU efficiency")

    return results


def main():
    """Run the full benchmark suite."""
    # Small model config for quick testing
    results = run_benchmark_suite(
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        max_seq_len=2048,
        batch_size=1,
    )

    # Also run with realistic model sizes if GPU available
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("Running with realistic model configuration...")
        print("=" * 70)
        results_large = run_benchmark_suite(
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=4096,
            batch_size=1,
        )
        results["large_config"] = results_large

    # Save results
    output_path = "benchmarks/kv_cache_layout_results.json"
    with open(output_path, "w") as f:
        # Convert to serializable format
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
