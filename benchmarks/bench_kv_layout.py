"""
Benchmark KV cache layouts for decode performance.

Tests three layouts:
- BHSD [batch, num_heads, seq_len, head_dim]
- BSHD [batch, seq_len, num_heads, head_dim]
- HBSD [num_heads, batch, seq_len, head_dim]

Measures:
1. Single-token write time (decode bottleneck)
2. Full-sequence read time (attention bottleneck)

Outputs JSON with results.
"""

import json
import time
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class LayoutConfig:
    name: str
    layout: Literal["BHSD", "BSHD", "HBSD"]
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    batch_size: int
    dtype: torch.dtype = torch.float16


def create_cache(config: LayoutConfig, device: torch.device):
    if config.layout == "BHSD":
        shape = (config.batch_size, config.num_kv_heads, config.max_seq_len, config.head_dim)
    elif config.layout == "BSHD":
        shape = (config.batch_size, config.max_seq_len, config.num_kv_heads, config.head_dim)
    else:
        shape = (config.num_kv_heads, config.batch_size, config.max_seq_len, config.head_dim)

    cache = torch.zeros(shape, dtype=config.dtype, device=device)
    return cache


def write_single_token_bhsd(cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: int):
    cache[:, :, pos : pos + 1, :] = k
    return cache[:, :, : pos + 1, :]


def write_single_token_bshd(cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: int):
    k_t = k.permute(0, 2, 1, 3)
    cache[:, pos : pos + 1, :, :] = k_t
    return cache[:, : pos + 1, :, :].permute(0, 2, 1, 3)


def write_single_token_hbsd(cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: int):
    k_t = k.permute(1, 0, 2, 3)
    cache[:, :, pos : pos + 1, :] = k_t
    return cache[:, :, : pos + 1, :].permute(1, 0, 2, 3)


def benchmark_single_token_write(config: LayoutConfig, device: torch.device) -> dict:
    k_cache, v_cache = create_cache(config, device), create_cache(config, device)

    B, H, D = config.batch_size, config.num_kv_heads, config.head_dim
    k = torch.randn(B, H, 1, D, dtype=config.dtype, device=device)
    v = torch.randn(B, H, 1, D, dtype=config.dtype, device=device)

    if config.layout == "BHSD":
        write_fn = write_single_token_bhsd
    elif config.layout == "BSHD":
        write_fn = write_single_token_bshd
    else:
        write_fn = write_single_token_hbsd

    for _ in range(10):
        write_fn(k_cache, k, v, 0)
        write_fn(v_cache, k, v, 0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(1000):
        write_fn(k_cache, k, v, 0)
        write_fn(v_cache, k, v, 0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "write_time_us": elapsed * 1e6 / 2000,
        "writes_per_sec": 2000 / elapsed,
    }


def benchmark_full_sequence_read(config: LayoutConfig, device: torch.device, seq_len: int) -> dict:
    k_cache, v_cache = create_cache(config, device), create_cache(config, device)

    B, H, S, D = config.batch_size, config.num_kv_heads, seq_len, config.head_dim

    if config.layout == "BHSD":
        k_cache[:, :, :S, :] = torch.randn(B, H, S, D, dtype=config.dtype, device=device)
        v_cache[:, :, :S, :] = torch.randn(B, H, S, D, dtype=config.dtype, device=device)
        read_fn = lambda: (k_cache[:, :, :S, :], v_cache[:, :, :S, :])
    elif config.layout == "BSHD":
        k_cache[:, :S, :, :] = torch.randn(B, S, H, D, dtype=config.dtype, device=device)
        v_cache[:, :S, :, :] = torch.randn(B, S, H, D, dtype=config.dtype, device=device)
        read_fn = lambda: (
            k_cache[:, :S, :, :].permute(0, 2, 1, 3),
            v_cache[:, :S, :, :].permute(0, 2, 1, 3),
        )
    else:
        k_cache[:, :, :S, :] = torch.randn(H, B, S, D, dtype=config.dtype, device=device)
        v_cache[:, :, :S, :] = torch.randn(H, B, S, D, dtype=config.dtype, device=device)
        read_fn = lambda: (
            k_cache[:, :, :S, :].permute(1, 0, 2, 3),
            v_cache[:, :, :S, :].permute(1, 0, 2, 3),
        )

    for _ in range(10):
        k, v = read_fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        k, v = read_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_read = 2 * config.num_layers * B * H * S * D * (2 if config.dtype == torch.float16 else 4)

    return {
        "read_time_us": elapsed * 1e6 / 100,
        "reads_per_sec": 100 / elapsed,
        "bandwidth_gbps": bytes_read * 100 / elapsed / 1e9,
    }


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    configs = [
        LayoutConfig("BHSD", "BHSD", 32, 32, 8, 128, 4096, 1),
        LayoutConfig("BSHD", "BSHD", 32, 32, 8, 128, 4096, 1),
        LayoutConfig("HBSD", "HBSD", 32, 32, 8, 128, 4096, 1),
    ]

    results = {
        "device": str(device),
        "configs": [],
        "write_benchmarks": {},
        "read_benchmarks": {},
    }

    print("KV Cache Layout Benchmark")
    print(f"Device: {device}\n")
    print("Single-Token Write Performance:")
    print(f"{'Layout':<10} {'Time (μs)':>15} {'Writes/sec':>15}")
    print("-" * 40)

    for config in configs:
        result = benchmark_single_token_write(config, device)
        results["write_benchmarks"][config.name] = result
        print(
            f"{config.name:<10} {result['write_time_us']:>15.2f} {result['writes_per_sec']:>15.1f}"
        )
        results["configs"].append(
            {
                "layout": config.layout,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "num_kv_heads": config.num_kv_heads,
                "head_dim": config.head_dim,
                "max_seq_len": config.max_seq_len,
                "batch_size": config.batch_size,
            }
        )

    print("\nFull-Sequence Read Performance:")
    print(f"{'Layout':<10} {'Seq':>5} {'Time (μs)':>15} {'Bandwidth (GB/s)':>18}")
    print("-" * 55)

    for seq_len in [512, 1024, 2048, 4096]:
        for config in configs:
            result = benchmark_full_sequence_read(config, device, seq_len)
            key = f"{config.name}_seq{seq_len}"
            results["read_benchmarks"][key] = result
            print(
                f"{config.name:<10} {seq_len:>5} {result['read_time_us']:>15.2f} {result['bandwidth_gbps']:>18.2f}"
            )

    with open("benchmarks/bench_kv_layout_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to benchmarks/bench_kv_layout_results.json")


if __name__ == "__main__":
    main()
