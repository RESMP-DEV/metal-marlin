#!/usr/bin/env python3
"""Profile KV cache update overhead (concat vs in-place).

Measures:
- Python heap allocations via tracemalloc (per token)
- MPS memory delta if available
- Time per token spent in cache update

Also estimates theoretical memory traffic for concat (O(n^2)) vs in-place (O(n)).
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Ensure metal_marlin is importable from project layout
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch  # noqa: E402


import os

# Check if running inside AlphaHENG task mode - skip to avoid memory bloat
if os.environ.get("ALPHAHENG_TASK_MODE") == "1":
    print("SKIP: Benchmark disabled in AlphaHENG task mode (ALPHAHENG_TASK_MODE=1)")
    print("Run benchmarks manually outside of agent tasks to avoid memory leaks.")
    sys.exit(0)

@dataclass
class BenchResult:
    seq_len: int
    tokens: int
    base_len: int
    mode: str
    time_ms_per_token: float
    py_alloc_per_token: float
    py_peak_bytes: int
    mps_delta_bytes: int
    mps_peak_bytes: int


def _format_bytes(num_bytes: float) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _sync(device: str) -> None:
    if device == "mps" and HAS_TORCH and HAS_MPS and torch is not None:
        torch.mps.synchronize()


def _mps_memory() -> dict[str, int]:
    if not (HAS_TORCH and HAS_MPS and torch is not None):
        return {"current": 0, "driver": 0}
    current = 0
    driver = 0
    if hasattr(torch.mps, "current_allocated_memory"):
        current = int(torch.mps.current_allocated_memory())
    if hasattr(torch.mps, "driver_allocated_memory"):
        driver = int(torch.mps.driver_allocated_memory())
    if driver == 0:
        driver = current
    return {"current": current, "driver": driver}


def _bytes_per_token(
    *,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    element_size: int,
) -> int:
    # K and V are both stored, so multiply by 2.
    return 2 * batch_size * num_kv_heads * head_dim * element_size


def _estimate_concat_traffic(seq_len: int, bytes_per_token: int) -> int:
    # Sum_{t=1..N} t = N(N+1)/2
    return int(bytes_per_token * (seq_len * (seq_len + 1) / 2))


def _estimate_inplace_traffic(seq_len: int, bytes_per_token: int) -> int:
    return int(bytes_per_token * seq_len)


def _concat_update(
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_cache is None:
        return k_new, v_new
    return torch.cat([k_cache, k_new], dim=2), torch.cat([v_cache, v_new], dim=2)


def _inplace_update(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    pos: int,
) -> None:
    k_cache[:, :, pos : pos + 1, :].copy_(k_new)
    v_cache[:, :, pos : pos + 1, :].copy_(v_new)


def _run_case(
    *,
    seq_len: int,
    tokens: int,
    mode: str,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    sync_each: bool,
    warmup: int,
) -> BenchResult:
    tokens = min(tokens, seq_len)
    base_len = max(seq_len - tokens, 0)

    k_new = torch.zeros((batch_size, num_kv_heads, 1, head_dim), device=device, dtype=dtype)
    v_new = torch.zeros((batch_size, num_kv_heads, 1, head_dim), device=device, dtype=dtype)

    if mode == "concat":
        k_cache = (
            torch.zeros((batch_size, num_kv_heads, base_len, head_dim), device=device, dtype=dtype)
            if base_len > 0
            else None
        )
        v_cache = (
            torch.zeros((batch_size, num_kv_heads, base_len, head_dim), device=device, dtype=dtype)
            if base_len > 0
            else None
        )
        for _ in range(warmup):
            k_cache, v_cache = _concat_update(k_cache, v_cache, k_new, v_new)
            if sync_each:
                _sync(device)
    else:
        k_cache = torch.zeros((batch_size, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)
        v_cache = torch.zeros((batch_size, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)
        pos = base_len
        for _ in range(warmup):
            _inplace_update(k_cache, v_cache, k_new, v_new, pos)
            pos += 1
            if sync_each:
                _sync(device)

    _sync(device)
    mps_before = _mps_memory()
    tracemalloc.start()
    start_current, _ = tracemalloc.get_traced_memory()

    times: list[float] = []
    if mode == "concat":
        k_cache_concat = k_cache
        v_cache_concat = v_cache
        for _ in range(tokens):
            t0 = time.perf_counter()
            k_cache_concat, v_cache_concat = _concat_update(
                k_cache_concat, v_cache_concat, k_new, v_new
            )
            if sync_each:
                _sync(device)
            times.append(time.perf_counter() - t0)
    else:
        pos = base_len
        for _ in range(tokens):
            t0 = time.perf_counter()
            _inplace_update(k_cache, v_cache, k_new, v_new, pos)
            if sync_each:
                _sync(device)
            times.append(time.perf_counter() - t0)
            pos += 1

    _sync(device)
    end_current, end_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mps_after = _mps_memory()

    py_alloc_per_token = max(0, end_peak - start_current) / max(tokens, 1)
    time_ms_per_token = (sum(times) / max(tokens, 1)) * 1000.0

    return BenchResult(
        seq_len=seq_len,
        tokens=tokens,
        base_len=base_len,
        mode=mode,
        time_ms_per_token=time_ms_per_token,
        py_alloc_per_token=py_alloc_per_token,
        py_peak_bytes=end_peak,
        mps_delta_bytes=mps_after["driver"] - mps_before["driver"],
        mps_peak_bytes=max(mps_before["driver"], mps_after["driver"]),
    )


def _print_results(results: Iterable[BenchResult]) -> None:
    header = (
        "seq_len  tokens  base  mode     time_ms/token  py_alloc/token  mps_delta  mps_peak"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.seq_len:<7}  {res.tokens:<6}  {res.base_len:<4}  {res.mode:<7}  "
            f"{res.time_ms_per_token:>13.3f}  "
            f"{_format_bytes(res.py_alloc_per_token):>13}  "
            f"{_format_bytes(res.mps_delta_bytes):>9}  "
            f"{_format_bytes(res.mps_peak_bytes):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile KV cache concat vs in-place updates")
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="128,512,2048,8192",
        help="Comma-separated sequence lengths",
    )
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate per case")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--mode", type=str, default="both", choices=["concat", "inplace", "both"])
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Synchronize per token for accurate MPS timing",
    )
    args = parser.parse_args()

    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for this benchmark (pip install torch).")

    device = args.device
    if device == "mps" and not HAS_MPS:
        raise RuntimeError("MPS requested but not available.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    if not seq_lens:
        raise ValueError("No sequence lengths specified.")

    sync_each = args.sync or device == "mps"

    results: list[BenchResult] = []
    for seq_len in seq_lens:
        if args.mode in ("concat", "both"):
            results.append(
                _run_case(
                    seq_len=seq_len,
                    tokens=args.tokens,
                    mode="concat",
                    batch_size=args.batch_size,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    device=device,
                    sync_each=sync_each,
                    warmup=args.warmup,
                )
            )
        if args.mode in ("inplace", "both"):
            results.append(
                _run_case(
                    seq_len=seq_len,
                    tokens=args.tokens,
                    mode="inplace",
                    batch_size=args.batch_size,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    device=device,
                    sync_each=sync_each,
                    warmup=args.warmup,
                )
            )

    print("KV Cache Update Profile")
    print(f"device={device} dtype={args.dtype} tokens={args.tokens} sync={sync_each}")
    print()
    _print_results(results)
    print()
    print("Estimated Memory Traffic (K+V)")
    print("seq_len  concat_traffic  inplace_traffic  ratio")
    print("----------------------------------------------")
    bytes_per_token = _bytes_per_token(
        batch_size=args.batch_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        element_size=torch.tensor([], dtype=dtype).element_size(),
    )
    for seq_len in seq_lens:
        concat_bytes = _estimate_concat_traffic(seq_len, bytes_per_token)
        inplace_bytes = _estimate_inplace_traffic(seq_len, bytes_per_token)
        ratio = concat_bytes / max(inplace_bytes, 1)
        print(
            f"{seq_len:<7}  {_format_bytes(concat_bytes):>14}  "
            f"{_format_bytes(inplace_bytes):>14}  x{ratio:>6.1f}"
        )

    print()
    print("Notes:")
    print("- tracemalloc tracks Python heap allocations only (not GPU driver memory).")
    print("- MPS allocations can be observed via torch.mps.*_allocated_memory when available.")


if __name__ == "__main__":
    main()
