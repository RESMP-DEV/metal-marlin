"""
Sliding Window Attention dispatch wrapper for Apple Silicon Metal kernels.

This module provides Python bindings to the optimized Sliding Window Attention
Metal kernels for models like Mistral that use local attention patterns.

Key benefit: O(seq * window) memory instead of O(seq^2), enabling efficient
inference on long sequences (e.g., Mistral's 32K context window).

Kernels:
    sliding_window_attention_prefill  - Prefill with sliding window
    sliding_window_attention_causal   - Prefill with causal + window mask
    sliding_window_attention_decode   - Single-query decode (seq_q=1)
    sliding_window_attention_gqa      - Grouped-Query Attention with window

Usage:
    from metal_marlin.sliding_window_attention import sliding_window_attention

    # Mistral-style sliding window attention
    output = sliding_window_attention(
        Q, K, V,
        window_size=4096,  # Mistral default
        causal=True,
    )

    # Decode phase (seq_q=1)
    output = sliding_window_attention_decode(
        Q, K, V,
        window_size=4096,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import HAS_MPS, HAS_PYOBJC_METAL, HAS_TORCH

if HAS_TORCH:
    import torch

if HAS_PYOBJC_METAL:
    import Metal

if TYPE_CHECKING:
    import torch


# Tile dimensions matching the Metal kernel
TILE_Q_SW = 8
TILE_KV_SW = 32
NUM_SIMDGROUPS_SW = 4
THREADS_PER_TG_SW = NUM_SIMDGROUPS_SW * 32  # 128


def _require_metal_sliding_window() -> None:
    """Raise if Metal dispatch is not available."""
    if not HAS_TORCH:
        raise RuntimeError(
            "Sliding Window Attention requires PyTorch. Install with: pip install torch"
        )
    if not HAS_MPS:
        raise RuntimeError(
            "Sliding Window Attention requires PyTorch MPS backend (Apple Silicon)."
        )
    if not HAS_PYOBJC_METAL:
        raise RuntimeError(
            "Sliding Window Attention requires PyObjC Metal. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )


@dataclass
class SlidingWindowParams:
    """Parameters for Sliding Window Attention kernel dispatch."""

    batch: int
    num_heads_q: int
    num_heads_kv: int
    seq_q: int
    seq_k: int
    head_dim: int
    scale: float
    gqa_ratio: int
    window_size: int
    is_causal: bool

    def to_buffer(self, device: Any) -> Any:
        """Pack parameters into Metal buffer for kernel dispatch.

        Args:
            device: MTLDevice for buffer allocation

        Returns:
            MTLBuffer containing packed parameters
        """
        # Pack as uint32 array (scale as reinterpreted float, is_causal as 0/1)
        data = np.array(
            [
                self.batch,
                self.num_heads_q,
                self.num_heads_kv,
                self.seq_q,
                self.seq_k,
                self.head_dim,
                np.float32(self.scale).view(np.uint32),  # reinterpret float as uint32
                self.gqa_ratio,
                self.window_size,
                1 if self.is_causal else 0,
            ],
            dtype=np.uint32,
        )
        buffer = device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )
        return buffer


def _get_kernel_source() -> str:
    """Load Sliding Window Attention Metal kernel source."""
    kernel_path = Path(__file__).parent.parent / "src" / "sliding_window_attention.metal"
    if kernel_path.exists():
        return kernel_path.read_text()
    raise FileNotFoundError(f"Sliding Window Attention kernel not found at {kernel_path}")


def _compute_grid(params: SlidingWindowParams, kernel_type: str) -> tuple[int, int, int]:
    """Compute Metal grid dimensions for dispatch."""
    if kernel_type == "decode":
        # Decode: one threadgroup per (sequence, head) pair
        return (params.batch * params.num_heads_q, 1, 1)
    elif kernel_type == "gqa":
        # GQA: dispatch by KV heads
        q_tiles = (params.seq_q + TILE_Q_SW - 1) // TILE_Q_SW
        return (params.num_heads_kv, q_tiles, params.batch)
    else:
        # Standard prefill: dispatch by Q heads
        q_tiles = (params.seq_q + TILE_Q_SW - 1) // TILE_Q_SW
        return (params.num_heads_q, q_tiles, params.batch)


def select_kernel(
    seq_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    is_causal: bool,
) -> str:
    """Select optimal kernel based on attention configuration.

    Args:
        seq_q: Query sequence length
        num_heads_q: Number of query heads
        num_heads_kv: Number of KV heads
        is_causal: Whether causal masking is applied

    Returns:
        Kernel name to use
    """
    gqa_ratio = num_heads_q // num_heads_kv if num_heads_kv > 0 else 1

    if seq_q == 1:
        # Decode phase: single query token
        return "sliding_window_attention_decode"

    if gqa_ratio >= 4:
        # GQA with high ratio: use specialized kernel
        return "sliding_window_attention_gqa"

    # Standard prefill - causal is the typical pattern for sliding window
    if is_causal:
        return "sliding_window_attention_causal"
    else:
        return "sliding_window_attention_prefill"


# ---------------------------------------------------------------------------
# Kernel library singleton
# ---------------------------------------------------------------------------

_kernel_lib: Any = None


def _get_kernel_library() -> Any:
    """Get or create the Sliding Window Attention kernel library."""
    global _kernel_lib
    if _kernel_lib is None:
        from .metal_dispatch import MetalKernelLibrary

        _kernel_lib = MetalKernelLibrary()
        source = _get_kernel_source()
        _kernel_lib.compile_source("sliding_window_attention", source)
    return _kernel_lib


def _mps_tensor_to_metal_buffer(tensor: torch.Tensor, device: Any) -> Any:
    """Get Metal buffer from PyTorch MPS tensor (shared memory, no copy)."""
    if not tensor.is_mps:
        raise ValueError("Tensor must be on MPS device")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    storage = tensor.untyped_storage()
    ptr = storage.data_ptr()
    size = storage.nbytes()

    buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
        ptr, size, Metal.MTLResourceStorageModeShared, None
    )

    if buffer is None:
        raise RuntimeError("Failed to create Metal buffer from tensor")

    return buffer


def _dispatch_attention_kernel(
    lib: Any,
    function_name: str,
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    buffers: list[Any],
    wait: bool = True,
) -> None:
    """Dispatch a Sliding Window Attention Metal kernel."""
    pipeline = lib.get_pipeline(function_name, library_name="sliding_window_attention")

    command_buffer = lib.command_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline)

    # Bind buffers
    for i, buf in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buf, 0, i)

    # Dispatch
    grid_size = Metal.MTLSizeMake(*grid)
    tg_size = Metal.MTLSizeMake(*threadgroup)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

    encoder.endEncoding()
    command_buffer.commit()

    if wait:
        command_buffer.waitUntilCompleted()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sliding_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Compute Sliding Window Attention with automatic kernel selection.

    Each token only attends to the most recent window_size tokens, providing
    O(seq * window) memory complexity instead of O(seq^2).

    Args:
        Q: Query tensor [batch, heads_q, seq_q, head_dim], MPS device
        K: Key tensor [batch, heads_kv, seq_k, head_dim], MPS device
        V: Value tensor [batch, heads_kv, seq_k, head_dim], MPS device
        window_size: Sliding window size (e.g., 4096 for Mistral)
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        causal: Apply causal masking within window

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim], MPS device
    """
    _require_metal_sliding_window()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    if scale is None:
        scale = head_dim**-0.5

    kernel_name = select_kernel(seq_q, num_heads_q, num_heads_kv, causal)

    if kernel_name == "sliding_window_attention_decode":
        return sliding_window_attention_decode(Q, K, V, window_size, scale)
    elif kernel_name == "sliding_window_attention_gqa":
        gqa_ratio = num_heads_q // num_heads_kv
        return sliding_window_attention_gqa(Q, K, V, window_size, scale, gqa_ratio, causal)
    else:
        return _sliding_window_attention_prefill(Q, K, V, window_size, scale, causal)


def _sliding_window_attention_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Sliding Window Attention prefill kernel dispatch."""
    _require_metal_sliding_window()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    gqa_ratio = num_heads_q // num_heads_kv if num_heads_kv > 0 else 1

    params = SlidingWindowParams(
        batch=batch,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seq_q=seq_q,
        seq_k=seq_k,
        head_dim=head_dim,
        scale=scale,
        gqa_ratio=gqa_ratio,
        window_size=window_size,
        is_causal=causal,
    )

    lib = _get_kernel_library()
    device = lib.device

    # Ensure tensors are FP16 contiguous on MPS
    Q = Q.to(device="mps", dtype=torch.float16).contiguous()
    K = K.to(device="mps", dtype=torch.float16).contiguous()
    V = V.to(device="mps", dtype=torch.float16).contiguous()

    # Allocate output
    O = torch.empty((batch, num_heads_q, seq_q, head_dim), dtype=torch.float16, device="mps")

    # Convert to Metal buffers
    Q_buf = _mps_tensor_to_metal_buffer(Q, device)
    K_buf = _mps_tensor_to_metal_buffer(K, device)
    V_buf = _mps_tensor_to_metal_buffer(V, device)
    O_buf = _mps_tensor_to_metal_buffer(O, device)
    params_buf = params.to_buffer(device)

    # Compute grid
    kernel_name = "sliding_window_attention_causal" if causal else "sliding_window_attention_prefill"
    grid = _compute_grid(params, "standard")

    # Dispatch
    _dispatch_attention_kernel(
        lib,
        kernel_name,
        grid=grid,
        threadgroup=(THREADS_PER_TG_SW, 1, 1),
        buffers=[Q_buf, K_buf, V_buf, O_buf, params_buf],
        wait=True,
    )

    return O


def sliding_window_attention_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Optimized decode kernel for seq_q=1 with sliding window.

    Only attends to the most recent window_size tokens in the KV cache.

    Args:
        Q: Query tensor [num_seqs, heads_q, 1, head_dim] or [num_seqs, heads_q, head_dim]
        K: Key cache [num_seqs, heads_kv, seq_k, head_dim]
        V: Value cache [num_seqs, heads_kv, seq_k, head_dim]
        window_size: Sliding window size
        scale: Attention scale factor

    Returns:
        Output tensor [num_seqs, heads_q, 1, head_dim] or [num_seqs, heads_q, head_dim]
    """
    _require_metal_sliding_window()

    # Handle 3D input (squeeze seq_q dimension)
    squeeze_output = Q.ndim == 3
    if squeeze_output:
        Q = Q[:, :, None, :]

    num_seqs, num_heads_q, _, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    if scale is None:
        scale = head_dim**-0.5

    lib = _get_kernel_library()
    device = lib.device

    # Ensure tensors are FP16 contiguous on MPS
    # Squeeze Q for decode kernel: [num_seqs, num_heads_q, head_dim]
    Q = Q.squeeze(2).to(device="mps", dtype=torch.float16).contiguous()
    K = K.to(device="mps", dtype=torch.float16).contiguous()
    V = V.to(device="mps", dtype=torch.float16).contiguous()

    # Allocate output: [num_seqs, num_heads_q, head_dim]
    O = torch.empty((num_seqs, num_heads_q, head_dim), dtype=torch.float16, device="mps")

    # Convert to Metal buffers
    Q_buf = _mps_tensor_to_metal_buffer(Q, device)
    K_buf = _mps_tensor_to_metal_buffer(K, device)
    V_buf = _mps_tensor_to_metal_buffer(V, device)
    O_buf = _mps_tensor_to_metal_buffer(O, device)

    # Create scalar constant buffers for decode kernel
    def make_uint32_buffer(val: int) -> Any:
        data = np.array([val], dtype=np.uint32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    def make_float_buffer(val: float) -> Any:
        data = np.array([val], dtype=np.float32)
        return device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )

    num_seqs_buf = make_uint32_buffer(num_seqs)
    num_heads_q_buf = make_uint32_buffer(num_heads_q)
    num_heads_kv_buf = make_uint32_buffer(num_heads_kv)
    seq_k_buf = make_uint32_buffer(seq_k)
    head_dim_buf = make_uint32_buffer(head_dim)
    scale_buf = make_float_buffer(scale)
    window_size_buf = make_uint32_buffer(window_size)

    # Grid: one threadgroup per (sequence, head) pair
    grid = (num_seqs * num_heads_q, 1, 1)

    # Dispatch
    _dispatch_attention_kernel(
        lib,
        "sliding_window_attention_decode",
        grid=grid,
        threadgroup=(THREADS_PER_TG_SW, 1, 1),
        buffers=[
            Q_buf,
            K_buf,
            V_buf,
            O_buf,
            num_seqs_buf,
            num_heads_q_buf,
            num_heads_kv_buf,
            seq_k_buf,
            head_dim_buf,
            scale_buf,
            window_size_buf,
        ],
        wait=True,
    )

    if squeeze_output:
        return O
    else:
        return O.unsqueeze(2)


def sliding_window_attention_gqa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
    scale: float | None = None,
    gqa_ratio: int | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Sliding Window Attention for Grouped-Query Attention.

    Optimized for models where multiple Q heads share K/V heads, combined
    with sliding window attention pattern.

    Args:
        Q: Query tensor [batch, heads_q, seq_q, head_dim]
        K: Key tensor [batch, heads_kv, seq_k, head_dim]
        V: Value tensor [batch, heads_kv, seq_k, head_dim]
        window_size: Sliding window size
        scale: Attention scale factor
        gqa_ratio: Q heads per KV head (auto-computed if None)
        causal: Apply causal masking within window

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim]
    """
    _require_metal_sliding_window()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    if scale is None:
        scale = head_dim**-0.5

    if gqa_ratio is None:
        gqa_ratio = num_heads_q // num_heads_kv

    params = SlidingWindowParams(
        batch=batch,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seq_q=seq_q,
        seq_k=seq_k,
        head_dim=head_dim,
        scale=scale,
        gqa_ratio=gqa_ratio,
        window_size=window_size,
        is_causal=causal,
    )

    lib = _get_kernel_library()
    device = lib.device

    # Ensure tensors are FP16 contiguous on MPS
    Q = Q.to(device="mps", dtype=torch.float16).contiguous()
    K = K.to(device="mps", dtype=torch.float16).contiguous()
    V = V.to(device="mps", dtype=torch.float16).contiguous()

    # Allocate output
    O = torch.empty((batch, num_heads_q, seq_q, head_dim), dtype=torch.float16, device="mps")

    # Convert to Metal buffers
    Q_buf = _mps_tensor_to_metal_buffer(Q, device)
    K_buf = _mps_tensor_to_metal_buffer(K, device)
    V_buf = _mps_tensor_to_metal_buffer(V, device)
    O_buf = _mps_tensor_to_metal_buffer(O, device)
    params_buf = params.to_buffer(device)

    # Compute grid for GQA: dispatch by KV heads
    grid = _compute_grid(params, "gqa")

    # Dispatch
    _dispatch_attention_kernel(
        lib,
        "sliding_window_attention_gqa",
        grid=grid,
        threadgroup=(THREADS_PER_TG_SW, 1, 1),
        buffers=[Q_buf, K_buf, V_buf, O_buf, params_buf],
        wait=True,
    )

    return O


def benchmark_sliding_window_attention(
    batch: int = 1,
    num_heads_q: int = 32,
    num_heads_kv: int = 8,
    seq_q: int = 4096,
    seq_k: int = 32768,
    head_dim: int = 128,
    window_size: int = 4096,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """Benchmark Sliding Window Attention performance.

    Args:
        batch: Batch size
        num_heads_q: Number of query heads
        num_heads_kv: Number of KV heads (for GQA)
        seq_q: Query sequence length
        seq_k: Key/value sequence length (KV cache size)
        head_dim: Dimension per head
        window_size: Sliding window size
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Dictionary with timing statistics and bandwidth utilization
    """
    _require_metal_sliding_window()

    import time

    # Create random inputs on MPS
    Q = torch.randn((batch, num_heads_q, seq_q, head_dim), dtype=torch.float16, device="mps")
    K = torch.randn((batch, num_heads_kv, seq_k, head_dim), dtype=torch.float16, device="mps")
    V = torch.randn((batch, num_heads_kv, seq_k, head_dim), dtype=torch.float16, device="mps")

    scale = head_dim**-0.5

    # Select kernel
    kernel_name = select_kernel(seq_q, num_heads_q, num_heads_kv, causal=True)

    # Warmup
    for _ in range(warmup):
        _ = sliding_window_attention(Q, K, V, window_size=window_size, scale=scale, causal=True)
        torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        O = sliding_window_attention(Q, K, V, window_size=window_size, scale=scale, causal=True)
        torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate memory bandwidth
    # Key insight: sliding window means we read less K/V data
    # For each query at position p, we read at most window_size K/V entries
    # Total K/V reads: seq_q * min(window_size, seq_k) (not seq_q * seq_k)
    gqa_ratio = num_heads_q // num_heads_kv

    # Q reads: full tensor
    q_bytes = batch * num_heads_q * seq_q * head_dim * 2

    # K/V reads: window_size per query position
    effective_kv_per_query = min(window_size, seq_k)
    # For prefill, average window size grows linearly to window_size
    avg_window = min(seq_q // 2, window_size) if seq_q > 1 else effective_kv_per_query
    kv_bytes = batch * num_heads_kv * seq_q * avg_window * head_dim * 2 * 2  # K and V

    # O writes
    o_bytes = batch * num_heads_q * seq_q * head_dim * 2

    total_bytes = q_bytes + kv_bytes + o_bytes
    bandwidth_gbps = total_bytes / min_time / 1e9

    # Compare to full attention
    full_kv_bytes = batch * num_heads_kv * seq_q * seq_k * head_dim * 2 * 2
    full_total_bytes = q_bytes + full_kv_bytes + o_bytes
    memory_savings = (full_total_bytes - total_bytes) / full_total_bytes * 100

    # M4 Max theoretical bandwidth: ~800 GB/s
    peak_bandwidth = 800  # GB/s
    bandwidth_util = bandwidth_gbps / peak_bandwidth * 100

    return {
        "kernel": kernel_name,
        "batch": batch,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "seq_q": seq_q,
        "seq_k": seq_k,
        "head_dim": head_dim,
        "window_size": window_size,
        "gqa_ratio": gqa_ratio,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "total_bytes": total_bytes,
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_util_pct": bandwidth_util,
        "memory_savings_pct": memory_savings,
    }


if __name__ == "__main__":
    # Run benchmark with Mistral configuration
    print("Sliding Window Attention Benchmark")
    print("=" * 60)

    configs = [
        # Mistral-7B: 32 heads, GQA=4, head_dim=128, window=4096
        {
            "num_heads_q": 32,
            "num_heads_kv": 8,
            "head_dim": 128,
            "seq_q": 2048,
            "seq_k": 32768,
            "window_size": 4096,
        },
        # Decode (seq_q=1) with long KV cache
        {
            "num_heads_q": 32,
            "num_heads_kv": 8,
            "head_dim": 128,
            "seq_q": 1,
            "seq_k": 32768,
            "window_size": 4096,
        },
        # Short sequence (window > seq)
        {
            "num_heads_q": 32,
            "num_heads_kv": 8,
            "head_dim": 128,
            "seq_q": 2048,
            "seq_k": 2048,
            "window_size": 4096,
        },
    ]

    for config in configs:
        result = benchmark_sliding_window_attention(**config)
        print(f"\nConfig: {config}")
        print(f"  Kernel: {result['kernel']}")
        print(f"  Avg time: {result['avg_time_ms']:.2f} ms")
        print(f"  Min time: {result['min_time_ms']:.2f} ms")
        print(
            f"  Bandwidth: {result['bandwidth_gbps']:.1f} GB/s ({result['bandwidth_util_pct']:.1f}%)"
        )
        print(f"  Memory savings vs full attention: {result['memory_savings_pct']:.1f}%")
