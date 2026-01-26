"""
Flash Attention V2 dispatch wrapper for Apple Silicon Metal kernels.

This module provides Python bindings to the optimized Flash Attention V2 Metal
kernels with automatic kernel selection based on attention configuration.

Kernels:
    flash_attention_v2         - Tiled prefill, non-causal
    flash_attention_v2_causal  - Tiled prefill, causal
    flash_attention_v2_decode  - Single-query decode (seq_q=1)
    flash_attention_v2_gqa     - Grouped-Query Attention
    flash_attention_v2_mqa     - Multi-Query Attention (single KV head)

Optimizations:
    - Process Q tiles (16 rows) for better arithmetic intensity
    - Vectorized 128-bit loads for memory bandwidth (>80% utilization)
    - Branchless causal masking
    - Double-buffered K/V tile streaming
    - Specialized decode kernel for autoregressive generation
    - GQA/MQA kernels that load K/V once for multiple Q heads

Precision:
    Currently uses FP16 (torch.float16) for attention computation because
    the Metal shaders use `half` type. BF16 input tensors are converted to
    FP16 before dispatch. This is acceptable because attention scores are
    bounded softmax outputs, and the FP16 range is sufficient.

    TODO: Add native BF16 shader variants using Metal's `bfloat` type
    (available on M3+) for end-to-end BF16 compute.

Usage:
    from metal_marlin.flash_attention_v2 import flash_attention_v2

    # Automatic kernel selection based on inputs
    output = flash_attention_v2(Q, K, V, scale=scale, causal=True)

    # GLM-4.7-Flash configuration (32 Q heads, 2 KV heads)
    output = flash_attention_v2_gqa(Q, K, V, gqa_ratio=16, causal=True)

    # Decode phase (seq_q=1)
    output = flash_attention_v2_decode(Q, K, V, scale=scale)
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
TILE_Q = 16
TILE_KV = 64
NUM_SIMDGROUPS = 4
THREADS_PER_TG = NUM_SIMDGROUPS * 32  # 128


def _require_metal_attention() -> None:
    """Raise if Metal dispatch is not available."""
    if not HAS_TORCH:
        raise RuntimeError("Flash Attention V2 requires PyTorch. Install with: pip install torch")
    if not HAS_MPS:
        raise RuntimeError("Flash Attention V2 requires PyTorch MPS backend (Apple Silicon).")
    if not HAS_PYOBJC_METAL:
        raise RuntimeError(
            "Flash Attention V2 requires PyObjC Metal. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )


@dataclass
class AttentionParams:
    """Parameters for Flash Attention kernel dispatch."""

    batch: int
    num_heads_q: int
    num_heads_kv: int
    seq_q: int
    seq_k: int
    head_dim: int
    scale: float
    gqa_ratio: int
    is_causal: bool

    def to_buffer(self, device: Any) -> Any:
        """Pack parameters into Metal buffer for kernel dispatch.

        Args:
            device: MTLDevice for buffer allocation

        Returns:
            MTLBuffer containing packed parameters
        """
        # Pack as uint32 array (is_causal as 0/1)
        # Note: scale is reinterpreted as uint32 for the Metal struct
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
                1 if self.is_causal else 0,
            ],
            dtype=np.uint32,
        )
        buffer = device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
        )
        return buffer


def _get_kernel_source() -> str:
    """Load Flash Attention V2 Metal kernel source."""
    kernel_path = Path(__file__).parent.parent / "src" / "flash_attention_v2.metal"
    if kernel_path.exists():
        return kernel_path.read_text()
    raise FileNotFoundError(f"Flash Attention V2 kernel not found at {kernel_path}")


def _compute_grid(params: AttentionParams, kernel_type: str) -> tuple[int, int, int]:
    """Compute Metal grid dimensions for dispatch."""
    if kernel_type == "decode":
        # Decode: one threadgroup per (sequence, head) pair
        return (params.batch * params.num_heads_q, 1, 1)
    elif kernel_type == "gqa":
        # GQA: dispatch by KV heads
        q_tiles = (params.seq_q + TILE_Q - 1) // TILE_Q
        return (params.num_heads_kv, q_tiles, params.batch)
    elif kernel_type == "mqa":
        # MQA: batch Q heads together
        head_groups = (params.num_heads_q + NUM_SIMDGROUPS - 1) // NUM_SIMDGROUPS
        q_tiles = (params.seq_q + TILE_Q - 1) // TILE_Q
        return (head_groups, q_tiles, params.batch)
    else:
        # Standard prefill: dispatch by Q heads
        q_tiles = (params.seq_q + TILE_Q - 1) // TILE_Q
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
        return "flash_attention_v2_decode"

    if num_heads_kv == 1:
        # MQA: all Q heads share single KV head
        return "flash_attention_v2_mqa"

    if gqa_ratio >= 4:
        # GQA with high ratio: use specialized kernel
        return "flash_attention_v2_gqa"

    # Standard prefill
    if is_causal:
        return "flash_attention_v2_causal"
    else:
        return "flash_attention_v2"


# ---------------------------------------------------------------------------
# Kernel library singleton
# ---------------------------------------------------------------------------

_kernel_lib: Any = None


def _get_kernel_library() -> Any:
    """Get or create the Flash Attention kernel library."""
    global _kernel_lib
    if _kernel_lib is None:
        from .metal_dispatch import MetalKernelLibrary

        _kernel_lib = MetalKernelLibrary()
        source = _get_kernel_source()
        _kernel_lib.compile_source("flash_attention_v2", source)
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
    """Dispatch a Flash Attention Metal kernel."""
    pipeline = lib.get_pipeline(function_name, library_name="flash_attention_v2")

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


def flash_attention_v2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Compute Flash Attention V2 with automatic kernel selection.

    Args:
        Q: Query tensor [batch, heads_q, seq_q, head_dim], MPS device
        K: Key tensor [batch, heads_kv, seq_k, head_dim], MPS device
        V: Value tensor [batch, heads_kv, seq_k, head_dim], MPS device
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        causal: Apply causal masking

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim], MPS device
    """
    _require_metal_attention()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    if scale is None:
        scale = head_dim**-0.5

    kernel_name = select_kernel(seq_q, num_heads_q, num_heads_kv, causal)

    if kernel_name == "flash_attention_v2_decode":
        return flash_attention_v2_decode(Q, K, V, scale)
    elif kernel_name == "flash_attention_v2_mqa":
        return flash_attention_v2_mqa(Q, K, V, scale, causal)
    elif kernel_name == "flash_attention_v2_gqa":
        gqa_ratio = num_heads_q // num_heads_kv
        return flash_attention_v2_gqa(Q, K, V, scale, gqa_ratio, causal)
    else:
        return _flash_attention_v2_prefill(Q, K, V, scale, causal)


def _flash_attention_v2_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Flash Attention V2 prefill kernel dispatch."""
    _require_metal_attention()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    gqa_ratio = num_heads_q // num_heads_kv if num_heads_kv > 0 else 1

    params = AttentionParams(
        batch=batch,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seq_q=seq_q,
        seq_k=seq_k,
        head_dim=head_dim,
        scale=scale,
        gqa_ratio=gqa_ratio,
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
    kernel_name = "flash_attention_v2_causal" if causal else "flash_attention_v2"
    grid = _compute_grid(params, "causal" if causal else "standard")

    # Dispatch
    _dispatch_attention_kernel(
        lib,
        kernel_name,
        grid=grid,
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[Q_buf, K_buf, V_buf, O_buf, params_buf],
        wait=True,
    )

    return O


def flash_attention_v2_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Optimized decode kernel for seq_q=1.

    Args:
        Q: Query tensor [num_seqs, heads_q, 1, head_dim] or [num_seqs, heads_q, head_dim]
        K: Key cache [num_seqs, heads_kv, seq_k, head_dim]
        V: Value cache [num_seqs, heads_kv, seq_k, head_dim]
        scale: Attention scale factor

    Returns:
        Output tensor [num_seqs, heads_q, 1, head_dim] or [num_seqs, heads_q, head_dim]
    """
    _require_metal_attention()

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
    # The decode kernel uses separate buffer arguments, not a struct
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

    # Grid: one threadgroup per (sequence, head) pair
    grid = (num_seqs * num_heads_q, 1, 1)

    # Dispatch
    _dispatch_attention_kernel(
        lib,
        "flash_attention_v2_decode",
        grid=grid,
        threadgroup=(THREADS_PER_TG, 1, 1),
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
        ],
        wait=True,
    )

    if squeeze_output:
        return O
    else:
        return O.unsqueeze(2)


def flash_attention_v2_gqa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    gqa_ratio: int | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Flash Attention for Grouped-Query Attention.

    Optimized for models like GLM-4.7-Flash where multiple Q heads share K/V.

    Args:
        Q: Query tensor [batch, heads_q, seq_q, head_dim]
        K: Key tensor [batch, heads_kv, seq_k, head_dim]
        V: Value tensor [batch, heads_kv, seq_k, head_dim]
        scale: Attention scale factor
        gqa_ratio: Q heads per KV head (auto-computed if None)
        causal: Apply causal masking

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim]
    """
    _require_metal_attention()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    if scale is None:
        scale = head_dim**-0.5

    if gqa_ratio is None:
        gqa_ratio = num_heads_q // num_heads_kv

    params = AttentionParams(
        batch=batch,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seq_q=seq_q,
        seq_k=seq_k,
        head_dim=head_dim,
        scale=scale,
        gqa_ratio=gqa_ratio,
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
        "flash_attention_v2_gqa",
        grid=grid,
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[Q_buf, K_buf, V_buf, O_buf, params_buf],
        wait=True,
    )

    return O


def flash_attention_v2_mqa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Flash Attention for Multi-Query Attention (single KV head).

    All Q heads share the same K/V, providing maximum memory savings.

    Args:
        Q: Query tensor [batch, heads_q, seq_q, head_dim]
        K: Key tensor [batch, 1, seq_k, head_dim]
        V: Value tensor [batch, 1, seq_k, head_dim]
        scale: Attention scale factor
        causal: Apply causal masking

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim]
    """
    _require_metal_attention()

    batch, num_heads_q, seq_q, head_dim = Q.shape
    _, num_heads_kv, seq_k, _ = K.shape

    if scale is None:
        scale = head_dim**-0.5

    gqa_ratio = num_heads_q // num_heads_kv if num_heads_kv > 0 else num_heads_q

    params = AttentionParams(
        batch=batch,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seq_q=seq_q,
        seq_k=seq_k,
        head_dim=head_dim,
        scale=scale,
        gqa_ratio=gqa_ratio,
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

    # Compute grid for MQA
    grid = _compute_grid(params, "mqa")

    # Dispatch
    _dispatch_attention_kernel(
        lib,
        "flash_attention_v2_mqa",
        grid=grid,
        threadgroup=(THREADS_PER_TG, 1, 1),
        buffers=[Q_buf, K_buf, V_buf, O_buf, params_buf],
        wait=True,
    )

    return O


def benchmark_flash_attention(
    batch: int = 1,
    num_heads_q: int = 32,
    num_heads_kv: int = 2,
    seq_q: int = 2048,
    seq_k: int = 2048,
    head_dim: int = 64,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """Benchmark Flash Attention V2 performance.

    Args:
        batch: Batch size
        num_heads_q: Number of query heads
        num_heads_kv: Number of KV heads (for GQA)
        seq_q: Query sequence length
        seq_k: Key/value sequence length
        head_dim: Dimension per head
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Dictionary with timing statistics and bandwidth utilization
    """
    _require_metal_attention()

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
        _ = flash_attention_v2(Q, K, V, scale=scale, causal=True)
        torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        O = flash_attention_v2(Q, K, V, scale=scale, causal=True)
        torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate memory bandwidth
    # Read: Q, K, V (all float16)
    # Write: O (float16)
    gqa_ratio = num_heads_q // num_heads_kv
    q_bytes = batch * num_heads_q * seq_q * head_dim * 2
    k_bytes = batch * num_heads_kv * seq_k * head_dim * 2
    v_bytes = batch * num_heads_kv * seq_k * head_dim * 2
    o_bytes = batch * num_heads_q * seq_q * head_dim * 2

    total_bytes = q_bytes + k_bytes + v_bytes + o_bytes
    bandwidth_gbps = total_bytes / min_time / 1e9

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
        "gqa_ratio": gqa_ratio,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "total_bytes": total_bytes,
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_util_pct": bandwidth_util,
    }


if __name__ == "__main__":
    # Run benchmark with GLM-4.7-Flash configuration
    print("Flash Attention V2 Benchmark")
    print("=" * 60)

    configs = [
        # GLM-4.7-Flash: 32 Q heads, 2 KV heads, head_dim=64
        {"num_heads_q": 32, "num_heads_kv": 2, "head_dim": 64, "seq_q": 2048, "seq_k": 2048},
        # Decode (seq_q=1)
        {"num_heads_q": 32, "num_heads_kv": 2, "head_dim": 64, "seq_q": 1, "seq_k": 2048},
        # Standard MHA
        {"num_heads_q": 32, "num_heads_kv": 32, "head_dim": 128, "seq_q": 2048, "seq_k": 2048},
    ]

    for config in configs:
        result = benchmark_flash_attention(**config)
        print(f"\nConfig: {config}")
        print(f"  Kernel: {result['kernel']}")
        print(f"  Avg time: {result['avg_time_ms']:.2f} ms")
        print(f"  Min time: {result['min_time_ms']:.2f} ms")
        print(
            f"  Bandwidth: {result['bandwidth_gbps']:.1f} GB/s ({result['bandwidth_util_pct']:.1f}%)"
        )
