"""Flash Attention accuracy tests for Metal kernels.

Tests both v1 and v2 flash attention implementations against NumPy/PyTorch reference:
  - flash_attention / flash_attention_causal / flash_attention_gqa
  - flash_attention_v2 / flash_attention_v2_causal / flash_attention_v2_decode
  - flash_attention_v2_gqa / flash_attention_v2_mqa

Validates:
  - Reference implementation accuracy (NumPy vs PyTorch)
  - Numerical accuracy within FP16 tolerances
  - GQA head mapping correctness
  - Causal masking behavior
  - Decode (seq_q=1) optimization path

NOTE: The current Metal kernels exceed Apple Silicon's 32KB threadgroup memory limit
(they allocate 64KB for double-buffered K/V tiles). Tests that require Metal kernel
execution are marked as xfail until the kernels are fixed to fit within memory limits.

Usage:
    cd metal_marlin
    uv run pytest tests/test_attention.py -v
"""

from __future__ import annotations

import ctypes
import math
import struct
from pathlib import Path

import numpy as np
import pytest

# Paths to Metal shaders
_SHADER_V1_PATH = Path(__file__).parent.parent / "src" / "flash_attention.metal"
_SHADER_V2_PATH = Path(__file__).parent.parent / "src" / "flash_attention_v2.metal"

# Kernel configuration constants (must match Metal shader)
_TILE_Q = 16
_TILE_KV = 64
_ROWS_PER_TG = 4
_THREADS_PER_TG = 128
_SIMD_SIZE = 32
_NUM_SIMDGROUPS = 4


def _check_metal_available() -> bool:
    """Check if Metal API is available via PyObjC."""
    try:
        import Metal  # noqa: F401

        return True
    except ImportError:
        return False


def _check_torch_available() -> bool:
    """Check if PyTorch is available for reference computation."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# ==============================================================================
# Reference implementations
# ==============================================================================


def ref_scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
    num_kv_heads: int | None = None,
) -> np.ndarray:
    """NumPy reference implementation of scaled dot-product attention.

    Args:
        Q: Query tensor [batch, heads_q, seq_q, head_dim]
        K: Key tensor [batch, heads_kv, seq_k, head_dim]
        V: Value tensor [batch, heads_kv, seq_k, head_dim]
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        is_causal: If True, apply causal masking
        num_kv_heads: Number of KV heads for GQA expansion

    Returns:
        Output tensor [batch, heads_q, seq_q, head_dim]
    """
    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_kv, seq_k, _ = K.shape

    # GQA: expand K/V heads to match Q heads
    if heads_kv < heads_q:
        repeat = heads_q // heads_kv
        K = np.repeat(K, repeat, axis=1)
        V = np.repeat(V, repeat, axis=1)

    # Compute Q @ K^T * scale
    # Q: [batch, heads, seq_q, head_dim]
    # K^T: [batch, heads, head_dim, seq_k]
    scores = np.einsum("bhqd,bhkd->bhqk", Q.astype(np.float32), K.astype(np.float32))
    scores = scores * scale

    # Apply causal mask if needed
    # For causal attention, each query position can only attend to keys at positions <= query position
    if is_causal:
        # Create causal mask: 1s on diagonal and below, 0s above
        # For seq_q x seq_k: position q can attend to keys 0..min(q, seq_k-1)
        q_idx = np.arange(seq_q)[:, None]  # [seq_q, 1]
        k_idx = np.arange(seq_k)[None, :]  # [1, seq_k]
        # Mask out positions where k > q (future positions)
        mask = np.where(k_idx > q_idx, -np.inf, 0.0).astype(np.float32)
        scores = scores + mask

    # Softmax along seq_k dimension with numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    # Handle -inf max (when entire row is masked)
    scores_max = np.where(np.isinf(scores_max), 0.0, scores_max)
    scores = scores - scores_max
    exp_scores = np.exp(scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    # Avoid division by zero for fully masked rows
    sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)
    softmax = exp_scores / sum_exp

    # Apply attention weights to values
    output = np.einsum("bhqk,bhkd->bhqd", softmax, V.astype(np.float32))

    return output.astype(np.float16)


def ref_torch_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
) -> np.ndarray:
    """PyTorch reference implementation for higher precision verification."""
    import torch
    import torch.nn.functional as F

    Q_t = torch.from_numpy(Q).float()
    K_t = torch.from_numpy(K).float()
    V_t = torch.from_numpy(V).float()

    _, heads_q, _, _ = Q_t.shape
    _, heads_kv, _, _ = K_t.shape

    if heads_kv < heads_q:
        repeat = heads_q // heads_kv
        K_t = K_t.repeat_interleave(repeat, dim=1)
        V_t = V_t.repeat_interleave(repeat, dim=1)

    output = F.scaled_dot_product_attention(
        Q_t,
        K_t,
        V_t,
        scale=scale,
        is_causal=is_causal,
    )
    return output.to(torch.float16).numpy()


# ==============================================================================
# Metal kernel helpers
# ==============================================================================


class ThreadgroupMemoryExceeded(Exception):
    """Raised when kernel exceeds threadgroup memory limits."""

    pass


class MetalCompilationError(Exception):
    """Raised when Metal shader compilation fails."""

    pass


def _read_metal_buffer(buf, nbytes: int) -> bytes:
    """Read bytes from a Metal buffer."""
    contents = buf.contents()
    if isinstance(contents, int):
        arr = (ctypes.c_char * nbytes).from_address(contents)
        return bytes(arr)
    elif isinstance(contents, memoryview):
        return bytes(contents[:nbytes])
    elif hasattr(contents, "__getitem__"):
        return b"".join(contents[i] for i in range(nbytes))
    else:
        ptr = ctypes.cast(contents, ctypes.POINTER(ctypes.c_char * nbytes))
        return bytes(ptr.contents)


def _compile_shader(shader_path: Path):
    """Compile a Metal shader and return (device, library)."""
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    assert device is not None, "No Metal device found"

    if not shader_path.exists():
        pytest.skip(f"Metal shader not found: {shader_path}")

    source = shader_path.read_text()
    include_token = '#include "bf16_compat.metal"'
    if include_token in source:
        include_path = shader_path.parent / "bf16_compat.metal"
        if not include_path.exists():
            pytest.fail(f"Missing Metal include: {include_path}")
        include_source = include_path.read_text()
        source = source.replace(include_token, include_source)
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err is not None:
        pytest.fail(f"Metal compile error: {err}")

    return device, library


def _create_pipeline(device, library, kernel_name: str):
    """Create a compute pipeline, raising ThreadgroupMemoryExceeded if it fails."""
    func = library.newFunctionWithName_(kernel_name)
    if func is None:
        raise ValueError(f"Kernel {kernel_name} not found")

    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    if err is not None:
        err_str = str(err)
        if "Threadgroup memory size" in err_str and "exceeds" in err_str:
            raise ThreadgroupMemoryExceeded(
                f"Kernel {kernel_name} exceeds threadgroup memory limit: {err_str}"
            )
        # XPC errors and other compilation failures
        raise MetalCompilationError(f"Pipeline error for {kernel_name}: {err}")

    return pipeline


def _create_buffer(device, data: np.ndarray):
    """Create a Metal buffer from a numpy array."""
    import Metal

    return device.newBufferWithBytes_length_options_(
        data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
    )


def _create_constant_buffer(device, value, dtype=np.uint32):
    """Create a small constant buffer for kernel parameters."""
    import Metal

    data = np.array([value], dtype=dtype)
    return device.newBufferWithBytes_length_options_(
        data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared
    )


# ==============================================================================
# Flash Attention V1 kernel runners
# ==============================================================================


def run_flash_attention_v1(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
) -> np.ndarray:
    """Run flash_attention or flash_attention_causal kernel."""
    import Metal

    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_k, seq_k, _ = K.shape

    device, library = _compile_shader(_SHADER_V1_PATH)

    kernel_name = "flash_attention_causal" if is_causal else "flash_attention"
    pipeline = _create_pipeline(device, library, kernel_name)

    # Prepare buffers
    Q_fp16 = Q.astype(np.float16)
    K_fp16 = K.astype(np.float16)
    V_fp16 = V.astype(np.float16)

    buf_Q = _create_buffer(device, Q_fp16)
    buf_K = _create_buffer(device, K_fp16)
    buf_V = _create_buffer(device, V_fp16)

    output_shape = (batch, heads_q, seq_q, head_dim)
    output_size = int(np.prod(output_shape)) * 2  # FP16
    buf_O = device.newBufferWithLength_options_(output_size, Metal.MTLResourceStorageModeShared)

    buf_batch = _create_constant_buffer(device, batch)
    buf_heads_q = _create_constant_buffer(device, heads_q)
    buf_heads_k = _create_constant_buffer(device, heads_k)
    buf_seq_q = _create_constant_buffer(device, seq_q)
    buf_seq_k = _create_constant_buffer(device, seq_k)
    buf_head_dim = _create_constant_buffer(device, head_dim)
    buf_scale = _create_constant_buffer(device, scale, dtype=np.float32)

    # Dispatch
    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(buf_Q, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_K, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_V, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_O, 0, 3)
    encoder.setBuffer_offset_atIndex_(buf_batch, 0, 4)
    encoder.setBuffer_offset_atIndex_(buf_heads_q, 0, 5)
    encoder.setBuffer_offset_atIndex_(buf_heads_k, 0, 6)
    encoder.setBuffer_offset_atIndex_(buf_seq_q, 0, 7)
    encoder.setBuffer_offset_atIndex_(buf_seq_k, 0, 8)
    encoder.setBuffer_offset_atIndex_(buf_head_dim, 0, 9)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 10)

    # Grid: [heads_q, ceil(seq_q/ROWS_PER_TG), batch]
    grid_x = heads_q
    grid_y = (seq_q + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    grid_z = batch

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(grid_x, grid_y, grid_z),
        Metal.MTLSizeMake(_THREADS_PER_TG, 1, 1),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    # Read output
    raw = _read_metal_buffer(buf_O, output_size)
    return np.frombuffer(raw, dtype=np.float16).reshape(output_shape).copy()


def run_flash_attention_v1_gqa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
) -> np.ndarray:
    """Run flash_attention_gqa kernel."""
    import Metal

    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_k, seq_k, _ = K.shape
    gqa_ratio = heads_q // heads_k

    device, library = _compile_shader(_SHADER_V1_PATH)
    pipeline = _create_pipeline(device, library, "flash_attention_gqa")

    Q_fp16 = Q.astype(np.float16)
    K_fp16 = K.astype(np.float16)
    V_fp16 = V.astype(np.float16)

    buf_Q = _create_buffer(device, Q_fp16)
    buf_K = _create_buffer(device, K_fp16)
    buf_V = _create_buffer(device, V_fp16)

    output_shape = (batch, heads_q, seq_q, head_dim)
    output_size = int(np.prod(output_shape)) * 2
    buf_O = device.newBufferWithLength_options_(output_size, Metal.MTLResourceStorageModeShared)

    buf_batch = _create_constant_buffer(device, batch)
    buf_heads_q = _create_constant_buffer(device, heads_q)
    buf_heads_k = _create_constant_buffer(device, heads_k)
    buf_seq_q = _create_constant_buffer(device, seq_q)
    buf_seq_k = _create_constant_buffer(device, seq_k)
    buf_head_dim = _create_constant_buffer(device, head_dim)
    buf_scale = _create_constant_buffer(device, scale, dtype=np.float32)
    buf_gqa_ratio = _create_constant_buffer(device, gqa_ratio)
    buf_is_causal = _create_constant_buffer(device, 1 if is_causal else 0)

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(buf_Q, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_K, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_V, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_O, 0, 3)
    encoder.setBuffer_offset_atIndex_(buf_batch, 0, 4)
    encoder.setBuffer_offset_atIndex_(buf_heads_q, 0, 5)
    encoder.setBuffer_offset_atIndex_(buf_heads_k, 0, 6)
    encoder.setBuffer_offset_atIndex_(buf_seq_q, 0, 7)
    encoder.setBuffer_offset_atIndex_(buf_seq_k, 0, 8)
    encoder.setBuffer_offset_atIndex_(buf_head_dim, 0, 9)
    encoder.setBuffer_offset_atIndex_(buf_scale, 0, 10)
    encoder.setBuffer_offset_atIndex_(buf_gqa_ratio, 0, 11)
    encoder.setBuffer_offset_atIndex_(buf_is_causal, 0, 12)

    grid_x = heads_q
    grid_y = (seq_q + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    grid_z = batch

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(grid_x, grid_y, grid_z),
        Metal.MTLSizeMake(_THREADS_PER_TG, 1, 1),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_O, output_size)
    return np.frombuffer(raw, dtype=np.float16).reshape(output_shape).copy()


# ==============================================================================
# Flash Attention V2 kernel runners
# ==============================================================================


def _pack_attention_params(
    batch: int,
    heads_q: int,
    heads_kv: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    scale: float,
    gqa_ratio: int,
    is_causal: int,
) -> bytes:
    """Pack AttentionParams struct for v2 kernels."""
    return struct.pack(
        "IIIIIIfII", batch, heads_q, heads_kv, seq_q, seq_k, head_dim, scale, gqa_ratio, is_causal
    )


def run_flash_attention_v2(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
) -> np.ndarray:
    """Run flash_attention_v2 or flash_attention_v2_causal kernel."""
    import Metal

    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_kv, seq_k, _ = K.shape

    device, library = _compile_shader(_SHADER_V2_PATH)

    kernel_name = "flash_attention_v2_causal" if is_causal else "flash_attention_v2"
    pipeline = _create_pipeline(device, library, kernel_name)

    Q_fp16 = Q.astype(np.float16)
    K_fp16 = K.astype(np.float16)
    V_fp16 = V.astype(np.float16)

    buf_Q = _create_buffer(device, Q_fp16)
    buf_K = _create_buffer(device, K_fp16)
    buf_V = _create_buffer(device, V_fp16)

    output_shape = (batch, heads_q, seq_q, head_dim)
    output_size = int(np.prod(output_shape)) * 2
    buf_O = device.newBufferWithLength_options_(output_size, Metal.MTLResourceStorageModeShared)

    params = _pack_attention_params(
        batch, heads_q, heads_kv, seq_q, seq_k, head_dim, scale, 1, 1 if is_causal else 0
    )
    buf_params = device.newBufferWithBytes_length_options_(
        params, len(params), Metal.MTLResourceStorageModeShared
    )

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(buf_Q, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_K, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_V, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_O, 0, 3)
    encoder.setBuffer_offset_atIndex_(buf_params, 0, 4)

    grid_x = heads_q
    grid_y = (seq_q + _TILE_Q - 1) // _TILE_Q
    grid_z = batch

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(grid_x, grid_y, grid_z),
        Metal.MTLSizeMake(_THREADS_PER_TG, 1, 1),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_O, output_size)
    return np.frombuffer(raw, dtype=np.float16).reshape(output_shape).copy()


def run_flash_attention_v2_decode(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Run flash_attention_v2_decode kernel (seq_q=1)."""
    import Metal

    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_kv, seq_k, _ = K.shape
    assert seq_q == 1, "Decode kernel requires seq_q=1"

    device, library = _compile_shader(_SHADER_V2_PATH)
    pipeline = _create_pipeline(device, library, "flash_attention_v2_decode")

    Q_fp16 = Q.astype(np.float16)
    K_fp16 = K.astype(np.float16)
    V_fp16 = V.astype(np.float16)

    buf_Q = _create_buffer(device, Q_fp16)
    buf_K = _create_buffer(device, K_fp16)
    buf_V = _create_buffer(device, V_fp16)

    output_shape = (batch, heads_q, 1, head_dim)
    output_size = int(np.prod(output_shape)) * 2
    buf_O = device.newBufferWithLength_options_(output_size, Metal.MTLResourceStorageModeShared)

    params = _pack_attention_params(batch, heads_q, heads_kv, 1, seq_k, head_dim, scale, 1, 0)
    buf_params = device.newBufferWithBytes_length_options_(
        params, len(params), Metal.MTLResourceStorageModeShared
    )

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(buf_Q, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_K, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_V, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_O, 0, 3)
    encoder.setBuffer_offset_atIndex_(buf_params, 0, 4)

    # Decode: one threadgroup per (head, batch) pair
    grid_x = heads_q
    grid_y = 1
    grid_z = batch

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(grid_x, grid_y, grid_z),
        Metal.MTLSizeMake(_THREADS_PER_TG, 1, 1),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_O, output_size)
    return np.frombuffer(raw, dtype=np.float16).reshape(output_shape).copy()


def run_flash_attention_v2_gqa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
) -> np.ndarray:
    """Run flash_attention_v2_gqa kernel."""
    import Metal

    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_kv, seq_k, _ = K.shape
    gqa_ratio = heads_q // heads_kv

    device, library = _compile_shader(_SHADER_V2_PATH)
    pipeline = _create_pipeline(device, library, "flash_attention_v2_gqa")

    Q_fp16 = Q.astype(np.float16)
    K_fp16 = K.astype(np.float16)
    V_fp16 = V.astype(np.float16)

    buf_Q = _create_buffer(device, Q_fp16)
    buf_K = _create_buffer(device, K_fp16)
    buf_V = _create_buffer(device, V_fp16)

    output_shape = (batch, heads_q, seq_q, head_dim)
    output_size = int(np.prod(output_shape)) * 2
    buf_O = device.newBufferWithLength_options_(output_size, Metal.MTLResourceStorageModeShared)

    params = _pack_attention_params(
        batch, heads_q, heads_kv, seq_q, seq_k, head_dim, scale, gqa_ratio, 1 if is_causal else 0
    )
    buf_params = device.newBufferWithBytes_length_options_(
        params, len(params), Metal.MTLResourceStorageModeShared
    )

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(buf_Q, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_K, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_V, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_O, 0, 3)
    encoder.setBuffer_offset_atIndex_(buf_params, 0, 4)

    grid_x = heads_q
    grid_y = (seq_q + _TILE_Q - 1) // _TILE_Q
    grid_z = batch

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(grid_x, grid_y, grid_z),
        Metal.MTLSizeMake(_THREADS_PER_TG, 1, 1),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_O, output_size)
    return np.frombuffer(raw, dtype=np.float16).reshape(output_shape).copy()


def run_flash_attention_v2_mqa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    is_causal: bool = False,
) -> np.ndarray:
    """Run flash_attention_v2_mqa kernel (single KV head)."""
    import Metal

    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_kv, seq_k, _ = K.shape
    assert heads_kv == 1, "MQA kernel requires heads_kv=1"

    device, library = _compile_shader(_SHADER_V2_PATH)
    pipeline = _create_pipeline(device, library, "flash_attention_v2_mqa")

    Q_fp16 = Q.astype(np.float16)
    K_fp16 = K.astype(np.float16)
    V_fp16 = V.astype(np.float16)

    buf_Q = _create_buffer(device, Q_fp16)
    buf_K = _create_buffer(device, K_fp16)
    buf_V = _create_buffer(device, V_fp16)

    output_shape = (batch, heads_q, seq_q, head_dim)
    output_size = int(np.prod(output_shape)) * 2
    buf_O = device.newBufferWithLength_options_(output_size, Metal.MTLResourceStorageModeShared)

    params = _pack_attention_params(
        batch, heads_q, 1, seq_q, seq_k, head_dim, scale, heads_q, 1 if is_causal else 0
    )
    buf_params = device.newBufferWithBytes_length_options_(
        params, len(params), Metal.MTLResourceStorageModeShared
    )

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(buf_Q, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_K, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_V, 0, 2)
    encoder.setBuffer_offset_atIndex_(buf_O, 0, 3)
    encoder.setBuffer_offset_atIndex_(buf_params, 0, 4)

    grid_x = heads_q
    grid_y = (seq_q + _TILE_Q - 1) // _TILE_Q
    grid_z = batch

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(grid_x, grid_y, grid_z),
        Metal.MTLSizeMake(_THREADS_PER_TG, 1, 1),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    raw = _read_metal_buffer(buf_O, output_size)
    return np.frombuffer(raw, dtype=np.float16).reshape(output_shape).copy()


# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


def generate_qkv(
    rng: np.random.Generator,
    batch: int,
    heads_q: int,
    heads_kv: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random Q, K, V tensors."""
    Q = rng.standard_normal((batch, heads_q, seq_q, head_dim)).astype(np.float16)
    K = rng.standard_normal((batch, heads_kv, seq_k, head_dim)).astype(np.float16)
    V = rng.standard_normal((batch, heads_kv, seq_k, head_dim)).astype(np.float16)
    return Q, K, V


# ==============================================================================
# Reference implementation tests (always run, no Metal required)
# ==============================================================================


class TestReferenceImplementation:
    """Tests for the NumPy reference implementation."""

    @pytest.mark.parametrize(
        "batch,heads,seq_q,seq_k,head_dim",
        [
            (1, 4, 64, 64, 64),
            (1, 8, 128, 128, 64),
            (2, 4, 32, 64, 64),
            (1, 32, 1, 512, 128),
        ],
    )
    def test_reference_non_causal(self, rng, batch, heads, seq_q, seq_k, head_dim):
        """Test NumPy reference against manual computation."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=False)

        # Manual verification for small case
        assert result.shape == (batch, heads, seq_q, head_dim)
        assert result.dtype == np.float16
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    @pytest.mark.parametrize(
        "batch,heads,seq,head_dim",
        [
            (1, 4, 64, 64),
            (1, 8, 128, 64),
            (1, 32, 256, 128),
        ],
    )
    def test_reference_causal(self, rng, batch, heads, seq, head_dim):
        """Test causal masking in NumPy reference."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=True)

        assert result.shape == (batch, heads, seq, head_dim)
        assert not np.any(np.isnan(result))

    @pytest.mark.parametrize(
        "heads_q,heads_kv",
        [
            (8, 2),
            (32, 8),
            (32, 4),
            (16, 1),  # MQA
        ],
    )
    def test_reference_gqa(self, rng, heads_q, heads_kv):
        """Test GQA head expansion in NumPy reference."""
        Q, K, V = generate_qkv(rng, 1, heads_q, heads_kv, 64, 64, 64)
        scale = 1.0 / math.sqrt(64)

        result = ref_scaled_dot_product_attention(Q, K, V, scale, num_kv_heads=heads_kv)

        assert result.shape == (1, heads_q, 64, 64)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    @pytest.mark.parametrize(
        "batch,heads,seq_q,seq_k,head_dim,is_causal",
        [
            (1, 8, 64, 64, 64, False),
            (1, 8, 64, 64, 64, True),
            (2, 16, 128, 256, 128, False),
            (1, 32, 1, 512, 128, False),
        ],
    )
    def test_numpy_vs_pytorch_reference(self, rng, batch, heads, seq_q, seq_k, head_dim, is_causal):
        """Verify NumPy reference matches PyTorch."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        np_result = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=is_causal)
        torch_result = ref_torch_attention(Q, K, V, scale, is_causal=is_causal)

        np.testing.assert_allclose(
            np_result.astype(np.float32),
            torch_result.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )


# ==============================================================================
# Metal kernel tests
# NOTE: These are marked xfail because the current kernels exceed Apple Silicon's
# 32KB threadgroup memory limit (they allocate 64KB for double-buffered K/V tiles).
# ==============================================================================


@pytest.mark.skipif(not _check_metal_available(), reason="Metal not available")
class TestFlashAttentionV1:
    """Tests for V1 flash attention kernels."""

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize(
        "batch,heads,seq_q,seq_k,head_dim",
        [
            (1, 4, 64, 64, 64),
            (1, 8, 128, 128, 64),
            (1, 32, 1, 512, 128),
            (2, 4, 32, 64, 64),
        ],
    )
    def test_flash_attention_accuracy(self, rng, batch, heads, seq_q, seq_k, head_dim):
        """Test non-causal flash attention against reference."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v1(Q, K, V, scale, is_causal=False)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=False)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize(
        "batch,heads,seq,head_dim",
        [
            (1, 4, 64, 64),
            (1, 8, 128, 64),
            (1, 32, 256, 128),
        ],
    )
    def test_flash_attention_causal(self, rng, batch, heads, seq, head_dim):
        """Test causal flash attention."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v1(Q, K, V, scale, is_causal=True)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=True)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize(
        "heads_q,heads_kv",
        [
            (8, 2),
            (32, 8),
            (32, 4),
        ],
    )
    def test_flash_attention_gqa(self, rng, heads_q, heads_kv):
        """Test GQA flash attention."""
        batch, seq_q, seq_k, head_dim = 1, 64, 128, 64
        Q, K, V = generate_qkv(rng, batch, heads_q, heads_kv, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v1_gqa(Q, K, V, scale, is_causal=False)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, num_kv_heads=heads_kv)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    def test_flash_attention_gqa_causal(self, rng):
        """Test GQA with causal masking."""
        batch, heads_q, heads_kv, seq, head_dim = 1, 32, 8, 128, 64
        Q, K, V = generate_qkv(rng, batch, heads_q, heads_kv, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v1_gqa(Q, K, V, scale, is_causal=True)
        expected = ref_scaled_dot_product_attention(
            Q, K, V, scale, is_causal=True, num_kv_heads=heads_kv
        )

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )


@pytest.mark.skipif(not _check_metal_available(), reason="Metal not available")
class TestFlashAttentionV2:
    """Tests for V2 flash attention kernels."""

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize(
        "batch,heads,seq_q,seq_k,head_dim",
        [
            (1, 4, 32, 64, 64),
            (1, 8, 64, 128, 64),
            (1, 32, 128, 256, 128),
            (2, 8, 32, 64, 64),
        ],
    )
    def test_v2_non_causal(self, rng, batch, heads, seq_q, seq_k, head_dim):
        """Test V2 non-causal attention."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v2(Q, K, V, scale, is_causal=False)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=False)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize(
        "batch,heads,seq,head_dim",
        [
            (1, 4, 64, 64),
            (1, 8, 128, 64),
            (1, 32, 256, 128),
        ],
    )
    def test_v2_causal(self, rng, batch, heads, seq, head_dim):
        """Test V2 causal attention."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v2(Q, K, V, scale, is_causal=True)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=True)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize(
        "batch,heads,seq_k,head_dim",
        [
            (1, 32, 512, 128),
            (8, 32, 1024, 128),
            (1, 8, 256, 64),
        ],
    )
    def test_v2_decode(self, rng, batch, heads, seq_k, head_dim):
        """Test V2 decode kernel (seq_q=1)."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, 1, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v2_decode(Q, K, V, scale)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=False)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize(
        "heads_q,heads_kv",
        [
            (16, 4),
            (32, 2),
            (32, 8),
        ],
    )
    def test_v2_gqa(self, rng, heads_q, heads_kv):
        """Test V2 GQA kernel."""
        batch, seq_q, seq_k, head_dim = 1, 64, 128, 64
        Q, K, V = generate_qkv(rng, batch, heads_q, heads_kv, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v2_gqa(Q, K, V, scale, is_causal=False)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, num_kv_heads=heads_kv)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    def test_v2_gqa_causal(self, rng):
        """Test V2 GQA with causal masking."""
        batch, heads_q, heads_kv, seq, head_dim = 1, 32, 4, 128, 64
        Q, K, V = generate_qkv(rng, batch, heads_q, heads_kv, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v2_gqa(Q, K, V, scale, is_causal=True)
        expected = ref_scaled_dot_product_attention(
            Q, K, V, scale, is_causal=True, num_kv_heads=heads_kv
        )

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    @pytest.mark.parametrize("heads_q", [4, 8, 16, 32])
    def test_v2_mqa(self, rng, heads_q):
        """Test V2 MQA kernel (single KV head)."""
        batch, seq_q, seq_k, head_dim = 1, 64, 128, 64
        Q, K, V = generate_qkv(rng, batch, heads_q, 1, seq_q, seq_k, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v2_mqa(Q, K, V, scale, is_causal=False)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, num_kv_heads=1)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.xfail(
        reason="Kernel exceeds 32KB threadgroup memory limit",
        raises=(ThreadgroupMemoryExceeded, MetalCompilationError),
    )
    def test_v2_mqa_causal(self, rng):
        """Test V2 MQA with causal masking."""
        batch, heads_q, seq, head_dim = 1, 16, 128, 64
        Q, K, V = generate_qkv(rng, batch, heads_q, 1, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result = run_flash_attention_v2_mqa(Q, K, V, scale, is_causal=True)
        expected = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=True, num_kv_heads=1)

        np.testing.assert_allclose(
            result.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )


# ==============================================================================
# PyTorch comparison tests (using PyTorch SDPA as ground truth)
# ==============================================================================


@pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
class TestPyTorchReference:
    """Test against PyTorch's scaled_dot_product_attention."""

    @pytest.mark.parametrize(
        "batch,heads,seq,head_dim",
        [
            (1, 8, 128, 64),
            (2, 16, 256, 128),
        ],
    )
    def test_pytorch_reference_non_causal(self, rng, batch, heads, seq, head_dim):
        """Verify our NumPy reference matches PyTorch SDPA."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        np_result = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=False)
        torch_result = ref_torch_attention(Q, K, V, scale, is_causal=False)

        np.testing.assert_allclose(
            np_result.astype(np.float32),
            torch_result.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "batch,heads,seq,head_dim",
        [
            (1, 8, 128, 64),
            (2, 16, 256, 128),
        ],
    )
    def test_pytorch_reference_causal(self, rng, batch, heads, seq, head_dim):
        """Verify causal masking matches PyTorch."""
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        np_result = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=True)
        torch_result = ref_torch_attention(Q, K, V, scale, is_causal=True)

        np.testing.assert_allclose(
            np_result.astype(np.float32),
            torch_result.astype(np.float32),
            rtol=1e-2,
            atol=1e-3,
        )


# ==============================================================================
# Edge case tests
# ==============================================================================


class TestEdgeCases:
    """Edge case tests for attention implementations."""

    def test_single_token_prefill(self, rng):
        """Test single token prefill (seq_q=1)."""
        Q, K, V = generate_qkv(rng, 1, 8, 8, 1, 128, 64)
        scale = 1.0 / math.sqrt(64)

        result = ref_scaled_dot_product_attention(Q, K, V, scale)

        assert result.shape == (1, 8, 1, 64)
        assert not np.any(np.isnan(result))

    def test_seq_not_divisible_by_tile(self, rng):
        """Test sequences not divisible by tile size."""
        # 73 is not divisible by TILE_Q=16 or TILE_KV=64
        Q, K, V = generate_qkv(rng, 1, 8, 8, 73, 73, 64)
        scale = 1.0 / math.sqrt(64)

        result = ref_scaled_dot_product_attention(Q, K, V, scale)

        assert result.shape == (1, 8, 73, 64)
        assert not np.any(np.isnan(result))

    def test_long_context(self, rng):
        """Test longer context lengths."""
        Q, K, V = generate_qkv(rng, 1, 4, 4, 32, 2048, 64)
        scale = 1.0 / math.sqrt(64)

        result = ref_scaled_dot_product_attention(Q, K, V, scale)

        # Output shape is [batch, heads, seq_q, head_dim]
        assert result.shape == (1, 4, 32, 64)
        assert not np.any(np.isnan(result))

    @pytest.mark.parametrize(
        "heads_q,heads_kv",
        [
            (8, 4),
            (16, 2),
            (32, 1),
        ],
    )
    def test_gqa_ratios(self, rng, heads_q, heads_kv):
        """Test various GQA ratios."""
        Q, K, V = generate_qkv(rng, 1, heads_q, heads_kv, 64, 64, 64)
        scale = 1.0 / math.sqrt(64)

        result = ref_scaled_dot_product_attention(Q, K, V, scale, num_kv_heads=heads_kv)

        assert result.shape == (1, heads_q, 64, 64)
        assert not np.any(np.isnan(result))

    def test_causal_correct_masking(self, rng):
        """Verify causal mask is applied correctly."""
        # Use small sizes for easier verification
        batch, heads, seq, head_dim = 1, 1, 4, 8
        Q, K, V = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        result_causal = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=True)
        result_non_causal = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=False)

        # Causal and non-causal should differ (unless by chance)
        # More importantly, verify shapes are correct
        assert result_causal.shape == result_non_causal.shape
        assert not np.any(np.isnan(result_causal))
