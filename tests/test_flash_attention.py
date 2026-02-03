"""Tests for flash_attention.metal implementation."""

from __future__ import annotations

import ctypes
import math
import struct
from pathlib import Path

import numpy as np
import pytest

# Paths to Metal shaders
_SHADER_PATH = Path(__file__).parent.parent / "src" / "flash_attention.metal"

# Kernel configuration constants (must match Metal shader)
_TILE_KV = 32
_ROWS_PER_TG = 4
_THREADS_PER_TG = 128


def _check_metal_available() -> bool:
    """Check if Metal API is available via PyObjC."""
    try:
        import Metal  # noqa: F401
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
    """NumPy reference implementation of scaled dot-product attention."""
    batch, heads_q, seq_q, head_dim = Q.shape
    _, heads_kv, seq_k, _ = K.shape

    # GQA: expand K/V heads to match Q heads
    if heads_kv < heads_q:
        repeat = heads_q // heads_kv
        K = np.repeat(K, repeat, axis=1)
        V = np.repeat(V, repeat, axis=1)

    # Compute Q @ K^T * scale
    scores = np.einsum("bhqd,bhkd->bhqk", Q.astype(np.float32), K.astype(np.float32))
    scores = scores * scale

    # Apply causal mask if needed
    if is_causal:
        q_idx = np.arange(seq_q)[:, None]
        k_idx = np.arange(seq_k)[None, :]
        mask = np.where(k_idx > q_idx, -np.inf, 0.0).astype(np.float32)
        scores = scores + mask

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_max = np.where(np.isinf(scores_max), 0.0, scores_max)
    scores = scores - scores_max
    exp_scores = np.exp(scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)
    softmax = exp_scores / sum_exp

    output = np.einsum("bhqk,bhkd->bhqd", softmax, V.astype(np.float32))
    return output.astype(np.float16)


# ==============================================================================
# Metal Helpers
# ==============================================================================

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
    options = Metal.MTLCompileOptions.new()
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_0)
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err is not None:
        pytest.fail(f"Metal compile error: {err}")

    return device, library

def _create_pipeline(device, library, kernel_name: str):
    """Create a compute pipeline."""
    func = library.newFunctionWithName_(kernel_name)
    if func is None:
        raise ValueError(f"Kernel {kernel_name} not found")

    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    if err is not None:
        raise RuntimeError(f"Pipeline error for {kernel_name}: {err}")

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
# Runners
# ==============================================================================

def run_flash_attention(
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

    device, library = _compile_shader(_SHADER_PATH)

    kernel_name = "flash_attention_causal" if is_causal else "flash_attention"
    pipeline = _create_pipeline(device, library, kernel_name)

    # Prepare buffers
    buf_Q = _create_buffer(device, Q.astype(np.float16))
    buf_K = _create_buffer(device, K.astype(np.float16))
    buf_V = _create_buffer(device, V.astype(np.float16))

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


def run_flash_attention_gqa(
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

    device, library = _compile_shader(_SHADER_PATH)
    pipeline = _create_pipeline(device, library, "flash_attention_gqa")

    buf_Q = _create_buffer(device, Q.astype(np.float16))
    buf_K = _create_buffer(device, K.astype(np.float16))
    buf_V = _create_buffer(device, V.astype(np.float16))

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
# Tests
# ==============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def generate_qkv(rng, batch, heads_q, heads_kv, seq_q, seq_k, head_dim):
    Q = rng.standard_normal((batch, heads_q, seq_q, head_dim)).astype(np.float16)
    K = rng.standard_normal((batch, heads_kv, seq_k, head_dim)).astype(np.float16)
    V = rng.standard_normal((batch, heads_kv, seq_k, head_dim)).astype(np.float16)
    return Q, K, V

@pytest.mark.skipif(not _check_metal_available(), reason="Metal not available")
class TestFlashAttentionMetal:

    @pytest.mark.parametrize("is_causal", [False, True])
    def test_accuracy(self, rng, is_causal):
        batch, heads, seq, head_dim = 1, 4, 128, 64
        scale = 1.0 / math.sqrt(head_dim)

        Q, K, V = generate_qkv(rng, batch, heads, heads, seq, seq, head_dim)

        ref = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=is_causal)
        out = run_flash_attention(Q, K, V, scale, is_causal=is_causal)

        np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("heads_q, heads_kv", [(8, 2), (8, 4)])
    def test_gqa_accuracy(self, rng, heads_q, heads_kv):
        batch, seq, head_dim = 1, 128, 64
        scale = 1.0 / math.sqrt(head_dim)
        is_causal = True

        Q, K, V = generate_qkv(rng, batch, heads_q, heads_kv, seq, seq, head_dim)

        ref = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=is_causal, num_kv_heads=heads_kv)
        out = run_flash_attention_gqa(Q, K, V, scale, is_causal=is_causal)

        np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)

    def test_decode_optimization(self, rng):
        # seq_q=1 triggers the GEMV optimization path (if implemented in kernel logic, usually implicit by seq_q=1)
        # Note: kernel logic handles this by loading Q into registers
        batch, heads, seq_k, head_dim = 1, 4, 128, 64
        seq_q = 1
        scale = 1.0 / math.sqrt(head_dim)

        Q, K, V = generate_qkv(rng, batch, heads, heads, seq_q, seq_k, head_dim)

        ref = ref_scaled_dot_product_attention(Q, K, V, scale, is_causal=False)
        out = run_flash_attention(Q, K, V, scale, is_causal=False)

        np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)
