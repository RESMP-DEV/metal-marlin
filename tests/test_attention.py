"""Flash attention accuracy tests against PyTorch reference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

_METAL_SHADER_PATH = Path(__file__).parent.parent / "src" / "flash_attention.metal"
_ROWS_PER_TG = 4
_THREADS_PER_TG = 128


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except (ImportError, ModuleNotFoundError):
        pytest.skip("torch not available")
    return torch, F


def _require_mlx():
    try:
        import mlx.core as mx
        return mx
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mlx not available")


def _load_metal_source() -> str:
    if not _METAL_SHADER_PATH.exists():
        pytest.skip(f"Metal shader not found: {_METAL_SHADER_PATH}")
    return _METAL_SHADER_PATH.read_text()


_KERNEL_CACHE: dict[str, object] = {}


def _get_flash_kernel(mx, name: str) -> object:
    if name in _KERNEL_CACHE:
        return _KERNEL_CACHE[name]
    source = _load_metal_source()
    _KERNEL_CACHE[name] = mx.fast.metal_kernel(
        name=name,
        input_names=[
            "Q",
            "K",
            "V",
            "batch",
            "num_heads_q",
            "num_heads_k",
            "seq_q",
            "seq_k",
            "head_dim",
            "scale",
        ],
        output_names=["O"],
        source=source,
    )
    return _KERNEL_CACHE[name]


def _run_flash_attention(mx, kernel_name: str, Q: np.ndarray, K: np.ndarray,
                         V: np.ndarray, scale: float, num_heads_k: int) -> np.ndarray:
    batch, num_heads_q, seq_q, head_dim = Q.shape
    seq_k = K.shape[2]

    kernel = _get_flash_kernel(mx, kernel_name)

    grid_x = num_heads_q
    grid_y = (seq_q + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    grid_z = batch

    outputs = kernel(
        inputs=[
            mx.array(Q, dtype=mx.float16),
            mx.array(K, dtype=mx.float16),
            mx.array(V, dtype=mx.float16),
            mx.array(np.uint32(batch)),
            mx.array(np.uint32(num_heads_q)),
            mx.array(np.uint32(num_heads_k)),
            mx.array(np.uint32(seq_q)),
            mx.array(np.uint32(seq_k)),
            mx.array(np.uint32(head_dim)),
            mx.array(np.float32(scale)),
        ],
        output_shapes=[(batch, num_heads_q, seq_q, head_dim)],
        output_dtypes=[mx.float16],
        grid=(grid_x, grid_y, grid_z),
        threadgroup=(_THREADS_PER_TG, 1, 1),
        init_value=0.0,
    )
    mx.eval(outputs[0])
    return np.array(outputs[0])


@pytest.mark.parametrize("batch,heads,seq_q,seq_k,head_dim", [
    (1, 32, 1, 512, 128),      # Single token decode
    (1, 32, 512, 512, 128),    # Prefill
    (8, 32, 1, 2048, 128),     # Batched decode, long context
    (1, 8, 1, 512, 128),       # GQA (fewer heads)
])
def test_flash_attention_accuracy(batch, heads, seq_q, seq_k, head_dim) -> None:
    torch, F = _require_torch()
    mx = _require_mlx()

    torch.manual_seed(0)
    Q = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float16)

    if heads == 8:
        num_heads_k = 2
    else:
        num_heads_k = heads

    K = torch.randn(batch, num_heads_k, seq_k, head_dim, dtype=torch.float16)
    V = torch.randn(batch, num_heads_k, seq_k, head_dim, dtype=torch.float16)

    scale = 1.0 / math.sqrt(head_dim)

    if num_heads_k != heads:
        repeat_factor = heads // num_heads_k
        K_ref = K.repeat_interleave(repeat_factor, dim=1)
        V_ref = V.repeat_interleave(repeat_factor, dim=1)
    else:
        K_ref = K
        V_ref = V

    ref = F.scaled_dot_product_attention(
        Q.float(), K_ref.float(), V_ref.float(), scale=scale
    ).to(dtype=torch.float16)

    result_np = _run_flash_attention(
        mx, "flash_attention", Q.cpu().numpy(), K.cpu().numpy(), V.cpu().numpy(),
        scale, num_heads_k
    )
    result = torch.from_numpy(result_np)

    assert torch.allclose(result, ref, rtol=1e-2, atol=1e-3)


def test_attention_causal_mask() -> None:
    torch, F = _require_torch()
    mx = _require_mlx()

    batch = 1
    heads = 4
    seq = 64
    head_dim = 64
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(123)
    Q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16)
    K = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16)
    V = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16)

    ref = F.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), scale=scale, is_causal=True
    ).to(dtype=torch.float16)

    result_np = _run_flash_attention(
        mx, "flash_attention_causal", Q.cpu().numpy(), K.cpu().numpy(),
        V.cpu().numpy(), scale, heads
    )
    result = torch.from_numpy(result_np)

    assert torch.allclose(result, ref, rtol=1e-2, atol=1e-3)


def test_attention_fp4_kv() -> None:
    mx = _require_mlx()
    source = _load_metal_source()
    if "flash_attention_kv_fp4" not in source:
        pytest.skip("flash_attention_kv_fp4 kernel not available")
    pytest.skip("FP4 KV attention test not implemented yet")
