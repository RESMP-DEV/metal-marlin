"""
Fused scaled dot-product attention with configurable Metal backend selection.

Backend selection is controlled by ``METAL_MARLIN_ATTENTION_BACKEND``:
    - ``v3`` (default): try Flash Attention v3, then v2, then plain SDPA
    - ``v2``: try Flash Attention v2, then plain SDPA
    - ``plain``: use PyTorch scaled_dot_product_attention directly
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import HAS_MPS, HAS_MPSGRAPH, HAS_PYOBJC_METAL, HAS_TORCH, Metal, torch
from ._compat import MPSGraph as MPSG
from .flash_attention_v2 import flash_attention_v2
from .metal_dispatch import mps_tensor_to_metal_buffer

# Module-level logger
_logger = logging.getLogger(__name__)

# Singleton to track which backend was selected at init time (once, not per-call)
_SELECTED_ATTENTION_BACKEND: str | None = None
_ATTENTION_BACKEND_LOGGED: bool = False
_ATTENTION_BACKEND_ENV = "METAL_MARLIN_ATTENTION_BACKEND"
_VALID_ATTENTION_BACKENDS = ("v3", "v2", "plain")


def _get_requested_attention_backend() -> str:
    """Return the requested attention backend, normalized from the environment."""
    backend = os.environ.get(_ATTENTION_BACKEND_ENV, "v3").strip().lower()
    if backend in _VALID_ATTENTION_BACKENDS:
        return backend

    _logger.warning(
        "[fused_attention_mps] Ignoring unsupported %s=%r; expected one of %s",
        _ATTENTION_BACKEND_ENV,
        backend,
        ", ".join(_VALID_ATTENTION_BACKENDS),
    )
    return "v3"


def _attention_backend_candidates() -> tuple[str, ...]:
    """Return the fallback chain for the requested attention backend."""
    requested = _get_requested_attention_backend()
    if requested == "plain":
        return ("plain",)
    if requested == "v2":
        return ("v2", "plain")
    return ("v3", "v2", "plain")


def _probe_v3_backend() -> bool:
    """Check whether Flash Attention v3 can be dispatched."""
    if not (HAS_TORCH and torch is not None and HAS_MPS):
        return False

    try:
        from .flash_attn_mla import flash_attention_v3_mla

        test_q = torch.zeros((1, 1, 4, 64), dtype=torch.float16, device="mps")
        test_k = torch.zeros((1, 1, 4, 64), dtype=torch.float16, device="mps")
        test_v = torch.zeros((1, 1, 4, 64), dtype=torch.float16, device="mps")
        flash_attention_v3_mla(test_q, test_k, test_v, scale=0.125, causal=False)
        return True
    except Exception:
        return False


def _probe_v2_backend() -> bool:
    """Check whether Flash Attention v2 can be dispatched."""
    if not (HAS_TORCH and torch is not None and HAS_MPS):
        return False

    try:
        test_q = torch.zeros((1, 1, 4, 16), dtype=torch.float16, device="mps")
        test_k = torch.zeros((1, 1, 4, 16), dtype=torch.float16, device="mps")
        test_v = torch.zeros((1, 1, 4, 16), dtype=torch.float16, device="mps")
        flash_attention_v2(test_q, test_k, test_v, scale=0.25, causal=False)
        return True
    except Exception:
        return False


def _probe_plain_backend() -> bool:
    """Check whether the plain PyTorch SDPA fallback is available."""
    return HAS_TORCH and torch is not None


def _run_v3_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None,
    causal: bool,
) -> torch.Tensor:
    """Run Flash Attention v3."""
    from .flash_attn_mla import flash_attention_v3_mla

    return flash_attention_v3_mla(q, k, v, scale=scale, causal=causal)


def _run_v2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None,
    causal: bool,
) -> torch.Tensor:
    """Run Flash Attention v2."""
    return flash_attention_v2(q, k, v, scale=scale, causal=causal)


def _run_plain_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    scale: float | None,
    causal: bool,
) -> torch.Tensor:
    """Run the plain PyTorch SDPA fallback."""
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for plain attention fallback.")

    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )


def _get_attention_backend() -> str:
    """Determine which attention backend will be used at runtime.

    Returns:
        Backend name: "v3", "v2", "plain", or "unavailable"
    """
    for backend in _attention_backend_candidates():
        if backend == "v3" and _probe_v3_backend():
            return "v3"
        if backend == "v2" and _probe_v2_backend():
            return "v2"
        if backend == "plain" and _probe_plain_backend():
            return "plain"

    return "unavailable"


def _log_attention_backend_once() -> str:
    """Log which attention backend is selected (once at init time, not per-call).

    Returns:
        The selected backend name.
    """
    global _SELECTED_ATTENTION_BACKEND, _ATTENTION_BACKEND_LOGGED

    if _ATTENTION_BACKEND_LOGGED:
        return _SELECTED_ATTENTION_BACKEND or "unknown"

    # Only log once
    _ATTENTION_BACKEND_LOGGED = True

    backend = _get_attention_backend()
    _SELECTED_ATTENTION_BACKEND = backend

    backend_descriptions = {
        "v3": "Flash Attention v3 (Metal MLA kernels)",
        "v2": "Flash Attention v2 (Metal kernels)",
        "plain": "PyTorch scaled_dot_product_attention",
        "unavailable": "No available attention backend",
    }

    desc = backend_descriptions.get(backend, backend)
    msg = f"[fused_attention_mps] Attention backend selected: {desc} ({backend})"
    _logger.info(msg)
    # Also print to stderr so the selection is visible even when logging is not configured
    print(msg, file=sys.stderr)

    return backend


# Log the selected backend at module import time (once, not per-call)
_log_attention_backend_once()


def _require_mpsgraph_attention() -> None:
    if not HAS_TORCH or torch is None:
        raise RuntimeError(
            "MPSGraph attention requires PyTorch. Install with: pip install torch")
    if not HAS_MPS:
        raise RuntimeError(
            "MPSGraph attention requires PyTorch MPS backend (Apple Silicon).")
    if not HAS_PYOBJC_METAL:
        raise RuntimeError(
            "MPSGraph attention requires PyObjC Metal. Install with:\n"
            "  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders"
        )
    if not HAS_MPSGRAPH or MPSG is None:
        raise RuntimeError(
            "MPSGraph attention requires MetalPerformanceShadersGraph. Install with:\n"
            "  pip install pyobjc-framework-MetalPerformanceShadersGraph"
        )


def _torch_dtype_to_mps(dtype: torch.dtype) -> Any:
    if dtype == torch.float16:
        return MPSG.MPSDataTypeFloat16
    if dtype == torch.float32:
        return MPSG.MPSDataTypeFloat32
    raise ValueError(f"Unsupported dtype for MPSGraph SDPA: {dtype}")


def _np_dtype_from_torch(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    raise ValueError(f"Unsupported dtype for MPSGraph SDPA: {dtype}")


def _scaled_dot_product_attention_op(
    graph: Any,
    query: Any,
    key: Any,
    value: Any,
    mask: Any | None,
    scale: float,
) -> Any:
    if hasattr(graph, "scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_"):
        return graph.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_(
            query,
            key,
            value,
            mask,
            scale,
        )
    if hasattr(graph, "scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name_"):
        return graph.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name_(
            query,
            key,
            value,
            mask,
            scale,
            None,
        )
    raise RuntimeError(
        "MPSGraph scaledDotProductAttention selector not found in PyObjC bindings.")


@dataclass(frozen=True)
class _GraphCacheKey:
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    mask_shape: tuple[int, ...] | None
    dtype: torch.dtype
    scale: float


@dataclass
class _GraphCacheEntry:
    graph: Any
    device: Any
    command_queue: Any
    q_placeholder: Any
    k_placeholder: Any
    v_placeholder: Any
    mask_placeholder: Any | None
    output_tensor: Any


_GRAPH_CACHE: dict[_GraphCacheKey, _GraphCacheEntry] = {}


def _get_graph_entry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> _GraphCacheEntry:
    key = _GraphCacheKey(
        q_shape=tuple(q.shape),
        k_shape=tuple(k.shape),
        v_shape=tuple(v.shape),
        mask_shape=tuple(mask.shape) if mask is not None else None,
        dtype=q.dtype,
        scale=scale,
    )
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    graph = MPSG.MPSGraph.alloc().init()
    mps_dtype = _torch_dtype_to_mps(q.dtype)

    q_placeholder = graph.placeholderWithShape_dataType_name_(
        list(q.shape), mps_dtype, None)
    k_placeholder = graph.placeholderWithShape_dataType_name_(
        list(k.shape), mps_dtype, None)
    v_placeholder = graph.placeholderWithShape_dataType_name_(
        list(v.shape), mps_dtype, None)
    mask_placeholder = None
    if mask is not None:
        mask_placeholder = graph.placeholderWithShape_dataType_name_(
            list(mask.shape), mps_dtype, None)

    output_tensor = _scaled_dot_product_attention_op(
        graph,
        q_placeholder,
        k_placeholder,
        v_placeholder,
        mask_placeholder,
        scale,
    )

    if Metal is None:
        raise RuntimeError(
            "Metal framework not available for MPSGraph attention.")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device available for MPSGraph attention.")
    command_queue = device.newCommandQueue()
    if command_queue is None:
        raise RuntimeError(
            "Failed to create Metal command queue for MPSGraph attention.")

    entry = _GraphCacheEntry(
        graph=graph,
        device=device,
        command_queue=command_queue,
        q_placeholder=q_placeholder,
        k_placeholder=k_placeholder,
        v_placeholder=v_placeholder,
        mask_placeholder=mask_placeholder,
        output_tensor=output_tensor,
    )
    _GRAPH_CACHE[key] = entry
    return entry


def _ensure_mps_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_mps:
        tensor = tensor.to(device="mps")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _build_causal_mask(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.triu(
        torch.full((seq_q, seq_k), float("-inf"), dtype=dtype, device="mps"),
        diagonal=1,
    )
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.expand(batch, heads, seq_q, seq_k)


def fused_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Fused SDPA using MPSGraph scaledDotProductAttention.

    Args:
        q: Query tensor [batch, heads, seq_q, head_dim], MPS device preferred.
        k: Key tensor [batch, heads, seq_k, head_dim].
        v: Value tensor [batch, heads, seq_k, head_dim].
        mask: Optional additive mask broadcastable to [batch, heads, seq_q, seq_k].
        scale: Optional scale factor (defaults to 1/sqrt(head_dim)).
        causal: Whether to apply causal masking when mask is not provided.
    """
    _require_mpsgraph_attention()

    q = _ensure_mps_tensor(q)
    k = _ensure_mps_tensor(k)
    v = _ensure_mps_tensor(v)

    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"Unsupported dtype for MPSGraph SDPA: {q.dtype}")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if mask is None and causal:
        batch, heads, seq_q, _ = q.shape
        seq_k = k.shape[2]
        mask = _build_causal_mask(batch, heads, seq_q, seq_k, q.dtype)
    if mask is not None:
        mask = _ensure_mps_tensor(mask.to(dtype=q.dtype))

    entry = _get_graph_entry(q, k, v, mask, scale)
    mps_dtype = _torch_dtype_to_mps(q.dtype)

    feeds = {
        entry.q_placeholder: MPSG.MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(
            mps_tensor_to_metal_buffer(q, entry.device),
            list(q.shape),
            mps_dtype,
        ),
        entry.k_placeholder: MPSG.MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(
            mps_tensor_to_metal_buffer(k, entry.device),
            list(k.shape),
            mps_dtype,
        ),
        entry.v_placeholder: MPSG.MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(
            mps_tensor_to_metal_buffer(v, entry.device),
            list(v.shape),
            mps_dtype,
        ),
    }
    if mask is not None and entry.mask_placeholder is not None:
        feeds[entry.mask_placeholder] = MPSG.MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(
            mps_tensor_to_metal_buffer(mask, entry.device),
            list(mask.shape),
            mps_dtype,
        )

    result = entry.graph.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(
        entry.command_queue,
        feeds,
        [entry.output_tensor],
        None,
    )
    output_data = result[entry.output_tensor]

    # Extract data via MPSNDArray.readBytes_strideBytes_
    # MPSGraphTensorData doesn't expose MTLBuffer directly, but mpsndarray() works
    ndarray = output_data.mpsndarray()
    if ndarray is None:
        raise RuntimeError("Failed to get MPSNDArray from MPSGraphTensorData")

    desc = ndarray.descriptor()
    dims = desc.numberOfDimensions()

    # MPS reports shape in reversed order (column-major convention)
    mps_shape = [desc.lengthOfDimension_(i) for i in range(dims)]

    # Calculate element size and total data size
    element_size = q.element_size()
    total_elements = 1
    for s in mps_shape:
        total_elements *= s
    data_size = total_elements * element_size

    # Create ctypes buffer to receive data
    output_buf = (ctypes.c_char * data_size)()

    # Calculate row-major strides for MPS shape (in bytes)
    strides = [element_size]
    for i in range(dims - 1):
        strides.append(strides[-1] * mps_shape[i])

    # Read data from MPSNDArray
    ndarray.readBytes_strideBytes_(output_buf, strides)

    # Convert to numpy, reshape to MPS order, then transpose to original order
    np_data = np.frombuffer(bytes(output_buf), dtype=_np_dtype_from_torch(q.dtype))
    np_mps = np_data.reshape(mps_shape)

    # Transpose from MPS order to original order
    # MPS reverses dimensions, so transpose reverses back
    transpose_axes = tuple(range(dims - 1, -1, -1))
    np_out = np_mps.transpose(transpose_axes).copy()

    return torch.from_numpy(np_out).to(device="mps", dtype=q.dtype)


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Dispatch attention according to the configured v3/v2/plain fallback chain."""
    debug = os.environ.get("FUSED_ATTN_DEBUG", "0") == "1"
    last_error: Exception | None = None

    for backend in _attention_backend_candidates():
        if backend in {"v3", "v2"} and mask is not None:
            if debug:
                print(
                    f"[FUSED_ATTN] Skipping flash_attention_{backend} because attn_mask is set"
                )
            continue

        label = (
            "flash_attention_v3"
            if backend == "v3"
            else "flash_attention_v2"
            if backend == "v2"
            else "F.scaled_dot_product_attention"
        )

        try:
            if backend == "v3":
                result = _run_v3_attention(q, k, v, scale=scale, causal=causal)
            elif backend == "v2":
                result = _run_v2_attention(q, k, v, scale=scale, causal=causal)
            else:
                result = _run_plain_attention(
                    q, k, v, mask=mask, scale=scale, causal=causal
                )

            if debug:
                print(f"[FUSED_ATTN] Using {label}")
            return result
        except Exception as exc:
            last_error = exc
            if debug:
                print(f"[FUSED_ATTN] {label} failed: {exc}")

    if last_error is not None:
        raise RuntimeError("No available backend for fused attention.") from last_error
    raise RuntimeError("No compatible attention backend for the provided inputs.")


def get_selected_attention_backend() -> str:
    """Get the name of the attention backend selected at init time.

    This function returns the backend that was selected and logged when the
    module was first imported. The backend is determined by testing availability
    in the configured fallback order (v3 -> v2 -> plain by default).

    Returns:
        Backend name: "v3", "v2", "plain", or "unavailable"
    """
    global _SELECTED_ATTENTION_BACKEND
    if _SELECTED_ATTENTION_BACKEND is None:
        # Fallback in case module was imported without running init
        return _get_attention_backend()
    return _SELECTED_ATTENTION_BACKEND


__all__ = [
    "fused_attention",
    "fused_scaled_dot_product_attention",
    "get_selected_attention_backend",
]
