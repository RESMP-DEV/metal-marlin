"""Qwen3.6-27B fused Metal kernel wrappers.

The fused path is intentionally opt-in through
``METAL_MARLIN_QWEN36_27B_MEGAKERNEL=1``.  These wrappers validate the dense
27B contract and dispatch kernels from the normal Metal Marlin metallib; they
do not load or depend on the standalone qwenmetal C API.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .. import launch_tracing
from ..metal_dispatch import (
    Metal,
    MetalKernelLibrary,
    dispatch_kernel,
    mps_tensor_to_metal_buffer,
    require_mps,
)
from ..metallib_loader import get_kernel_from_metallib, get_staleness_details
from ..qwen36_27b_artifact import (
    Int4TensorArtifact,
    ManifestCoverageKind,
    TensorRole,
    artifact_coverage,
    expected_tensor_shape,
    read_manifest,
)
from ..qwen36_27b_profile import (
    FEATURE_FLAG,
    QWEN36_27B_PROFILE,
    Qwen36ModelProfile,
    is_qwen36_27b_config,
    validate_supported_profile,
)

logger = logging.getLogger(__name__)
_qwen36_kernel_library: MetalKernelLibrary | None = None

KERNEL_QKVB = "qwen36_27b_int4_qkvb"
KERNEL_LINEAR_AZ = "qwen36_27b_int4_linear_az"
KERNEL_DELTANET_UPDATE = "qwen36_27b_deltanet_update"
KERNEL_DELTANET_INTERVAL = "qwen36_27b_deltanet_interval4"
KERNEL_LINEAR_OUT = "qwen36_27b_linear_o_residual"
KERNEL_DENSE_GATE_UP = "qwen36_27b_dense_gate_up_silu"
KERNEL_DENSE_DOWN = "qwen36_27b_dense_down_residual"
KERNEL_RMSNORM = "qwen36_27b_rmsnorm_hidden"
KERNEL_LINEAR_GATED_NORM = "qwen36_27b_linear_rmsnorm_gated"
KERNEL_ATTENTION_QKV = "qwen36_27b_int4_attention_qkv"
KERNEL_ATTENTION_CACHE = "qwen36_27b_attention_cache_write"
KERNEL_ATTENTION_DECODE = "qwen36_27b_attention_decode"
KERNEL_ATTENTION_OUT = "qwen36_27b_attention_o_residual"
KERNEL_LM_HEAD = "qwen36_27b_lm_head_logits"
KERNEL_ARGMAX = "qwen36_27b_argmax_f16"

REQUIRED_KERNELS = (
    KERNEL_QKVB,
    KERNEL_LINEAR_AZ,
    KERNEL_DELTANET_UPDATE,
    KERNEL_DELTANET_INTERVAL,
    KERNEL_LINEAR_OUT,
    KERNEL_DENSE_GATE_UP,
    KERNEL_DENSE_DOWN,
    KERNEL_RMSNORM,
    KERNEL_LINEAR_GATED_NORM,
    KERNEL_ATTENTION_QKV,
    KERNEL_ATTENTION_CACHE,
    KERNEL_ATTENTION_DECODE,
    KERNEL_ATTENTION_OUT,
    KERNEL_LM_HEAD,
    KERNEL_ARGMAX,
)

_INT4_DTYPES = {torch.int32}
if hasattr(torch, "uint32"):
    _INT4_DTYPES.add(torch.uint32)

QWEN36_PROJECTION_THREADS_PER_GROUP = 64
QWEN36_INT4_GEMV_THREADS_PER_GROUP = 64
QWEN36_DELTANET_THREADS_PER_GROUP = 128


def get_qwen36_kernel_library() -> MetalKernelLibrary:
    """Return a Qwen3.6-27B library that dispatches from the tracked metallib."""

    global _qwen36_kernel_library
    if _qwen36_kernel_library is None:
        _qwen36_kernel_library = MetalKernelLibrary()
    return _qwen36_kernel_library


@dataclass(frozen=True)
class RuntimeDecision:
    enabled: bool
    reason: str
    coverage_kind: ManifestCoverageKind | None = None
    coverage_layers: int | None = None
    coverage_tensors: int | None = None


@dataclass(frozen=True)
class PackedInt4Matrix:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    in_features: int
    out_features: int
    group_size: int = QWEN36_27B_PROFILE.group_size
    role: TensorRole | None = None


def _read_tensor_file(path: Path, dtype: torch.dtype, expected_elements: int) -> torch.Tensor:
    data = path.read_bytes()
    item_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = expected_elements * item_size
    if len(data) != expected_bytes:
        raise ValueError(f"{path} has {len(data)} bytes, expected {expected_bytes}")
    return torch.frombuffer(bytearray(data), dtype=dtype).clone()


def load_packed_int4_matrix(
    artifact: Int4TensorArtifact,
    base_dir: str | Path,
    *,
    device: str | torch.device | None = None,
) -> PackedInt4Matrix:
    """Load one manifest tensor into a wrapper-ready packed int4 matrix."""
    artifact.validate()
    root = Path(base_dir)
    groups = (artifact.in_features + artifact.group_size - 1) // artifact.group_size
    packed_k = artifact.in_features // 8
    qweight = _read_tensor_file(
        root / artifact.qweight,
        torch.uint32,
        packed_k * artifact.out_features,
    ).reshape(packed_k, artifact.out_features)
    scales = _read_tensor_file(
        root / artifact.scales,
        torch.float16,
        groups * artifact.out_features,
    ).reshape(groups, artifact.out_features)
    zeros = _read_tensor_file(
        root / artifact.zeros,
        torch.float16,
        groups * artifact.out_features,
    ).reshape(groups, artifact.out_features)
    if device is not None:
        qweight = qweight.to(device=device)
        scales = scales.to(device=device)
        zeros = zeros.to(device=device)
    return PackedInt4Matrix(
        qweight=qweight,
        scales=scales,
        zeros=zeros,
        in_features=artifact.in_features,
        out_features=artifact.out_features,
        group_size=artifact.group_size,
        role=artifact.role,
    )


@dataclass(frozen=True)
class QkvbProjectionOutput:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class FullAttentionProjectionOutput:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class LinearAzProjectionOutput:
    a: torch.Tensor
    z: torch.Tensor


def feature_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(FEATURE_FLAG) == "1"


def decide_runtime_path(
    config: dict[str, Any] | None,
    *,
    env: Mapping[str, str] | None = None,
) -> RuntimeDecision:
    if not feature_enabled(env):
        return RuntimeDecision(False, f"{FEATURE_FLAG} is not set")
    if config is None:
        return RuntimeDecision(False, "missing config")
    if not is_qwen36_27b_config(config):
        return RuntimeDecision(False, "config is not dense Qwen3.6-27B")
    return RuntimeDecision(True, "dense Qwen3.6-27B fused path selected")


def decide_fused_artifact_path(
    config: dict[str, Any] | None,
    artifact_manifest_path: str | Path | None,
    *,
    env: Mapping[str, str] | None = None,
    require_fresh_metallib: bool = True,
    require_kernel_symbols: bool = True,
    library: Any | None = None,
) -> RuntimeDecision:
    """Return whether the artifact-backed Qwen3.6-27B fused path is usable."""

    base_decision = decide_runtime_path(config, env=env)
    if not base_decision.enabled:
        return base_decision
    if artifact_manifest_path is None:
        return RuntimeDecision(False, "missing artifact manifest")

    path = Path(artifact_manifest_path)
    try:
        manifest = read_manifest(path)
        coverage = artifact_coverage(manifest)
    except (OSError, ValueError, TypeError) as exc:
        return RuntimeDecision(False, f"invalid artifact manifest: {exc}")

    if require_fresh_metallib:
        staleness = get_staleness_details()
        if staleness["is_stale"]:
            return RuntimeDecision(False, f"metallib is stale: {staleness['reason']}")

    if require_kernel_symbols:
        missing = sorted(missing_kernel_symbols(library=library))
        if missing:
            return RuntimeDecision(False, f"missing Qwen3.6-27B kernels: {missing}")

    return RuntimeDecision(
        True,
        f"dense Qwen3.6-27B fused artifact path selected ({coverage.kind})",
        coverage_kind=coverage.kind,
        coverage_layers=coverage.layers,
        coverage_tensors=coverage.tensors,
    )


def available_kernel_symbols(library: Any | None = None) -> set[str]:
    found: set[str] = set()
    for name in REQUIRED_KERNELS:
        if get_kernel_from_metallib(name, library=library) is not None:
            found.add(name)
    return found


def missing_kernel_symbols(library: Any | None = None) -> set[str]:
    return set(REQUIRED_KERNELS) - available_kernel_symbols(library=library)


def _ensure_mps_half_vector(tensor: torch.Tensor, name: str, expected: int) -> torch.Tensor:
    if tensor.numel() != expected:
        raise ValueError(f"{name} must have {expected} elements, got {tensor.numel()}")
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"{name} must be float16 or bfloat16, got {tensor.dtype}")
    if not tensor.is_mps:
        raise ValueError(f"{name} must be on MPS for Metal dispatch")
    return tensor.reshape(expected).contiguous().to(dtype=torch.float16)


def _ensure_mutable_mps_half_vector(
    tensor: torch.Tensor,
    name: str,
    min_elements: int,
    *,
    exact: bool = True,
) -> torch.Tensor:
    if exact and tensor.numel() != min_elements:
        raise ValueError(f"{name} must have {min_elements} elements, got {tensor.numel()}")
    if not exact and tensor.numel() < min_elements:
        raise ValueError(f"{name} must have at least {min_elements} elements, got {tensor.numel()}")
    if tensor.dtype != torch.float16:
        raise TypeError(f"{name} must be float16 because the kernel mutates it, got {tensor.dtype}")
    if not tensor.is_mps:
        raise ValueError(f"{name} must be on MPS because the kernel mutates it")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous because the kernel mutates it")
    return tensor.view(-1)


def _ensure_matrix(matrix: PackedInt4Matrix, role: TensorRole) -> PackedInt4Matrix:
    expected_in, expected_out = expected_tensor_shape(role)
    if (matrix.in_features, matrix.out_features) != (expected_in, expected_out):
        raise ValueError(
            f"{role} expected ({expected_in}, {expected_out}), "
            f"got ({matrix.in_features}, {matrix.out_features})"
        )
    if matrix.group_size != QWEN36_27B_PROFILE.group_size:
        raise ValueError(f"{role} requires group_size={QWEN36_27B_PROFILE.group_size}")
    if matrix.qweight.dtype not in _INT4_DTYPES:
        raise TypeError(f"{role}.qweight must be int32/uint32, got {matrix.qweight.dtype}")
    if matrix.scales.dtype != torch.float16 or matrix.zeros.dtype != torch.float16:
        raise TypeError(f"{role}.scales and zeros must be float16")
    packed_rows = matrix.in_features // 8
    groups = (matrix.in_features + matrix.group_size - 1) // matrix.group_size
    if tuple(matrix.qweight.shape) != (packed_rows, matrix.out_features):
        raise ValueError(f"{role}.qweight must be {(packed_rows, matrix.out_features)}")
    if tuple(matrix.scales.shape) != (groups, matrix.out_features):
        raise ValueError(f"{role}.scales must be {(groups, matrix.out_features)}")
    if tuple(matrix.zeros.shape) != (groups, matrix.out_features):
        raise ValueError(f"{role}.zeros must be {(groups, matrix.out_features)}")
    return PackedInt4Matrix(
        qweight=matrix.qweight.contiguous(),
        scales=matrix.scales.contiguous(),
        zeros=matrix.zeros.contiguous(),
        in_features=matrix.in_features,
        out_features=matrix.out_features,
        group_size=matrix.group_size,
        role=role,
    )


def _buffer(tensor: torch.Tensor, lib: MetalKernelLibrary, *, copy_back: bool = False) -> Any:
    if not tensor.is_mps:
        tensor = tensor.to("mps")
    return mps_tensor_to_metal_buffer(tensor.contiguous(), lib.device, copy_back=copy_back)


def _mutable_buffer(tensor: torch.Tensor, lib: MetalKernelLibrary) -> Any:
    if not tensor.is_mps:
        tensor = tensor.to("mps")
    return mps_tensor_to_metal_buffer(
        tensor.contiguous(),
        lib.device,
        copy_back=True,
        initialize_copy_back=True,
    )


def _params(values: list[int], lib: MetalKernelLibrary) -> Any:
    tensor = torch.tensor(values, dtype=torch.int32, device="mps")
    return mps_tensor_to_metal_buffer(tensor, lib.device)


def _shared_buffer(byte_size: int, lib: MetalKernelLibrary) -> Any:
    buffer = lib.device.newBufferWithLength_options_(
        byte_size,
        Metal.MTLResourceStorageModeShared,
    )
    if buffer is None:
        raise RuntimeError(f"failed to allocate {byte_size} byte Metal buffer")
    return buffer


def dispatch_qkvb_projection(
    hidden: torch.Tensor,
    q: PackedInt4Matrix,
    k: PackedInt4Matrix,
    v: PackedInt4Matrix,
    beta: PackedInt4Matrix,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> QkvbProjectionOutput:
    """Run the fused Q/K/V/Beta int4 projection for one decode token."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    hidden = _ensure_mps_half_vector(hidden, "hidden", profile.hidden_size)
    q = _ensure_matrix(q, "linear_attn_q")
    k = _ensure_matrix(k, "linear_attn_k")
    v = _ensure_matrix(v, "linear_attn_v")
    beta = _ensure_matrix(beta, "linear_attn_beta")

    q_out = torch.empty(profile.delta.q_features, dtype=torch.float16, device=hidden.device)
    k_out = torch.empty(profile.delta.k_features, dtype=torch.float16, device=hidden.device)
    v_out = torch.empty(profile.delta.v_features, dtype=torch.float16, device=hidden.device)
    beta_out = torch.empty(profile.delta.beta_features, dtype=torch.float16, device=hidden.device)
    total_cols = (
        profile.delta.q_features
        + profile.delta.k_features
        + profile.delta.v_features
        + profile.delta.beta_features
    )
    buffers = [
        _buffer(hidden, lib),
        _buffer(q.qweight, lib),
        _buffer(q.scales, lib),
        _buffer(q.zeros, lib),
        _buffer(k.qweight, lib),
        _buffer(k.scales, lib),
        _buffer(k.zeros, lib),
        _buffer(v.qweight, lib),
        _buffer(v.scales, lib),
        _buffer(v.zeros, lib),
        _buffer(beta.qweight, lib),
        _buffer(beta.scales, lib),
        _buffer(beta.zeros, lib),
        _buffer(q_out, lib, copy_back=True),
        _buffer(k_out, lib, copy_back=True),
        _buffer(v_out, lib, copy_back=True),
        _buffer(beta_out, lib, copy_back=True),
        _params([profile.group_size, total_cols], lib),
    ]
    launch_tracing.record_dispatch(KERNEL_QKVB, total_cols=total_cols)
    dispatch_kernel(
        lib,
        KERNEL_QKVB,
        (total_cols, 1, 1),
        (QWEN36_PROJECTION_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return QkvbProjectionOutput(q=q_out, k=k_out, v=v_out, beta=beta_out)


def dispatch_attention_qkv_projection(
    hidden: torch.Tensor,
    q: PackedInt4Matrix,
    k: PackedInt4Matrix,
    v: PackedInt4Matrix,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> FullAttentionProjectionOutput:
    """Run fused full-attention Q/K/V int4 projections for one decode token."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    hidden = _ensure_mps_half_vector(hidden, "hidden", profile.hidden_size)
    q = _ensure_matrix(q, "full_attn_q")
    k = _ensure_matrix(k, "full_attn_k")
    v = _ensure_matrix(v, "full_attn_v")

    q_out = torch.empty(profile.attention.q_features, dtype=torch.float16, device=hidden.device)
    k_out = torch.empty(profile.attention.kv_features, dtype=torch.float16, device=hidden.device)
    v_out = torch.empty(profile.attention.kv_features, dtype=torch.float16, device=hidden.device)
    total_cols = (
        profile.attention.q_features
        + profile.attention.kv_features
        + profile.attention.kv_features
    )
    buffers = [
        _buffer(hidden, lib),
        _buffer(q.qweight, lib),
        _buffer(q.scales, lib),
        _buffer(q.zeros, lib),
        _buffer(k.qweight, lib),
        _buffer(k.scales, lib),
        _buffer(k.zeros, lib),
        _buffer(v.qweight, lib),
        _buffer(v.scales, lib),
        _buffer(v.zeros, lib),
        _buffer(q_out, lib, copy_back=True),
        _buffer(k_out, lib, copy_back=True),
        _buffer(v_out, lib, copy_back=True),
        _params([profile.group_size, total_cols], lib),
    ]
    launch_tracing.record_dispatch(KERNEL_ATTENTION_QKV, total_cols=total_cols)
    dispatch_kernel(
        lib,
        KERNEL_ATTENTION_QKV,
        (total_cols, 1, 1),
        (QWEN36_INT4_GEMV_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return FullAttentionProjectionOutput(q=q_out, k=k_out, v=v_out)


def dispatch_linear_az_projection(
    hidden: torch.Tensor,
    a: PackedInt4Matrix,
    z: PackedInt4Matrix,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> LinearAzProjectionOutput:
    """Run fused linear-attention A/Z int4 projections for one decode token."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    hidden = _ensure_mps_half_vector(hidden, "hidden", profile.hidden_size)
    a = _ensure_matrix(a, "linear_attn_a")
    z = _ensure_matrix(z, "linear_attn_z")

    a_out = torch.empty(profile.delta.beta_features, dtype=torch.float16, device=hidden.device)
    z_out = torch.empty(profile.delta.v_features, dtype=torch.float16, device=hidden.device)
    total_cols = profile.delta.beta_features + profile.delta.v_features
    buffers = [
        _buffer(hidden, lib),
        _buffer(a.qweight, lib),
        _buffer(a.scales, lib),
        _buffer(a.zeros, lib),
        _buffer(z.qweight, lib),
        _buffer(z.scales, lib),
        _buffer(z.zeros, lib),
        _buffer(a_out, lib, copy_back=True),
        _buffer(z_out, lib, copy_back=True),
        _params([profile.group_size, total_cols], lib),
    ]
    launch_tracing.record_dispatch(KERNEL_LINEAR_AZ, total_cols=total_cols)
    dispatch_kernel(
        lib,
        KERNEL_LINEAR_AZ,
        (total_cols, 1, 1),
        (QWEN36_INT4_GEMV_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return LinearAzProjectionOutput(a=a_out, z=z_out)


def dispatch_deltanet_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run one Qwen3.6-27B DeltaNet update/readout."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    q = _ensure_mps_half_vector(q, "q", profile.delta.q_features)
    k = _ensure_mps_half_vector(k, "k", profile.delta.k_features)
    v = _ensure_mps_half_vector(v, "v", profile.delta.v_features)
    beta = _ensure_mps_half_vector(beta, "beta", profile.delta.beta_features)
    state = _ensure_mutable_mps_half_vector(state, "state", profile.delta.state_elements)
    y = torch.empty(profile.delta.v_features, dtype=torch.float16, device=q.device)
    buffers = [
        _buffer(q, lib),
        _buffer(k, lib),
        _buffer(v, lib),
        _buffer(beta, lib),
        _mutable_buffer(state, lib),
        _buffer(y, lib, copy_back=True),
    ]
    launch_tracing.record_dispatch(KERNEL_DELTANET_UPDATE, value_heads=profile.delta.value_heads)
    value_block_cols = 16
    deltanet_blocks = profile.delta.value_heads * (profile.delta.value_dim // value_block_cols)
    dispatch_kernel(
        lib,
        KERNEL_DELTANET_UPDATE,
        (deltanet_blocks, 1, 1),
        (QWEN36_DELTANET_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return y


def dispatch_deltanet_interval4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run the three-linear-layer interval DeltaNet update/readout kernel."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    layers = 3
    q = _ensure_mps_half_vector(q, "q", layers * profile.delta.q_features)
    k = _ensure_mps_half_vector(k, "k", layers * profile.delta.k_features)
    v = _ensure_mps_half_vector(v, "v", layers * profile.delta.v_features)
    beta = _ensure_mps_half_vector(beta, "beta", layers * profile.delta.beta_features)
    state = _ensure_mutable_mps_half_vector(
        state,
        "state",
        layers * profile.delta.state_elements,
    )
    y = torch.empty(layers * profile.delta.v_features, dtype=torch.float16, device=q.device)
    buffers = [
        _buffer(q, lib),
        _buffer(k, lib),
        _buffer(v, lib),
        _buffer(beta, lib),
        _mutable_buffer(state, lib),
        _buffer(y, lib, copy_back=True),
    ]
    total_values = layers * profile.delta.v_features
    launch_tracing.record_dispatch(
        KERNEL_DELTANET_INTERVAL,
        value_heads=layers * profile.delta.value_heads,
    )
    dispatch_kernel(
        lib,
        KERNEL_DELTANET_INTERVAL,
        ((total_values + 63) // 64, 1, 1),
        (64, 1, 1),
        buffers,
        wait=wait,
    )
    return y


def dispatch_linear_attention(
    hidden: torch.Tensor,
    q: PackedInt4Matrix,
    k: PackedInt4Matrix,
    v: PackedInt4Matrix,
    beta: PackedInt4Matrix,
    state: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
) -> torch.Tensor:
    """Run fused projection plus DeltaNet update in one command buffer."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    hidden = _ensure_mps_half_vector(hidden, "hidden", profile.hidden_size)
    q = _ensure_matrix(q, "linear_attn_q")
    k = _ensure_matrix(k, "linear_attn_k")
    v = _ensure_matrix(v, "linear_attn_v")
    beta = _ensure_matrix(beta, "linear_attn_beta")
    state = _ensure_mutable_mps_half_vector(state, "state", profile.delta.state_elements)

    q_scratch = _shared_buffer(profile.delta.q_features * 2, lib)
    k_scratch = _shared_buffer(profile.delta.k_features * 2, lib)
    v_scratch = _shared_buffer(profile.delta.v_features * 2, lib)
    beta_scratch = _shared_buffer(profile.delta.beta_features * 2, lib)
    y = torch.empty(profile.delta.v_features, dtype=torch.float16, device=hidden.device)

    total_cols = (
        profile.delta.q_features
        + profile.delta.k_features
        + profile.delta.v_features
        + profile.delta.beta_features
    )
    qkvb_buffers = [
        _buffer(hidden, lib),
        _buffer(q.qweight, lib),
        _buffer(q.scales, lib),
        _buffer(q.zeros, lib),
        _buffer(k.qweight, lib),
        _buffer(k.scales, lib),
        _buffer(k.zeros, lib),
        _buffer(v.qweight, lib),
        _buffer(v.scales, lib),
        _buffer(v.zeros, lib),
        _buffer(beta.qweight, lib),
        _buffer(beta.scales, lib),
        _buffer(beta.zeros, lib),
        q_scratch,
        k_scratch,
        v_scratch,
        beta_scratch,
        _params([profile.group_size, total_cols], lib),
    ]
    deltanet_buffers = [
        q_scratch,
        k_scratch,
        v_scratch,
        beta_scratch,
        _mutable_buffer(state, lib),
        _buffer(y, lib, copy_back=True),
    ]

    with lib.batch_dispatch(wait=True):
        launch_tracing.record_dispatch(KERNEL_QKVB, total_cols=total_cols)
        dispatch_kernel(
            lib,
            KERNEL_QKVB,
            (total_cols, 1, 1),
            (QWEN36_PROJECTION_THREADS_PER_GROUP, 1, 1),
            qkvb_buffers,
            wait=False,
        )
        launch_tracing.record_dispatch(
            KERNEL_DELTANET_UPDATE,
            value_heads=profile.delta.value_heads,
        )
        value_block_cols = 16
        deltanet_blocks = profile.delta.value_heads * (
            profile.delta.value_dim // value_block_cols
        )
        dispatch_kernel(
            lib,
            KERNEL_DELTANET_UPDATE,
            (deltanet_blocks, 1, 1),
            (QWEN36_DELTANET_THREADS_PER_GROUP, 1, 1),
            deltanet_buffers,
            wait=False,
        )
    return y


def dispatch_linear_o_residual(
    linear_out: torch.Tensor,
    out_proj: PackedInt4Matrix,
    residual: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run linear-attention output projection and residual add."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    linear_out = _ensure_mps_half_vector(linear_out, "linear_out", profile.delta.v_features)
    residual = _ensure_mps_half_vector(residual, "residual", profile.hidden_size)
    out_proj = _ensure_matrix(out_proj, "linear_attn_out")
    out = torch.empty(profile.hidden_size, dtype=torch.float16, device=linear_out.device)
    buffers = [
        _buffer(linear_out, lib),
        _buffer(out_proj.qweight, lib),
        _buffer(out_proj.scales, lib),
        _buffer(out_proj.zeros, lib),
        _buffer(residual, lib),
        _buffer(out, lib, copy_back=True),
        _params([profile.group_size], lib),
    ]
    launch_tracing.record_dispatch(KERNEL_LINEAR_OUT, hidden_size=profile.hidden_size)
    dispatch_kernel(
        lib,
        KERNEL_LINEAR_OUT,
        (profile.hidden_size, 1, 1),
        (QWEN36_INT4_GEMV_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return out


def dispatch_dense_gate_up_silu(
    hidden: torch.Tensor,
    gate: PackedInt4Matrix,
    up: PackedInt4Matrix,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run fused dense MLP gate/up projection plus SiLU activation."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    hidden = _ensure_mps_half_vector(hidden, "hidden", profile.hidden_size)
    gate = _ensure_matrix(gate, "mlp_gate")
    up = _ensure_matrix(up, "mlp_up")
    intermediate = torch.empty(
        profile.dense_mlp.intermediate_size,
        dtype=torch.float16,
        device=hidden.device,
    )
    buffers = [
        _buffer(hidden, lib),
        _buffer(gate.qweight, lib),
        _buffer(gate.scales, lib),
        _buffer(gate.zeros, lib),
        _buffer(up.qweight, lib),
        _buffer(up.scales, lib),
        _buffer(up.zeros, lib),
        _buffer(intermediate, lib, copy_back=True),
        _params([profile.group_size], lib),
    ]
    launch_tracing.record_dispatch(
        KERNEL_DENSE_GATE_UP,
        intermediate_size=profile.dense_mlp.intermediate_size,
    )
    dispatch_kernel(
        lib,
        KERNEL_DENSE_GATE_UP,
        (profile.dense_mlp.intermediate_size, 1, 1),
        (QWEN36_INT4_GEMV_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return intermediate


def dispatch_dense_down_residual(
    intermediate: torch.Tensor,
    down: PackedInt4Matrix,
    residual: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run dense MLP down projection and residual add."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    intermediate = _ensure_mps_half_vector(
        intermediate,
        "intermediate",
        profile.dense_mlp.intermediate_size,
    )
    residual = _ensure_mps_half_vector(residual, "residual", profile.hidden_size)
    down = _ensure_matrix(down, "mlp_down")
    out = torch.empty(profile.hidden_size, dtype=torch.float16, device=intermediate.device)
    buffers = [
        _buffer(intermediate, lib),
        _buffer(down.qweight, lib),
        _buffer(down.scales, lib),
        _buffer(down.zeros, lib),
        _buffer(residual, lib),
        _buffer(out, lib, copy_back=True),
        _params([profile.group_size], lib),
    ]
    launch_tracing.record_dispatch(KERNEL_DENSE_DOWN, hidden_size=profile.hidden_size)
    dispatch_kernel(
        lib,
        KERNEL_DENSE_DOWN,
        (profile.hidden_size, 1, 1),
        (QWEN36_INT4_GEMV_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return out


def dispatch_linear_rmsnorm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run Qwen3.6-27B linear-attention gated RMSNorm."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    x = _ensure_mps_half_vector(x, "x", profile.delta.v_features)
    gate = _ensure_mps_half_vector(gate, "gate", profile.delta.v_features)
    weight = _ensure_mps_half_vector(weight, "weight", profile.delta.value_dim)
    out = torch.empty(profile.delta.v_features, dtype=torch.float16, device=x.device)
    buffers = [
        _buffer(x, lib),
        _buffer(gate, lib),
        _buffer(weight, lib),
        _buffer(out, lib, copy_back=True),
    ]
    launch_tracing.record_dispatch(
        KERNEL_LINEAR_GATED_NORM,
        value_heads=profile.delta.value_heads,
    )
    dispatch_kernel(
        lib,
        KERNEL_LINEAR_GATED_NORM,
        (profile.delta.value_heads, 1, 1),
        (profile.delta.value_dim, 1, 1),
        buffers,
        wait=wait,
    )
    return out


def dispatch_rmsnorm_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run the Qwen3.6-27B hidden-size RMSNorm kernel."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    hidden = _ensure_mps_half_vector(hidden, "hidden", profile.hidden_size)
    weight = _ensure_mps_half_vector(weight, "weight", profile.hidden_size)
    out = torch.empty(profile.hidden_size, dtype=torch.float16, device=hidden.device)
    buffers = [
        _buffer(hidden, lib),
        _buffer(weight, lib),
        _buffer(out, lib, copy_back=True),
    ]
    launch_tracing.record_dispatch(KERNEL_RMSNORM, hidden_size=profile.hidden_size)
    dispatch_kernel(
        lib,
        KERNEL_RMSNORM,
        (1, 1, 1),
        (256, 1, 1),
        buffers,
        wait=wait,
    )
    return out


def dispatch_attention_cache_write(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_position: int,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> None:
    """Write one full-attention K/V projection into cache buffers."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    k = _ensure_mps_half_vector(k, "k", profile.attention.kv_features)
    v = _ensure_mps_half_vector(v, "v", profile.attention.kv_features)
    min_cache_elements = (token_position + 1) * profile.attention.kv_features
    k_cache = _ensure_mutable_mps_half_vector(
        k_cache,
        "k_cache",
        min_cache_elements,
        exact=False,
    )
    v_cache = _ensure_mutable_mps_half_vector(
        v_cache,
        "v_cache",
        min_cache_elements,
        exact=False,
    )
    buffers = [
        _buffer(k, lib),
        _buffer(v, lib),
        _mutable_buffer(k_cache, lib),
        _mutable_buffer(v_cache, lib),
        _params([token_position], lib),
    ]
    launch_tracing.record_dispatch(
        KERNEL_ATTENTION_CACHE,
        token_position=token_position,
    )
    dispatch_kernel(
        lib,
        KERNEL_ATTENTION_CACHE,
        ((profile.attention.kv_features + 63) // 64, 1, 1),
        (64, 1, 1),
        buffers,
        wait=wait,
    )


def dispatch_attention_decode(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_position: int,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run the full-attention decode helper for one token."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    q_proj = _ensure_mps_half_vector(q_proj, "q_proj", profile.attention.q_features)
    k_proj = _ensure_mps_half_vector(k_proj, "k_proj", profile.attention.kv_features)
    v_proj = _ensure_mps_half_vector(v_proj, "v_proj", profile.attention.kv_features)
    q_norm_weight = _ensure_mps_half_vector(
        q_norm_weight,
        "q_norm_weight",
        profile.attention.head_dim,
    )
    k_norm_weight = _ensure_mps_half_vector(
        k_norm_weight,
        "k_norm_weight",
        profile.attention.head_dim,
    )
    min_cache_elements = (token_position + 1) * profile.attention.kv_features
    k_cache = _ensure_mutable_mps_half_vector(
        k_cache,
        "k_cache",
        min_cache_elements,
        exact=False,
    )
    v_cache = _ensure_mutable_mps_half_vector(
        v_cache,
        "v_cache",
        min_cache_elements,
        exact=False,
    )
    out = torch.empty(profile.attention.o_features, dtype=torch.float16, device=q_proj.device)
    buffers = [
        _buffer(q_proj, lib),
        _buffer(k_proj, lib),
        _buffer(v_proj, lib),
        _buffer(q_norm_weight, lib),
        _buffer(k_norm_weight, lib),
        _mutable_buffer(k_cache, lib),
        _mutable_buffer(v_cache, lib),
        _buffer(out, lib, copy_back=True),
        _params([token_position], lib),
    ]
    launch_tracing.record_dispatch(
        KERNEL_ATTENTION_DECODE,
        heads=profile.attention.heads,
        token_position=token_position,
    )
    dispatch_kernel(
        lib,
        KERNEL_ATTENTION_DECODE,
        (profile.attention.heads, 1, 1),
        (profile.attention.head_dim, 1, 1),
        buffers,
        wait=wait,
    )
    return out


def dispatch_attention_o_residual(
    attn_out: torch.Tensor,
    out_proj: PackedInt4Matrix,
    residual: torch.Tensor,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run full-attention output projection and residual add."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    attn_out = _ensure_mps_half_vector(attn_out, "attn_out", profile.attention.o_features)
    residual = _ensure_mps_half_vector(residual, "residual", profile.hidden_size)
    out_proj = _ensure_matrix(out_proj, "full_attn_o")
    out = torch.empty(profile.hidden_size, dtype=torch.float16, device=attn_out.device)
    buffers = [
        _buffer(attn_out, lib),
        _buffer(out_proj.qweight, lib),
        _buffer(out_proj.scales, lib),
        _buffer(out_proj.zeros, lib),
        _buffer(residual, lib),
        _buffer(out, lib, copy_back=True),
        _params([profile.group_size], lib),
    ]
    launch_tracing.record_dispatch(KERNEL_ATTENTION_OUT, hidden_size=profile.hidden_size)
    dispatch_kernel(
        lib,
        KERNEL_ATTENTION_OUT,
        (profile.hidden_size, 1, 1),
        (QWEN36_INT4_GEMV_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return out


def dispatch_lm_head_logits(
    hidden: torch.Tensor,
    lm_head: PackedInt4Matrix,
    *,
    lib: MetalKernelLibrary | None = None,
    wait: bool = True,
) -> torch.Tensor:
    """Run Qwen3.6-27B int4 LM head projection."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    profile = QWEN36_27B_PROFILE
    hidden = _ensure_mps_half_vector(hidden, "hidden", profile.hidden_size)
    lm_head = _ensure_matrix(lm_head, "lm_head")
    logits = torch.empty(profile.vocab_size, dtype=torch.float16, device=hidden.device)
    buffers = [
        _buffer(hidden, lib),
        _buffer(lm_head.qweight, lib),
        _buffer(lm_head.scales, lib),
        _buffer(lm_head.zeros, lib),
        _buffer(logits, lib, copy_back=True),
        _params([profile.group_size], lib),
    ]
    launch_tracing.record_dispatch(KERNEL_LM_HEAD, vocab_size=profile.vocab_size)
    dispatch_kernel(
        lib,
        KERNEL_LM_HEAD,
        (profile.vocab_size, 1, 1),
        (QWEN36_INT4_GEMV_THREADS_PER_GROUP, 1, 1),
        buffers,
        wait=wait,
    )
    return logits


def dispatch_argmax(logits: torch.Tensor, *, lib: MetalKernelLibrary | None = None) -> torch.Tensor:
    """Run the Qwen3.6-27B vocabulary argmax kernel."""
    require_mps()
    lib = get_qwen36_kernel_library() if lib is None else lib
    logits = _ensure_mps_half_vector(logits, "logits", QWEN36_27B_PROFILE.vocab_size)
    token = torch.empty(1, dtype=torch.int32, device=logits.device)
    buffers = [_buffer(logits, lib), _buffer(token, lib, copy_back=True)]
    launch_tracing.record_dispatch(KERNEL_ARGMAX, vocab_size=QWEN36_27B_PROFILE.vocab_size)
    dispatch_kernel(
        lib,
        KERNEL_ARGMAX,
        (1, 1, 1),
        (256, 1, 1),
        buffers,
        wait=True,
    )
    return token


def validate_profile(profile: Qwen36ModelProfile = QWEN36_27B_PROFILE) -> None:
    validate_supported_profile(profile)
