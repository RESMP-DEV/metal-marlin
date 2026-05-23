#!/usr/bin/env python3
"""Qwen3.6-27B fused block-skeleton benchmark.

This is the repo-native replacement for the prototype ``qwen36_e2e_bench``
surface. It intentionally measures the current Metal Marlin fused-wrapper
skeleton over imported QMI4 artifacts; it is not a quality or serving claim
because the default local artifact reuses template layer-0/layer-3 tensors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metal_marlin import launch_tracing  # noqa: E402
from metal_marlin._compat import HAS_TORCH, torch  # noqa: E402
from metal_marlin.kernels.qwen36_27b import (  # noqa: E402
    KERNEL_ARGMAX,
    KERNEL_ATTENTION_CACHE,
    KERNEL_ATTENTION_DECODE,
    KERNEL_ATTENTION_OUT,
    KERNEL_ATTENTION_QKV,
    KERNEL_DELTANET_UPDATE,
    KERNEL_DENSE_DOWN,
    KERNEL_DENSE_GATE_UP,
    KERNEL_LINEAR_AZ,
    KERNEL_LINEAR_GATED_NORM,
    KERNEL_LINEAR_OUT,
    KERNEL_LM_HEAD,
    KERNEL_QKVB,
    KERNEL_RMSNORM,
    dispatch_argmax,
    dispatch_attention_cache_write,
    dispatch_attention_decode,
    dispatch_attention_o_residual,
    dispatch_attention_qkv_projection,
    dispatch_dense_down_residual,
    dispatch_dense_gate_up_silu,
    dispatch_linear_attention,
    dispatch_linear_az_projection,
    dispatch_linear_o_residual,
    dispatch_linear_rmsnorm_gated,
    dispatch_lm_head_logits,
    dispatch_rmsnorm_hidden,
    get_qwen36_kernel_library,
    load_packed_int4_matrix,
)
from metal_marlin.metal_dispatch import Metal, dispatch_kernel  # noqa: E402
from metal_marlin.qwen36_27b_artifact import (  # noqa: E402
    GLOBAL_TENSOR_ROLES,
    LAYER_LOCAL_TENSOR_ROLES,
    ManifestCoverageKind,
    ManifestTensorKey,
    Qwen36ArtifactManifest,
    TensorRole,
    artifact_coverage,
    manifest_tensor_index,
    read_manifest,
)
from metal_marlin.qwen36_27b_profile import QWEN36_27B_PROFILE, Qwen36ModelProfile  # noqa: E402
from metal_marlin.qwen36_27b_validation import default_block_layer0_manifest_path  # noqa: E402

logger = logging.getLogger(__name__)

TEMPLATE_WEIGHT_NOTES = (
    "Template-weight block skeleton for launch/timing evidence only; "
    "pack every layer and run generation/perplexity before quality claims."
)
FULL_LAYER_NOTES = (
    "Full-layer block skeleton for launch/timing evidence only; "
    "run generation/perplexity before quality claims."
)


@dataclass(frozen=True)
class Qwen36BlockSkeletonSummary:
    manifest_path: str | None
    runner: str
    decode_tokens: int
    warmup_tokens: int
    elapsed_ms: float
    decode_tok_per_s: float
    dispatch_count: int
    dispatches_per_token: float
    command_buffers: int
    command_buffers_per_token: float
    expected_dispatches_per_token: int
    kernel_counts: dict[str, int]
    expected_kernel_counts: dict[str, int]
    coverage_kind: ManifestCoverageKind | None = None
    coverage_layers: int | None = None
    coverage_tensors: int | None = None
    template_weight_reuse: bool = True
    quality_claim: bool = False
    notes: str = TEMPLATE_WEIGHT_NOTES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestTensorLookup:
    coverage_kind: ManifestCoverageKind
    coverage_layers: int
    coverage_tensors: int
    values: dict[ManifestTensorKey, Any]
    template_keys: dict[TensorRole, ManifestTensorKey]
    global_keys: dict[TensorRole, ManifestTensorKey]

    @property
    def template_weight_reuse(self) -> bool:
        return self.coverage_kind == "template"

    def tensor(self, role: TensorRole, *, layer_index: int | None = None) -> Any:
        if role in GLOBAL_TENSOR_ROLES:
            return self.values[self.global_keys[role]]

        if role not in LAYER_LOCAL_TENSOR_ROLES:
            raise ValueError(f"unsupported Qwen3.6-27B tensor role: {role}")

        if self.template_weight_reuse:
            return self.values[self.template_keys[role]]

        if layer_index is None:
            raise ValueError(f"layer_index is required for full-layer role {role}")

        key = (layer_index, role)
        if key not in self.values:
            raise ValueError(f"role {role} is not available for layer {layer_index}")
        return self.values[key]


def manifest_tensor_lookup(manifest: Qwen36ArtifactManifest) -> ManifestTensorLookup:
    coverage = artifact_coverage(manifest)
    values = manifest_tensor_index(manifest)
    template_keys: dict[TensorRole, ManifestTensorKey] = {}
    global_keys: dict[TensorRole, ManifestTensorKey] = {}

    for key, tensor in values.items():
        if tensor.role in GLOBAL_TENSOR_ROLES:
            global_keys[tensor.role] = key
        elif coverage.kind == "template":
            template_keys[tensor.role] = key

    return ManifestTensorLookup(
        coverage_kind=coverage.kind,
        coverage_layers=coverage.layers,
        coverage_tensors=coverage.tensors,
        values=values,
        template_keys=template_keys,
        global_keys=global_keys,
    )


def _with_loaded_values(
    lookup: ManifestTensorLookup,
    values: dict[ManifestTensorKey, Any],
) -> ManifestTensorLookup:
    return ManifestTensorLookup(
        coverage_kind=lookup.coverage_kind,
        coverage_layers=lookup.coverage_layers,
        coverage_tensors=lookup.coverage_tensors,
        values=values,
        template_keys=lookup.template_keys,
        global_keys=lookup.global_keys,
    )


def expected_kernel_breakdown(
    profile: Qwen36ModelProfile = QWEN36_27B_PROFILE,
) -> dict[str, int]:
    """Return the current wrapper-level block-skeleton dispatch plan."""

    linear_layers = profile.num_linear_attention_layers
    full_layers = profile.num_full_attention_layers
    return {
        KERNEL_RMSNORM: profile.num_hidden_layers * 2 + 1,
        KERNEL_QKVB: linear_layers,
        KERNEL_DELTANET_UPDATE: linear_layers,
        KERNEL_LINEAR_AZ: linear_layers,
        KERNEL_LINEAR_GATED_NORM: linear_layers,
        KERNEL_LINEAR_OUT: linear_layers,
        KERNEL_ATTENTION_QKV: full_layers,
        KERNEL_ATTENTION_CACHE: full_layers,
        KERNEL_ATTENTION_DECODE: full_layers,
        KERNEL_ATTENTION_OUT: full_layers,
        KERNEL_DENSE_GATE_UP: profile.num_hidden_layers,
        KERNEL_DENSE_DOWN: profile.num_hidden_layers,
        KERNEL_LM_HEAD: 1,
        KERNEL_ARGMAX: 1,
    }


def expected_dispatches_per_token(
    profile: Qwen36ModelProfile = QWEN36_27B_PROFILE,
) -> int:
    return sum(expected_kernel_breakdown(profile).values())


def kernel_counts_from_names(kernel_names: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(kernel_names).items()))


def wrapper_command_buffers_from_kernel_names(kernel_names: Sequence[str]) -> int:
    """Estimate command buffers for the public wrapper skeleton.

    ``dispatch_linear_attention`` batches each QKVB + DeltaNet pair in one
    command buffer; the other first-wave wrappers currently wait per dispatch.
    """

    counts = Counter(kernel_names)
    fused_linear_pairs = min(counts[KERNEL_QKVB], counts[KERNEL_DELTANET_UPDATE])
    return len(kernel_names) - fused_linear_pairs


def prewarm_qwen36_block_skeleton_pipelines() -> None:
    """Create metallib-backed pipeline states before timed decode measurement."""

    lib = get_qwen36_kernel_library()
    for kernel_name in expected_kernel_breakdown():
        lib.get_function(kernel_name)


def make_summary(
    *,
    manifest_path: str | None,
    runner: str,
    decode_tokens: int,
    warmup_tokens: int,
    elapsed_ms: float,
    kernel_names: Sequence[str],
    command_buffers: int = 0,
    coverage_kind: ManifestCoverageKind | None = None,
    coverage_layers: int | None = None,
    coverage_tensors: int | None = None,
) -> Qwen36BlockSkeletonSummary:
    dispatch_count = len(kernel_names)
    active_tokens = max(decode_tokens, 0)
    expected = expected_dispatches_per_token()
    template_weight_reuse = coverage_kind != "full_layers"
    return Qwen36BlockSkeletonSummary(
        manifest_path=manifest_path,
        runner=runner,
        decode_tokens=decode_tokens,
        warmup_tokens=warmup_tokens,
        elapsed_ms=elapsed_ms,
        decode_tok_per_s=0.0 if elapsed_ms <= 0 else active_tokens / (elapsed_ms / 1000.0),
        dispatch_count=dispatch_count,
        dispatches_per_token=0.0 if active_tokens == 0 else dispatch_count / active_tokens,
        command_buffers=command_buffers,
        command_buffers_per_token=(
            0.0 if active_tokens == 0 else command_buffers / active_tokens
        ),
        expected_dispatches_per_token=expected,
        kernel_counts=kernel_counts_from_names(kernel_names),
        expected_kernel_counts=expected_kernel_breakdown(),
        coverage_kind=coverage_kind,
        coverage_layers=coverage_layers,
        coverage_tensors=coverage_tensors,
        template_weight_reuse=template_weight_reuse,
        notes=FULL_LAYER_NOTES if coverage_kind == "full_layers" else TEMPLATE_WEIGHT_NOTES,
    )


def assert_launch_budget(
    summary: Qwen36BlockSkeletonSummary,
    *,
    max_dispatches_per_token: float | None = None,
    max_command_buffers_per_token: float | None = None,
) -> None:
    """Raise if the measured skeleton exceeds a declared launch budget."""

    if (
        max_dispatches_per_token is not None
        and summary.dispatches_per_token > max_dispatches_per_token
    ):
        raise ValueError(
            "dispatch budget exceeded: "
            f"{summary.dispatches_per_token:.3f} > {max_dispatches_per_token:.3f}"
        )
    if (
        max_command_buffers_per_token is not None
        and summary.command_buffers_per_token > max_command_buffers_per_token
    ):
        raise ValueError(
            "command-buffer budget exceeded: "
            f"{summary.command_buffers_per_token:.3f} > "
            f"{max_command_buffers_per_token:.3f}"
        )


def _role_artifacts(
    manifest_path: Path,
    *,
    device: str,
) -> ManifestTensorLookup:
    manifest = read_manifest(manifest_path)
    lookup = manifest_tensor_lookup(manifest)
    values = {
        key: load_packed_int4_matrix(tensor, manifest_path.parent, device=device)
        for key, tensor in lookup.values.items()
    }
    return _with_loaded_values(lookup, values)


def _manifest_tensor_map(manifest_path: Path) -> ManifestTensorLookup:
    manifest = read_manifest(manifest_path)
    return manifest_tensor_lookup(manifest)


def _buffer_from_bytes(lib, data: bytes, label: str):
    buffer = lib.device.newBufferWithBytes_length_options_(
        data,
        len(data),
        Metal.MTLResourceStorageModeShared,
    )
    if buffer is None:
        raise RuntimeError(f"failed to allocate Metal buffer for {label}")
    return buffer


def _buffer_from_file(lib, path: Path, expected_bytes: int):
    data = path.read_bytes()
    if len(data) != expected_bytes:
        raise ValueError(f"{path} has {len(data)} bytes, expected {expected_bytes}")
    return _buffer_from_bytes(lib, data, str(path))


def _empty_buffer(lib, byte_size: int, label: str):
    buffer = lib.device.newBufferWithLength_options_(
        byte_size,
        Metal.MTLResourceStorageModeShared,
    )
    if buffer is None:
        raise RuntimeError(f"failed to allocate {byte_size} byte Metal buffer for {label}")
    return buffer


def _filled_half_buffer(lib, elements: int, value: float, label: str):
    import numpy as np

    return _buffer_from_bytes(
        lib,
        np.full(elements, value, dtype=np.float16).tobytes(),
        label,
    )


def _params_buffer(lib, values: Sequence[int], label: str):
    import numpy as np

    return _buffer_from_bytes(
        lib,
        np.asarray(values, dtype=np.int32).tobytes(),
        label,
    )


@dataclass(frozen=True)
class _MetalInt4Matrix:
    qweight: Any
    scales: Any
    zeros: Any


@dataclass(frozen=True)
class _DirectBuffers:
    weights: ManifestTensorLookup
    hidden_a: Any
    hidden_b: Any
    norm: Any
    q: Any
    k: Any
    v: Any
    beta: Any
    a: Any
    z: Any
    linear_out: Any
    linear_gated: Any
    attn_q: Any
    attn_k: Any
    attn_v: Any
    attn_out: Any
    intermediate: Any
    logits: Any
    token: Any
    linear_states: list[Any]
    k_caches: list[Any]
    v_caches: list[Any]
    input_norm: Any
    post_norm: Any
    final_norm: Any
    linear_norm: Any
    q_norm: Any
    k_norm: Any
    qkvb_params: Any
    linear_az_params: Any
    group_params: Any
    dense_gate_params: Any
    attention_qkv_params: Any
    token_params: list[Any]


def _load_direct_buffers(
    manifest_path: Path,
    *,
    max_context: int,
) -> _DirectBuffers:
    profile = QWEN36_27B_PROFILE
    lib = get_qwen36_kernel_library()
    tensor_map = _manifest_tensor_map(manifest_path)
    weights: dict[ManifestTensorKey, _MetalInt4Matrix] = {}
    for key, tensor in tensor_map.values.items():
        groups = (tensor.in_features + tensor.group_size - 1) // tensor.group_size
        weights[key] = _MetalInt4Matrix(
            qweight=_buffer_from_file(
                lib,
                manifest_path.parent / tensor.qweight,
                (tensor.in_features // 8) * tensor.out_features * 4,
            ),
            scales=_buffer_from_file(
                lib,
                manifest_path.parent / tensor.scales,
                groups * tensor.out_features * 2,
            ),
            zeros=_buffer_from_file(
                lib,
                manifest_path.parent / tensor.zeros,
                groups * tensor.out_features * 2,
            ),
        )

    half = 2
    int32 = 4
    qkvb_cols = (
        profile.delta.q_features
        + profile.delta.k_features
        + profile.delta.v_features
        + profile.delta.beta_features
    )
    linear_az_cols = profile.delta.beta_features + profile.delta.v_features
    attention_qkv_cols = (
        profile.attention.q_features
        + profile.attention.kv_features
        + profile.attention.kv_features
    )
    cache_elements = max_context * profile.attention.kv_features
    return _DirectBuffers(
        weights=_with_loaded_values(tensor_map, weights),
        hidden_a=_filled_half_buffer(lib, profile.hidden_size, 1.0, "hidden_a"),
        hidden_b=_filled_half_buffer(lib, profile.hidden_size, 0.0, "hidden_b"),
        norm=_empty_buffer(lib, profile.hidden_size * half, "norm"),
        q=_empty_buffer(lib, profile.delta.q_features * half, "linear_q"),
        k=_empty_buffer(lib, profile.delta.k_features * half, "linear_k"),
        v=_empty_buffer(lib, profile.delta.v_features * half, "linear_v"),
        beta=_empty_buffer(lib, profile.delta.beta_features * half, "linear_beta"),
        a=_empty_buffer(lib, profile.delta.beta_features * half, "linear_a"),
        z=_empty_buffer(lib, profile.delta.v_features * half, "linear_z"),
        linear_out=_empty_buffer(lib, profile.delta.v_features * half, "linear_out"),
        linear_gated=_empty_buffer(lib, profile.delta.v_features * half, "linear_gated"),
        attn_q=_empty_buffer(lib, profile.attention.q_features * half, "attn_q"),
        attn_k=_empty_buffer(lib, profile.attention.kv_features * half, "attn_k"),
        attn_v=_empty_buffer(lib, profile.attention.kv_features * half, "attn_v"),
        attn_out=_empty_buffer(lib, profile.attention.o_features * half, "attn_out"),
        intermediate=_empty_buffer(
            lib,
            profile.dense_mlp.intermediate_size * half,
            "intermediate",
        ),
        logits=_empty_buffer(lib, profile.vocab_size * half, "logits"),
        token=_empty_buffer(lib, int32, "token"),
        linear_states=[
            _filled_half_buffer(lib, profile.delta.state_elements, 0.0, f"linear_state_{idx}")
            for idx in range(profile.num_linear_attention_layers)
        ],
        k_caches=[
            _filled_half_buffer(lib, cache_elements, 0.0, f"k_cache_{idx}")
            for idx in range(profile.num_full_attention_layers)
        ],
        v_caches=[
            _filled_half_buffer(lib, cache_elements, 0.0, f"v_cache_{idx}")
            for idx in range(profile.num_full_attention_layers)
        ],
        input_norm=_filled_half_buffer(lib, profile.hidden_size, 1.0, "input_norm"),
        post_norm=_filled_half_buffer(lib, profile.hidden_size, 1.0, "post_norm"),
        final_norm=_filled_half_buffer(lib, profile.hidden_size, 1.0, "final_norm"),
        linear_norm=_filled_half_buffer(lib, profile.delta.value_dim, 1.0, "linear_norm"),
        q_norm=_filled_half_buffer(lib, profile.attention.head_dim, 1.0, "q_norm"),
        k_norm=_filled_half_buffer(lib, profile.attention.head_dim, 1.0, "k_norm"),
        qkvb_params=_params_buffer(lib, [profile.group_size, qkvb_cols], "qkvb_params"),
        linear_az_params=_params_buffer(
            lib,
            [profile.group_size, linear_az_cols],
            "linear_az_params",
        ),
        group_params=_params_buffer(lib, [profile.group_size], "group_params"),
        dense_gate_params=_params_buffer(lib, [profile.group_size], "dense_gate_params"),
        attention_qkv_params=_params_buffer(
            lib,
            [profile.group_size, attention_qkv_cols],
            "attention_qkv_params",
        ),
        token_params=[
            _params_buffer(lib, [token_position], f"token_params_{token_position}")
            for token_position in range(max_context)
        ],
    )


def _ones(size: int, *, device: str):
    return torch.ones(size, dtype=torch.float16, device=device)


def _zeros(size: int, *, device: str):
    return torch.zeros(size, dtype=torch.float16, device=device)


def _run_one_token(
    hidden,
    artifacts: ManifestTensorLookup,
    linear_states: list[Any],
    k_caches: list[Any],
    v_caches: list[Any],
    token_position: int,
    *,
    device: str,
):
    profile = QWEN36_27B_PROFILE
    input_norm = _ones(profile.hidden_size, device=device)
    post_norm = _ones(profile.hidden_size, device=device)
    final_norm = _ones(profile.hidden_size, device=device)
    linear_norm = _ones(profile.delta.value_dim, device=device)
    q_norm = _ones(profile.attention.head_dim, device=device)
    k_norm = _ones(profile.attention.head_dim, device=device)
    linear_index = 0
    full_index = 0

    for layer_index, layer_type in enumerate(profile.layer_types):
        residual = hidden
        hidden = dispatch_rmsnorm_hidden(hidden, input_norm)

        if layer_type == "linear_attention":
            linear_out = dispatch_linear_attention(
                hidden,
                artifacts.tensor("linear_attn_q", layer_index=layer_index),
                artifacts.tensor("linear_attn_k", layer_index=layer_index),
                artifacts.tensor("linear_attn_v", layer_index=layer_index),
                artifacts.tensor("linear_attn_beta", layer_index=layer_index),
                linear_states[linear_index],
            )
            az = dispatch_linear_az_projection(
                hidden,
                artifacts.tensor("linear_attn_a", layer_index=layer_index),
                artifacts.tensor("linear_attn_z", layer_index=layer_index),
            )
            linear_out = dispatch_linear_rmsnorm_gated(linear_out, az.z, linear_norm)
            hidden = dispatch_linear_o_residual(
                linear_out,
                artifacts.tensor("linear_attn_out", layer_index=layer_index),
                residual,
            )
            linear_index += 1
        else:
            projected = dispatch_attention_qkv_projection(
                hidden,
                artifacts.tensor("full_attn_q", layer_index=layer_index),
                artifacts.tensor("full_attn_k", layer_index=layer_index),
                artifacts.tensor("full_attn_v", layer_index=layer_index),
            )
            dispatch_attention_cache_write(
                projected.k,
                projected.v,
                k_caches[full_index],
                v_caches[full_index],
                token_position,
            )
            attn_out = dispatch_attention_decode(
                projected.q,
                projected.k,
                projected.v,
                q_norm,
                k_norm,
                k_caches[full_index],
                v_caches[full_index],
                token_position,
            )
            hidden = dispatch_attention_o_residual(
                attn_out,
                artifacts.tensor("full_attn_o", layer_index=layer_index),
                residual,
            )
            full_index += 1

        residual = hidden
        hidden = dispatch_rmsnorm_hidden(hidden, post_norm)
        intermediate = dispatch_dense_gate_up_silu(
            hidden,
            artifacts.tensor("mlp_gate", layer_index=layer_index),
            artifacts.tensor("mlp_up", layer_index=layer_index),
        )
        hidden = dispatch_dense_down_residual(
            intermediate,
            artifacts.tensor("mlp_down", layer_index=layer_index),
            residual,
        )

    hidden = dispatch_rmsnorm_hidden(hidden, final_norm)
    logits = dispatch_lm_head_logits(hidden, artifacts.tensor("lm_head"))
    token = dispatch_argmax(logits)
    return hidden, int(token.cpu().item())


def _emit(kernel_name: str, grid: tuple[int, int, int], threadgroup: tuple[int, int, int], buffers):
    lib = get_qwen36_kernel_library()
    launch_tracing.record_dispatch(kernel_name)
    dispatch_kernel(lib, kernel_name, grid, threadgroup, buffers, wait=False)


def _next_hidden_buffer(buffers: _DirectBuffers, current):
    return buffers.hidden_b if current is buffers.hidden_a else buffers.hidden_a


def _run_one_token_direct(buffers: _DirectBuffers, token_position: int, hidden):
    profile = QWEN36_27B_PROFILE
    linear_index = 0
    full_index = 0

    for layer_index, layer_type in enumerate(profile.layer_types):
        residual = hidden
        _emit(
            KERNEL_RMSNORM,
            (1, 1, 1),
            (256, 1, 1),
            [hidden, buffers.input_norm, buffers.norm],
        )

        next_hidden = _next_hidden_buffer(buffers, hidden)
        if layer_type == "linear_attention":
            weights = buffers.weights
            q_weight = weights.tensor("linear_attn_q", layer_index=layer_index)
            k_weight = weights.tensor("linear_attn_k", layer_index=layer_index)
            v_weight = weights.tensor("linear_attn_v", layer_index=layer_index)
            beta_weight = weights.tensor("linear_attn_beta", layer_index=layer_index)
            _emit(
                KERNEL_QKVB,
                (
                    profile.delta.q_features
                    + profile.delta.k_features
                    + profile.delta.v_features
                    + profile.delta.beta_features,
                    1,
                    1,
                ),
                (64, 1, 1),
                [
                    buffers.norm,
                    q_weight.qweight,
                    q_weight.scales,
                    q_weight.zeros,
                    k_weight.qweight,
                    k_weight.scales,
                    k_weight.zeros,
                    v_weight.qweight,
                    v_weight.scales,
                    v_weight.zeros,
                    beta_weight.qweight,
                    beta_weight.scales,
                    beta_weight.zeros,
                    buffers.q,
                    buffers.k,
                    buffers.v,
                    buffers.beta,
                    buffers.qkvb_params,
                ],
            )
            _emit(
                KERNEL_DELTANET_UPDATE,
                (
                    profile.delta.value_heads * (profile.delta.value_dim // 16),
                    1,
                    1,
                ),
                (128, 1, 1),
                [
                    buffers.q,
                    buffers.k,
                    buffers.v,
                    buffers.beta,
                    buffers.linear_states[linear_index],
                    buffers.linear_out,
                ],
            )
            a_weight = weights.tensor("linear_attn_a", layer_index=layer_index)
            z_weight = weights.tensor("linear_attn_z", layer_index=layer_index)
            _emit(
                KERNEL_LINEAR_AZ,
                (
                    profile.delta.beta_features + profile.delta.v_features,
                    1,
                    1,
                ),
                (64, 1, 1),
                [
                    buffers.norm,
                    a_weight.qweight,
                    a_weight.scales,
                    a_weight.zeros,
                    z_weight.qweight,
                    z_weight.scales,
                    z_weight.zeros,
                    buffers.a,
                    buffers.z,
                    buffers.linear_az_params,
                ],
            )
            _emit(
                KERNEL_LINEAR_GATED_NORM,
                (profile.delta.value_heads, 1, 1),
                (profile.delta.value_dim, 1, 1),
                [buffers.linear_out, buffers.z, buffers.linear_norm, buffers.linear_gated],
            )
            out_weight = weights.tensor("linear_attn_out", layer_index=layer_index)
            _emit(
                KERNEL_LINEAR_OUT,
                (profile.hidden_size, 1, 1),
                (64, 1, 1),
                [
                    buffers.linear_gated,
                    out_weight.qweight,
                    out_weight.scales,
                    out_weight.zeros,
                    residual,
                    next_hidden,
                    buffers.group_params,
                ],
            )
            linear_index += 1
        else:
            weights = buffers.weights
            q_weight = weights.tensor("full_attn_q", layer_index=layer_index)
            k_weight = weights.tensor("full_attn_k", layer_index=layer_index)
            v_weight = weights.tensor("full_attn_v", layer_index=layer_index)
            _emit(
                KERNEL_ATTENTION_QKV,
                (
                    profile.attention.q_features
                    + profile.attention.kv_features
                    + profile.attention.kv_features,
                    1,
                    1,
                ),
                (64, 1, 1),
                [
                    buffers.norm,
                    q_weight.qweight,
                    q_weight.scales,
                    q_weight.zeros,
                    k_weight.qweight,
                    k_weight.scales,
                    k_weight.zeros,
                    v_weight.qweight,
                    v_weight.scales,
                    v_weight.zeros,
                    buffers.attn_q,
                    buffers.attn_k,
                    buffers.attn_v,
                    buffers.attention_qkv_params,
                ],
            )
            token_params = buffers.token_params[token_position]
            _emit(
                KERNEL_ATTENTION_CACHE,
                ((profile.attention.kv_features + 63) // 64, 1, 1),
                (64, 1, 1),
                [
                    buffers.attn_k,
                    buffers.attn_v,
                    buffers.k_caches[full_index],
                    buffers.v_caches[full_index],
                    token_params,
                ],
            )
            _emit(
                KERNEL_ATTENTION_DECODE,
                (profile.attention.heads, 1, 1),
                (profile.attention.head_dim, 1, 1),
                [
                    buffers.attn_q,
                    buffers.attn_k,
                    buffers.attn_v,
                    buffers.q_norm,
                    buffers.k_norm,
                    buffers.k_caches[full_index],
                    buffers.v_caches[full_index],
                    buffers.attn_out,
                    token_params,
                ],
            )
            out_weight = weights.tensor("full_attn_o", layer_index=layer_index)
            _emit(
                KERNEL_ATTENTION_OUT,
                (profile.hidden_size, 1, 1),
                (64, 1, 1),
                [
                    buffers.attn_out,
                    out_weight.qweight,
                    out_weight.scales,
                    out_weight.zeros,
                    residual,
                    next_hidden,
                    buffers.group_params,
                ],
            )
            full_index += 1

        hidden = next_hidden
        residual = hidden
        _emit(
            KERNEL_RMSNORM,
            (1, 1, 1),
            (256, 1, 1),
            [hidden, buffers.post_norm, buffers.norm],
        )
        gate_weight = buffers.weights.tensor("mlp_gate", layer_index=layer_index)
        up_weight = buffers.weights.tensor("mlp_up", layer_index=layer_index)
        _emit(
            KERNEL_DENSE_GATE_UP,
            (profile.dense_mlp.intermediate_size, 1, 1),
            (64, 1, 1),
            [
                buffers.norm,
                gate_weight.qweight,
                gate_weight.scales,
                gate_weight.zeros,
                up_weight.qweight,
                up_weight.scales,
                up_weight.zeros,
                buffers.intermediate,
                buffers.dense_gate_params,
            ],
        )
        next_hidden = _next_hidden_buffer(buffers, hidden)
        down_weight = buffers.weights.tensor("mlp_down", layer_index=layer_index)
        _emit(
            KERNEL_DENSE_DOWN,
            (profile.hidden_size, 1, 1),
            (64, 1, 1),
            [
                buffers.intermediate,
                down_weight.qweight,
                down_weight.scales,
                down_weight.zeros,
                residual,
                next_hidden,
                buffers.group_params,
            ],
        )
        hidden = next_hidden

    _emit(
        KERNEL_RMSNORM,
        (1, 1, 1),
        (256, 1, 1),
        [hidden, buffers.final_norm, buffers.norm],
    )
    lm_head = buffers.weights.tensor("lm_head")
    _emit(
        KERNEL_LM_HEAD,
        (profile.vocab_size, 1, 1),
        (64, 1, 1),
        [
            buffers.norm,
            lm_head.qweight,
            lm_head.scales,
            lm_head.zeros,
            buffers.logits,
            buffers.group_params,
        ],
    )
    _emit(KERNEL_ARGMAX, (1, 1, 1), (256, 1, 1), [buffers.logits, buffers.token])
    return hidden


def run_direct_benchmark(
    manifest_path: Path,
    *,
    decode_tokens: int,
    warmup_tokens: int,
    max_context: int,
) -> Qwen36BlockSkeletonSummary:
    if decode_tokens <= 0:
        raise ValueError("--decode-tokens must be > 0")
    if warmup_tokens < 0:
        raise ValueError("--warmup-tokens must be >= 0")
    if max_context < decode_tokens + warmup_tokens:
        raise ValueError("--max-context must cover warmup plus decode tokens")

    buffers = _load_direct_buffers(manifest_path, max_context=max_context)
    lib = get_qwen36_kernel_library()
    hidden = buffers.hidden_a

    for token_position in range(warmup_tokens):
        with lib.batch_dispatch(wait=True):
            hidden = _run_one_token_direct(buffers, token_position, hidden)

    launch_tracing.enable_for_testing()
    launch_tracing.reset()
    start = time.perf_counter()
    for offset in range(decode_tokens):
        with lib.batch_dispatch(wait=True):
            hidden = _run_one_token_direct(buffers, warmup_tokens + offset, hidden)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return make_summary(
        manifest_path=str(manifest_path),
        runner="direct-metal-buffer",
        decode_tokens=decode_tokens,
        warmup_tokens=warmup_tokens,
        elapsed_ms=elapsed_ms,
        kernel_names=launch_tracing.kernel_names(),
        command_buffers=decode_tokens,
        coverage_kind=buffers.weights.coverage_kind,
        coverage_layers=buffers.weights.coverage_layers,
        coverage_tensors=buffers.weights.coverage_tensors,
    )


def run_wrapper_benchmark(
    manifest_path: Path,
    *,
    decode_tokens: int,
    warmup_tokens: int,
    max_context: int,
    device: str = "mps",
) -> Qwen36BlockSkeletonSummary:
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required to run this benchmark.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Qwen3.6-27B block-skeleton benchmark.")
    if decode_tokens <= 0:
        raise ValueError("--decode-tokens must be > 0")
    if warmup_tokens < 0:
        raise ValueError("--warmup-tokens must be >= 0")
    if max_context < decode_tokens + warmup_tokens:
        raise ValueError("--max-context must cover warmup plus decode tokens")

    profile = QWEN36_27B_PROFILE
    artifacts = _role_artifacts(manifest_path, device=device)
    prewarm_qwen36_block_skeleton_pipelines()
    hidden = _ones(profile.hidden_size, device=device)
    linear_states = [
        _zeros(profile.delta.state_elements, device=device)
        for _ in range(profile.num_linear_attention_layers)
    ]
    cache_elements = max_context * profile.attention.kv_features
    k_caches = [_zeros(cache_elements, device=device) for _ in range(profile.num_full_attention_layers)]
    v_caches = [_zeros(cache_elements, device=device) for _ in range(profile.num_full_attention_layers)]

    for token_position in range(warmup_tokens):
        hidden, _ = _run_one_token(
            hidden,
            artifacts,
            linear_states,
            k_caches,
            v_caches,
            token_position,
            device=device,
        )

    launch_tracing.enable_for_testing()
    launch_tracing.reset()
    start = time.perf_counter()
    last_token = -1
    for offset in range(decode_tokens):
        hidden, last_token = _run_one_token(
            hidden,
            artifacts,
            linear_states,
            k_caches,
            v_caches,
            warmup_tokens + offset,
            device=device,
        )
    if device == "mps":
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    kernel_names = launch_tracing.kernel_names()
    logger.info("last_argmax_token=%s", last_token)
    return make_summary(
        manifest_path=str(manifest_path),
        runner="python-wrapper",
        decode_tokens=decode_tokens,
        warmup_tokens=warmup_tokens,
        elapsed_ms=elapsed_ms,
        kernel_names=kernel_names,
        command_buffers=wrapper_command_buffers_from_kernel_names(kernel_names),
        coverage_kind=artifacts.coverage_kind,
        coverage_layers=artifacts.coverage_layers,
        coverage_tensors=artifacts.coverage_tensors,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Qwen3.6-27B fused block skeleton over local artifacts."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_ROOT / default_block_layer0_manifest_path(),
        help="Qwen3.6-27B block artifact manifest.",
    )
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--max-dispatches-per-token",
        type=float,
        help="Fail if measured dispatches/token exceeds this launch budget.",
    )
    parser.add_argument(
        "--max-command-buffers-per-token",
        type=float,
        help="Fail if measured command buffers/token exceeds this launch budget.",
    )
    parser.add_argument(
        "--runner",
        choices=("direct", "wrapper"),
        default="direct",
        help="Use direct Metal buffers or the public Python wrapper path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expected launch plan without loading weights or running MPS kernels.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging from Metal setup and benchmark internals.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.dry_run:
        summary = make_summary(
            manifest_path=str(args.manifest),
            runner=args.runner,
            decode_tokens=args.decode_tokens,
            warmup_tokens=args.warmup_tokens,
            elapsed_ms=0.0,
            kernel_names=[],
        )
    elif args.runner == "direct":
        summary = run_direct_benchmark(
            args.manifest,
            decode_tokens=args.decode_tokens,
            warmup_tokens=args.warmup_tokens,
            max_context=args.max_context,
        )
    else:
        summary = run_wrapper_benchmark(
            args.manifest,
            decode_tokens=args.decode_tokens,
            warmup_tokens=args.warmup_tokens,
            max_context=args.max_context,
            device=args.device,
        )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    assert_launch_budget(
        summary,
        max_dispatches_per_token=args.max_dispatches_per_token,
        max_command_buffers_per_token=args.max_command_buffers_per_token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
