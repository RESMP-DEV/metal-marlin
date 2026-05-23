"""Qwen3.6-27B fused-path artifact metadata.

The prototype qwenmetal layout is supported only through this explicit schema:
packed unsigned int4 weights as ``qweight[(K / 8), N]`` plus FP16 per-group
``scales`` and ``zeros`` shaped ``[ceil(K / group_size), N]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .qwen36_27b_profile import MODEL_ID, QWEN36_27B_PROFILE

SCHEMA_VERSION = "qwen36_27b_int4_v1"
TensorRole = Literal[
    "linear_attn_q",
    "linear_attn_k",
    "linear_attn_v",
    "linear_attn_beta",
    "linear_attn_a",
    "linear_attn_z",
    "linear_attn_out",
    "full_attn_q",
    "full_attn_k",
    "full_attn_v",
    "full_attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "lm_head",
]
ManifestTensorKey = tuple[int | None, TensorRole]
ManifestCoverageKind = Literal["template", "full_layers"]

REQUIRED_TENSOR_ROLES: tuple[TensorRole, ...] = (
    "linear_attn_q",
    "linear_attn_k",
    "linear_attn_v",
    "linear_attn_beta",
    "linear_attn_a",
    "linear_attn_z",
    "linear_attn_out",
    "full_attn_q",
    "full_attn_k",
    "full_attn_v",
    "full_attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "lm_head",
)
LINEAR_ATTENTION_TENSOR_ROLES: tuple[TensorRole, ...] = (
    "linear_attn_q",
    "linear_attn_k",
    "linear_attn_v",
    "linear_attn_beta",
    "linear_attn_a",
    "linear_attn_z",
    "linear_attn_out",
)
FULL_ATTENTION_TENSOR_ROLES: tuple[TensorRole, ...] = (
    "full_attn_q",
    "full_attn_k",
    "full_attn_v",
    "full_attn_o",
)
DENSE_MLP_TENSOR_ROLES: tuple[TensorRole, ...] = (
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)
GLOBAL_TENSOR_ROLES: tuple[TensorRole, ...] = ("lm_head",)
LAYER_LOCAL_TENSOR_ROLES: tuple[TensorRole, ...] = (
    LINEAR_ATTENTION_TENSOR_ROLES
    + FULL_ATTENTION_TENSOR_ROLES
    + DENSE_MLP_TENSOR_ROLES
)


ROLE_SHAPES: dict[str, tuple[int, int]] = {
    "linear_attn_q": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.q_features,
    ),
    "linear_attn_k": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.k_features,
    ),
    "linear_attn_v": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.v_features,
    ),
    "linear_attn_beta": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.beta_features,
    ),
    "linear_attn_a": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.beta_features,
    ),
    "linear_attn_z": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.delta.v_features,
    ),
    "linear_attn_out": (
        QWEN36_27B_PROFILE.delta.v_features,
        QWEN36_27B_PROFILE.hidden_size,
    ),
    "full_attn_q": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.attention.q_features,
    ),
    "full_attn_k": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.attention.kv_features,
    ),
    "full_attn_v": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.attention.kv_features,
    ),
    "full_attn_o": (
        QWEN36_27B_PROFILE.attention.o_features,
        QWEN36_27B_PROFILE.hidden_size,
    ),
    "mlp_gate": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.dense_mlp.intermediate_size,
    ),
    "mlp_up": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.dense_mlp.intermediate_size,
    ),
    "mlp_down": (
        QWEN36_27B_PROFILE.dense_mlp.intermediate_size,
        QWEN36_27B_PROFILE.hidden_size,
    ),
    "lm_head": (
        QWEN36_27B_PROFILE.hidden_size,
        QWEN36_27B_PROFILE.vocab_size,
    ),
}


@dataclass(frozen=True)
class Int4TensorArtifact:
    role: TensorRole
    layer_index: int | None
    qweight: str
    scales: str
    zeros: str
    in_features: int
    out_features: int
    group_size: int = QWEN36_27B_PROFILE.group_size
    dtype: str = "uint4_asym"
    qweight_layout: str = "packed_k_major_u32"
    metadata_layout: str = "group_major_f16"
    calibration: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.role not in ROLE_SHAPES:
            raise ValueError(f"unsupported Qwen3.6-27B tensor role: {self.role}")
        expected_in, expected_out = ROLE_SHAPES[self.role]
        if (self.in_features, self.out_features) != (expected_in, expected_out):
            raise ValueError(
                f"{self.role} expected shape ({expected_in}, {expected_out}), "
                f"got ({self.in_features}, {self.out_features})"
            )
        if self.in_features % 8 != 0:
            raise ValueError("qweight layout requires in_features divisible by 8")
        if self.group_size <= 0 or self.group_size % 8 != 0:
            raise ValueError("group_size must be a positive multiple of 8")
        if self.group_size != QWEN36_27B_PROFILE.group_size:
            raise ValueError(
                f"{self.role} requires group_size={QWEN36_27B_PROFILE.group_size}"
            )
        if self.dtype != "uint4_asym":
            raise ValueError(f"{self.role} expected dtype=uint4_asym, got {self.dtype}")
        if self.qweight_layout != "packed_k_major_u32":
            raise ValueError(
                f"{self.role} expected qweight_layout=packed_k_major_u32, "
                f"got {self.qweight_layout}"
            )
        if self.metadata_layout != "group_major_f16":
            raise ValueError(
                f"{self.role} expected metadata_layout=group_major_f16, "
                f"got {self.metadata_layout}"
            )
        if self.layer_index is not None and not 0 <= self.layer_index < QWEN36_27B_PROFILE.num_hidden_layers:
            raise ValueError(f"layer_index out of range: {self.layer_index}")


@dataclass(frozen=True)
class Qwen36ArtifactManifest:
    tensors: list[Int4TensorArtifact]
    schema_version: str = SCHEMA_VERSION
    model_id: str = MODEL_ID
    source_checkpoint: str = MODEL_ID
    notes: str = "Qwen3.6-27B dense fused-path artifact"

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.model_id != MODEL_ID:
            raise ValueError(f"unsupported model_id: {self.model_id}")
        for tensor in self.tensors:
            tensor.validate()


@dataclass(frozen=True)
class Qwen36ArtifactCoverage:
    kind: ManifestCoverageKind
    layers: int
    tensors: int

    @property
    def is_template(self) -> bool:
        return self.kind == "template"

    @property
    def is_full_layers(self) -> bool:
        return self.kind == "full_layers"


def write_manifest(manifest: Qwen36ArtifactManifest, path: str | Path) -> Path:
    manifest.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_manifest(path: str | Path) -> Qwen36ArtifactManifest:
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    tensors = [Int4TensorArtifact(**item) for item in payload.get("tensors", [])]
    manifest = Qwen36ArtifactManifest(
        tensors=tensors,
        schema_version=payload.get("schema_version", SCHEMA_VERSION),
        model_id=payload.get("model_id", MODEL_ID),
        source_checkpoint=payload.get("source_checkpoint", MODEL_ID),
        notes=payload.get("notes", ""),
    )
    manifest.validate()
    return manifest


def manifest_tensor_index(
    manifest: Qwen36ArtifactManifest,
) -> dict[ManifestTensorKey, Int4TensorArtifact]:
    """Index artifacts by ``(layer_index, role)`` and reject duplicates."""

    manifest.validate()
    index: dict[ManifestTensorKey, Int4TensorArtifact] = {}
    for tensor in manifest.tensors:
        key = (tensor.layer_index, tensor.role)
        if key in index:
            raise ValueError(
                f"manifest has duplicate tensor for layer={tensor.layer_index} "
                f"role={tensor.role}"
            )
        index[key] = tensor
    return index


def expected_roles_for_layer(layer_index: int) -> tuple[TensorRole, ...]:
    """Return the concrete tensor roles required by one Qwen3.6-27B layer."""

    if not 0 <= layer_index < QWEN36_27B_PROFILE.num_hidden_layers:
        raise ValueError(f"layer_index out of range: {layer_index}")
    attention_roles = (
        FULL_ATTENTION_TENSOR_ROLES
        if layer_index in QWEN36_27B_PROFILE.full_attention_layer_indices
        else LINEAR_ATTENTION_TENSOR_ROLES
    )
    return attention_roles + DENSE_MLP_TENSOR_ROLES


def tensors_for_layer(
    manifest: Qwen36ArtifactManifest,
    layer_index: int,
    roles: tuple[TensorRole, ...] | None = None,
) -> dict[TensorRole, Int4TensorArtifact]:
    """Return required role artifacts for one concrete layer index."""

    if not 0 <= layer_index < QWEN36_27B_PROFILE.num_hidden_layers:
        raise ValueError(f"layer_index out of range: {layer_index}")
    required_roles = expected_roles_for_layer(layer_index) if roles is None else roles
    index = manifest_tensor_index(manifest)
    result: dict[TensorRole, Int4TensorArtifact] = {}
    missing: list[TensorRole] = []
    for role in required_roles:
        tensor = index.get((layer_index, role))
        if tensor is None:
            missing.append(role)
        else:
            result[role] = tensor
    if missing:
        raise ValueError(f"manifest missing roles for layer {layer_index}: {missing}")
    return result


def validate_required_roles_for_layer(
    manifest: Qwen36ArtifactManifest,
    layer_index: int,
    roles: tuple[TensorRole, ...] | None = None,
) -> None:
    """Ensure a layered manifest has all required roles for one layer."""

    tensors_for_layer(manifest, layer_index, roles)


def validate_full_layer_coverage(manifest: Qwen36ArtifactManifest) -> None:
    """Ensure a manifest covers every concrete layer plus global tensors."""

    manifest.validate()
    manifest_tensor_index(manifest)

    template_layer_roles = sorted(
        tensor.role
        for tensor in manifest.tensors
        if tensor.layer_index is None and tensor.role in LAYER_LOCAL_TENSOR_ROLES
    )
    if template_layer_roles:
        raise ValueError(
            "full-layer manifest has template layer-local roles: "
            f"{template_layer_roles}"
        )

    for layer_index in range(QWEN36_27B_PROFILE.num_hidden_layers):
        expected_roles = set(expected_roles_for_layer(layer_index))
        actual_layer_roles = {
            tensor.role
            for tensor in manifest.tensors
            if tensor.layer_index == layer_index and tensor.role in LAYER_LOCAL_TENSOR_ROLES
        }
        unexpected = sorted(actual_layer_roles - expected_roles)
        if unexpected:
            raise ValueError(
                f"manifest has unexpected roles for layer {layer_index}: {unexpected}"
            )
        validate_required_roles_for_layer(manifest, layer_index)
    for role in GLOBAL_TENSOR_ROLES:
        count = sum(1 for tensor in manifest.tensors if tensor.role == role)
        if count == 0:
            raise ValueError(f"manifest missing global role: {role}")
        if count > 1:
            raise ValueError(f"manifest has duplicate global role: {role}")


def artifact_coverage(manifest: Qwen36ArtifactManifest) -> Qwen36ArtifactCoverage:
    """Classify whether a manifest is template-only or full-layer coverage."""

    try:
        validate_required_roles(manifest)
    except ValueError as template_exc:
        try:
            validate_full_layer_coverage(manifest)
        except ValueError as full_exc:
            raise ValueError(
                "manifest does not satisfy template or full-layer coverage: "
                f"template={template_exc}; full_layers={full_exc}"
            ) from full_exc
        return Qwen36ArtifactCoverage(
            kind="full_layers",
            layers=QWEN36_27B_PROFILE.num_hidden_layers,
            tensors=len(manifest.tensors),
        )
    return Qwen36ArtifactCoverage(
        kind="template",
        layers=1,
        tensors=len(manifest.tensors),
    )


def expected_tensor_shape(role: TensorRole) -> tuple[int, int]:
    return ROLE_SHAPES[role]


def validate_required_roles(
    manifest: Qwen36ArtifactManifest,
    roles: tuple[TensorRole, ...] = REQUIRED_TENSOR_ROLES,
) -> None:
    """Ensure a manifest has exactly one artifact for each required role."""

    manifest.validate()
    counts: dict[TensorRole, int] = {role: 0 for role in roles}
    for tensor in manifest.tensors:
        if tensor.role in counts:
            counts[tensor.role] += 1
    missing = [role for role, count in counts.items() if count == 0]
    duplicated = [role for role, count in counts.items() if count > 1]
    if missing:
        raise ValueError(f"manifest missing required roles: {missing}")
    if duplicated:
        raise ValueError(f"manifest has duplicate required roles: {duplicated}")
