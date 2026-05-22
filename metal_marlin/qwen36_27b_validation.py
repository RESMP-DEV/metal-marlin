"""Optional local-weight validation for Qwen3.6-27B fused artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .kernels.qwen36_27b import (
    PackedInt4Matrix,
    dispatch_attention_o_residual,
    dispatch_attention_qkv_projection,
    dispatch_dense_down_residual,
    dispatch_dense_gate_up_silu,
    dispatch_qkvb_projection,
    load_packed_int4_matrix,
)
from .metallib_loader import get_staleness_details
from .qwen36_27b_artifact import Int4TensorArtifact, TensorRole, read_manifest
from .qwen36_27b_profile import QWEN36_27B_PROFILE

LINEAR_QKVB_ROLES: tuple[TensorRole, ...] = (
    "linear_attn_q",
    "linear_attn_k",
    "linear_attn_v",
    "linear_attn_beta",
)
DENSE_MLP_ROLES: tuple[TensorRole, ...] = (
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)
FULL_ATTENTION_ROLES: tuple[TensorRole, ...] = (
    "full_attn_q",
    "full_attn_k",
    "full_attn_v",
    "full_attn_o",
)


@dataclass(frozen=True)
class ArtifactParityResult:
    role: TensorRole
    checked_columns: tuple[int, ...]
    max_abs_error: float
    max_rel_error: float


@dataclass(frozen=True)
class Layer0ProjectionValidation:
    manifest_path: Path
    hidden_l2: float
    results: tuple[ArtifactParityResult, ...]

    @property
    def max_abs_error(self) -> float:
        return max((result.max_abs_error for result in self.results), default=0.0)

    @property
    def max_rel_error(self) -> float:
        return max((result.max_rel_error for result in self.results), default=0.0)

    def assert_within(self, *, atol: float, rtol: float) -> None:
        for result in self.results:
            if result.max_abs_error > atol and result.max_rel_error > rtol:
                raise AssertionError(
                    f"{result.role} parity failed: "
                    f"max_abs_error={result.max_abs_error:.6g} > {atol}, "
                    f"max_rel_error={result.max_rel_error:.6g} > {rtol}"
                )


@dataclass(frozen=True)
class Layer0MlpValidation:
    manifest_path: Path
    hidden_l2: float
    intermediate_l2: float
    gate_up: ArtifactParityResult
    down: ArtifactParityResult

    @property
    def max_abs_error(self) -> float:
        return max(self.gate_up.max_abs_error, self.down.max_abs_error)

    @property
    def max_rel_error(self) -> float:
        return max(self.gate_up.max_rel_error, self.down.max_rel_error)

    @property
    def results(self) -> tuple[ArtifactParityResult, ...]:
        return (self.gate_up, self.down)

    def assert_within(self, *, atol: float, rtol: float) -> None:
        for result in self.results:
            if result.max_abs_error > atol and result.max_rel_error > rtol:
                raise AssertionError(
                    f"{result.role} parity failed: "
                    f"max_abs_error={result.max_abs_error:.6g} > {atol}, "
                    f"max_rel_error={result.max_rel_error:.6g} > {rtol}"
                )


@dataclass(frozen=True)
class Layer0FullAttentionValidation:
    manifest_path: Path
    hidden_l2: float
    attn_out_l2: float
    results: tuple[ArtifactParityResult, ...]

    @property
    def max_abs_error(self) -> float:
        return max((result.max_abs_error for result in self.results), default=0.0)

    @property
    def max_rel_error(self) -> float:
        return max((result.max_rel_error for result in self.results), default=0.0)

    def assert_within(self, *, atol: float, rtol: float) -> None:
        for result in self.results:
            if result.max_abs_error > atol and result.max_rel_error > rtol:
                raise AssertionError(
                    f"{result.role} parity failed: "
                    f"max_abs_error={result.max_abs_error:.6g} > {atol}, "
                    f"max_rel_error={result.max_rel_error:.6g} > {rtol}"
                )


def default_layer0_manifest_path() -> Path:
    return (
        Path("agent_workspace")
        / "qwen36_27b"
        / "artifacts"
        / "prototype_layer0"
        / "manifest.json"
    )


def default_block_layer0_manifest_path() -> Path:
    return (
        Path("agent_workspace")
        / "qwen36_27b"
        / "artifacts"
        / "prototype_block_layer0"
        / "manifest.json"
    )


def _artifact_by_role(
    manifest_path: Path,
    roles: tuple[TensorRole, ...],
    description: str,
) -> dict[TensorRole, Int4TensorArtifact]:
    manifest = read_manifest(manifest_path)
    by_role: dict[TensorRole, Int4TensorArtifact] = {}
    for tensor in manifest.tensors:
        if tensor.role in roles:
            by_role[tensor.role] = tensor
    missing = [role for role in roles if role not in by_role]
    if missing:
        raise ValueError(f"manifest missing {description} roles: {missing}")
    return by_role


def _sample_columns(out_features: int, max_columns: int) -> tuple[int, ...]:
    if max_columns <= 0:
        raise ValueError("max_columns must be positive")
    if out_features <= max_columns:
        return tuple(range(out_features))
    anchors = {0, out_features - 1, out_features // 2}
    step = max(1, out_features // max_columns)
    anchors.update(range(0, out_features, step))
    return tuple(sorted(anchors)[:max_columns])


def _reference_columns(
    hidden: torch.Tensor,
    matrix: PackedInt4Matrix,
    columns: tuple[int, ...],
) -> torch.Tensor:
    qweight = matrix.qweight.cpu().to(torch.uint32)
    scales = matrix.scales.cpu().float()
    zeros = matrix.zeros.cpu().float()
    hidden = hidden.cpu().float()
    group_size = matrix.group_size
    values: list[float] = []
    for column in columns:
        acc = 0.0
        for packed_k in range(qweight.shape[0]):
            packed = int(qweight[packed_k, column].item())
            base_k = packed_k * 8
            for lane in range(8):
                k = base_k + lane
                group = k // group_size
                nibble = (packed >> (lane * 4)) & 0xF
                weight = (float(nibble) - float(zeros[group, column])) * float(
                    scales[group, column]
                )
                acc += float(hidden[k]) * weight
        values.append(acc)
    return torch.tensor(values, dtype=torch.float32)


def _compare_role(
    role: TensorRole,
    hidden: torch.Tensor,
    matrix: PackedInt4Matrix,
    actual: torch.Tensor,
    *,
    max_columns: int,
) -> ArtifactParityResult:
    columns = _sample_columns(matrix.out_features, max_columns)
    expected = _reference_columns(hidden, matrix, columns)
    actual_columns = actual.cpu().float()[list(columns)]
    abs_error = torch.abs(actual_columns - expected)
    rel_error = abs_error / torch.clamp(torch.abs(expected), min=1e-6)
    return ArtifactParityResult(
        role=role,
        checked_columns=columns,
        max_abs_error=float(abs_error.max().item()) if abs_error.numel() else 0.0,
        max_rel_error=float(rel_error.max().item()) if rel_error.numel() else 0.0,
    )


def _compare_expected_columns(
    role: TensorRole,
    columns: tuple[int, ...],
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> ArtifactParityResult:
    actual_columns = actual.cpu().float()[list(columns)]
    expected = expected.cpu().float()
    abs_error = torch.abs(actual_columns - expected)
    rel_error = abs_error / torch.clamp(torch.abs(expected), min=1e-6)
    return ArtifactParityResult(
        role=role,
        checked_columns=columns,
        max_abs_error=float(abs_error.max().item()) if abs_error.numel() else 0.0,
        max_rel_error=float(rel_error.max().item()) if rel_error.numel() else 0.0,
    )


def validate_layer0_qkvb_artifacts(
    manifest_path: str | Path | None = None,
    *,
    max_columns_per_role: int = 8,
) -> Layer0ProjectionValidation:
    """Validate present layer-0 qmi4 artifacts against the fused projection kernel."""

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for fused Qwen3.6-27B artifact validation")
    staleness = get_staleness_details()
    if staleness["is_stale"]:
        raise RuntimeError(f"metallib is stale: {staleness['reason']}")

    path = Path(manifest_path) if manifest_path is not None else default_layer0_manifest_path()
    if not path.exists():
        raise FileNotFoundError(path)
    base_dir = path.parent
    by_role = _artifact_by_role(path, LINEAR_QKVB_ROLES, "layer-0 projection")
    matrices = {
        role: load_packed_int4_matrix(artifact, base_dir, device="mps")
        for role, artifact in by_role.items()
    }

    profile = QWEN36_27B_PROFILE
    hidden_cpu = torch.linspace(
        -0.75,
        0.75,
        profile.hidden_size,
        dtype=torch.float32,
    )
    hidden = hidden_cpu.to(dtype=torch.float16, device="mps")
    projected = dispatch_qkvb_projection(
        hidden,
        matrices["linear_attn_q"],
        matrices["linear_attn_k"],
        matrices["linear_attn_v"],
        matrices["linear_attn_beta"],
    )
    actual_by_role = {
        "linear_attn_q": projected.q,
        "linear_attn_k": projected.k,
        "linear_attn_v": projected.v,
        "linear_attn_beta": projected.beta,
    }
    results = tuple(
        _compare_role(
            role,
            hidden_cpu.to(dtype=torch.float16),
            matrices[role],
            actual_by_role[role],
            max_columns=max_columns_per_role,
        )
        for role in LINEAR_QKVB_ROLES
    )
    return Layer0ProjectionValidation(
        manifest_path=path,
        hidden_l2=float(torch.linalg.vector_norm(hidden_cpu).item()),
        results=results,
    )


def validate_layer0_mlp_artifacts(
    manifest_path: str | Path | None = None,
    *,
    max_columns_per_role: int = 8,
) -> Layer0MlpValidation:
    """Validate present layer-0 dense MLP qmi4 artifacts against fused kernels."""

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for fused Qwen3.6-27B artifact validation")
    staleness = get_staleness_details()
    if staleness["is_stale"]:
        raise RuntimeError(f"metallib is stale: {staleness['reason']}")

    path = (
        Path(manifest_path)
        if manifest_path is not None
        else default_block_layer0_manifest_path()
    )
    if not path.exists():
        raise FileNotFoundError(path)
    base_dir = path.parent
    by_role = _artifact_by_role(path, DENSE_MLP_ROLES, "layer-0 dense MLP")
    matrices = {
        role: load_packed_int4_matrix(artifact, base_dir, device="mps")
        for role, artifact in by_role.items()
    }

    profile = QWEN36_27B_PROFILE
    hidden_cpu = torch.linspace(
        -0.75,
        0.75,
        profile.hidden_size,
        dtype=torch.float32,
    )
    hidden = hidden_cpu.to(dtype=torch.float16, device="mps")
    intermediate = dispatch_dense_gate_up_silu(
        hidden,
        matrices["mlp_gate"],
        matrices["mlp_up"],
    )
    out = dispatch_dense_down_residual(
        intermediate,
        matrices["mlp_down"],
        hidden,
    )

    gate_up_columns = _sample_columns(
        profile.dense_mlp.intermediate_size,
        max_columns_per_role,
    )
    hidden_reference = hidden_cpu.to(dtype=torch.float16)
    gate_expected = _reference_columns(
        hidden_reference,
        matrices["mlp_gate"],
        gate_up_columns,
    )
    up_expected = _reference_columns(
        hidden_reference,
        matrices["mlp_up"],
        gate_up_columns,
    )
    gate_up_expected = torch.nn.functional.silu(gate_expected) * up_expected
    gate_up_result = _compare_expected_columns(
        "mlp_gate",
        gate_up_columns,
        gate_up_expected.to(dtype=torch.float16).float(),
        intermediate,
    )

    down_columns = _sample_columns(profile.hidden_size, max_columns_per_role)
    intermediate_reference = intermediate.cpu().to(dtype=torch.float16)
    down_expected = _reference_columns(
        intermediate_reference,
        matrices["mlp_down"],
        down_columns,
    )
    out_expected = hidden_reference.float()[list(down_columns)] + down_expected
    down_result = _compare_expected_columns(
        "mlp_down",
        down_columns,
        out_expected.to(dtype=torch.float16).float(),
        out,
    )

    return Layer0MlpValidation(
        manifest_path=path,
        hidden_l2=float(torch.linalg.vector_norm(hidden_cpu).item()),
        intermediate_l2=float(
            torch.linalg.vector_norm(intermediate.cpu().float()).item()
        ),
        gate_up=gate_up_result,
        down=down_result,
    )


def validate_layer0_full_attention_artifacts(
    manifest_path: str | Path | None = None,
    *,
    max_columns_per_role: int = 8,
) -> Layer0FullAttentionValidation:
    """Validate layer-0 full-attention qmi4 artifacts against fused wrappers."""

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for fused Qwen3.6-27B artifact validation")
    staleness = get_staleness_details()
    if staleness["is_stale"]:
        raise RuntimeError(f"metallib is stale: {staleness['reason']}")

    path = (
        Path(manifest_path)
        if manifest_path is not None
        else default_block_layer0_manifest_path()
    )
    if not path.exists():
        raise FileNotFoundError(path)
    base_dir = path.parent
    by_role = _artifact_by_role(path, FULL_ATTENTION_ROLES, "layer-0 full attention")
    matrices = {
        role: load_packed_int4_matrix(artifact, base_dir, device="mps")
        for role, artifact in by_role.items()
    }

    profile = QWEN36_27B_PROFILE
    hidden_cpu = torch.linspace(
        -0.75,
        0.75,
        profile.hidden_size,
        dtype=torch.float32,
    )
    hidden = hidden_cpu.to(dtype=torch.float16, device="mps")
    projected = dispatch_attention_qkv_projection(
        hidden,
        matrices["full_attn_q"],
        matrices["full_attn_k"],
        matrices["full_attn_v"],
    )
    actual_by_role = {
        "full_attn_q": projected.q,
        "full_attn_k": projected.k,
        "full_attn_v": projected.v,
    }
    hidden_reference = hidden_cpu.to(dtype=torch.float16)
    projection_results = tuple(
        _compare_role(
            role,
            hidden_reference,
            matrices[role],
            actual_by_role[role],
            max_columns=max_columns_per_role,
        )
        for role in FULL_ATTENTION_ROLES[:3]
    )

    attn_out_cpu = torch.linspace(
        -0.5,
        0.5,
        profile.attention.o_features,
        dtype=torch.float32,
    )
    attn_out = attn_out_cpu.to(dtype=torch.float16, device="mps")
    out = dispatch_attention_o_residual(
        attn_out,
        matrices["full_attn_o"],
        hidden,
    )
    out_columns = _sample_columns(profile.hidden_size, max_columns_per_role)
    out_expected = hidden_reference.float()[list(out_columns)] + _reference_columns(
        attn_out_cpu.to(dtype=torch.float16),
        matrices["full_attn_o"],
        out_columns,
    )
    out_result = _compare_expected_columns(
        "full_attn_o",
        out_columns,
        out_expected.to(dtype=torch.float16).float(),
        out,
    )

    return Layer0FullAttentionValidation(
        manifest_path=path,
        hidden_l2=float(torch.linalg.vector_norm(hidden_cpu).item()),
        attn_out_l2=float(torch.linalg.vector_norm(attn_out_cpu).item()),
        results=projection_results + (out_result,),
    )
