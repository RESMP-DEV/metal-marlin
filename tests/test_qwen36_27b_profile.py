from __future__ import annotations

import json

import pytest

from metal_marlin.qwen36_27b_artifact import (
    DENSE_MLP_TENSOR_ROLES,
    FULL_ATTENTION_TENSOR_ROLES,
    LAYER_LOCAL_TENSOR_ROLES,
    LINEAR_ATTENTION_TENSOR_ROLES,
    REQUIRED_TENSOR_ROLES,
    Int4TensorArtifact,
    Qwen36ArtifactManifest,
    TensorRole,
    artifact_coverage,
    expected_roles_for_layer,
    expected_tensor_shape,
    manifest_tensor_index,
    read_manifest,
    tensors_for_layer,
    validate_required_roles_for_layer,
    write_manifest,
)
from metal_marlin.qwen36_27b_profile import (
    MODEL_ID,
    QWEN36_27B_PROFILE,
    is_qwen36_27b_config,
    profile_from_hf_config,
    shape_contract_payload,
)
from metal_marlin.serving.engine import _detect_model_format, _normalize_model_name


def _hf_config() -> dict:
    return {
        "_name_or_path": MODEL_ID,
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "intermediate_size": 17408,
            "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            * 16,
        },
    }


def test_profile_matches_dense_qwen36_27b_contract() -> None:
    profile = QWEN36_27B_PROFILE
    assert profile.hidden_size == 5120
    assert profile.num_hidden_layers == 64
    assert profile.num_linear_attention_layers == 48
    assert profile.num_full_attention_layers == 16
    assert profile.full_attention_layer_indices[:4] == [3, 7, 11, 15]
    assert profile.delta.v_features == 6144
    assert profile.dense_mlp.intermediate_size == 17408


def test_profile_from_hf_config_and_shape_payload() -> None:
    profile = profile_from_hf_config(_hf_config())
    assert profile == QWEN36_27B_PROFILE
    assert is_qwen36_27b_config(_hf_config(), MODEL_ID) is True

    payload = shape_contract_payload(profile)
    assert payload["model_label"] == "Qwen 3.6 27B"
    assert payload["mlp_kind"] == "dense"
    assert payload["weights_loaded"] is False
    assert payload["layer_types"].count("full_attention") == 16


def test_qwen36_27b_artifact_manifest_roundtrip(tmp_path) -> None:
    in_features, out_features = expected_tensor_shape("linear_attn_q")
    tensor = Int4TensorArtifact(
        role="linear_attn_q",
        layer_index=0,
        qweight="layer_0000/linear_attn_q.qweight",
        scales="layer_0000/linear_attn_q.scales",
        zeros="layer_0000/linear_attn_q.zeros",
        in_features=in_features,
        out_features=out_features,
        calibration={"method": "awq_diag_hessian"},
    )
    manifest = Qwen36ArtifactManifest(tensors=[tensor])
    path = write_manifest(manifest, tmp_path / "manifest.json")
    loaded = read_manifest(path)
    assert loaded == manifest


def _artifact(role: TensorRole, layer_index: int | None) -> Int4TensorArtifact:
    in_features, out_features = expected_tensor_shape(role)
    prefix = "global" if layer_index is None else f"layer_{layer_index:04d}"
    return Int4TensorArtifact(
        role=role,
        layer_index=layer_index,
        qweight=f"{prefix}/{role}.qweight",
        scales=f"{prefix}/{role}.scales",
        zeros=f"{prefix}/{role}.zeros",
        in_features=in_features,
        out_features=out_features,
    )


def test_layered_artifact_manifest_indexes_by_layer_and_role() -> None:
    manifest = Qwen36ArtifactManifest(
        tensors=[
            *(_artifact(role, 0) for role in LAYER_LOCAL_TENSOR_ROLES),
            *(_artifact(role, 3) for role in LAYER_LOCAL_TENSOR_ROLES),
            _artifact("lm_head", None),
        ]
    )

    layer0 = tensors_for_layer(manifest, 0)
    layer3 = tensors_for_layer(manifest, 3)
    validate_required_roles_for_layer(manifest, 0)
    validate_required_roles_for_layer(manifest, 3)

    assert set(layer0) == set(LINEAR_ATTENTION_TENSOR_ROLES + DENSE_MLP_TENSOR_ROLES)
    assert set(layer3) == set(FULL_ATTENTION_TENSOR_ROLES + DENSE_MLP_TENSOR_ROLES)
    assert layer0["linear_attn_q"].layer_index == 0
    assert layer3["full_attn_q"].layer_index == 3
    assert manifest_tensor_index(manifest)[(None, "lm_head")].role == "lm_head"


def test_expected_roles_follow_qwen36_hybrid_cadence() -> None:
    assert set(expected_roles_for_layer(0)) == set(
        LINEAR_ATTENTION_TENSOR_ROLES + DENSE_MLP_TENSOR_ROLES
    )
    assert set(expected_roles_for_layer(3)) == set(
        FULL_ATTENTION_TENSOR_ROLES + DENSE_MLP_TENSOR_ROLES
    )


def test_artifact_coverage_accepts_template_manifest() -> None:
    manifest = Qwen36ArtifactManifest(
        tensors=[
            _artifact(role, 0 if role != "lm_head" else None)
            for role in REQUIRED_TENSOR_ROLES
        ]
    )

    coverage = artifact_coverage(manifest)

    assert coverage.kind == "template"
    assert coverage.layers == 1
    assert coverage.tensors == len(REQUIRED_TENSOR_ROLES)


def test_artifact_coverage_accepts_full_layer_manifest() -> None:
    tensors: list[Int4TensorArtifact] = []
    for layer_index in range(QWEN36_27B_PROFILE.num_hidden_layers):
        tensors.extend(_artifact(role, layer_index) for role in expected_roles_for_layer(layer_index))
    tensors.append(_artifact("lm_head", None))
    manifest = Qwen36ArtifactManifest(tensors=tensors)

    coverage = artifact_coverage(manifest)

    assert coverage.kind == "full_layers"
    assert coverage.layers == QWEN36_27B_PROFILE.num_hidden_layers
    assert coverage.tensors == len(tensors)


def test_artifact_coverage_rejects_cadence_incompatible_layer_roles() -> None:
    tensors: list[Int4TensorArtifact] = []
    for layer_index in range(QWEN36_27B_PROFILE.num_hidden_layers):
        tensors.extend(_artifact(role, layer_index) for role in expected_roles_for_layer(layer_index))
    tensors.append(_artifact("linear_attn_q", 3))
    tensors.append(_artifact("lm_head", None))
    manifest = Qwen36ArtifactManifest(tensors=tensors)

    with pytest.raises(ValueError, match="unexpected roles for layer 3"):
        artifact_coverage(manifest)


def test_layered_artifact_manifest_rejects_duplicate_layer_role() -> None:
    manifest = Qwen36ArtifactManifest(
        tensors=[
            _artifact("linear_attn_q", 0),
            _artifact("linear_attn_q", 0),
        ]
    )

    with pytest.raises(ValueError, match="duplicate tensor"):
        manifest_tensor_index(manifest)


def test_layered_artifact_manifest_reports_missing_layer_role() -> None:
    manifest = Qwen36ArtifactManifest(tensors=[_artifact("linear_attn_q", 0)])

    with pytest.raises(ValueError, match="manifest missing roles for layer 0"):
        validate_required_roles_for_layer(manifest, 0)


def test_artifact_rejects_35b_shape_for_27b_role() -> None:
    tensor = Int4TensorArtifact(
        role="linear_attn_q",
        layer_index=0,
        qweight="q",
        scales="s",
        zeros="z",
        in_features=2048,
        out_features=2048,
    )
    with pytest.raises(ValueError, match="linear_attn_q expected shape"):
        tensor.validate()


def test_serving_detects_dense_qwen36_27b_mmfp4(tmp_path) -> None:
    model_dir = tmp_path / "Qwen3.6-27B-MMFP4"
    model_dir.mkdir()
    config = _hf_config()
    config["quantization_config"] = {"format": "mmfp4_e2m1_marlin"}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert _detect_model_format(str(model_dir)) == "mmfp4"
    assert _normalize_model_name(str(model_dir)) == "Qwen/Qwen3.6-27B"
