from __future__ import annotations

import json

import pytest

from metal_marlin.qwen36_27b_artifact import (
    Int4TensorArtifact,
    Qwen36ArtifactManifest,
    expected_tensor_shape,
    read_manifest,
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

