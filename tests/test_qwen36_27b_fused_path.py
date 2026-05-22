from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from metal_marlin import launch_tracing
from metal_marlin.kernels.qwen36_27b import (
    FEATURE_FLAG,
    QWEN36_27B_PROFILE,
    REQUIRED_KERNELS,
    PackedInt4Matrix,
    _ensure_matrix,
    decide_runtime_path,
    dispatch_argmax,
    dispatch_attention_cache_write,
    dispatch_attention_decode,
    dispatch_deltanet_interval4,
    dispatch_deltanet_update,
    dispatch_dense_down_residual,
    dispatch_dense_gate_up_silu,
    dispatch_linear_attention,
    dispatch_linear_rmsnorm_gated,
    dispatch_qkvb_projection,
    dispatch_rmsnorm_hidden,
    missing_kernel_symbols,
)
from metal_marlin.metallib_loader import get_staleness_details
from metal_marlin.qwen36_27b_artifact import expected_tensor_shape
from metal_marlin.qwen36_27b_profile import MODEL_ID

REPO_ROOT = Path(__file__).resolve().parent.parent
SHADER_PATH = REPO_ROOT / "src" / "qwen36_27b_decode.metal"


def _config() -> dict:
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
        },
    }


def _require_fresh_mps_metallib() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    staleness = get_staleness_details()
    if staleness["is_stale"]:
        pytest.skip(f"metallib is stale: {staleness['reason']}")


def _filled_int4_shape(
    in_features: int,
    out_features: int,
    nibble: int,
    scale: float = 1.0,
) -> PackedInt4Matrix:
    profile = QWEN36_27B_PROFILE
    packed_word = sum((nibble & 0xF) << (4 * i) for i in range(8))
    return PackedInt4Matrix(
        qweight=torch.full(
            (in_features // 8, out_features),
            packed_word,
            dtype=torch.uint32,
            device="mps",
        ),
        scales=torch.full(
            (in_features // profile.group_size, out_features),
            scale,
            dtype=torch.float16,
            device="mps",
        ),
        zeros=torch.zeros(
            (in_features // profile.group_size, out_features),
            dtype=torch.float16,
            device="mps",
        ),
        in_features=in_features,
        out_features=out_features,
    )


def _filled_int4_matrix(out_features: int, nibble: int, scale: float = 1.0) -> PackedInt4Matrix:
    return _filled_int4_shape(QWEN36_27B_PROFILE.hidden_size, out_features, nibble, scale)


def _deltanet_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    profile = QWEN36_27B_PROFILE
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    next_state = state.float().clone().reshape(
        profile.delta.value_heads,
        profile.delta.key_dim,
        profile.delta.value_dim,
    )
    y = torch.empty(profile.delta.v_features, dtype=torch.float32)
    for value_head in range(profile.delta.value_heads):
        key_head = value_head // 3
        q_head = q[key_head * profile.delta.key_dim : (key_head + 1) * profile.delta.key_dim]
        k_head = k[key_head * profile.delta.key_dim : (key_head + 1) * profile.delta.key_dim]
        v_head = v[
            value_head * profile.delta.value_dim : (value_head + 1) * profile.delta.value_dim
        ]
        state_head = next_state[value_head]
        prediction = torch.matmul(k_head, state_head)
        delta = v_head - prediction
        state_head += beta[value_head] * torch.outer(k_head, delta)
        y_head = torch.matmul(q_head, state_head)
        y[
            value_head * profile.delta.value_dim : (value_head + 1) * profile.delta.value_dim
        ] = y_head
    return y, next_state.reshape(-1)


def test_feature_flag_runtime_decision() -> None:
    disabled = decide_runtime_path(_config(), env={})
    assert disabled.enabled is False
    assert FEATURE_FLAG in disabled.reason

    enabled = decide_runtime_path(_config(), env={FEATURE_FLAG: "1"})
    assert enabled.enabled is True
    assert "fused path selected" in enabled.reason


def test_runtime_decision_rejects_35b_a3b_shape() -> None:
    config = _config()
    config["_name_or_path"] = "Qwen/Qwen3.6-35B-A3B"
    config["text_config"]["hidden_size"] = 2048
    config["text_config"]["num_hidden_layers"] = 40
    decision = decide_runtime_path(config, env={FEATURE_FLAG: "1"})
    assert decision.enabled is False
    assert decision.reason == "config is not dense Qwen3.6-27B"


def test_matrix_validation_uses_profile_shapes() -> None:
    in_features, out_features = expected_tensor_shape("linear_attn_beta")
    matrix = PackedInt4Matrix(
        qweight=torch.zeros((in_features // 8, out_features), dtype=torch.int32),
        scales=torch.ones((in_features // 128, out_features), dtype=torch.float16),
        zeros=torch.zeros((in_features // 128, out_features), dtype=torch.float16),
        in_features=in_features,
        out_features=out_features,
    )
    checked = _ensure_matrix(matrix, "linear_attn_beta")
    assert checked.role == "linear_attn_beta"


def test_matrix_validation_rejects_bad_metadata_shape() -> None:
    in_features, out_features = expected_tensor_shape("linear_attn_beta")
    matrix = PackedInt4Matrix(
        qweight=torch.zeros((in_features // 8, out_features), dtype=torch.int32),
        scales=torch.ones((1, out_features), dtype=torch.float16),
        zeros=torch.zeros((in_features // 128, out_features), dtype=torch.float16),
        in_features=in_features,
        out_features=out_features,
    )
    with pytest.raises(ValueError, match="linear_attn_beta.scales"):
        _ensure_matrix(matrix, "linear_attn_beta")


def test_shader_contains_required_kernel_symbols() -> None:
    source = SHADER_PATH.read_text(encoding="utf-8")
    found = set(re.findall(r"kernel\s+void\s+(\w+)\s*\(", source))
    assert set(REQUIRED_KERNELS) <= found


def test_launch_budget_records_two_dispatches_for_linear_attention_block() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    hidden = torch.ones((profile.hidden_size,), dtype=torch.float16, device="mps")
    state = torch.zeros((profile.delta.state_elements,), dtype=torch.float16, device="mps")
    scale = 1.0 / profile.hidden_size

    launch_tracing.enable_for_testing()
    launch_tracing.reset()
    y = dispatch_linear_attention(
        hidden,
        _filled_int4_matrix(profile.delta.q_features, 1, scale),
        _filled_int4_matrix(profile.delta.k_features, 1, scale),
        _filled_int4_matrix(profile.delta.v_features, 1, scale),
        _filled_int4_matrix(profile.delta.beta_features, 1, scale),
        state,
    )

    assert launch_tracing.dispatch_count() == 2
    assert launch_tracing.kernel_names() == [
        "qwen36_27b_int4_qkvb",
        "qwen36_27b_deltanet_update",
    ]
    assert torch.allclose(y.cpu().float(), torch.full((profile.delta.v_features,), 128.0))
    assert torch.allclose(state.cpu().float(), torch.ones((profile.delta.state_elements,)))
    launch_tracing.reset()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_metallib_has_no_missing_qwen36_symbols_when_built() -> None:
    _require_fresh_mps_metallib()
    missing = missing_kernel_symbols()
    assert missing == set()


def test_argmax_dispatch_matches_torch_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    logits = torch.full((profile.vocab_size,), -12.0, dtype=torch.float16, device="mps")
    logits[12345] = 7.0
    logits[54321] = 8.0

    token = dispatch_argmax(logits)

    assert token.cpu().item() == torch.argmax(logits.cpu()).item()


def test_qkvb_projection_dispatch_matches_uniform_int4_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    hidden = torch.ones((profile.hidden_size,), dtype=torch.float16, device="mps")
    q = _filled_int4_matrix(profile.delta.q_features, 1)
    k = _filled_int4_matrix(profile.delta.k_features, 1)
    v = _filled_int4_matrix(profile.delta.v_features, 1)
    beta = _filled_int4_matrix(profile.delta.beta_features, 1)

    out = dispatch_qkvb_projection(hidden, q, k, v, beta)

    expected = torch.full((), float(profile.hidden_size), dtype=torch.float32)
    assert torch.allclose(out.q.cpu().float(), expected.expand(profile.delta.q_features))
    assert torch.allclose(out.k.cpu().float(), expected.expand(profile.delta.k_features))
    assert torch.allclose(out.v.cpu().float(), expected.expand(profile.delta.v_features))
    assert torch.allclose(out.beta.cpu().float(), expected.expand(profile.delta.beta_features))


def test_deltanet_update_dispatch_matches_cpu_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    q = torch.linspace(-0.25, 0.25, profile.delta.q_features, dtype=torch.float32)
    k = torch.linspace(0.1, -0.1, profile.delta.k_features, dtype=torch.float32)
    v = torch.linspace(-0.2, 0.2, profile.delta.v_features, dtype=torch.float32)
    beta = torch.linspace(0.01, 0.03, profile.delta.beta_features, dtype=torch.float32)
    state = torch.zeros(profile.delta.state_elements, dtype=torch.float32)
    expected_y, expected_state = _deltanet_reference(q, k, v, beta, state)
    state_mps = state.to(dtype=torch.float16, device="mps")

    actual_y = dispatch_deltanet_update(
        q.to(dtype=torch.float16, device="mps"),
        k.to(dtype=torch.float16, device="mps"),
        v.to(dtype=torch.float16, device="mps"),
        beta.to(dtype=torch.float16, device="mps"),
        state_mps,
    )

    torch.testing.assert_close(
        actual_y.cpu().float(),
        expected_y.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=5e-4,
    )
    torch.testing.assert_close(
        state_mps.cpu().float(),
        expected_state.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=7e-4,
    )


def test_deltanet_interval4_dispatch_matches_cpu_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    layers = 3
    q = torch.linspace(-0.1, 0.1, layers * profile.delta.q_features, dtype=torch.float32)
    k = torch.linspace(0.08, -0.08, layers * profile.delta.k_features, dtype=torch.float32)
    v = torch.linspace(-0.05, 0.05, layers * profile.delta.v_features, dtype=torch.float32)
    beta = torch.linspace(0.01, 0.02, layers * profile.delta.beta_features, dtype=torch.float32)
    state = torch.zeros(layers * profile.delta.state_elements, dtype=torch.float32)
    expected_y = []
    expected_state = []
    for layer in range(layers):
        y_layer, state_layer = _deltanet_reference(
            q[layer * profile.delta.q_features : (layer + 1) * profile.delta.q_features],
            k[layer * profile.delta.k_features : (layer + 1) * profile.delta.k_features],
            v[layer * profile.delta.v_features : (layer + 1) * profile.delta.v_features],
            beta[layer * profile.delta.beta_features : (layer + 1) * profile.delta.beta_features],
            state[
                layer * profile.delta.state_elements : (layer + 1) * profile.delta.state_elements
            ],
        )
        expected_y.append(y_layer)
        expected_state.append(state_layer)
    expected_y_tensor = torch.cat(expected_y)
    expected_state_tensor = torch.cat(expected_state)
    state_mps = state.to(dtype=torch.float16, device="mps")

    actual_y = dispatch_deltanet_interval4(
        q.to(dtype=torch.float16, device="mps"),
        k.to(dtype=torch.float16, device="mps"),
        v.to(dtype=torch.float16, device="mps"),
        beta.to(dtype=torch.float16, device="mps"),
        state_mps,
    )

    torch.testing.assert_close(
        actual_y.cpu().float(),
        expected_y_tensor.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=5e-4,
    )
    torch.testing.assert_close(
        state_mps.cpu().float(),
        expected_state_tensor.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=7e-4,
    )


def test_rmsnorm_dispatch_matches_cpu_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    hidden = torch.linspace(-1.0, 1.0, profile.hidden_size, dtype=torch.float32)
    weight = torch.linspace(0.75, 1.25, profile.hidden_size, dtype=torch.float32)
    expected = hidden * torch.rsqrt(hidden.square().mean() + profile.rms_norm_eps) * weight

    actual = dispatch_rmsnorm_hidden(
        hidden.to(dtype=torch.float16, device="mps"),
        weight.to(dtype=torch.float16, device="mps"),
    )

    torch.testing.assert_close(
        actual.cpu().float(),
        expected.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=7e-4,
    )


def test_linear_gated_rmsnorm_dispatch_matches_cpu_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    x = torch.linspace(-0.5, 0.5, profile.delta.v_features, dtype=torch.float32)
    gate = torch.linspace(-0.2, 0.2, profile.delta.v_features, dtype=torch.float32)
    weight = torch.linspace(0.8, 1.2, profile.delta.value_dim, dtype=torch.float32)
    x_heads = x.reshape(profile.delta.value_heads, profile.delta.value_dim)
    gate_heads = gate.reshape(profile.delta.value_heads, profile.delta.value_dim)
    inv_rms = torch.rsqrt(x_heads.square().mean(dim=1, keepdim=True) + profile.rms_norm_eps)
    expected = (x_heads * inv_rms * weight.reshape(1, -1) * torch.nn.functional.silu(gate_heads)).reshape(-1)

    actual = dispatch_linear_rmsnorm_gated(
        x.to(dtype=torch.float16, device="mps"),
        gate.to(dtype=torch.float16, device="mps"),
        weight.to(dtype=torch.float16, device="mps"),
    )

    torch.testing.assert_close(
        actual.cpu().float(),
        expected.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=7e-4,
    )


def test_dense_mlp_dispatch_matches_uniform_int4_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    hidden = torch.ones((profile.hidden_size,), dtype=torch.float16, device="mps")
    gate = _filled_int4_shape(
        profile.hidden_size,
        profile.dense_mlp.intermediate_size,
        1,
        1.0 / profile.hidden_size,
    )
    up = _filled_int4_shape(
        profile.hidden_size,
        profile.dense_mlp.intermediate_size,
        1,
        1.0 / profile.hidden_size,
    )
    down = _filled_int4_shape(
        profile.dense_mlp.intermediate_size,
        profile.hidden_size,
        1,
        1.0 / profile.dense_mlp.intermediate_size,
    )
    expected_intermediate = torch.full(
        (profile.dense_mlp.intermediate_size,),
        float(torch.nn.functional.silu(torch.tensor(1.0))),
        dtype=torch.float32,
    )

    intermediate = dispatch_dense_gate_up_silu(hidden, gate, up)
    out = dispatch_dense_down_residual(intermediate, down, hidden)

    torch.testing.assert_close(
        intermediate.cpu().float(),
        expected_intermediate.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=7e-4,
    )
    torch.testing.assert_close(
        out.cpu().float(),
        torch.full((profile.hidden_size,), 1.0 + expected_intermediate[0].item()),
        rtol=5e-3,
        atol=7e-4,
    )


def test_attention_cache_write_dispatch_mutates_cache_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    token_position = 1
    k = torch.arange(profile.attention.kv_features, dtype=torch.float16, device="mps")
    v = k + 2
    k_cache = torch.zeros(3 * profile.attention.kv_features, dtype=torch.float16, device="mps")
    v_cache = torch.zeros_like(k_cache)

    dispatch_attention_cache_write(k, v, k_cache, v_cache, token_position)

    start = token_position * profile.attention.kv_features
    stop = start + profile.attention.kv_features
    torch.testing.assert_close(k_cache[start:stop].cpu(), k.cpu())
    torch.testing.assert_close(v_cache[start:stop].cpu(), v.cpu())
    assert torch.count_nonzero(k_cache[:start]).cpu().item() == 0
    assert torch.count_nonzero(v_cache[:start]).cpu().item() == 0


def test_attention_decode_token0_dispatch_matches_reference_on_mps() -> None:
    _require_fresh_mps_metallib()
    profile = QWEN36_27B_PROFILE
    q_proj = torch.zeros(profile.attention.q_features, dtype=torch.float16, device="mps")
    q_proj[profile.attention.o_features :] = 0.5
    k_proj = torch.linspace(-0.2, 0.2, profile.attention.kv_features, dtype=torch.float32)
    v_proj = torch.linspace(-1.0, 1.0, profile.attention.kv_features, dtype=torch.float32)
    q_norm_weight = torch.ones(profile.attention.head_dim, dtype=torch.float16, device="mps")
    k_norm_weight = torch.linspace(0.9, 1.1, profile.attention.head_dim, dtype=torch.float32)
    k_cache = torch.zeros(2 * profile.attention.kv_features, dtype=torch.float16, device="mps")
    v_cache = torch.zeros_like(k_cache)
    gate = torch.sigmoid(torch.tensor(0.5, dtype=torch.float32))
    expected = torch.empty(profile.attention.o_features, dtype=torch.float32)
    v_heads = v_proj.reshape(profile.attention.kv_heads, profile.attention.head_dim)
    for head in range(profile.attention.heads):
        kv_head = head // (profile.attention.heads // profile.attention.kv_heads)
        start = head * profile.attention.head_dim
        expected[start : start + profile.attention.head_dim] = v_heads[kv_head] * gate
    k_heads = k_proj.reshape(profile.attention.kv_heads, profile.attention.head_dim)
    k_inv_rms = torch.rsqrt(k_heads.square().mean(dim=1, keepdim=True) + profile.rms_norm_eps)
    expected_k_cache = (k_heads * k_inv_rms * k_norm_weight.reshape(1, -1)).reshape(-1)

    out = dispatch_attention_decode(
        q_proj,
        k_proj.to(dtype=torch.float16, device="mps"),
        v_proj.to(dtype=torch.float16, device="mps"),
        q_norm_weight,
        k_norm_weight.to(dtype=torch.float16, device="mps"),
        k_cache,
        v_cache,
        token_position=0,
    )

    torch.testing.assert_close(
        out.cpu().float(),
        expected.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=7e-4,
    )
    torch.testing.assert_close(
        k_cache[: profile.attention.kv_features].cpu().float(),
        expected_k_cache.to(dtype=torch.float16).float(),
        rtol=5e-3,
        atol=7e-4,
    )
    torch.testing.assert_close(v_cache[: profile.attention.kv_features].cpu(), v_proj.half())
