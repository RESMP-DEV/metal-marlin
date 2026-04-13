import pytest
import torch

import metal_marlin.fused_attention_mps as fused_attention_mps


@pytest.fixture
def attention_inputs():
    batch = 1
    num_heads_q = 8
    num_heads_kv = 2
    seq_q = 32
    seq_k = 64
    head_dim = 64

    # CPU tensors are enough here because dispatch is mocked.
    q = torch.randn(batch, num_heads_q, seq_q, head_dim, dtype=torch.float16)
    k = torch.randn(batch, num_heads_kv, seq_k, head_dim, dtype=torch.float16)
    v = torch.randn(batch, num_heads_kv, seq_k, head_dim, dtype=torch.float16)
    return q, k, v


def _shape_result(q: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(q)


def test_default_attention_prefers_v3(attention_inputs, monkeypatch):
    q, k, v = attention_inputs
    calls: list[str] = []

    monkeypatch.delenv("METAL_MARLIN_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_v3_attention",
        lambda *args, **kwargs: calls.append("v3") or _shape_result(q),
    )
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_v2_attention",
        lambda *args, **kwargs: calls.append("v2") or _shape_result(q),
    )
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_plain_attention",
        lambda *args, **kwargs: calls.append("plain") or _shape_result(q),
    )

    output = fused_attention_mps.fused_attention(q, k, v, causal=True)

    assert calls == ["v3"]
    assert output.shape == q.shape


def test_attention_backend_override_v2_skips_v3(attention_inputs, monkeypatch):
    q, k, v = attention_inputs
    calls: list[str] = []

    monkeypatch.setenv("METAL_MARLIN_ATTENTION_BACKEND", "v2")
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_v3_attention",
        lambda *args, **kwargs: calls.append("v3") or _shape_result(q),
    )
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_v2_attention",
        lambda *args, **kwargs: calls.append("v2") or _shape_result(q),
    )
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_plain_attention",
        lambda *args, **kwargs: calls.append("plain") or _shape_result(q),
    )

    output = fused_attention_mps.fused_attention(q, k, v, causal=True)

    assert calls == ["v2"]
    assert output.shape == q.shape


def test_attention_backend_override_plain_uses_plain_only(attention_inputs, monkeypatch):
    q, k, v = attention_inputs
    calls: list[str] = []

    monkeypatch.setenv("METAL_MARLIN_ATTENTION_BACKEND", "plain")
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_v3_attention",
        lambda *args, **kwargs: calls.append("v3") or _shape_result(q),
    )
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_v2_attention",
        lambda *args, **kwargs: calls.append("v2") or _shape_result(q),
    )
    monkeypatch.setattr(
        fused_attention_mps,
        "_run_plain_attention",
        lambda *args, **kwargs: calls.append("plain") or _shape_result(q),
    )

    output = fused_attention_mps.fused_attention(q, k, v, causal=True)

    assert calls == ["plain"]
    assert output.shape == q.shape


def test_default_attention_falls_back_v3_to_v2_to_plain(attention_inputs, monkeypatch):
    q, k, v = attention_inputs
    calls: list[str] = []

    monkeypatch.delenv("METAL_MARLIN_ATTENTION_BACKEND", raising=False)

    def fail_v3(*args, **kwargs):
        calls.append("v3")
        raise RuntimeError("v3 unavailable")

    def fail_v2(*args, **kwargs):
        calls.append("v2")
        raise RuntimeError("v2 unavailable")

    def run_plain(*args, **kwargs):
        calls.append("plain")
        return _shape_result(q)

    monkeypatch.setattr(fused_attention_mps, "_run_v3_attention", fail_v3)
    monkeypatch.setattr(fused_attention_mps, "_run_v2_attention", fail_v2)
    monkeypatch.setattr(fused_attention_mps, "_run_plain_attention", run_plain)

    output = fused_attention_mps.fused_attention(q, k, v, causal=True)

    assert calls == ["v3", "v2", "plain"]
    assert output.shape == q.shape


def test_all_backends_produce_compatible_output_shapes(attention_inputs, monkeypatch):
    """Verify that v3, v2, and plain backends return outputs matching the query shape."""
    q, k, v = attention_inputs

    for backend in ("v3", "v2", "plain"):
        monkeypatch.setenv("METAL_MARLIN_ATTENTION_BACKEND", backend)
        monkeypatch.setattr(
            fused_attention_mps,
            "_run_v3_attention",
            lambda *args, **kwargs: _shape_result(q),
        )
        monkeypatch.setattr(
            fused_attention_mps,
            "_run_v2_attention",
            lambda *args, **kwargs: _shape_result(q),
        )
        monkeypatch.setattr(
            fused_attention_mps,
            "_run_plain_attention",
            lambda *args, **kwargs: _shape_result(q),
        )

        output = fused_attention_mps.fused_attention(q, k, v, causal=True)

        assert output.shape == q.shape, (
            f"Backend {backend} returned shape {output.shape}, expected {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"Backend {backend} returned dtype {output.dtype}, expected {q.dtype}"
        )
