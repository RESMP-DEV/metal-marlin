from __future__ import annotations

from types import SimpleNamespace

from metal_marlin.inference.mmfp4_pipeline import MMFP4Pipeline


class _DummyAttention:
    def __init__(self) -> None:
        self.use_paged_attention = False
        self.use_fused_decode = False
        self.prefer_glm4_fused_kernel = False


class _DummyLayer:
    def __init__(self) -> None:
        self.self_attn = _DummyAttention()


class _DummyModel:
    def __init__(self, device: str = "cpu", num_layers: int = 2) -> None:
        self.device = device
        self.config = SimpleNamespace()
        self.model = SimpleNamespace(
            layers=[_DummyLayer() for _ in range(num_layers)],
        )

    def eval(self) -> _DummyModel:
        return self


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _build_pipeline(device: str = "cpu") -> MMFP4Pipeline:
    model = _DummyModel(device=device)
    tokenizer = _DummyTokenizer()
    return MMFP4Pipeline(model, tokenizer, use_paged_attention=False)


def test_mla_glm4_kernel_preference_uses_runtime_library_fallback(
    monkeypatch,
) -> None:
    class _RuntimeLibrary:
        def get_function(self, function_name: str) -> object:
            if function_name != "mla_fused_attention_decode_glm4":
                raise KeyError(function_name)
            return object()

    monkeypatch.setattr("metal_marlin.metal_dispatch.get_kernel", lambda _: None)
    monkeypatch.setattr(
        "metal_marlin.mla_fused._get_metal_library",
        lambda: _RuntimeLibrary(),
    )

    pipeline = _build_pipeline(device="mps")

    assert pipeline._glm4_mla_decode_kernel_available is True
    for layer in pipeline.model.model.layers:
        attn = layer.self_attn
        assert attn.use_fused_decode is True
        assert attn.prefer_glm4_fused_kernel is True


def test_mla_glm4_kernel_preference_falls_back_when_unavailable(
    monkeypatch,
) -> None:
    class _NoKernelLibrary:
        def get_function(self, function_name: str) -> object:
            raise KeyError(function_name)

    monkeypatch.setattr("metal_marlin.metal_dispatch.get_kernel", lambda _: None)
    monkeypatch.setattr(
        "metal_marlin.mla_fused._get_metal_library",
        lambda: _NoKernelLibrary(),
    )

    pipeline = _build_pipeline(device="mps")

    assert pipeline._glm4_mla_decode_kernel_available is False
    for layer in pipeline.model.model.layers:
        attn = layer.self_attn
        assert attn.use_fused_decode is True
        assert attn.prefer_glm4_fused_kernel is False
