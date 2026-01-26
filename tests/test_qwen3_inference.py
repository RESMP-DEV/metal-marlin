from __future__ import annotations

from pathlib import Path

import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH

if not HAS_TORCH:
    pytest.skip("PyTorch not available", allow_module_level=True)

try:
    from metal_marlin.inference import MetalInferenceEngine, load_quantized_model
except Exception as exc:  # pragma: no cover - optional dependency path
    pytest.skip(f"Metal inference engine unavailable: {exc}", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]


def _resolve_model_path(model_path: str) -> Path:
    resolved = ROOT / model_path
    if not resolved.exists():
        pytest.skip(f"Model artifacts not found: {resolved}")
    return resolved


@pytest.mark.parametrize(
    "model_path,expected_layers",
    [
        ("benchmarks/results/qwen3_4b_fp4", 36),
        ("benchmarks/results/qwen3_32b_fp4", 64),
        ("benchmarks/results/qwen3_30b_fp8_int2", 48),
    ],
)
def test_qwen3_loads(model_path: str, expected_layers: int) -> None:
    model, _tokenizer = load_quantized_model(_resolve_model_path(model_path))
    if hasattr(model, "layers"):
        assert len(model.layers) == expected_layers
    else:
        assert getattr(model, "num_layers", None) == expected_layers


@pytest.mark.smoke
@pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)")
def test_qwen3_4b_generates() -> None:
    """Quick smoke test with small model."""
    engine = MetalInferenceEngine(str(_resolve_model_path("benchmarks/results/qwen3_4b_fp4")))
    output = engine.generate("What is 2+2?", max_tokens=20)
    assert "4" in output or "four" in output.lower()
