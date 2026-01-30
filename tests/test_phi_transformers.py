"""Phi Transformers integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_METAL_MARLIN_ROOT = _ROOT / "contrib" / "metal_marlin"
if _METAL_MARLIN_ROOT.exists():
    sys.path.insert(0, str(_METAL_MARLIN_ROOT))

RUN_PHI = os.environ.get("RUN_PHI_TRANSFORMERS") == "1"
if not RUN_PHI:
    pytest.skip(
        "Set RUN_PHI_TRANSFORMERS=1 to run Phi Transformers tests.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

if not torch.backends.mps.is_available():
    pytest.skip("MPS backend required for Phi Transformers tests.", allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from metal_marlin.layer_replacement import replace_linear_layers  # noqa: E402
except Exception as exc:  # pragma: no cover - surfaces missing integration modules
    raise ImportError(
        "Metal Marlin Transformers integration modules are missing. "
        "Ensure layer_replacement is available."
    ) from exc


def _resolve_phi_model_ids() -> list[str]:
    """Return Phi model IDs to test, honoring env overrides."""
    extra = os.environ.get("PHI_MODEL_IDS")
    if extra:
        model_ids = [model_id.strip() for model_id in extra.split(",") if model_id.strip()]
        if model_ids:
            return model_ids
    return [
        "microsoft/phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
    ]


PHI_MODEL_IDS = _resolve_phi_model_ids()


def _load_phi_model(model_id: str):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="mps",
            trust_remote_code=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "out of memory" in message or "not enough memory" in message:
            pytest.skip(f"Skipping {model_id}: insufficient memory on MPS ({exc})")
        if "401" in message or "403" in message or "gated" in message or "permission" in message:
            pytest.skip(f"Skipping {model_id}: no access to gated weights ({exc})")
        pytest.skip(f"Skipping {model_id}: unable to load model ({exc})")
    model.eval()
    return model


@pytest.fixture(scope="session", params=PHI_MODEL_IDS, ids=lambda m: m.rsplit("/", 1)[-1])
def phi_model_id(request):
    return request.param


@pytest.fixture(scope="session")
def phi_model(phi_model_id):
    return _load_phi_model(phi_model_id)


@pytest.fixture(scope="session")
def phi_tokenizer(phi_model_id):
    return AutoTokenizer.from_pretrained(phi_model_id, trust_remote_code=True)


class TestPhiTransformersIntegration:
    def test_phi_config(self, phi_model_id):
        config = AutoConfig.from_pretrained(phi_model_id, trust_remote_code=True)
        assert config.model_type in {"phi", "phi3"}
        if config.architectures:
            assert any("Phi" in arch for arch in config.architectures)

    @pytest.mark.slow
    def test_phi_quantization(self, phi_model, phi_tokenizer):
        stats = replace_linear_layers(phi_model, bits=4)
        assert stats["replaced_count"] > 0

        input_ids = phi_tokenizer("def hello():\n    ", return_tensors="pt").input_ids.to("mps")
        output = phi_model.generate(input_ids, max_new_tokens=20, do_sample=False)
        text = phi_tokenizer.decode(output[0], skip_special_tokens=True)

        # Should generate Python code
        assert "print" in text or "return" in text
