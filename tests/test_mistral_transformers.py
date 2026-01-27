"""Mistral Transformers integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_METAL_MARLIN_ROOT = _ROOT / "contrib" / "metal_marlin"
if _METAL_MARLIN_ROOT.exists():
    sys.path.insert(0, str(_METAL_MARLIN_ROOT))

RUN_MISTRAL = os.environ.get("RUN_MISTRAL_TRANSFORMERS") == "1"
if not RUN_MISTRAL:
    pytest.skip(
        "Set RUN_MISTRAL_TRANSFORMERS=1 to run Mistral Transformers tests.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

if not torch.backends.mps.is_available():
    pytest.skip("MPS backend required for Mistral tests.", allow_module_level=True)

from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralForCausalLM,
)

try:
    from metal_marlin.layer_replacement import replace_linear_layers  # noqa: E402
except Exception as exc:  # pragma: no cover - surfaces missing integration modules
    raise ImportError(
        "Metal Marlin Transformers integration modules are missing. "
        "Ensure layer_replacement is available."
    ) from exc

MISTRAL_7B = "mistralai/Mistral-7B-v0.1"


def _load_mistral_model():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_7B,
            torch_dtype=torch.bfloat16,
            device_map="mps",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "out of memory" in message or "not enough memory" in message:
            pytest.skip(f"Skipping {MISTRAL_7B}: insufficient memory on MPS ({exc})")
        if "401" in message or "403" in message or "gated" in message or "permission" in message:
            pytest.skip(f"Skipping {MISTRAL_7B}: no access to gated weights ({exc})")
        pytest.skip(f"Skipping {MISTRAL_7B}: unable to load model ({exc})")
    model.eval()
    return model


@pytest.fixture(scope="session")
def mistral_model():
    return _load_mistral_model()


@pytest.fixture(scope="session")
def mistral_tokenizer():
    return AutoTokenizer.from_pretrained(MISTRAL_7B)


class TestMistralTransformersIntegration:
    def test_mistral_config(self):
        config = AutoConfig.from_pretrained(MISTRAL_7B)
        assert config.model_type == "mistral"
        if config.architectures:
            assert "MistralForCausalLM" in config.architectures
        assert hasattr(config, "sliding_window")
        assert config.sliding_window is not None
        assert config.sliding_window > 0

    @pytest.mark.slow
    def test_mistral_quantization_sliding_window(self, mistral_model, mistral_tokenizer):
        """Verify Mistral sliding window attention survives quantization."""
        window = getattr(mistral_model.config, "sliding_window", None)
        assert window is not None

        stats = replace_linear_layers(mistral_model, bits=4)
        assert stats["replaced_count"] > 0
        assert isinstance(mistral_model, MistralForCausalLM)

        assert mistral_model.config.sliding_window == window
        if hasattr(mistral_model, "model") and hasattr(mistral_model.model, "layers"):
            attn = getattr(mistral_model.model.layers[0], "self_attn", None)
            if attn is not None and hasattr(attn, "sliding_window"):
                assert attn.sliding_window == window

        input_ids = mistral_tokenizer("Hello", return_tensors="pt").input_ids.to("mps")
        output = mistral_model.generate(input_ids, max_new_tokens=10, do_sample=False)
        assert output.shape[1] > input_ids.shape[1]
