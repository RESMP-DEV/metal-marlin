"""Qwen3 dense Transformers integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_METAL_MARLIN_ROOT = _ROOT / "contrib" / "metal_marlin"
if _METAL_MARLIN_ROOT.exists():
    sys.path.insert(0, str(_METAL_MARLIN_ROOT))

RUN_QWEN3 = os.environ.get("RUN_QWEN3_TRANSFORMERS") == "1"
if not RUN_QWEN3:
    pytest.skip(
        "Set RUN_QWEN3_TRANSFORMERS=1 to run Qwen3 Transformers tests.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

if not torch.backends.mps.is_available():
    pytest.skip("MPS backend required for Qwen3 tests.", allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from metal_marlin.layer_replacement import replace_linear_layers  # noqa: E402
except Exception as exc:  # pragma: no cover - surfaces missing integration modules
    raise ImportError(
        "Metal Marlin Transformers integration modules are missing. "
        "Ensure layer_replacement is available."
    ) from exc

QWEN3_4B = "Qwen/Qwen3-4B"
_QWEN3_32B_CANDIDATES = (
    "Qwen/Qwen3-32B-Dense",
    "Qwen/Qwen3-32B-dense",
    "Qwen/Qwen3-32B",
)


def _resolve_qwen3_32b_dense() -> str | None:
    for model_id in _QWEN3_32B_CANDIDATES:
        try:
            config = AutoConfig.from_pretrained(model_id)
        except Exception:
            continue
        if config.model_type == "qwen3" and not hasattr(config, "num_experts"):
            return model_id
    return None


@pytest.fixture(scope="session")
def qwen3_model():
    """Load Qwen3-4B for testing."""
    return AutoModelForCausalLM.from_pretrained(
        QWEN3_4B,
        torch_dtype=torch.bfloat16,
        device_map="mps",
    )


@pytest.fixture(scope="session")
def qwen3_tokenizer():
    return AutoTokenizer.from_pretrained(QWEN3_4B)


class TestQwen3DenseTransformersIntegration:
    def test_qwen3_is_dense(self):
        """Verify Qwen3-4B is dense (not MoE)."""
        config = AutoConfig.from_pretrained(QWEN3_4B)
        assert config.model_type == "qwen3"  # Not qwen3_moe
        assert not hasattr(config, "num_experts")

    @pytest.mark.slow
    def test_replacement_on_qwen3(self, qwen3_model):
        """Test layer replacement on Qwen3."""
        stats = replace_linear_layers(qwen3_model, bits=4)
        assert stats["replaced_count"] > 0

    @pytest.mark.slow
    def test_qwen3_generation(self, qwen3_model, qwen3_tokenizer):
        """Test generation after quantization."""
        replace_linear_layers(qwen3_model, bits=4)

        prompt = "Python is"
        input_ids = qwen3_tokenizer(prompt, return_tensors="pt").input_ids.to("mps")
        output_ids = qwen3_model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
        )
        output_text = qwen3_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        assert any(
            word in output_text.lower()
            for word in ("programming", "language", "code")
        )

    def test_qwen3_32b_dense_checkpoint_if_available(self):
        """Verify Qwen3-32B-dense config if the checkpoint exists."""
        model_id = _resolve_qwen3_32b_dense()
        if model_id is None:
            pytest.skip("Qwen3-32B dense checkpoint not available.")
        config = AutoConfig.from_pretrained(model_id)
        assert config.model_type == "qwen3"
        assert not hasattr(config, "num_experts")
