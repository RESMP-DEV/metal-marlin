"""Llama Transformers integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_METAL_MARLIN_ROOT = _ROOT / "contrib" / "metal_marlin"
if _METAL_MARLIN_ROOT.exists():
    sys.path.insert(0, str(_METAL_MARLIN_ROOT))

RUN_LLAMA = os.environ.get("RUN_LLAMA_TRANSFORMERS") == "1"
if not RUN_LLAMA:
    pytest.skip(
        "Set RUN_LLAMA_TRANSFORMERS=1 to run Llama Transformers tests.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

if not torch.backends.mps.is_available():
    pytest.skip("MPS backend required for Llama Transformers tests.", allow_module_level=True)

from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

try:
    from metal_marlin.benchmarks.quality import compare_models  # noqa: E402
    from metal_marlin.layer_replacement import replace_linear_layers  # noqa: E402
except Exception as exc:  # pragma: no cover - surfaces missing integration modules
    raise ImportError(
        "Metal Marlin Transformers integration modules are missing. "
        "Ensure layer_replacement and benchmarks.quality are available."
    ) from exc


def _resolve_model_ids() -> list[str]:
    """Return model IDs to test, honoring env overrides."""
    extra = os.environ.get("LLAMA_MODEL_IDS")
    if extra:
        model_ids = [model_id.strip() for model_id in extra.split(",") if model_id.strip()]
        if model_ids:
            return model_ids

    model_ids = ["meta-llama/Llama-3.2-1B"]
    if os.environ.get("RUN_LLAMA_8B") == "1":
        model_ids.append("meta-llama/Llama-3.1-8B")
    if os.environ.get("RUN_LLAMA_70B") == "1":
        model_ids.append("meta-llama/Llama-3.1-70B")
    return model_ids


LLAMA_MODEL_IDS = _resolve_model_ids()


def _load_llama_model(model_id: str):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="mps",
            trust_remote_code=False,
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


@pytest.fixture(scope="session", params=LLAMA_MODEL_IDS, ids=lambda m: m.rsplit("/", 1)[-1])
def llama_model_id(request):
    return request.param


@pytest.fixture(scope="session")
def llama_model(llama_model_id):
    """Load the requested Llama variant for testing."""
    return _load_llama_model(llama_model_id)


@pytest.fixture(scope="session")
def llama_tokenizer(llama_model_id):
    return AutoTokenizer.from_pretrained(llama_model_id)


class TestLlamaTransformersIntegration:
    def test_llama_architecture_recognized(self, llama_model_id):
        """Verify Llama loads as standard architecture."""
        config = AutoConfig.from_pretrained(llama_model_id)
        assert config.model_type == "llama"
        if config.architectures:
            assert "LlamaForCausalLM" in config.architectures

    @pytest.mark.slow
    def test_layer_replacement_on_llama(self, llama_model):
        """Verify layer replacement works on Llama."""
        linear_count = sum(
            1 for module in llama_model.modules() if isinstance(module, torch.nn.Linear)
        )

        stats = replace_linear_layers(llama_model, bits=4, group_size=128)

        assert stats["replaced_count"] > 0
        assert stats["replaced_count"] + stats["skipped_count"] == linear_count
        assert isinstance(llama_model, LlamaForCausalLM)

    @pytest.mark.slow
    def test_llama_generation_after_quantization(self, llama_model, llama_tokenizer):
        """Verify generation works after quantization."""
        replace_linear_layers(llama_model, bits=4)

        prompt = "The quick brown fox"
        input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to("mps")

        output_ids = llama_model.generate(input_ids, max_new_tokens=20, do_sample=False)
        output_text = llama_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        assert len(output_text) > len(prompt)
        assert not output_text.endswith(prompt)

    @pytest.mark.slow
    def test_llama_quality_acceptable(self, llama_model, llama_tokenizer, llama_model_id):
        """Verify quantization quality is acceptable."""
        ref_model = _load_llama_model(llama_model_id)

        replace_linear_layers(llama_model, bits=4)

        metrics = compare_models(
            ref_model,
            llama_model,
            llama_tokenizer,
            num_samples=20,
            max_length=256,
        )

        assert metrics.perplexity_delta_pct < 10.0
        assert metrics.kl_divergence_mean < 0.1
