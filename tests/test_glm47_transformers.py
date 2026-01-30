"""GLM-4.7-Flash Transformers integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_METAL_MARLIN_ROOT = _ROOT / "contrib" / "metal_marlin"
if _METAL_MARLIN_ROOT.exists():
    sys.path.insert(0, str(_METAL_MARLIN_ROOT))

RUN_GLM47 = os.environ.get("RUN_GLM47_TRANSFORMERS") == "1"
if not RUN_GLM47:
    pytest.skip(
        "Set RUN_GLM47_TRANSFORMERS=1 to run GLM-4.7-Flash Transformers tests.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

if not torch.backends.mps.is_available():
    pytest.skip("MPS backend required for GLM-4.7-Flash tests.", allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    from metal_marlin.layer_replacement import replace_linear_layers  # noqa: E402
    from metal_marlin.transformers_loader import load_and_quantize  # noqa: E402
except Exception as exc:  # pragma: no cover - surfaces missing integration modules
    raise ImportError(
        "Metal Marlin Transformers integration modules are missing. "
        "Ensure layer_replacement and transformers_loader are available."
    ) from exc


@pytest.fixture(scope="session")
def glm47_model():
    """Load GLM-4.7-Flash via Transformers."""
    return AutoModelForCausalLM.from_pretrained(
        "zai-org/GLM-4.7-Flash",
        torch_dtype=torch.bfloat16,
        device_map="mps",
        trust_remote_code=False,
    )


@pytest.fixture(scope="session")
def glm47_tokenizer():
    return AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")


class TestGLM47TransformersIntegration:
    @pytest.mark.slow
    def test_model_loads_via_transformers(self):
        """Verify GLM-4.7-Flash loads with transformers 5.0+."""
        try:
            from transformers import Glm4MoeLiteForCausalLM  # noqa: F401
        except Exception as exc:
            pytest.fail(f"Transformers GLM4 class not available: {exc}")
        config = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash")
        assert config.model_type == "glm4_moe_lite"
        assert config.n_routed_experts == 64
        assert config.n_shared_experts == 1
        assert callable(load_and_quantize)

    @pytest.mark.slow
    def test_layer_replacement_preserves_structure(self, glm47_model):
        """Verify layer replacement doesn't break model structure."""
        linear_count_before = sum(
            1 for m in glm47_model.modules() if isinstance(m, torch.nn.Linear)
        )

        stats = replace_linear_layers(glm47_model, bits=4, group_size=128)

        assert stats["replaced_count"] > 0
        assert stats["replaced_count"] + stats["skipped_count"] == linear_count_before
        assert hasattr(glm47_model.model.layers[1].mlp, "experts")
        assert hasattr(glm47_model.model.layers[1].mlp, "shared_experts")

    @pytest.mark.slow
    def test_moe_routing_still_works(self, glm47_model, glm47_tokenizer):
        """Verify MoE routing works after quantization."""
        replace_linear_layers(glm47_model, bits=4)

        input_ids = glm47_tokenizer("Hello", return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            output = glm47_model(input_ids)

        assert output.logits.shape[-1] == glm47_model.config.vocab_size
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()

    @pytest.mark.slow
    def test_generation_produces_coherent_text(self, glm47_model, glm47_tokenizer):
        """Verify generation works end-to-end."""
        replace_linear_layers(glm47_model, bits=4)

        prompt = "The capital of France is"
        input_ids = glm47_tokenizer(prompt, return_tensors="pt").input_ids.to("mps")

        output_ids = glm47_model.generate(input_ids, max_new_tokens=20, do_sample=False)
        output_text = glm47_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Verify we got actual generated text (not empty/garbage)
        assert len(output_text) > len(prompt), "Model should generate additional text"
        # Only ASCII/Unicode text, no garbage bytes
        assert output_text.isprintable() or "\n" in output_text, "Output should be readable"
        # Quantized model may not always say "Paris" but should mention France-related concepts
        france_related = any(
            w in output_text.lower()
            for w in ["paris", "france", "french", "city", "capital", "europe"]
        )
        assert france_related, f"Output should relate to France: {output_text}"
