"""Transformers integration tests for Qwen3-30B-A3B (MoE)."""

from __future__ import annotations

import os
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

pytestmark = pytest.mark.slow

_DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B"


def _require_qwen3_moe_class():
    try:
        from transformers import Qwen3MoeForCausalLM  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on transformers version
        pytest.skip(f"Qwen3 MoE not available in transformers: {exc}")
    return Qwen3MoeForCausalLM


def _load_config(model_id: str):
    try:
        return AutoConfig.from_pretrained(model_id)
    except Exception:
        try:
            return AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            pytest.skip(f"Could not load Qwen3 MoE config from {model_id}: {exc}")


def _clone_config(config: Any, overrides: dict[str, Any]) -> Any:
    data = config.to_dict()
    for key, value in overrides.items():
        if key in data:
            data[key] = value
    try:
        return config.__class__.from_dict(data)
    except Exception:
        return config.__class__(**data)


def _get_moe_blocks(model: Any) -> list[Any]:
    blocks = [m for m in model.modules() if m.__class__.__name__ == "Qwen3MoeSparseMoeBlock"]
    if not blocks:
        pytest.skip("Qwen3MoeSparseMoeBlock not found in model")
    return blocks


def _find_router_linear(block: Any, num_experts: int):
    candidates: list[tuple[str, Any]] = []
    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == num_experts:
            candidates.append((name, module))
    if not candidates:
        return None
    preferred = [item for item in candidates if item[0].endswith(("gate", "router", "gate_proj"))]
    return (preferred[0][1] if preferred else candidates[0][1])


def _assert_no_shared_expert(block: Any) -> None:
    for name in ("shared_expert", "shared_experts", "shared_expert_mlp", "shared_mlp"):
        if not hasattr(block, name):
            continue
        value = getattr(block, name)
        if value is None:
            continue
        if isinstance(value, (list, tuple, torch.nn.ModuleList)) and len(value) == 0:
            continue
        assert False, f"Unexpected shared expert found on block: {name}"


def _expert_layer_filter(name: str, _module: Any) -> bool:
    return ".experts.0." in name or name.endswith((".gate", ".router", ".gate_proj"))


def _select_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.fixture(scope="session")
def qwen3_moe_config():
    model_id = os.environ.get("QWEN3_MOE_MODEL", _DEFAULT_MODEL_ID)
    config = _load_config(model_id)
    assert config.model_type == "qwen3_moe"
    assert config.num_experts == 128
    assert config.num_experts_per_tok == 8
    if hasattr(config, "num_shared_experts"):
        assert getattr(config, "num_shared_experts") in (0, None)
    return config


@pytest.fixture
def qwen3_moe_tiny_config(qwen3_moe_config):
    overrides = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "vocab_size": 4096,
        "max_position_embeddings": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
    }
    return _clone_config(qwen3_moe_config, overrides)


@pytest.fixture(scope="session")
def qwen3_moe_pretrained():
    model_id = os.environ.get("QWEN3_MOE_MODEL")
    if not model_id:
        pytest.skip("Set QWEN3_MOE_MODEL to a local path or HF repo to run pretrained tests.")

    device = _select_device()
    if device is None:
        pytest.skip("No accelerator available for Qwen3-30B-A3B pretrained load.")

    Qwen3MoeForCausalLM = _require_qwen3_moe_class()
    try:
        model = Qwen3MoeForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        try:
            model = Qwen3MoeForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            pytest.skip(f"Failed to load pretrained Qwen3 MoE model: {exc}")

    model.to(device)
    model.eval()
    return model, tokenizer


def test_qwen3_moe_model_loads(qwen3_moe_tiny_config):
    Qwen3MoeForCausalLM = _require_qwen3_moe_class()
    model = Qwen3MoeForCausalLM(qwen3_moe_tiny_config)
    assert model.config.model_type == "qwen3_moe"
    assert _get_moe_blocks(model)


def test_qwen3_moe_layer_replacement_preserves_experts(qwen3_moe_tiny_config):
    pytest.importorskip("metal_marlin")
    from metal_marlin.quantize_model import quantize_model

    Qwen3MoeForCausalLM = _require_qwen3_moe_class()
    model = Qwen3MoeForCausalLM(qwen3_moe_tiny_config)
    moe_blocks = _get_moe_blocks(model)
    before_counts = [len(block.experts) for block in moe_blocks]

    stats = quantize_model(
        model,
        group_size=128,
        layer_filter=_expert_layer_filter,
    )
    if stats["replaced_count"] == 0:
        pytest.skip("No linear layers replaced; check model naming for expert layers.")

    after_counts = [len(block.experts) for block in moe_blocks]
    assert before_counts == after_counts
    for block in moe_blocks:
        assert isinstance(block.experts, torch.nn.ModuleList)
        assert len(block.experts) == 128
        _assert_no_shared_expert(block)


def test_qwen3_moe_routing_topk(qwen3_moe_tiny_config):
    Qwen3MoeForCausalLM = _require_qwen3_moe_class()
    model = Qwen3MoeForCausalLM(qwen3_moe_tiny_config)
    block = _get_moe_blocks(model)[0]

    router = _find_router_linear(block, qwen3_moe_tiny_config.num_experts)
    if router is None:
        pytest.skip("MoE router linear not found for Qwen3 MoE block.")

    hidden = torch.randn(2, 4, qwen3_moe_tiny_config.hidden_size)
    logits = router(hidden)

    assert logits.shape[-1] == qwen3_moe_tiny_config.num_experts
    topk = torch.topk(logits, k=qwen3_moe_tiny_config.num_experts_per_tok, dim=-1)
    assert topk.indices.shape[-1] == 8
    assert (topk.indices >= 0).all()
    assert (topk.indices < qwen3_moe_tiny_config.num_experts).all()


def test_qwen3_moe_pretrained_loads(qwen3_moe_pretrained):
    model, _tokenizer = qwen3_moe_pretrained
    assert model.config.model_type == "qwen3_moe"


def test_qwen3_moe_generation_quality(qwen3_moe_pretrained):
    model, tokenizer = qwen3_moe_pretrained
    device = next(model.parameters()).device

    prompt = "Explain what a transformer model is in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    assert len(decoded) > len(prompt) + 10
    assert decoded.strip() != prompt.strip()
    assert sum(char.isalpha() for char in decoded) > 20
