"""MoE quantization accuracy tests for GLM-4.7-Flash (FP16 vs FP4)."""

from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metal_marlin._compat import HAS_MPS, HAS_TORCH, torch

RUN_GLM47_MOE_ACCURACY = os.environ.get("RUN_GLM47_MOE_ACCURACY") == "1"
if not RUN_GLM47_MOE_ACCURACY:
    pytest.skip(
        "Set RUN_GLM47_MOE_ACCURACY=1 to run GLM-4.7-Flash MoE accuracy tests.",
        allow_module_level=True,
    )

if not HAS_TORCH or torch is None:
    pytest.skip("PyTorch is required for MoE accuracy tests.", allow_module_level=True)

pytest.importorskip("transformers")

if not HAS_MPS:
    pytest.skip(
        "MPS backend required for GLM-4.7-Flash MoE accuracy tests.", allow_module_level=True
    )

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from metal_marlin.eval import (  # noqa: E402
    compute_perplexity_from_logits,
    load_wikitext2,
)
from metal_marlin.inference_metal import MetalQuantizedLinear  # noqa: E402
from metal_marlin.layer_replacement import replace_linear_layers  # noqa: E402
from metal_marlin.quantize import unpack_fp4_weights  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.requires_mps]

MODEL_ID = "zai-org/GLM-4.7-Flash"
PROMPT = "The capital of France is"
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "glm47_moe_accuracy_golden.json"

_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


@dataclass(frozen=True)
class LayerMSE:
    per_layer: dict[int, float]
    max_mse: float


def _get_layers(model: Any) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return list(model.transformer.layers)
    raise RuntimeError("Could not locate transformer layers for GLM-4.7-Flash model.")


def _make_input_ids(tokenizer: Any, prompt: str, length: int, device: str) -> Any:
    tokens = tokenizer.encode(prompt)
    if not tokens:
        raise ValueError("Tokenizer produced empty token list.")
    if len(tokens) < length:
        repeats = (length // len(tokens)) + 1
        tokens = (tokens * repeats)[:length]
    else:
        tokens = tokens[:length]
    return torch.tensor(tokens, device=device).unsqueeze(0)


def _forward_with_hooks(
    model: Any,
    input_ids: Any,
    *,
    sample_tokens: int = 64,
    output_router_logits: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, Any]:
    layers = _get_layers(model)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def _hook(idx: int):
        def _inner(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return
            seq_len = min(hidden.shape[1], sample_tokens)
            captured[idx] = hidden[:, :seq_len, :].detach().float().cpu()

        return _inner

    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(_hook(idx)))

    try:
        with torch.no_grad():
            try:
                outputs = model(
                    input_ids,
                    use_cache=False,
                    output_router_logits=output_router_logits,
                    return_dict=True,
                )
            except TypeError:
                outputs = model(input_ids, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()

    logits = outputs.logits.detach().float().cpu()
    return captured, logits, outputs


def _compute_layer_mse(
    ref_states: dict[int, torch.Tensor], quant_states: dict[int, torch.Tensor]
) -> LayerMSE:
    per_layer: dict[int, float] = {}
    for idx, ref in ref_states.items():
        quant = quant_states.get(idx)
        if quant is None:
            continue
        seq_len = min(ref.shape[1], quant.shape[1])
        hidden = min(ref.shape[2], quant.shape[2])
        ref_slice = ref[:, :seq_len, :hidden]
        quant_slice = quant[:, :seq_len, :hidden]
        mse = torch.mean((ref_slice - quant_slice) ** 2).item()
        per_layer[idx] = float(mse)
    max_mse = max(per_layer.values()) if per_layer else 0.0
    return LayerMSE(per_layer=per_layer, max_mse=float(max_mse))


def _routing_summary(outputs: Any) -> dict[str, float] | None:
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is None:
        return None
    if isinstance(router_logits, torch.Tensor):
        router_logits = (router_logits,)

    unique_counts = []
    for layer_logits in router_logits:
        if not torch.is_tensor(layer_logits):
            continue
        logits = layer_logits
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.shape[-1])
        top1 = logits.argmax(dim=-1)
        unique_counts.append(int(torch.unique(top1).numel()))

    if not unique_counts:
        return None
    return {
        "mean_unique": float(np.mean(unique_counts)),
        "min_unique": float(np.min(unique_counts)),
        "max_unique": float(np.max(unique_counts)),
    }


def _logits_fn(model: Any, device: str):
    def _fn(input_ids_np: np.ndarray) -> np.ndarray:
        input_ids = torch.from_numpy(input_ids_np).to(device)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False, return_dict=True)
            logits = outputs.logits
        result = logits.detach().float().cpu().numpy()
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        return result

    return _fn


def _unpack_quant_weight(layer: MetalQuantizedLinear) -> np.ndarray:
    meta = {
        "orig_K": layer.in_features,
        "orig_N": layer.out_features,
        "padded_K": layer.in_features,
        "padded_N": layer._padded_out_features,
        "group_size": layer.group_size,
    }
    quant_k_n = unpack_fp4_weights(
        layer.weight_packed,
        layer.scales,
        meta,
        weights_dtype=np.float32,
        output_backend="numpy",
    )
    return quant_k_n.T


def _compute_per_expert_error(ref_model: Any, quant_model: Any) -> dict[int, float]:
    ref_linears = {
        name: module
        for name, module in ref_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    errors: dict[int, list[float]] = {}
    for name, module in quant_model.named_modules():
        if not isinstance(module, MetalQuantizedLinear):
            continue
        match = _EXPERT_RE.search(name)
        if match is None:
            continue
        ref_layer = ref_linears.get(name)
        if ref_layer is None:
            continue
        quant_weight = _unpack_quant_weight(module)
        ref_weight = ref_layer.weight.detach().float().cpu().numpy()
        if ref_weight.shape != quant_weight.shape:
            continue
        diff = ref_weight - quant_weight
        rmse = float(np.sqrt(np.mean(diff**2)))
        idx = int(match.group(1))
        errors.setdefault(idx, []).append(rmse)

    return {idx: float(np.mean(vals)) for idx, vals in errors.items() if vals}


@pytest.fixture(scope="session")
def glm47_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="session")
def glm47_fp16_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=False,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def glm47_fp4_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=False,
    )
    replace_linear_layers(model, bits=4, group_size=128)
    model.eval()
    return model


@pytest.mark.smoke
@pytest.mark.parametrize("seq_len", [1])
def test_single_token_mse(glm47_fp16_model, glm47_fp4_model, glm47_tokenizer, seq_len: int) -> None:
    input_ids = _make_input_ids(glm47_tokenizer, PROMPT, seq_len, "mps")

    ref_states, ref_logits, _ = _forward_with_hooks(
        glm47_fp16_model, input_ids, sample_tokens=seq_len
    )
    quant_states, quant_logits, _ = _forward_with_hooks(
        glm47_fp4_model, input_ids, sample_tokens=seq_len
    )

    assert ref_logits.shape == quant_logits.shape
    layer_mse = _compute_layer_mse(ref_states, quant_states)

    assert layer_mse.max_mse < 0.01, f"Per-layer MSE too high: {layer_mse.max_mse:.6f}"


@pytest.mark.parametrize("seq_len", [128])
def test_short_sequence_mse(
    glm47_fp16_model, glm47_fp4_model, glm47_tokenizer, seq_len: int
) -> None:
    input_ids = _make_input_ids(glm47_tokenizer, PROMPT, seq_len, "mps")

    ref_states, ref_logits, _ = _forward_with_hooks(glm47_fp16_model, input_ids, sample_tokens=64)
    quant_states, quant_logits, _ = _forward_with_hooks(
        glm47_fp4_model, input_ids, sample_tokens=64
    )

    assert ref_logits.shape == quant_logits.shape
    layer_mse = _compute_layer_mse(ref_states, quant_states)

    assert layer_mse.max_mse < 0.01, f"Per-layer MSE too high: {layer_mse.max_mse:.6f}"


@pytest.mark.expensive
@pytest.mark.parametrize("seq_len", [2048])
def test_long_sequence_mse(
    glm47_fp16_model, glm47_fp4_model, glm47_tokenizer, seq_len: int
) -> None:
    input_ids = _make_input_ids(glm47_tokenizer, PROMPT, seq_len, "mps")

    ref_states, ref_logits, _ = _forward_with_hooks(glm47_fp16_model, input_ids, sample_tokens=32)
    quant_states, quant_logits, _ = _forward_with_hooks(
        glm47_fp4_model, input_ids, sample_tokens=32
    )

    assert ref_logits.shape == quant_logits.shape
    layer_mse = _compute_layer_mse(ref_states, quant_states)

    assert layer_mse.max_mse < 0.01, f"Per-layer MSE too high: {layer_mse.max_mse:.6f}"


@pytest.mark.expensive
def test_mixed_routing_patterns(glm47_fp16_model, glm47_fp4_model, glm47_tokenizer) -> None:
    concentrated_ids = _make_input_ids(glm47_tokenizer, "the ", 128, "mps")
    diverse_ids = _make_input_ids(
        glm47_tokenizer,
        "The quick brown fox jumps over the lazy dog. ",
        128,
        "mps",
    )

    ref_states_c, _, outputs_c = _forward_with_hooks(
        glm47_fp16_model, concentrated_ids, sample_tokens=64, output_router_logits=True
    )
    quant_states_c, _, _ = _forward_with_hooks(
        glm47_fp4_model, concentrated_ids, sample_tokens=64, output_router_logits=True
    )
    ref_states_d, _, outputs_d = _forward_with_hooks(
        glm47_fp16_model, diverse_ids, sample_tokens=64, output_router_logits=True
    )
    quant_states_d, _, _ = _forward_with_hooks(
        glm47_fp4_model, diverse_ids, sample_tokens=64, output_router_logits=True
    )

    mse_c = _compute_layer_mse(ref_states_c, quant_states_c)
    mse_d = _compute_layer_mse(ref_states_d, quant_states_d)

    assert mse_c.max_mse < 0.01, f"Per-layer MSE too high (concentrated): {mse_c.max_mse:.6f}"
    assert mse_d.max_mse < 0.01, f"Per-layer MSE too high (diverse): {mse_d.max_mse:.6f}"

    summary_c = _routing_summary(outputs_c)
    summary_d = _routing_summary(outputs_d)
    if summary_c and summary_d:
        assert summary_d["mean_unique"] >= summary_c["mean_unique"] * 0.8


@pytest.mark.expensive
def test_per_expert_weight_error(glm47_fp16_model, glm47_fp4_model) -> None:
    errors = _compute_per_expert_error(glm47_fp16_model, glm47_fp4_model)
    assert errors, "No per-expert weights found to compute quantization error."
    assert all(np.isfinite(val) for val in errors.values())


@pytest.mark.expensive
def test_perplexity_increase(glm47_fp16_model, glm47_fp4_model, glm47_tokenizer) -> None:
    texts = load_wikitext2(max_samples=8)

    ref_ppl = compute_perplexity_from_logits(
        _logits_fn(glm47_fp16_model, "mps"),
        glm47_tokenizer,
        texts,
        max_length=256,
        verbose=False,
    )
    quant_ppl = compute_perplexity_from_logits(
        _logits_fn(glm47_fp4_model, "mps"),
        glm47_tokenizer,
        texts,
        max_length=256,
        verbose=False,
    )

    assert np.isfinite(ref_ppl)
    assert np.isfinite(quant_ppl)

    delta_pct = (quant_ppl - ref_ppl) / ref_ppl * 100.0
    assert delta_pct < 5.0, (
        f"Perplexity increase too high: ref={ref_ppl:.4f}, quant={quant_ppl:.4f} "
        f"delta={delta_pct:.2f}%"
    )


@pytest.mark.expensive
def test_golden_output_regression(glm47_fp16_model, glm47_fp4_model, glm47_tokenizer) -> None:
    update_golden = os.environ.get("UPDATE_GLM47_MOE_GOLDEN") == "1"

    input_ids = _make_input_ids(glm47_tokenizer, PROMPT, 16, "mps")

    with torch.no_grad():
        fp16_ids = glm47_fp16_model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
        )
        fp4_ids = glm47_fp4_model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
        )

    fp16_text = glm47_tokenizer.decode(fp16_ids[0], skip_special_tokens=True).strip()
    fp4_text = glm47_tokenizer.decode(fp4_ids[0], skip_special_tokens=True).strip()

    if update_golden:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(
                {
                    "prompt": PROMPT,
                    "fp16": fp16_text,
                    "fp4": fp4_text,
                },
                indent=2,
                sort_keys=True,
            )
        )
        pytest.skip("Golden outputs updated.")

    if not GOLDEN_PATH.exists():
        pytest.fail("Golden file missing. Run with UPDATE_GLM47_MOE_GOLDEN=1 to generate.")

    golden = json.loads(GOLDEN_PATH.read_text())
    assert fp16_text == golden.get("fp16"), "FP16 output drifted from golden."
    assert fp4_text == golden.get("fp4"), "FP4 output drifted from golden."
