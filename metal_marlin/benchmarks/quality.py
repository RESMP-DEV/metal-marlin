"""Quality comparison utilities for Transformers-based Metal Marlin models."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .._compat import HAS_TORCH, torch
from ..eval_kl_divergence import evaluate_kl_divergence
from ..eval_perplexity import compute_perplexity_from_logits, load_wikitext2
from ..inference_metal import MetalQuantizedLinear
from ..quantize import unpack_fp4_weights


@dataclass(frozen=True)
class QualityMetrics:
    """Quality metrics comparing reference and quantized models."""

    perplexity_ref: float
    perplexity_quant: float
    perplexity_delta_pct: float
    kl_divergence_mean: float
    kl_divergence_max: float
    kl_divergence_std: float
    kl_divergence_p95: float
    mean_rmse: float
    layer_rmse: list[dict[str, float]]
    num_samples: int
    max_length: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def compare_models(
    ref_model: Any,
    quant_model: Any,
    tokenizer: Any,
    *,
    num_samples: int = 50,
    max_length: int = 256,
    texts: list[str] | None = None,
    verbose: bool = True,
) -> QualityMetrics:
    """Compare reference and quantized models using perplexity, KL, and RMSE."""
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for quality comparison.")

    ref_model.eval()
    quant_model.eval()

    if texts is None:
        texts = load_wikitext2(num_samples)
    else:
        texts = texts[:num_samples]

    if verbose:
        print(f"Loaded {len(texts)} WikiText-2 samples for quality evaluation.")

    ref_device = _get_model_device(ref_model)
    quant_device = _get_model_device(quant_model)

    ref_logits_fn = _logits_fn(ref_model, ref_device)
    quant_logits_fn = _logits_fn(quant_model, quant_device)

    if verbose:
        print("Computing perplexity (reference)...")
    ppl_ref = compute_perplexity_from_logits(
        ref_logits_fn, tokenizer, texts, max_length=max_length, verbose=verbose
    )

    if verbose:
        print("Computing perplexity (quantized)...")
    ppl_quant = compute_perplexity_from_logits(
        quant_logits_fn, tokenizer, texts, max_length=max_length, verbose=verbose
    )

    if verbose:
        print("Computing KL divergence...")
    kl_result = evaluate_kl_divergence(
        ref_logits_fn,
        quant_logits_fn,
        tokenizer,
        texts,
        max_length=max_length,
        temperature=1.0,
        verbose=verbose,
    )

    mean_rmse, layer_rmse = _compute_layer_rmse(ref_model, quant_model)

    delta_pct = 0.0
    if ppl_ref > 0:
        delta_pct = (ppl_quant - ppl_ref) / ppl_ref * 100.0

    return QualityMetrics(
        perplexity_ref=float(ppl_ref),
        perplexity_quant=float(ppl_quant),
        perplexity_delta_pct=float(delta_pct),
        kl_divergence_mean=float(kl_result.kl_mean),
        kl_divergence_max=float(kl_result.kl_max),
        kl_divergence_std=float(kl_result.kl_std),
        kl_divergence_p95=float(kl_result.kl_p95),
        mean_rmse=float(mean_rmse),
        layer_rmse=layer_rmse,
        num_samples=len(texts),
        max_length=max_length,
    )


def _get_model_device(model: Any) -> str:
    if not HAS_TORCH or torch is None:
        return "cpu"
    try:
        return next(model.parameters()).device.type
    except StopIteration:
        return "cpu"


def _logits_fn(model: Any, device: str):
    def _fn(input_ids_np: np.ndarray) -> np.ndarray:
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch is required for logits evaluation.")
        input_ids = torch.from_numpy(input_ids_np).to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        result = logits.detach().float().cpu().numpy()
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        return result

    return _fn


def _compute_layer_rmse(ref_model: Any, quant_model: Any) -> tuple[float, list[dict[str, float]]]:
    if not HAS_TORCH or torch is None:
        return 0.0, []

    ref_linears: dict[str, torch.nn.Linear] = {}
    for name, module in ref_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            ref_linears[name] = module

    layer_stats: list[dict[str, float]] = []

    for name, module in quant_model.named_modules():
        if not isinstance(module, MetalQuantizedLinear):
            continue
        if module.bits != 4:
            continue
        ref_layer = ref_linears.get(name)
        if ref_layer is None:
            continue

        meta = {
            "orig_K": module.in_features,
            "orig_N": module.out_features,
            "padded_K": module.in_features,
            "padded_N": module._padded_out_features,
            "group_size": module.group_size,
        }

        quant_k_n = unpack_fp4_weights(
            module.weight_packed,
            module.scales,
            meta,
            weights_dtype=np.float32,
            output_backend="numpy",
        )
        quant_weight = quant_k_n.T
        ref_weight = ref_layer.weight.detach().float().cpu().numpy()
        if ref_weight.shape != quant_weight.shape:
            continue

        diff = ref_weight.astype(np.float32) - quant_weight.astype(np.float32)
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_error = float(np.max(np.abs(diff)))

        layer_stats.append({"layer": name, "rmse": rmse, "max_error": max_error})

    mean_rmse = float(np.mean([s["rmse"] for s in layer_stats])) if layer_stats else 0.0
    return mean_rmse, layer_stats
