# Supported Models (Tested)

This file is the authoritative list of models validated by tests in this repo.
Cross-checked against STATUS.md and README.md. If a model appears there but is
not covered by tests, it is listed under "Not yet covered by tests" below.

## Test coverage notes

- Transformers path = AutoModelForCausalLM + replace_linear_layers(...)
- Marlin checkpoint path = pre-quantized Metal Marlin artifacts under benchmarks/results/
- Most model tests are opt-in (env vars) and require Apple Silicon MPS.
- Memory figures are approximate and meant for capacity planning only. If a
  value is not published elsewhere, it is marked "est.".

## Tested models

| Model | HuggingFace ID | Quantization formats tested | Memory (approx) | Known issues / limitations | Test file |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | meta-llama/Llama-3.2-1B | FP4 (replace_linear_layers, bits=4, group_size=128) | est. ~0.5 GB FP4 | RUN_LLAMA_TRANSFORMERS=1, MPS, gated weights | [tests/test_llama_transformers.py](../../tests/test_llama_transformers.py) |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B | FP4 (replace_linear_layers, bits=4, group_size=128) | ~4 GB FP4 (README/STATUS) | RUN_LLAMA_8B=1, gated weights, MPS | [tests/test_llama_transformers.py](../../tests/test_llama_transformers.py) |
| Llama-3.1-70B | meta-llama/Llama-3.1-70B | FP4 (replace_linear_layers, bits=4, group_size=128) | ~40.2 GB FP4 weights (docs/mr_gptq) | RUN_LLAMA_70B=1, very large memory, gated weights, MPS | [tests/test_llama_transformers.py](../../tests/test_llama_transformers.py) |
| Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | FP4 (replace_linear_layers, bits=4) | est. ~3.5 GB FP4 | RUN_MISTRAL_TRANSFORMERS=1, MPS | [tests/test_mistral_transformers.py](../../tests/test_mistral_transformers.py) |
| Phi-2 | microsoft/phi-2 | FP4 (replace_linear_layers, bits=4) | est. ~1.35 GB FP4 | RUN_PHI_TRANSFORMERS=1, trust_remote_code, MPS | [tests/test_phi_transformers.py](../../tests/test_phi_transformers.py) |
| Phi-3-mini-4k-instruct | microsoft/Phi-3-mini-4k-instruct | FP4 (replace_linear_layers, bits=4) | est. ~1.9 GB FP4 | RUN_PHI_TRANSFORMERS=1, trust_remote_code, MPS | [tests/test_phi_transformers.py](../../tests/test_phi_transformers.py) |
| TinyLlama-1.1B-Chat | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | FP4 (MarlinLinear; group_size 32 and 128) | est. ~0.55 GB FP4 | Perplexity-only test (no generation); uses replace_linear_with_marlin_torch | [tests/test_perplexity.py](../../tests/test_perplexity.py) |
| Qwen3-4B | Qwen/Qwen3-4B | FP4 (replace_linear_layers, bits=4)<br>FP4 Marlin checkpoint (benchmarks/results/qwen3_4b_fp4) | ~2 GB FP4 (README/STATUS) | RUN_QWEN3_TRANSFORMERS=1 for HF path; local artifacts required for checkpoint test; MPS | [tests/test_qwen3_dense_transformers.py](../../tests/test_qwen3_dense_transformers.py)<br>[tests/test_inference.py](../../tests/test_inference.py) |
| Qwen3-32B (Dense) | Qwen/Qwen3-32B (or Qwen/Qwen3-32B-Dense) | FP4 Marlin checkpoint (benchmarks/results/qwen3_32b_fp4)<br>HF config check only (no quantize/generate) | ~18.5 GB FP4 weights (docs/mr_gptq) | Transformers test is config-only; local artifacts required for checkpoint test; large memory | [tests/test_qwen3_dense_transformers.py](../../tests/test_qwen3_dense_transformers.py)<br>[tests/test_inference.py](../../tests/test_inference.py) |
| Qwen3-30B-A3B (MoE) | Qwen/Qwen3-30B-A3B | FP4 (quantize_model/replace_linear_layers on tiny config)<br>Mixed FP8/INT2 Marlin checkpoint (benchmarks/results/qwen3_30b_fp8_int2) | est. ~15 GB FP4 (MoE total) | Pretrained load is opt-in (QWEN3_MOE_MODEL); local artifacts required for checkpoint test; MPS | [tests/test_qwen3_moe_transformers.py](../../tests/test_qwen3_moe_transformers.py)<br>[tests/test_inference.py](../../tests/test_inference.py) |
| Qwen3.5-35B-A3B (MoE) | Qwen/Qwen3.5-35B-A3B | MMFP4 via `scripts/quantize_qwen35_35b_a3b_mmfp4.py` (MR-GPTQ, group_size=128, FP4 E2M1, Metal/MPS)<br>Config-level integration via `tests/test_qwen35_support.py` | est. ~18 GB FP4 (MoE total, 256 experts × 512-dim FFN) | ⚠ MoE expert weights: fused `gate_up_proj` with missing `.weight` suffix in checkpoint index; requires special key-splitting in weight loader<br>⚠ DeltaNet layers: 4-tensor projection layout (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) must be handled distinctly from flat-layout models<br>⚠ Nested `text_config`: top-level config fields (`qwen3_5_moe`) do not contain text architecture params; all text fields live inside `text_config` (`qwen3_5_moe_text`)<br>⚠ Hybrid architecture: `full_attention_interval=4` with explicit 40-entry `layer_types` array — every 4th layer (indices 3, 7, …, 39) uses full attention; remaining layers use DeltaNet linear attention<br>⚠ No end-to-end generation test; served inference not yet validated<br>Quantization script requires `--no-calibration` for RTN fallback without a local HF cache | [tests/test_qwen35_support.py](../../tests/test_qwen35_support.py)<br>[scripts/quantize_qwen35_35b_a3b_mmfp4.py](../../scripts/quantize_qwen35_35b_a3b_mmfp4.py) |
| GLM-4.7-Flash | zai-org/GLM-4.7-Flash | FP4 (replace_linear_layers, bits=4)<br>Mixed FP8/INT2 Marlin checkpoint (benchmarks/results/glm47_sensitivity_fp8_int2) | ~15 GB FP4 (README/STATUS) | Requires transformers>=5.0 (GLM4 class); RUN_GLM47_TRANSFORMERS=1 or RUN_GLM47_MOE_ACCURACY=1; MPS; local artifacts for checkpoint test | [tests/test_glm47_transformers.py](../../tests/test_glm47_transformers.py)<br>[tests/test_moe_accuracy.py](../../tests/test_moe_accuracy.py)<br>[tests/test_glm_flash.py](../../tests/test_glm_flash.py) |

## Not yet covered by tests

These models appear in README.md or STATUS.md but do not have direct, model-specific
coverage in tests. They should not be treated as validated until tests are added.

| Model | HuggingFace ID | Notes |
| --- | --- | --- |
| Mixtral-8x7B | mistralai/Mixtral-8x7B-v0.1 | Listed in README/STATUS; no Transformers integration test (only generic MoE unit tests) |
| Qwen3.5-122B-A10B | Qwen/Qwen3.5-122B-A10B | MMFP4 CUDA quantization script available (`scripts/quantize_qwen35_122b_a10b_mmfp4_cuda.py`); full end-to-end model test not yet in CI |
| Qwen3.6-35B-A3B | Qwen/Qwen3.6-35B-A3B | MMFP4 Metal quantization script exists (`scripts/quantize_qwen36_35b_a3b_mmfp4.py`); config-level integration tests (`tests/test_qwen35_support.py` test `ModelConfig` with `qwen3_6_moe`/`qwen3_6_moe_text` model types and nested `text_config`); no end-to-end generation test or served inference validation yet — see note below |

> **Qwen3.6-35B-A3B note:** `test_qwen35_support.py` contains `ModelConfig` parsing tests for `qwen3_6_moe`/`qwen3_6_moe_text` model types with nested `text_config`, and `test_qwen36.py` exercises the `_qwen_moe_shared` module. However, no dedicated end-to-end quantization → load → generate test exists in this batch. The script `quantize_qwen36_35b_a3b_mmfp4.py` reuses the same architecture assumptions as `quantize_qwen35_35b_a3b_mmfp4.py` (256 experts, 512-dim FFN, `model.language_model.layers.*` prefix, 4-tensor DeltaNet layout) and is structured to be cloned from the Qwen3.5 variant. Treat the quantization pipeline as **partially validated** until an end-to-end test confirms correctness.

## Architecture notes for Qwen3.5 / Qwen3.6 MoE models

Both Qwen3.5-35B-A3B and Qwen3.6-35B-A3B share a hybrid **DeltaNet + full-attention** architecture inside a multimodal MoE wrapper. Key structural facts:

- **Multimodal wrapper**: `model_type = qwen3_5_moe` at the top level; actual text model fields live under `text_config` (`model_type = qwen3_5_moe_text`). All config readers must recurse into `text_config` when it is present.
- **Hybrid layer pattern**: 40 layers total; `full_attention_interval = 4` with an explicit 40-entry `layer_types` array. Every 4th layer (indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39) uses full standard attention; the remaining 30 layers use DeltaNet (linear) attention.
- **DeltaNet projection tensors** on linear-attention layers are split into **4 separate weight tensors** (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) — distinct from flat-layout models that use a fused `in_proj_qkvz` + `in_proj_ba`. DeltaNet projection layers are kept in full precision during MMFP4 quantization.
- **Expert tensor naming irregularity**: Expert `down_proj` and `gate_up_proj` tensors in the checkpoint index omit the `.weight` suffix (a serialization artifact). The weight loader must handle both suffixed and unsuffixed variants.

For the complete tensor naming inventory (including the `language_model.` prefix nesting and the `shared_expert` vs `shared_experts` mismatch), see [`docs/audits/qwen35_qwen36_deltanet_inventory_2026_04_21.md`](../../docs/audits/qwen35_qwen36_deltanet_inventory_2026_04_21.md).
