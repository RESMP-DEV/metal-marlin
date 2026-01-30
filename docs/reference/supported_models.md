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
| Llama-3.2-1B | meta-llama/Llama-3.2-1B | FP4 (replace_linear_layers, bits=4, group_size=128) | est. ~0.5 GB FP4 | RUN_LLAMA_TRANSFORMERS=1, MPS, gated weights | [tests/test_llama_transformers.py](../tests/test_llama_transformers.py) |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B | FP4 (replace_linear_layers, bits=4, group_size=128) | ~4 GB FP4 (README/STATUS) | RUN_LLAMA_8B=1, gated weights, MPS | [tests/test_llama_transformers.py](../tests/test_llama_transformers.py) |
| Llama-3.1-70B | meta-llama/Llama-3.1-70B | FP4 (replace_linear_layers, bits=4, group_size=128) | ~40.2 GB FP4 weights (docs/mr_gptq) | RUN_LLAMA_70B=1, very large memory, gated weights, MPS | [tests/test_llama_transformers.py](../tests/test_llama_transformers.py) |
| Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | FP4 (replace_linear_layers, bits=4) | est. ~3.5 GB FP4 | RUN_MISTRAL_TRANSFORMERS=1, MPS | [tests/test_mistral_transformers.py](../tests/test_mistral_transformers.py) |
| Phi-2 | microsoft/phi-2 | FP4 (replace_linear_layers, bits=4) | est. ~1.35 GB FP4 | RUN_PHI_TRANSFORMERS=1, trust_remote_code, MPS | [tests/test_phi_transformers.py](../tests/test_phi_transformers.py) |
| Phi-3-mini-4k-instruct | microsoft/Phi-3-mini-4k-instruct | FP4 (replace_linear_layers, bits=4) | est. ~1.9 GB FP4 | RUN_PHI_TRANSFORMERS=1, trust_remote_code, MPS | [tests/test_phi_transformers.py](../tests/test_phi_transformers.py) |
| TinyLlama-1.1B-Chat | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | FP4 (MarlinLinear; group_size 32 and 128) | est. ~0.55 GB FP4 | Perplexity-only test (no generation); uses replace_linear_with_marlin_torch | [tests/test_perplexity.py](../tests/test_perplexity.py) |
| Qwen3-4B | Qwen/Qwen3-4B | FP4 (replace_linear_layers, bits=4)<br>FP4 Marlin checkpoint (benchmarks/results/qwen3_4b_fp4) | ~2 GB FP4 (README/STATUS) | RUN_QWEN3_TRANSFORMERS=1 for HF path; local artifacts required for checkpoint test; MPS | [tests/test_qwen3_dense_transformers.py](../tests/test_qwen3_dense_transformers.py)<br>[tests/test_qwen3_inference.py](../tests/test_qwen3_inference.py) |
| Qwen3-32B (Dense) | Qwen/Qwen3-32B (or Qwen/Qwen3-32B-Dense) | FP4 Marlin checkpoint (benchmarks/results/qwen3_32b_fp4)<br>HF config check only (no quantize/generate) | ~18.5 GB FP4 weights (docs/mr_gptq) | Transformers test is config-only; local artifacts required for checkpoint test; large memory | [tests/test_qwen3_dense_transformers.py](../tests/test_qwen3_dense_transformers.py)<br>[tests/test_qwen3_inference.py](../tests/test_qwen3_inference.py) |
| Qwen3-30B-A3B (MoE) | Qwen/Qwen3-30B-A3B | FP4 (quantize_model/replace_linear_layers on tiny config)<br>Mixed FP8/INT2 Marlin checkpoint (benchmarks/results/qwen3_30b_fp8_int2) | est. ~15 GB FP4 (MoE total) | Pretrained load is opt-in (QWEN3_MOE_MODEL); local artifacts required for checkpoint test; MPS | [tests/test_qwen3_moe_transformers.py](../tests/test_qwen3_moe_transformers.py)<br>[tests/test_qwen3_inference.py](../tests/test_qwen3_inference.py) |
| GLM-4.7-Flash | zai-org/GLM-4.7-Flash | FP4 (replace_linear_layers, bits=4)<br>Mixed FP8/INT2 Marlin checkpoint (benchmarks/results/glm47_sensitivity_fp8_int2) | ~15 GB FP4 (README/STATUS) | Requires transformers>=5.0 (GLM4 class); RUN_GLM47_TRANSFORMERS=1 or RUN_GLM47_MOE_ACCURACY=1; MPS; local artifacts for checkpoint test | [tests/test_glm47_transformers.py](../tests/test_glm47_transformers.py)<br>[tests/test_moe_accuracy.py](../tests/test_moe_accuracy.py)<br>[tests/test_glm4_integration.py](../tests/test_glm4_integration.py) |

## Not yet covered by tests

These models appear in README.md or STATUS.md but do not have direct, model-specific
coverage in tests. They should not be treated as validated until tests are added.

| Model | HuggingFace ID | Notes |
| --- | --- | --- |
| Mixtral-8x7B | mistralai/Mixtral-8x7B-v0.1 | Listed in README/STATUS; no Transformers integration test (only generic MoE unit tests) |
