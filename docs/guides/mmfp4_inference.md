# MMFP4 Inference Guide

## Overview

MMFP4 (Mixed-Mode FP4) is a 4-bit quantization format that achieves near-FP8 accuracy with 50% memory reduction. This format is specifically supported by GLM-4.7-Flash models quantized with Marlin.

### Key Features

- **4-bit weights, 8-bit activations**: Optimal balance of precision and efficiency
- **Marlin kernel acceleration**: Optimized Metal shaders for Apple Silicon
- **Fused MoE support**: Reduces memory bandwidth by ~40% for mixture-of-experts models
- **GLM-4.7-Flash optimized**: Purpose-built for the GLM-4.7-Flash architecture

## Quick Start

### Generate Text

```bash
metal-marlin mmfp4 generate \
  -m models/GLM-4.7-Flash-Marlin-MMFP4 \
  -p "Hello, how are you?"
```

### Serve API

```bash
metal-marlin mmfp4 serve \
  -m models/GLM-4.7-Flash-Marlin-MMFP4 \
  --port 8000
```

The server implements an OpenAI-compatible chat completions API at `http://localhost:8000/v1/chat/completions`.

### Benchmark Performance

```bash
metal-marlin mmfp4 bench \
  -m models/GLM-4.7-Flash-Marlin-MMFP4
```

> **⚠️ Warning**: Benchmarks load multi-GB models and should never be run in automated CI/agent tasks. They will automatically skip when `ALPHAHENG_TASK_MODE=1` is set.

## Performance Expectations

> **⚠️ Current Status (Feb 2026):** MMFP4 is significantly slower than expected.
> Correctness is verified (perplexity 17.15), but performance needs optimization.

| Metric | Measured Value | Expected Value | Gap |
|--------|----------------|----------------|-----|
| Load Time | ~15 seconds | ~12 seconds | On target |
| Prefill (1024 ctx) | 39 tok/s | 500-2000 tok/s | 13-50× slower |
| Decode (with KV cache) | 0.27 tok/s | 5-20 tok/s | 20-75× slower |
| Decode (manual loop) | 0.07 tok/s | N/A | No KV cache |
| Memory Usage | ~17 GB | ~17 GB | On target |

### Why Is MMFP4 So Slow?

The performance gap is under investigation. Suspected causes:

1. **MMFP4 GEMM kernel efficiency** - May need optimization for M4 Max
2. **PagedAttention not wired** - `paged_attention_v1_fp4` exists but unused
3. **MoE dispatch overhead** - Despite fused kernel, something is still slow

### Comparison with Other Backends

| Backend | Decode | Notes |
|---------|--------|-------|
| Metal Marlin (non-MMFP4 Trellis) | 5.4 tok/s | Working baseline |
| llama.cpp Q4_K_M | ~3 tok/s | Different quant format |
| **MMFP4 (current)** | **0.27 tok/s** | **Needs optimization** |

### Critical: Always Use KV Cache

**Without KV cache (manual loop):** 0.07 tok/s (full context recompute every token!)
**With KV cache (model.generate):** 0.27 tok/s (4× faster)

```python
# WRONG - 4x slower (no KV cache)
for _ in range(50):
    logits = model(input_ids)
    next_token = sample(logits)
    input_ids = torch.cat([input_ids, next_token], dim=1)

# CORRECT - uses MLAKVCache automatically
output = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True  # Critical!
)
```

### Optimizing Throughput

Enable fused MoE for best performance:

```bash
metal-marlin mmfp4 generate \
  -m models/GLM-4.7-Flash-Marlin-MMFP4 \
  -p "Hello" \
  --use-fused-moe
```

## Troubleshooting

### NaN Issues

**Symptom**: Model outputs `NaN` or gibberish tokens.

**Solution**: Ensure you have the latest code with the `wait=True` synchronization fix. Update to the most recent commit and rebuild:

```bash
git pull origin main
cd contrib/metal_marlin && pip install -e .
```

The `wait=True` fix ensures proper synchronization between Metal command buffers and CPU access, preventing race conditions that cause NaN outputs.

### Slow Performance

**Symptom**: Throughput significantly below 5 tok/s.

**Solution**: Verify `use_fused_moe` is enabled:

```python
# Check configuration
from metal_marlin import load_model

model = load_model("models/GLM-4.7-Flash-Marlin-MMFP4")
print(f"Fused MoE enabled: {model.config.use_fused_moe}")
```

If disabled, enable it:

```bash
metal-marlin mmfp4 generate -m models/GLM-4.7-Flash-Marlin-MMFP4 -p "Test" --use-fused-moe
```

### Memory Issues

**Symptom**: Out-of-memory errors on 16 GB systems.

**Solution**: MMFP4 models require ~17 GB peak memory during loading. Options:

1. Use a system with 32 GB unified memory
2. Close other memory-intensive applications
3. Consider using the smaller GLM-4.7 variants

### Shader Compilation Errors

**Symptom**: `MTLLibrary` creation fails or kernel not found.

**Solution**: Ensure macOS is updated and Metal is properly configured:

```bash
# Verify Metal availability
python -c "import metal_marlin; metal_marlin.check_metal()"
```

## Model Format Details

MMFP4 uses the following quantization scheme:

- **Weights**: 4-bit per element (E2M1 format)
- **Scales**: 8-bit per block (typically 128-element blocks)
- **Activations**: 8-bit (dynamic quantization)

This format requires the Marlin weight layout and is not compatible with standard GGUF or GPTQ formats.

## See Also

- [Metal Marlin README](../README.md)
- [GLM-4.7-Flash Model Card](https://huggingface.co/THUDM/GLM-4.7-Flash)
- [Marlin Paper](https://arxiv.org/abs/2310.15531)

## PagedAttention for Decode

### Enabling PagedAttention

For faster decode-phase inference, enable PagedAttention:

```python
mla = MMFP4MLA(
    ...,
    use_paged_attention=True,
)
```

This uses FP4-quantized KV cache with paged memory management,
reducing memory bandwidth by 4x compared to FP16.

### When to Use

- Prefill (seq_len > 1): Uses standard SDPA (optimal for batched prefill)
- Decode (seq_len = 1): Uses PagedAttention (optimal for autoregressive)

### Performance

Decode throughput improvement: ~2-3x for long sequences (>2K tokens)
Memory reduction: ~4x for KV cache