# Metal Marlin vs vLLM: Quantization Comparison

This document compares Metal Marlin's quantization implementation with vLLM's reference implementation to identify feature parity, format compatibility, and opportunities for improvement.

## Feature Parity Matrix

| Feature | vLLM | Metal Marlin | Notes |
|---------|------|--------------|-------|
| **Weight Quantization** | | | |
| FP4 (E2M1) | Via compressed-tensors | Native | Core format |
| INT4 (U4) | GPTQ/AWQ | Native | With zero points |
| INT8 | Native | Implemented | Per-channel |
| FP8 (E4M3) | Native | Implemented | W8A16 kernels |
| INT2 | Not available | Implemented | 16 weights/uint32 |
| INT3 | Not available | Implemented | 10 weights/uint32 |
| **Scale Handling** | | | |
| Per-tensor | Yes | No | Add for simpler formats |
| Per-channel | Yes | Yes | INT8 path |
| Per-group | Yes | Yes | Default for FP4/INT4 |
| Block-wise (2D) | Yes (DeepSeek) | No | Add for DeepSeek models |
| **Activation Quantization** | | | |
| Static (calibrated) | Yes | No | Add via calibration |
| Dynamic (per-batch) | Yes | No | Would need runtime quant |
| FP8 activations | Yes | No | Requires W8A8 kernels |
| **Architectural** | | | |
| MoE support | Native FusedMoE | Via layer dispatch | No fused MoE kernel |
| Paged attention | PagedAttention v1/v2 | Implemented | vLLM-style blocks |
| Flash attention | FlashAttention-2 | Implemented | FP4 KV variant |
| KV cache quant | FP8 native | FP4 implemented | vLLM uses FP8 |
| Tensor parallelism | Native | Not implemented | Single-GPU focus |
| **Calibration** | | | |
| GPTQ-style | External | Not implemented | Add for accuracy |
| AWQ-style | External | Not implemented | Consider adding |
| Static per-layer | Yes | Bartowski loader | Via calibration.py |
| **Weight Formats** | | | |
| GGUF loading | Via llama.cpp | Implemented | gguf_loader.py |
| Safetensors | Native | Native | safetensors_loader.py |
| ONNX | Not native | Implemented | onnx_loader.py |
| vLLM quant configs | Native | Partial | Can read some configs |

## Weight Packing Format Comparison

### vLLM GPTQ/AWQ Format
```
qweight: [in_features // pack_factor, out_features]  # INT32 packed
qzeros:  [num_groups, out_features // pack_factor]   # INT32 packed
scales:  [num_groups, out_features]                  # FP16/FP32
g_idx:   [in_features]                               # Optional reorder
```

- Pack along input dimension (K/reduction)
- Zeros are packed (8 INT4 per INT32)
- Supports activation reordering (desc_act)

### Metal Marlin Format
```
B_packed: [K // 8, N]     # UINT32 packed (FP4) or [K // 8, N] (INT4)
scales:   [K // group_size, N]  # FP16
zeros:    [K // group_size, N]  # FP16 (INT4 only, unpacked)
```

- Pack along K dimension (8 values per uint32)
- Zeros are unpacked FP16 (simpler kernel, more memory)
- No activation reordering support

### Key Differences

1. **Packing Axis**: vLLM packs along input features, Metal Marlin packs along K (both along the reduction dimension, but column-packed vs row-packed access patterns differ).

2. **Zero Point Storage**: vLLM packs zeros into INT32 (memory efficient), Metal Marlin stores as FP16 (simpler dequant, 2x memory for zeros).

3. **Activation Reordering**: vLLM supports g_idx for GPTQ's activation-order quantization; Metal Marlin does not.

**Format Compatibility**: Cannot directly load vLLM GPTQ/AWQ quants. Would need conversion that:
- Repacks weights from column-major to row-major
- Unpacks zeros from INT32 to FP16
- Handles g_idx reordering if present

## Dequantization Kernel Comparison

### vLLM FP4 (via compressed-tensors)
- Uses CUTLASS/Triton backends
- Supports block-wise quantization
- Per-tensor or per-group scales
- Integrated with scaled_mm kernels

### Metal Marlin FP4 (E2M1)
```metal
// Branchless bitwise dequant (no LUT)
inline half dequant_fp4_bitwise(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1;
    uint exp_bits = (nibble >> 1) & 0x3;
    uint man_bit  = nibble & 1;

    half magnitude;
    if (exp_bits == 0) {
        magnitude = half(man_bit) * half(0.25h);
    } else {
        half power = half(1u << (exp_bits - 1));
        half mantissa = half(1.0h) + half(man_bit) * half(0.5h);
        magnitude = power * mantissa;
    }
    return guard_finite(sign_bit ? -magnitude : magnitude);
}
```

- Pure ALU (no LUT, better for Metal's shared memory constraints)
- Float intermediate to avoid Metal half precision bug
- Guard against NaN/Inf from extreme scales

### vLLM INT4 (GPTQ/AWQ)
```python
# Marlin kernel style
weight = ((qweight >> (i * 4)) & 0xF).to(torch.int8) - 8
dequant = weight * scale
```

### Metal Marlin INT4 (U4)
```metal
// Magic bias trick for fast INT4 extraction
constant constexpr uint32_t FUSED_MAGIC_BIAS = 0x64006400u;
constant constexpr uint32_t FUSED_LO_MASK    = 0x000F000Fu;

inline void fused_dequant_u4x8(uint32_t packed, half scale, half zero_point, thread half* out) {
    half2 bias = as_type<half2>(FUSED_MAGIC_BIAS);
    // Extract 2 nibbles at once using magic bias
    uint32_t n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS;
    half2 v0 = as_type<half2>(n0) - bias;
    // ... extracts 8 nibbles total
}
```

- Magic bias trick avoids 8 separate shifts
- Processes 2 nibbles per instruction
- Uses float intermediates for precision

## Layer Classification Comparison

### vLLM Approach
- Per-module dynamic config (regex patterns)
- Supports mixed precision within model
- Automatic layer detection from module names
- Can exclude specific layers (modules_to_not_convert)

### Metal Marlin Approach (`mixed_precision.py`)
```python
LAYER_PATTERNS = {
    "embeddings": ["embed_tokens", "wte", ...],
    "lm_head": ["lm_head", "output.weight", ...],
    "norms": ["layernorm", "rmsnorm", ...],
    "moe_router": ["router", "gate.weight", ...],
    "moe_experts": ["experts.", "expert.", ...],
    "attention_qkv": ["q_proj", "k_proj", ...],
    ...
}
```

- Pattern-based classification
- Predefined precision presets (quality_first, speed_first, default_moe)
- MoE-aware with shared vs routed expert handling
- MTP head support for speculative decoding

**Recommendation**: Metal Marlin's approach is simpler and covers common cases. Consider adding vLLM-style regex overrides for edge cases.

## MoE Handling Comparison

### vLLM FusedMoE
- Dedicated fused expert kernels
- Per-expert quantization scales
- Supports W8A8/W4A16 for experts
- Router kept at higher precision
- Expert parallel dispatch

### Metal Marlin MoE
- Layer-level dispatch (not fused)
- Shared expert gets tighter quantization (group_size=64)
- Routed experts more aggressive (group_size=128)
- Router kept at BF16/FP16
- No expert parallelism

**Gap**: Metal Marlin lacks fused MoE kernels. For models like Mixtral or GLM-4.7-Flash, this means:
- Extra kernel launches (one per expert)
- No overlap between expert computation
- Higher memory bandwidth for routing

**Recommendation**: Implement fused MoE kernel for Metal:
```
FusedMoEKernel(hidden_states, gate_logits, expert_weights, top_k)
```

## KV Cache Quantization Comparison

### vLLM FP8 KV Cache
- Per-TP-rank, per-layer scales
- E4M3 format (8-bit)
- Integrated with PagedAttention
- 2x memory savings

### Metal Marlin FP4 KV Cache
- Per-row scales
- E2M1 format (4-bit)
- Integrated with paged_attention.metal
- 4x memory savings (but more accuracy loss)

**Trade-off**: Metal Marlin is more aggressive (FP4 vs FP8). For long context:
- FP4 enables 2x longer context than FP8 at same memory
- FP8 provides better accuracy for attention scores

**Recommendation**: Add FP8 KV cache option for quality-sensitive use cases.

## Paged Attention Comparison

### vLLM PagedAttention
- V1: Single-pass for short context
- V2: Multi-partition for long context (>1024 tokens)
- Block size: 16 tokens (configurable)
- Supports variable-length batches

### Metal Marlin Paged Attention
```metal
// paged_attention.metal
constant constexpr uint BLOCK_SIZE = 16;      // Same as vLLM
constant constexpr uint PARTITION_SIZE = 256; // For V2

kernel void paged_attention_v1(...) {
    // Single-pass decode attention
    // 4 simdgroups: 1 compute, 3 load
}

kernel void paged_attention_v2(...) {
    // Multi-partition for long context
    // Partial softmax reduction
}
```

**Parity**: Good alignment with vLLM's design. Key differences:
- vLLM uses CUDA warps (32 threads), Metal uses simdgroups (32 threads) - same
- vLLM has more variants (GQA-specific, etc.)

## Calibration Comparison

### vLLM Calibration (External)
- GPTQ: Hessian-weighted layer-wise
- AWQ: Activation-aware per-group
- Loads pre-calibrated weights from checkpoints

### Metal Marlin Calibration
```python
# calibration.py
class BartowskiCalibration:
    """Downloads Bartowski calibration dataset."""

    @classmethod
    def v3(cls) -> CalibrationDataset:
        # Fetch from GitHub Gist
        # Returns text samples for calibration
```

**Gap**: Metal Marlin lacks runtime calibration. Currently:
- Uses simple max-abs scaling
- No Hessian or activation-aware methods
- Relies on pre-quantized models or simple packing

**Recommendation**: Add calibration support:
1. GPTQ-style layer-wise (for accuracy)
2. AWQ-style activation-aware (for MoE models)
3. Cache calibration stats for reproducibility

## Performance Comparison Methodology

To compare Metal Marlin vs vLLM performance fairly:

### Metrics
1. **Throughput** (tokens/sec): Measure at various batch sizes
2. **Latency** (time-to-first-token): Single request decode start
3. **Memory efficiency** (GB/B params): Memory per billion parameters
4. **Accuracy** (perplexity): WikiText-2 or C4 validation

### Benchmark Suite
```python
# benchmarks/vllm_comparison.py
class ComparisonBenchmark:
    models = [
        "Llama-3.1-8B",      # Dense baseline
        "Mistral-7B",        # GQA
        "Mixtral-8x7B",      # MoE
        "GLM-4-9B-Chat",     # MoE + MTP
    ]

    batch_sizes = [1, 4, 16, 64]
    seq_lengths = [128, 512, 2048, 8192]
    quant_formats = ["fp4", "int4", "fp8"]
```

### Test Protocol
1. **Prefill**: Measure throughput at batch=1, varying seq_len
2. **Decode**: Measure tokens/sec with KV cache
3. **Batched decode**: Throughput vs batch size
4. **Memory**: Peak GPU memory at max context

## Recommendations Summary

### High Priority
1. **Add FP8 KV cache** - vLLM's default, better accuracy than FP4
2. **Implement weight format converter** - Load vLLM GPTQ/AWQ quants
3. **Add calibration pipeline** - GPTQ-style for accuracy

### Medium Priority
4. **Fused MoE kernel** - Critical for Mixtral/DeepSeek performance
5. **Block-wise 2D scales** - For DeepSeek models
6. **Activation quantization** - W8A8 for throughput

### Lower Priority
7. **Tensor parallelism** - Multi-GPU (not Metal's strength)
8. **g_idx reordering** - Full GPTQ compatibility
9. **Dynamic activation quant** - Per-batch FP8 scaling

## Appendix: vLLM Quantization Architecture

### Registration System
```python
@register_quantization_config("custom_quant")
class CustomConfig(QuantizationConfig):
    def get_quant_method(self, layer, prefix) -> QuantizeMethodBase:
        return CustomLinearMethod(self)
```

### Key Abstractions
- `QuantizationConfig`: Config + factory for quant methods
- `QuantizeMethodBase`: apply(), create_weights(), process_weights_after_loading()
- `GroupShape`: (row, col) scale grouping
- `QuantKey`: (dtype, scale_desc, scale2, symmetric)

### Supported Methods (28+)
GPTQ, AWQ, FP8, GPTQ-Marlin, AWQ-Marlin, compressed-tensors, MoE-WNA16, Int8, BitBLAS, GGUF, BitsAndBytes, EETQ, DeepSpeedFP, Experts-Int8, FBGemm-FP8, FP4-NV, GPTBigCode, HQQ, IPEx, Marlin, ModelOpt, Neuron, QQQQQ, QuIP#, SmoothQuant, SqueezeLLM, TorchAO, TPU-Int8
