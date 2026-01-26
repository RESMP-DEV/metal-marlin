# Inference Architecture

Metal Marlin uses a format-agnostic, layered architecture. Instead of hardcoding model implementations (like the old `llama.py`), the system loads any transformer from standard weight formats and executes through a generic graph or protocol-based pipeline.

## Stack Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          User API                                           │
│              MarlinPipeline  /  ONNXExecutor  /  CLI                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Graph Executor                                        │
│         ONNXExecutor: parse graph, dispatch ops to kernels                  │
│         MarlinPipeline: protocol-based forward pass                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Format Loaders                                        │
│    ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐          │
│    │  ONNX        │   │   Safetensors    │   │     GGUF         │          │
│    │  .onnx graph │   │   .safetensors   │   │  Q4_0/Q4_1/Q8_0  │          │
│    │  + weights   │   │   stream+quant   │   │  dequant→FP4     │          │
│    └──────────────┘   └──────────────────┘   └──────────────────┘          │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Quantization Layer                                    │
│         pack_fp4_weights / pack_int4_weights / per-group scales             │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Kernel Library (Metal Shaders)                         │
│    ┌───────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  │
│    │  GEMM     │  │  Attention    │  │  Dequant      │  │  Sparse      │  │
│    │  12 vars  │  │  Flash + GQA  │  │  FP4/INT4/FP8 │  │  2:4 N:M    │  │
│    └───────────┘  └───────────────┘  └───────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                       MLX Runtime                                           │
│              metal_kernel() dispatch, lazy evaluation, unified memory       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Layer 1: Kernel Library (Metal Shaders)

The foundation is a set of Metal compute shaders in `src/`. These are pure GPU kernels with no knowledge of model architectures.

### GEMM Kernels (`marlin_gemm.metal`)

12 kernel variants optimized for different scenarios:

| Kernel | Strategy | Use Case |
|--------|----------|----------|
| `marlin_gemm_fp4` | Double-buffered, 2D dispatch | Standard inference |
| `marlin_gemm_fp4_3stage` | Triple-buffered | High-latency tolerance |
| `marlin_gemm_fp4_single_stage` | Non-pipelined | Reference / debugging |
| `marlin_gemm_fp16_single_stage` | FP16 reference | Measuring dequant overhead |
| `marlin_gemm_fp4_striped` | Stripe-partitioned 1D | Load balancing |
| `marlin_gemm_fused_fp4` | Register-resident dequant | Reduced register pressure |
| `marlin_gemm_fused_u4` | INT4 fused variant | INT4 quantized models |
| `marlin_gemm_fp4_fp32acc` | FP32 accumulator | K > 8192, precision-critical |
| `marlin_gemm_fused_fp4_fp32acc` | Fused + FP32 acc | Combined benefits |
| `marlin_gemm_divergent_fp4` | Divergent load/compute | Simdgroup specialization |
| `marlin_gemm_fp8_e4m3` | W8A16, FP8 weights | FP8 quantized models |
| `marlin_gemm_fused_fp8_e4m3` | Fused W8A16 | FP8 fused variant |

All use 64x64x32 (MxNxK) tile sizes, tuned for Apple M4 Max.

### Attention Kernels (`flash_attention.metal`)

Flash attention with optional KV cache quantization:

- `flash_attention_kv_fp4`: FP4-quantized KV cache
- `flash_attention_kv_int4`: INT4-quantized KV cache
- Grouped-query attention (GQA) support via head mapping

### Dequantization Kernels

- `dequant.metal`: INT4/U4 using magic bias trick (branchless, no LUTs)
- `dequant_int8.metal`: INT8 variants
- `dequant_fp8.metal`: FP8 E4M3 and E5M2 with IEEE-like subnormal handling

### Sparse Kernels

- `sparse_gemm.metal`: 2:4 structured sparsity (NVIDIA-compatible metadata)
- `sparse.metal`: General N:M patterns with compact combination-index decoding

### Supporting Shaders

| File | Purpose |
|------|---------|
| `batched_gemm.metal` | Batch matrix multiply, grouped GQA GEMM |
| `gemm_epilogue.metal` | Bias addition, activation fusion |
| `kernels_autotune.metal` | Template variants for auto-tuning |
| `bf16_compat.metal` | BF16 compatibility layer |

## Layer 2: Format Loaders

Format loaders convert standard weight files into Marlin's packed FP4/INT4 representation. They have no knowledge of model architecture; they operate on raw tensors.

### ONNX (`converters/onnx_executor.py`)

Parses the ONNX protobuf graph and loads initializers (weights) as MLX arrays. Weight tensors from MatMul/Gemm nodes are automatically quantized to FP4 during load.

```python
executor = ONNXExecutor.from_file("model.onnx", quantize=True)
output = executor(input_ids=tokens)
```

### Safetensors (`safetensors_loader.py`)

Streams safetensors files and quantizes weights on-the-fly, producing `.marlin.safetensors` output. Avoids loading the full FP16 model into memory at once.

### GGUF (`gguf_loader.py`)

Standalone GGUF parser (no external `gguf` package). Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 quantization types. Dequantizes GGML format to FP16, then re-quantizes to Marlin FP4 with per-group scales.

```python
from metal_marlin.gguf_loader import load_gguf
weights = load_gguf("model.gguf")  # Returns {name: (packed, scales)}
```

## Layer 3: Graph Executor

Two execution paths, both architecture-agnostic:

### ONNXExecutor

Topologically executes an ONNX graph. Each node dispatches to the appropriate kernel via an op-type table:

```
MatMul / Gemm       → marlin_gemm_fp4 (if weight is quantized)
                    → mx.matmul (fallback for non-quantized)
LayerNormalization  → standard MLX implementation
Softmax             → mx.softmax
Add / Mul           → element-wise ops
Reshape / Transpose → shape manipulation
```

The executor knows nothing about Llama, Mistral, or any other architecture. Any transformer exported to ONNX runs without modification.

### MarlinPipeline (Protocol-Based)

For non-ONNX use, `MarlinPipeline` accepts any model implementing the `MarlinModel` protocol:

```python
@runtime_checkable
class MarlinModel(Protocol):
    def __call__(self, input_ids: mx.array, kv_cache=None) -> mx.array: ...
    def create_kv_cache(self, batch_size: int = 1): ...
```

Models built from `MarlinLinear`, `MarlinMLP`, and `MarlinTransformerBlock` satisfy this protocol automatically. The pipeline handles tokenization, prefill/decode phases, sampling, and streaming.

## Why We Removed Model-Specific Code

The old architecture had `models/llama.py` with a hardcoded `MarlinLlamaForCausalLM` class. This was problematic:

1. **Every new architecture required new code.** Supporting Mistral, Gemma, Qwen, etc. each needed a dedicated Python file duplicating 90% of the same logic.

2. **Config proliferation.** Each model's config (hidden_size, num_heads, intermediate_size) was validated and wired separately, despite all being standard transformer parameters.

3. **Tight coupling to HuggingFace conventions.** Weight names, config keys, and layer ordering were baked in.

The current design eliminates this entirely:

- **ONNX path**: Export any model to ONNX once, run it forever via `ONNXExecutor`. No Python model code needed.
- **Safetensors path**: Load raw weights, build layers from `MarlinLinear` / `MarlinTransformerBlock`. The `ModelConfig` dataclass covers all standard transformer parameters generically.
- **GGUF path**: Convert GGML quantized models directly to Marlin format. Community quantizations work out of the box.

The old `llama.py` is preserved in `_archived/` for reference but is no longer part of the active codebase.

## End-to-End Data Flow

```
                    ┌──────────────────────┐
                    │   Weight File         │
                    │  .onnx / .safetensors │
                    │  / .gguf             │
                    └──────────┬───────────┘
                               │
                     ┌─────────▼─────────┐
                     │   Format Loader    │
                     │   Parse + Quantize │
                     │   → packed FP4     │
                     │   → per-group      │
                     │     scales         │
                     └─────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                   │
            ▼                  ▼                   ▼
  ┌─────────────────┐ ┌───────────────┐ ┌─────────────────┐
  │  ONNXExecutor   │ │MarlinPipeline │ │  Direct Kernel  │
  │  (graph walk)   │ │  (protocol)   │ │    Calls        │
  └────────┬────────┘ └───────┬───────┘ └────────┬────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                     ┌─────────▼──────────┐
                     │   Kernel Dispatch   │
                     │   kernels.py        │
                     │   (MLX wrappers)    │
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │   Metal Shaders     │
                     │   marlin_gemm.metal │
                     │   flash_attention   │
                     │   dequant / sparse  │
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │   Apple GPU         │
                     │   (M4 Max, etc.)    │
                     └────────────────────┘
```

## Prefill / Decode Phases

Both execution paths (ONNX and pipeline) follow the same two-phase pattern:

**Prefill**: Process the full prompt in a single batched pass. All tokens attend to each other. KV cache is populated for all layers. The last logit position produces the first generated token.

**Decode**: One token at a time. Each step appends to the KV cache. Attention is over the full cached context. Sampling (top-p, top-k, temperature, repetition penalty) selects the next token. Continues until EOS or max length.

```python
# ONNX path
executor = ONNXExecutor.from_file("model.onnx")
logits = executor(input_ids=prompt_tokens)

# Pipeline path
pipe = MarlinPipeline(model, tokenizer)
for token in pipe.generate_stream("prompt", max_tokens=256):
    print(tokenizer.decode([token]), end="")
```

## Performance Characteristics

| Phase | Bottleneck | Kernel Selection |
|-------|-----------|-----------------|
| Prefill (large batch) | Memory bandwidth | `marlin_gemm_fp4` (double-buffered) |
| Prefill (very large K) | Accumulation precision | `marlin_gemm_fp4_fp32acc` |
| Decode (M=1) | Compute | `marlin_gemm_fused_fp4` (register-resident) |
| Long context attention | KV cache memory | `flash_attention_kv_fp4` |
| Sparse models | Reduced FLOPS | `marlin_gemm_sparse_2_4` |

The autotune framework (`autotune.py`) selects the optimal kernel variant per (M, N, K) problem size automatically.

## File Map

```
src/
├── marlin_gemm.metal          # 12 GEMM kernel variants
├── flash_attention.metal      # Attention with quantized KV
├── sparse_gemm.metal          # 2:4 structured sparsity
├── dequant.metal              # INT4/U4 dequant (magic bias)
├── dequant_int8.metal         # INT8 dequant
├── dequant_fp8.metal          # FP8 E4M3/E5M2
├── batched_gemm.metal         # Batch multiply, GQA
├── gemm_epilogue.metal        # Bias, activation fusion
├── sparse.metal               # General N:M sparsity
├── bf16_compat.metal          # BF16 layer
├── kernels_autotune.metal     # Autotune templates
└── debug.metal                # Profiling kernels

metal_marlin/
├── kernels.py                 # MLX metal_kernel() wrappers
├── layers.py                  # MarlinLinear (quantized nn.Linear)
├── mlp.py                     # MarlinMLP (SwiGLU / standard)
├── transformer.py             # MarlinTransformerBlock, RMSNorm
├── attention.py               # MarlinAttention + RoPE
├── kv_cache.py                # KV cache with optional quantization
├── quantize.py                # Weight packing (FP4, INT4)
├── inference.py               # MarlinPipeline (protocol-based)
├── generate.py                # Sampling, beam search, streaming
├── gguf_loader.py             # GGUF format support
├── safetensors_loader.py      # Safetensors streaming loader
├── autotune.py                # Per-problem-size kernel selection
└── cli.py                     # Command-line interface

converters/
└── onnx_executor.py           # ONNX graph → Metal Marlin dispatch

_archived/
└── llama.py                   # Old model-specific code (deprecated)
```
