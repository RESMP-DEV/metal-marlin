# Inference Architecture

Metal Marlin uses a format-agnostic, layered architecture. Instead of hardcoding model implementations, the system integrates with HuggingFace Transformers by replacing `nn.Linear` layers, and it can also execute ONNX graphs directly.

## Stack Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          User API                                           │
│      Transformers + replace_linear_layers  /  ONNXExecutor  /  CLI          │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Model Integration                                     │
│  Transformers: forward/generate with MetalQuantizedLinear                   │
│  ONNXExecutor: parse graph, dispatch ops to kernels                         │
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
│                       Metal Runtime                                          │
│              Kernel dispatch, unified memory, Apple GPU execution            │
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

Format loaders and conversion utilities transform standard weight files into Marlin's packed FP4/INT4 representation. They have no knowledge of model architecture; they operate on raw tensors.

### ONNX (`converters/onnx_executor.py`, `metal_marlin/onnx_loader.py`)

Parses the ONNX protobuf graph and loads initializers (weights). Weight tensors from MatMul/Gemm nodes can be quantized to FP4 during load.

```python
executor = ONNXExecutor.from_file("model.onnx", quantize=True)
output = executor(input_ids=tokens)
```

### Safetensors (`metal_marlin/safetensors_loader.py`)

Streams safetensors files and quantizes weights on-the-fly, producing `.marlin.safetensors` output. Avoids loading the full FP16 model into memory at once.

### GGUF (`gguf_loader.py`)

Standalone GGUF parser (no external `gguf` package). Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 quantization types. Dequantizes GGML format to FP16, then re-quantizes to Marlin FP4 with per-group scales.

```python
from metal_marlin.gguf_loader import load_gguf
weights = load_gguf("model.gguf")  # Returns {name: (packed, scales)}
```

## Layer 3: Model Integration

Two execution paths, both architecture-agnostic:

### Transformers Integration (Recommended)

Use HuggingFace Transformers for model structure and generation, then swap
`nn.Linear` layers to Metal-backed quantized layers with `replace_linear_layers()`.
This keeps the model code untouched while routing GEMM-heavy ops through Metal.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from metal_marlin import replace_linear_layers
from metal_marlin.inference import TransformersMarlinPipeline

model = AutoModelForCausalLM.from_pretrained("your-model-id")
tokenizer = AutoTokenizer.from_pretrained("your-model-id")
replace_linear_layers(model, bits=4, group_size=128)

pipe = TransformersMarlinPipeline(model, tokenizer)
output = pipe("Prompt", max_tokens=128)
```

### ONNXExecutor

Topologically executes an ONNX graph. Each node dispatches to the appropriate kernel via an op-type table:

```
MatMul / Gemm       → marlin_gemm_fp4 (if weight is quantized)
                    → dense_gemm (fallback for non-quantized)
LayerNormalization  → standard implementation
Softmax             → softmax kernel
Add / Mul           → element-wise ops
Reshape / Transpose → shape manipulation
```

The executor knows nothing about Llama, Mistral, or any other architecture. Any transformer exported to ONNX runs without modification.

## Why We Avoid Model-Specific Code

Model-specific Python implementations do not scale: every new architecture would
require a dedicated file, duplicate logic, and hard-code weight naming
conventions. The current design removes this entire class of work:

- **Transformers path**: Use HuggingFace models as-is and swap `nn.Linear` layers
  with `replace_linear_layers()`. No model reimplementation.
- **ONNX path**: Export any model to ONNX once and run it forever via `ONNXExecutor`.
- **GGUF path**: Convert GGML quantized models directly to Marlin format. Community
  quantizations work out of the box.

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
  ┌─────────────────┐ ┌────────────────────────────┐ ┌─────────────────┐
  │  ONNXExecutor   │ │ Transformers + layer swap  │ │  Direct Kernel  │
  │  (graph walk)   │ │  replace_linear_layers()   │ │    Calls        │
  └────────┬────────┘ └──────────────┬─────────────┘ └────────┬────────┘
           │                          │                        │
           └──────────────────────────┼────────────────────────┘
                               │
                     ┌─────────▼──────────┐
                     │   Kernel Dispatch   │
                     │   kernels.py        │
                     │   (Metal wrappers)  │
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

Both execution paths (ONNX and Transformers integration) follow the same two-phase pattern:

**Prefill**: Process the full prompt in a single batched pass. All tokens attend to each other. KV cache is populated for all layers. The last logit position produces the first generated token.

**Decode**: One token at a time. Each step appends to the KV cache. Attention is over the full cached context. Sampling (top-p, top-k, temperature, repetition penalty) selects the next token. Continues until EOS or max length.

```python
# ONNX path
executor = ONNXExecutor.from_file("model.onnx")
logits = executor(input_ids=prompt_tokens)

# Transformers path
from transformers import AutoModelForCausalLM, AutoTokenizer
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained("your-model-id")
tokenizer = AutoTokenizer.from_pretrained("your-model-id")
replace_linear_layers(model, bits=4, group_size=128)
model.to("mps")

inputs = tokenizer("prompt", return_tensors="pt").to("mps")
output_ids = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
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

## File Map (Key Files)

```
src/
├── marlin_gemm.metal          # GEMM kernel variants
├── dense_gemm.metal           # FP16/FP32 dense GEMM fallback
├── flash_attention.metal      # Attention with quantized KV
├── flash_attention_v2.metal   # Updated flash attention kernels
├── attention.metal            # Attention primitives
├── dequant.metal              # INT4/U4 dequant (magic bias)
├── dequant_int8.metal         # INT8 dequant
├── dequant_fp8.metal          # FP8 E4M3/E5M2
├── dequant_sub4bit.metal      # Sub-4bit dequant paths
├── batched_gemm.metal         # Batch multiply, GQA
├── gemm_epilogue.metal        # Bias, activation fusion
├── sparse_gemm.metal          # 2:4 structured sparsity
├── sparse.metal               # General N:M sparsity
├── bf16_compat.metal          # BF16 layer
└── kernels_autotune.metal     # Autotune templates

metal_marlin/
├── layer_replacement.py       # replace_linear_layers() / MetalQuantizedLinear
├── transformers_loader.py     # HF model loading + layer swap helpers
├── quantize_model.py          # Convenience wrapper for layer replacement
├── inference/pipeline_v2.py   # TransformersMarlinPipeline
├── inference_metal.py         # MetalQuantizedLinear + kernel dispatch
├── kernels.py                 # Metal kernel wrappers
├── kv_cache.py                # KV cache with optional quantization
├── quantize.py                # Weight packing (FP4/INT4/FP8)
├── onnx_graph.py              # ONNX graph parsing utilities
├── onnx_loader.py             # ONNX weight loading helpers
├── gguf_loader.py             # GGUF format support
├── safetensors_loader.py      # Safetensors streaming loader
├── generate.py                # Sampling + streaming utilities
└── cli.py                     # Command-line interface

converters/
├── onnx_executor.py           # ONNX graph → Metal Marlin dispatch
└── safetensors_loader.py      # HF config mapping + quantization
```
