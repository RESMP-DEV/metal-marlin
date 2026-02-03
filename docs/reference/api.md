# API Reference

Metal Marlin provides native Metal compute kernels for quantized LLM inference on Apple Silicon. All APIs use PyTorch MPS tensors with custom Metal shaders dispatched via PyObjC.

## Transformers Integration (Recommended)

Metal Marlin integrates directly with HuggingFace Transformers by replacing
`nn.Linear` layers with Metal-accelerated quantized layers. This keeps
Transformer generation logic intact while swapping in Metal kernels for GEMM.

### `replace_linear_layers`

```python
replace_linear_layers(
    model: torch.nn.Module,
    bits: Literal[2, 4, 8] = 4,
    group_size: int = 128,
    skip_layers: set[str] | None = None,
    layer_filter: Callable[[str, nn.Module], bool] | None = None,
) -> dict[str, int]
```

Walks the model and replaces eligible `nn.Linear` layers with
`MetalQuantizedLinear` variants in-place.

**Returns:**
- `replaced_count`: number of layers replaced
- `skipped_count`: number of layers skipped (shape or filter mismatch)

**Example:**
```python
from transformers import AutoModelForCausalLM
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash")
stats = replace_linear_layers(model, bits=4, group_size=128)
print(stats)
```

### `load_and_quantize`

```python
load_and_quantize(
    model_id_or_path: str,
    bits: Literal[2, 4, 8] = 4,
    group_size: int = 128,
    device: str = "mps",
) -> tuple[torch.nn.Module, dict[str, int]]
```

Convenience wrapper to load a Transformers model and immediately quantize
its linear layers.

### `save_quantized`

```python
save_quantized(model: torch.nn.Module, output_dir: str | Path) -> None
```

Save a quantized model to disk (weights + config) for fast reloads.

### `load_quantized`

```python
load_quantized(path: str | Path, device: str = "mps") -> torch.nn.Module
```

Load a previously quantized checkpoint from disk.

For migration details from legacy classes, see `migration_guide.md`.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- PyTorch 2.0+ with MPS backend
- PyObjC Metal bindings: `pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders`

---

## Metal Kernel Dispatch

### `MetalKernelLibrary`

```python
from metal_marlin.metal_dispatch import MetalKernelLibrary

class MetalKernelLibrary:
    def __init__(self, device: MTLDevice | None = None)

    @classmethod
    def from_source_dir(cls, src_dir: Path | None = None) -> MetalKernelLibrary

    def compile_source(self, name: str, source: str) -> MTLLibrary

    def get_pipeline(
        self,
        function_name: str,
        library_name: str | None = None,
    ) -> MTLComputePipelineState

    def list_functions(self, library_name: str) -> list[str]

    @property
    def device(self) -> MTLDevice

    @property
    def command_queue(self) -> MTLCommandQueue
```

Manages compilation of `.metal` source files and provides access to compute pipeline states for kernel dispatch.

**Example:**
```python
lib = MetalKernelLibrary.from_source_dir()
pipeline = lib.get_pipeline("marlin_gemm_fp4")
```

---

### `dispatch_kernel`

```python
dispatch_kernel(
    lib: MetalKernelLibrary,
    function_name: str,
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    buffers: Sequence[MTLBuffer],
    wait: bool = True,
) -> None
```

Low-level dispatch for Metal compute kernels.

**Parameters:**
- `lib`: MetalKernelLibrary with compiled shaders
- `function_name`: Kernel function to dispatch
- `grid`: Grid dimensions (threadgroups in X, Y, Z)
- `threadgroup`: Threadgroup dimensions (threads in X, Y, Z)
- `buffers`: Sequence of MTLBuffer arguments (in order)
- `wait`: If True, wait for kernel completion

---

### `get_default_library`

```python
get_default_library() -> MetalKernelLibrary
```

Get or create the default kernel library singleton. Compiles all shaders in `metal_marlin/src/` on first call.

---

## Quantized GEMM Operations

### `dispatch_gemm_fp4`

```python
dispatch_gemm_fp4(
    lib: MetalKernelLibrary,
    A: torch.Tensor,           # [M, K], fp16, MPS
    B_packed: torch.Tensor,    # [K//8, N], uint32, MPS
    scales: torch.Tensor,      # [K//group_size, N], fp16, MPS
    M: int,
    N: int,
    K: int,
    group_size: int = 32,
) -> torch.Tensor              # [M, N], fp16, MPS
```

FP4 E2M1 quantized GEMM: `C = A @ dequant(B_packed, scales)`

**Parameters:**
- `A`: Input activations
- `B_packed`: Packed FP4 weights (8 values per uint32)
- `scales`: Per-group scales for dequantization
- `M`: Rows of A (batch * seq for transformers)
- `N`: Output features
- `K`: Input features
- `group_size`: Quantization group size

**Example:**
```python
lib = get_default_library()
M, N, K = 32, 4096, 4096

A = torch.randn(M, K, dtype=torch.float16, device="mps")
B_packed = torch.randint(0, 2**32, (K//8, N), dtype=torch.uint32, device="mps")
scales = torch.randn(K//32, N, dtype=torch.float16, device="mps") * 0.1

C = dispatch_gemm_fp4(lib, A, B_packed, scales, M, N, K, group_size=32)
```

---

### `dispatch_gemm_fp8`

```python
dispatch_gemm_fp8(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
) -> torch.Tensor
```

FP8 E4M3 quantized GEMM. Higher precision than FP4, 4 values packed per uint32.

---

### `dispatch_gemm_int2`

```python
dispatch_gemm_int2(
    lib: MetalKernelLibrary,
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
) -> torch.Tensor
```

INT2 quantized GEMM for extreme compression (cold MoE experts). 16 values packed per uint32.

---

## Layer Classes

### `MetalQuantizedLinear`

```python
class MetalQuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        bias: bool = False,
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
    ) -> MetalQuantizedLinear
```

Drop-in replacement for `nn.Linear` using Metal quantized GEMM kernels.

**Supported quantization formats:**
- `bits=4`: FP4 E2M1 (default, 8 values per uint32)
- `bits=8`: FP8 E4M3 (4 values per uint32)
- `bits=2`: INT2 symmetric (16 values per uint32)

**Example:**
```python
from metal_marlin.inference_metal import MetalQuantizedLinear

# Create from scratch
layer = MetalQuantizedLinear(4096, 11008, bits=4, group_size=128)

# Quantize an existing linear layer
original = nn.Linear(4096, 11008)
quantized = MetalQuantizedLinear.from_float(original, bits=4)
```

---

### `MetalAttention`

```python
class MetalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
        bias: bool = False,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: MetalKVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor
```

Multi-head attention with quantized projections.

**Features:**
- Standard MHA (`num_heads == num_kv_heads`)
- Grouped Query Attention (`num_kv_heads < num_heads`)
- RoPE position embeddings with configurable base and ratio
- KV caching for autoregressive generation
- Uses PyTorch `scaled_dot_product_attention` (Metal-accelerated on MPS)

---

### `MetalMLAAttention`

```python
class MetalMLAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 64,
        head_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        bias: bool = False,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor
```

Multi-head Latent Attention for compressed KV cache (GLM-4.7-Flash, DeepSeek).

---

### `MetalMoELayer`

```python
class MetalMoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
    )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor
```

Mixture of Experts layer with top-k routing and quantized expert MLPs.

---

### `MetalTransformerBlock`

```python
class MetalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: int | None = None,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        max_position_embeddings: int = 4096,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: MetalKVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor
```

Pre-norm transformer decoder block: `x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +`

---

## Token Sampling

### `MetalSampler`

```python
class MetalSampler:
    def __init__(
        self,
        vocab_size: int,
        lib: MetalKernelLibrary | None = None,
        seed: int | None = None,
    )

    def argmax(self, logits: torch.Tensor) -> int

    def sample_categorical(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> int

    def sample_top_k(
        self,
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> int

    def sample_top_p(
        self,
        logits: torch.Tensor,
        p: float,
        temperature: float = 1.0,
    ) -> int

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float,
    ) -> torch.Tensor

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        generated_ids: list[int] | None = None,
    ) -> int
```

Metal-accelerated token sampler using GPU kernels.

**Sampling methods:**
- `argmax`: Greedy decoding
- `sample_categorical`: Gumbel-max trick for GPU sampling
- `sample_top_k`: Top-k filtering
- `sample_top_p`: Nucleus (top-p) sampling
- `sample`: Combined strategy with all modifiers

**Example:**
```python
from metal_marlin.sampler import MetalSampler

sampler = MetalSampler(vocab_size=32000)

# Greedy decoding
token_id = sampler.argmax(logits)

# Temperature + top-p sampling
token_id = sampler.sample(logits, temperature=0.7, top_p=0.9)

# With repetition penalty
token_id = sampler.sample(
    logits,
    temperature=0.7,
    top_p=0.9,
    generated_ids=[1, 2, 3],
    repetition_penalty=1.1
)
```

---

## Generation API

### `generate`

```python
from metal_marlin.generate import generate, GenerationConfig

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    eos_token_id: int = 2
    pad_token_id: int = 0
    do_sample: bool = True

def generate(
    model: CausalLM,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    kv_cache: KVCacheTorch | None = None,
    streamer: Callable[[int], None] | None = None,
) -> torch.Tensor
```

High-level generation function for any model implementing the `CausalLM` protocol.

**Example:**
```python
config = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=0.9)
output_ids = generate(model, input_ids, config=config)
```

---

## KV Cache

### `MetalKVCache`

```python
@dataclass
class MetalKVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype = torch.float16

class MetalKVCache:
    def __init__(self, config: MetalKVCacheConfig, batch_size: int = 1)

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]

    def advance(self, num_tokens: int = 1) -> None

    def reset(self) -> None
```

Pre-allocated KV cache for efficient autoregressive generation.

---

## MoE Dispatch Functions

### `dispatch_moe_optimized`

```python
dispatch_moe_optimized(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,         # [batch, hidden_dim]
    router_weights: torch.Tensor,      # [hidden_dim, num_experts]
    expert_weights: torch.Tensor,      # [num_experts, K/8, N]
    expert_scales: torch.Tensor,       # [num_experts, K/group_size, N]
    shared_weights: torch.Tensor | None,
    shared_scales: torch.Tensor | None,
    batch_size: int,
    hidden_dim: int,
    out_dim: int,
    num_experts: int = 64,
    top_k: int = 4,
    group_size: int = 128,
) -> torch.Tensor
```

Single-kernel fused routing + GEMM + combination. Optimized for GLM-4.7-Flash with 64 experts, top-k=4, and shared expert.

---

### `dispatch_moe_decode`

```python
dispatch_moe_decode(
    lib: MetalKernelLibrary,
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    shared_weights: torch.Tensor | None,
    shared_scales: torch.Tensor | None,
    hidden_dim: int,
    out_dim: int,
    num_experts: int = 64,
    top_k: int = 4,
    group_size: int = 128,
) -> torch.Tensor
```

Single-token MoE optimized for minimal latency in autoregressive generation.

---

## FP4 Dequantization

### `dispatch_dequant_fp4`

```python
dispatch_dequant_fp4(
    lib: MetalKernelLibrary,
    packed: torch.Tensor,      # [K/8, N], uint32
    scales: torch.Tensor,      # [K/group_size, N], fp16
    K: int,
    N: int,
    group_size: int = 32,
) -> torch.Tensor              # [K, N], fp16
```

Row-major optimized FP4 dequantization kernel.

---

### `benchmark_dequant_fp4`

```python
benchmark_dequant_fp4(
    lib: MetalKernelLibrary,
    num_packed: int = 16 * 1024 * 1024,
    group_size: int = 32,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> dict[str, float]
```

Benchmark FP4 dequantization bandwidth. Returns `time_ms`, `bandwidth_gb_s`, `throughput_gop_s`.

---

## Buffer Interop

### `mps_tensor_to_metal_buffer`

```python
mps_tensor_to_metal_buffer(
    tensor: torch.Tensor,
    device: MTLDevice,
) -> MTLBuffer
```

Get Metal buffer from PyTorch MPS tensor (zero-copy, shared memory).

---

### `metal_buffer_to_numpy`

```python
metal_buffer_to_numpy(
    buffer: MTLBuffer,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray
```

Read Metal buffer contents to numpy array (copy).

---

## Vision Operations

Metal-accelerated vision preprocessing operations. Replaces CPU-bound PIL/torchvision
resize/normalize pipelines with Metal compute shaders.

### `VisionMetal`

```python
from metal_marlin.vision.vision_metal import VisionMetal

class VisionMetal:
    def __init__(self, device: torch.device | None = None)

    def uint8_to_float(self, image: torch.Tensor) -> torch.Tensor
    
    def center_crop(
        self,
        image: torch.Tensor,
        size: tuple[int, int],
        nhwc: bool = False,
    ) -> torch.Tensor

    def resize_bilinear(
        self,
        input: torch.Tensor,
        target_size: tuple[int, int],
        nhwc: bool = False,
    ) -> torch.Tensor

    def resize_bicubic(
        self,
        input: torch.Tensor,
        target_size: tuple[int, int],
        nhwc: bool = False,
    ) -> torch.Tensor

    def normalize(
        self,
        image: torch.Tensor,
        mean: tuple[float, ...] | list[float] | torch.Tensor,
        std: tuple[float, ...] | list[float] | torch.Tensor,
        nhwc: bool = False,
    ) -> torch.Tensor

    def resize_and_normalize(
        self,
        image: torch.Tensor,
        size: tuple[int, int],
        mean: tuple[float, ...] | list[float] | torch.Tensor,
        std: tuple[float, ...] | list[float] | torch.Tensor,
        nhwc: bool = False,
    ) -> torch.Tensor

    def extract_patches(
        self,
        image: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor

    def preprocess_qwen2vl(
        self,
        image: torch.Tensor,
        max_pixels: int = 1024 * 1024,
        patch_size: int = 14,
        nhwc: bool = False,
    ) -> torch.Tensor
```

---

### `preprocess_for_vit`

```python
from metal_marlin.vision.vision_metal import preprocess_for_vit

def preprocess_for_vit(
    images: list[torch.Tensor] | torch.Tensor,
    target_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor
```

---

## Deprecated Classes (Legacy)

These classes are deprecated and will be removed in a future major release.
Use the Transformers integration (`replace_linear_layers`) or
`TransformersMarlinPipeline` instead. See `migration_guide.md` for migration steps.

### `MetalGLM47Model`

**Deprecated:** Use `AutoModelForCausalLM` + `replace_linear_layers()` instead.

```python
class MetalGLM47Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        intermediate_size: int = 11008,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_ratio: float = 1.0,
        bits: Literal[2, 4, 8] = 4,
        group_size: int = 128,
    )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: MetalKVCache | None = None,
    ) -> torch.Tensor

    def create_kv_cache(self, batch_size: int = 1) -> MetalKVCache

    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 2,
        streamer: Callable[[int], None] | None = None,
    ) -> torch.Tensor

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: MetalGenerationConfig,
    ) -> Iterator[int]

    @classmethod
    def from_quantized(
        cls,
        model_path: Path | str,
        bits: Literal[2, 4, 8] = 4,
    ) -> MetalGLM47Model
```

**Recommended replacement:**
```python
from transformers import AutoModelForCausalLM
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash")
replace_linear_layers(model, bits=4, group_size=128)
```

---

### `QuantizedLlamaLayer`

**Deprecated:** Use `AutoModelForCausalLM` + `replace_linear_layers()` instead.

```python
class QuantizedLlamaLayer(nn.Module):
    def __init__(
        self,
        quantized_model: QuantizedModel,
        layer_idx: int,
        warn_if_standalone: bool = True,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor
```

**Recommended replacement:**
```python
from transformers import AutoModelForCausalLM
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
replace_linear_layers(model, bits=4, group_size=128)
```

---

### `MarlinPipeline`

**Deprecated:** Use `TransformersMarlinPipeline` instead (or Transformers +
`replace_linear_layers()` for manual control).

```python
class MarlinPipeline:
    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        device: str | None = "mps",
    ) -> None

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        quant_type: str = "fp4",
        device: str | None = None,
        **kwargs: Any,
    ) -> MarlinPipeline

    def __call__(
        self,
        prompt: str | list[str],
        **kwargs: Any,
    ) -> str | Iterator[str] | list[str]
```

**Recommended replacement:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from metal_marlin import replace_linear_layers

model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash")
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
replace_linear_layers(model, bits=4, group_size=128)
```
