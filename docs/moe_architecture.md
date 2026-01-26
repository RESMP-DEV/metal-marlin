# MoE Kernel Architecture

Metal Marlin's Mixture-of-Experts (MoE) strategy is fundamentally shaped by Apple Silicon's Unified Memory architecture. On traditional discrete GPUs, expert weights live in VRAM and MoE execution involves costly PCIe transfers or sophisticated expert caching. Apple Silicon eliminates this constraint entirely: all expert weights are always resident in the same memory pool as the CPU, with no transfer penalty.

This document describes the kernel design, dispatch strategy, and performance characteristics for MoE inference on Metal.

## Why Unified Memory Changes Everything

### Traditional GPU MoE Challenges

On NVIDIA GPUs with discrete VRAM, MoE models face a fundamental tension:

```
+------------------+                    +------------------+
|   System RAM     |  <-- PCIe 4.0 -->  |   GPU VRAM       |
|  (infinite but   |     ~32 GB/s       |  (fast but       |
|   slow access)   |                    |   limited: 24GB) |
+------------------+                    +------------------+
```

For a model like Mixtral-8x7B (8 experts, 7B each = 56B total parameters), you cannot fit all experts in VRAM. Solutions include:
- **Expert offloading**: Page experts in/out (latency disaster)
- **Expert parallelism**: Distribute across GPUs (needs NVLink)
- **Capacity limits**: Only run smaller MoE models

### Apple Silicon Unified Memory

```
+------------------------------------------------------------------+
|                    Unified Memory (up to 192GB)                    |
|                                                                    |
|  +--------------+  +--------------+  +------------------------+   |
|  | CPU Cores    |  | GPU Cores    |  | Neural Engine          |   |
|  | (full access)|  | (full access)|  | (specialized tensor)   |   |
|  +--------------+  +--------------+  +------------------------+   |
|                                                                    |
|  Memory bandwidth: 400+ GB/s (M4 Max), 800+ GB/s (M4 Ultra)       |
+------------------------------------------------------------------+
```

Implications for MoE:
1. **All experts always resident**: No offloading needed, no PCIe bottleneck
2. **Instant expert access**: Any expert can be accessed with L2/DRAM latency
3. **No expert parallelism needed**: Single-device can run 64+ expert models
4. **Cache-friendly patterns**: Focus on maximizing cache reuse, not hiding latency

The design principle: **Don't copy expert weights; just index into them.**

## MoE Execution Flow

### Token-to-Expert Dispatch

```
                                Input Tokens
                                [batch, hidden]
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      Router GEMM        │
                         │  [batch, hidden] @      │
                         │  [hidden, num_experts]  │
                         │  → [batch, num_experts] │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │     Softmax + Top-K     │
                         │  expert_ids [batch, k]  │
                         │  expert_probs [batch, k]│
                         └───────────┬─────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
                 ▼                   ▼                   ▼
    ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
    │  Expert 0 GEMM     │ │  Expert 1 GEMM     │ │  Expert 63 GEMM    │
    │  (only for tokens  │ │  (only for tokens  │ │  (only for tokens  │
    │   assigned to E0)  │ │   assigned to E1)  │ │   assigned to E63) │
    └──────────┬─────────┘ └──────────┬─────────┘ └──────────┬─────────┘
               │                      │                      │
               └──────────────────────┴──────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Weighted Aggregation  │
                         │  output[t] = Σ prob[e]  │
                         │            × expert[e]  │
                         │                (x[t])   │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                              Output Tokens
                              [batch, hidden]
```

### Shared Expert Pattern (GLM-4, Qwen3-MoE)

Some architectures add a "shared expert" that processes all tokens:

```
┌──────────────────────────────────────────────────────────────────┐
│                         MoE Layer                                 │
│                                                                   │
│  Input x [batch, hidden]                                         │
│       │                                                          │
│       ├──────────────────────┐                                   │
│       │                      │                                   │
│       ▼                      ▼                                   │
│  ┌──────────────┐    ┌─────────────────────────────────┐        │
│  │Shared Expert │    │        Routed Experts            │        │
│  │  (always)    │    │  (variable: top-k per token)     │        │
│  └──────┬───────┘    └──────────────┬──────────────────┘        │
│         │                           │                            │
│         │                           │                            │
│         ▼                           ▼                            │
│  shared_out              Σ prob[e] × expert_out[e]               │
│         │                           │                            │
│         └───────────┬───────────────┘                            │
│                     │                                            │
│                     ▼                                            │
│              shared_out + routed_sum                             │
│                     │                                            │
│                     ▼                                            │
│              Output [batch, hidden]                              │
└──────────────────────────────────────────────────────────────────┘
```

## Kernel Design: Batched vs Per-Token

### The Naive Approach (Per-Token)

```python
# Pseudo-code: per-token dispatch
for token_idx in range(batch_size):
    for k in range(top_k):
        expert_id = expert_ids[token_idx, k]
        prob = expert_probs[token_idx, k]

        # Individual small GEMM (inefficient)
        expert_out = expert_weights[expert_id] @ activations[token_idx]
        output[token_idx] += prob * expert_out
```

Problems:
- **Kernel launch overhead**: One kernel per token × expert
- **Poor GPU utilization**: Single-token GEMMs are memory-bound
- **No batching benefit**: Cannot amortize load/store across tokens

### Batched Expert GEMM (Planned)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dynamic Token Grouping                                │
│                                                                          │
│  Before grouping:              After grouping (by expert):              │
│  Token 0 → Expert 3, 7         Expert 0: [Token 2, Token 5, Token 9]    │
│  Token 1 → Expert 1, 4         Expert 1: [Token 1, Token 6]             │
│  Token 2 → Expert 0, 5         Expert 2: [Token 3, Token 8]             │
│  Token 3 → Expert 2, 6         Expert 3: [Token 0, Token 4, Token 7]    │
│  ...                           ...                                       │
│                                                                          │
│  Benefit: Each expert processes a batch of tokens, enabling              │
│           efficient GEMM with good compute utilization.                  │
└─────────────────────────────────────────────────────────────────────────┘
```

The batched kernel processes all tokens assigned to an expert in one operation:

```metal
// Planned: moe_expert_gemm.metal
kernel void moe_expert_gemm_fp4(
    device const half* activations,     // [batch, hidden]
    device const uint* expert_weights,  // [num_experts, out/8, in] packed FP4
    device const half* scales,          // [num_experts, num_groups, in]
    device const uint* expert_ids,      // [batch, top_k]
    device const half* expert_probs,    // [batch, top_k]
    device half* output,                // [batch, out]
    constant MoEParams& params,
    ...
)
```

Key optimizations:
1. **Token grouping on CPU**: Sort tokens by expert assignment before GPU dispatch
2. **Single kernel launch**: All experts processed in one dispatch
3. **Coalesced access**: Grouped tokens access contiguous weight regions
4. **Fused weighting**: Multiply by expert probability in the kernel epilogue

## Router Fusion Rationale

### Unfused Router (Current State)

```
Step 1: Router GEMM
        [batch, hidden] @ [hidden, num_experts] → [batch, num_experts]
        ↓ (write to global memory)

Step 2: Softmax
        Read [batch, num_experts], compute softmax, write back
        ↓ (write to global memory)

Step 3: Top-K Selection
        Read [batch, num_experts], find top-k indices and values
        ↓ (write expert_ids, expert_probs)

Total: 3 kernel launches, 3× global memory round-trips
```

### Fused Router Kernel (Planned)

```
┌──────────────────────────────────────────────────────────────────┐
│              moe_router_topk_fused                                │
│                                                                   │
│  Input: hidden [batch, hidden_dim]                               │
│         router_weights [hidden_dim, num_experts]                 │
│                                                                   │
│  Per threadgroup (one row of output):                            │
│    1. Compute GEMM: hidden[row] @ router_weights → logits[64]   │
│    2. Find max (for softmax stability)                           │
│    3. Compute exp(logits - max), accumulate sum                  │
│    4. Divide by sum (in-register softmax)                        │
│    5. Partial sort for top-k (k typically 2-8)                   │
│    6. Renormalize top-k probabilities                            │
│    7. Write expert_ids[row, k], expert_probs[row, k]            │
│                                                                   │
│  Output: expert_ids [batch, top_k]  uint32                       │
│          expert_probs [batch, top_k] half                        │
└──────────────────────────────────────────────────────────────────┘
```

Why fusion matters for routers:
1. **Router is tiny**: Typically 4096 hidden × 64 experts = 256K elements
2. **Bandwidth-bound**: Three separate kernels spend more time on memory than compute
3. **Top-k is cheap**: For k ≤ 8, partial sorting is O(k × num_experts)
4. **BF16 accumulation**: Router softmax benefits from BF16's larger dynamic range

## Memory Layout for Expert Weights

### Single Expert Weight Matrix

```
Expert e weight matrix: [out_features, in_features]

Packed (FP4): [out_features, in_features // 8] as uint32
              Each uint32 holds 8 consecutive weights along K

Scales: [num_groups, out_features]
        One scale per group_size elements along K, per output row

Memory layout for expert e:
┌────────────────────────────────────────────────────────────────────┐
│  B_packed[e, out_row, k_pack]                                      │
│                                                                    │
│  k_pack = 0:  weights[out_row, 0:8]   packed into uint32          │
│  k_pack = 1:  weights[out_row, 8:16]  packed into uint32          │
│  ...                                                               │
│  k_pack = K/8: weights[out_row, K-8:K] packed into uint32         │
└────────────────────────────────────────────────────────────────────┘
```

### All Experts Stacked

```
expert_weights: [num_experts, out_features, in_features // 8]

Memory view (contiguous):
┌─────────────────────────────────────────────────────────────────────┐
│ Expert 0                                                            │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ out=0: [k0-7][k8-15][k16-23]...[kN-8 to kN-1]                   │ │
│ │ out=1: [k0-7][k8-15][k16-23]...[kN-8 to kN-1]                   │ │
│ │ ...                                                              │ │
│ │ out=M: [k0-7][k8-15][k16-23]...[kN-8 to kN-1]                   │ │
│ └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ Expert 1                                                            │
│ └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ ...                                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ Expert 63                                                           │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

Scale layout: [num_experts, num_groups, out_features]
              Indexed as: scales[expert_id, k // group_size, out_col]
```

### Indexing Strategy

On Unified Memory, expert selection is just pointer arithmetic:

```metal
// Accessing expert e's weight tile
device const uint* expert_base = expert_weights +
    e * (out_features * (in_features / 8));

// Reading packed weights for expert e, output row out_row, k-group k_pack
uint packed = expert_base[out_row * (in_features / 8) + k_pack];
```

No copying, no staging buffers. The GPU indexes directly into the unified address space.

## Expert Caching Strategy

### Why Cache Dequantized Tiles?

Even with Unified Memory, dequantization has a cost:

```
Memory Bandwidth:     400 GB/s (M4 Max)
Dequant Throughput:   ~200B weights/cycle (per simdgroup)
                      = 64 weights × 4 simdgroups × 0.8 (utilization)

For FP4: 8 weights/uint32, each needs ALU ops to dequant
        Reading packed + dequant > reading dequanted FP16
```

The insight: **Dequantization is the bottleneck, not memory transfer.**

### LRU Dequant Tile Cache (Planned)

```
┌────────────────────────────────────────────────────────────────────┐
│                    ExpertCache                                      │
│                                                                     │
│  Cache: dict[(expert_id, tile_idx), dequantized_tile]              │
│  Size: 512MB (configurable)                                        │
│  Policy: LRU eviction                                              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Tile 0 (E3)  │  Tile 2 (E7)  │  Tile 5 (E3)  │ ... │ LRU │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         ▲                                                           │
│         │                                                           │
│  On cache miss:                                                     │
│    1. Read packed tile from expert_weights                         │
│    2. Dequantize tile to FP16                                      │
│    3. Insert into cache, evict LRU if full                         │
│    4. Return dequantized tile                                      │
│                                                                     │
│  On cache hit:                                                      │
│    1. Return cached dequantized tile directly                      │
│    2. Update LRU order                                             │
└────────────────────────────────────────────────────────────────────┘
```

Cache benefit analysis:
- **Temporal locality**: Same experts often selected for nearby tokens
- **Spatial locality**: Expert tiles accessed in order during GEMM
- **Batch locality**: Grouped tokens for same expert reuse tiles

## Performance Characteristics

### Kernel Selection Strategy

| Phase | Compute Pattern | Bottleneck | Recommended Kernel |
|-------|-----------------|------------|-------------------|
| Router | Small GEMM [B, H] × [H, E] | Memory bandwidth | `moe_router_topk_fused` |
| Token grouping | Sort by expert | CPU | Python `argsort` + `cumsum` |
| Expert GEMM | Variable per expert | Dequant + compute | `moe_expert_gemm_fp4` |
| Shared expert | Full batch GEMM | Same as dense GEMM | `marlin_gemm_fused_fp4` |
| Aggregation | Weighted sum | Memory bandwidth | Fused in expert kernel |

### Expected Performance (M4 Max)

Based on existing GEMM kernel benchmarks and MoE model characteristics:

| Model | Experts | Active | Dense Equivalent | Expected Throughput |
|-------|---------|--------|------------------|---------------------|
| GLM-4.7-Flash | 64 | 2 | ~2B params | ~150 tok/s prefill |
| Qwen3-30B-A3B | 128 | 8 | ~3B params | ~100 tok/s prefill |
| Mixtral-8x7B | 8 | 2 | ~14B params | ~50 tok/s prefill |

Decode throughput scales with batch size due to better GPU utilization.

## Comparison with vLLM/TensorRT-LLM

### vLLM FusedMoE

vLLM's approach targets NVIDIA GPUs with discrete VRAM:

```python
# vLLM kernel signature (simplified)
fused_moe(
    hidden_states,      # [num_tokens, hidden_size]
    w1, w2,             # [num_experts, intermediate_size, hidden_size]
    topk_weights,       # [num_tokens, num_experts_per_token]
    topk_ids,           # [num_tokens, num_experts_per_token]
    ...
)
```

Key differences from Metal Marlin:

| Aspect | vLLM | Metal Marlin |
|--------|------|--------------|
| Memory model | PCIe + VRAM | Unified Memory |
| Expert weights | May need offloading | Always resident |
| Quantization | W8A8, W4A16 via Marlin | FP4, INT4, sub-4-bit |
| Token grouping | Triton kernel | CPU + GPU kernel |
| Expert parallelism | Across GPUs | Not needed |
| Router fusion | Separate kernels | Fused (planned) |

### TensorRT-LLM MoE

TensorRT-LLM uses a similar CUDA-based approach with additional optimizations:

- **Expert parallelism across GPUs**: Splits experts for multi-GPU
- **Paged expert weights**: For very large MoE models
- **Custom Cutlass kernels**: Highly tuned GEMM primitives

Metal Marlin's advantages:
1. **Simpler dispatch**: No paging, no expert parallelism complexity
2. **Unified address space**: Direct indexing into all experts
3. **Lower latency**: No PCIe round-trips for expert access

### Key Architectural Trade-offs

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MoE Architecture Comparison                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  vLLM/TensorRT (NVIDIA)          Metal Marlin (Apple Silicon)        │
│  ─────────────────────           ────────────────────────────        │
│                                                                       │
│  ┌─────────┐   PCIe    ┌─────┐   ┌──────────────────────────────┐   │
│  │ CPU RAM │ ◄──────► │ GPU │   │     Unified Memory            │   │
│  │(experts)│   copy    │VRAM │   │  CPU + GPU + Experts         │   │
│  └─────────┘           └─────┘   └──────────────────────────────┘   │
│                                                                       │
│  Complexity:                     Complexity:                         │
│  - Expert offloading             - None (always resident)            │
│  - Expert parallelism            - None (single device)              │
│  - Memory management             - Standard allocation               │
│                                                                       │
│  Bandwidth:                      Bandwidth:                          │
│  - PCIe: 32 GB/s                 - UMA: 400-800 GB/s                 │
│  - HBM: 2+ TB/s (H100)           - No transfer needed                │
│                                                                       │
│  Optimal for:                    Optimal for:                        │
│  - Large MoE (100+ experts)      - All MoE models up to memory       │
│  - Multi-GPU clusters            - Single-device inference           │
│  - Batch size > 64               - Batch size 1-32                   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Quantization Strategy for MoE

### Mixed Precision by Layer Type

```python
# From mixed_precision.py

MixedPrecisionConfig.default_moe():
    moe_router      = BF16     # Critical for expert selection
    moe_shared      = FP4/g64  # Always active, tighter quantization
    moe_experts     = FP4/g128 # Redundant (2 of 64), looser OK
    attention_qkv   = FP4/g64  # Position-sensitive
```

### Aggressive Sub-4-bit for Cold Experts

Most routed experts are "cold" (rarely activated). Aggressive quantization works:

```python
MixedPrecisionConfig.aggressive_moe():
    moe_router      = BF16     # Still critical
    moe_shared      = INT4/g64 # Active every token
    moe_experts     = NF3/g64  # 3-bit NormalFloat for cold experts
```

Rationale:
- With 64 experts and top-2 routing, 62 experts are unused per token
- Community benchmarks (llama.cpp IQ2_XXS, IQ3_XXS) show 2-3 bit works
- NF3 (NormalFloat 3-bit) is optimal for Gaussian-distributed weights

### Memory Savings

| Model | FP16 Size | FP4 All | FP4 + NF3 Experts |
|-------|-----------|---------|-------------------|
| GLM-4.7-Flash | 18 GB | 5.5 GB | 4.2 GB |
| Mixtral-8x7B | 112 GB | 28 GB | 21 GB |

## Implementation Status

### Implemented

- [x] FP4/INT4/FP8 dequantization kernels
- [x] Generic batched GEMM (`batched_gemm.metal`)
- [x] Mixed precision configuration system
- [x] MoE model loading (GGUF, safetensors, ONNX)
- [x] Sub-4-bit quantization module (INT2, INT3, NF2, NF3)
- [x] Layer classification for MoE patterns

### Planned (Phase 20)

- [ ] `moe_expert_gemm.metal`: Batched expert GEMM
- [ ] `moe_router.metal`: Fused router + top-k
- [ ] `moe_shared_expert.metal`: Shared expert fusion
- [ ] `moe_dispatch.py`: Dynamic token-to-expert grouping
- [ ] `expert_cache.py`: LRU dequant tile cache
- [ ] `dequant_sub4bit.metal`: Metal kernels for INT2/INT3/NF3
- [ ] `test_sub4bit.py`: Comprehensive sub-4-bit tests

## References

- [vLLM FusedMoE](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe)
- [TensorRT-LLM MoE](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/models/moe)
- [llama.cpp IQ Quantization](https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c)
- [Mixtral Paper (arxiv:2401.04088)](https://arxiv.org/abs/2401.04088)
- [GLM-4 Technical Report](https://github.com/THUDM/GLM-4)
