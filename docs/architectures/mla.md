# Multi-head Latent Attention (MLA)

Multi-head Latent Attention is an attention architecture designed to reduce KV cache memory during inference. First introduced in DeepSeek-V2, MLA compresses key-value projections through a learned latent bottleneck, achieving up to 93% KV cache reduction compared to standard Multi-Head Attention (MHA) while maintaining or exceeding quality.

Reference models: GLM-4.7-Flash, DeepSeek-V2, DeepSeek-V2.5, DeepSeek-V3.

## Architecture Overview

### Standard MHA KV Cache

In standard Multi-Head Attention, each token stores separate K and V vectors for every head:

```
KV cache per layer = seq_len × num_heads × head_dim × 2(K+V)

Example: 32 heads, head_dim=128, seq=4096
= 4096 × 32 × 128 × 2 × 2 bytes = 64 MB per layer
```

For a 32-layer model with 4K context: **2 GB** just for KV cache.

### MLA Latent Compression

MLA introduces a latent bottleneck that compresses KV into a shared low-rank representation:

```
┌─────────────┐
│   hidden    │  (d_model)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  kv_a_proj  │  hidden → kv_lora_rank + qk_rope_head_dim
└──────┬──────┘
       │
       ├───────────────┐
       │               │
       ▼               ▼
┌─────────────┐  ┌──────────────┐
│   c_kv      │  │  k_pe (RoPE) │
│  (latent)   │  │  positional  │
└──────┬──────┘  └──────┬───────┘
       │               │
       ▼               │
┌─────────────┐        │
│  kv_b_proj  │        │
└──────┬──────┘        │
       │               │
       ▼               │
┌──────┴───────────────┴──────┐
│   K, V (num_heads × head_dim × 2) │
└─────────────────────────────┘
```

The key insight: **only store `c_kv` (the latent) in the KV cache**, not the full K and V. The decompression through `kv_b_proj` happens at attention time.

```
MLA KV cache per layer = seq_len × kv_lora_rank

Example: kv_lora_rank=512, seq=4096
= 4096 × 512 × 2 bytes = 4 MB per layer

Compression: 64 MB → 4 MB = 16× reduction
```

## Layer Projections

### Query Path

MLA can optionally compress queries too (DeepSeek-V2.5+), though GLM-4.7-Flash uses it:

| Layer | Input Shape | Output Shape | Purpose |
|-------|-------------|--------------|---------|
| `q_a_proj` | `[batch, seq, d_model]` | `[batch, seq, q_lora_rank]` | Query down-projection to latent |
| `q_b_proj` | `[batch, seq, q_lora_rank]` | `[batch, seq, num_heads × head_dim]` | Query up-projection to full dim |

**Typical dimensions (GLM-4.7-Flash):**
- d_model = 4096
- q_lora_rank = 1536
- num_heads = 32
- head_dim = 128

### KV Path

| Layer | Input Shape | Output Shape | Purpose |
|-------|-------------|--------------|---------|
| `kv_a_proj` | `[batch, seq, d_model]` | `[batch, seq, kv_lora_rank + qk_rope_head_dim]` | KV down-projection |
| `kv_b_proj` | `[batch, seq, kv_lora_rank]` | `[batch, seq, num_heads × head_dim × 2]` | KV up-projection |

**Typical dimensions (GLM-4.7-Flash):**
- kv_lora_rank = 512
- qk_rope_head_dim = 64 (position encoding dimension)
- Output of kv_a_proj: 576 (512 latent + 64 RoPE)

### Output Projection

Standard output projection, same as MHA:

| Layer | Input Shape | Output Shape | Purpose |
|-------|-------------|--------------|---------|
| `o_proj` | `[batch, seq, num_heads × head_dim]` | `[batch, seq, d_model]` | Attention output projection |

## Forward Pass

```python
def mla_attention(hidden_states, kv_cache):
    # Query path
    q_latent = q_a_proj(hidden_states)      # [B, S, q_lora_rank]
    q = q_b_proj(q_latent)                   # [B, S, num_heads × head_dim]
    q = q.view(B, S, num_heads, head_dim)

    # KV path
    kv_compressed = kv_a_proj(hidden_states) # [B, S, kv_lora_rank + rope_dim]
    c_kv = kv_compressed[:, :, :kv_lora_rank]           # latent
    k_pe = kv_compressed[:, :, kv_lora_rank:]           # positional

    # Store compressed KV in cache (NOT full K, V)
    kv_cache.append(c_kv, k_pe)

    # Decompress for attention
    kv_full = kv_b_proj(kv_cache.c_kv)       # [B, cache_len, num_heads × head_dim × 2]
    k, v = kv_full.chunk(2, dim=-1)

    # Apply RoPE to k
    k = apply_rope(k, kv_cache.k_pe)
    q = apply_rope(q, position_ids)

    # Standard attention
    attn_output = scaled_dot_product_attention(q, k, v)
    return o_proj(attn_output)
```

## Memory Comparison

| Architecture | KV Cache Formula | 32L / 4K context | 32L / 32K context |
|--------------|------------------|------------------|-------------------|
| **MHA** | `2 × L × S × H × D × 2` | 4.0 GB | 32.0 GB |
| **GQA-8** | `2 × L × S × (H/4) × D × 2` | 1.0 GB | 8.0 GB |
| **MQA** | `2 × L × S × 1 × D × 2` | 0.25 GB | 2.0 GB |
| **MLA-512** | `L × S × kv_rank × 2` | 0.13 GB | 1.0 GB |

Where:
- L = layers (32)
- S = sequence length
- H = num_heads (32)
- D = head_dim (128)
- kv_rank = kv_lora_rank (512)

**MLA achieves 32× reduction vs MHA for equivalent quality.**

## Quantization Sensitivity Analysis

Based on empirical analysis from `eval_glm4_flash.py`:

### Layer Sensitivity Ranking (Most → Least Sensitive)

```
q_a_proj    ████████████████████  (Highest - directly affects attention scores)
q_b_proj    ████████████████░░░░
kv_a_proj   ███████████████░░░░░  (KV cache quality)
kv_b_proj   ██████████████░░░░░░
o_proj      ████████████░░░░░░░░  (Least sensitive in MLA layers)
```

### Quantization Recommendations by Layer Type

| Layer | Sensitivity | Recommended Precision | Group Size | Use Hadamard |
|-------|-------------|----------------------|------------|--------------|
| `q_a_proj` | Critical | FP4 or FP8 | 32-64 | Yes if outliers |
| `q_b_proj` | High | FP4 | 64 | Yes |
| `kv_a_proj` | High | FP4 | 64 | Yes (significant benefit) |
| `kv_b_proj` | Medium-High | FP4 | 64-128 | Yes |
| `o_proj` | Medium | FP4 | 128 | Optional |

### Why q_a_proj is Most Sensitive

The query down-projection (`q_a_proj`) compresses the entire query representation into a latent space. Quantization errors here propagate through:

1. The up-projection (`q_b_proj`) amplifies small errors
2. Errors in Q directly affect attention scores (Q @ K^T)
3. Softmax is sensitive to the relative magnitudes of scores
4. This affects which tokens receive attention weight

For high-quality inference, consider keeping `q_a_proj` at FP8 or even FP16.

### Hadamard Transform Benefit

MLA's latent projections often have concentrated outliers due to the compression bottleneck. Hadamard rotation disperses these outliers:

```
Before Hadamard: max/mean ratio ≈ 47.3
After Hadamard:  max/mean ratio ≈ 3.8

Result: 12× improvement in outlier distribution
```

Layers benefiting most from Hadamard:
- `kv_a_proj` (15-20% RMSE reduction)
- `q_b_proj` (10-15% RMSE reduction)
- `q_a_proj` (5-10% RMSE reduction)

## Optimal Quantization Configurations

### Quality-First (Minimal Degradation)

For applications where accuracy is paramount:

```python
mla_config = MixedPrecisionConfig(
    mla_q_a=LayerQuantConfig(Precision.BF16),      # Keep FP16
    mla_q_b=LayerQuantConfig(Precision.FP4, 32),   # Tight FP4
    mla_kv_a=LayerQuantConfig(Precision.FP4, 32),
    mla_kv_b=LayerQuantConfig(Precision.FP4, 64),
    attention_out=LayerQuantConfig(Precision.FP4, 64),
)
```

Expected: <0.1 perplexity increase vs BF16.

### Balanced (Default)

Standard quality/compression trade-off:

```python
mla_config = MixedPrecisionConfig(
    mla_q_a=LayerQuantConfig(Precision.FP4, 64),
    mla_q_b=LayerQuantConfig(Precision.FP4, 64),
    mla_kv_a=LayerQuantConfig(Precision.FP4, 64),
    mla_kv_b=LayerQuantConfig(Precision.FP4, 64),
    attention_out=LayerQuantConfig(Precision.FP4, 128),
)
```

Expected: 0.1-0.3 perplexity increase.

### Speed-First (Maximum Compression)

For memory-constrained deployment:

```python
mla_config = MixedPrecisionConfig(
    mla_q_a=LayerQuantConfig(Precision.FP4, 128),
    mla_q_b=LayerQuantConfig(Precision.FP4, 128),
    mla_kv_a=LayerQuantConfig(Precision.FP4, 128),
    mla_kv_b=LayerQuantConfig(Precision.FP4, 128),
    attention_out=LayerQuantConfig(Precision.FP4, 256),
)
```

Expected: 0.3-0.5 perplexity increase.

## MLA + MoE (GLM-4.7-Flash)

GLM-4.7-Flash combines MLA attention with Mixture of Experts, creating compound memory savings:

```
                     ┌────────────────────────────┐
                     │      Hidden State          │
                     └───────────┬────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
            ▼                                         ▼
    ┌───────────────┐                        ┌───────────────┐
    │  MLA Attention │                        │   MoE FFN     │
    │  (compressed   │                        │  64 experts   │
    │   KV cache)    │                        │  2 active     │
    └───────────────┘                        └───────────────┘
```

### Quantization Strategy for MLA+MoE

```python
config = MixedPrecisionConfig(
    # MLA layers - use recommended MLA settings
    mla_q_a=LayerQuantConfig(Precision.FP4, 64),
    mla_q_b=LayerQuantConfig(Precision.FP4, 64),
    mla_kv_a=LayerQuantConfig(Precision.FP4, 64),
    mla_kv_b=LayerQuantConfig(Precision.FP4, 64),
    attention_out=LayerQuantConfig(Precision.FP4, 128),

    # MoE layers - router critical, experts can be aggressive
    moe_router=LayerQuantConfig(Precision.BF16),        # Never quantize router
    moe_shared_expert=LayerQuantConfig(Precision.FP4, 64),  # More used
    moe_experts=LayerQuantConfig(Precision.FP4, 128),       # Sparse activation
)
```

## KV Cache Quantization

For MLA models, the KV cache stores the compressed latent `c_kv`, not full K/V. This changes quantization strategy:

### Latent Cache vs Full Cache

| Aspect | Full K/V Cache | MLA Latent Cache |
|--------|---------------|------------------|
| Values stored | K, V tensors | Compressed c_kv |
| Elements per position | 2 × num_heads × head_dim | kv_lora_rank |
| Typical size | 8192 | 512 |
| Quantization impact | Affects attention directly | Amplified by kv_b_proj |

### Latent Cache Quantization

Because the latent is smaller but errors are amplified by `kv_b_proj`, use tighter quantization:

```python
kv_cache_config = CacheConfig(
    quantize=True,
    format="fp4",
    group_size=32,      # Tighter than weight quantization
    per_head_scale=True # Critical for MLA latents
)
```

Error amplification factor: approximately `sqrt(num_heads × head_dim / kv_lora_rank)` ≈ 2.8× for typical configs.

## Implementation Notes

### Layer Detection Patterns

```python
MLA_LAYER_PATTERNS = {
    "q_a_proj": ["q_a_proj", "q_down_proj", "query_a_proj"],
    "q_b_proj": ["q_b_proj", "q_up_proj", "query_b_proj"],
    "kv_a_proj": ["kv_a_proj", "kv_down_proj", "kv_a_layernorm"],
    "kv_b_proj": ["kv_b_proj", "kv_up_proj"],
    "o_proj": ["o_proj", "out_proj", "dense"],
}
```

### Dimension Validation

When quantizing MLA models, validate dimensions:

```python
def validate_mla_dimensions(config):
    assert config.kv_lora_rank % 8 == 0, "kv_lora_rank must be divisible by 8 for FP4"
    assert config.q_lora_rank % 8 == 0, "q_lora_rank must be divisible by 8 for FP4"

    # Check projection shapes
    kv_a_out = config.kv_lora_rank + config.qk_rope_head_dim
    kv_b_out = config.num_heads * config.head_dim * 2

    # Ensure group_size divides hidden dimensions
    assert config.hidden_size % group_size == 0
    assert kv_a_out % group_size == 0
```

## RoPE Fusion Optimization

Metal Marlin supports fusing RoPE with MLA latent projections for improved performance. This optimization applies RoPE in the compressed space when mathematically equivalent.

### Standard vs Optimized Flow

```
Standard:  hidden → kv_a_proj → latent → kv_b_proj → RoPE → K/V
Optimized: hidden → kv_a_proj → latent_RoPE → kv_b_proj → K/V
```

### When Fusion is Mathematically Valid

RoPE fusion is valid when one of these conditions holds:

1. **Decoupled RoPE (GLM-style)**: RoPE is applied to a separate `qk_rope_head_dim` portion, not to `kv_lora_rank`
2. **Orthogonal kv_b_proj**: If W_kv_b is orthogonal, then RoPE(W_kv_b @ x) = W_kv_b @ RoPE(x)
3. **Commuting structures**: When the projection preserves rotation equivariance

Many modern MLA implementations use decoupled RoPE, making fusion safe. Coupled
variants that apply RoPE after kv_b_proj over the full head_dim must keep the
standard path.

### Supported Model Variants

| Model | RoPE Style | Fusion Support | Notes |
|-------|-----------|----------------|-------|
| **GLM-4.7-Flash** | Decoupled | ✅ Full | `rope_ratio` scaling, separate `qk_rope_head_dim=64` |
| **DeepSeek-V2** | Decoupled | ✅ Full | Position encoding split from latent |
| **DeepSeek-V2.5** | Decoupled | ✅ Full | Enhanced MLA with query compression |
| **DeepSeek-V3** | Decoupled | ✅ Full | Latest variant, same fusion approach |
| **Coupled MLA (single RoPE on full K/V)** | Coupled | ❌ None | RoPE applied after kv_b_proj; fusion breaks equivalence |

### GLM rope_ratio Scaling

GLM-4.7-Flash uses `rope_ratio` to scale RoPE frequencies:

```python
# Standard RoPE: inv_freq = 1 / (base^(2i/dim))
# GLM RoPE:      inv_freq = rope_ratio / (base^(2i/dim))

# Example: rope_ratio=0.5 doubles effective context length
```

The `rope_ratio` is typically set in model config and handled automatically by `MLAAttention`.

### Metal Kernel Support

The following kernels support MLA RoPE operations:

| Kernel | Purpose | Latent Support |
|--------|---------|----------------|
| `rope_forward` | Standard RoPE for Q/K | Full head_dim |
| `rope_mla_latent` | RoPE for kv_a_proj output | `kv_lora_rank + rope_dim` |
| `rope_mla_split_fused` | Fused split + RoPE | Optimal for decoupled MLA |
| `rope_small_dim` | Simdgroup-optimized | dim <= 64 |
| `rope_mla_latent_small` | Simdgroup RoPE for MLA latents | rope_dim <= 64 |
| `rope_mla_latent_small_scaled` | Simdgroup RoPE with rope_ratio | rope_dim <= 64 |
| `rope_generate_cache` | On-the-fly cache with rope_ratio | Any dim |

Projection fusion is provided by `mla_proj_with_rope_fp4` in `contrib/metal_marlin/src/mla_proj.metal`,
which applies RoPE to the decoupled rope_dim portion during kv_a_proj.

### Using Fused RoPE in Code

```python
from metal_marlin.mla_attention import MLAAttention

# GLM-4.7-Flash configuration
attn = MLAAttention(
    hidden_size=4096,
    num_heads=32,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    rope_theta=10000.0,
    rope_ratio=1.0,  # GLM's frequency scaling
)

# The MLAAttention class automatically uses the optimal RoPE path
output = attn(hidden_states, kv_cache=cache, layer_idx=0)
```

### Performance Impact

Fused RoPE provides:
- **Reduced memory traffic**: RoPE applied during split, not separate pass
- **Better cache utilization**: cos/sin values loaded once for both c_kv and k_pe
- **Smaller compute footprint**: Only `qk_rope_head_dim` (64) rotated, not full K

Typical speedup: 5-15% for attention-bound workloads.

## References

1. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (https://arxiv.org/abs/2405.04434)
2. GLM-4.7-Flash Technical Report
3. Metal Marlin mixed_precision.py implementation
4. eval_glm4_flash.py sensitivity analysis

## Related Documentation

- [KV Cache Design](../concepts/kv_cache.md) - General KV cache architecture
- [Mixed Precision](../concepts/mixed_precision.md) - Layer-wise precision configuration
- [MoE Architecture](../concepts/moe_architecture.md) - MoE-specific considerations
- [MR-GPTQ](../formats/mr_gptq.md) - Hadamard rotation and Hessian calibration
- [RoPE Implementation](../audits/metal_kernel_audit.md#position-encoding-15-variants) - RoPE kernel documentation
