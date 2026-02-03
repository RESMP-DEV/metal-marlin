# Metal Shader Structural Audit

Target Models:
- **GLM-4.7-Flash**: 30B-A3B MoE, MLA attention (kv_lora=512, q_lora=768)
- **Qwen3-30B-A3B**: 128 experts, top-8, GQA 8:1

## Critical Bug: FP4 Subnormal Dequantization Mismatch

The FP4 E2M1 dequantization has **different implementations** across files:

| File | Subnormal Formula | Code |
|------|-------------------|------|
| `marlin_gemm.metal` | `m * 0.5` | `magnitude = half(man_bit) * half(0.5h)` |
| `moe_expert_gemm.metal` | `m * 0.25` | `magnitude = half(man_bit) * half(0.25h)` |
| `moe_dispatch.metal` | `m * 0.25` | (same as moe_expert_gemm) |
| `mla_proj.metal` | `m * 0.25` | (same as moe_expert_gemm) |

**Mathematical Impact**: For subnormal FP4 values (exp=0), marlin_gemm produces
values **2x larger** than the MoE/MLA kernels. This causes:
- Inconsistent inference results between GEMM paths
- Potential accuracy degradation in MoE models

**Correct Formula** (FP4 E2M1 standard):
- Subnormal: `(-1)^s * m * 2^(-2) = m * 0.25` (exponent bias = 1, min exp = -2)
- **marlin_gemm.metal is WRONG** - should be 0.25h, not 0.5h

## Structural Issues

### 1. "Representative Expert" Hack in MoE Dispatch

**Location**: `moe_dispatch.metal` line ~300

```metal
// For simplicity, use the first valid token's expert for this slot
// (Assumption: tokens in same tile often share experts due to locality)
uint representative_expert = tile_expert_ids[0][slot];
```

**Problem**: Assumes all 32 tokens in a tile use the same expert. With:
- GLM-4.7: 64 experts, top-4 → only 6.25% chance each token shares expert
- Qwen3: 128 experts, top-8 → even worse

**Impact**: ~75% of MoE compute wasted on wrong experts for many tokens.

**Fix**: Use `moe_expert_gemm_fp4_grouped` kernel which properly sorts tokens by expert.

### 2. Code Duplication: 4 Copies of FP4 Dequant

Same logic duplicated with slight variations:
- `marlin_gemm.metal::dequant_fp4_bitwise()`
- `moe_expert_gemm.metal::moe_dequant_fp4_bitwise()`
- `moe_dispatch.metal::dispatch_dequant_fp4_scalar()`
- `mla_proj.metal::dequant_fp4_scalar()`

**Fix**: Extract to `dequant.metal` shared header and `#include`.

### 3. SIMDGROUPS_PER_TG Mismatch

**marlin_gemm.metal**:
- Comment: "4 simdgroups per threadgroup (128 threads)"
- Code: `SIMDGROUPS_PER_TG = 2` (64 threads!)

This discrepancy suggests either:
1. Code regressed and wasn't caught
2. Comment is stale from earlier version

### 4. Tile Size vs Model Dimensions

Current hardcoded tiles vs actual model needs:

| Kernel | TILE_N | GLM-4.7 intermediate | Qwen3 intermediate |
|--------|--------|---------------------|-------------------|
| MOE GEMM | 64 | 1536 (24 tiles) | 768 (12 tiles) |
| MLA proj | 64 | 512/768 (kv/q lora) | N/A |

For small dimensions (768), TILE_N=64 means only 12 full tiles.
Consider TILE_N=32 for better occupancy with small N.

## Model-Specific Problem Sizes

### GLM-4.7-Flash

```python
# MoE Expert GEMM shapes
up_proj:   (batch, 2048) @ (2048, 1536) → (batch, 1536)  # per expert
down_proj: (batch, 1536) @ (1536, 2048) → (batch, 2048)  # per expert

# MLA Projections
kv_a_proj: (batch, 2048) @ (2048, 512)  → (batch, 512)   # compress KV
kv_b_proj: (batch, 512)  @ (512, 256*20) → (batch, 5120) # expand KV
q_proj:    (batch, 2048) @ (2048, 768)  → (batch, 768)   # query latent
```

### Qwen3-30B-A3B

```python
# MoE Expert GEMM shapes  
gate_proj: (batch, 2048) @ (2048, 768) → (batch, 768)  # per expert
up_proj:   (batch, 2048) @ (2048, 768) → (batch, 768)  # per expert
down_proj: (batch, 768)  @ (768, 2048) → (batch, 2048) # per expert

# Standard GQA (not MLA)
q_proj: (batch, 2048) @ (2048, 4096) → (batch, 4096)  # 32 heads * 128 dim
k_proj: (batch, 2048) @ (2048, 512)  → (batch, 512)   # 4 KV heads * 128 dim
v_proj: (batch, 2048) @ (2048, 512)  → (batch, 512)   # 4 KV heads * 128 dim
```

## Recommended Fixes (Priority Order)

### P0: Critical Bugs
1. Fix FP4 subnormal dequant in `marlin_gemm.metal` (0.5h → 0.25h)
2. Fix SIMDGROUPS_PER_TG comment/code mismatch

### P1: Performance
1. Enable `moe_dispatch_grouped` as default (fix representative expert hack)
2. Add TILE_N=32 variant for small output dimensions
3. Extract shared `fp4_dequant.h` header

### P2: Maintainability  
1. Add model-specific autotune configs for GLM-4.7 and Qwen3
2. Document correct FP4 E2M1 formula in code comments
3. Add integration tests for cross-kernel consistency
