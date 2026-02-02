# MoE Kernel Architecture

This document describes the Metal-based Mixture-of-Experts (MoE) kernel architecture for efficient inference on Apple Silicon.

## Overview

The MoE system consists of four core components:

1. **Router**: Computes expert selection probabilities and selects top-k experts per token
2. **Expert GEMM**: Executes quantized matrix multiplications for selected experts
3. **Dispatch**: Coordinates token-to-expert routing and result aggregation
4. **Shared Expert**: Handles always-active expert pathway (for architectures like GLM-4)

```
                         ┌─────────────────────────────────────────────────────┐
                         │                    MoE Layer                        │
                         └─────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Input: [batch, hidden_dim]                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         │                         ▼
         ┌──────────────────┐                 │              ┌──────────────────┐
         │      Router      │                 │              │  Shared Expert   │
         │  (moe_router)    │                 │              │   (optional)     │
         └──────────────────┘                 │              └──────────────────┘
                    │                         │                         │
                    ▼                         │                         │
         ┌──────────────────┐                 │                         │
         │  Top-K Selection │                 │                         │
         │  + Renormalize   │                 │                         │
         └──────────────────┘                 │                         │
                    │                         │                         │
                    ▼                         │                         │
         ┌──────────────────┐                 │                         │
         │   expert_ids     │                 │                         │
         │   expert_probs   │                 │                         │
         │  [batch, top_k]  │                 │                         │
         └──────────────────┘                 │                         │
                    │                         │                         │
                    ▼                         │                         │
         ┌──────────────────────────────┐     │                         │
         │        Expert GEMM           │◄────┘                         │
         │  (parallel per-expert)       │                               │
         │                              │                               │
         │  For each expert slot:       │                               │
         │    output += prob[k] *       │                               │
         │      expert[id[k]](input)    │                               │
         └──────────────────────────────┘                               │
                    │                                                   │
                    └───────────────────────┬───────────────────────────┘
                                            │
                                            ▼
         ┌──────────────────────────────────────────────────────────────────────┐
         │  Aggregate: shared_out + sum(prob[k] * routed_expert_out[k])        │
         └──────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
         ┌──────────────────────────────────────────────────────────────────────┐
         │  Output: [batch, hidden_dim]                                        │
         └──────────────────────────────────────────────────────────────────────┘
```

## Buffer Layouts

### Packed Weights (FP4 Quantization)

Expert weights are stored in FP4 format with 8 values packed per uint32:

```
expert_weights: [num_experts, hidden_dim/8, out_dim] packed FP4

Memory layout per expert:
┌─────────────────────────────────────────────────────────────┐
│  Expert 0                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Row 0 (K=0..7 packed): [uint32 × out_dim]               ││
│  │ Row 1 (K=8..15 packed): [uint32 × out_dim]              ││
│  │ ...                                                      ││
│  │ Row K/8-1: [uint32 × out_dim]                           ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Expert 1                                                   │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘

FP4 E2M1 bit layout (4 bits per value):
┌───┬───────┬─────┐
│ S │  E E  │  M  │
│ 1 │  2b   │ 1b  │
└───┴───────┴─────┘
  │     │      └── Mantissa bit
  │     └──────── Exponent (2 bits)
  └────────────── Sign bit

Dequantization formula:
  if (exp == 0): magnitude = man * 0.25
  else: magnitude = 2^(exp-1) * (1.0 + man * 0.5)
  value = (-1)^sign * magnitude * scale
```

### Scales

Per-group scales for dequantization:

```
scales: [num_experts, n_groups, out_features]

where n_groups = ceil(hidden_dim / group_size)

Indexing for weight at (expert, k, n):
  group_idx = k / group_size
  scale = scales[expert, group_idx, n]
```

### Sign Vectors (Trellis Quantization)

For Trellis 3-bit quantization, sign flips are stored separately:

```
su: [num_experts, in_features]   # Sign flips for input dimension (K)
sv: [num_experts, out_features]  # Sign flips for output dimension (N)

Final dequantized value:
  value = grid[codebook_idx] * scale * su[k] * sv[n]
```

### Trellis Weight Packing

For 3-bit Trellis quantization, weights are packed in 16x16 tiles:

```
packed_weights: [num_experts, tiles_k, tiles_n, 96 bytes]

where:
  tiles_k = ceil(K / 16)
  tiles_n = ceil(N / 16)
  96 bytes = 16 * 16 * 3 bits / 8 = 96 bytes per tile

Tile layout (transposed for coalesced memory access):
  idx_in_tile = n_in_tile * 16 + k_in_tile
  bit_offset = idx_in_tile * 3
```

## Kernel Execution Model

### Grid Dimensions

The kernels use different grid layouts depending on the dispatch strategy:

**Per-token dispatch** (`moe_expert_gemm_fp4`):
```
Grid: [ceil(out_dim/64), ceil(batch/64)]
  - tgid.x: Output column block
  - tgid.y: Token batch block

Each threadgroup processes:
  - MOE_TILE_M (64) tokens
  - MOE_TILE_N (64) output columns
  - Loops over top_k expert slots
```

**Grouped dispatch** (`moe_expert_gemm_fp4_grouped`, `moe_dispatch_grouped`):
```
Grid: [ceil(out_dim/64), num_experts]
  - tgid.x: Output column block
  - tgid.y: Expert index

Each threadgroup processes:
  - All tokens assigned to one expert
  - MOE_TILE_N (64) output columns
  - Token batching within expert
```

**Trellis MoE with SwiGLU** (`moe_trellis_swiglu`):
```
Grid: [ceil(hidden_dim/64), M, top_k]
  - tgid.x: Output column block (hidden_dim)
  - tgid.y: Token index
  - tgid.z: Expert slot (0 to top_k-1)

Each threadgroup processes:
  - 1 token
  - 1 expert slot
  - MOE_TILE_N (64) output columns
  - Full intermediate streaming (chunks of 64)
```

### Threadgroup Configuration

```
Threads per threadgroup: 128 (4 simdgroups × 32 threads)

Simdgroup layout for GEMM:
  ┌────────────────────────────────────────────────────────────────┐
  │                        MOE_TILE_N (64)                         │
  │  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
  │  │  Simdgroup 0 │  Simdgroup 1 │  Simdgroup 2 │  Simdgroup 3 │ │
  │  │   8×16       │    8×16      │    8×16      │    8×16      │ │
  │  │  (rows 0-7)  │  (rows 8-15) │ (rows 16-23) │ (rows 24-31) │ │
  │  └──────────────┴──────────────┴──────────────┴──────────────┘ │
  │           ↑                                                     │
  │    MOE_TILE_M (32-64 rows depending on kernel)                 │
  └────────────────────────────────────────────────────────────────┘

Alternative layout (expert_gemm):
  - All simdgroups cover same M rows
  - Each simdgroup handles different N columns
  - sg_col_offset = simd_id * (SG_N_TILES * 8)
```

### Memory Access Patterns

**Double-buffered K-loop**:
```
Threadgroup memory:
  A_tiles[2][TILE_M][TILE_K]  // Activation tiles (ping-pong)
  B_tiles[2][TILE_K][TILE_N]  // Weight tiles (ping-pong)

Loop structure:
  buf_compute = 0
  load(A_tiles[0], B_tiles[0], k=0)
  barrier()

  for kt = 0 to num_k_tiles:
    buf_load = 1 - buf_compute

    if (next_k < K):
      load(A_tiles[buf_load], B_tiles[buf_load], k=next_k)

    compute(A_tiles[buf_compute], B_tiles[buf_compute])
    barrier()

    buf_compute = buf_load
```

**Cooperative tile loading**:
```
// 128 threads load TILE_M×TILE_K = 64×32 = 2048 elements
// 16 elements per thread

elems_per_thread = (TILE_M * TILE_K) / THREADS  // 16

for i = 0 to elems_per_thread:
  flat_idx = thread_idx * elems_per_thread + i
  row = flat_idx / TILE_K
  col = flat_idx % TILE_K
  A_buf[row][col] = A[global_row + row, global_col + col]
```

**Simdgroup matrix operations**:
```
simdgroup_matrix<half, 8, 8> acc[SG_M_TILES][SG_N_TILES];

for kt = 0 to K_TILES:
  for mi = 0 to SG_M_TILES:
    simdgroup_load(a_frag, &A_buf[sg_row + mi*8][kt*8], TILE_K);

    for ni = 0 to SG_N_TILES:
      simdgroup_load(b_frag, &B_buf[kt*8][sg_col + ni*8], TILE_N);
      simdgroup_multiply_accumulate(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
```

## Known Limitations and Workarounds

### 1. Atomic Output Accumulation

Multiple threadgroups (one per expert slot in Trellis MoE, or per-expert in grouped dispatch) write to the same output locations.

**Problem**: Metal doesn't support native FP16 atomics.

**Workaround**: Use FP32 output buffer with CAS (compare-and-swap) atomic add:
```metal
device atomic_uint* atomic_ptr = (device atomic_uint*)(&output[idx]);
uint old_bits = atomic_load_explicit(atomic_ptr, memory_order_relaxed);
uint new_bits;
do {
    float old_val = as_type<float>(old_bits);
    float new_val = old_val + weighted;
    new_bits = as_type<uint>(new_val);
    success = atomic_compare_exchange_weak_explicit(...);
} while (!success);
```

Post-processing converts FP32 output to FP16 on the host.

### 2. Representative Expert Approximation

In `moe_dispatch_fused`, all tokens in a tile are assumed to use the same expert.

**Problem**: Tokens in the same tile may have different expert assignments.

**Workaround**: Use grouped dispatch (`moe_expert_gemm_fp4_grouped`) for accurate per-token routing, or accept the approximation for large batches where expert diversity per tile is low.

### 3. Small Batch Decode Efficiency

For single-token decode (M=1), the standard tiled GEMM underutilizes GPU resources.

**Workaround**: `moe_shared_expert_scatter` kernel uses per-thread output column striding:
```metal
for (out_d = thread_idx; out_d < intermediate_dim; out_d += THREADS) {
    // Each thread computes one output element
    // Direct dot product, no simdgroup matrices
}
```

### 4. Intermediate Dimension Larger Than Tile Size

For SwiGLU experts, intermediate_dim can be much larger than the GEMM tile size.

**Problem**: Cannot materialize full intermediate result in threadgroup memory.

**Workaround**: Streaming chunk approach in `moe_trellis_swiglu`:
```
for chunk in intermediate_chunks:
    1. Compute gate[chunk] and up[chunk] via full K-reduction
    2. Apply SwiGLU: swiglu[chunk] = silu(gate[chunk]) * up[chunk]
    3. Partial down: acc_down += swiglu[chunk] @ down_weights[chunk]
```

### 5. Metal Half Precision Bugs

Some Metal implementations have precision issues with FP16 intermediate calculations.

**Workaround**: Use FP32 accumulators in critical paths (`moe_trellis_swiglu_fp32acc`), convert to FP16 only for final output.

### 6. Load Balancing Statistics

Computing auxiliary loss for training requires tracking per-expert token counts.

**Solution**: `moe_router_with_aux_loss` kernel accumulates atomic statistics:
```metal
device atomic_float* expert_load;   // Sum of routing probs per expert
device atomic_uint* expert_count;   // Count of tokens selecting each expert

atomic_fetch_add_explicit(expert_load + e, prob, memory_order_relaxed);
atomic_fetch_add_explicit(expert_count + topk_ids[i], 1u, memory_order_relaxed);
```

### 7. Threadgroup Memory Limits

Apple Silicon has 32KB threadgroup memory limit.

**Constraint**: Tile sizes are chosen to fit within this limit:
- A_tile: 64×32×2 = 4KB
- B_tile: 32×64×2 = 4KB
- Double buffering: 2× = 16KB total for tiles
- Plus staging, routing info, etc.

For Trellis MoE with separate gate/up/down tiles:
- A_tile: 16×2 = 32 bytes (single token)
- B_gate/up/down: 16×64×2 = 2KB each = 6KB
- Total: ~7KB (well within limit)
