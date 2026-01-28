# Batched GEMM for Multi-Head Attention

Multi-head attention uses two batched GEMMs:

```
Q: [batch, num_heads, seq_len, head_dim]
K: [batch, num_heads, seq_len, head_dim]
V: [batch, num_heads, seq_len, head_dim]

scores = Q @ K^T  ->  [batch, num_heads, seq_len, seq_len]
output = softmax(scores) @ V  ->  [batch, num_heads, seq_len, head_dim]
```

This document defines a batched GEMM API for both steps and compares two
common dispatch strategies.

## Operations

### 1) Q x K^T (attention scores)

- A = Q: [B, H, M, K]
- B = K: [B, H, N, K]  (transposed in math)
- C = scores: [B, H, M, N]

With:

- B = batch
- H = num_heads
- M = seq_len
- N = seq_len
- K = head_dim

### 2) Scores x V (attention output)

- A = scores: [B, H, M, N]
- B = V: [B, H, N, K]
- C = output: [B, H, M, K]

## Dispatch Strategies

### Strategy A: Strided batched

All heads live in a single contiguous allocation. The kernel computes the
3D grid over (batch, head, tile_mn). Strides describe how to move between
batches and heads in memory.

#### Interface sketch

```
struct BatchedGemmStridedArgs {
  void *A;
  void *B;
  void *C;

  uint32_t M;
  uint32_t N;
  uint32_t K;

  uint32_t batch;
  uint32_t num_heads;

  int64_t stride_a_batch;
  int64_t stride_a_head;
  int64_t stride_a_row;

  int64_t stride_b_batch;
  int64_t stride_b_head;
  int64_t stride_b_row;

  int64_t stride_c_batch;
  int64_t stride_c_head;
  int64_t stride_c_row;

  bool b_transposed; // true for Q x K^T
};

// Grid: [num_heads, ceil_div(M, TILE_MN), batch]
void batched_gemm_strided(const BatchedGemmStridedArgs &args);
```

Notes:
- `stride_*_row` is in elements (or bytes), matching the kernel convention.
- For Q x K^T, `b_transposed = true` and B is stored as [B, H, N, K].
- For scores x V, `b_transposed = false` and B is stored as [B, H, N, K].

#### Advantages

- Single kernel launch for all batches and heads.
- Highest cache locality for Q/K/V when stored in standard attention layouts.
- Simplest host-side dispatch (one descriptor).

#### Trade-offs

- Requires contiguous layout for all heads.
- Less flexible if each head is independently allocated.

### Strategy B: Array of pointers

Each (batch, head) pair has its own allocation. The kernel receives a list
of pointers for A, B, and C, and launches once per batch (or once for the
entire list, depending on runtime support).

#### Interface sketch

```
struct BatchedGemmPtrArgs {
  const void **A_ptrs;
  const void **B_ptrs;
  void **C_ptrs;

  uint32_t M;
  uint32_t N;
  uint32_t K;

  uint32_t batch;
  uint32_t num_heads;
  bool b_transposed;
};

// Grid: [num_heads, ceil_div(M, TILE_MN), batch]
void batched_gemm_ptrs(const BatchedGemmPtrArgs &args);
```

#### Advantages

- Works with heterogeneous or fragmented allocations.
- Can mix different head layouts or data types if needed.

#### Trade-offs

- Higher dispatch overhead (pointer setup and potentially more launches).
- Reduced memory coalescing if head allocations are scattered.

## Recommendation

Use **strided batched** for typical attention workloads where Q/K/V are
stored in standard contiguous layouts. It minimizes launch overhead and
maximizes memory coalescing. Reserve the pointer-array strategy for cases
where heads are independently allocated or layouts are irregular.

## Suggested Defaults for Attention

For Q x K^T:
- A: Q [B, H, M, K]
- B: K [B, H, N, K]
- C: scores [B, H, M, N]
- `b_transposed = true`

For scores x V:
- A: scores [B, H, M, N]
- B: V [B, H, N, K]
- C: output [B, H, M, K]
- `b_transposed = false`
