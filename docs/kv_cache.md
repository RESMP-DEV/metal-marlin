# KV Cache Design

This document describes the key-value cache architecture used for efficient autoregressive inference in Metal Marlin.

## Purpose

During transformer inference, the attention mechanism computes `softmax(Q @ K^T / sqrt(d)) @ V`. Without caching, every generated token would require recomputing K and V projections for all previous tokens, making generation cost quadratic in sequence length. The KV cache stores K and V tensors from prior positions, reducing each decode step to a single new token's projections plus a cache lookup.

## Inference Phases

```
Prefill (prompt processing):
  Q: [batch, num_heads, seq_len, head_dim]
  K: [batch, num_kv_heads, seq_len, head_dim]  → Store in cache
  V: [batch, num_kv_heads, seq_len, head_dim]  → Store in cache

  All prompt tokens processed in parallel via flash attention.
  KV cache populated with full prompt context.

Decode (autoregressive generation):
  Q_new: [batch, num_heads, 1, head_dim]           ← Single new token
  K_cached: [batch, num_kv_heads, seq_len, head_dim]  ← From cache
  V_cached: [batch, num_kv_heads, seq_len, head_dim]  ← From cache

  Attention: Q_new @ K_cached^T → [batch, num_heads, 1, seq_len]
  Output: softmax(scores / sqrt(d)) @ V_cached → [batch, num_heads, 1, head_dim]

  Append new K, V to cache. Advance position.
```

## Memory Layout

The cache uses a 4D contiguous layout optimized for sequential writes and tiled reads:

```
[batch, num_kv_heads, max_seq_len, head_dim]
  │         │              │           │
  │         │              │           └─ Innermost: 128 elements (or 64/80)
  │         │              └─ Pre-allocated to max context length
  │         └─ GQA: can be fewer than num_q_heads
  └─ Batch dimension (typically 1 for single-sequence inference)
```

GQA mapping during attention dispatch:
```
head_k = head_q * num_kv_heads / num_heads_q
```
Multiple Q heads share a single K/V head, reducing cache memory proportionally.

## Design Decisions

### 1. Pre-allocation

The cache allocates tensors for `max_seq_len` at initialization. This avoids per-token reallocation during decode, which on Metal would trigger new buffer allocations and invalidate command encoder state.

```python
cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
k_cache = mx.zeros(cache_shape, dtype=mx.float16)  # Per layer
v_cache = mx.zeros(cache_shape, dtype=mx.float16)
```

A position counter (`seq_len`) tracks how much of the buffer is valid. Reads slice `[:, :, :seq_len, :]`; writes target `[:, :, seq_len:seq_len+new, :]`.

Memory cost at initialization for a 32-layer, 8-head GQA model with head_dim=128:
```
FP16: 32 layers * 2(K+V) * 8 heads * 4096 seq * 128 dim * 2 bytes = 4 GB
FP4:  32 layers * 2(K+V) * 8 heads * 4096 seq * 128 dim * 0.5 bytes + scales ≈ 1.06 GB
```

### 2. Quantized KV Cache (FP4 / INT4)

For long-context inference where the KV cache dominates memory, optional quantization stores values in 4-bit format with per-row scales.

**FP4 E2M1 format** (same as weight quantization):
- 1 sign bit, 2 exponent bits, 1 mantissa bit
- Range: 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
- 8 values packed per `uint32`

**Packed storage layout:**
```
[batch, num_kv_heads, max_seq_len, head_dim // 8]  dtype=uint32
```

**Per-row scales:**
```
[batch, num_kv_heads, max_seq_len, 1]  dtype=float16
```

Each row (one position, one head) gets a single scale factor derived from its absolute maximum:
```
scale = max(|row|) / 6.0    (6.0 = FP4 E2M1 representable max)
quantized = round(value / scale)
```

Dequantization happens on-the-fly during attention computation. The flash attention kernels `flash_attention_kv_fp4` and `flash_attention_kv_int4` load packed tiles, dequantize per-element in registers, and proceed with the standard online softmax algorithm.

**INT4 variant:**
- Symmetric signed 4-bit integers [-8, 7]
- Same packing density (8 per uint32)
- Per-row scale: `scale = max(|row|) / 7.0`
- Slightly simpler dequant (subtract bias, multiply scale)

### 3. Memory Savings

| Format | Bytes/element | 4K context (32L, 8H, D=128) | 32K context |
|--------|---------------|------------------------------|-------------|
| FP16   | 2.0           | 4.0 GB                       | 32.0 GB     |
| FP4    | 0.5 + scale   | ~1.06 GB (3.8x savings)      | ~8.5 GB     |
| INT4   | 0.5 + scale   | ~1.06 GB (3.8x savings)      | ~8.5 GB     |

Scale overhead is negligible: 2 bytes per row vs 256 bytes of data (head_dim=128).

### 4. Paged Attention (Future Work)

For very long sequences (>32K tokens) or batch serving, paged attention avoids pre-allocating the full max_seq_len:

- Divide KV cache into fixed-size pages (e.g., 256 tokens per page)
- Maintain a page table mapping logical positions to physical pages
- Allocate pages on demand as the sequence grows
- Enables memory sharing between sequences with common prefixes

This is not yet implemented. Current workloads (single-sequence, up to 4K-8K context) fit comfortably with pre-allocation on M4 Max unified memory.

### 5. Ring Buffer (Sliding Window)

For models using sliding window attention (e.g., Mistral with 4K window), the cache can operate as a ring buffer:

```
write_pos = seq_len % window_size

On each decode step:
  cache[:, :, write_pos, :] = new_kv
  write_pos = (write_pos + 1) % window_size
```

This bounds cache memory to `window_size * head_dim` regardless of total sequence length. Not yet implemented; current models use full context.

## Integration with Flash Attention Kernels

The flash attention kernels in `src/flash_attention.metal` consume the KV cache directly:

**Standard path** (`flash_attention`, `flash_attention_causal`, `flash_attention_gqa`):
- K/V pointers point into the cache buffer
- `seq_k` parameter set to current `seq_len`
- Tiled iteration over K/V dimension with `TILE_KV = 64`

**Quantized path** (`flash_attention_kv_fp4`, `flash_attention_kv_int4`):
- K/V pointers are packed `uint32` buffers
- Additional scale buffer pointers
- Tile loaders unpack 8 FP4/INT4 values per uint32 in registers
- Dequantized values used for QK^T dot product and PV accumulation
- No intermediate FP16 materialization of full KV; dequant is fused into the tile loop

**Dispatch grid:**
```
threadgroups = [num_heads_q, ceil(seq_q / ROWS_PER_TG), batch]
threads_per_tg = THREADS_PER_ROW * ROWS_PER_TG = 128
```

Each threadgroup handles 4 query rows. K/V tiles are loaded cooperatively by all 128 threads, then each simdgroup (32 threads) computes the attention for its assigned query row.

## Python API

```python
from metal_marlin.python.kv_cache import KVCache, CacheConfig

config = CacheConfig(
    num_layers=32,
    num_heads=32,        # Q heads
    num_kv_heads=8,      # KV heads (GQA with 4:1 ratio)
    head_dim=128,
    max_seq_len=4096,
    dtype=mx.float16,
    quantize=False,      # Set True for FP4 KV cache
)

cache = KVCache(config, batch_size=1)

# Prefill: process full prompt
k_full, v_full = cache.update(layer_idx=0, k_new=k_prompt, v_new=v_prompt)
cache.advance(num_tokens=prompt_len)

# Decode loop: one token at a time
k_full, v_full = cache.update(layer_idx=0, k_new=k_step, v_new=v_step)
cache.advance(num_tokens=1)

# Memory monitoring
print(f"Cache usage: {cache.memory_usage_mb():.1f} MB")

# Reset for new sequence
cache.reset()
```

## Performance Considerations

**Prefill phase** is compute-bound: the full sequence is processed via flash attention with no cache reads. The bottleneck is the GEMM throughput for QKV projections and the attention kernel itself.

**Decode phase** is memory-bandwidth-bound: each step reads the entire cached K and V for all layers. With a 4K context, 32 layers, 8 KV heads, and head_dim=128, a single decode step reads:
```
2(K+V) * 32 layers * 8 heads * 4096 * 128 * 2 bytes = 4 GB
```

At M4 Max bandwidth (~400 GB/s), this takes ~10ms per token just for cache reads, making KV cache bandwidth the primary decode bottleneck. FP4 quantization reduces this to ~1 GB of reads (~2.5ms), directly improving token latency for long contexts.

**Threadgroup memory** in flash attention kernels holds one K tile and one V tile simultaneously:
```
K_tile: TILE_KV * head_dim * sizeof(half) = 64 * 128 * 2 = 16 KB
V_tile: same = 16 KB
Total: 32 KB per threadgroup
```

This fits within Apple Silicon's 32 KB threadgroup memory limit. Double-buffering uses ping-pong between two tile slots (not doubling memory, but overlapping loads with compute via the Metal command encoder's implicit pipelining).
