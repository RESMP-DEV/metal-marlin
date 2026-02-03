# Copy-on-Write Prompt Sharing

## Overview

The `cache_manager.py` module implements zero-copy prompt sharing using Copy-on-Write (COW) semantics. This enables efficient memory usage when multiple sequences share common prompt prefixes.

## Key Features

### 1. Zero-Copy Block Sharing
- Sequences with identical prompt prefixes share physical KV cache blocks
- Sharing implemented via reference counting on blocks
- No memory copy until divergence occurs

### 2. Automatic Copy-on-Write
- First write to a shared block automatically triggers COW
- Original sequence remains unchanged
- Child sequence gets private copy only when needed

### 3. Prompt Prefix Caching
- Prompts are hashed and cached for reuse
- Subsequent sequences with same prompt can fork from cached version
- Automatic cache invalidation when source sequences are removed

### 4. Memory Statistics
- Tracks number of shared blocks created
- Counts COW operations performed
- Measures prompt cache hit rate

## API

### Basic Usage

```python
from metal_marlin.paged.cache_manager import PagedKVCache
from metal_marlin.paged.kv_block import KVBlockConfig
import numpy as np

# Create cache
config = KVBlockConfig(block_size=16, num_heads=32, head_dim=128)
cache = PagedKVCache(config, num_blocks=1024)

# Add parent sequence
cache.add_sequence(seq_id=0)
for _ in range(100):
    k = np.random.randn(32, 128).astype(np.float16)
    v = np.random.randn(32, 128).astype(np.float16)
    cache.append_kv(seq_id=0, key=k, value=v)

# Fork child sequences (zero-copy)
cache.fork_sequence(src_id=0, dst_id=1)
cache.fork_sequence(src_id=0, dst_id=2)

# Children share parent's blocks until they diverge
# Writing to child triggers COW automatically
k_new = np.random.randn(32, 128).astype(np.float16)
v_new = np.random.randn(32, 128).astype(np.float16)
cache.append_kv(seq_id=1, key=k_new, value=v_new)  # COW happens here
```

### Prompt Caching

```python
# First sequence with prompt
prompt_tokens = [1, 2, 3, ..., 100]
cache.add_sequence_with_prompt(seq_id=0, prompt_tokens=prompt_tokens)

# Subsequent sequences reuse cached blocks
cache.add_sequence_with_prompt(seq_id=1, prompt_tokens=prompt_tokens)  # Cache hit!
cache.add_sequence_with_prompt(seq_id=2, prompt_tokens=prompt_tokens)  # Cache hit!

# Check statistics
stats = cache.get_stats()
print(f"Prompt cache hits: {stats.prompt_cache_hits}")
print(f"Shared blocks: {stats.shared_prompt_blocks}")
```

### Batch Prompt Sharing

```python
# Share prompt across multiple sequences
cache.add_sequence(0)
# ... fill with prompt tokens ...

dst_ids = [1, 2, 3, 4, 5]
results = cache.share_prompt_blocks(
    src_seq_id=0,
    dst_seq_ids=dst_ids,
    num_prefix_tokens=50,  # Share first 50 tokens
)
# All sequences now share the first 50 tokens
```

## Use Cases

1. **Batch Inference with System Prompts**
   - Single system prompt shared across all requests
   - Saves memory proportional to batch size × prompt length

2. **Few-Shot Learning**
   - Examples shared, queries unique
   - Efficient for large few-shot contexts

3. **Beam Search / Speculative Decoding**
   - Multiple candidates share prefix
   - Diverge only as predictions differ

4. **Multi-Turn Conversations**
   - History shared across turns
   - New responses extend without copying history

## Implementation Details

### Reference Counting

Each block maintains a `ref_count`:
- `ref_count == 1`: Exclusive access, can write directly
- `ref_count > 1`: Shared, COW required before write

### COW Operation

When writing to a shared block:
1. Check `ref_count > 1`
2. Allocate new physical block
3. Copy content from shared block to new block
4. Decrement old block's refcount
5. Update page table to point to new block
6. Write to new block

### Memory Savings

For N sequences sharing M blocks:
- **Without COW**: N × M blocks allocated
- **With COW**: M + diverged blocks allocated
- **Savings**: (N-1) × M blocks (until divergence)

Example: 10 sequences sharing 100-block prompt:
- Traditional: 1000 blocks
- With COW: 100 blocks (until writes occur)
- **90% memory reduction**

## Performance

- **Fork operation**: O(M) where M = number of blocks (refcount increment)
- **COW trigger**: O(1) block allocation + O(block_size) copy
- **Prompt cache lookup**: O(1) hash table lookup
- **Memory overhead**: Minimal (refcount per block, hash table for cache)

## Thread Safety

⚠️  **Not thread-safe by default**. If using from multiple threads:
- Protect cache operations with locks
- Or use separate cache instances per thread

## Limitations

1. COW granularity is per-block (typically 16 tokens)
   - Sub-block divergence still triggers full block copy
   
2. Prompt cache uses content hashing
   - Same token sequence = same hash (expected)
   - Different tokenizations won't match (expected)

3. Cache invalidation is lazy
   - Stale entries cleaned on next access to that hash

## Testing

Run tests to verify COW functionality:

```bash
cd contrib/metal_marlin
uv run python -c "from metal_marlin.paged.cache_manager import PagedKVCache; ..."
```

## References

- Based on vLLM's PagedAttention architecture (Kwon et al., SOSP 2023)
- COW mechanism inspired by Unix fork() semantics
- Block-level granularity chosen for GPU efficiency
