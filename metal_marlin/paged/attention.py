"""Paged attention: scaled dot-product attention over block-table KV storage.

Reads K/V from a contiguous block pool indexed by per-sequence block tables,
avoiding the need to gather/scatter individual sequence KV caches into
contiguous buffers before computing attention.

Two APIs are provided:

1. paged_attention() - Uses unified block pool layout
   [num_blocks, 2, block_size, num_kv_heads, head_dim].
   Materializes gathered KV for attention (simpler, GQA-aware, supports prefill).

2. paged_attention_v1() - Uses separate k_cache/v_cache with layout
   [num_blocks, block_size, num_heads, head_dim].
   Dispatches to a Metal kernel with online softmax for decode-only attention
   (no attention matrix materialization). Falls back to reference impl if
   use_metal=False or head_dim > 128.

Usage:
    # Unified pool API (prefill + decode):
    logits = paged_attention(
        query=q,                      # [num_seqs, num_heads, seq_len, head_dim]
        block_pool=allocator._storage, # [num_blocks, 2, block_size, num_kv_heads, head_dim]
        block_tables=block_tables,     # [num_seqs, max_blocks_per_seq]
        context_lens=context_lens,     # [num_seqs]
        scale=head_dim ** -0.5,
        num_kv_heads=8,
    )

    # Separate cache API (decode, Metal kernel):
    out = paged_attention_v1(
        query=q,                  # [num_seqs, num_heads, head_dim]
        k_cache=k_cache,          # [num_blocks, block_size, num_heads, head_dim]
        v_cache=v_cache,          # [num_blocks, block_size, num_heads, head_dim]
        block_tables=block_tables,# [num_seqs, max_blocks_per_seq] int32
        context_lens=context_lens,# [num_seqs] int32
        scale=head_dim ** -0.5,
    )
"""

from __future__ import annotations

import mlx.core as mx

# ---------------------------------------------------------------------------
# Metal kernel source for paged_attention_v1 (online softmax, decode-only)
# ---------------------------------------------------------------------------

_PAGED_ATTN_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# The kernel body uses compile-time template constants injected by MLX:
#   NUM_SEQS, NUM_HEADS, HEAD_DIM, BLOCK_SIZE, MAX_BLOCKS_PER_SEQ,
#   SCALE_NUM, SCALE_DEN
#
# Each threadgroup handles one (sequence, head) pair.
# Threads cooperatively compute Q*K dot products via simd_sum reduction
# and accumulate V with online softmax rescaling.
_PAGED_ATTN_SOURCE = """
    const uint seq_idx = threadgroup_position_in_grid.x;
    const uint head_idx = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;

    if (seq_idx >= NUM_SEQS || head_idx >= NUM_HEADS) return;

    const float scale = float(SCALE_NUM) / float(SCALE_DEN);

    // Load query vector for this (seq, head) into threadgroup memory
    // Q layout (flattened): [num_seqs, num_heads, head_dim]
    const uint q_base = (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;

    threadgroup float q_shared[256];  // HEAD_DIM <= 256
    for (uint d = tid; d < HEAD_DIM; d += THREADS_PER_TG) {
        q_shared[d] = float(Q[q_base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ctx_len = context_lens[seq_idx];
    const int num_blocks_used = (ctx_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Online softmax state + output accumulator (per-thread partial dims)
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Each thread accumulates a strided subset of HEAD_DIM
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // Thread handles dims: tid, tid+TG, tid+2*TG, tid+3*TG (if < HEAD_DIM)

    // Shared memory for broadcasting dot product results
    threadgroup float score_shared[1];

    for (int blk = 0; blk < num_blocks_used; blk++) {
        const int block_physical = block_tables[seq_idx * MAX_BLOCKS_PER_SEQ + blk];
        const int block_start = blk * BLOCK_SIZE;
        const int block_end = min(block_start + BLOCK_SIZE, ctx_len);
        const int tokens_in_block = block_end - block_start;

        for (int t = 0; t < tokens_in_block; t++) {
            // K layout (flattened): [num_blocks, block_size, num_heads, head_dim]
            const uint k_base = ((uint(block_physical) * BLOCK_SIZE + uint(t))
                                 * NUM_HEADS + head_idx) * HEAD_DIM;

            // Cooperative dot product: each thread handles strided dims
            float partial_dot = 0.0f;
            for (uint d = tid; d < HEAD_DIM; d += THREADS_PER_TG) {
                partial_dot += q_shared[d] * float(k_cache[k_base + d]);
            }

            // Reduce within simdgroup
            float sg_dot = simd_sum(partial_dot);

            // Cross-simdgroup reduction via shared memory
            const uint sg_idx = tid / 32;
            const uint sg_lane = tid % 32;

            threadgroup float sg_partials[4];
            if (sg_lane == 0) {
                sg_partials[sg_idx] = sg_dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                for (uint s = 0; s < (THREADS_PER_TG + 31) / 32; s++) {
                    total += sg_partials[s];
                }
                score_shared[0] = total * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float score = score_shared[0];

            // Online softmax update
            float old_max = running_max;
            running_max = max(running_max, score);
            float rescale = exp(old_max - running_max);
            running_sum = running_sum * rescale + exp(score - running_max);

            // Rescale existing accumulator
            for (int i = 0; i < 4; i++) {
                acc[i] *= rescale;
            }

            // Accumulate V weighted by unnormalized attention weight
            // V layout same as K: [num_blocks, block_size, num_heads, head_dim]
            const uint v_base = ((uint(block_physical) * BLOCK_SIZE + uint(t))
                                 * NUM_HEADS + head_idx) * HEAD_DIM;
            float attn_w = exp(score - running_max);

            for (int i = 0; i < 4; i++) {
                uint d = tid + uint(i) * THREADS_PER_TG;
                if (d < HEAD_DIM) {
                    acc[i] += attn_w * float(v_cache[v_base + d]);
                }
            }
        }
    }

    // Normalize and write output
    // O layout: [num_seqs, num_heads, head_dim]
    const uint o_base = (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

    for (int i = 0; i < 4; i++) {
        uint d = tid + uint(i) * THREADS_PER_TG;
        if (d < HEAD_DIM) {
            O[o_base + d] = half(acc[i] * inv_sum);
        }
    }
"""

_PAGED_ATTN_THREADS_PER_TG = 128

# ---------------------------------------------------------------------------
# Kernel cache (lazy initialization)
# ---------------------------------------------------------------------------

_paged_attn_kernel: object | None = None


def _get_paged_attn_kernel() -> object:
    global _paged_attn_kernel
    if _paged_attn_kernel is None:
        _paged_attn_kernel = mx.fast.metal_kernel(
            name="paged_attention_v1",
            input_names=["Q", "k_cache", "v_cache", "block_tables", "context_lens"],
            output_names=["O"],
            source=_PAGED_ATTN_SOURCE,
            header=_PAGED_ATTN_HEADER,
            ensure_row_contiguous=True,
        )
    return _paged_attn_kernel


# ---------------------------------------------------------------------------
# Public API: paged_attention_v1 (separate k/v cache, Metal kernel path)
# ---------------------------------------------------------------------------


def paged_attention_v1(
    query: mx.array,         # [num_seqs, num_heads, head_dim]
    k_cache: mx.array,       # [num_blocks, block_size, num_heads, head_dim]
    v_cache: mx.array,       # [num_blocks, block_size, num_heads, head_dim]
    block_tables: mx.array,  # [num_seqs, max_blocks_per_seq] int32
    context_lens: mx.array,  # [num_seqs] int32
    scale: float | None = None,
    use_metal: bool = True,
) -> mx.array:
    """Paged attention forward pass with separate K/V caches.

    Implements vLLM-style paged attention for decode-only workloads.
    The Metal kernel streams through KV blocks using online softmax,
    avoiding materialization of the full attention matrix.

    For head_dim > 512, falls back to the reference implementation
    since the kernel's per-thread accumulator assumes head_dim <= 4*TG.

    Args:
        query: Query vectors [num_seqs, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_heads, head_dim].
        block_tables: Logical-to-physical block mapping
            [num_seqs, max_blocks_per_seq] as int32.
        context_lens: Context length per sequence [num_seqs] as int32.
        scale: Attention scale (default: 1/sqrt(head_dim)).
        use_metal: Dispatch to Metal kernel if True (default).

    Returns:
        Attention output [num_seqs, num_heads, head_dim] as float16.
    """
    num_seqs, num_heads, head_dim = query.shape
    block_size = k_cache.shape[1]
    max_blocks_per_seq = block_tables.shape[1]

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Validate shapes
    assert k_cache.shape == v_cache.shape, (
        f"k_cache {k_cache.shape} != v_cache {v_cache.shape}"
    )
    assert k_cache.shape[2] == num_heads, (
        f"Cache heads {k_cache.shape[2]} != query heads {num_heads}"
    )
    assert k_cache.shape[3] == head_dim
    assert block_tables.shape[0] == num_seqs
    assert context_lens.shape[0] == num_seqs

    # Kernel accumulator supports head_dim <= 4 * THREADS_PER_TG = 512
    if not use_metal or head_dim > 4 * _PAGED_ATTN_THREADS_PER_TG:
        return _paged_attention_v1_ref(
            query, k_cache, v_cache, block_tables, context_lens, scale
        )

    # Encode scale as fixed-point rational for integer template params
    scale_den = 1000000
    scale_num = int(round(scale * scale_den))

    # Ensure types
    q = query.astype(mx.float16)
    k_c = k_cache.astype(mx.float16)
    v_c = v_cache.astype(mx.float16)
    bt = block_tables.astype(mx.int32)
    cl = context_lens.astype(mx.int32)

    # One threadgroup per (sequence, head) pair
    grid = (num_seqs, num_heads, 1)
    threadgroup = (_PAGED_ATTN_THREADS_PER_TG, 1, 1)

    kernel = _get_paged_attn_kernel()
    outputs = kernel(
        inputs=[
            q.reshape(-1),
            k_c.reshape(-1),
            v_c.reshape(-1),
            bt.reshape(-1),
            cl.reshape(-1),
        ],
        template=[
            ("NUM_SEQS", num_seqs),
            ("NUM_HEADS", num_heads),
            ("HEAD_DIM", head_dim),
            ("BLOCK_SIZE", block_size),
            ("MAX_BLOCKS_PER_SEQ", max_blocks_per_seq),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
            ("THREADS_PER_TG", _PAGED_ATTN_THREADS_PER_TG),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(num_seqs * num_heads * head_dim,)],
        output_dtypes=[mx.float16],
    )

    return outputs[0].reshape(num_seqs, num_heads, head_dim)


def _paged_attention_v1_ref(
    query: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    scale: float,
) -> mx.array:
    """Reference implementation for paged_attention_v1.

    Gathers KV from physical blocks and computes standard attention per sequence.
    """
    num_seqs, num_heads, head_dim = query.shape
    block_size = k_cache.shape[1]

    outputs = []
    for seq_idx in range(num_seqs):
        ctx_len = int(context_lens[seq_idx])
        q = query[seq_idx]  # [num_heads, head_dim]
        num_blocks_used = (ctx_len + block_size - 1) // block_size

        # Gather KV from physical blocks
        keys = []
        values = []
        for blk in range(num_blocks_used):
            phys = int(block_tables[seq_idx, blk])
            keys.append(k_cache[phys])      # [block_size, num_heads, head_dim]
            values.append(v_cache[phys])

        k = mx.concatenate(keys, axis=0)[:ctx_len]    # [ctx_len, num_heads, head_dim]
        v = mx.concatenate(values, axis=0)[:ctx_len]

        # scores: [num_heads, ctx_len]
        scores = mx.einsum("hd,thd->ht", q, k) * scale
        attn = mx.softmax(scores, axis=-1)
        # out: [num_heads, head_dim]
        out = mx.einsum("ht,thd->hd", attn, v)
        outputs.append(out)

    return mx.stack(outputs, axis=0).astype(mx.float16)


# ---------------------------------------------------------------------------
# Public API: paged_attention (unified block pool, gather-based)
# ---------------------------------------------------------------------------


def paged_attention(
    query: mx.array,
    block_pool: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    scale: float,
    num_kv_heads: int,
    block_size: int = 16,
) -> mx.array:
    """Compute scaled dot-product attention with paged KV cache.

    Gathers K/V from the block pool using per-sequence block tables,
    expands KV heads for GQA, and computes masked attention.

    Args:
        query: Query tensor [num_seqs, num_heads, seq_len, head_dim].
        block_pool: Pre-allocated KV storage
            [num_blocks, 2, block_size, num_kv_heads, head_dim].
        block_tables: Block indices per sequence [num_seqs, max_blocks_per_seq].
        context_lens: Number of valid KV tokens per sequence [num_seqs].
        scale: Attention scale factor (typically head_dim ** -0.5).
        num_kv_heads: Number of KV heads (for GQA expansion).
        block_size: Tokens per block.

    Returns:
        Attention output [num_seqs, num_heads, seq_len, head_dim].
    """
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    max_blocks = block_tables.shape[1]
    max_context = max_blocks * block_size

    # Gather K and V from block pool for each sequence.
    # block_pool: [num_blocks, 2, block_size, num_kv_heads, head_dim]
    # We need to gather blocks for each sequence and reshape to
    # [num_seqs, max_context, num_kv_heads, head_dim]

    # Flatten block indices for gather
    flat_indices = block_tables.reshape(-1)  # [num_seqs * max_blocks]

    # Gather: [num_seqs * max_blocks, 2, block_size, num_kv_heads, head_dim]
    gathered = block_pool[flat_indices]

    # Reshape to [num_seqs, max_blocks * block_size, num_kv_heads, head_dim]
    gathered = gathered.reshape(num_seqs, max_blocks, 2, block_size, num_kv_heads, head_dim)
    gathered = gathered.transpose(0, 2, 1, 3, 4, 5)  # [num_seqs, 2, max_blocks, block_size, ...]
    gathered = gathered.reshape(num_seqs, 2, max_context, num_kv_heads, head_dim)

    # Split K and V: each [num_seqs, max_context, num_kv_heads, head_dim]
    keys = gathered[:, 0]    # [num_seqs, max_context, num_kv_heads, head_dim]
    values = gathered[:, 1]  # [num_seqs, max_context, num_kv_heads, head_dim]

    # Transpose to [num_seqs, num_kv_heads, max_context, head_dim]
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # GQA expansion: repeat KV heads to match query heads
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        keys = mx.repeat(keys, repeat_factor, axis=1)
        values = mx.repeat(values, repeat_factor, axis=1)

    # Compute attention scores: [num_seqs, num_heads, seq_len, max_context]
    attn_weights = (query @ keys.transpose(0, 1, 3, 2)) * scale

    # Build validity mask from context_lens.
    # For each sequence, positions >= context_len should be masked out.
    # context_lens: [num_seqs]
    kv_positions = mx.arange(max_context)[None, :]  # [1, max_context]
    context_lens_2d = context_lens[:, None]           # [num_seqs, 1]
    # valid: [num_seqs, max_context] - True where position < context_len
    valid_mask = kv_positions < context_lens_2d

    # Expand for broadcasting: [num_seqs, 1, 1, max_context]
    valid_mask = valid_mask[:, None, None, :]
    attn_weights = mx.where(valid_mask, attn_weights, mx.array(float("-inf")))

    # Causal mask for prefill (seq_len > 1)
    if seq_len > 1:
        # Query position i can attend to KV positions <= i
        # For prefill, query positions map to the last seq_len positions
        # of the context (positions context_len-seq_len through context_len-1)
        q_positions = mx.arange(seq_len)[None, None, :, None]  # [1, 1, seq_len, 1]
        kv_pos_expanded = kv_positions[None, None, None, :]     # [1, 1, 1, max_context]

        # Each query at position q_pos can attend to kv_pos where:
        # kv_pos <= (context_len - seq_len + q_pos)
        # i.e., the query at offset i in the prefill chunk sees all prior
        # context plus positions 0..i of the current chunk.
        offsets = (context_lens[:, None, None, None] - seq_len + q_positions)
        causal_mask = kv_pos_expanded <= offsets
        attn_weights = mx.where(causal_mask, attn_weights, mx.array(float("-inf")))

    attn_weights = mx.softmax(attn_weights, axis=-1)

    # Compute output: [num_seqs, num_heads, seq_len, head_dim]
    output = attn_weights @ values

    return output


def write_kv_to_blocks(
    block_pool: mx.array,
    block_tables: mx.array,
    keys: mx.array,
    values: mx.array,
    slot_offsets: mx.array,
    block_size: int = 16,
) -> mx.array:
    """Write K/V vectors into the block pool at specified positions.

    Used after computing K/V projections to store them in the paged cache.
    For decode, writes a single token per sequence. For prefill, writes
    the full prompt KV.

    Args:
        block_pool: Mutable KV storage
            [num_blocks, 2, block_size, num_kv_heads, head_dim].
        block_tables: Block indices per sequence [num_seqs, max_blocks].
        keys: Key vectors to write [num_seqs, num_tokens, num_kv_heads, head_dim].
        values: Value vectors to write [num_seqs, num_tokens, num_kv_heads, head_dim].
        slot_offsets: Starting token offset within the sequence for each
            sequence [num_seqs]. Determines which block/slot to write into.
        block_size: Tokens per block.

    Returns:
        Updated block_pool array.
    """
    num_seqs = keys.shape[0]
    num_tokens = keys.shape[1]

    for seq_idx in range(num_seqs):
        offset = int(slot_offsets[seq_idx].item())
        seq_blocks = block_tables[seq_idx]

        for tok_idx in range(num_tokens):
            abs_pos = offset + tok_idx
            block_idx_in_seq = abs_pos // block_size
            slot_in_block = abs_pos % block_size
            physical_block = int(seq_blocks[block_idx_in_seq].item())

            # Write K
            block_pool = block_pool.at[physical_block, 0, slot_in_block].add(
                keys[seq_idx, tok_idx] - block_pool[physical_block, 0, slot_in_block]
            )
            # Write V
            block_pool = block_pool.at[physical_block, 1, slot_in_block].add(
                values[seq_idx, tok_idx] - block_pool[physical_block, 1, slot_in_block]
            )

    return block_pool
