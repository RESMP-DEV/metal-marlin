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
   Pure numpy reference implementation for decode-only attention.

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

    # Separate cache API (decode):
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

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax


def paged_attention_v1(
    query: NDArray[Any],  # [num_seqs, num_heads, head_dim]
    k_cache: NDArray[Any],  # [num_blocks, block_size, num_heads, head_dim]
    v_cache: NDArray[Any],  # [num_blocks, block_size, num_heads, head_dim]
    block_tables: NDArray[Any],  # [num_seqs, max_blocks_per_seq] int32
    context_lens: NDArray[Any],  # [num_seqs] int32
    scale: float | None = None,
) -> NDArray[Any]:
    """Paged attention forward pass with separate K/V caches.

    Implements vLLM-style paged attention for decode-only workloads.
    Pure numpy reference implementation.

    Args:
        query: Query vectors [num_seqs, num_heads, head_dim].
        k_cache: Paged key cache [num_blocks, block_size, num_heads, head_dim].
        v_cache: Paged value cache [num_blocks, block_size, num_heads, head_dim].
        block_tables: Logical-to-physical block mapping
            [num_seqs, max_blocks_per_seq] as int32.
        context_lens: Context length per sequence [num_seqs] as int32.
        scale: Attention scale (default: 1/sqrt(head_dim)).

    Returns:
        Attention output [num_seqs, num_heads, head_dim] as float16.
    """
    num_seqs, num_heads, head_dim = query.shape
    block_size = k_cache.shape[1]

    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    # Validate shapes
    assert k_cache.shape == v_cache.shape, f"k_cache {k_cache.shape} != v_cache {v_cache.shape}"
    assert k_cache.shape[2] == num_heads, (
        f"Cache heads {k_cache.shape[2]} != query heads {num_heads}"
    )
    assert k_cache.shape[3] == head_dim
    assert block_tables.shape[0] == num_seqs
    assert context_lens.shape[0] == num_seqs

    # Reference implementation
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
            keys.append(k_cache[phys])  # [block_size, num_heads, head_dim]
            values.append(v_cache[phys])

        k = np.concatenate(keys, axis=0)[:ctx_len]  # [ctx_len, num_heads, head_dim]
        v = np.concatenate(values, axis=0)[:ctx_len]

        # scores: [num_heads, ctx_len]
        scores = np.einsum("hd,thd->ht", q, k) * scale
        attn = softmax(scores, axis=-1)
        # out: [num_heads, head_dim]
        out = np.einsum("ht,thd->hd", attn, v)
        outputs.append(out)

    return np.stack(outputs, axis=0).astype(np.float16)


def paged_attention(
    query: NDArray[Any],
    block_pool: NDArray[Any],
    block_tables: NDArray[Any],
    context_lens: NDArray[Any],
    scale: float,
    num_kv_heads: int,
    block_size: int = 16,
) -> NDArray[Any]:
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
    gathered = np.transpose(
        gathered, (0, 2, 1, 3, 4, 5)
    )  # [num_seqs, 2, max_blocks, block_size, ...]
    gathered = gathered.reshape(num_seqs, 2, max_context, num_kv_heads, head_dim)

    # Split K and V: each [num_seqs, max_context, num_kv_heads, head_dim]
    keys = gathered[:, 0]  # [num_seqs, max_context, num_kv_heads, head_dim]
    values = gathered[:, 1]  # [num_seqs, max_context, num_kv_heads, head_dim]

    # Transpose to [num_seqs, num_kv_heads, max_context, head_dim]
    keys = np.transpose(keys, (0, 2, 1, 3))
    values = np.transpose(values, (0, 2, 1, 3))

    # GQA expansion: repeat KV heads to match query heads
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        keys = np.repeat(keys, repeat_factor, axis=1)
        values = np.repeat(values, repeat_factor, axis=1)

    # Compute attention scores: [num_seqs, num_heads, seq_len, max_context]
    attn_weights = (query @ np.transpose(keys, (0, 1, 3, 2))) * scale

    # Build validity mask from context_lens.
    # For each sequence, positions >= context_len should be masked out.
    # context_lens: [num_seqs]
    kv_positions = np.arange(max_context)[None, :]  # [1, max_context]
    context_lens_2d = context_lens[:, None]  # [num_seqs, 1]
    # valid: [num_seqs, max_context] - True where position < context_len
    valid_mask = kv_positions < context_lens_2d

    # Expand for broadcasting: [num_seqs, 1, 1, max_context]
    valid_mask = valid_mask[:, None, None, :]
    attn_weights = np.where(valid_mask, attn_weights, float("-inf"))

    # Causal mask for prefill (seq_len > 1)
    if seq_len > 1:
        # Query position i can attend to KV positions <= i
        # For prefill, query positions map to the last seq_len positions
        # of the context (positions context_len-seq_len through context_len-1)
        q_positions = np.arange(seq_len)[None, None, :, None]  # [1, 1, seq_len, 1]
        kv_pos_expanded = kv_positions[None, None, None, :]  # [1, 1, 1, max_context]

        # Each query at position q_pos can attend to kv_pos where:
        # kv_pos <= (context_len - seq_len + q_pos)
        # i.e., the query at offset i in the prefill chunk sees all prior
        # context plus positions 0..i of the current chunk.
        offsets = context_lens[:, None, None, None] - seq_len + q_positions
        causal_mask = kv_pos_expanded <= offsets
        attn_weights = np.where(causal_mask, attn_weights, float("-inf"))

    attn_weights = softmax(attn_weights, axis=-1)

    # Compute output: [num_seqs, num_heads, seq_len, head_dim]
    output = attn_weights @ values

    return output


def write_kv_to_blocks(
    block_pool: NDArray[Any],
    block_tables: NDArray[Any],
    keys: NDArray[Any],
    values: NDArray[Any],
    slot_offsets: NDArray[Any],
    block_size: int = 16,
) -> NDArray[Any]:
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

    # Make a copy to avoid modifying the original
    block_pool = block_pool.copy()

    for seq_idx in range(num_seqs):
        offset = int(slot_offsets[seq_idx])
        seq_blocks = block_tables[seq_idx]

        for tok_idx in range(num_tokens):
            abs_pos = offset + tok_idx
            block_idx_in_seq = abs_pos // block_size
            slot_in_block = abs_pos % block_size
            physical_block = int(seq_blocks[block_idx_in_seq])

            # Write K
            block_pool[physical_block, 0, slot_in_block] = keys[seq_idx, tok_idx]
            # Write V
            block_pool[physical_block, 1, slot_in_block] = values[seq_idx, tok_idx]

    return block_pool
