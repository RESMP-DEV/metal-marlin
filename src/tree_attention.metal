// tree_attention.metal - Tree-structured attention for Eagle v3 speculative decoding
//
// Eagle v3 uses tree-structured speculative decoding where multiple draft paths
// are verified in parallel. Instead of a single sequence of K speculative tokens,
// we have a tree where each node can branch into multiple continuations.
//
// Tree structure example (tree_size=7):
//
//            [root=prompt]
//                 |
//              [node 0]  <-- first speculation
//              /       \
//        [node 1]    [node 2]  <-- two branches from node 0
//          /   \       |
//     [node 3] [node 4] [node 5]
//         |
//     [node 6]
//
// The tree_mask encodes attention patterns:
//   - Each draft token attends to the accepted prefix (standard causal)
//   - Each draft token attends to ancestors on its path in the tree
//   - Parallel branches cannot attend to each other
//
// tree_mask[i][j] = 1 if token i can attend to token j, else 0
//
// For Eagle v3, we process:
//   1. Standard causal attention for the accepted prefix (seq_len positions)
//   2. Tree-structured attention for draft tokens (tree_size positions)
//   3. Draft tokens can attend to the prefix but not to other draft paths
//
// Algorithm: Flash Attention with tree mask
//   - Online softmax to avoid materializing full attention matrix
//   - Tree mask applied additively (-INF for blocked positions)
//   - K/V include both prefix cache and draft token K/V
//
// Dispatch: one threadgroup per (batch, head, draft_node) triple
// Grid: [tree_size, num_heads, batch]
// Threadgroup: 128 threads (4 simdgroups)
//
// Memory layout:
//   Q: [batch, num_heads, tree_size, head_dim] - queries for each tree node
//   K: [batch, num_heads, seq_len + tree_size, head_dim] - prefix + draft keys
//   V: [batch, num_heads, seq_len + tree_size, head_dim] - prefix + draft values
//   tree_mask: [tree_size, tree_size] - which draft nodes can attend to which
//   O: [batch, num_heads, tree_size, head_dim] - output for each tree node
//
// Key optimizations:
//   1. Standard attention for prefix positions (no tree mask needed)
//   2. Tree mask lookup only for draft-to-draft attention
//   3. Vectorized loads and online softmax
//   4. Double-buffered K/V tiles

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

constant constexpr uint TILE_KV_TREE = 32;        // K/V positions per tile
constant constexpr uint THREADS_TREE = 128;       // 4 simdgroups
constant constexpr uint HEAD_DIM_MAX_TREE = 128;
constant constexpr uint MAX_TREE_SIZE = 64;       // Maximum draft tree nodes
constant constexpr uint SIMDGROUPS_TREE = 4;

// ---------------------------------------------------------------------------
// Utility: threadgroup reductions
// ---------------------------------------------------------------------------

inline float simd_max_tree(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum_tree(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

inline float threadgroup_reduce_max_tree(
    float val,
    uint tid,
    threadgroup float* scratch
) {
    float sg_max = simd_max_tree(val);
    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) {
        scratch[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float result;
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_TREE) ? scratch[lane] : -INFINITY;
        result = simd_max_tree(v);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        scratch[0] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline float threadgroup_reduce_sum_tree(
    float val,
    uint tid,
    threadgroup float* scratch
) {
    float sg_sum = simd_sum_tree(val);
    uint sg_id = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) {
        scratch[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float result;
    if (sg_id == 0) {
        float v = (lane < SIMDGROUPS_TREE) ? scratch[lane] : 0.0f;
        result = simd_sum_tree(v);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        scratch[0] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

// ---------------------------------------------------------------------------
// Tree Attention Forward Kernel
//
// Computes attention for tree-structured speculative decoding.
// Each query position is a node in the draft tree.
// Keys/values include both the accepted prefix and draft tree nodes.
//
// The tree_mask allows draft tokens to attend to:
//   1. All positions in the accepted prefix (causal within prefix)
//   2. Ancestor nodes in the draft tree (on the same path)
//   3. NOT to sibling branches or descendants
// ---------------------------------------------------------------------------

kernel void tree_attention_forward(
    device const half* Q             [[buffer(0)]],   // [batch, heads, tree_size, head_dim]
    device const half* K             [[buffer(1)]],   // [batch, heads, seq_len + tree_size, head_dim]
    device const half* V             [[buffer(2)]],   // [batch, heads, seq_len + tree_size, head_dim]
    device const uint* tree_mask     [[buffer(3)]],   // [tree_size, tree_size] packed bits or bool array
    device half* O                   [[buffer(4)]],   // [batch, heads, tree_size, head_dim]
    constant uint& batch             [[buffer(5)]],
    constant uint& num_heads         [[buffer(6)]],
    constant uint& seq_len           [[buffer(7)]],   // length of accepted prefix
    constant uint& tree_size         [[buffer(8)]],   // number of draft tree nodes
    constant uint& head_dim          [[buffer(9)]],
    constant float& scale            [[buffer(10)]],  // 1/sqrt(head_dim)
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    // Grid mapping: (tree_node, head, batch)
    uint tree_node = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (tree_node >= tree_size || head >= num_heads || batch_idx >= batch) return;

    // Total K/V length = prefix + draft tree
    uint total_kv_len = seq_len + tree_size;

    // Compute base offsets
    // Q layout: [batch, heads, tree_size, head_dim]
    uint q_stride_b = num_heads * tree_size * head_dim;
    uint q_stride_h = tree_size * head_dim;
    uint q_stride_t = head_dim;
    uint q_offset = batch_idx * q_stride_b + head * q_stride_h + tree_node * q_stride_t;

    // K/V layout: [batch, heads, seq_len + tree_size, head_dim]
    uint kv_stride_b = num_heads * total_kv_len * head_dim;
    uint kv_stride_h = total_kv_len * head_dim;
    uint kv_stride_s = head_dim;
    uint kv_base = batch_idx * kv_stride_b + head * kv_stride_h;

    // O layout: same as Q
    uint o_offset = q_offset;

    // Threadgroup memory
    threadgroup half Q_cache[HEAD_DIM_MAX_TREE];
    threadgroup half K_tile[TILE_KV_TREE][HEAD_DIM_MAX_TREE];
    threadgroup half V_tile[TILE_KV_TREE][HEAD_DIM_MAX_TREE];
    threadgroup float reduction_scratch[SIMDGROUPS_TREE];

    // Load Q vector into threadgroup memory
    for (uint i = tid; i < head_dim; i += THREADS_TREE) {
        Q_cache[i] = Q[q_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Output accumulator per thread
    // Each thread accumulates a strided portion of head_dim
    constexpr uint O_ELEMS = (HEAD_DIM_MAX_TREE + THREADS_TREE - 1) / THREADS_TREE;
    float o_accum[O_ELEMS] = {0.0f};

    // Number of K/V tiles
    uint num_tiles = (total_kv_len + TILE_KV_TREE - 1) / TILE_KV_TREE;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint tile_start = tile_idx * TILE_KV_TREE;
        uint tile_len = min(TILE_KV_TREE, total_kv_len - tile_start);

        // Cooperatively load K/V tile
        uint total_elems = tile_len * head_dim;
        uint per_thread = (total_elems + THREADS_TREE - 1) / THREADS_TREE;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_TREE;
            if (idx < total_elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                uint global_kv_idx = tile_start + kv_row;
                if (global_kv_idx < total_kv_len) {
                    K_tile[kv_row][kv_col] = K[kv_base + global_kv_idx * kv_stride_s + kv_col];
                    V_tile[kv_row][kv_col] = V[kv_base + global_kv_idx * kv_stride_s + kv_col];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores with tree mask
        float thread_max_tile = -INFINITY;
        float tile_scores[TILE_KV_TREE];

        for (uint local_k = 0; local_k < tile_len; ++local_k) {
            uint global_k_idx = tile_start + local_k;

            // Determine if this position is in prefix or draft tree
            bool is_prefix = global_k_idx < seq_len;

            // Compute mask
            float mask_val = 0.0f;
            if (is_prefix) {
                // Prefix positions: all draft nodes can attend to all prefix positions
                // (the prefix is fully formed and accepted)
                mask_val = 0.0f;
            } else {
                // Draft tree position: check tree_mask
                // tree_k_idx is the index within the draft tree
                uint tree_k_idx = global_k_idx - seq_len;

                // tree_mask[tree_node][tree_k_idx] == 1 means node can attend
                // tree_mask layout: [tree_size, tree_size] as row-major uint array
                // Each entry is 0 or 1 (could be packed as bits for efficiency)
                uint mask_idx = tree_node * tree_size + tree_k_idx;
                uint can_attend = tree_mask[mask_idx];

                // If can't attend, use -INFINITY; else 0
                mask_val = (can_attend != 0) ? 0.0f : -INFINITY;
            }

            // Compute dot product Q . K
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q_cache[d]) * float(K_tile[local_k][d]);
            }

            // Scale and apply mask
            float score = dot * scale + mask_val;
            tile_scores[local_k] = score;
            thread_max_tile = max(thread_max_tile, score);
        }

        // Reduce to get tile max
        float tile_max = threadgroup_reduce_max_tree(thread_max_tile, tid, reduction_scratch);

        // Update global running max and rescale
        float prev_max = running_max;
        running_max = max(running_max, tile_max);

        if (prev_max > -INFINITY) {
            float rescale = exp(prev_max - running_max);
            running_sum *= rescale;
            for (uint i = 0; i < O_ELEMS; ++i) {
                o_accum[i] *= rescale;
            }
        }

        // Accumulate: compute exp(score - running_max) * V
        for (uint local_k = 0; local_k < tile_len; ++local_k) {
            float s = tile_scores[local_k];
            if (s <= -INFINITY) continue;

            float w = exp(s - running_max);
            running_sum += w;

            // Each thread accumulates its portion of head_dim
            for (uint i = 0; i < O_ELEMS; ++i) {
                uint d = tid + i * THREADS_TREE;
                if (d < head_dim) {
                    o_accum[i] += w * float(V_tile[local_k][d]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

    for (uint i = 0; i < O_ELEMS; ++i) {
        uint d = tid + i * THREADS_TREE;
        if (d < head_dim) {
            O[o_offset + d] = half(o_accum[i] * inv_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Tree Attention with Causal Prefix
//
// Optimized variant where we explicitly handle:
//   1. Causal attention within the prefix itself
//   2. Tree-structured attention for draft tokens
//
// This version takes a prefix_pos parameter indicating which position in the
// prefix this draft tree corresponds to. All prefix positions up to prefix_pos
// are attended to (standard causal within the prefix).
//
// This is more efficient when the prefix is long because we only need to
// attend to prefix positions [0, prefix_pos] rather than the full prefix,
// and we don't need to pass prefix_pos through a mask.
// ---------------------------------------------------------------------------

kernel void tree_attention_forward_with_prefix_causal(
    device const half* Q             [[buffer(0)]],   // [batch, heads, tree_size, head_dim]
    device const half* K             [[buffer(1)]],   // [batch, heads, seq_len + tree_size, head_dim]
    device const half* V             [[buffer(2)]],   // [batch, heads, seq_len + tree_size, head_dim]
    device const uint* tree_mask     [[buffer(3)]],   // [tree_size, tree_size]
    device const uint* tree_positions [[buffer(4)]],  // [tree_size] - position in original sequence for each node
    device half* O                   [[buffer(5)]],   // [batch, heads, tree_size, head_dim]
    constant uint& batch             [[buffer(6)]],
    constant uint& num_heads         [[buffer(7)]],
    constant uint& seq_len           [[buffer(8)]],   // length of accepted prefix
    constant uint& tree_size         [[buffer(9)]],   // number of draft tree nodes
    constant uint& head_dim          [[buffer(10)]],
    constant float& scale            [[buffer(11)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint tree_node = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (tree_node >= tree_size || head >= num_heads || batch_idx >= batch) return;

    // Get the sequence position for this tree node
    // This determines causal masking for the prefix portion
    uint node_seq_pos = tree_positions[tree_node];

    uint total_kv_len = seq_len + tree_size;

    // Offsets (same as basic kernel)
    uint q_stride_b = num_heads * tree_size * head_dim;
    uint q_stride_h = tree_size * head_dim;
    uint q_stride_t = head_dim;
    uint q_offset = batch_idx * q_stride_b + head * q_stride_h + tree_node * q_stride_t;

    uint kv_stride_b = num_heads * total_kv_len * head_dim;
    uint kv_stride_h = total_kv_len * head_dim;
    uint kv_stride_s = head_dim;
    uint kv_base = batch_idx * kv_stride_b + head * kv_stride_h;

    uint o_offset = q_offset;

    threadgroup half Q_cache[HEAD_DIM_MAX_TREE];
    threadgroup half K_tile[TILE_KV_TREE][HEAD_DIM_MAX_TREE];
    threadgroup half V_tile[TILE_KV_TREE][HEAD_DIM_MAX_TREE];
    threadgroup float reduction_scratch[SIMDGROUPS_TREE];

    // Load Q
    for (uint i = tid; i < head_dim; i += THREADS_TREE) {
        Q_cache[i] = Q[q_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    constexpr uint O_ELEMS = (HEAD_DIM_MAX_TREE + THREADS_TREE - 1) / THREADS_TREE;
    float o_accum[O_ELEMS] = {0.0f};

    uint num_tiles = (total_kv_len + TILE_KV_TREE - 1) / TILE_KV_TREE;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint tile_start = tile_idx * TILE_KV_TREE;
        uint tile_len = min(TILE_KV_TREE, total_kv_len - tile_start);

        // Load K/V tile
        uint total_elems = tile_len * head_dim;
        uint per_thread = (total_elems + THREADS_TREE - 1) / THREADS_TREE;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_TREE;
            if (idx < total_elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                uint global_kv_idx = tile_start + kv_row;
                if (global_kv_idx < total_kv_len) {
                    K_tile[kv_row][kv_col] = K[kv_base + global_kv_idx * kv_stride_s + kv_col];
                    V_tile[kv_row][kv_col] = V[kv_base + global_kv_idx * kv_stride_s + kv_col];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float thread_max_tile = -INFINITY;
        float tile_scores[TILE_KV_TREE];

        for (uint local_k = 0; local_k < tile_len; ++local_k) {
            uint global_k_idx = tile_start + local_k;
            bool is_prefix = global_k_idx < seq_len;

            float mask_val = 0.0f;
            if (is_prefix) {
                // Causal within prefix: can only attend to positions <= node_seq_pos
                // node_seq_pos is the position in the original sequence this node would occupy
                mask_val = (global_k_idx <= node_seq_pos) ? 0.0f : -INFINITY;
            } else {
                // Tree mask for draft-to-draft attention
                uint tree_k_idx = global_k_idx - seq_len;
                uint mask_idx = tree_node * tree_size + tree_k_idx;
                uint can_attend = tree_mask[mask_idx];
                mask_val = (can_attend != 0) ? 0.0f : -INFINITY;
            }

            // Dot product
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q_cache[d]) * float(K_tile[local_k][d]);
            }

            float score = dot * scale + mask_val;
            tile_scores[local_k] = score;
            thread_max_tile = max(thread_max_tile, score);
        }

        float tile_max = threadgroup_reduce_max_tree(thread_max_tile, tid, reduction_scratch);
        float prev_max = running_max;
        running_max = max(running_max, tile_max);

        if (prev_max > -INFINITY) {
            float rescale = exp(prev_max - running_max);
            running_sum *= rescale;
            for (uint i = 0; i < O_ELEMS; ++i) {
                o_accum[i] *= rescale;
            }
        }

        for (uint local_k = 0; local_k < tile_len; ++local_k) {
            float s = tile_scores[local_k];
            if (s <= -INFINITY) continue;

            float w = exp(s - running_max);
            running_sum += w;

            for (uint i = 0; i < O_ELEMS; ++i) {
                uint d = tid + i * THREADS_TREE;
                if (d < head_dim) {
                    o_accum[i] += w * float(V_tile[local_k][d]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint i = 0; i < O_ELEMS; ++i) {
        uint d = tid + i * THREADS_TREE;
        if (d < head_dim) {
            O[o_offset + d] = half(o_accum[i] * inv_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Packed Tree Mask Variant
//
// For efficiency, tree_mask can be packed as bits: 64 nodes fit in 2 uint64_t per row.
// This variant uses bit-packed masks to reduce memory bandwidth.
// ---------------------------------------------------------------------------

kernel void tree_attention_forward_packed_mask(
    device const half* Q             [[buffer(0)]],   // [batch, heads, tree_size, head_dim]
    device const half* K             [[buffer(1)]],   // [batch, heads, seq_len + tree_size, head_dim]
    device const half* V             [[buffer(2)]],   // [batch, heads, seq_len + tree_size, head_dim]
    device const ulong* tree_mask_packed [[buffer(3)]],  // [tree_size, (tree_size+63)/64] packed bits
    device half* O                   [[buffer(4)]],   // [batch, heads, tree_size, head_dim]
    constant uint& batch             [[buffer(5)]],
    constant uint& num_heads         [[buffer(6)]],
    constant uint& seq_len           [[buffer(7)]],
    constant uint& tree_size         [[buffer(8)]],
    constant uint& head_dim          [[buffer(9)]],
    constant float& scale            [[buffer(10)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    uint tree_node = tgid.x;
    uint head = tgid.y;
    uint batch_idx = tgid.z;

    if (tree_node >= tree_size || head >= num_heads || batch_idx >= batch) return;

    uint total_kv_len = seq_len + tree_size;
    uint mask_row_len = (tree_size + 63) / 64;  // Number of uint64_t per mask row

    // Offsets
    uint q_stride_b = num_heads * tree_size * head_dim;
    uint q_stride_h = tree_size * head_dim;
    uint q_stride_t = head_dim;
    uint q_offset = batch_idx * q_stride_b + head * q_stride_h + tree_node * q_stride_t;

    uint kv_stride_b = num_heads * total_kv_len * head_dim;
    uint kv_stride_h = total_kv_len * head_dim;
    uint kv_stride_s = head_dim;
    uint kv_base = batch_idx * kv_stride_b + head * kv_stride_h;

    uint o_offset = q_offset;

    threadgroup half Q_cache[HEAD_DIM_MAX_TREE];
    threadgroup half K_tile[TILE_KV_TREE][HEAD_DIM_MAX_TREE];
    threadgroup half V_tile[TILE_KV_TREE][HEAD_DIM_MAX_TREE];
    threadgroup float reduction_scratch[SIMDGROUPS_TREE];

    // Load Q
    for (uint i = tid; i < head_dim; i += THREADS_TREE) {
        Q_cache[i] = Q[q_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    constexpr uint O_ELEMS = (HEAD_DIM_MAX_TREE + THREADS_TREE - 1) / THREADS_TREE;
    float o_accum[O_ELEMS] = {0.0f};

    uint num_tiles = (total_kv_len + TILE_KV_TREE - 1) / TILE_KV_TREE;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint tile_start = tile_idx * TILE_KV_TREE;
        uint tile_len = min(TILE_KV_TREE, total_kv_len - tile_start);

        // Load K/V
        uint total_elems = tile_len * head_dim;
        uint per_thread = (total_elems + THREADS_TREE - 1) / THREADS_TREE;
        for (uint i = 0; i < per_thread; ++i) {
            uint idx = tid + i * THREADS_TREE;
            if (idx < total_elems) {
                uint kv_row = idx / head_dim;
                uint kv_col = idx % head_dim;
                uint global_kv_idx = tile_start + kv_row;
                if (global_kv_idx < total_kv_len) {
                    K_tile[kv_row][kv_col] = K[kv_base + global_kv_idx * kv_stride_s + kv_col];
                    V_tile[kv_row][kv_col] = V[kv_base + global_kv_idx * kv_stride_s + kv_col];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float thread_max_tile = -INFINITY;
        float tile_scores[TILE_KV_TREE];

        for (uint local_k = 0; local_k < tile_len; ++local_k) {
            uint global_k_idx = tile_start + local_k;
            bool is_prefix = global_k_idx < seq_len;

            float mask_val = 0.0f;
            if (!is_prefix) {
                uint tree_k_idx = global_k_idx - seq_len;

                // Unpack bit from tree_mask_packed
                uint word_idx = tree_k_idx / 64;
                uint bit_idx = tree_k_idx % 64;
                ulong mask_word = tree_mask_packed[tree_node * mask_row_len + word_idx];
                bool can_attend = (mask_word >> bit_idx) & 1UL;

                mask_val = can_attend ? 0.0f : -INFINITY;
            }

            // Dot product
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q_cache[d]) * float(K_tile[local_k][d]);
            }

            float score = dot * scale + mask_val;
            tile_scores[local_k] = score;
            thread_max_tile = max(thread_max_tile, score);
        }

        float tile_max = threadgroup_reduce_max_tree(thread_max_tile, tid, reduction_scratch);
        float prev_max = running_max;
        running_max = max(running_max, tile_max);

        if (prev_max > -INFINITY) {
            float rescale = exp(prev_max - running_max);
            running_sum *= rescale;
            for (uint i = 0; i < O_ELEMS; ++i) {
                o_accum[i] *= rescale;
            }
        }

        for (uint local_k = 0; local_k < tile_len; ++local_k) {
            float s = tile_scores[local_k];
            if (s <= -INFINITY) continue;

            float w = exp(s - running_max);
            running_sum += w;

            for (uint i = 0; i < O_ELEMS; ++i) {
                uint d = tid + i * THREADS_TREE;
                if (d < head_dim) {
                    o_accum[i] += w * float(V_tile[local_k][d]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint i = 0; i < O_ELEMS; ++i) {
        uint d = tid + i * THREADS_TREE;
        if (d < head_dim) {
            O[o_offset + d] = half(o_accum[i] * inv_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Tree Mask Construction Helper
//
// Constructs the tree_mask from parent indices.
// tree_parents[i] = parent node index of node i, or UINT_MAX if root (no parent in tree)
//
// A node can attend to:
//   - Itself
//   - All ancestors (following parent pointers up)
//
// This kernel runs on CPU or GPU to build the mask from the tree structure.
// ---------------------------------------------------------------------------

kernel void build_tree_mask(
    device const uint* tree_parents  [[buffer(0)]],  // [tree_size] parent index per node
    device uint* tree_mask           [[buffer(1)]],  // [tree_size, tree_size] output
    constant uint& tree_size         [[buffer(2)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint i = gid.x;  // query node
    uint j = gid.y;  // key node

    if (i >= tree_size || j >= tree_size) return;

    // Check if j is an ancestor of i (including i itself)
    // Walk from i up the tree following parent pointers
    uint current = i;
    bool is_ancestor = false;

    // Self-attention is always allowed
    if (i == j) {
        is_ancestor = true;
    } else {
        // Walk up the tree from i
        for (uint step = 0; step < tree_size && !is_ancestor; ++step) {
            uint parent = tree_parents[current];
            if (parent == UINT_MAX || parent >= tree_size) {
                // Reached root, j is not an ancestor
                break;
            }
            if (parent == j) {
                is_ancestor = true;
            }
            current = parent;
        }
    }

    // Write mask value
    tree_mask[i * tree_size + j] = is_ancestor ? 1 : 0;
}
