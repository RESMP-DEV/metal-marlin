/**
 * Multi-GPU P2P Transfer Kernels for GPTQ
 *
 * Metal Peer-to-Peer (P2P) transfer operations for distributed GPTQ inference.
 * Enables efficient inter-GPU communication for tensor parallelism and pipeline
 * parallelism with quantized weights.
 *
 * Architecture:
 * - Direct GPU-to-GPU memory transfers using Metal shared memory
 * - Peer-aware buffer allocation for zero-copy transfers
 * - Optimized for Apple Silicon multi-die configurations (future M-series Ultra)
 *
 * Supported Operations:
 * - P2P weight transfer (GPTQ packed format)
 * - P2P gradient transfer (FP16/BF16)
 * - P2P activation transfer with optional dequantization
 * - Broadcast/gather/scatter patterns for tensor parallelism
 *
 * Memory Layout:
 * - Packed weights: [K/8, N] uint32 (4-bit GPTQ format)
 * - Scales/zeros: [n_groups, N] half
 * - Activations: [batch, seq_len, hidden] half
 *
 * Note:
 *   Metal does not expose explicit P2P APIs like CUDA. This implementation
 *   uses shared memory regions and peer-aware buffer allocation to simulate
 *   P2P behavior. On Apple Silicon, the Unified Memory Architecture (UMA)
 *   provides efficient cross-GPU access when devices are on the same die.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Configuration
// ============================================================================

constant constexpr uint32_t MAGIC_BIAS_U32 = 0x64006400u;
constant constexpr uint32_t LO_NIBBLE_MASK = 0x000F000Fu;
constant constexpr uint32_t THREADGROUP_SIZE = 256;
constant constexpr uint32_t SIMDGROUP_SIZE = 32;
constant constexpr uint32_t WARP_SIZE = 32;

// P2P transfer types
enum class P2PTransferType : uint8_t {
    DIRECT = 0,         // Direct memory copy
    BROADCAST = 1,      // One-to-many
    SCATTER = 2,        // One-to-many with indexing
    GATHER = 3,         // Many-to-one with indexing
    ALL_GATHER = 4,     // All-to-all gather
    REDUCE_SCATTER = 5  // All-to-all reduce-scatter
};

// Reduction operations for reduce-scatter
enum class ReduceOp : uint8_t {
    SUM = 0,
    MEAN = 1,
    MAX = 2,
    MIN = 3
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Dequantize 8 unsigned INT4 values packed in a uint32_t to 8 FP16 values.
 * Used for on-the-fly dequantization during P2P transfers.
 */
inline void dequant_u4x8_gptq(uint32_t packed,
                              half scale,
                              half zero_point,
                              thread half4 &out_lo,
                              thread half4 &out_hi) {
    // Nibbles 0 and 4
    uint32_t n0_biased = (packed & LO_NIBBLE_MASK) | MAGIC_BIAS_U32;
    half2 n0_pair = as_type<half2>(n0_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // Nibbles 1 and 5
    uint32_t n1_shifted = (packed >> 4u) & LO_NIBBLE_MASK;
    uint32_t n1_biased = n1_shifted | MAGIC_BIAS_U32;
    half2 n1_pair = as_type<half2>(n1_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // Nibbles 2 and 6
    uint32_t n2_shifted = (packed >> 8u) & LO_NIBBLE_MASK;
    uint32_t n2_biased = n2_shifted | MAGIC_BIAS_U32;
    half2 n2_pair = as_type<half2>(n2_biased) - as_type<half2>(MAGIC_BIAS_U32);

    // Nibbles 3 and 7
    uint32_t n3_shifted = (packed >> 12u) & LO_NIBBLE_MASK;
    uint32_t n3_biased = n3_shifted | MAGIC_BIAS_U32;
    half2 n3_pair = as_type<half2>(n3_biased) - as_type<half2>(MAGIC_BIAS_U32);

    out_lo = half4(n0_pair.x, n1_pair.x, n2_pair.x, n3_pair.x);
    out_hi = half4(n0_pair.y, n1_pair.y, n2_pair.y, n3_pair.y);

    // GPTQ dequant: (q - z) * s
    out_lo = (out_lo - zero_point) * scale;
    out_hi = (out_hi - zero_point) * scale;
}

/**
 * Compute peer buffer offset for multi-GPU layout.
 * Assumes peer buffers are laid out consecutively in a shared heap.
 */
inline uint64_t compute_peer_offset(uint32_t peer_id,
                                    uint32_t num_peers,
                                    uint64_t buffer_size_per_peer) {
    return peer_id * buffer_size_per_peer;
}

// ============================================================================
// P2P Weight Transfer (Packed GPTQ Format)
// ============================================================================

/**
 * Direct P2P transfer of packed GPTQ weights between GPUs.
 *
 * Transfers quantized weights from source GPU to destination GPU without
 * dequantization. Useful for distributing weight shards in tensor parallelism.
 *
 * Arguments:
 *   src_weights: Source packed weights [K/8, N] uint32
 *   dst_weights: Destination packed weights [K/8, N] uint32
 *   num_elements: Total number of uint32 elements to transfer
 *   src_offset: Offset in source buffer (in uint32 units)
 *   dst_offset: Offset in destination buffer (in uint32 units)
 */
kernel void p2p_transfer_packed_weights(
    device const uint32_t* src_weights [[buffer(0)]],
    device uint32_t* dst_weights [[buffer(1)]],
    constant uint32_t& num_elements [[buffer(2)]],
    constant uint32_t& src_offset [[buffer(3)]],
    constant uint32_t& dst_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_elements) return;

    // Direct memory copy
    dst_weights[dst_offset + gid] = src_weights[src_offset + gid];
}

/**
 * P2P transfer with on-the-fly dequantization.
 *
 * Transfers packed GPTQ weights and dequantizes them during the transfer.
 * Reduces memory bandwidth at the cost of compute.
 *
 * Arguments:
 *   src_weights: Source packed weights [K/8, N] uint32
 *   src_scales: Source scales [n_groups, N] half
 *   src_zeros: Source zeros [n_groups, N] half
 *   dst_weights: Destination dequantized weights [K, N] half
 *   K: Number of input features
 *   N: Number of output features
 *   group_size: GPTQ group size (e.g., 128)
 *   src_offset: Offset in source buffer
 *   dst_offset: Offset in destination buffer
 */
kernel void p2p_transfer_dequant_weights(
    device const uint32_t* src_weights [[buffer(0)]],
    device const half* src_scales [[buffer(1)]],
    device const half* src_zeros [[buffer(2)]],
    device half* dst_weights [[buffer(3)]],
    constant uint32_t& K [[buffer(4)]],
    constant uint32_t& N [[buffer(5)]],
    constant uint32_t& group_size [[buffer(6)]],
    constant uint32_t& src_offset [[buffer(7)]],
    constant uint32_t& dst_offset [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint n_idx = gid.x;
    uint k_block = gid.y; // Block index along K (8 weights per block)

    uint k_start = k_block * 8u;
    if (n_idx >= N || k_start >= K) return;

    // Load packed weights from source
    uint packed_idx = src_offset + k_block * N + n_idx;
    uint32_t packed = src_weights[packed_idx];

    // Load scale and zero for this group
    uint group_idx = k_start / group_size;
    uint param_idx = group_idx * N + n_idx;
    
    half scale = src_scales[param_idx];
    half zero = src_zeros[param_idx];

    // Dequantize
    half4 lo, hi;
    dequant_u4x8_gptq(packed, scale, zero, lo, hi);

    // Write to destination
    uint out_base = dst_offset + k_start * N + n_idx;
    uint k_remain = min(8u, K - k_start);
    
    if (k_remain == 8u) {
        dst_weights[out_base + 0u * N] = lo.x;
        dst_weights[out_base + 1u * N] = lo.y;
        dst_weights[out_base + 2u * N] = lo.z;
        dst_weights[out_base + 3u * N] = lo.w;
        dst_weights[out_base + 4u * N] = hi.x;
        dst_weights[out_base + 5u * N] = hi.y;
        dst_weights[out_base + 6u * N] = hi.z;
        dst_weights[out_base + 7u * N] = hi.w;
    } else {
        half vals[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        for (uint i = 0; i < k_remain; i++) {
            dst_weights[out_base + i * N] = vals[i];
        }
    }
}

// ============================================================================
// P2P Activation Transfer
// ============================================================================

/**
 * Direct P2P transfer of FP16 activations between GPUs.
 *
 * Transfers activation tensors during forward/backward passes in tensor
 * parallelism or pipeline parallelism.
 *
 * Arguments:
 *   src_activations: Source activations [batch, seq_len, hidden] half
 *   dst_activations: Destination activations [batch, seq_len, hidden] half
 *   num_elements: Total number of half elements to transfer
 *   src_offset: Offset in source buffer (in half units)
 *   dst_offset: Offset in destination buffer (in half units)
 */
kernel void p2p_transfer_activations(
    device const half* src_activations [[buffer(0)]],
    device half* dst_activations [[buffer(1)]],
    constant uint32_t& num_elements [[buffer(2)]],
    constant uint32_t& src_offset [[buffer(3)]],
    constant uint32_t& dst_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_elements) return;

    // Vectorized transfer (4 elements per thread)
    uint vec_idx = gid * 4u;
    if (vec_idx + 3u < num_elements) {
        half4 data = *((device const half4*)(src_activations + src_offset + vec_idx));
        *((device half4*)(dst_activations + dst_offset + vec_idx)) = data;
    } else {
        // Handle remainder
        for (uint i = 0; i < 4u && (vec_idx + i) < num_elements; i++) {
            dst_activations[dst_offset + vec_idx + i] = src_activations[src_offset + vec_idx + i];
        }
    }
}

/**
 * Batched P2P transfer of activations with strided layout.
 *
 * Transfers multiple activation tensors with strided memory layout.
 * Used for pipeline parallelism with micro-batching.
 *
 * Arguments:
 *   src_activations: Source activations buffer
 *   dst_activations: Destination activations buffer
 *   batch_size: Number of batches
 *   seq_len: Sequence length
 *   hidden_dim: Hidden dimension
 *   src_stride: Stride between batches in source
 *   dst_stride: Stride between batches in destination
 */
kernel void p2p_transfer_batched_activations(
    device const half* src_activations [[buffer(0)]],
    device half* dst_activations [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    constant uint32_t& seq_len [[buffer(3)]],
    constant uint32_t& hidden_dim [[buffer(4)]],
    constant uint32_t& src_stride [[buffer(5)]],
    constant uint32_t& dst_stride [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch_idx = gid.z;
    uint seq_idx = gid.y;
    uint hidden_idx = gid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_dim) return;

    uint src_idx = batch_idx * src_stride + seq_idx * hidden_dim + hidden_idx;
    uint dst_idx = batch_idx * dst_stride + seq_idx * hidden_dim + hidden_idx;

    dst_activations[dst_idx] = src_activations[src_idx];
}

// ============================================================================
// Collective Operations (Broadcast, Scatter, Gather)
// ============================================================================

/**
 * P2P broadcast from source GPU to multiple destination GPUs.
 *
 * Broadcasts a tensor from source GPU (rank 0) to all other GPUs.
 * Each thread handles a contiguous chunk of data.
 *
 * Arguments:
 *   src_buffer: Source buffer on rank 0
 *   dst_buffers: Array of destination buffers (one per peer)
 *   num_elements: Number of elements to broadcast
 *   num_peers: Total number of GPUs
 *   peer_id: Current GPU ID
 */
kernel void p2p_broadcast_weights(
    device const half* src_buffer [[buffer(0)]],
    device half* dst_buffers [[buffer(1)]],
    constant uint32_t& num_elements [[buffer(2)]],
    constant uint32_t& num_peers [[buffer(3)]],
    constant uint32_t& peer_id [[buffer(4)]],
    constant uint64_t& buffer_size_per_peer [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_elements) return;

    half value = src_buffer[gid];

    // Write to all peer buffers (excluding self)
    for (uint32_t p = 0; p < num_peers; p++) {
        if (p != peer_id) {
            uint64_t peer_offset = compute_peer_offset(p, num_peers, buffer_size_per_peer);
            dst_buffers[peer_offset + gid] = value;
        }
    }
}

/**
 * P2P scatter: distribute shards from source to multiple GPUs.
 *
 * Scatters a tensor into equal shards and distributes to all GPUs.
 * Used for splitting tensors in tensor parallelism.
 *
 * Arguments:
 *   src_buffer: Source buffer (full tensor)
 *   dst_buffers: Array of destination buffers (one per peer)
 *   total_elements: Total elements in source
 *   shard_size: Elements per shard (total_elements / num_peers)
 *   num_peers: Number of GPUs
 *   peer_id: Current GPU ID
 */
kernel void p2p_scatter_tensor(
    device const half* src_buffer [[buffer(0)]],
    device half* dst_buffers [[buffer(1)]],
    constant uint32_t& total_elements [[buffer(2)]],
    constant uint32_t& shard_size [[buffer(3)]],
    constant uint32_t& num_peers [[buffer(4)]],
    constant uint32_t& peer_id [[buffer(5)]],
    constant uint64_t& buffer_size_per_peer [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_elements) return;

    // Determine which peer this element belongs to
    uint32_t target_peer = gid / shard_size;
    uint32_t local_idx = gid % shard_size;

    if (target_peer < num_peers) {
        uint64_t peer_offset = compute_peer_offset(target_peer, num_peers, buffer_size_per_peer);
        dst_buffers[peer_offset + local_idx] = src_buffer[gid];
    }
}

/**
 * P2P gather: collect shards from multiple GPUs to source.
 *
 * Gathers tensor shards from all GPUs into a single output tensor.
 * Inverse operation of scatter.
 *
 * Arguments:
 *   src_buffers: Array of source buffers (one per peer)
 *   dst_buffer: Destination buffer (full tensor)
 *   shard_size: Elements per shard
 *   num_peers: Number of GPUs
 *   peer_id: Current GPU ID
 */
kernel void p2p_gather_tensor(
    device const half* src_buffers [[buffer(0)]],
    device half* dst_buffer [[buffer(1)]],
    constant uint32_t& shard_size [[buffer(2)]],
    constant uint32_t& num_peers [[buffer(3)]],
    constant uint32_t& peer_id [[buffer(4)]],
    constant uint64_t& buffer_size_per_peer [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint32_t total_elements = shard_size * num_peers;
    if (gid >= total_elements) return;

    // Determine which peer this element comes from
    uint32_t source_peer = gid / shard_size;
    uint32_t local_idx = gid % shard_size;

    uint64_t peer_offset = compute_peer_offset(source_peer, num_peers, buffer_size_per_peer);
    dst_buffer[gid] = src_buffers[peer_offset + local_idx];
}

/**
 * P2P all-gather: each GPU gathers full tensor from all shards.
 *
 * All GPUs perform a gather operation simultaneously. Each GPU receives
 * the full concatenated tensor from all peers.
 *
 * Arguments:
 *   local_shard: This GPU's local shard
 *   peer_shards: Array of shards from all peers
 *   dst_buffer: Destination for full gathered tensor
 *   shard_size: Elements per shard
 *   num_peers: Number of GPUs
 *   peer_id: Current GPU ID
 */
kernel void p2p_all_gather_tensor(
    device const half* local_shard [[buffer(0)]],
    device const half* peer_shards [[buffer(1)]],
    device half* dst_buffer [[buffer(2)]],
    constant uint32_t& shard_size [[buffer(3)]],
    constant uint32_t& num_peers [[buffer(4)]],
    constant uint32_t& peer_id [[buffer(5)]],
    constant uint64_t& buffer_size_per_peer [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint32_t total_elements = shard_size * num_peers;
    if (gid >= total_elements) return;

    uint32_t source_peer = gid / shard_size;
    uint32_t local_idx = gid % shard_size;

    if (source_peer == peer_id) {
        // Copy from local shard
        dst_buffer[gid] = local_shard[local_idx];
    } else {
        // Copy from peer shard
        uint64_t peer_offset = compute_peer_offset(source_peer, num_peers, buffer_size_per_peer);
        dst_buffer[gid] = peer_shards[peer_offset + local_idx];
    }
}

// ============================================================================
// Reduce-Scatter Operations
// ============================================================================

/**
 * P2P reduce-scatter: reduce across all GPUs and scatter results.
 *
 * Reduces tensors from all GPUs using the specified reduction operation,
 * then scatters the reduced result back to all GPUs.
 *
 * Used for gradient synchronization in data parallelism.
 *
 * Arguments:
 *   local_tensor: This GPU's local tensor
 *   peer_tensors: Array of tensors from all peers
 *   dst_tensor: Destination for reduced shard
 *   total_elements: Total elements in full tensor
 *   shard_size: Elements per shard after reduction
 *   num_peers: Number of GPUs
 *   peer_id: Current GPU ID
 *   op: Reduction operation (0=sum, 1=mean, 2=max, 3=min)
 */
kernel void p2p_reduce_scatter_tensor(
    device const half* local_tensor [[buffer(0)]],
    device const half* peer_tensors [[buffer(1)]],
    device half* dst_tensor [[buffer(2)]],
    constant uint32_t& total_elements [[buffer(3)]],
    constant uint32_t& shard_size [[buffer(4)]],
    constant uint32_t& num_peers [[buffer(5)]],
    constant uint32_t& peer_id [[buffer(6)]],
    constant uint8_t& op [[buffer(7)]],
    constant uint64_t& buffer_size_per_peer [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= shard_size) return;

    // Compute global index for this thread's element
    uint32_t global_idx = peer_id * shard_size + gid;
    if (global_idx >= total_elements) return;

    ReduceOp reduce_op = static_cast<ReduceOp>(op);
    float acc;

    // Initialize accumulator
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc = 0.0f;
            break;
        case ReduceOp::MAX:
            acc = -INFINITY;
            break;
        case ReduceOp::MIN:
            acc = INFINITY;
            break;
    }

    // Reduce from local tensor
    float val = float(local_tensor[global_idx]);
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc += val;
            break;
        case ReduceOp::MAX:
            acc = max(acc, val);
            break;
        case ReduceOp::MIN:
            acc = min(acc, val);
            break;
    }

    // Reduce from all peer tensors
    for (uint32_t p = 0; p < num_peers; p++) {
        if (p != peer_id) {
            uint64_t peer_offset = compute_peer_offset(p, num_peers, buffer_size_per_peer);
            float peer_val = float(peer_tensors[peer_offset + global_idx]);

            switch (reduce_op) {
                case ReduceOp::SUM:
                case ReduceOp::MEAN:
                    acc += peer_val;
                    break;
                case ReduceOp::MAX:
                    acc = max(acc, peer_val);
                    break;
                case ReduceOp::MIN:
                    acc = min(acc, peer_val);
                    break;
            }
        }
    }

    // Finalize
    if (reduce_op == ReduceOp::MEAN) {
        acc /= float(num_peers);
    }

    dst_tensor[gid] = half(acc);
}

/**
 * P2P all-reduce: reduce across all GPUs and replicate result.
 *
 * Each GPU reduces contributions from all peers and stores the full
 * reduced tensor locally. Combination of reduce-scatter + all-gather.
 *
 * Arguments:
 *   local_tensor: This GPU's local tensor
 *   peer_tensors: Array of tensors from all peers
 *   dst_tensor: Destination for reduced tensor
 *   num_elements: Number of elements
 *   num_peers: Number of GPUs
 *   peer_id: Current GPU ID
 *   op: Reduction operation
 */
kernel void p2p_all_reduce_tensor(
    device const half* local_tensor [[buffer(0)]],
    device const half* peer_tensors [[buffer(1)]],
    device half* dst_tensor [[buffer(2)]],
    constant uint32_t& num_elements [[buffer(3)]],
    constant uint32_t& num_peers [[buffer(4)]],
    constant uint32_t& peer_id [[buffer(5)]],
    constant uint8_t& op [[buffer(6)]],
    constant uint64_t& buffer_size_per_peer [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_elements) return;

    ReduceOp reduce_op = static_cast<ReduceOp>(op);
    float acc;

    // Initialize accumulator
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc = 0.0f;
            break;
        case ReduceOp::MAX:
            acc = -INFINITY;
            break;
        case ReduceOp::MIN:
            acc = INFINITY;
            break;
    }

    // Reduce from local tensor
    float val = float(local_tensor[gid]);
    switch (reduce_op) {
        case ReduceOp::SUM:
        case ReduceOp::MEAN:
            acc += val;
            break;
        case ReduceOp::MAX:
            acc = max(acc, val);
            break;
        case ReduceOp::MIN:
            acc = min(acc, val);
            break;
    }

    // Reduce from all peer tensors
    for (uint32_t p = 0; p < num_peers; p++) {
        if (p != peer_id) {
            uint64_t peer_offset = compute_peer_offset(p, num_peers, buffer_size_per_peer);
            float peer_val = float(peer_tensors[peer_offset + gid]);

            switch (reduce_op) {
                case ReduceOp::SUM:
                case ReduceOp::MEAN:
                    acc += peer_val;
                    break;
                case ReduceOp::MAX:
                    acc = max(acc, peer_val);
                    break;
                case ReduceOp::MIN:
                    acc = min(acc, peer_val);
                    break;
            }
        }
    }

    // Finalize
    if (reduce_op == ReduceOp::MEAN) {
        acc /= float(num_peers);
    }

    dst_tensor[gid] = half(acc);
}

// ============================================================================
// Optimized P2P Operations with Threadgroup Synchronization
// ============================================================================

/**
 * Threadgroup-optimized P2P reduce-scatter for small tensors.
 *
 * Uses threadgroup memory for staging and threadgroup barriers for
 * synchronization. Reduces latency for small transfers.
 *
 * Best for: Shard sizes < 64KB
 */
kernel void p2p_reduce_scatter_tg_optimized(
    device const half* local_tensor [[buffer(0)]],
    device const half* peer_tensors [[buffer(1)]],
    device half* dst_tensor [[buffer(2)]],
    constant uint32_t& total_elements [[buffer(3)]],
    constant uint32_t& shard_size [[buffer(4)]],
    constant uint32_t& num_peers [[buffer(5)]],
    constant uint32_t& peer_id [[buffer(6)]],
    constant uint8_t& op [[buffer(7)]],
    constant uint64_t& buffer_size_per_peer [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    threadgroup half* shared_mem [[threadgroup(0)]])
{
    if (gid >= shard_size) return;

    uint32_t global_idx = peer_id * shard_size + gid;
    if (global_idx >= total_elements) return;

    // Load local value into shared memory
    shared_mem[tid] = local_tensor[global_idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform reduction
    ReduceOp reduce_op = static_cast<ReduceOp>(op);
    float acc = float(shared_mem[tid]);

    // Reduce across peers
    for (uint32_t p = 0; p < num_peers; p++) {
        if (p != peer_id) {
            uint64_t peer_offset = compute_peer_offset(p, num_peers, buffer_size_per_peer);
            float peer_val = float(peer_tensors[peer_offset + global_idx]);

            switch (reduce_op) {
                case ReduceOp::SUM:
                case ReduceOp::MEAN:
                    acc += peer_val;
                    break;
                case ReduceOp::MAX:
                    acc = max(acc, peer_val);
                    break;
                case ReduceOp::MIN:
                    acc = min(acc, peer_val);
                    break;
            }
        }
    }

    if (reduce_op == ReduceOp::MEAN) {
        acc /= float(num_peers);
    }

    dst_tensor[gid] = half(acc);
}

/**
 * Simdgroup-optimized P2P broadcast for small tensors.
 *
 * Uses simdgroup shuffle operations for efficient intra-warp communication.
 * Reduces latency for broadcasts within a single simdgroup.
 *
 * Best for: Tensor sizes <= 32 elements (one simdgroup)
 */
kernel void p2p_broadcast_simd_optimized(
    device const half* src_buffer [[buffer(0)]],
    device half* dst_buffers [[buffer(1)]],
    constant uint32_t& num_elements [[buffer(2)]],
    constant uint32_t& num_peers [[buffer(3)]],
    constant uint32_t& peer_id [[buffer(4)]],
    constant uint64_t& buffer_size_per_peer [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (gid >= num_elements) return;

    // Load value from lane 0 (broadcast source)
    float value = (simd_lane == 0) ? float(src_buffer[gid]) : 0.0f;
    value = simd_broadcast(value, 0);

    // Write to all peer buffers
    for (uint32_t p = 0; p < num_peers; p++) {
        if (p != peer_id) {
            uint64_t peer_offset = compute_peer_offset(p, num_peers, buffer_size_per_peer);
            dst_buffers[peer_offset + gid] = half(value);
        }
    }
}

// ============================================================================
// End of distributed_gptq.metal
// ============================================================================
