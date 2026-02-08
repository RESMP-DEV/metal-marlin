// moe_router_dispatch.hpp - Optimized CPU-side MoE router dispatch path
//
// This module provides a high-performance router dispatcher for decode/prefill
// critical path workloads:
// - Batched router forward over BF16 activations
// - SIMD-assisted top-k selection
// - Double-buffered async overlap with expert execution
// - Hot expert-pair caching for faster dispatch packing

#pragma once

#include "moe_manager.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <vector>

namespace metal_marlin {
namespace moe {

struct RouterBatchOutput {
    int32_t num_tokens = 0;
    int32_t top_k = 0;
    int32_t num_experts = 0;

    // [num_tokens, num_experts] FP32 router logits.
    std::vector<float> logits;

    // [num_tokens, top_k] selected expert IDs.
    std::vector<int32_t> topk_expert_ids;

    // [num_tokens, top_k] normalized top-k probabilities.
    std::vector<float> topk_probs;

    // Expert-grouped dispatch layout for expert GEMM.
    DispatchInfo dispatch_info;
};

struct RouterBuffer {
    uint64_t sequence_id = 0;
    RouterBatchOutput output;
};

class FastRouterDispatcher {
public:
    using ExpertLaunchFn = std::function<std::shared_future<void>(const RouterBuffer&)>;

    FastRouterDispatcher(int32_t num_experts,
                         int32_t hidden_dim,
                         int32_t top_k,
                         int32_t max_batch_tokens = 128,
                         size_t hot_pair_cache_capacity = 256,
                         uint32_t hot_pair_threshold = 8);

    ~FastRouterDispatcher();

    FastRouterDispatcher(const FastRouterDispatcher&) = delete;
    FastRouterDispatcher& operator=(const FastRouterDispatcher&) = delete;
    FastRouterDispatcher(FastRouterDispatcher&&) = delete;
    FastRouterDispatcher& operator=(FastRouterDispatcher&&) = delete;

    // Synchronous batched router forward + top-k + dispatch packing.
    RouterBatchOutput route_batch(const uint16_t* token_activations_bf16,
                                  int32_t num_tokens,
                                  const float* router_weights,
                                  const float* router_bias = nullptr);

    // Asynchronous overlap path:
    // - CPU computes router for sequence N into one of two buffers.
    // - ExpertLaunchFn is invoked when router output is ready.
    // - While experts execute N, caller can submit N+1.
    // The returned future resolves when ExpertLaunchFn's future resolves.
    std::shared_future<void> submit_async(const uint16_t* token_activations_bf16,
                                          int32_t num_tokens,
                                          const float* router_weights,
                                          const float* router_bias,
                                          ExpertLaunchFn launch_experts);

    RouterBuffer current_router_buffer() const;
    RouterBuffer previous_router_buffer() const;

    void reset_hot_pair_cache();

    int32_t num_experts() const { return num_experts_; }
    int32_t hidden_dim() const { return hidden_dim_; }
    int32_t top_k() const { return top_k_; }
    size_t hot_pair_count() const;

private:
    static constexpr uint32_t kInvalidPairKey = 0xFFFFFFFFu;
    static constexpr int32_t kPreferredFastPairFirst = 5;
    static constexpr int32_t kPreferredFastPairSecond = 12;

    struct TopKCandidate {
        float value = 0.0f;
        int32_t expert_id = -1;
    };

    uint32_t encode_pair(int32_t first_expert, int32_t second_expert) const;
    void precompute_pair_lookup();
    void reserve_batch_output_buffers(RouterBatchOutput* output) const;

    void compute_logits_batch(const uint16_t* token_activations_bf16,
                              int32_t num_tokens,
                              const float* router_weights,
                              const float* router_bias,
                              float* out_logits) const;

    void route_batch_into(const uint16_t* token_activations_bf16,
                          int32_t num_tokens,
                          const float* router_weights,
                          const float* router_bias,
                          RouterBatchOutput* out);

    void select_topk_batch(const float* logits,
                           int32_t num_tokens,
                           int32_t* out_topk_expert_ids,
                           float* out_topk_probs) const;

    void build_dispatch_with_cache(const int32_t* topk_expert_ids,
                                   int32_t num_tokens,
                                   DispatchInfo* out_info);

    void select_topk_for_token(const float* token_logits,
                               TopKCandidate* topk) const;

    void select_top2_for_token_simd(const float* token_logits,
                                    TopKCandidate* topk) const;

    void update_hot_pair_stats(int32_t first_expert, int32_t second_expert);
    bool is_hot_pair(int32_t first_expert, int32_t second_expert) const;
    bool is_preferred_fast_pair(uint32_t pair_key) const;

    int32_t num_experts_;
    int32_t hidden_dim_;
    int32_t top_k_;
    int32_t max_batch_tokens_;
    size_t hot_pair_cache_capacity_;
    uint32_t hot_pair_threshold_;

    // Precomputed pair->expert lookup for hot top-2 paths.
    std::vector<std::array<int32_t, 2>> pair_lookup_;
    uint32_t preferred_fast_pair_key_ = kInvalidPairKey;

    // Frequency tracking for expert pairs.
    std::vector<uint32_t> pair_hit_counts_;
    std::vector<uint8_t> hot_pair_mask_;
    size_t hot_pair_count_ = 0;

    // Double-buffered router outputs for async overlap.
    std::array<RouterBuffer, 2> router_buffers_;
    std::array<std::shared_future<void>, 2> async_stage_futures_;

    mutable std::mutex cache_mutex_;
    mutable std::mutex async_mutex_;
    mutable std::mutex submit_mutex_;

    uint64_t submitted_sequence_ = 0;
    int32_t current_buffer_index_ = 0;
    int32_t previous_buffer_index_ = 1;
};

} // namespace moe
} // namespace metal_marlin
