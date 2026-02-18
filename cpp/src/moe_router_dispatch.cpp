// moe_router_dispatch.cpp - Optimized CPU router dispatch for MoE decode path

#include "moe_router_dispatch.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(__ARM_NEON)
#include <arm_neon.h>
// Additional ARM headers for optimization hints
#if defined(__APPLE__)
#include <os/lock.h>
#endif
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

// Compiler optimization hints
#if defined(__GNUC__) || defined(__clang__)
#define MM_LIKELY(x) __builtin_expect(!!(x), 1)
#define MM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define MM_INLINE __attribute__((always_inline)) inline
#define MM_HOT __attribute__((hot))
#define MM_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#else
#define MM_LIKELY(x) (x)
#define MM_UNLIKELY(x) (x)
#define MM_INLINE inline
#define MM_HOT
#define MM_PREFETCH(addr, rw, locality)
#endif

namespace {

MM_INLINE MM_HOT float max4_simd(const float* values) {
#if defined(__ARM_NEON)
#if defined(__aarch64__)
    const float32x4_t v = vld1q_f32(values);
    return vmaxvq_f32(v);
#else
    const float32x4_t v = vld1q_f32(values);
    float32x2_t m = vmax_f32(vget_low_f32(v), vget_high_f32(v));
    m = vpmax_f32(m, m);
    return vget_lane_f32(m, 0);
#endif
#elif defined(__SSE2__)
    const __m128 v = _mm_loadu_ps(values);
    __m128 m = _mm_max_ps(v, _mm_movehl_ps(v, v));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    return _mm_cvtss_f32(m);
#else
    return std::max(std::max(values[0], values[1]), std::max(values[2], values[3]));
#endif
}

MM_INLINE MM_HOT void bf16_row_to_float(const uint16_t* bf16_row,
                              float* __restrict fp32_row,
                              int32_t hidden_dim) {
#if defined(__ARM_NEON)
    int32_t d = 0;
    // Main loop: 16 elements at a time with prefetching
    for (; d + 16 <= hidden_dim; d += 16) {
        MM_PREFETCH(bf16_row + d + 64, 0, 3);
        
        const uint16x8_t p0 = vld1q_u16(bf16_row + d);
        const uint16x8_t p1 = vld1q_u16(bf16_row + d + 8);
        
        const uint32x4_t l0 = vshll_n_u16(vget_low_u16(p0), 16);
        const uint32x4_t h0 = vshll_n_u16(vget_high_u16(p0), 16);
        const uint32x4_t l1 = vshll_n_u16(vget_low_u16(p1), 16);
        const uint32x4_t h1 = vshll_n_u16(vget_high_u16(p1), 16);
        
        vst1q_f32(fp32_row + d, vreinterpretq_f32_u32(l0));
        vst1q_f32(fp32_row + d + 4, vreinterpretq_f32_u32(h0));
        vst1q_f32(fp32_row + d + 8, vreinterpretq_f32_u32(l1));
        vst1q_f32(fp32_row + d + 12, vreinterpretq_f32_u32(h1));
    }
    // Secondary loop: 8 elements
    for (; d + 8 <= hidden_dim; d += 8) {
        const uint16x8_t packed = vld1q_u16(bf16_row + d);
        const uint32x4_t low = vshll_n_u16(vget_low_u16(packed), 16);
        const uint32x4_t high = vshll_n_u16(vget_high_u16(packed), 16);
        vst1q_f32(fp32_row + d, vreinterpretq_f32_u32(low));
        vst1q_f32(fp32_row + d + 4, vreinterpretq_f32_u32(high));
    }
    // Cleanup loop
    for (; d < hidden_dim; ++d) {
        uint32_t val = static_cast<uint32_t>(bf16_row[d]) << 16;
        std::memcpy(&fp32_row[d], &val, sizeof(float));
    }
#else
    for (int32_t d = 0; d < hidden_dim; ++d) {
        uint32_t val = static_cast<uint32_t>(bf16_row[d]) << 16;
        std::memcpy(&fp32_row[d], &val, sizeof(float));
    }
#endif
}

MM_INLINE MM_HOT float dot_product_simd(const float* __restrict lhs,
                              const float* __restrict rhs,
                              int32_t hidden_dim) {
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int32_t d = 0;
    
    // Unrolled 16-element loop with prefetching
    for (; d + 16 <= hidden_dim; d += 16) {
        MM_PREFETCH(lhs + d + 64, 0, 3);
        MM_PREFETCH(rhs + d + 64, 0, 3);
        
        acc0 = vfmaq_f32(acc0, vld1q_f32(lhs + d), vld1q_f32(rhs + d));
        acc1 = vfmaq_f32(acc1, vld1q_f32(lhs + d + 4), vld1q_f32(rhs + d + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(lhs + d + 8), vld1q_f32(rhs + d + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(lhs + d + 12), vld1q_f32(rhs + d + 12));
    }
    // 8-element loop
    for (; d + 8 <= hidden_dim; d += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(lhs + d), vld1q_f32(rhs + d));
        acc1 = vfmaq_f32(acc1, vld1q_f32(lhs + d + 4), vld1q_f32(rhs + d + 4));
    }
    // 4-element loop
    for (; d + 4 <= hidden_dim; d += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(lhs + d), vld1q_f32(rhs + d));
    }
    
    // Horizontal reduction
    float32x4_t acc01 = vaddq_f32(acc0, acc1);
    float32x4_t acc23 = vaddq_f32(acc2, acc3);
    float32x4_t acc = vaddq_f32(acc01, acc23);
    float out = vaddvq_f32(acc);
    
    // Cleanup
    for (; d < hidden_dim; ++d) {
        out += lhs[d] * rhs[d];
    }
    return out;
#elif defined(__SSE2__)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    int32_t d = 0;
    
    for (; d + 16 <= hidden_dim; d += 16) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(lhs + d), _mm_loadu_ps(rhs + d)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(lhs + d + 4), _mm_loadu_ps(rhs + d + 4)));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(lhs + d + 8), _mm_loadu_ps(rhs + d + 8)));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(lhs + d + 12), _mm_loadu_ps(rhs + d + 12)));
    }
    for (; d + 8 <= hidden_dim; d += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(lhs + d), _mm_loadu_ps(rhs + d)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(lhs + d + 4), _mm_loadu_ps(rhs + d + 4)));
    }
    for (; d + 4 <= hidden_dim; d += 4) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(lhs + d), _mm_loadu_ps(rhs + d)));
    }
    
    __m128 acc01 = _mm_add_ps(acc0, acc1);
    __m128 acc23 = _mm_add_ps(acc2, acc3);
    __m128 acc = _mm_add_ps(acc01, acc23);
    __m128 sum = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    float out = _mm_cvtss_f32(sum);
    for (; d < hidden_dim; ++d) {
        out += lhs[d] * rhs[d];
    }
    return out;
#else
    float out = 0.0f;
    for (int32_t d = 0; d < hidden_dim; ++d) {
        out += lhs[d] * rhs[d];
    }
    return out;
#endif
}

} // namespace

namespace metal_marlin {
namespace moe {

FastRouterDispatcher::FastRouterDispatcher(int32_t num_experts,
                                           int32_t hidden_dim,
                                           int32_t top_k,
                                           int32_t max_batch_tokens,
                                           size_t hot_pair_cache_capacity,
                                           uint32_t hot_pair_threshold)
    : num_experts_(num_experts),
      hidden_dim_(hidden_dim),
      top_k_(top_k),
      max_batch_tokens_(max_batch_tokens > 0 ? max_batch_tokens : 128),
      hot_pair_cache_capacity_(hot_pair_cache_capacity),
      hot_pair_threshold_(hot_pair_threshold > 0 ? hot_pair_threshold : 8) {
    if (num_experts_ <= 0) {
        throw std::invalid_argument("FastRouterDispatcher: num_experts must be > 0");
    }
    if (hidden_dim_ <= 0) {
        throw std::invalid_argument("FastRouterDispatcher: hidden_dim must be > 0");
    }
    if (top_k_ <= 0 || top_k_ > num_experts_) {
        throw std::invalid_argument("FastRouterDispatcher: top_k must be in [1, num_experts]");
    }

    precompute_pair_lookup();
    reserve_batch_output_buffers(&router_buffers_[0].output);
    reserve_batch_output_buffers(&router_buffers_[1].output);
}

FastRouterDispatcher::~FastRouterDispatcher() {
    std::shared_future<void> pending0;
    std::shared_future<void> pending1;
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        if (async_stage_futures_[0].valid()) {
            pending0 = async_stage_futures_[0];
        }
        if (async_stage_futures_[1].valid()) {
            pending1 = async_stage_futures_[1];
        }
    }
    if (pending0.valid()) {
        pending0.wait();
    }
    if (pending1.valid()) {
        pending1.wait();
    }
}

uint32_t FastRouterDispatcher::encode_pair(int32_t first_expert,
                                           int32_t second_expert) const {
    if (first_expert < 0 || second_expert < 0 ||
        first_expert >= num_experts_ || second_expert >= num_experts_) {
        return kInvalidPairKey;
    }
    return static_cast<uint32_t>(first_expert * num_experts_ + second_expert);
}

void FastRouterDispatcher::precompute_pair_lookup() {
    const size_t pair_count =
        static_cast<size_t>(num_experts_) * static_cast<size_t>(num_experts_);

    pair_lookup_.resize(pair_count);
    pair_hit_counts_.assign(pair_count, 0U);
    hot_pair_mask_.assign(pair_count, 0U);
    hot_pair_count_ = 0;

    for (int32_t first = 0; first < num_experts_; ++first) {
        for (int32_t second = 0; second < num_experts_; ++second) {
            const uint32_t key = encode_pair(first, second);
            pair_lookup_[key] = {first, second};
        }
    }

    preferred_fast_pair_key_ =
        encode_pair(kPreferredFastPairFirst, kPreferredFastPairSecond);
    if (preferred_fast_pair_key_ != kInvalidPairKey &&
        hot_pair_cache_capacity_ > 0) {
        hot_pair_mask_[preferred_fast_pair_key_] = 1U;
        hot_pair_count_ = 1;
        const uint32_t reverse_pair =
            encode_pair(kPreferredFastPairSecond, kPreferredFastPairFirst);
        if (reverse_pair != kInvalidPairKey &&
            reverse_pair != preferred_fast_pair_key_ &&
            hot_pair_count_ < hot_pair_cache_capacity_) {
            hot_pair_mask_[reverse_pair] = 1U;
            ++hot_pair_count_;
        }
    }
}

bool FastRouterDispatcher::is_preferred_fast_pair(uint32_t pair_key) const {
    if (pair_key == kInvalidPairKey || preferred_fast_pair_key_ == kInvalidPairKey) {
        return false;
    }
    if (pair_key == preferred_fast_pair_key_) {
        return true;
    }
    const uint32_t reverse_pair =
        encode_pair(kPreferredFastPairSecond, kPreferredFastPairFirst);
    return reverse_pair != kInvalidPairKey && pair_key == reverse_pair;
}

void FastRouterDispatcher::reserve_batch_output_buffers(RouterBatchOutput* output) const {
    if (!output) {
        return;
    }

    const size_t max_tokens = static_cast<size_t>(max_batch_tokens_);
    output->logits.reserve(max_tokens * static_cast<size_t>(num_experts_));
    output->topk_expert_ids.reserve(max_tokens * static_cast<size_t>(top_k_));
    output->topk_probs.reserve(max_tokens * static_cast<size_t>(top_k_));
    output->dispatch_info.sorted_token_indices.reserve(max_tokens * static_cast<size_t>(top_k_));
    output->dispatch_info.sorted_expert_indices.reserve(max_tokens * static_cast<size_t>(top_k_));
    output->dispatch_info.inverse_indices.reserve(max_tokens * static_cast<size_t>(top_k_));
    output->dispatch_info.expert_offsets.resize(static_cast<size_t>(num_experts_) + 1U, 0);
}

void FastRouterDispatcher::compute_logits_batch(const uint16_t* __restrict token_activations_bf16,
                                                int32_t num_tokens,
                                                const float* __restrict router_weights,
                                                const float* __restrict router_bias,
                                                float* __restrict out_logits) const {
    // Decode path optimization: batch tokens to reuse expert weight rows in cache
    // Increased block size for better cache utilization on modern CPUs
    constexpr int32_t kTokenBlock = 8;

    thread_local std::vector<float> token_block_fp32;
    const size_t required =
        static_cast<size_t>(kTokenBlock) * static_cast<size_t>(hidden_dim_);
    if (MM_UNLIKELY(token_block_fp32.size() < required)) {
        token_block_fp32.resize(required);
    }

    alignas(64) std::array<const float*, kTokenBlock> token_rows{};
    alignas(64) std::array<float*, kTokenBlock> logits_rows{};

    for (int32_t token_base = 0; token_base < num_tokens; token_base += kTokenBlock) {
        const int32_t block_tokens = std::min(kTokenBlock, num_tokens - token_base);

        // Phase 1: Convert BF16 to FP32 for all tokens in block
        for (int32_t local_token = 0; local_token < block_tokens; ++local_token) {
            const int32_t token = token_base + local_token;
            const uint16_t* token_row =
                token_activations_bf16 + static_cast<size_t>(token) * hidden_dim_;
            float* fp32_row =
                token_block_fp32.data() + static_cast<size_t>(local_token) * hidden_dim_;

            bf16_row_to_float(token_row, fp32_row, hidden_dim_);
            token_rows[local_token] = fp32_row;
            logits_rows[local_token] =
                out_logits + static_cast<size_t>(token) * num_experts_;
        }

        // Phase 2: Compute dot products with weight rows
        // Process experts in chunks to improve weight cache locality
        for (int32_t expert = 0; expert < num_experts_; ++expert) {
            const float* weight_row =
                router_weights + static_cast<size_t>(expert) * hidden_dim_;
            const float bias = router_bias ? router_bias[expert] : 0.0f;

            // Prefetch next expert's weights
            if (MM_LIKELY(expert + 1 < num_experts_)) {
                MM_PREFETCH(router_weights + static_cast<size_t>(expert + 1) * hidden_dim_, 0, 1);
            }

            // Unrolled loop for common small block sizes
            int32_t local_token = 0;
            
            // Handle 4 tokens at a time (common for top-k=4)
            for (; local_token + 4 <= block_tokens; local_token += 4) {
                logits_rows[local_token][expert] =
                    bias + dot_product_simd(token_rows[local_token], weight_row, hidden_dim_);
                logits_rows[local_token + 1][expert] =
                    bias + dot_product_simd(token_rows[local_token + 1], weight_row, hidden_dim_);
                logits_rows[local_token + 2][expert] =
                    bias + dot_product_simd(token_rows[local_token + 2], weight_row, hidden_dim_);
                logits_rows[local_token + 3][expert] =
                    bias + dot_product_simd(token_rows[local_token + 3], weight_row, hidden_dim_);
            }
            // Handle remaining tokens
            for (; local_token < block_tokens; ++local_token) {
                logits_rows[local_token][expert] =
                    bias + dot_product_simd(token_rows[local_token], weight_row, hidden_dim_);
            }
        }
    }
}

void FastRouterDispatcher::select_top2_for_token_simd(const float* __restrict token_logits,
                                                      TopKCandidate* topk) const {
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Use vectorized comparison where possible
#if defined(__ARM_NEON) && defined(__aarch64__)
    // Initialize with first two elements
    float best0 = token_logits[0];
    float best1 = token_logits[1];
    int32_t idx0 = 0;
    int32_t idx1 = 1;
    
    if (best1 > best0) {
        std::swap(best0, best1);
        std::swap(idx0, idx1);
    }

    int32_t expert = 2;
    
    // Process 4 experts at a time using SIMD
    for (; expert + 4 <= num_experts_; expert += 4) {
        float32x4_t vals = vld1q_f32(token_logits + expert);
        
        // Extract and compare each value
        float v[4];
        vst1q_f32(v, vals);
        
        for (int i = 0; i < 4; ++i) {
            const float value = v[i];
            if (value > best0) {
                best1 = best0;
                idx1 = idx0;
                best0 = value;
                idx0 = expert + i;
            } else if (value > best1) {
                best1 = value;
                idx1 = expert + i;
            }
        }
    }
    // Process remaining experts
    for (; expert < num_experts_; ++expert) {
        const float value = token_logits[expert];
        if (value > best0) {
            best1 = best0;
            idx1 = idx0;
            best0 = value;
            idx0 = expert;
        } else if (value > best1) {
            best1 = value;
            idx1 = expert;
        }
    }
#else
    // Scalar implementation with branch prediction hints
    float best0 = neg_inf;
    float best1 = neg_inf;
    int32_t idx0 = -1;
    int32_t idx1 = -1;

    for (int32_t expert = 0; expert < num_experts_; ++expert) {
        const float value = token_logits[expert];
        if (MM_LIKELY(value <= best1)) {
            continue;
        }
        if (MM_LIKELY(value > best0)) {
            best1 = best0;
            idx1 = idx0;
            best0 = value;
            idx0 = expert;
        } else {
            best1 = value;
            idx1 = expert;
        }
    }
#endif

    topk[0].value = best0;
    topk[0].expert_id = idx0;
    topk[1].value = best1;
    topk[1].expert_id = idx1;
}

void FastRouterDispatcher::select_topk_for_token(const float* token_logits,
                                                 TopKCandidate* topk) const {
    if (top_k_ == 2) {
        select_top2_for_token_simd(token_logits, topk);
        return;
    }

    auto maybe_insert = [&](float value, int32_t expert_id) {
        if (value <= topk[top_k_ - 1].value) {
            return;
        }

        int32_t pos = top_k_ - 1;
        while (pos > 0 && value > topk[pos - 1].value) {
            topk[pos] = topk[pos - 1];
            --pos;
        }
        topk[pos].value = value;
        topk[pos].expert_id = expert_id;
    };

    int32_t expert = 0;
    for (; expert + 4 <= num_experts_; expert += 4) {
        const float block_max = max4_simd(token_logits + expert);
        if (block_max <= topk[top_k_ - 1].value) {
            continue;
        }

        maybe_insert(token_logits[expert], expert);
        maybe_insert(token_logits[expert + 1], expert + 1);
        maybe_insert(token_logits[expert + 2], expert + 2);
        maybe_insert(token_logits[expert + 3], expert + 3);
    }
    for (; expert < num_experts_; ++expert) {
        maybe_insert(token_logits[expert], expert);
    }
}

void FastRouterDispatcher::select_topk_batch(const float* logits,
                                             int32_t num_tokens,
                                             int32_t* out_topk_expert_ids,
                                             float* out_topk_probs) const {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<TopKCandidate> topk(static_cast<size_t>(top_k_));

    for (int32_t token = 0; token < num_tokens; ++token) {
        for (int32_t i = 0; i < top_k_; ++i) {
            topk[i].value = neg_inf;
            topk[i].expert_id = -1;
        }

        const float* token_logits = logits + static_cast<size_t>(token) * num_experts_;
        select_topk_for_token(token_logits, topk.data());

        float max_selected = neg_inf;
        for (int32_t i = 0; i < top_k_; ++i) {
            if (topk[i].expert_id >= 0) {
                max_selected = std::max(max_selected, topk[i].value);
            }
        }

        float prob_sum = 0.0f;
        for (int32_t i = 0; i < top_k_; ++i) {
            const size_t out_idx = static_cast<size_t>(token) * top_k_ + i;
            const int32_t expert_id = topk[i].expert_id >= 0 ? topk[i].expert_id : 0;
            out_topk_expert_ids[out_idx] = expert_id;

            if (topk[i].expert_id < 0 || max_selected == neg_inf) {
                out_topk_probs[out_idx] = 0.0f;
                continue;
            }

            const float exp_value = std::exp(topk[i].value - max_selected);
            out_topk_probs[out_idx] = exp_value;
            prob_sum += exp_value;
        }

        if (prob_sum > 0.0f) {
            for (int32_t i = 0; i < top_k_; ++i) {
                const size_t out_idx = static_cast<size_t>(token) * top_k_ + i;
                out_topk_probs[out_idx] /= prob_sum;
            }
        }
    }
}

void FastRouterDispatcher::update_hot_pair_stats(int32_t first_expert, int32_t second_expert) {
    const uint32_t key = encode_pair(first_expert, second_expert);
    if (key == kInvalidPairKey) {
        return;
    }

    uint32_t& hits = pair_hit_counts_[key];
    if (hits != std::numeric_limits<uint32_t>::max()) {
        ++hits;
    }

    if (!hot_pair_mask_[key] && hits >= hot_pair_threshold_ &&
        hot_pair_count_ < hot_pair_cache_capacity_) {
        hot_pair_mask_[key] = 1;
        ++hot_pair_count_;
    }
}

bool FastRouterDispatcher::is_hot_pair(int32_t first_expert, int32_t second_expert) const {
    const uint32_t key = encode_pair(first_expert, second_expert);
    if (key == kInvalidPairKey) {
        return false;
    }
    return hot_pair_mask_[key] != 0 || is_preferred_fast_pair(key);
}

void FastRouterDispatcher::build_dispatch_with_cache(const int32_t* topk_expert_ids,
                                                     int32_t num_tokens,
                                                     DispatchInfo* out_info) {
    const int32_t total_assignments = num_tokens * top_k_;

    out_info->num_tokens = num_tokens;
    out_info->top_k = top_k_;
    out_info->num_experts = num_experts_;
    out_info->sorted_token_indices.resize(total_assignments);
    out_info->sorted_expert_indices.resize(total_assignments);
    out_info->inverse_indices.resize(total_assignments);
    out_info->expert_offsets.assign(static_cast<size_t>(num_experts_) + 1U, 0);

    std::vector<int32_t> expert_counts(static_cast<size_t>(num_experts_), 0);
    std::vector<uint32_t> token_pair_keys(static_cast<size_t>(num_tokens), kInvalidPairKey);
    std::vector<uint8_t> fast_pair_token(static_cast<size_t>(num_tokens), 0U);

    if (top_k_ == 2) {
        for (int32_t token = 0; token < num_tokens; ++token) {
            const int32_t base = token * top_k_;
            token_pair_keys[static_cast<size_t>(token)] =
                encode_pair(topk_expert_ids[base], topk_expert_ids[base + 1]);
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (int32_t token = 0; token < num_tokens; ++token) {
            const uint32_t pair_key = token_pair_keys[static_cast<size_t>(token)];
            if (pair_key == kInvalidPairKey) {
                continue;
            }
            const auto& pair = pair_lookup_[pair_key];
            update_hot_pair_stats(pair[0], pair[1]);
            if (hot_pair_mask_[pair_key] != 0U || is_preferred_fast_pair(pair_key)) {
                fast_pair_token[static_cast<size_t>(token)] = 1U;
            }
        }
    }

    for (int32_t token = 0; token < num_tokens; ++token) {
        const int32_t base = token * top_k_;
        const uint32_t pair_key = token_pair_keys[static_cast<size_t>(token)];
        if (top_k_ == 2 && pair_key != kInvalidPairKey &&
            fast_pair_token[static_cast<size_t>(token)] != 0U) {
            const auto& pair = pair_lookup_[pair_key];
            ++expert_counts[pair[0]];
            ++expert_counts[pair[1]];
            continue;
        }

        for (int32_t slot = 0; slot < top_k_; ++slot) {
            const int32_t raw_expert = topk_expert_ids[base + slot];
            const int32_t safe_expert =
                std::clamp(raw_expert, int32_t{0}, num_experts_ - 1);
            ++expert_counts[safe_expert];
        }
    }

    out_info->expert_offsets[0] = 0;
    for (int32_t expert = 0; expert < num_experts_; ++expert) {
        out_info->expert_offsets[expert + 1] =
            out_info->expert_offsets[expert] + expert_counts[expert];
    }

    std::vector<int32_t> cursor = out_info->expert_offsets;
    for (int32_t token = 0; token < num_tokens; ++token) {
        const int32_t base = token * top_k_;
        const uint32_t pair_key = token_pair_keys[static_cast<size_t>(token)];

        if (top_k_ == 2 && pair_key != kInvalidPairKey &&
            fast_pair_token[static_cast<size_t>(token)] != 0U) {
            const auto& pair = pair_lookup_[pair_key];
            const int32_t out0 = cursor[pair[0]]++;
            const int32_t out1 = cursor[pair[1]]++;

            out_info->sorted_token_indices[out0] = token;
            out_info->sorted_expert_indices[out0] = 0;
            out_info->inverse_indices[base] = out0;

            out_info->sorted_token_indices[out1] = token;
            out_info->sorted_expert_indices[out1] = 1;
            out_info->inverse_indices[base + 1] = out1;
            continue;
        }

        for (int32_t slot = 0; slot < top_k_; ++slot) {
            const int32_t raw_expert = topk_expert_ids[base + slot];
            const int32_t safe_expert =
                std::clamp(raw_expert, int32_t{0}, num_experts_ - 1);
            const int32_t out_pos = cursor[safe_expert]++;
            out_info->sorted_token_indices[out_pos] = token;
            out_info->sorted_expert_indices[out_pos] = slot;
            out_info->inverse_indices[base + slot] = out_pos;
        }
    }
}

void FastRouterDispatcher::route_batch_into(const uint16_t* token_activations_bf16,
                                            int32_t num_tokens,
                                            const float* router_weights,
                                            const float* router_bias,
                                            RouterBatchOutput* out) {
    if (!out) {
        throw std::invalid_argument("FastRouterDispatcher::route_batch_into: output is null");
    }
    if (!token_activations_bf16) {
        throw std::invalid_argument("FastRouterDispatcher::route_batch_into: token activations is null");
    }
    if (!router_weights) {
        throw std::invalid_argument("FastRouterDispatcher::route_batch_into: router weights is null");
    }
    if (num_tokens <= 0) {
        throw std::invalid_argument("FastRouterDispatcher::route_batch_into: num_tokens must be > 0");
    }

    out->num_tokens = num_tokens;
    out->top_k = top_k_;
    out->num_experts = num_experts_;

    const size_t total_logits = static_cast<size_t>(num_tokens) * num_experts_;
    const size_t total_topk = static_cast<size_t>(num_tokens) * top_k_;

    out->logits.resize(total_logits);
    out->topk_expert_ids.resize(total_topk);
    out->topk_probs.resize(total_topk);

    compute_logits_batch(token_activations_bf16,
                         num_tokens,
                         router_weights,
                         router_bias,
                         out->logits.data());

    select_topk_batch(out->logits.data(),
                      num_tokens,
                      out->topk_expert_ids.data(),
                      out->topk_probs.data());

    build_dispatch_with_cache(out->topk_expert_ids.data(),
                              num_tokens,
                              &out->dispatch_info);
}

RouterBatchOutput FastRouterDispatcher::route_batch(const uint16_t* token_activations_bf16,
                                                    int32_t num_tokens,
                                                    const float* router_weights,
                                                    const float* router_bias) {
    RouterBatchOutput output;
    reserve_batch_output_buffers(&output);
    route_batch_into(token_activations_bf16,
                     num_tokens,
                     router_weights,
                     router_bias,
                     &output);
    return output;
}

std::shared_future<void> FastRouterDispatcher::submit_async(
    const uint16_t* token_activations_bf16,
    int32_t num_tokens,
    const float* router_weights,
    const float* router_bias,
    ExpertLaunchFn launch_experts) {
    if (!launch_experts) {
        throw std::invalid_argument("FastRouterDispatcher::submit_async: launch_experts is empty");
    }
    std::lock_guard<std::mutex> submit_lock(submit_mutex_);

    int32_t buffer_index = 0;
    uint64_t sequence_id = 0;
    std::shared_future<void> prior_stage;
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        ++submitted_sequence_;
        sequence_id = submitted_sequence_;
        buffer_index = static_cast<int32_t>(sequence_id & 1ULL);
        if (async_stage_futures_[buffer_index].valid()) {
            prior_stage = async_stage_futures_[buffer_index];
        }
    }

    if (prior_stage.valid()) {
        prior_stage.wait();
    }

    // Compute token N routing on CPU while experts for token N-1 are running.
    thread_local RouterBatchOutput stage_output;
    reserve_batch_output_buffers(&stage_output);
    route_batch_into(token_activations_bf16,
                     num_tokens,
                     router_weights,
                     router_bias,
                     &stage_output);

    const RouterBuffer* ready_buffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        router_buffers_[buffer_index].sequence_id = sequence_id;
        std::swap(router_buffers_[buffer_index].output, stage_output);
        previous_buffer_index_ = current_buffer_index_;
        current_buffer_index_ = buffer_index;
        ready_buffer = &router_buffers_[buffer_index];
    }

    std::shared_future<void> expert_done = launch_experts(*ready_buffer);
    if (!expert_done.valid()) {
        std::promise<void> done;
        done.set_value();
        expert_done = done.get_future().share();
    }

    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_stage_futures_[buffer_index] = expert_done;
    }

    return expert_done;
}

RouterBuffer FastRouterDispatcher::current_router_buffer() const {
    std::lock_guard<std::mutex> lock(async_mutex_);
    return router_buffers_[current_buffer_index_];
}

RouterBuffer FastRouterDispatcher::previous_router_buffer() const {
    std::lock_guard<std::mutex> lock(async_mutex_);
    return router_buffers_[previous_buffer_index_];
}

void FastRouterDispatcher::reset_hot_pair_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    std::fill(pair_hit_counts_.begin(), pair_hit_counts_.end(), 0U);
    std::fill(hot_pair_mask_.begin(), hot_pair_mask_.end(), 0U);
    hot_pair_count_ = 0;

    if (preferred_fast_pair_key_ != kInvalidPairKey &&
        hot_pair_cache_capacity_ > 0) {
        hot_pair_mask_[preferred_fast_pair_key_] = 1U;
        hot_pair_count_ = 1;
        const uint32_t reverse_pair =
            encode_pair(kPreferredFastPairSecond, kPreferredFastPairFirst);
        if (reverse_pair != kInvalidPairKey &&
            reverse_pair != preferred_fast_pair_key_ &&
            hot_pair_count_ < hot_pair_cache_capacity_) {
            hot_pair_mask_[reverse_pair] = 1U;
            ++hot_pair_count_;
        }
    }
}

size_t FastRouterDispatcher::hot_pair_count() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return hot_pair_count_;
}

} // namespace moe
} // namespace metal_marlin
