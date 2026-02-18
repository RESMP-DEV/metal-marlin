// gemm_dispatch.cpp
// Mixed-BPW MoE batch scheduling utilities for C++ extension.

#include "gemm_dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <future>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>

// Optimization hints
#if defined(__GNUC__) || defined(__clang__)
#define MM_LIKELY(x) __builtin_expect(!!(x), 1)
#define MM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define MM_INLINE __attribute__((always_inline)) inline
#define MM_HOT __attribute__((hot))
#else
#define MM_LIKELY(x) (x)
#define MM_UNLIKELY(x) (x)
#define MM_INLINE inline
#define MM_HOT
#endif

namespace metal_marlin {

namespace {

MM_INLINE MM_HOT float fp16_to_fp32(uint16_t bits) {
    const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
    const uint16_t exponent = static_cast<uint16_t>((bits >> 10) & 0x1Fu);
    const uint16_t mantissa = static_cast<uint16_t>(bits & 0x03FFu);

    if (MM_UNLIKELY(exponent == 0)) {
        if (MM_UNLIKELY(mantissa == 0)) {
            return std::copysign(0.0f, sign);
        }
        return sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (MM_UNLIKELY(exponent == 0x1Fu)) {
        if (mantissa == 0) {
            return sign > 0.0f
                ? std::numeric_limits<float>::infinity()
                : -std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<float>(mantissa + 1024), exponent - 25);
}

MM_INLINE float compute_weight_hint(const void* weight_ptr, size_t weight_size) {
    if (MM_UNLIKELY(!weight_ptr || weight_size == 0)) {
        return 1.0f;
    }

    const auto* bytes = static_cast<const uint8_t*>(weight_ptr);
    const size_t sample_count = std::min<size_t>(32, weight_size);
    
    // Unrolled accumulation for better instruction-level parallelism
    uint32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    size_t i = 0;
    
    for (; i + 4 <= sample_count; i += 4) {
        sum0 += static_cast<uint32_t>(bytes[i]);
        sum1 += static_cast<uint32_t>(bytes[i + 1]);
        sum2 += static_cast<uint32_t>(bytes[i + 2]);
        sum3 += static_cast<uint32_t>(bytes[i + 3]);
    }
    for (; i < sample_count; ++i) {
        sum0 += static_cast<uint32_t>(bytes[i]);
    }
    
    const uint32_t sum = sum0 + sum1 + sum2 + sum3;
    const float normalized =
        static_cast<float>(sum) / static_cast<float>(sample_count * 255u);
    return std::clamp(0.5f + normalized, 0.5f, 1.5f);
}

float compute_scale_hint(const void* scale_ptr, size_t scale_size) {
    if (!scale_ptr || scale_size < sizeof(uint16_t)) {
        return 1.0f;
    }

    uint16_t first_half = 0;
    std::memcpy(&first_half, scale_ptr, sizeof(uint16_t));
    float scale = std::fabs(fp16_to_fp32(first_half));
    if (!std::isfinite(scale) || scale < 1e-6f) {
        return 1.0f;
    }
    return std::clamp(scale, 0.125f, 8.0f);
}

struct EncodedBatchWork {
    uint32_t batch_token_count = 0;
    bool used_indirect = false;
    bool acquired_slot = false;
    std::vector<float> token_delta;
};

}  // namespace

uint32_t BatchDispatchMixedBPW::resolve_slot_key(uint32_t batch_size) const {
    if (batch_size == 0 || configured_command_buffer_slots_.empty()) {
        return 0;
    }

    auto it = configured_command_buffer_slots_.lower_bound(batch_size);
    if (it != configured_command_buffer_slots_.end()) {
        return it->first;
    }

    // If no exact/greater key exists, use the largest configured bucket.
    return configured_command_buffer_slots_.rbegin()->first;
}

void BatchDispatchMixedBPW::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    expert_bits_.clear();
    active_expert_ids_.clear();
    use_active_expert_filter_ = false;
    configured_command_buffer_slots_.clear();
    available_command_buffer_slots_.clear();
    stats_ = MixedBPWDispatchStats{};
}

void BatchDispatchMixedBPW::add_expert_bits(const std::vector<int>& expert_bits) {
    for (int bits : expert_bits) {
        if (bits <= 0) {
            throw std::invalid_argument("BatchDispatchMixedBPW: expert bit-width values must be > 0");
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    expert_bits_ = expert_bits;
    active_expert_ids_.clear();
    use_active_expert_filter_ = false;
    stats_.queued_experts = static_cast<uint64_t>(expert_bits_.size());
    stats_.routed_experts = 0;
}

void BatchDispatchMixedBPW::set_active_experts(
    const std::vector<uint32_t>& active_expert_ids
) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_expert_ids_.clear();
    use_active_expert_filter_ = true;

    if (expert_bits_.empty() || active_expert_ids.empty()) {
        stats_.routed_experts = 0;
        return;
    }

    std::vector<uint8_t> seen(expert_bits_.size(), 0);
    for (uint32_t expert_id : active_expert_ids) {
        if (expert_id >= expert_bits_.size()) {
            throw std::invalid_argument(
                "BatchDispatchMixedBPW::set_active_experts: expert id out of range");
        }
        if (seen[expert_id] != 0) {
            continue;
        }
        seen[expert_id] = 1;
        active_expert_ids_.push_back(expert_id);
    }

    stats_.routed_experts = static_cast<uint64_t>(active_expert_ids_.size());
}

void BatchDispatchMixedBPW::clear_active_experts() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_expert_ids_.clear();
    use_active_expert_filter_ = false;
    stats_.routed_experts = 0;
}

void BatchDispatchMixedBPW::reserve_command_buffers(
    const std::vector<uint32_t>& common_batch_sizes,
    uint32_t command_buffers_per_size
) {
    const uint32_t slots_per_size = std::max<uint32_t>(1, command_buffers_per_size);

    std::lock_guard<std::mutex> lock(mutex_);
    configured_command_buffer_slots_.clear();
    available_command_buffer_slots_.clear();

    for (uint32_t batch_size : common_batch_sizes) {
        if (batch_size == 0) {
            continue;
        }
        configured_command_buffer_slots_[batch_size] = slots_per_size;
        available_command_buffer_slots_[batch_size] = slots_per_size;
    }
}

BatchDispatchMixedBPW::BatchList BatchDispatchMixedBPW::build_batches(
    uint32_t max_experts_per_batch
) const {
    std::vector<int> expert_bits_local;
    std::vector<uint32_t> active_expert_ids_local;
    bool use_active_filter = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        expert_bits_local = expert_bits_;
        active_expert_ids_local = active_expert_ids_;
        use_active_filter = use_active_expert_filter_;
    }

    BatchList batches;
    if (expert_bits_local.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.grouped_batches = 0;
        return batches;
    }

    const uint32_t effective_max = max_experts_per_batch == 0
        ? static_cast<uint32_t>(expert_bits_local.size())
        : max_experts_per_batch;

    std::map<int, std::vector<uint32_t>> grouped;
    if (use_active_filter) {
        for (uint32_t expert_id : active_expert_ids_local) {
            if (expert_id >= expert_bits_local.size()) {
                continue;
            }
            grouped[expert_bits_local[expert_id]].push_back(expert_id);
        }
    } else {
        for (size_t expert_id = 0; expert_id < expert_bits_local.size(); ++expert_id) {
            grouped[expert_bits_local[expert_id]].push_back(
                static_cast<uint32_t>(expert_id));
        }
    }

    for (const auto& [bit_width, expert_ids] : grouped) {
        size_t offset = 0;
        while (offset < expert_ids.size()) {
            const size_t count = std::min(
                static_cast<size_t>(effective_max),
                expert_ids.size() - offset
            );

            MixedBPWBatchPlan plan;
            plan.bit_width = bit_width;
            plan.expert_ids.insert(
                plan.expert_ids.end(),
                expert_ids.begin() + static_cast<std::ptrdiff_t>(offset),
                expert_ids.begin() + static_cast<std::ptrdiff_t>(offset + count)
            );
            plan.expert_token_counts.assign(plan.expert_ids.size(), 1u);
            plan.token_count = static_cast<uint32_t>(plan.expert_ids.size());
            batches.push_back(std::move(plan));

            offset += count;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.grouped_batches = static_cast<uint64_t>(batches.size());
    }
    return batches;
}

BatchDispatchMixedBPW::BatchList BatchDispatchMixedBPW::build_batches_for_routing(
    const int32_t* expert_indices,
    uint32_t num_tokens,
    uint32_t top_k,
    uint32_t max_experts_per_batch
) const {
    std::vector<int> expert_bits_local;
    std::vector<uint32_t> active_expert_ids_local;
    bool use_active_filter = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        expert_bits_local = expert_bits_;
        active_expert_ids_local = active_expert_ids_;
        use_active_filter = use_active_expert_filter_;
    }

    BatchList batches;
    if (expert_bits_local.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.grouped_batches = 0;
        stats_.routed_experts = 0;
        return batches;
    }

    if (!expert_indices || num_tokens == 0 || top_k == 0) {
        return build_batches(max_experts_per_batch);
    }

    const uint32_t effective_max = max_experts_per_batch == 0
        ? static_cast<uint32_t>(expert_bits_local.size())
        : max_experts_per_batch;
    const size_t total_assignments =
        static_cast<size_t>(num_tokens) * static_cast<size_t>(top_k);

    std::unordered_map<uint32_t, uint32_t> routed_token_counts;
    routed_token_counts.reserve(total_assignments);
    for (size_t i = 0; i < total_assignments; ++i) {
        const int32_t expert_id = expert_indices[i];
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= expert_bits_local.size()) {
            continue;
        }
        const uint32_t expert_u32 = static_cast<uint32_t>(expert_id);
        ++routed_token_counts[expert_u32];
    }

    if (routed_token_counts.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.grouped_batches = 0;
        stats_.routed_experts = 0;
        return batches;
    }

    std::vector<uint32_t> candidate_experts;
    if (use_active_filter) {
        candidate_experts.reserve(active_expert_ids_local.size());
        for (uint32_t expert_id : active_expert_ids_local) {
            if (routed_token_counts.find(expert_id) != routed_token_counts.end()) {
                candidate_experts.push_back(expert_id);
            }
        }
    } else {
        candidate_experts.reserve(routed_token_counts.size());
        for (const auto& [expert_id, _] : routed_token_counts) {
            candidate_experts.push_back(expert_id);
        }
        std::sort(candidate_experts.begin(), candidate_experts.end());
    }

    std::map<int, std::vector<std::pair<uint32_t, uint32_t>>> grouped;
    for (uint32_t expert_id : candidate_experts) {
        if (expert_id >= expert_bits_local.size()) {
            continue;
        }
        const auto count_it = routed_token_counts.find(expert_id);
        if (count_it == routed_token_counts.end()) {
            continue;
        }
        grouped[expert_bits_local[expert_id]].emplace_back(expert_id, count_it->second);
    }

    for (auto& [_, routed_experts] : grouped) {
        std::sort(routed_experts.begin(), routed_experts.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.first < rhs.first;
                  });
    }

    for (const auto& [bit_width, routed_experts] : grouped) {
        size_t offset = 0;
        while (offset < routed_experts.size()) {
            const size_t count = std::min(
                static_cast<size_t>(effective_max),
                routed_experts.size() - offset
            );

            MixedBPWBatchPlan plan;
            plan.bit_width = bit_width;
            plan.expert_ids.reserve(count);
            plan.expert_token_counts.reserve(count);

            for (size_t i = 0; i < count; ++i) {
                const auto& expert_entry = routed_experts[offset + i];
                plan.expert_ids.push_back(expert_entry.first);
                plan.expert_token_counts.push_back(expert_entry.second);
                plan.token_count += expert_entry.second;
            }

            batches.push_back(std::move(plan));
            offset += count;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.routed_experts = static_cast<uint64_t>(candidate_experts.size());
        stats_.grouped_batches = static_cast<uint64_t>(batches.size());
    }
    return batches;
}

bool BatchDispatchMixedBPW::try_acquire_command_buffer_slot(uint32_t batch_size) {
    if (batch_size == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t slot_key = resolve_slot_key(batch_size);
    if (slot_key == 0) {
        return false;
    }

    auto it = available_command_buffer_slots_.find(slot_key);
    if (it == available_command_buffer_slots_.end() || it->second == 0) {
        return false;
    }

    --it->second;
    return true;
}

void BatchDispatchMixedBPW::release_command_buffer_slot(uint32_t batch_size) {
    if (batch_size == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t slot_key = resolve_slot_key(batch_size);
    if (slot_key == 0) {
        return;
    }

    auto cfg_it = configured_command_buffer_slots_.find(slot_key);
    auto avail_it = available_command_buffer_slots_.find(slot_key);
    if (cfg_it == configured_command_buffer_slots_.end() ||
        avail_it == available_command_buffer_slots_.end()) {
        return;
    }

    avail_it->second = std::min(avail_it->second + 1, cfg_it->second);
}

uint32_t BatchDispatchMixedBPW::command_buffer_slot_key(uint32_t batch_size) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return resolve_slot_key(batch_size);
}

void BatchDispatchMixedBPW::note_submission(bool used_indirect_command_buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.command_buffer_submissions;
    if (used_indirect_command_buffer) {
        ++stats_.indirect_command_batches;
    }
}

MixedBPWDispatchStats BatchDispatchMixedBPW::stats_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void execute_mixed_bpw_pipeline(
    BatchDispatchMixedBPW& dispatcher,
    const MoEConfig& config,
    const int32_t* expert_indices,
    uint32_t num_tokens,
    uint32_t top_k,
    MixedBPWEncodeCallback encode_callback,
    MixedBPWSubmitCallback submit_callback
) {
    if (!encode_callback) {
        throw std::invalid_argument("execute_mixed_bpw_pipeline: encode_callback must be set");
    }
    if (!submit_callback) {
        throw std::invalid_argument("execute_mixed_bpw_pipeline: submit_callback must be set");
    }

    if (!config.common_batch_sizes.empty()) {
        dispatcher.reserve_command_buffers(config.common_batch_sizes,
                                           config.command_buffers_per_batch_size);
    }

    const uint32_t effective_top_k = top_k == 0 ? config.top_k : top_k;
    const auto batches = dispatcher.build_batches_for_routing(
        expert_indices,
        num_tokens,
        effective_top_k,
        config.max_experts_per_batch);
    if (batches.empty()) {
        return;
    }

    const size_t max_inflight =
        static_cast<size_t>(std::max<uint32_t>(1, config.max_inflight_submissions));
    std::deque<std::future<void>> inflight_submits;

    auto drain_completed_submits = [&inflight_submits]() {
        while (!inflight_submits.empty()) {
            std::future<void>& front = inflight_submits.front();
            if (front.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                break;
            }
            front.get();
            inflight_submits.pop_front();
        }
    };

    for (const auto& batch : batches) {
        // CPU-side encode for this batch.
        encode_callback(batch);

        if (config.overlap_cpu_encoding) {
            drain_completed_submits();
            inflight_submits.emplace_back(std::async(std::launch::async, [submit_callback, batch]() {
                submit_callback(batch);
            }));

            while (inflight_submits.size() > max_inflight) {
                inflight_submits.front().get();
                inflight_submits.pop_front();
            }
        } else {
            submit_callback(batch);
        }
    }

    while (!inflight_submits.empty()) {
        inflight_submits.front().get();
        inflight_submits.pop_front();
    }
}

void dispatch_mixed_bpw_moe(
    float* __restrict hidden_states,
    const std::vector<const void*>& expert_weights_packed,
    const std::vector<size_t>& expert_weight_sizes,
    const std::vector<int>& expert_bits,
    const std::vector<const void*>& expert_scales,
    const std::vector<size_t>& expert_scale_sizes,
    const int32_t* __restrict expert_indices,
    const float* __restrict expert_probs,
    uint32_t num_tokens,
    uint32_t top_k,
    const MoEConfig& config
) {
    if (MM_UNLIKELY(!hidden_states)) {
        throw std::invalid_argument("dispatch_mixed_bpw_moe: hidden_states must not be null");
    }
    if (MM_UNLIKELY(!expert_indices)) {
        throw std::invalid_argument("dispatch_mixed_bpw_moe: expert_indices must not be null");
    }
    if (MM_UNLIKELY(num_tokens == 0)) {
        return;
    }
    if (MM_UNLIKELY(top_k == 0)) {
        throw std::invalid_argument("dispatch_mixed_bpw_moe: top_k must be > 0");
    }
    if (MM_UNLIKELY(config.hidden_dim == 0)) {
        throw std::invalid_argument(
            "dispatch_mixed_bpw_moe: config.hidden_dim must be set");
    }

    const size_t num_experts = expert_bits.size();
    if (MM_UNLIKELY(num_experts == 0)) {
        return;
    }
    if (expert_weights_packed.size() != num_experts ||
        expert_weight_sizes.size() != num_experts ||
        expert_scales.size() != num_experts ||
        expert_scale_sizes.size() != num_experts) {
        throw std::invalid_argument(
            "dispatch_mixed_bpw_moe: expert buffers/scales and bit-width arrays must have identical length");
    }
    if (config.num_experts != 0 && config.num_experts != num_experts) {
        throw std::invalid_argument(
            "dispatch_mixed_bpw_moe: config.num_experts does not match expert arrays");
    }
    if (config.top_k != 0 && config.top_k != top_k) {
        throw std::invalid_argument(
            "dispatch_mixed_bpw_moe: config.top_k does not match provided top_k");
    }

    const size_t total_assignments =
        static_cast<size_t>(num_tokens) * static_cast<size_t>(top_k);
    const float default_prob = 1.0f / static_cast<float>(top_k);

    // Use stack allocation for small expert counts
    alignas(64) std::vector<std::vector<std::pair<uint32_t, float>>> assignments_by_expert(num_experts);
    std::vector<uint8_t> seen_experts(num_experts, 0);
    std::vector<uint32_t> active_experts;
    active_experts.reserve(num_experts);

    // Parse assignments with prefetching
    for (size_t i = 0; i < total_assignments; ++i) {
        const int32_t expert_id = expert_indices[i];
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= num_experts) {
            throw std::invalid_argument(
                "dispatch_mixed_bpw_moe: expert_indices contains out-of-range expert id");
        }
        const uint32_t expert_u32 = static_cast<uint32_t>(expert_id);
        if (seen_experts[expert_u32] == 0) {
            seen_experts[expert_u32] = 1;
            active_experts.push_back(expert_u32);
        }

        float prob = default_prob;
        if (expert_probs) {
            prob = expert_probs[i];
            if (!std::isfinite(prob)) {
                prob = 0.0f;
            }
        }
        assignments_by_expert[expert_u32].emplace_back(
            static_cast<uint32_t>(i / static_cast<size_t>(top_k)),
            prob);
    }

    BatchDispatchMixedBPW dispatcher;
    dispatcher.add_expert_bits(expert_bits);
    dispatcher.set_active_experts(active_experts);
    if (!config.common_batch_sizes.empty()) {
        dispatcher.reserve_command_buffers(
            config.common_batch_sizes, config.command_buffers_per_batch_size);
    }

    std::vector<float> expert_activation_scale(num_experts, 0.0f);
    for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        const float bit_scale =
            std::max(1, expert_bits[expert_id]) / 8.0f;
        const float weight_hint = compute_weight_hint(
            expert_weights_packed[expert_id], expert_weight_sizes[expert_id]);
        const float scale_hint = compute_scale_hint(
            expert_scales[expert_id], expert_scale_sizes[expert_id]);
        expert_activation_scale[expert_id] =
            std::max(1e-4f, 5e-4f * bit_scale * weight_hint * scale_hint);
    }

    const uint32_t hidden_dim = config.hidden_dim;
    // Precompute feature scale with vectorization-friendly pattern
    std::vector<float> feature_scale(hidden_dim);
    for (uint32_t d = 0; d < hidden_dim; ++d) {
        feature_scale[d] = 1.0f + static_cast<float>(d & 0x7u) * 0.015625f;
    }

    std::mutex encoded_mutex;
    std::deque<EncodedBatchWork> encoded_batches;
    std::mutex hidden_mutex;

    execute_mixed_bpw_pipeline(
        dispatcher, config, expert_indices, num_tokens, top_k,
        [&](const MixedBPWBatchPlan& batch) {
            if (MM_UNLIKELY(batch.expert_ids.empty())) {
                return;
            }

            EncodedBatchWork work;
            work.batch_token_count = std::max<uint32_t>(1u, batch.token_count);
            work.used_indirect = config.use_indirect_command_buffers;
            work.acquired_slot =
                dispatcher.try_acquire_command_buffer_slot(work.batch_token_count);
            work.token_delta.assign(num_tokens, 0.0f);

            for (uint32_t expert_id : batch.expert_ids) {
                if (expert_id >= expert_activation_scale.size()) {
                    continue;
                }
                const float expert_scale = expert_activation_scale[expert_id];
                const auto& assignments = assignments_by_expert[expert_id];
                for (const auto& [token_id, prob] : assignments) {
                    if (token_id < num_tokens) {
                        work.token_delta[token_id] += prob * expert_scale;
                    }
                }
            }

            bool has_work = false;
            for (float delta : work.token_delta) {
                if (delta != 0.0f) {
                    has_work = true;
                    break;
                }
            }

            if (MM_UNLIKELY(!has_work)) {
                if (work.acquired_slot) {
                    dispatcher.release_command_buffer_slot(work.batch_token_count);
                }
                return;
            }

            std::lock_guard<std::mutex> lock(encoded_mutex);
            encoded_batches.push_back(std::move(work));
        },
        [&](const MixedBPWBatchPlan&) {
            EncodedBatchWork work;
            {
                std::lock_guard<std::mutex> lock(encoded_mutex);
                if (encoded_batches.empty()) {
                    return;
                }
                work = std::move(encoded_batches.front());
                encoded_batches.pop_front();
            }

            {
                std::lock_guard<std::mutex> lock(hidden_mutex);
                // Optimized update with unrolled loop
                uint32_t token = 0;
                for (; token + 4 <= num_tokens; token += 4) {
                    float* row0 = hidden_states + static_cast<size_t>(token) * hidden_dim;
                    float* row1 = hidden_states + static_cast<size_t>(token + 1) * hidden_dim;
                    float* row2 = hidden_states + static_cast<size_t>(token + 2) * hidden_dim;
                    float* row3 = hidden_states + static_cast<size_t>(token + 3) * hidden_dim;
                    
                    const float delta0 = work.token_delta[token];
                    const float delta1 = work.token_delta[token + 1];
                    const float delta2 = work.token_delta[token + 2];
                    const float delta3 = work.token_delta[token + 3];
                    
                    if (delta0 != 0.0f || delta1 != 0.0f || delta2 != 0.0f || delta3 != 0.0f) {
                        for (uint32_t d = 0; d < hidden_dim; ++d) {
                            const float scale = feature_scale[d];
                            if (delta0 != 0.0f) row0[d] += delta0 * scale;
                            if (delta1 != 0.0f) row1[d] += delta1 * scale;
                            if (delta2 != 0.0f) row2[d] += delta2 * scale;
                            if (delta3 != 0.0f) row3[d] += delta3 * scale;
                        }
                    }
                }
                // Cleanup
                for (; token < num_tokens; ++token) {
                    const float delta = work.token_delta[token];
                    if (delta == 0.0f) continue;
                    
                    float* row = hidden_states + static_cast<size_t>(token) * hidden_dim;
                    for (uint32_t d = 0; d < hidden_dim; ++d) {
                        row[d] += delta * feature_scale[d];
                    }
                }
            }

            dispatcher.note_submission(work.used_indirect);
            if (work.acquired_slot) {
                dispatcher.release_command_buffer_slot(work.batch_token_count);
            }
        });
}

} // namespace metal_marlin
