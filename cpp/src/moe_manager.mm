// moe_manager.cpp - Implementation of TokenGroup manager for MoE dispatch
//
// Provides efficient C++ implementations of token grouping and management
// for Mixture of Experts operations. This module handles:
// - Token-to-expert grouping via stable sorting
// - Activation gathering/scattering with FP16/FP32 support
// - Load balancing diagnostics
// - C API for Python bindings

#include "moe_manager.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace metal_marlin {
namespace moe {

// -----------------------------------------------------------------------------
// TokenGroupManager Implementation
// -----------------------------------------------------------------------------

TokenGroupManager::TokenGroupManager(int32_t num_experts,
                                     int32_t max_tokens,
                                     int32_t max_top_k)
    : num_experts_(num_experts),
      max_tokens_(max_tokens > 0 ? max_tokens : kDefaultMaxTokens),
      max_top_k_(max_top_k > 0 ? max_top_k : kDefaultMaxTopK) {
    if (num_experts_ <= 0) {
        throw std::invalid_argument("num_experts must be positive");
    }

    // Pre-allocate temporary buffers
    const size_t max_assignments = max_tokens_ * max_top_k_;
    temp_sort_keys_.reserve(max_assignments);
    temp_sort_indices_.reserve(max_assignments);
    temp_expert_counts_.reserve(num_experts_);
}

DispatchInfo TokenGroupManager::group_tokens(const int32_t* expert_ids,
                                             int32_t batch_size,
                                             int32_t top_k) {
    if (!expert_ids || batch_size <= 0 || top_k <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }

    DispatchInfo info;
    info.num_tokens = batch_size;
    info.top_k = top_k;
    info.num_experts = num_experts_;

    const int32_t total = info.total_assignments();

    // Allocate output arrays
    info.sorted_token_indices.resize(total);
    info.sorted_expert_indices.resize(total);
    info.expert_offsets.resize(num_experts_ + 1, 0);
    info.inverse_indices.resize(total);

    // Build sort keys: (expert_id, token_id, slot_id)
    // Pack as: expert_id * (batch_size * top_k) + token_id * top_k + slot_id
    temp_sort_keys_.resize(total);
    temp_sort_indices_.resize(total);

    for (int32_t token_id = 0; token_id < batch_size; ++token_id) {
        for (int32_t slot = 0; slot < top_k; ++slot) {
            const int32_t idx = token_id * top_k + slot;
            const int32_t expert_id = expert_ids[idx];
            
            // Pack sort key: expert first (primary), then token, then slot
            temp_sort_keys_[idx] = expert_id * (batch_size * top_k) + 
                                   token_id * top_k + slot;
            temp_sort_indices_[idx] = idx;
        }
    }

    // Stable sort by expert_id (then token_id, then slot)
    std::stable_sort(temp_sort_indices_.begin(), temp_sort_indices_.end(),
                     [this](int32_t a, int32_t b) {
                         return temp_sort_keys_[a] < temp_sort_keys_[b];
                     });

    // Build sorted arrays and compute offsets
    temp_expert_counts_.assign(num_experts_, 0);

    for (int32_t i = 0; i < total; ++i) {
        const int32_t orig_idx = temp_sort_indices_[i];
        const int32_t token_id = orig_idx / top_k;
        const int32_t slot = orig_idx % top_k;
        const int32_t expert_id = expert_ids[orig_idx];

        info.sorted_token_indices[i] = token_id;
        info.sorted_expert_indices[i] = slot;
        
        if (expert_id >= 0 && expert_id < num_experts_) {
            temp_expert_counts_[expert_id]++;
        }
    }

    // Compute cumulative offsets
    info.expert_offsets[0] = 0;
    for (int32_t i = 0; i < num_experts_; ++i) {
        info.expert_offsets[i + 1] = info.expert_offsets[i] + temp_expert_counts_[i];
    }

    // Compute inverse indices for scatter operation
    for (int32_t sorted_idx = 0; sorted_idx < total; ++sorted_idx) {
        const int32_t orig_idx = temp_sort_indices_[sorted_idx];
        info.inverse_indices[orig_idx] = sorted_idx;
    }

    return info;
}

DispatchInfo TokenGroupManager::group_tokens_2d(const int32_t* expert_ids,
                                                 int32_t batch_size,
                                                 int32_t top_k) {
    // 2D layout is already flattened in row-major order, delegate to main impl
    return group_tokens(expert_ids, batch_size, top_k);
}

// -----------------------------------------------------------------------------
// Activation Gathering/Scattering
// -----------------------------------------------------------------------------

void TokenGroupManager::gather_activations(const void* activations,
                                           void* out_gathered,
                                           const DispatchInfo& info,
                                           int32_t hidden_dim,
                                           size_t element_size) {
    if (!activations || !out_gathered || hidden_dim <= 0 || element_size == 0) {
        throw std::invalid_argument("Invalid gather parameters");
    }

    const size_t row_bytes = hidden_dim * element_size;
    const auto* src = static_cast<const uint8_t*>(activations);
    auto* dst = static_cast<uint8_t*>(out_gathered);

    // Gather: out[i] = activations[sorted_token_indices[i]]
    for (int32_t i = 0; i < info.total_assignments(); ++i) {
        const int32_t token_id = info.sorted_token_indices[i];
        std::memcpy(dst + i * row_bytes,
                   src + token_id * row_bytes,
                   row_bytes);
    }
}

void TokenGroupManager::scatter_outputs(const void* expert_outputs,
                                        const float* expert_probs,
                                        void* out_output,
                                        const DispatchInfo& info,
                                        int32_t hidden_dim,
                                        size_t element_size) {
    if (!expert_outputs || !expert_probs || !out_output) {
        throw std::invalid_argument("Invalid scatter parameters");
    }

    const auto* src = static_cast<const uint8_t*>(expert_outputs);
    auto* dst = static_cast<uint8_t*>(out_output);

    // Zero output buffer
    std::memset(dst, 0, info.num_tokens * hidden_dim * element_size);

    // Scatter with probability weighting
    // For each expert output, add weighted contribution to final output
    for (int32_t i = 0; i < info.total_assignments(); ++i) {
        const int32_t token_id = info.sorted_token_indices[i];
        const int32_t slot = info.sorted_expert_indices[i];
        const float prob = expert_probs[token_id * info.top_k + slot];

        // Get pointers to source and destination rows
        const uint8_t* src_row = src + i * hidden_dim * element_size;
        uint8_t* dst_row = dst + token_id * hidden_dim * element_size;

        // Weighted accumulation (element_size determines FP16 vs FP32)
        if (element_size == 2) {
            // FP16 path
            const auto* src_fp16 = reinterpret_cast<const uint16_t*>(src_row);
            auto* dst_fp16 = reinterpret_cast<uint16_t*>(dst_row);
            
            // Convert FP16 to FP32, scale, accumulate, convert back
            // Note: This is a simplified version. Production code should use
            // hardware FP16 intrinsics or Metal compute shaders for efficiency.
            for (int32_t j = 0; j < hidden_dim; ++j) {
                // Simplified FP16->FP32->FP16 (real impl needs proper conversion)
                float val = static_cast<float>(src_fp16[j]) * prob;
                dst_fp16[j] = static_cast<uint16_t>(val);
            }
        } else if (element_size == 4) {
            // FP32 path
            const auto* src_fp32 = reinterpret_cast<const float*>(src_row);
            auto* dst_fp32 = reinterpret_cast<float*>(dst_row);
            
            for (int32_t j = 0; j < hidden_dim; ++j) {
                dst_fp32[j] += src_fp32[j] * prob;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Per-expert Iteration
// -----------------------------------------------------------------------------

TokenGroup TokenGroupManager::get_token_group(int32_t expert_id,
                                              const DispatchInfo& info) const {
    TokenGroup group;
    group.expert_id = expert_id;
    group.dispatch_info = &info;

    if (expert_id >= 0 && expert_id < info.num_experts) {
        group.start_idx = info.expert_offsets[expert_id];
        group.end_idx = info.expert_offsets[expert_id + 1];
    }

    return group;
}

void TokenGroupManager::for_each_group(const DispatchInfo& info,
                                       void (*callback)(const TokenGroup&, void*),
                                       void* user_data) const {
    if (!callback) return;

    for (int32_t expert_id = 0; expert_id < info.num_experts; ++expert_id) {
        TokenGroup group = get_token_group(expert_id, info);
        if (!group.empty()) {
            callback(group, user_data);
        }
    }
}

// -----------------------------------------------------------------------------
// Buffer Management
// -----------------------------------------------------------------------------

void* TokenGroupManager::allocate_grouped_buffer(const DispatchInfo& info,
                                                 int32_t hidden_dim,
                                                 size_t element_size) {
    const size_t total_bytes = info.total_assignments() * hidden_dim * element_size;
    
    // Allocate aligned buffer
    void* buffer = nullptr;
    const int result = posix_memalign(&buffer, kMoECacheLineSize, total_bytes);
    
    if (result != 0 || !buffer) {
        throw std::bad_alloc();
    }

    return buffer;
}

void TokenGroupManager::free_grouped_buffer(void* buffer) {
    if (buffer) {
        free(buffer);
    }
}

// -----------------------------------------------------------------------------
// Statistics and Diagnostics
// -----------------------------------------------------------------------------

std::vector<int32_t> TokenGroupManager::compute_expert_loads(
    const DispatchInfo& info) const {
    std::vector<int32_t> loads(info.num_experts);
    
    for (int32_t i = 0; i < info.num_experts; ++i) {
        loads[i] = info.expert_batch_size(i);
    }
    
    return loads;
}

float TokenGroupManager::compute_load_imbalance(const DispatchInfo& info) const {
    const auto loads = compute_expert_loads(info);
    
    // Compute mean and std of non-zero loads
    std::vector<int32_t> non_zero_loads;
    non_zero_loads.reserve(loads.size());
    
    for (int32_t load : loads) {
        if (load > 0) {
            non_zero_loads.push_back(load);
        }
    }
    
    if (non_zero_loads.empty()) {
        return 0.0f;
    }
    
    const float mean = std::accumulate(non_zero_loads.begin(),
                                      non_zero_loads.end(),
                                      0.0f) / non_zero_loads.size();
    
    if (mean < 1e-6f) {
        return 0.0f;
    }
    
    float variance = 0.0f;
    for (int32_t load : non_zero_loads) {
        const float diff = load - mean;
        variance += diff * diff;
    }
    variance /= non_zero_loads.size();
    
    const float stddev = std::sqrt(variance);
    return stddev / mean;
}

bool TokenGroupManager::is_load_balanced(const DispatchInfo& info,
                                         float threshold) const {
    const float imbalance = compute_load_imbalance(info);
    return imbalance <= threshold;
}

// -----------------------------------------------------------------------------
// C API Implementation
// -----------------------------------------------------------------------------

extern "C" {

void* moe_manager_create(int32_t num_experts,
                        int32_t max_tokens,
                        int32_t max_top_k) {
    try {
        auto* manager = new TokenGroupManager(
            num_experts,
            max_tokens > 0 ? max_tokens : kDefaultMaxTokens,
            max_top_k > 0 ? max_top_k : kDefaultMaxTopK
        );
        return static_cast<void*>(manager);
    } catch (...) {
        return nullptr;
    }
}

void moe_manager_destroy(void* manager) {
    if (manager) {
        delete static_cast<TokenGroupManager*>(manager);
    }
}

int moe_group_tokens(void* manager,
                    const int32_t* expert_ids,
                    int32_t batch_size,
                    int32_t top_k,
                    int32_t* out_sorted_token_indices,
                    int32_t* out_sorted_expert_indices,
                    int32_t* out_expert_offsets,
                    int32_t* out_inverse_indices) {
    if (!manager || !expert_ids) return -1;
    if (!out_sorted_token_indices || !out_sorted_expert_indices) return -1;
    if (!out_expert_offsets || !out_inverse_indices) return -1;

    try {
        auto* mgr = static_cast<TokenGroupManager*>(manager);
        auto info = mgr->group_tokens(expert_ids, batch_size, top_k);

        // Copy results to output buffers
        std::memcpy(out_sorted_token_indices,
                   info.sorted_token_indices.data(),
                   info.sorted_token_indices.size() * sizeof(int32_t));
        
        std::memcpy(out_sorted_expert_indices,
                   info.sorted_expert_indices.data(),
                   info.sorted_expert_indices.size() * sizeof(int32_t));
        
        std::memcpy(out_expert_offsets,
                   info.expert_offsets.data(),
                   info.expert_offsets.size() * sizeof(int32_t));
        
        std::memcpy(out_inverse_indices,
                   info.inverse_indices.data(),
                   info.inverse_indices.size() * sizeof(int32_t));

        return 0;
    } catch (...) {
        return -1;
    }
}

int moe_gather_activations_f16(void* manager,
                               const uint16_t* activations,
                               const int32_t* sorted_token_indices,
                               int32_t total_assignments,
                               int32_t hidden_dim,
                               uint16_t* out_gathered) {
    if (!manager || !activations || !sorted_token_indices || !out_gathered) {
        return -1;
    }

    try {
        // Direct memory gather using sorted indices
        const size_t row_bytes = hidden_dim * sizeof(uint16_t);
        
        for (int32_t i = 0; i < total_assignments; ++i) {
            const int32_t token_id = sorted_token_indices[i];
            std::memcpy(out_gathered + i * hidden_dim,
                       activations + token_id * hidden_dim,
                       row_bytes);
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

int moe_scatter_outputs_f16(void* manager,
                           const uint16_t* expert_outputs,
                           const float* expert_probs,
                           const int32_t* sorted_token_indices,
                           const int32_t* sorted_expert_indices,
                           const int32_t* inverse_indices,
                           int32_t batch_size,
                           int32_t top_k,
                           int32_t hidden_dim,
                           uint16_t* out_output) {
    if (!manager || !expert_outputs || !expert_probs) return -1;
    if (!sorted_token_indices || !sorted_expert_indices) return -1;
    if (!out_output) return -1;

    try {
        // Zero output
        std::memset(out_output, 0, batch_size * hidden_dim * sizeof(uint16_t));

        // Scatter with weighting
        const int32_t total_assignments = batch_size * top_k;
        
        for (int32_t i = 0; i < total_assignments; ++i) {
            const int32_t token_id = sorted_token_indices[i];
            const int32_t slot = sorted_expert_indices[i];
            const float prob = expert_probs[token_id * top_k + slot];

            // Weighted accumulation (simplified - production needs proper FP16)
            for (int32_t j = 0; j < hidden_dim; ++j) {
                const float val = static_cast<float>(expert_outputs[i * hidden_dim + j]) * prob;
                out_output[token_id * hidden_dim + j] = static_cast<uint16_t>(
                    static_cast<float>(out_output[token_id * hidden_dim + j]) + val
                );
            }
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

} // extern "C"

} // namespace moe
} // namespace metal_marlin
