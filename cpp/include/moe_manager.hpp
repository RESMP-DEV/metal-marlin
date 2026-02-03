// moe_manager.hpp - TokenGroup manager for MoE dispatch
//
// This module provides C++ implementations of token grouping and management
// for Mixture of Experts (MoE) dispatch operations. It mirrors the Python
// token_dispatcher.py functionality but with lower overhead for hot-path
// operations.
//
// Key features:
// - Efficient token-to-expert grouping using sorting
// - Buffer management for grouped activations
// - Offset tracking for per-expert batch sizes
// - GIL-free operation during dispatch preparation
//
// Thread Safety:
// - TokenGroupManager is NOT thread-safe (create one per thread)
// - All internal buffers are owned by the manager instance
//
// Usage:
//   TokenGroupManager manager(num_experts, max_tokens, top_k);
//   auto info = manager.group_tokens(expert_ids);  // expert_ids: [batch, top_k]
//   auto gathered = manager.gather_activations(activations, info);

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

// Forward declarations for Metal types
#ifdef __OBJC__
@protocol MTLBuffer;
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;
@protocol MTLComputePipelineState;
@protocol MTLLibrary;
typedef id<MTLBuffer> MTLBufferRef;
#else
namespace MTL {
    class Buffer;
}
typedef MTL::Buffer* MTLBufferRef;
#endif

namespace metal_marlin {
namespace moe {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum number of tokens for pre-allocation
constexpr size_t kDefaultMaxTokens = 8192;

/// Default maximum top-k value
constexpr size_t kDefaultMaxTopK = 8;

/// Cache line alignment for buffers
constexpr size_t kMoECacheLineSize = 64;

// -----------------------------------------------------------------------------
// Dispatch Info - Result of token grouping
// -----------------------------------------------------------------------------

/**
 * @brief Dispatch information for grouped MoE execution.
 *
 * Contains all indexing tensors needed to:
 * 1. Reorder tokens by expert for batched GEMM
 * 2. Apply correct expert probabilities to outputs
 * 3. Restore original token order after expert computation
 *
 * All indices are stored as 32-bit integers for GPU compatibility.
 */
struct DispatchInfo {
    // [total_assignments] indices into original batch
    // Sorted so tokens for same expert are contiguous
    std::vector<int32_t> sorted_token_indices;

    // [total_assignments] which expert slot (0 to top_k-1)
    std::vector<int32_t> sorted_expert_indices;

    // [num_experts + 1] start index for each expert's assignments
    std::vector<int32_t> expert_offsets;

    // [total_assignments] indices to scatter expert outputs back
    std::vector<int32_t> inverse_indices;

    // Metadata
    int32_t num_tokens = 0;
    int32_t top_k = 0;
    int32_t num_experts = 0;

    /// Total number of token-expert assignments
    int32_t total_assignments() const { return num_tokens * top_k; }

    /// Get number of tokens assigned to a specific expert
    int32_t expert_batch_size(int32_t expert_id) const {
        if (expert_id < 0 || expert_id >= num_experts) return 0;
        return expert_offsets[expert_id + 1] - expert_offsets[expert_id];
    }

    /// Check if an expert has any assigned tokens
    bool is_expert_active(int32_t expert_id) const {
        return expert_batch_size(expert_id) > 0;
    }

    /// Get number of active experts
    int32_t active_expert_count() const {
        int32_t count = 0;
        for (int32_t i = 0; i < num_experts; ++i) {
            if (is_expert_active(i)) ++count;
        }
        return count;
    }

    /// Clear all data
    void clear() {
        sorted_token_indices.clear();
        sorted_expert_indices.clear();
        expert_offsets.clear();
        inverse_indices.clear();
        num_tokens = 0;
        top_k = 0;
        num_experts = 0;
    }
};

// -----------------------------------------------------------------------------
// Token Group - A group of tokens assigned to the same expert
// -----------------------------------------------------------------------------

/**
 * @brief Represents a group of tokens assigned to a single expert.
 *
 * This is a lightweight view into the DispatchInfo that provides
 * convenient access to tokens for a specific expert.
 */
struct TokenGroup {
    int32_t expert_id = -1;
    int32_t start_idx = 0;  // Start index in sorted arrays
    int32_t end_idx = 0;    // End index (exclusive)
    const DispatchInfo* dispatch_info = nullptr;

    /// Number of tokens in this group
    int32_t size() const { return end_idx - start_idx; }

    /// Check if group is empty
    bool empty() const { return start_idx >= end_idx; }

    /// Get token index at position within this group
    int32_t token_index(int32_t pos) const {
        if (!dispatch_info || pos < 0 || pos >= size()) return -1;
        return dispatch_info->sorted_token_indices[start_idx + pos];
    }

    /// Get expert slot at position within this group
    int32_t expert_slot(int32_t pos) const {
        if (!dispatch_info || pos < 0 || pos >= size()) return -1;
        return dispatch_info->sorted_expert_indices[start_idx + pos];
    }

    /// Check if this group is valid
    bool is_valid() const {
        return expert_id >= 0 && dispatch_info != nullptr;
    }
};

// -----------------------------------------------------------------------------
// Token Group Manager - Main class for MoE token management
// -----------------------------------------------------------------------------

/**
 * @brief Manages token grouping and dispatch for MoE operations.
 *
 * The TokenGroupManager handles:
 * 1. Grouping tokens by their assigned expert
 * 2. Managing temporary buffers for grouped activations
 * 3. Tracking offsets for per-expert batch execution
 *
 * This class is designed for single-threaded use per instance.
 * Create multiple instances for multi-threaded operation.
 */
class TokenGroupManager {
public:
    /**
     * @brief Construct a TokenGroupManager.
     * @param num_experts Total number of experts in the MoE layer
     * @param max_tokens Maximum number of tokens to support (for pre-allocation)
     * @param max_top_k Maximum top-k value (for pre-allocation)
     */
    TokenGroupManager(int32_t num_experts,
                      int32_t max_tokens = kDefaultMaxTokens,
                      int32_t max_top_k = kDefaultMaxTopK);

    ~TokenGroupManager() = default;

    // Non-copyable, movable
    TokenGroupManager(const TokenGroupManager&) = delete;
    TokenGroupManager& operator=(const TokenGroupManager&) = delete;
    TokenGroupManager(TokenGroupManager&&) noexcept = default;
    TokenGroupManager& operator=(TokenGroupManager&&) noexcept = default;

    // -------------------------------------------------------------------------
    // Core token grouping operations
    // -------------------------------------------------------------------------

    /**
     * @brief Group tokens by their assigned expert.
     *
     * Given expert assignments [batch, top_k], produces a DispatchInfo
     * that reorders tokens so all tokens assigned to the same expert
     * are contiguous.
     *
     * @param expert_ids Pointer to [batch, top_k] expert assignments
     * @param batch_size Number of tokens in batch
     * @param top_k Number of experts per token
     * @return DispatchInfo with all indexing information
     */
    DispatchInfo group_tokens(const int32_t* expert_ids,
                              int32_t batch_size,
                              int32_t top_k);

    /**
     * @brief Group tokens from 2D array layout.
     *
     * @param expert_ids Flattened [batch, top_k] array in row-major order
     * @param batch_size Number of tokens
     * @param top_k Number of experts per token
     * @return DispatchInfo with grouping information
     */
    DispatchInfo group_tokens_2d(const int32_t* expert_ids,
                                  int32_t batch_size,
                                  int32_t top_k);

    // -------------------------------------------------------------------------
    // Activation gathering/scattering (CPU-side)
    // -------------------------------------------------------------------------

    /**
     * @brief Gather activations in expert-sorted order.
     *
     * Reorders activations so tokens for the same expert are contiguous.
     *
     * @param activations Pointer to [batch, hidden_dim] input (FP16 or FP32)
     * @param out_gathered Output buffer [total_assignments, hidden_dim]
     * @param info DispatchInfo from group_tokens
     * @param hidden_dim Hidden dimension size
     * @param element_size Size of each element (2 for FP16, 4 for FP32)
     */
    void gather_activations(const void* activations,
                            void* out_gathered,
                            const DispatchInfo& info,
                            int32_t hidden_dim,
                            size_t element_size);

    /**
     * @brief Scatter expert outputs back to original token order.
     *
     * @param expert_outputs [total_assignments, hidden_dim] outputs from experts
     * @param expert_probs [batch, top_k] routing probabilities
     * @param out_output [batch, hidden_dim] output buffer
     * @param info DispatchInfo from group_tokens
     * @param hidden_dim Hidden dimension size
     * @param element_size Size of each element
     */
    void scatter_outputs(const void* expert_outputs,
                         const float* expert_probs,
                         void* out_output,
                         const DispatchInfo& info,
                         int32_t hidden_dim,
                         size_t element_size);

    // -------------------------------------------------------------------------
    // Per-expert iteration
    // -------------------------------------------------------------------------

    /**
     * @brief Get TokenGroup for a specific expert.
     * @param expert_id Expert index
     * @param info DispatchInfo from group_tokens
     * @return TokenGroup view into the dispatch info
     */
    TokenGroup get_token_group(int32_t expert_id, const DispatchInfo& info) const;

    /**
     * @brief Iterate over all non-empty token groups.
     *
     * @param info DispatchInfo from group_tokens
     * @param callback Function to call for each non-empty group
     */
    void for_each_group(const DispatchInfo& info,
                        void (*callback)(const TokenGroup& group, void* user_data),
                        void* user_data) const;

    // -------------------------------------------------------------------------
    // Buffer management
    // -------------------------------------------------------------------------

    /**
     * @brief Allocate temporary buffer for grouped activations.
     * @param info DispatchInfo with total_assignments
     * @param hidden_dim Hidden dimension
     * @param element_size Element size in bytes
     * @return Pointer to allocated buffer (caller must free)
     */
    void* allocate_grouped_buffer(const DispatchInfo& info,
                                  int32_t hidden_dim,
                                  size_t element_size);

    /**
     * @brief Release buffer allocated by allocate_grouped_buffer.
     * @param buffer Buffer pointer
     */
    void free_grouped_buffer(void* buffer);

    // -------------------------------------------------------------------------
    // Statistics and diagnostics
    // -------------------------------------------------------------------------

    /**
     * @brief Compute load balancing statistics.
     * @param info DispatchInfo from group_tokens
     * @return Vector of token counts per expert
     */
    std::vector<int32_t> compute_expert_loads(const DispatchInfo& info) const;

    /**
     * @brief Compute load imbalance metric (std/mean of non-zero loads).
     * @param info DispatchInfo from group_tokens
     * @return Load imbalance ratio
     */
    float compute_load_imbalance(const DispatchInfo& info) const;

    /**
     * @brief Check if dispatch has acceptable load balance.
     * @param info DispatchInfo from group_tokens
     * @param threshold Maximum acceptable imbalance (default 2.0)
     * @return True if load is balanced
     */
    bool is_load_balanced(const DispatchInfo& info, float threshold = 2.0f) const;

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    int32_t num_experts() const { return num_experts_; }
    int32_t max_tokens() const { return max_tokens_; }
    int32_t max_top_k() const { return max_top_k_; }

private:
    int32_t num_experts_;
    int32_t max_tokens_;
    int32_t max_top_k_;

    // Internal temporaries for sorting (reused across calls)
    std::vector<int32_t> temp_sort_keys_;
    std::vector<int32_t> temp_sort_indices_;
    std::vector<int32_t> temp_expert_counts_;
};

// -----------------------------------------------------------------------------
// C API for Python bindings
// -----------------------------------------------------------------------------

extern "C" {

/**
 * @brief Create a new TokenGroupManager.
 * @param num_experts Number of experts
 * @param max_tokens Maximum tokens (0 for default)
 * @param max_top_k Maximum top-k (0 for default)
 * @return Opaque pointer to manager, or nullptr on failure
 */
void* moe_manager_create(int32_t num_experts,
                         int32_t max_tokens,
                         int32_t max_top_k);

/**
 * @brief Destroy a TokenGroupManager.
 * @param manager Manager pointer from moe_manager_create
 */
void moe_manager_destroy(void* manager);

/**
 * @brief Group tokens by expert.
 * @param manager Manager pointer
 * @param expert_ids Flattened [batch, top_k] expert assignments
 * @param batch_size Number of tokens
 * @param top_k Number of experts per token
 * @param out_sorted_token_indices Output buffer [batch * top_k]
 * @param out_sorted_expert_indices Output buffer [batch * top_k]
 * @param out_expert_offsets Output buffer [num_experts + 1]
 * @param out_inverse_indices Output buffer [batch * top_k]
 * @return 0 on success, -1 on failure
 */
int moe_group_tokens(void* manager,
                     const int32_t* expert_ids,
                     int32_t batch_size,
                     int32_t top_k,
                     int32_t* out_sorted_token_indices,
                     int32_t* out_sorted_expert_indices,
                     int32_t* out_expert_offsets,
                     int32_t* out_inverse_indices);

/**
 * @brief Gather activations (FP16 version).
 * @param manager Manager pointer
 * @param activations [batch, hidden_dim] FP16 activations
 * @param sorted_token_indices From moe_group_tokens
 * @param total_assignments Total assignments (batch * top_k)
 * @param hidden_dim Hidden dimension
 * @param out_gathered [total_assignments, hidden_dim] output
 * @return 0 on success
 */
int moe_gather_activations_f16(void* manager,
                               const uint16_t* activations,
                               const int32_t* sorted_token_indices,
                               int32_t total_assignments,
                               int32_t hidden_dim,
                               uint16_t* out_gathered);

/**
 * @brief Scatter outputs (FP16 version).
 * @param manager Manager pointer
 * @param expert_outputs [total_assignments, hidden_dim] FP16
 * @param expert_probs [batch, top_k] float probabilities
 * @param sorted_token_indices From moe_group_tokens
 * @param sorted_expert_indices From moe_group_tokens
 * @param inverse_indices From moe_group_tokens
 * @param batch_size Number of tokens
 * @param top_k Top-k value
 * @param hidden_dim Hidden dimension
 * @param out_output [batch, hidden_dim] output
 * @return 0 on success
 */
int moe_scatter_outputs_f16(void* manager,
                            const uint16_t* expert_outputs,
                            const float* expert_probs,
                            const int32_t* sorted_token_indices,
                            const int32_t* sorted_expert_indices,
                            const int32_t* inverse_indices,
                            int32_t batch_size,
                            int32_t top_k,
                            int32_t hidden_dim,
                            uint16_t* out_output);

} // extern "C"

} // namespace moe
} // namespace metal_marlin
