#pragma once

#include "metal_device.hpp"
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace metal_marlin {

// -----------------------------------------------------------------------------
// Constants (must match Python metal_dispatch.py)
// -----------------------------------------------------------------------------

constexpr uint32_t kTileM = 64;
constexpr uint32_t kTileN = 64;
constexpr uint32_t kTileK = 32;
constexpr uint32_t kThreadsPerTG = 128;
constexpr uint32_t kFp32AccumKThreshold = 256;
constexpr uint32_t kPadMultiple = 8;
constexpr size_t kAsyncTransferThreshold = 1024 * 1024;  // 1MB

// -----------------------------------------------------------------------------
// GPU Family Detection
// -----------------------------------------------------------------------------

enum class GPUFamily : uint32_t {
    Apple7 = 7,  // M1
    Apple8 = 8,  // M2
    Apple9 = 9,  // M3+
    Unknown = 0
};

/**
 * @brief Get GPU family from Metal device.
 * @param device MTLDevice pointer.
 * @return GPUFamily enum value.
 */
GPUFamily get_gpu_family(void* device);

// -----------------------------------------------------------------------------
// Kernel Selection
// -----------------------------------------------------------------------------

/**
 * @brief Select FP4 GEMM kernel name based on parameters.
 * @param family GPU family.
 * @param K Inner dimension.
 * @return Kernel function name.
 */
std::string select_fp4_kernel(GPUFamily family, uint32_t K);

/**
 * @brief Select FP8 GEMM kernel name.
 * @return Fixed kernel name for FP8.
 */
inline std::string select_fp8_kernel() {
    return "marlin_gemm_fp8_e4m3";
}

/**
 * @brief Select INT2 GEMM kernel name.
 * @return Fixed kernel name for INT2.
 */
inline std::string select_int2_kernel() {
    return "marlin_gemm_int2";
}

// -----------------------------------------------------------------------------
// Buffer Management for GEMM
// -----------------------------------------------------------------------------

/**
 * @brief Padded tensor info for tracking padding.
 */
struct PaddedTensor {
    void* buffer = nullptr;      // MTLBuffer
    uint32_t M = 0;           // Padded rows
    uint32_t N = 0;           // Padded columns
    uint32_t K = 0;           // Packed K dimension
    uint32_t pad_m = 0;        // M padding
    uint32_t pad_n = 0;        // N padding
    bool owns_buffer = false;   // Whether we created the buffer
};

/**
 * @brief Buffer wrapper for constant parameters.
 */
struct ConstantBuffer {
    void* buffer = nullptr;  // MTLBuffer
    uint32_t value = 0;

    ConstantBuffer() = default;
    ~ConstantBuffer();
    ConstantBuffer(const ConstantBuffer&) = delete;
    ConstantBuffer& operator=(const ConstantBuffer&) = delete;
    ConstantBuffer(ConstantBuffer&& other) noexcept;
    ConstantBuffer& operator=(ConstantBuffer&& other) noexcept;
};

/**
 * @brief Manages GEMM buffer resources with automatic cleanup.
 */
class GEMMBufferManager {
public:
    GEMMBufferManager(void* device) : device_(device) {}
    ~GEMMBufferManager();

    // Non-copyable
    GEMMBufferManager(const GEMMBufferManager&) = delete;
    GEMMBufferManager& operator=(const GEMMBufferManager&) = delete;

    // Movable
    GEMMBufferManager(GEMMBufferManager&&) noexcept;
    GEMMBufferManager& operator=(GEMMBufferManager&&) noexcept;

    /**
     * @brief Create shared buffer from bytes.
     */
    [[nodiscard]] void* create_buffer(const void* data, size_t size);

    /**
     * @brief Create shared buffer (zero-initialized).
     */
    [[nodiscard]] void* create_buffer(size_t size);

    /**
     * @brief Create private buffer from bytes.
     */
    [[nodiscard]] void* create_private_buffer(const void* data, size_t size);

    /**
     * @brief Create private buffer (zero-initialized).
     */
    [[nodiscard]] void* create_private_buffer(size_t size);

    /**
     * @brief Create constant parameter buffer.
     */
    [[nodiscard]] ConstantBuffer create_constant_buffer(uint32_t value);

    /**
     * @brief Get buffer pointer from PyTorch MPS tensor.
     * This is a placeholder - actual implementation needs PyTorch interop.
     */
    [[nodiscard]] void* get_mps_buffer(void* tensor_ptr);

    /**
     * @brief Copy data back from GPU.
     */
    void copy_from_gpu(void* dst, void* src_buffer, size_t size);

    /**
     * @brief Release a buffer.
     */
    void release_buffer(void* buffer);

private:
    void* device_ = nullptr;
    std::vector<void*> buffers_;  // Owned buffers to release
};

// -----------------------------------------------------------------------------
// Padding Helpers
// -----------------------------------------------------------------------------

/**
 * @brief Round up to nearest multiple.
 */
constexpr uint32_t round_up(uint32_t value, uint32_t multiple) noexcept {
    return ((value + multiple - 1) / multiple) * multiple;
}

/**
 * @brief Compute max of three values.
 */
constexpr uint32_t max3(uint32_t a, uint32_t b, uint32_t c) noexcept {
    return (a >= b && a >= c) ? a : (b >= c ? b : c);
}

/**
 * @brief Pad packed FP4 tensor to target dimensions.
 * @param packed Input packed tensor [K/8, N].
 * @param k_packs Target K packs (K/8).
 * @param n_target Target N.
 * @param padded_k Output padded K.
 * @param padded_n Output padded N.
 * @param manager Buffer manager for allocations.
 * @return Padded buffer.
 */
PaddedTensor pad_packed_fp4(
    const void* packed,
    uint32_t packed_k,
    uint32_t packed_n,
    uint32_t k_packs_target,
    uint32_t n_target,
    GEMMBufferManager& manager
);

/**
 * @brief Pad scales tensor to target dimensions.
 * @param scales Input scales [K/group_size, N].
 * @param k_groups Target K groups.
 * @param n_target Target N.
 * @param manager Buffer manager.
 * @return Padded scales buffer.
 */
void* pad_scales(
    const void* scales,
    uint32_t scales_k_groups,
    uint32_t scales_n,
    uint32_t k_groups_target,
    uint32_t n_target,
    GEMMBufferManager& manager
);

/**
 * @brief Pad packed tensor along N dimension.
 * @param packed Input packed tensor.
 * @param k_target Target K.
 * @param n_target Target N.
 * @param pack_factor Values per packed element.
 * @param manager Buffer manager.
 * @return Packed tensor info.
 */
PaddedTensor pad_packed_n(
    const void* packed,
    uint32_t packed_k,
    uint32_t packed_n,
    uint32_t k_target,
    uint32_t n_target,
    uint32_t pack_factor,
    GEMMBufferManager& manager
);

// -----------------------------------------------------------------------------
// GEMM Dispatch Functions
// -----------------------------------------------------------------------------

/**
 * @brief Dispatch FP4 quantized GEMM: C = A @ dequant(B).
 *
 * This is the C++ equivalent of dispatch_gemm_fp4 in metal_dispatch.py.
 *
 * Args:
 *   device: Metal device (or nullptr for default).
 *   library: Metal library with compiled kernels.
 *   A_buf: Input activations buffer [M, K], fp16.
 *   B_buf: Packed FP4 weights buffer [(K+pad)//8, N+pad], uint32.
 *   S_buf: Per-group scales buffer [K//group_size, N], fp16.
 *   C_buf: Output buffer [M, N], fp16.
 *   M: Rows of A.
 *   N: Columns of B (output features).
 *   K: Inner dimension.
 *   group_size: Quantization group size.
 *   enable_padding: Whether to enable padding (default: true).
 *
 * Returns:
 *   Padded M and N values (for unpadding).
 */
struct GEMMResult {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t pad_m = 0;
    uint32_t pad_n = 0;
};

GEMMResult dispatch_gemm_fp4(
    void* device,
    void* library,
    void* A_buf,
    void* B_buf,
    void* S_buf,
    void* C_buf,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t group_size = 32,
    bool enable_padding = true
);

/**
 * @brief Dispatch FP8 E4M3 quantized GEMM: C = A @ dequant(B).
 *
 * Equivalent to dispatch_gemm_fp8 in metal_dispatch.py.
 */
GEMMResult dispatch_gemm_fp8(
    void* device,
    void* library,
    void* A_buf,
    void* B_buf,
    void* S_buf,
    void* C_buf,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t group_size = 128,
    bool enable_padding = true
);

/**
 * @brief Dispatch INT2 quantized GEMM: C = A @ dequant(B).
 *
 * Equivalent to dispatch_gemm_int2 in metal_dispatch.py.
 * Note: Metal INT2 kernel may not be implemented yet.
 */
GEMMResult dispatch_gemm_int2(
    void* device,
    void* library,
    void* A_buf,
    void* B_buf,
    void* S_buf,
    void* C_buf,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t group_size = 128,
    bool enable_padding = true
);

/**
 * @brief High-level dispatch with tensor inputs.
 *
 * This handles buffer creation, padding, and dispatch in one call.
 * Used when calling from Python via ctypes/pybind11.
 *
 * Args:
 *   device: Metal device.
 *   library: Metal library.
 *   A_ptr: PyTorch MPS tensor pointer for A.
 *   B_ptr: PyTorch MPS tensor pointer for packed B.
 *   S_ptr: PyTorch MPS tensor pointer for scales.
 *   M, N, K: Dimensions.
 *   format: 0=FP4, 1=FP8, 2=INT2.
 *   group_size: Quantization group size.
 *   enable_padding: Enable padding.
 *
 * Returns:
 *   GEMMResult with padding info.
 */
GEMMResult dispatch_gemm_auto(
    void* device,
    void* library,
    void* A_ptr,
    void* B_ptr,
    void* S_ptr,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int format,  // 0=FP4, 1=FP8, 2=INT2
    uint32_t group_size = 128,
    bool enable_padding = true
);

// -----------------------------------------------------------------------------
// Environment Configuration
// -----------------------------------------------------------------------------

/**
 * @brief Check if GEMM padding is enabled via environment.
 *
 * Checks METAL_MARLIN_GEMM_PADDING environment variable.
 * Default: enabled.
 */
bool is_padding_enabled();

/**
 * @brief Set padding enabled state.
 */
void set_padding_enabled(bool enabled);

// -----------------------------------------------------------------------------
// Error Handling
// -----------------------------------------------------------------------------

/**
 * @brief Exception for GEMM dispatch errors.
 */
class GEMMDispatchError : public std::runtime_error {
public:
    explicit GEMMDispatchError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @brief Exception for dimension mismatch errors.
 */
class DimensionError : public GEMMDispatchError {
public:
    DimensionError(const std::string& msg) : GEMMDispatchError(msg) {}
};

// -----------------------------------------------------------------------------
// Mixed-BPW MoE batch dispatch
// -----------------------------------------------------------------------------

/**
 * @brief Runtime configuration for mixed bit-width MoE dispatch.
 */
struct MoEConfig {
    uint32_t hidden_dim = 0;
    uint32_t intermediate_dim = 0;
    uint32_t num_experts = 0;
    uint32_t top_k = 2;
    uint32_t max_experts_per_batch = 64;
    uint32_t command_buffers_per_batch_size = 2;
    uint32_t max_inflight_submissions = 2;
    uint32_t threadgroup_size_x = kThreadsPerTG;
    std::vector<uint32_t> common_batch_sizes = {32, 64, 128, 256};
    bool use_indirect_command_buffers = true;
    bool overlap_cpu_encoding = true;
    bool wait_for_completion = true;
    std::string kernel_name = "moe_trellis_swiglu_grouped";
    std::string metallib_path;
};

/**
 * @brief Single grouped mixed-BPW dispatch batch.
 */
struct MixedBPWBatchPlan {
    int bit_width = 0;
    std::vector<uint32_t> expert_ids;
    std::vector<uint32_t> expert_token_counts;
    uint32_t token_count = 0;
};

/**
 * @brief Dispatcher statistics for observability.
 */
struct MixedBPWDispatchStats {
    uint64_t queued_experts = 0;
    uint64_t routed_experts = 0;
    uint64_t grouped_batches = 0;
    uint64_t command_buffer_submissions = 0;
    uint64_t indirect_command_batches = 0;
};

/**
 * @brief Group experts by bit-width and coordinate batch submission.
 *
 * This class is a CPU-side scheduler that:
 * 1. Accepts per-expert bit-width arrays.
 * 2. Builds grouped expert batches by bit-width.
 * 3. Tracks pre-allocated command-buffer slots for common batch sizes.
 */
class BatchDispatchMixedBPW {
public:
    using BatchList = std::vector<MixedBPWBatchPlan>;

    BatchDispatchMixedBPW() = default;
    ~BatchDispatchMixedBPW() = default;

    void reset();
    void add_expert_bits(const std::vector<int>& expert_bits);
    void set_active_experts(const std::vector<uint32_t>& active_expert_ids);
    void clear_active_experts();

    void reserve_command_buffers(
        const std::vector<uint32_t>& common_batch_sizes,
        uint32_t command_buffers_per_size = 2
    );

    [[nodiscard]] BatchList build_batches(
        uint32_t max_experts_per_batch = 0
    ) const;
    [[nodiscard]] BatchList build_batches_for_routing(
        const int32_t* expert_indices,
        uint32_t num_tokens,
        uint32_t top_k,
        uint32_t max_experts_per_batch = 0
    ) const;

    [[nodiscard]] bool try_acquire_command_buffer_slot(uint32_t batch_size);
    void release_command_buffer_slot(uint32_t batch_size);
    [[nodiscard]] uint32_t command_buffer_slot_key(uint32_t batch_size) const;

    void note_submission(bool used_indirect_command_buffer);

    [[nodiscard]] const MixedBPWDispatchStats& stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] MixedBPWDispatchStats stats_snapshot() const;

private:
    [[nodiscard]] uint32_t resolve_slot_key(uint32_t batch_size) const;

    std::vector<int> expert_bits_;
    std::vector<uint32_t> active_expert_ids_;
    std::map<uint32_t, uint32_t> configured_command_buffer_slots_;
    std::map<uint32_t, uint32_t> available_command_buffer_slots_;
    bool use_active_expert_filter_ = false;
    mutable MixedBPWDispatchStats stats_;
    mutable std::mutex mutex_;
};

using MixedBPWEncodeCallback = std::function<void(const MixedBPWBatchPlan&)>;
using MixedBPWSubmitCallback = std::function<void(const MixedBPWBatchPlan&)>;

/**
 * @brief Execute grouped mixed-BPW batches with optional CPU encode/GPU submit overlap.
 *
 * The encode callback runs on the caller thread. The submit callback can be
 * pipelined asynchronously so CPU encoding of batch N+1 overlaps GPU execution
 * of batch N.
 *
 * If routing inputs are provided (`expert_indices`, `num_tokens`, `top_k`),
 * batches are built from active routed experts and per-expert token counts.
 * Passing null/zero routing inputs falls back to static expert grouping.
 */
void execute_mixed_bpw_pipeline(
    BatchDispatchMixedBPW& dispatcher,
    const MoEConfig& config,
    const int32_t* expert_indices,
    uint32_t num_tokens,
    uint32_t top_k,
    MixedBPWEncodeCallback encode_callback,
    MixedBPWSubmitCallback submit_callback
);

/**
 * @brief High-level mixed bit-width MoE dispatch.
 * 
 * This function orchestrates the entire MoE dispatch process, including
 * expert grouping, command buffer management, and pipeline execution.
 * 
 * @param hidden_states Input/output activations [num_tokens, hidden_dim].
 * @param expert_weights_packed Vector of packed weight buffers.
 * @param expert_weight_sizes Sizes of each weight buffer in bytes.
 * @param expert_bits Bit-width for each expert.
 * @param expert_scales Vector of scale buffers (fp16).
 * @param expert_scale_sizes Sizes of each scale buffer in bytes.
 * @param expert_indices Expert indices per token [num_tokens, top_k].
 * @param expert_probs Routing probabilities per token [num_tokens, top_k].
 * @param num_tokens Number of tokens in batch.
 * @param top_k Number of experts per token.
 * @param config MoE configuration and optimizations.
 */
void dispatch_mixed_bpw_moe(
    float* hidden_states,
    const std::vector<const void*>& expert_weights_packed,
    const std::vector<size_t>& expert_weight_sizes,
    const std::vector<int>& expert_bits,
    const std::vector<const void*>& expert_scales,
    const std::vector<size_t>& expert_scale_sizes,
    const int32_t* expert_indices,
    const float* expert_probs,
    uint32_t num_tokens,
    uint32_t top_k,
    const MoEConfig& config
);

} // namespace metal_marlin
