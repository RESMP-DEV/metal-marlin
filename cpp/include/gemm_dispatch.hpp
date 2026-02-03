#pragma once

#include "metal_device.hpp"
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
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

} // namespace metal_marlin
