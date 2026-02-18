/**
 * @file norm_ops.hpp
 * @brief C++ norm operations (LayerNorm, RMSNorm) for Metal Marlin
 *
 * Provides C++ wrappers for norm operations that can be called from Python
 * via the nanobind bindings. These operations dispatch to optimized Metal
 * kernels for GPU acceleration.
 */

#ifndef METAL_MARLIN_NORM_OPS_HPP
#define METAL_MARLIN_NORM_OPS_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace metal_marlin {

/**
 * @brief Configuration for LayerNorm operation
 */
struct LayerNormConfig {
    uint32_t num_tokens = 0;      ///< Number of tokens (batch * seq_len)
    uint32_t hidden_dim = 0;      ///< Hidden dimension size
    float eps = 1e-5f;            ///< Epsilon for numerical stability
    bool use_fused = false;       ///< Whether to use fused kernels if available
};

/**
 * @brief Configuration for RMSNorm operation
 */
struct RMSNormConfig {
    uint32_t num_tokens = 0;      ///< Number of tokens (batch * seq_len)
    uint32_t hidden_dim = 0;      ///< Hidden dimension size
    float eps = 1e-6f;            ///< Epsilon for numerical stability
    bool use_fused = false;       ///< Whether to use fused residual kernels
};

/**
 * @brief Result buffer for norm operations
 */
struct NormResult {
    std::vector<float> output;    ///< Normalized output data
    std::vector<float> residual;  ///< Residual output (for fused ops)
    bool success = false;         ///< Whether operation succeeded
    std::string error_msg;        ///< Error message if failed
};

/**
 * @brief C++ interface for LayerNorm operations
 *
 * Implements: output = (x - mean) / sqrt(var + eps) * gamma + beta
 */
class LayerNormOp {
public:
    explicit LayerNormOp(const LayerNormConfig& config);
    ~LayerNormOp();

    /**
     * @brief Perform LayerNorm on input data
     *
     * @param input Input tensor [num_tokens, hidden_dim]
     * @param gamma Scale weights [hidden_dim]
     * @param beta Bias weights [hidden_dim], can be empty for no bias
     * @return NormResult with normalized output
     */
    NormResult forward(const std::vector<float>& input,
                       const std::vector<float>& gamma,
                       const std::vector<float>& beta);

    /**
     * @brief Get the configuration
     */
    const LayerNormConfig& config() const { return config_; }

private:
    LayerNormConfig config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief C++ interface for RMSNorm operations
 *
 * Implements: output = x / sqrt(mean(x^2) + eps) * gamma
 */
class RMSNormOp {
public:
    explicit RMSNormOp(const RMSNormConfig& config);
    ~RMSNormOp();

    /**
     * @brief Perform RMSNorm on input data
     *
     * @param input Input tensor [num_tokens, hidden_dim]
     * @param gamma Scale weights [hidden_dim]
     * @return NormResult with normalized output
     */
    NormResult forward(const std::vector<float>& input,
                       const std::vector<float>& gamma);

    /**
     * @brief Perform fused residual add + RMSNorm
     *
     * @param input Input tensor [num_tokens, hidden_dim]
     * @param residual Residual tensor [num_tokens, hidden_dim]
     * @param gamma Scale weights [hidden_dim]
     * @return NormResult with normalized output and residual_out
     */
    NormResult forward_fused(const std::vector<float>& input,
                             const std::vector<float>& residual,
                             const std::vector<float>& gamma);

    /**
     * @brief Get the configuration
     */
    const RMSNormConfig& config() const { return config_; }

private:
    RMSNormConfig config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Utility functions for norm operations
 */
namespace norm_utils {

/**
 * @brief Compute mean of a vector
 */
float compute_mean(const float* data, size_t size);

/**
 * @brief Compute variance of a vector (with precomputed mean)
 */
float compute_variance(const float* data, size_t size, float mean);

/**
 * @brief Compute RMS (root mean square) of a vector
 */
float compute_rms(const float* data, size_t size);

/**
 * @brief Check if hidden dimension is supported by optimized kernels
 */
bool is_hidden_dim_supported(uint32_t hidden_dim);

/**
 * @brief Get recommended threadgroup size for given hidden dimension
 */
uint32_t get_recommended_threadgroup_size(uint32_t hidden_dim);

} // namespace norm_utils

} // namespace metal_marlin

#endif // METAL_MARLIN_NORM_OPS_HPP
