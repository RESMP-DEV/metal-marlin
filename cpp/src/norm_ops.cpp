/**
 * @file norm_ops.cpp
 * @brief Implementation of norm operations (LayerNorm, RMSNorm)
 */

#include "norm_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace metal_marlin {

// ============================================================================
// Utility Functions
// ============================================================================

namespace norm_utils {

float compute_mean(const float* data, size_t size) {
    if (size == 0) return 0.0f;
    
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    return static_cast<float>(sum / size);
}

float compute_variance(const float* data, size_t size, float mean) {
    if (size == 0) return 0.0f;
    
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return static_cast<float>(sum_sq_diff / size);
}

float compute_rms(const float* data, size_t size) {
    if (size == 0) return 0.0f;
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += data[i] * data[i];
    }
    return static_cast<float>(std::sqrt(sum_sq / size));
}

bool is_hidden_dim_supported(uint32_t hidden_dim) {
    // Supported hidden dimensions: common LLM sizes
    const uint32_t supported_dims[] = {
        768,    // BERT-base, GPT-2 small
        1024,   // BERT-large, GPT-2 medium
        1280,   // GPT-2 large
        1536,   // Custom
        1600,   // GPT-2 XL
        2048,   // Small LLMs
        2560,   // Medium LLMs
        4096,   // Llama 7B, Mistral 7B
        5120,   // Llama 13B
        6144,   // Custom
        8192,   // Llama 70B, Large models
        12288,  // GLM-4 9B
        16384,  // Very large models
    };
    
    for (uint32_t dim : supported_dims) {
        if (hidden_dim == dim) return true;
    }
    return false;
}

uint32_t get_recommended_threadgroup_size(uint32_t hidden_dim) {
    // Use 256 threads (8 simdgroups) for most cases
    // For small hidden dims, use fewer threads
    if (hidden_dim <= 1024) return 128;
    if (hidden_dim <= 2048) return 128;
    return 256;
}

} // namespace norm_utils

// ============================================================================
// LayerNorm Implementation
// ============================================================================

class LayerNormOp::Impl {
public:
    explicit Impl(const LayerNormConfig& config) : config_(config) {
        if (config_.num_tokens == 0 || config_.hidden_dim == 0) {
            throw std::invalid_argument("num_tokens and hidden_dim must be > 0");
        }
        if (config_.eps <= 0.0f) {
            throw std::invalid_argument("eps must be positive");
        }
    }

    NormResult forward(const std::vector<float>& input,
                       const std::vector<float>& gamma,
                       const std::vector<float>& beta) {
        NormResult result;
        
        // Validate input sizes
        const size_t expected_size = static_cast<size_t>(config_.num_tokens) * config_.hidden_dim;
        if (input.size() != expected_size) {
            result.error_msg = "Input size mismatch: expected " + std::to_string(expected_size) +
                               ", got " + std::to_string(input.size());
            return result;
        }
        if (gamma.size() != config_.hidden_dim) {
            result.error_msg = "Gamma size mismatch: expected " + std::to_string(config_.hidden_dim) +
                               ", got " + std::to_string(gamma.size());
            return result;
        }
        if (!beta.empty() && beta.size() != config_.hidden_dim) {
            result.error_msg = "Beta size mismatch: expected " + std::to_string(config_.hidden_dim) +
                               ", got " + std::to_string(beta.size());
            return result;
        }

        // Allocate output
        result.output.resize(expected_size);
        
        // Process each token (row)
        std::vector<float> row(config_.hidden_dim);
        
        for (uint32_t token = 0; token < config_.num_tokens; ++token) {
            // Copy row
            const size_t offset = static_cast<size_t>(token) * config_.hidden_dim;
            std::memcpy(row.data(), input.data() + offset, config_.hidden_dim * sizeof(float));
            
            // Compute mean
            float mean = norm_utils::compute_mean(row.data(), config_.hidden_dim);
            
            // Compute variance
            float variance = norm_utils::compute_variance(row.data(), config_.hidden_dim, mean);
            
            // Compute inverse standard deviation
            float inv_std = 1.0f / std::sqrt(variance + config_.eps);
            
            // Normalize, scale, and shift
            for (uint32_t i = 0; i < config_.hidden_dim; ++i) {
                float normalized = (row[i] - mean) * inv_std;
                float g = gamma[i];
                float b = beta.empty() ? 0.0f : beta[i];
                result.output[offset + i] = normalized * g + b;
            }
        }
        
        result.success = true;
        return result;
    }

private:
    LayerNormConfig config_;
};

LayerNormOp::LayerNormOp(const LayerNormConfig& config) 
    : config_(config), impl_(std::make_unique<Impl>(config)) {}

LayerNormOp::~LayerNormOp() = default;

NormResult LayerNormOp::forward(const std::vector<float>& input,
                                const std::vector<float>& gamma,
                                const std::vector<float>& beta) {
    return impl_->forward(input, gamma, beta);
}

// ============================================================================
// RMSNorm Implementation
// ============================================================================

class RMSNormOp::Impl {
public:
    explicit Impl(const RMSNormConfig& config) : config_(config) {
        if (config_.num_tokens == 0 || config_.hidden_dim == 0) {
            throw std::invalid_argument("num_tokens and hidden_dim must be > 0");
        }
        if (config_.eps <= 0.0f) {
            throw std::invalid_argument("eps must be positive");
        }
    }

    NormResult forward(const std::vector<float>& input,
                       const std::vector<float>& gamma) {
        NormResult result;
        
        // Validate input sizes
        const size_t expected_size = static_cast<size_t>(config_.num_tokens) * config_.hidden_dim;
        if (input.size() != expected_size) {
            result.error_msg = "Input size mismatch: expected " + std::to_string(expected_size) +
                               ", got " + std::to_string(input.size());
            return result;
        }
        if (gamma.size() != config_.hidden_dim) {
            result.error_msg = "Gamma size mismatch: expected " + std::to_string(config_.hidden_dim) +
                               ", got " + std::to_string(gamma.size());
            return result;
        }

        // Allocate output
        result.output.resize(expected_size);
        
        // Process each token (row)
        std::vector<float> row(config_.hidden_dim);
        
        for (uint32_t token = 0; token < config_.num_tokens; ++token) {
            // Copy row
            const size_t offset = static_cast<size_t>(token) * config_.hidden_dim;
            std::memcpy(row.data(), input.data() + offset, config_.hidden_dim * sizeof(float));
            
            // Compute RMS
            float rms = norm_utils::compute_rms(row.data(), config_.hidden_dim);
            float rms_inv = 1.0f / (rms + config_.eps);
            
            // Normalize and scale
            for (uint32_t i = 0; i < config_.hidden_dim; ++i) {
                result.output[offset + i] = row[i] * rms_inv * gamma[i];
            }
        }
        
        result.success = true;
        return result;
    }

    NormResult forward_fused(const std::vector<float>& input,
                             const std::vector<float>& residual,
                             const std::vector<float>& gamma) {
        NormResult result;
        
        // Validate input sizes
        const size_t expected_size = static_cast<size_t>(config_.num_tokens) * config_.hidden_dim;
        if (input.size() != expected_size) {
            result.error_msg = "Input size mismatch: expected " + std::to_string(expected_size) +
                               ", got " + std::to_string(input.size());
            return result;
        }
        if (residual.size() != expected_size) {
            result.error_msg = "Residual size mismatch: expected " + std::to_string(expected_size) +
                               ", got " + std::to_string(residual.size());
            return result;
        }
        if (gamma.size() != config_.hidden_dim) {
            result.error_msg = "Gamma size mismatch: expected " + std::to_string(config_.hidden_dim) +
                               ", got " + std::to_string(gamma.size());
            return result;
        }

        // Allocate outputs
        result.output.resize(expected_size);
        result.residual.resize(expected_size);
        
        // Process each token (row)
        std::vector<float> row(config_.hidden_dim);
        
        for (uint32_t token = 0; token < config_.num_tokens; ++token) {
            // Compute residual add
            const size_t offset = static_cast<size_t>(token) * config_.hidden_dim;
            for (uint32_t i = 0; i < config_.hidden_dim; ++i) {
                row[i] = input[offset + i] + residual[offset + i];
                result.residual[offset + i] = row[i];
            }
            
            // Compute RMS
            float rms = norm_utils::compute_rms(row.data(), config_.hidden_dim);
            float rms_inv = 1.0f / (rms + config_.eps);
            
            // Normalize and scale
            for (uint32_t i = 0; i < config_.hidden_dim; ++i) {
                result.output[offset + i] = row[i] * rms_inv * gamma[i];
            }
        }
        
        result.success = true;
        return result;
    }

private:
    RMSNormConfig config_;
};

RMSNormOp::RMSNormOp(const RMSNormConfig& config) 
    : config_(config), impl_(std::make_unique<Impl>(config)) {}

RMSNormOp::~RMSNormOp() = default;

NormResult RMSNormOp::forward(const std::vector<float>& input,
                              const std::vector<float>& gamma) {
    return impl_->forward(input, gamma);
}

NormResult RMSNormOp::forward_fused(const std::vector<float>& input,
                                    const std::vector<float>& residual,
                                    const std::vector<float>& gamma) {
    return impl_->forward_fused(input, residual, gamma);
}

} // namespace metal_marlin
