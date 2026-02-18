/**
 * @file weights_ops.cpp
 * @brief C++ implementation of 2:4 structured sparsity pruning operations.
 * 
 * Implements NVIDIA's 2:4 fine-grained structured sparsity pattern:
 * for every contiguous block of 4 elements along K, exactly 2 are kept
 * (the largest by magnitude) and 2 are zeroed.
 */

#include "weights_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace metal_marlin {

// ============================================================================
// FP16 Conversion Utilities
// ============================================================================

float fp16_to_fp32(uint16_t h) {
    // Extract components
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        // Zero or subnormal
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal: value = (-1)^sign * 2^-14 * (mant / 1024)
        float val = static_cast<float>(mant) / 1024.0f * std::pow(2.0f, -14.0f);
        return sign ? -val : val;
    } else if (exp == 0x1F) {
        // Infinity or NaN
        uint32_t f32 = (sign << 31) | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &f32, sizeof(result));
        return result;
    }
    
    // Normal: value = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t f32;
    std::memcpy(&f32, &f, sizeof(f32));
    
    uint32_t sign = (f32 >> 31) & 0x1;
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;
    
    // Handle special cases
    if (exp == 0 && mant == 0) {
        return static_cast<uint16_t>(sign << 15);  // Zero
    }
    if (exp == 0xFF) {
        // Infinity or NaN
        uint16_t h = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant >> 13));
        return h;
    }
    
    // Normalized float to half
    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
    
    if (new_exp >= 31) {
        // Overflow to infinity
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    } else if (new_exp <= 0) {
        // Underflow to zero or subnormal
        if (new_exp < -10) {
            return static_cast<uint16_t>(sign << 15);  // Too small, round to zero
        }
        // Subnormal
        mant = (mant | 0x800000) >> (1 - new_exp);
        return static_cast<uint16_t>((sign << 15) | (mant >> 13));
    }
    
    // Normal case
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(new_exp) << 10) | (mant >> 13));
}

float fp16_abs(uint16_t h) {
    // Clear sign bit and convert to float
    return fp16_to_fp32(h & 0x7FFF);
}

// ============================================================================
// 2:4 Structured Sparsity Pruning
// ============================================================================

Prune2to4Result prune_to_2_4(
    const uint16_t* weights,
    size_t K,
    size_t N
) {
    Prune2to4Result result;
    
    // Validate inputs
    if (K % 4 != 0) {
        result.error_msg = "K=" + std::to_string(K) + " must be divisible by 4 for 2:4 sparsity";
        return result;
    }
    if (weights == nullptr || K == 0 || N == 0) {
        result.error_msg = "Invalid input: null pointer or zero dimensions";
        return result;
    }
    
    const size_t num_blocks = K / 4;
    result.K = K;
    result.N = N;
    result.sparse_weights.resize(num_blocks * 2 * N);
    result.metadata.resize(num_blocks * N);
    
    // Process each column independently
    for (size_t n = 0; n < N; ++n) {
        for (size_t block = 0; block < num_blocks; ++block) {
            // Load 4 values from this block
            std::array<uint16_t, 4> vals;
            std::array<float, 4> abs_vals;
            
            for (int i = 0; i < 4; ++i) {
                size_t idx = (block * 4 + i) * N + n;
                vals[i] = weights[idx];
                abs_vals[i] = fp16_abs(vals[i]);
            }
            
            // Find top-2 indices by magnitude
            // Simple sorting network for 4 elements
            std::array<int, 4> indices = {0, 1, 2, 3};
            
            // Sort by descending absolute value (bubble sort for 4 elements)
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3 - i; ++j) {
                    if (abs_vals[indices[j]] < abs_vals[indices[j + 1]]) {
                        std::swap(indices[j], indices[j + 1]);
                    }
                }
            }
            
            // Top 2 are indices[0] and indices[1]
            int idx0 = indices[0];
            int idx1 = indices[1];
            
            // Ensure canonical ordering (idx0 < idx1)
            if (idx0 > idx1) {
                std::swap(idx0, idx1);
            }
            
            // Store kept values in packed format
            size_t sparse_idx = (block * 2) * N + n;
            result.sparse_weights[sparse_idx] = vals[idx0];
            result.sparse_weights[sparse_idx + N] = vals[idx1];
            
            // Pack metadata: bits [1:0] = first position, bits [3:2] = second position
            uint8_t meta = static_cast<uint8_t>((idx0 & 0x3) | ((idx1 & 0x3) << 2));
            result.metadata[block * N + n] = meta;
        }
    }
    
    result.success = true;
    return result;
}

Unprune2to4Result unprune_2_4(
    const uint16_t* sparse_weights,
    const uint8_t* metadata,
    size_t half_K,
    size_t N
) {
    Unprune2to4Result result;
    
    if (sparse_weights == nullptr || metadata == nullptr || half_K == 0 || N == 0) {
        result.error_msg = "Invalid input: null pointer or zero dimensions";
        return result;
    }
    
    if (half_K % 2 != 0) {
        result.error_msg = "half_K must be even (sparse_weights rows must be divisible by 2)";
        return result;
    }
    
    const size_t num_blocks = half_K / 2;
    const size_t K = num_blocks * 4;
    result.K = K;
    result.N = N;
    result.dense_weights.resize(K * N, 0);  // Initialize to zeros
    
    // Process each column
    for (size_t n = 0; n < N; ++n) {
        for (size_t block = 0; block < num_blocks; ++block) {
            // Extract position indices from metadata
            uint8_t meta = metadata[block * N + n];
            int keep_first = static_cast<int>(meta & 0x3);
            int keep_second = static_cast<int>((meta >> 2) & 0x3);
            
            // Extract kept values
            size_t sparse_idx = (block * 2) * N + n;
            uint16_t val0 = sparse_weights[sparse_idx];
            uint16_t val1 = sparse_weights[sparse_idx + N];
            
            // Scatter back to dense positions
            size_t block_base = block * 4;
            result.dense_weights[(block_base + keep_first) * N + n] = val0;
            result.dense_weights[(block_base + keep_second) * N + n] = val1;
        }
    }
    
    result.success = true;
    return result;
}

PruningMetrics measure_pruning_loss(
    const uint16_t* original,
    const uint16_t* pruned,
    size_t K,
    size_t N
) {
    PruningMetrics metrics;
    
    if (original == nullptr || pruned == nullptr || K == 0 || N == 0) {
        return metrics;  // Return zeroed metrics on invalid input
    }
    
    const size_t total_elements = K * N;
    double mse_sum = 0.0;
    double orig_norm_sq = 0.0;
    float max_abs_err = 0.0f;
    size_t num_zeros = 0;
    
    for (size_t i = 0; i < total_elements; ++i) {
        float orig = fp16_to_fp32(original[i]);
        float prun = fp16_to_fp32(pruned[i]);
        
        float diff = orig - prun;
        float abs_diff = std::abs(diff);
        
        mse_sum += static_cast<double>(diff * diff);
        orig_norm_sq += static_cast<double>(orig * orig);
        max_abs_err = std::max(max_abs_err, abs_diff);
        
        if (prun == 0.0f) {
            num_zeros++;
        }
    }
    
    metrics.mse = static_cast<float>(mse_sum / static_cast<double>(total_elements));
    metrics.rmse = std::sqrt(metrics.mse);
    
    double orig_norm = std::sqrt(orig_norm_sq);
    if (orig_norm > 0.0) {
        double diff_norm = std::sqrt(mse_sum * static_cast<double>(total_elements));
        metrics.relative_error = static_cast<float>(diff_norm / orig_norm);
    } else {
        metrics.relative_error = std::numeric_limits<float>::infinity();
    }
    
    metrics.sparsity = static_cast<float>(num_zeros) / static_cast<float>(total_elements);
    metrics.max_abs_error = max_abs_err;
    
    return metrics;
}

// ============================================================================
// Batch Processing
// ============================================================================

std::vector<Prune2to4Result> batch_prune_to_2_4(
    const uint16_t* const* weights_list,
    const size_t* K_list,
    const size_t* N_list,
    size_t count
) {
    std::vector<Prune2to4Result> results;
    results.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        results.push_back(prune_to_2_4(weights_list[i], K_list[i], N_list[i]));
    }
    
    return results;
}

} // namespace metal_marlin
