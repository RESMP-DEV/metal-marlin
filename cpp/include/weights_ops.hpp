#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <tuple>

namespace metal_marlin {

// ============================================================================
// 2:4 Structured Sparsity Pruning
// ============================================================================

/// Pruning result containing sparse weights and metadata
struct Prune2to4Result {
    std::vector<uint16_t> sparse_weights;  // FP16 packed kept values [K/2, N]
    std::vector<uint8_t> metadata;         // Position indices [K/4, N]
    size_t K = 0;                          // Original rows
    size_t N = 0;                          // Columns
    
    bool success = false;
    std::string error_msg;
};

/// Unpruning result containing reconstructed dense weights
struct Unprune2to4Result {
    std::vector<uint16_t> dense_weights;   // FP16 reconstructed [K, N]
    size_t K = 0;
    size_t N = 0;
    
    bool success = false;
    std::string error_msg;
};

/// Pruning error metrics
struct PruningMetrics {
    float mse = 0.0f;
    float rmse = 0.0f;
    float relative_error = 0.0f;
    float sparsity = 0.0f;
    float max_abs_error = 0.0f;
};

// ----------------------------------------------------------------------------
// Core Pruning Operations
// ----------------------------------------------------------------------------

/**
 * @brief Prune weights to 2:4 structured sparsity.
 * 
 * For each contiguous block of 4 elements along the K dimension,
 * keeps the 2 largest-magnitude values and discards the rest.
 * 
 * @param weights Input weight matrix [K, N] as FP16 (uint16_t bits)
 * @param K Number of rows (must be divisible by 4)
 * @param N Number of columns
 * @return Prune2to4Result containing sparse weights and metadata
 */
Prune2to4Result prune_to_2_4(
    const uint16_t* weights,
    size_t K,
    size_t N
);

/**
 * @brief Reconstruct a dense weight matrix from 2:4 sparse format.
 * 
 * Inverse of prune_to_2_4: places kept values back at their original
 * positions and fills the pruned positions with zeros.
 * 
 * @param sparse_weights Packed kept values [K/2, N] as FP16
 * @param metadata Position indices [K/4, N]
 * @param half_K K/2 (sparse weight rows)
 * @param N Columns
 * @return Unprune2to4Result containing reconstructed dense weights
 */
Unprune2to4Result unprune_2_4(
    const uint16_t* sparse_weights,
    const uint8_t* metadata,
    size_t half_K,
    size_t N
);

/**
 * @brief Measure information loss from 2:4 structured pruning.
 * 
 * @param original Original dense weight matrix [K, N] as FP16
 * @param pruned Reconstructed weight matrix [K, N] as FP16 (with zeros)
 * @param K Number of rows
 * @param N Number of columns
 * @return PruningMetrics containing error measurements
 */
PruningMetrics measure_pruning_loss(
    const uint16_t* original,
    const uint16_t* pruned,
    size_t K,
    size_t N
);

// ----------------------------------------------------------------------------
// Utility Functions
// ----------------------------------------------------------------------------

/**
 * @brief Convert FP16 bits to float32.
 */
float fp16_to_fp32(uint16_t h);

/**
 * @brief Convert float32 to FP16 bits.
 */
uint16_t fp32_to_fp16(float f);

/**
 * @brief Compute absolute value of FP16 value (as FP32 for comparison).
 */
float fp16_abs(uint16_t h);

// ----------------------------------------------------------------------------
// Batch Processing
// ----------------------------------------------------------------------------

/**
 * @brief Batch prune multiple weight matrices.
 * 
 * @param weights_list List of weight matrix pointers
 * @param K_list List of row counts (each must be divisible by 4)
 * @param N_list List of column counts
 * @param count Number of matrices
 * @return Vector of pruning results
 */
std::vector<Prune2to4Result> batch_prune_to_2_4(
    const uint16_t* const* weights_list,
    const size_t* K_list,
    const size_t* N_list,
    size_t count
);

} // namespace metal_marlin
