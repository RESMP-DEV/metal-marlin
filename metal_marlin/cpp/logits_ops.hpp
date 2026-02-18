#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace metal_marlin {

// -----------------------------------------------------------------------------
// Optimized Logits Operations
// -----------------------------------------------------------------------------

/**
 * @brief Apply temperature scaling to logits in-place.
 *
 * out[i] = logits[i] / temperature
 *
 * Optimized with SIMD vectorization and cache-friendly access patterns.
 *
 * @param logits Input/output logits array [batch_size, vocab_size].
 * @param batch_size Number of sequences in batch.
 * @param vocab_size Size of vocabulary.
 * @param temperature Temperature value (> 0).
 */
void temperature_scale_f32(
    float* __restrict logits,
    size_t batch_size,
    size_t vocab_size,
    float temperature
);

/**
 * @brief Apply temperature scaling to fp16 logits in-place.
 *
 * @param logits Input/output logits array [batch_size, vocab_size] as uint16_t.
 * @param batch_size Number of sequences in batch.
 * @param vocab_size Size of vocabulary.
 * @param temperature Temperature value (> 0).
 */
void temperature_scale_f16(
    uint16_t* __restrict logits,
    size_t batch_size,
    size_t vocab_size,
    float temperature
);

/**
 * @brief Compute softmax on logits in-place.
 *
 * out[i] = exp(logits[i] - max) / sum(exp(logits[j] - max))
 *
 * Uses numerically stable algorithm (subtract max before exp).
 * Optimized with SIMD vectorization and streaming stores.
 *
 * @param logits Input/output logits array [batch_size, vocab_size].
 * @param batch_size Number of sequences in batch.
 * @param vocab_size Size of vocabulary.
 */
void softmax_f32(
    float* __restrict logits,
    size_t batch_size,
    size_t vocab_size
);

/**
 * @brief Compute softmax on fp16 logits, output as fp32.
 *
 * @param logits Input logits array [batch_size, vocab_size] as uint16_t.
 * @param out Output probabilities [batch_size, vocab_size] as float.
 * @param batch_size Number of sequences in batch.
 * @param vocab_size Size of vocabulary.
 */
void softmax_f16_to_f32(
    const uint16_t* __restrict logits,
    float* __restrict out,
    size_t batch_size,
    size_t vocab_size
);

// -----------------------------------------------------------------------------
// Top-K Selection
// -----------------------------------------------------------------------------

/**
 * @brief Select top-k elements using optimized partial sorting.
 *
 * Uses a min-heap of size k for O(vocab * log(k)) complexity.
 * Optimized for small k (typical MoE: k=2-8).
 *
 * @param values Input values [vocab_size].
 * @param vocab_size Size of vocabulary.
 * @param k Number of top elements to select.
 * @param out_values Output top-k values [k].
 * @param out_indices Output top-k indices [k].
 */
void topk_select_f32(
    const float* __restrict values,
    size_t vocab_size,
    size_t k,
    float* __restrict out_values,
    uint32_t* __restrict out_indices
);

/**
 * @brief Batch top-k selection for multiple sequences.
 *
 * @param values Input values [batch_size, vocab_size].
 * @param batch_size Number of sequences.
 * @param vocab_size Size of vocabulary.
 * @param k Number of top elements to select.
 * @param out_values Output top-k values [batch_size, k].
 * @param out_indices Output top-k indices [batch_size, k].
 */
void topk_select_batch_f32(
    const float* __restrict values,
    size_t batch_size,
    size_t vocab_size,
    size_t k,
    float* __restrict out_values,
    uint32_t* __restrict out_indices
);

// -----------------------------------------------------------------------------
// Fused Operations
// -----------------------------------------------------------------------------

/**
 * @brief Fused temperature scaling + softmax.
 *
 * Computes: softmax(logits / temperature)
 *
 * More efficient than separate operations due to fused memory access.
 *
 * @param logits Input/output logits array [batch_size, vocab_size].
 * @param batch_size Number of sequences in batch.
 * @param vocab_size Size of vocabulary.
 * @param temperature Temperature value (> 0).
 */
void temperature_softmax_f32(
    float* __restrict logits,
    size_t batch_size,
    size_t vocab_size,
    float temperature
);

/**
 * @brief Fused softmax + top-k for MoE routing.
 *
 * This is the main optimization target for MoE models.
 * Computes softmax then selects top-k experts in a single pass
 * without materializing full softmax output.
 *
 * @param logits Input logits [batch_size, num_experts].
 * @param batch_size Number of tokens.
 * @param num_experts Number of experts.
 * @param top_k Number of experts to select.
 * @param out_weights Output routing weights [batch_size, top_k].
 * @param out_indices Output expert indices [batch_size, top_k].
 */
void softmax_topk_f32(
    const float* __restrict logits,
    size_t batch_size,
    size_t num_experts,
    size_t top_k,
    float* __restrict out_weights,
    uint32_t* __restrict out_indices
);

/**
 * @brief Fused temperature + softmax + top-k for MoE routing.
 *
 * Most efficient path for MoE decode: single-pass through logits
 * with streaming stores and no intermediate allocations.
 *
 * @param logits Input logits [batch_size, num_experts].
 * @param batch_size Number of tokens.
 * @param num_experts Number of experts.
 * @param top_k Number of experts to select.
 * @param temperature Temperature for scaling (> 0).
 * @param out_weights Output routing weights [batch_size, top_k].
 * @param out_indices Output expert indices [batch_size, top_k].
 */
void temperature_softmax_topk_f32(
    const float* __restrict logits,
    size_t batch_size,
    size_t num_experts,
    size_t top_k,
    float temperature,
    float* __restrict out_weights,
    uint32_t* __restrict out_indices
);

// -----------------------------------------------------------------------------
// Repetition Penalty
// -----------------------------------------------------------------------------

/**
 * @brief Apply repetition penalty to logits in-place.
 *
 * For each token in generated_ids:
 *   if logits[token] > 0: logits[token] /= penalty
 *   else: logits[token] *= penalty
 *
 * @param logits Input/output logits [vocab_size].
 * @param vocab_size Size of vocabulary.
 * @param generated_ids Array of previously generated token IDs.
 * @param num_generated Number of generated tokens.
 * @param penalty Repetition penalty (> 1.0 increases penalty).
 */
void apply_repetition_penalty_f32(
    float* __restrict logits,
    size_t vocab_size,
    const uint32_t* __restrict generated_ids,
    size_t num_generated,
    float penalty
);

// -----------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------

/**
 * @brief Compute logits statistics (min, max, mean, variance).
 *
 * Uses Welford's online algorithm for numerical stability.
 *
 * @param logits Input logits [vocab_size].
 * @param vocab_size Size of vocabulary.
 * @param out_min Output minimum value.
 * @param out_max Output maximum value.
 * @param out_mean Output mean.
 * @param out_var Output variance.
 */
void logits_stats_f32(
    const float* __restrict logits,
    size_t vocab_size,
    float* out_min,
    float* out_max,
    float* out_mean,
    float* out_var
);

} // namespace metal_marlin
