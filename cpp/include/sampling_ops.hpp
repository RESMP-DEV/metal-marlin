#pragma once

#include "metal_device.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <optional>

namespace metal_marlin {

namespace nb = nanobind;

struct SamplingConfig {
    float temperature = 1.0f;
    int32_t top_k = 0;
    float top_p = 1.0f;
    bool multinominal = false; // If true, use multinomial sampling from logits
    uint64_t seed = 0;
};

// Main sampling function
// logits: [batch_size, vocab_size]
// output_tokens: [batch_size] (pre-allocated)
void sample_tokens(
    MetalContext& ctx,
    nb::bytes logits,        // Float32 or Float16
    nb::bytes output_tokens, // Uint32
    uint32_t batch_size,
    uint32_t vocab_size,
    const SamplingConfig& config,
    bool wait = true
);

// Individual operations (exposed for testing/advanced usage)
void softmax(
    MetalContext& ctx,
    nb::bytes logits,
    nb::bytes probs,
    uint32_t batch_size,
    uint32_t vocab_size,
    bool wait = true
);

void argmax(
    MetalContext& ctx,
    nb::bytes logits,
    nb::bytes indices,
    uint32_t batch_size,
    uint32_t vocab_size,
    bool wait = true
);

void sample_top_k(
    MetalContext& ctx,
    nb::bytes logits,
    nb::bytes indices,
    uint32_t batch_size,
    uint32_t vocab_size,
    int32_t k,
    uint64_t seed,
    bool wait = true
);

void sample_top_p(
    MetalContext& ctx,
    nb::bytes logits,
    nb::bytes indices,
    nb::bytes workspace, // [batch, vocab_size * 2]
    uint32_t batch_size,
    uint32_t vocab_size,
    float p,
    uint64_t seed,
    bool wait = true
);

void sample_categorical(
    MetalContext& ctx,
    nb::bytes probs,
    nb::bytes indices,
    uint32_t batch_size,
    uint32_t vocab_size,
    uint64_t seed,
    bool wait = true
);

} // namespace metal_marlin
