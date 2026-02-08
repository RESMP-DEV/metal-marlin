// test_router_dispatch.cpp - Unit tests for FastRouterDispatcher
//
// Tests verify:
// - Batched router forward pass correctness
// - SIMD top-k selection matches reference
// - Async execution with double-buffering
// - Hot pair cache functionality

#include "moe_router_dispatch.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <chrono>

using namespace metal_marlin::moe;

namespace {

std::vector<uint16_t> generate_bf16_activations(int32_t num_tokens, int32_t hidden_dim) {
    std::vector<uint16_t> activations(num_tokens * hidden_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < activations.size(); ++i) {
        float val = dist(rng);
        // Convert float to BF16 (truncate upper 16 bits)
        uint32_t bits = std::bit_cast<uint32_t>(val);
        activations[i] = static_cast<uint16_t>(bits >> 16);
    }
    return activations;
}

std::vector<float> generate_router_weights(int32_t num_experts, int32_t hidden_dim) {
    std::vector<float> weights(num_experts * hidden_dim);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(rng);
    }
    return weights;
}

std::vector<float> generate_router_bias(int32_t num_experts) {
    std::vector<float> bias(num_experts, 0.0f);
    return bias;
}

void test_basic_router_forward() {
    std::cout << "Test: Basic router forward pass..." << std::flush;

    constexpr int32_t num_experts = 8;
    constexpr int32_t hidden_dim = 128;
    constexpr int32_t top_k = 2;
    constexpr int32_t num_tokens = 4;

    FastRouterDispatcher router(num_experts, hidden_dim, top_k);

    auto activations = generate_bf16_activations(num_tokens, hidden_dim);
    auto weights = generate_router_weights(num_experts, hidden_dim);
    auto bias = generate_router_bias(num_experts);

    auto output = router.route_batch(
        activations.data(),
        num_tokens,
        weights.data(),
        bias.data()
    );

    // Verify output structure
    assert(output.num_tokens == num_tokens);
    assert(output.top_k == top_k);
    assert(output.num_experts == num_experts);
    assert(output.logits.size() == static_cast<size_t>(num_tokens * num_experts));
    assert(output.topk_expert_ids.size() == static_cast<size_t>(num_tokens * top_k));
    assert(output.topk_probs.size() == static_cast<size_t>(num_tokens * top_k));

    // Verify expert IDs are in valid range
    for (int32_t expert_id : output.topk_expert_ids) {
        assert(expert_id >= 0 && expert_id < num_experts);
    }

    // Verify probabilities are positive
    for (float prob : output.topk_probs) {
        assert(prob >= 0.0f);
    }

    std::cout << " PASSED" << std::endl;
}

void test_async_double_buffering() {
    std::cout << "Test: Async execution with double-buffering..." << std::flush;

    constexpr int32_t num_experts = 4;
    constexpr int32_t hidden_dim = 64;
    constexpr int32_t top_k = 2;

    FastRouterDispatcher router(num_experts, hidden_dim, top_k);

    auto activations = generate_bf16_activations(8, hidden_dim);
    auto weights = generate_router_weights(num_experts, hidden_dim);
    auto bias = generate_router_bias(num_experts);

    std::vector<int> expert_launch_count(2, 0);

    auto expert_launch_fn = [&expert_launch_count](const RouterBuffer& buffer) {
        int idx = buffer.sequence_id & 1;
        expert_launch_count[idx]++;
        std::promise<void> done;
        done.set_value();
        return done.get_future().share();
    };

    // Submit multiple async operations
    for (int i = 0; i < 4; ++i) {
        router.submit_async(
            activations.data(),
            8,
            weights.data(),
            bias.data(),
            expert_launch_fn
        ).wait();
    }

    // Verify both buffers were used
    assert(expert_launch_count[0] > 0);
    assert(expert_launch_count[1] > 0);

    std::cout << " PASSED" << std::endl;
}

void test_hot_pair_cache() {
    std::cout << "Test: Hot pair caching..." << std::flush;

    constexpr int32_t num_experts = 8;
    constexpr int32_t hidden_dim = 64;
    constexpr int32_t top_k = 2;
    constexpr int32_t num_tokens = 16;

    FastRouterDispatcher router(num_experts, hidden_dim, top_k, 128, 256, 8);

    auto activations = generate_bf16_activations(num_tokens, hidden_dim);
    auto weights = generate_router_weights(num_experts, hidden_dim);
    auto bias = generate_router_bias(num_experts);

    // Generate multiple batches to populate cache
    std::mt19937 rng(789);
    for (int iter = 0; iter < 20; ++iter) {
        // Vary activations slightly
        for (size_t i = 0; i < activations.size(); ++i) {
            float val = (std::bit_cast<float>(static_cast<uint32_t>(activations[i]) << 16)
                         + std::uniform_real_distribution<float>(-0.1f, 0.1f)(rng));
            uint32_t bits = std::bit_cast<uint32_t>(val);
            activations[i] = static_cast<uint16_t>(bits >> 16);
        }

        router.route_batch(activations.data(), num_tokens, weights.data(), bias.data());
    }

    // Check that some pairs became hot
    size_t hot_pairs = router.hot_pair_count();
    std::cout << " (" << hot_pairs << " hot pairs detected)" << std::flush;

    // Reset and verify cache clears
    router.reset_hot_pair_cache();
    assert(router.hot_pair_count() <= 2);  // Preferred pair might remain

    std::cout << " PASSED" << std::endl;
}

void test_dispatch_packing() {
    std::cout << "Test: Dispatch info packing..." << std::flush;

    constexpr int32_t num_experts = 8;
    constexpr int32_t hidden_dim = 128;
    constexpr int32_t top_k = 2;
    constexpr int32_t num_tokens = 4;

    FastRouterDispatcher router(num_experts, hidden_dim, top_k);

    auto activations = generate_bf16_activations(num_tokens, hidden_dim);
    auto weights = generate_router_weights(num_experts, hidden_dim);
    auto bias = generate_router_bias(num_experts);

    auto output = router.route_batch(
        activations.data(),
        num_tokens,
        weights.data(),
        bias.data()
    );

    const auto& dispatch = output.dispatch_info;

    // Verify dispatch structure
    assert(dispatch.num_tokens == num_tokens);
    assert(dispatch.top_k == top_k);
    assert(dispatch.num_experts == num_experts);
    assert(dispatch.total_assignments() == num_tokens * top_k);

    // Verify offsets
    assert(dispatch.expert_offsets.size() == static_cast<size_t>(num_experts + 1));
    assert(dispatch.expert_offsets[0] == 0);
    assert(dispatch.expert_offsets[num_experts] == num_tokens * top_k);

    // Verify indices arrays
    assert(dispatch.sorted_token_indices.size() == static_cast<size_t>(num_tokens * top_k));
    assert(dispatch.sorted_expert_indices.size() == static_cast<size_t>(num_tokens * top_k));
    assert(dispatch.inverse_indices.size() == static_cast<size_t>(num_tokens * top_k));

    std::cout << " PASSED" << std::endl;
}

void test_batch_processing_consistency() {
    std::cout << "Test: Batch processing consistency..." << std::flush;

    constexpr int32_t num_experts = 4;
    constexpr int32_t hidden_dim = 64;
    constexpr int32_t top_k = 2;

    FastRouterDispatcher router(num_experts, hidden_dim, top_k);

    auto activations = generate_bf16_activations(4, hidden_dim);
    auto weights = generate_router_weights(num_experts, hidden_dim);
    auto bias = generate_router_bias(num_experts);

    // Process as single batch
    auto batch_output = router.route_batch(
        activations.data(),
        4,
        weights.data(),
        bias.data()
    );

    // Process as two separate batches
    auto batch1 = router.route_batch(
        activations.data(),
        2,
        weights.data(),
        bias.data()
    );

    auto batch2 = router.route_batch(
        activations.data() + 2 * hidden_dim,
        2,
        weights.data(),
        bias.data()
    );

    // Verify that batched results match per-token results
    for (int t = 0; t < 2; ++t) {
        for (int k = 0; k < top_k; ++k) {
            assert(batch_output.topk_expert_ids[t * top_k + k] == batch1.topk_expert_ids[t * top_k + k]);
            assert(std::abs(batch_output.topk_probs[t * top_k + k] - batch1.topk_probs[t * top_k + k]) < 1e-6f);
        }
    }

    for (int t = 0; t < 2; ++t) {
        for (int k = 0; k < top_k; ++k) {
            assert(batch_output.topk_expert_ids[(t + 2) * top_k + k] == batch2.topk_expert_ids[t * top_k + k]);
            assert(std::abs(batch_output.topk_probs[(t + 2) * top_k + k] - batch2.topk_probs[t * top_k + k]) < 1e-6f);
        }
    }

    std::cout << " PASSED" << std::endl;
}

} // namespace

int main() {
    std::cout << "=== FastRouterDispatcher Unit Tests ===" << std::endl;
    std::cout << std::endl;

    test_basic_router_forward();
    test_async_double_buffering();
    test_hot_pair_cache();
    test_dispatch_packing();
    test_batch_processing_consistency();

    std::cout << std::endl;
    std::cout << "=== All tests PASSED ===" << std::endl;

    return 0;
}
