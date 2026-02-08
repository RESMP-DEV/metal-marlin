// bench_router_dispatch.cpp - Performance benchmarks for FastRouterDispatcher
//
// Measures:
// - BF16→FP32 conversion throughput
// - Batched router matrix multiplication
// - SIMD top-k selection speed
// - Async dispatch latency
// - Hot pair cache hit rates

#include "moe_router_dispatch.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace metal_marlin::moe;
using namespace std::chrono;

namespace {

// Generate random BF16 activations
std::vector<uint16_t> generate_bf16_activations(int32_t num_tokens, int32_t hidden_dim) {
    std::vector<uint16_t> activations(num_tokens * hidden_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < activations.size(); ++i) {
        float val = dist(rng);
        uint32_t bits = std::bit_cast<uint32_t>(val);
        activations[i] = static_cast<uint16_t>(bits >> 16);
    }
    return activations;
}

// Generate random router weights
std::vector<float> generate_router_weights(int32_t num_experts, int32_t hidden_dim) {
    std::vector<float> weights(num_experts * hidden_dim);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(rng);
    }
    return weights;
}

// BF16→FP32 conversion benchmark
void bench_bf16_conversion(int32_t num_tokens, int32_t hidden_dim) {
    auto activations = generate_bf16_activations(num_tokens, hidden_dim);
    std::vector<float> fp32_buffer(num_tokens * hidden_dim);
    
    auto start = high_resolution_clock::now();
    
    for (int32_t t = 0; t < num_tokens; ++t) {
        const uint16_t* bf16_row = activations.data() + t * hidden_dim;
        float* fp32_row = fp32_buffer.data() + t * hidden_dim;
        
        for (int32_t d = 0; d < hidden_dim; ++d) {
            fp32_row[d] = std::bit_cast<float>(static_cast<uint32_t>(bf16_row[d]) << 16);
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    
    double total_elements = static_cast<double>(num_tokens) * hidden_dim;
    double gbps = (total_elements * sizeof(uint16_t)) / (duration.count() / 1e9) / 1e9;
    
    std::cout << "  BF16→FP32 conversion: " << std::setw(10) << duration.count() << " ns"
              << " (" << std::fixed << std::setprecision(2) << gbps << " GB/s)" << std::endl;
}

// SIMD dot product benchmark
void bench_simd_dot_product(int32_t num_pairs, int32_t hidden_dim) {
    std::vector<float> a(num_pairs * hidden_dim);
    std::vector<float> b(num_pairs * hidden_dim);
    std::vector<float> results(num_pairs);
    
    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : a) val = dist(rng);
    for (auto& val : b) val = dist(rng);
    
    auto start = high_resolution_clock::now();
    
    for (int32_t i = 0; i < num_pairs; ++i) {
        const float* lhs = a.data() + i * hidden_dim;
        const float* rhs = b.data() + i * hidden_dim;
        
        float dot = 0.0f;
        for (int32_t d = 0; d < hidden_dim; ++d) {
            dot += lhs[d] * rhs[d];
        }
        results[i] = dot;
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    
    double total_flops = static_cast<double>(num_pairs) * hidden_dim * 2;  // multiply + add
    double gflops = total_flops / (duration.count() / 1e9) / 1e9;
    
    std::cout << "  Scalar dot product:   " << std::setw(10) << duration.count() << " ns"
              << " (" << std::fixed << std::setprecision(2) << gflops << " GFLOPS)" << std::endl;
}

// Full router forward pass benchmark
void bench_router_forward(FastRouterDispatcher& router,
                         const uint16_t* activations,
                         const float* weights,
                         const float* bias,
                         int32_t num_tokens) {
    auto start = high_resolution_clock::now();
    
    auto output = router.route_batch(activations, num_tokens, weights, bias);
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    
    double total_operations = static_cast<double>(num_tokens) * router.num_experts() * 
                              router.hidden_dim() * 2;  // multiply + add per expert
    
    double gflops = total_operations / (duration.count() / 1e9) / 1e9;
    double tokens_per_sec = num_tokens / (duration.count() / 1e9);
    
    std::cout << "  Router forward:       " << std::setw(10) << duration.count() << " ns"
              << " (" << std::fixed << std::setprecision(2) << gflops << " GFLOPS, "
              << tokens_per_sec / 1e3 << " K tokens/sec)" << std::endl;
}

// Async dispatch benchmark
void bench_async_dispatch(FastRouterDispatcher& router,
                         const uint16_t* activations,
                         const float* weights,
                         const float* bias,
                         int32_t num_tokens,
                         int32_t iterations) {
    std::vector<std::shared_future<void>> futures;
    futures.reserve(iterations);
    
    auto expert_launch_fn = [](const RouterBuffer&) {
        std::promise<void> done;
        done.set_value();
        return done.get_future().share();
    };
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto future = router.submit_async(
            activations,
            num_tokens,
            weights,
            bias,
            expert_launch_fn
        );
        futures.push_back(future);
    }
    
    // Wait for all
    for (auto& future : futures) {
        future.wait();
    }
    
    auto end = high_resolution_clock::now();
    auto total_duration = duration_cast<nanoseconds>(end - start);
    auto avg_duration = total_duration / iterations;
    
    std::cout << "  Async dispatch (avg): " << std::setw(10) << avg_duration.count() << " ns"
              << " (" << std::fixed << std::setprecision(2) 
              << (num_tokens / (avg_duration.count() / 1e9)) / 1e3 << " K tokens/sec)" << std::endl;
}

// Generate random router bias
std::vector<float> generate_router_bias(int32_t num_experts) {
    std::vector<float> bias(num_experts, 0.0f);
    return bias;
}

// Cache hit rate benchmark
void bench_hot_pair_cache(FastRouterDispatcher& router,
                         const uint16_t* activations,
                         const float* weights,
                         const float* bias,
                         int32_t num_tokens,
                         int32_t iterations) {
    // Warm up cache
    for (int i = 0; i < iterations / 2; ++i) {
        router.route_batch(activations, num_tokens, weights, bias);
    }
    
    size_t hot_pairs_after_warmup = router.hot_pair_count();
    
    // Measure dispatch with warmed cache
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        router.route_batch(activations, num_tokens, weights, bias);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    
    std::cout << "  With hot pair cache:  " << std::setw(10) << (duration.count() / iterations) << " ns"
              << " (" << hot_pairs_after_warmup << " hot pairs)" << std::endl;
}

} // namespace

int main() {
    std::cout << std::endl;
    std::cout << "=== FastRouterDispatcher Performance Benchmarks ===" << std::endl;
    std::cout << std::endl;
    
    // Configuration 1: Small batch (decode path)
    {
        std::cout << "Configuration 1: Small batch (decode path)" << std::endl;
        std::cout << "  num_experts=8, hidden_dim=128, top_k=2, batch_size=1" << std::endl;
        
        constexpr int32_t num_experts = 8;
        constexpr int32_t hidden_dim = 128;
        constexpr int32_t top_k = 2;
        constexpr int32_t batch_size = 1;
        
        FastRouterDispatcher router(num_experts, hidden_dim, top_k, 128);
        
        auto activations = generate_bf16_activations(batch_size, hidden_dim);
        auto weights = generate_router_weights(num_experts, hidden_dim);
        auto bias = generate_router_bias(num_experts);
        
        bench_bf16_conversion(1000, hidden_dim);
        bench_simd_dot_product(1000, hidden_dim);
        bench_router_forward(router, activations.data(), weights.data(), bias.data(), batch_size);
        bench_async_dispatch(router, activations.data(), weights.data(), bias.data(), batch_size, 100);
        bench_hot_pair_cache(router, activations.data(), weights.data(), bias.data(), batch_size, 100);
        
        std::cout << std::endl;
    }
    
    // Configuration 2: Medium batch (prefill)
    {
        std::cout << "Configuration 2: Medium batch (prefill)" << std::endl;
        std::cout << "  num_experts=16, hidden_dim=256, top_k=2, batch_size=32" << std::endl;
        
        constexpr int32_t num_experts = 16;
        constexpr int32_t hidden_dim = 256;
        constexpr int32_t top_k = 2;
        constexpr int32_t batch_size = 32;
        
        FastRouterDispatcher router(num_experts, hidden_dim, top_k, 256);
        
        auto activations = generate_bf16_activations(batch_size, hidden_dim);
        auto weights = generate_router_weights(num_experts, hidden_dim);
        auto bias = generate_router_bias(num_experts);
        
        bench_bf16_conversion(100, hidden_dim);
        bench_simd_dot_product(100, hidden_dim);
        bench_router_forward(router, activations.data(), weights.data(), bias.data(), batch_size);
        bench_async_dispatch(router, activations.data(), weights.data(), bias.data(), batch_size, 20);
        bench_hot_pair_cache(router, activations.data(), weights.data(), bias.data(), batch_size, 50);
        
        std::cout << std::endl;
    }
    
    // Configuration 3: Large expert count (sparse MoE)
    {
        std::cout << "Configuration 3: Large expert count" << std::endl;
        std::cout << "  num_experts=64, hidden_dim=512, top_k=2, batch_size=8" << std::endl;
        
        constexpr int32_t num_experts = 64;
        constexpr int32_t hidden_dim = 512;
        constexpr int32_t top_k = 2;
        constexpr int32_t batch_size = 8;
        
        FastRouterDispatcher router(num_experts, hidden_dim, top_k, 128, 1024, 16);
        
        auto activations = generate_bf16_activations(batch_size, hidden_dim);
        auto weights = generate_router_weights(num_experts, hidden_dim);
        auto bias = generate_router_bias(num_experts);
        
        bench_bf16_conversion(50, hidden_dim);
        bench_simd_dot_product(50, hidden_dim);
        bench_router_forward(router, activations.data(), weights.data(), bias.data(), batch_size);
        bench_async_dispatch(router, activations.data(), weights.data(), bias.data(), batch_size, 10);
        bench_hot_pair_cache(router, activations.data(), weights.data(), bias.data(), batch_size, 30);
        
        std::cout << std::endl;
    }
    
    std::cout << "=== Benchmark completed ===" << std::endl;
    std::cout << std::endl;
    
    return 0;
}
