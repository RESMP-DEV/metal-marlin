#pragma once

#include "direct_access.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace metal_marlin {

struct PipelineConfig {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    float rms_norm_eps;
    float rope_theta;
};

class Pipeline {
public:
    Pipeline(direct_access::MetalDeviceDirect& device);

    void load_weight(const std::string& name, const void* data, size_t size);
    
    // Simple forward pass for testing/wiring
    void forward(
        const void* input_ids, // [batch, seq_len]
        size_t batch_size,
        size_t seq_len,
        void* logits // [batch, seq_len, vocab_size]
    );

private:
    direct_access::MetalDeviceDirect& device_;
    std::unordered_map<std::string, direct_access::BufferPtr> weights_;
    PipelineConfig config_;
};

} // namespace metal_marlin