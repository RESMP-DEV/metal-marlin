#include "pipeline.hpp"
#include <iostream>

#import <Metal/Metal.h>

namespace metal_marlin {

using namespace direct_access;

Pipeline::Pipeline(MetalDeviceDirect& device) : device_(device) {}

void Pipeline::load_weight(const std::string& name, const void* data, size_t size) {
    // Create a managed buffer for the weight
    // MTLResourceStorageModeShared = 0
    auto buffer = device_.create_buffer_from_bytes(data, size, MTLResourceStorageModeShared);
    weights_[name] = std::move(buffer);
}

void Pipeline::forward(
    const void* input_ids, 
    size_t batch_size,
    size_t seq_len,
    void* logits
) {
    // Placeholder implementation
    // In a real implementation, this would:
    // 1. Embedding lookup
    // 2. Loop over layers
    //    - Attention
    //    - MLP
    // 3. RMSNorm
    // 4. LM Head
    
    // For now, just print to confirm wiring
    // std::cout << "Pipeline::forward called with batch_size=" << batch_size << ", seq_len=" << seq_len << std::endl;
}

} // namespace metal_marlin