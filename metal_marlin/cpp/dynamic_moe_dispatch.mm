// dynamic_moe_dispatch.mm - Objective-C++ implementation
#include "dynamic_moe_dispatch.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace metal_marlin {

DynamicMoEDispatcher::DynamicMoEDispatcher(MTL::Device* device, MTL::Library* library)
    : device_(device), library_(library) {
    device_->retain();
    library_->retain();
    
    // Create command queue
    queue_ = device_->newCommandQueue();
    
    // Get kernel pipeline state
    NS::Error* error = nullptr;
    auto func = library_->newFunction(NS::String::string("moe_trellis_swiglu_dynamic", NS::UTF8StringEncoding));
    if (!func) {
        throw std::runtime_error("Failed to find moe_trellis_swiglu_dynamic kernel");
    }
    dynamic_kernel_pso_ = device_->newComputePipelineState(func, &error);
    func->release();
    if (!dynamic_kernel_pso_) {
        throw std::runtime_error("Failed to create PSO: " + 
            std::string(error->localizedDescription()->utf8String()));
    }
    
    // Create grid offsets buffer (constant)
    grid_offsets_buf_ = device_->newBuffer(GRID_OFFSETS.data(), 
        GRID_OFFSETS.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

DynamicMoEDispatcher::~DynamicMoEDispatcher() {
    if (expert_bits_buf_) expert_bits_buf_->release();
    if (grid_offsets_buf_) grid_offsets_buf_->release();
    if (grids_buf_) grids_buf_->release();
    if (params_buf_) params_buf_->release();
    if (output_buf_) output_buf_->release();
    if (dynamic_kernel_pso_) dynamic_kernel_pso_->release();
    if (queue_) queue_->release();
    library_->release();
    device_->release();
}

void DynamicMoEDispatcher::set_expert_bits(const std::vector<ExpertBitConfig>& expert_bits) {
    num_experts_ = expert_bits.size();
    if (expert_bits_buf_) expert_bits_buf_->release();
    expert_bits_buf_ = device_->newBuffer(expert_bits.data(),
        expert_bits.size() * sizeof(ExpertBitConfig), MTL::ResourceStorageModeShared);
    initialized_ = true;
}

void DynamicMoEDispatcher::build_grid_buffer(
    const std::vector<std::vector<float>>& grids_by_bits
) {
    // Concatenate grids: 2-bit (4 vals), 3-bit (8 vals), 4-bit (16 vals), etc.
    std::vector<uint16_t> concat_grids;  // half precision
    for (int bits = 2; bits <= 6; bits++) {
        if (bits - 2 < grids_by_bits.size()) {
            for (float val : grids_by_bits[bits - 2]) {
                // Convert float to half (simplified)
                concat_grids.push_back(static_cast<uint16_t>(val * 65504.0f));
            }
        }
    }
    
    if (grids_buf_) grids_buf_->release();
    grids_buf_ = device_->newBuffer(concat_grids.data(),
        concat_grids.size() * sizeof(uint16_t), MTL::ResourceStorageModeShared);
}

MTL::Buffer* DynamicMoEDispatcher::dispatch(
    MTL::Buffer* activations,
    MTL::Buffer* expert_ids,
    MTL::Buffer* expert_probs,
    MTL::Buffer* gate_weights,
    MTL::Buffer* gate_scales,
    MTL::Buffer* up_weights,
    MTL::Buffer* up_scales,
    MTL::Buffer* down_weights,
    MTL::Buffer* down_scales,
    MTL::Buffer* gate_su,
    MTL::Buffer* gate_sv,
    MTL::Buffer* up_su,
    MTL::Buffer* up_sv,
    MTL::Buffer* down_su,
    MTL::Buffer* down_sv,
    const DynamicMoEParams& params
) {
    // Ensure output buffer is allocated
    size_t output_size = params.batch_size * params.hidden_dim * sizeof(uint16_t);
    if (!output_buf_ || output_buf_->length() < output_size) {
        if (output_buf_) output_buf_->release();
        output_buf_ = device_->newBuffer(output_size, MTL::ResourceStorageModeShared);
    }
    
    // Update params buffer
    if (!params_buf_) {
        params_buf_ = device_->newBuffer(sizeof(DynamicMoEParams), MTL::ResourceStorageModeShared);
    }
    memcpy(params_buf_->contents(), &params, sizeof(DynamicMoEParams));
    
    // Create command buffer
    auto cmd = queue_->commandBuffer();
    auto encoder = cmd->computeCommandEncoder();
    
    encoder->setComputePipelineState(dynamic_kernel_pso_);
    
    // Set buffers (order must match kernel signature)
    encoder->setBuffer(activations, 0, 0);
    encoder->setBuffer(gate_weights, 0, 1);
    encoder->setBuffer(gate_scales, 0, 2);
    encoder->setBuffer(up_weights, 0, 3);
    encoder->setBuffer(up_scales, 0, 4);
    encoder->setBuffer(down_weights, 0, 5);
    encoder->setBuffer(down_scales, 0, 6);
    encoder->setBuffer(gate_su, 0, 7);
    encoder->setBuffer(gate_sv, 0, 8);
    encoder->setBuffer(up_su, 0, 9);
    encoder->setBuffer(up_sv, 0, 10);
    encoder->setBuffer(down_su, 0, 11);
    encoder->setBuffer(down_sv, 0, 12);
    encoder->setBuffer(grids_buf_, 0, 13);
    encoder->setBuffer(grid_offsets_buf_, 0, 14);
    encoder->setBuffer(expert_bits_buf_, 0, 15);
    encoder->setBuffer(expert_ids, 0, 16);
    encoder->setBuffer(expert_probs, 0, 17);
    encoder->setBuffer(output_buf_, 0, 18);
    encoder->setBuffer(params_buf_, 0, 19);
    
    // Dispatch grid
    uint32_t tile_n = 64;  // MOE_TILE_N
    MTL::Size grid(
        (params.hidden_dim + tile_n - 1) / tile_n,  // X: column blocks
        params.batch_size,                           // Y: tokens
        params.top_k                                 // Z: expert slots
    );
    MTL::Size tg(128, 1, 1);  // 4 simdgroups
    
    encoder->dispatchThreadgroups(grid, tg);
    encoder->endEncoding();
    
    cmd->commit();
    cmd->waitUntilCompleted();
    
    return output_buf_;
}

} // namespace metal_marlin
