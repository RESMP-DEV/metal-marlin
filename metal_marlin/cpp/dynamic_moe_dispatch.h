// dynamic_moe_dispatch.h - C++ dispatcher for dynamic mixed-precision MoE
#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>
#include <vector>
#include <array>
#include <span>

namespace metal_marlin {

/// Per-expert bit configuration (matches Metal ExpertBitConfig)
struct ExpertBitConfig {
    uint8_t gate_bits;
    uint8_t up_bits;
    uint8_t down_bits;
    uint8_t padding;
};
static_assert(sizeof(ExpertBitConfig) == 4, "ExpertBitConfig must be 4 bytes");

/// MoE dispatch parameters
struct DynamicMoEParams {
    uint32_t batch_size;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t tile_size;
};

/// Grid offset table for concatenated dequant grids
constexpr std::array<uint32_t, 7> GRID_OFFSETS = {
    0,    // 0-bit (unused)
    0,    // 1-bit (unused)
    0,    // 2-bit
    4,    // 3-bit (after 4 2-bit values)
    12,   // 4-bit (after 4+8 values)
    28,   // 5-bit (after 4+8+16 values)
    60    // 6-bit (after 4+8+16+32 values)
};

/// C++ dispatcher for dynamic mixed-precision MoE
/// Handles token sorting by bit-group and kernel dispatch
class DynamicMoEDispatcher {
public:
    DynamicMoEDispatcher(MTL::Device* device, MTL::Library* library);
    ~DynamicMoEDispatcher();
    
    // Initialize with expert bit configurations (call once after loading model)
    void set_expert_bits(const std::vector<ExpertBitConfig>& expert_bits);
    
    // Pre-allocate concatenated grid buffer
    void build_grid_buffer(
        const std::vector<std::vector<float>>& grids_by_bits  // [bits][values]
    );
    
    /// Dispatch MoE layer with dynamic bit widths
    /// Returns output buffer pointer
    MTL::Buffer* dispatch(
        MTL::Buffer* activations,      // [batch, hidden_dim]
        MTL::Buffer* expert_ids,       // [batch, top_k] uint32
        MTL::Buffer* expert_probs,     // [batch, top_k] half
        MTL::Buffer* gate_weights,     // [num_experts, ...]
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
    );
    
    // Synchronous dispatch with output copy back
    void dispatch_sync(/* same args */);
    
private:
    MTL::Device* device_;
    MTL::Library* library_;
    MTL::CommandQueue* queue_;
    MTL::ComputePipelineState* dynamic_kernel_pso_;
    
    // Cached buffers
    MTL::Buffer* expert_bits_buf_;    // [num_experts] ExpertBitConfig
    MTL::Buffer* grid_offsets_buf_;   // [7] uint32
    MTL::Buffer* grids_buf_;          // Concatenated grid values
    MTL::Buffer* params_buf_;         // MoE params
    MTL::Buffer* output_buf_;         // Pre-allocated output
    
    // State
    uint32_t num_experts_ = 0;
    bool initialized_ = false;
};

} // namespace metal_marlin
