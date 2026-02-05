// moe_dispatcher.h - MoE Dispatcher for Metal Marlin
//
// Implements efficient Mixture of Experts dispatch using:
// - Single MTLCommandBuffer per MoE layer (not per expert)
// - Buffer pool for MTLBuffer reuse
// - Indirect dispatch for variable expert counts
// - Blit encoder for inter-kernel data transfers

#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace metal_marlin {

// Forward declaration
class BufferPool;

// --------------------------------------------------------------------------
// Expert Dispatch Configuration
// --------------------------------------------------------------------------

struct ExpertDispatchConfig {
    uint32_t expert_id;           // Which expert to dispatch
    uint32_t num_tokens;          // Number of tokens for this expert
    uint32_t hidden_dim;          // Hidden dimension
    uint32_t intermediate_dim;    // Intermediate dimension (for FFN)
    
    // Workgroup configuration
    uint32_t threadgroup_size[3];
    uint32_t grid_size[3];
};

// --------------------------------------------------------------------------
// Buffer Pool for MTLBuffer Reuse
// --------------------------------------------------------------------------

class BufferPool {
public:
    explicit BufferPool(MTL::Device* device, MTL::ResourceOptions options = MTL::ResourceStorageModeShared);
    ~BufferPool();

    // Non-copyable
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    // Get a buffer of at least the requested size
    MTL::Buffer* acquire(size_t size);
    
    // Return a buffer to the pool
    void release(MTL::Buffer* buffer);
    
    // Clear all pooled buffers
    void clear();
    
    // Statistics
    size_t pooled_count() const;
    size_t pooled_bytes() const;

private:
    size_t align_size(size_t size) const;

    MTL::Device* device_;
    MTL::ResourceOptions options_;
    mutable std::mutex mutex_;
    
    // Pooled buffers organized by aligned size
    std::unordered_map<size_t, std::vector<MTL::Buffer*>> pools_;
};

// --------------------------------------------------------------------------
// MoE Dispatcher - Main class for dispatching MoE operations
// --------------------------------------------------------------------------

class MoEDispatcher {
public:
    MoEDispatcher(MTL::Device* device, MTL::Library* library);
    ~MoEDispatcher();

    // Non-copyable
    MoEDispatcher(const MoEDispatcher&) = delete;
    MoEDispatcher& operator=(const MoEDispatcher&) = delete;

    // ----------------------------------------------------------------------
    // Core dispatch operations
    // ----------------------------------------------------------------------

    // Dispatch all experts in a single command buffer
    // activations: [batch, hidden_dim] input activations
    // expert_ids: [batch, top_k] expert assignments
    // expert_probs: [batch, top_k] routing probabilities
    // Returns: [batch, hidden_dim] output tensor
    void* dispatch(
        void* activations,           // Input buffer
        void* expert_ids,            // Expert assignment buffer
        void* expert_probs,          // Expert probability buffer
        const std::vector<void*>& expert_weights,  // Expert weight buffers
        int32_t batch_size,
        int32_t hidden_dim,
        int32_t intermediate_dim,
        int32_t num_experts,
        int32_t top_k
    );

    // Prepare a dispatch operation (for repeated execution)
    void prepare_dispatch(
        int32_t batch_size,
        int32_t hidden_dim,
        int32_t intermediate_dim,
        int32_t num_experts,
        int32_t top_k
    );

    // Execute a prepared dispatch
    void execute_prepared(
        void* activations,
        void* expert_ids,
        void* expert_probs,
        const std::vector<void*>& expert_weights,
        void* output
    );

    // ----------------------------------------------------------------------
    // Buffer management
    // ----------------------------------------------------------------------

    // Get or create a buffer for temporary data
    MTL::Buffer* get_temp_buffer(size_t size);
    
    // Release a temporary buffer back to the pool
    void release_temp_buffer(MTL::Buffer* buffer);

    // ----------------------------------------------------------------------
    // Command buffer operations
    // ----------------------------------------------------------------------

    // Begin encoding a new command buffer
    void begin_command_buffer();
    
    // Commit and wait for completion
    void commit_and_wait();
    
    // Get the current command buffer (for advanced usage)
    MTL::CommandBuffer* current_command_buffer() { return current_cmd_; }

    // ----------------------------------------------------------------------
    // Expert dispatch encoding
    // ----------------------------------------------------------------------

    // Encode a single expert dispatch using indirect command buffer
    void encode_expert_dispatch(
        const ExpertDispatchConfig& config,
        MTL::Buffer* input_buffer,
        MTL::Buffer* weight_buffer,
        MTL::Buffer* output_buffer,
        MTL::Buffer* params_buffer = nullptr
    );

    // Encode multiple expert dispatches batched together
    void encode_batched_experts(
        const std::vector<ExpertDispatchConfig>& configs,
        const std::vector<MTL::Buffer*>& input_buffers,
        const std::vector<MTL::Buffer*>& weight_buffers,
        const std::vector<MTL::Buffer*>& output_buffers
    );

    // ----------------------------------------------------------------------
    // Blit operations for data transfer
    // ----------------------------------------------------------------------

    // Encode a buffer copy using blit encoder
    void encode_buffer_copy(
        MTL::Buffer* src,
        MTL::Buffer* dst,
        size_t src_offset = 0,
        size_t dst_offset = 0,
        size_t size = 0  // 0 = copy entire buffer
    );

    // Encode a buffer fill operation
    void encode_buffer_fill(
        MTL::Buffer* buffer,
        uint32_t value,
        size_t offset = 0,
        size_t size = 0
    );

    // ----------------------------------------------------------------------
    // Synchronization
    // ----------------------------------------------------------------------

    // Wait for all pending operations to complete
    void wait_until_completed();
    
    // Check if device supports indirect command buffers
    bool supports_indirect_command_buffers() const;

    // ----------------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------------

    MTL::Device* device() const { return device_; }
    BufferPool* buffer_pool() const { return buffer_pool_.get(); }

private:
    // Metal resources
    MTL::Device* device_;
    MTL::Library* library_;
    MTL::CommandQueue* command_queue_;
    MTL::CommandBuffer* current_cmd_;
    MTL::ComputeCommandEncoder* current_encoder_;
    
    // Buffer pool
    std::unique_ptr<BufferPool> buffer_pool_;
    
    // Track active temp buffers for cleanup
    std::vector<MTL::Buffer*> active_temp_buffers_;
    mutable std::mutex temp_buffer_mutex_;
    
    // Pipeline states cache
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
    mutable std::mutex pipeline_mutex_;
    
    // Prepared dispatch state
    struct PreparedState {
        int32_t batch_size = 0;
        int32_t hidden_dim = 0;
        int32_t intermediate_dim = 0;
        int32_t num_experts = 0;
        int32_t top_k = 0;
        bool is_prepared = false;
    };
    PreparedState prepared_state_;
    
    // Helper methods
    MTL::ComputePipelineState* get_pipeline(const std::string& kernel_name);
    void end_current_encoder();
    void begin_compute_encoder();
};

} // namespace metal_marlin
