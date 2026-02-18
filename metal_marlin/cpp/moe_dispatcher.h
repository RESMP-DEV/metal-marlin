#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <torch/extension.h>
#include <vector>
#include <memory>

namespace metal_marlin {

// Forward declaration
class MoEDispatcherImpl;

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
    
    // Move-constructible
    MoEDispatcher(MoEDispatcher&&) noexcept;
    MoEDispatcher& operator=(MoEDispatcher&&) noexcept;

    // ----------------------------------------------------------------------
    // Core dispatch operations
    // ----------------------------------------------------------------------

    // Dispatch all experts in a single command buffer
    torch::Tensor dispatch(
        const torch::Tensor& activations,
        const torch::Tensor& expert_ids,
        const torch::Tensor& expert_probs,
        const std::vector<torch::Tensor>& expert_weights,
        int32_t hidden_dim,
        int32_t intermediate_dim,
        int32_t num_experts,
        int32_t top_k
    );

    // Prepare a dispatch operation (for repeated execution)
    void prepare_dispatch(
        int32_t batch_size,
        int32_t hidden_dim,
        int32_t num_experts,
        int32_t top_k
    );

    // Execute a prepared dispatch
    void execute_prepared(
        const torch::Tensor& activations,
        const torch::Tensor& expert_ids,
        const torch::Tensor& expert_probs,
        torch::Tensor& output
    );

    // ----------------------------------------------------------------------
    // Buffer management (exposed for Python bindings)
    // ----------------------------------------------------------------------
    MTL::Buffer* get_temp_buffer(size_t size);
    void release_temp_buffer(MTL::Buffer* buffer);

    // ----------------------------------------------------------------------
    // Synchronization
    // ----------------------------------------------------------------------
    void wait_until_completed();
    
    // Check if device supports indirect command buffers
    bool supports_indirect_command_buffers() const;

private:
    std::unique_ptr<MoEDispatcherImpl> impl_;
};

} // namespace metal_marlin