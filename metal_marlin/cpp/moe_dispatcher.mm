/**
 * MoEDispatcher - High-performance Metal dispatch for Mixture of Experts layers.
 */

#import "moe_dispatcher.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <torch/extension.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace metal_marlin {

// -----------------------------------------------------------------------------
// BufferPool Implementation
// -----------------------------------------------------------------------------

class BufferPool {
public:
    explicit BufferPool(MTL::Device* device, MTL::ResourceOptions options = MTL::ResourceStorageModeShared)
        : device_(device), options_(options) {
        device_->retain();
    }
    
    ~BufferPool() {
        clear();
        device_->release();
    }
    
    MTL::Buffer* acquire(size_t size) {
        size_t aligned = align_size(size);
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pools_.find(aligned);
            if (it != pools_.end() && !it->second.empty()) {
                MTL::Buffer* buf = it->second.back();
                it->second.pop_back();
                return buf;
            }
        }
        
        MTL::Buffer* buf = device_->newBuffer(aligned, options_);
        if (!buf) {
            throw std::runtime_error("Failed to allocate Metal buffer");
        }
        return buf;
    }
    
    void release(MTL::Buffer* buf) {
        if (!buf) return;
        size_t size = buf->length();
        std::lock_guard<std::mutex> lock(mutex_);
        pools_[size].push_back(buf);
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : pools_) {
            for (auto* buf : pair.second) {
                buf->release();
            }
        }
        pools_.clear();
    }

private:
    static size_t align_size(size_t size) {
        constexpr size_t ALIGNMENT = 256;
        return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }
    
    MTL::Device* device_;
    MTL::ResourceOptions options_;
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<MTL::Buffer*>> pools_;
};

// -----------------------------------------------------------------------------
// MoEDispatcher Implementation
// -----------------------------------------------------------------------------

class MoEDispatcherImpl {
public:
    explicit MoEDispatcherImpl(MTL::Device* device, MTL::Library* library)
        : device_(device), library_(library) {
        device_->retain();
        library_->retain();
        command_queue_ = device_->newCommandQueue();
        buffer_pool_ = std::make_unique<BufferPool>(device_);
        setup_pipeline_states();
    }
    
    ~MoEDispatcherImpl() {
        for (auto& pair : pipelines_) {
            pair.second->release();
        }
        pipelines_.clear();
        if (command_queue_) command_queue_->release();
        if (library_) library_->release();
        if (device_) device_->release();
    }
    
    void setup_pipeline_states() {
        const char* kernel_names[] = {
            "moe_gather_tokens", "moe_scatter_outputs", "moe_compute_grouping",
            "moe_expert_forward", "moe_expert_forward_fp4"
        };
        for (const char* name : kernel_names) {
            // Pre-load if possible, ignore errors (lazy load later)
            try { get_pipeline(name); } catch (...) {}
        }
    }
    
    MTL::ComputePipelineState* get_pipeline(const char* kernel_name) {
        auto it = pipelines_.find(kernel_name);
        if (it != pipelines_.end()) return it->second;
        
        NS::String* fn_name = NS::String::string(kernel_name, NS::UTF8StringEncoding);
        MTL::Function* function = library_->newFunction(fn_name);
        if (!function) return nullptr;
        
        NS::Error* error = nullptr;
        MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(function, &error);
        function->release();
        
        if (pipeline) pipelines_[kernel_name] = pipeline;
        return pipeline;
    }
    
    torch::Tensor dispatch(
        const torch::Tensor& activations,
        const torch::Tensor& expert_ids,
        const torch::Tensor& expert_probs,
        const std::vector<torch::Tensor>& expert_weights,
        int hidden_dim,
        int intermediate_dim,
        int num_experts,
        int top_k
    ) {
        TORCH_CHECK(activations.is_mps(), "activations must be on MPS device");
        auto output = torch::empty_like(activations);
        
        MTL::CommandBuffer* cmd_buffer = command_queue_->commandBuffer();
        
        encode_token_grouping(cmd_buffer, expert_ids, num_experts, top_k);
        
        MTL::ComputeCommandEncoder* gather_encoder = cmd_buffer->computeCommandEncoder();
        encode_gather_tokens(gather_encoder, activations, expert_ids, num_experts, top_k);
        gather_encoder->endEncoding();
        
        for (int i = 0; i < num_experts; ++i) {
            MTL::ComputeCommandEncoder* expert_encoder = cmd_buffer->computeCommandEncoder();
            encode_expert_dispatch(expert_encoder, i, expert_weights[i], hidden_dim, intermediate_dim);
            expert_encoder->endEncoding();
        }
        
        MTL::ComputeCommandEncoder* scatter_encoder = cmd_buffer->computeCommandEncoder();
        encode_scatter_outputs(scatter_encoder, output, expert_probs, num_experts, top_k);
        scatter_encoder->endEncoding();
        
        cmd_buffer->commit();
        cmd_buffer->waitUntilCompleted();
        return output;
    }
    
    void prepare_dispatch(int batch_size, int hidden_dim, int num_experts, int top_k) {
        // Placeholder
    }
    
    void execute_prepared(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&) {
        // Placeholder
    }

    MTL::Buffer* get_temp_buffer(size_t size) {
        return buffer_pool_->acquire(size);
    }

    void release_temp_buffer(MTL::Buffer* buffer) {
        buffer_pool_->release(buffer);
    }

    void wait_until_completed() {
        // Simple barrier
        MTL::CommandBuffer* cmd = command_queue_->commandBuffer();
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    bool supports_indirect_command_buffers() const {
        return device_->supportsFamily(MTL::GPUFamilyApple2);
    }
    
private:
    void encode_token_grouping(MTL::CommandBuffer* cmd_buffer, const torch::Tensor& expert_ids, int num_experts, int top_k) {
        auto pipeline = get_pipeline("moe_compute_grouping");
        if (!pipeline) return;
        
        MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();
        encoder->setComputePipelineState(pipeline);
        
        MTL::Buffer* expert_ids_buf = (MTL::Buffer*)expert_ids.data_ptr();
        encoder->setBuffer(expert_ids_buf, 0, 0);
        
        uint32_t batch_size = expert_ids.size(0);
        encoder->setBytes(&batch_size, sizeof(uint32_t), 4);
        encoder->setBytes(&top_k, sizeof(uint32_t), 5);
        encoder->setBytes(&num_experts, sizeof(uint32_t), 6);
        
        MTL::Size grid((batch_size * top_k + 255) / 256, 1, 1);
        MTL::Size threadgroup(256, 1, 1);
        encoder->dispatchThreadgroups(grid, threadgroup);
        encoder->endEncoding();
    }
    
    void encode_gather_tokens(MTL::ComputeCommandEncoder* encoder, const torch::Tensor& activations, const torch::Tensor& expert_ids, int num_experts, int top_k) {
        auto pipeline = get_pipeline("moe_gather_tokens");
        if (!pipeline) return;
        
        encoder->setComputePipelineState(pipeline);
        
        size_t indices_size = activations.size(0) * top_k * sizeof(int32_t);
        MTL::Buffer* indices_buf = buffer_pool_->acquire(indices_size);
        encoder->setBuffer(indices_buf, 0, 0);
        
        uint32_t batch_size = activations.size(0);
        uint32_t hidden = activations.size(1);
        encoder->setBytes(&batch_size, sizeof(uint32_t), 2);
        encoder->setBytes(&hidden, sizeof(uint32_t), 3);
        encoder->setBytes(&top_k, sizeof(uint32_t), 4);
        
        buffer_pool_->release(indices_buf);
    }
    
    void encode_expert_dispatch(MTL::ComputeCommandEncoder* encoder, int expert_id, const torch::Tensor& weight, int hidden_dim, int intermediate_dim) {
        const char* kernel_name = "moe_expert_forward";
        if (weight.dtype() == at::kInt || weight.dtype() == c10::ScalarType::Byte) {
            kernel_name = "moe_expert_forward_fp4";
        }
        auto pipeline = get_pipeline(kernel_name);
        if (!pipeline) return;
        
        encoder->setComputePipelineState(pipeline);
        encoder->setBytes(&expert_id, sizeof(uint32_t), 0);
        encoder->setBytes(&hidden_dim, sizeof(uint32_t), 1);
        encoder->setBytes(&intermediate_dim, sizeof(uint32_t), 2);
        
        // Direct dispatch with fixed grid size - eliminates indirect buffer overhead
        uint32_t grid_x = (intermediate_dim + 127) / 128;
        MTL::Size grid(grid_x, 1, 1);
        MTL::Size threadgroup(128, 1, 1);
        encoder->dispatchThreadgroups(grid, threadgroup);
    }
    
    void encode_scatter_outputs(MTL::ComputeCommandEncoder* encoder, torch::Tensor& output, const torch::Tensor&, int num_experts, int top_k) {
        auto pipeline = get_pipeline("moe_scatter_outputs");
        if (!pipeline) return;
        
        encoder->setComputePipelineState(pipeline);
        MTL::Buffer* output_buf = (MTL::Buffer*)output.data_ptr();
        encoder->setBuffer(output_buf, 0, 0);
        
        uint32_t batch_size = output.size(0);
        uint32_t hidden = output.size(1);
        encoder->setBytes(&batch_size, sizeof(uint32_t), 1);
        encoder->setBytes(&hidden, sizeof(uint32_t), 2);
        encoder->setBytes(&top_k, sizeof(uint32_t), 3);
        
        MTL::Size grid((batch_size * hidden + 255) / 256, 1, 1);
        MTL::Size threadgroup(256, 1, 1);
        encoder->dispatchThreadgroups(grid, threadgroup);
    }
    
    MTL::Device* device_;
    MTL::Library* library_;
    MTL::CommandQueue* command_queue_;
    std::unique_ptr<BufferPool> buffer_pool_;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
};

// -----------------------------------------------------------------------------
// Public Interface
// -----------------------------------------------------------------------------

MoEDispatcher::MoEDispatcher(MTL::Device* device, MTL::Library* library)
    : impl_(std::make_unique<MoEDispatcherImpl>(device, library)) {}

MoEDispatcher::~MoEDispatcher() = default;
MoEDispatcher::MoEDispatcher(MoEDispatcher&&) noexcept = default;
MoEDispatcher& MoEDispatcher::operator=(MoEDispatcher&&) noexcept = default;

torch::Tensor MoEDispatcher::dispatch(const torch::Tensor& activations, const torch::Tensor& expert_ids, const torch::Tensor& expert_probs, const std::vector<torch::Tensor>& expert_weights, int hidden_dim, int intermediate_dim, int num_experts, int top_k) {
    return impl_->dispatch(activations, expert_ids, expert_probs, expert_weights, hidden_dim, intermediate_dim, num_experts, top_k);
}

void MoEDispatcher::prepare_dispatch(int batch_size, int hidden_dim, int num_experts, int top_k) {
    impl_->prepare_dispatch(batch_size, hidden_dim, num_experts, top_k);
}

void MoEDispatcher::execute_prepared(const torch::Tensor& activations, const torch::Tensor& expert_ids, const torch::Tensor& expert_probs, torch::Tensor& output) {
    impl_->execute_prepared(activations, expert_ids, expert_probs, output);
}

MTL::Buffer* MoEDispatcher::get_temp_buffer(size_t size) { return impl_->get_temp_buffer(size); }
void MoEDispatcher::release_temp_buffer(MTL::Buffer* buffer) { impl_->release_temp_buffer(buffer); }
void MoEDispatcher::wait_until_completed() { impl_->wait_until_completed(); }
bool MoEDispatcher::supports_indirect_command_buffers() const { return impl_->supports_indirect_command_buffers(); }

} // namespace metal_marlin
