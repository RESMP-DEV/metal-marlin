#include "expert_buffer_pool.hpp"
#include <stdexcept>

namespace metal_marlin {

ExpertBufferPool& ExpertBufferPool::instance() {
    static ExpertBufferPool instance;
    return instance;
}

void ExpertBufferPool::initialize(void* device_ptr, size_t heap_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_) {
        // Use PINNED priority for weights
        // Create pool with managed or shared memory depending on device capability
        // For now, default to SHARED as it works on all Apple Silicon
        pool_ = std::make_unique<BufferPool>(device_ptr, heap_size, StorageMode::SHARED);
    }
}

bool ExpertBufferPool::is_initialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_ != nullptr;
}

BufferPool& ExpertBufferPool::get_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_) {
        throw std::runtime_error("ExpertBufferPool not initialized. Call initialize() first.");
    }
    return *pool_;
}

std::optional<BufferHandle> ExpertBufferPool::allocate_weight(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_) {
        return std::nullopt;
    }
    // Weights are pinned by default in this specialized pool
    return pool_->allocate(size, BufferPriority::PINNED);
}

void ExpertBufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_) {
        pool_->clear();
    }
}

const BufferPoolMetrics* ExpertBufferPool::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_) {
        return &pool_->metrics();
    }
    return nullptr;
}

} // namespace metal_marlin
