/**
 * @file buffer_manager.cpp
 * @brief Zero-copy Buffer sharing between MPS and C++ extension
 */

#include "buffer_manager.hpp"
#include "buffer_pool.hpp"

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <queue>
#include <algorithm>

namespace metal_marlin {

// =============================================================================
// Alignment utilities
// =============================================================================

constexpr size_t CACHE_LINE_SIZE = 128;         // M-series cache line
constexpr size_t PAGE_SIZE = 16 * 1024;         // 16KB page boundary

static inline size_t align_to_cache_line(size_t size) {
    return (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
}

static inline size_t align_buffer_size(size_t size) {
    if (size >= LARGE_BUFFER_THRESHOLD) {
        return (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    }
    return align_to_cache_line(size);
}

// =============================================================================
// MPSBuffer Implementation
// =============================================================================

MPSBuffer::MPSBuffer(MTL::Buffer* buffer,
          size_t size,
          Ownership ownership,
          BufferPoolManager* pool)
    : buffer_(buffer)
    , size_(size)
    , ownership_(ownership)
    , pool_(pool)
{
    if (buffer_) {
        buffer_->retain();
    }
}

MPSBuffer::~MPSBuffer() {
    release();
}

MPSBuffer::MPSBuffer(MPSBuffer&& other) noexcept
    : buffer_(other.buffer_)
    , size_(other.size_)
    , ownership_(other.ownership_)
    , pool_(other.pool_)
{
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.pool_ = nullptr;
}

MPSBuffer& MPSBuffer::operator=(MPSBuffer&& other) noexcept {
    if (this != &other) {
        release();
        buffer_ = other.buffer_;
        size_ = other.size_;
        ownership_ = other.ownership_;
        pool_ = other.pool_;
        other.buffer_ = nullptr;
        other.size_ = 0;
        other.pool_ = nullptr;
    }
    return *this;
}

size_t MPSBuffer::allocated_size() const noexcept {
    return buffer_ ? buffer_->length() : 0;
}

void* MPSBuffer::contents() const noexcept {
    return buffer_ ? buffer_->contents() : nullptr;
}

uint64_t MPSBuffer::gpu_address() const noexcept {
    return buffer_ ? buffer_->gpuAddress() : 0;
}

void MPSBuffer::retain() const {
    if (buffer_) {
        buffer_->retain();
    }
}

void MPSBuffer::release() {
    if (!buffer_) return;

    switch (ownership_) {
        case Ownership::Owned:
            buffer_->release();
            break;

        case Ownership::Wrapped:
            buffer_->release();
            break;

        case Ownership::Pooled:
            return_to_pool();
            break;
    }

    buffer_ = nullptr;
    size_ = 0;
}

void MPSBuffer::did_modify_range(size_t offset, size_t size) {
    if (buffer_) {
        buffer_->didModifyRange(NS::Range::Make(offset, size));
    }
}

void MPSBuffer::return_to_pool() {
    if (pool_ && buffer_) {
        pool_->release_buffer(buffer_, allocated_size());
    } else if (buffer_) {
        buffer_->release();
    }
}

// =============================================================================
// BufferPoolManager Implementation
// =============================================================================

BufferPoolManager::BufferPoolManager(MTL::Device* device,
                           MTL::ResourceOptions options,
                           size_t max_pool_size)
    : device_(device)
    , options_(options)
    , max_pool_size_(max_pool_size)
    , current_pool_size_(0)
{
    if (device_) {
        device_->retain();
    }
}

BufferPoolManager::~BufferPoolManager() {
    clear();
    if (device_) {
        device_->release();
    }
}

MPSBuffer BufferPoolManager::acquire(size_t size) {
    size_t aligned_size = align_buffer_size(size);

    // Try to get from pool first
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        auto it = pools_.find(aligned_size);
        if (it != pools_.end() && !it->second.empty()) {
            MTL::Buffer* buf = it->second.back();
            it->second.pop_back();
            current_pool_size_ -= aligned_size;
            hits_++;

            return MPSBuffer(buf, size, MPSBuffer::Ownership::Pooled, this);
        }
    }

    // Allocate new buffer
    misses_++;
    MTL::Buffer* buf = device_->newBuffer(aligned_size, options_);
    if (!buf) {
        throw std::runtime_error("Failed to allocate Metal buffer of size " +
                                 std::to_string(aligned_size));
    }

    return MPSBuffer(buf, size, MPSBuffer::Ownership::Pooled, this);
}

void BufferPoolManager::release_buffer(MTL::Buffer* buffer, size_t size) {
    if (!buffer) return;

    size_t aligned_size = align_buffer_size(size);

    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Check if we can add to pool, evict if needed
    if (current_pool_size_ + aligned_size > max_pool_size_) {
        evict_to_make_space(aligned_size);
    }
    
    // If still no space, just release the buffer
    if (current_pool_size_ + aligned_size > max_pool_size_) {
        buffer->release();
        return;
    }

    // Add to pool
    pools_[aligned_size].push_back(buffer);
    current_pool_size_ += aligned_size;
}

MPSBuffer BufferPoolManager::wrap_external_memory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return MPSBuffer();
    }

    size_t aligned_size = align_buffer_size(size);

    // Create a buffer that wraps the existing memory
    // Using nullptr deallocator since we don't own the memory
    MTL::Buffer* buf = device_->newBuffer(
        ptr,
        aligned_size,
        options_,
        nullptr  // No deallocator - external owner manages lifetime
    );

    if (!buf) {
        throw std::runtime_error("Failed to wrap external memory at " +
                                 std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    }

    return MPSBuffer(buf, size, MPSBuffer::Ownership::Wrapped);
}

MPSBuffer BufferPoolManager::wrap_mps_tensor(uintptr_t data_ptr, size_t size) {
    return wrap_external_memory(reinterpret_cast<void*>(data_ptr), size);
}

void BufferPoolManager::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto& pair : pools_) {
        for (MTL::Buffer* buf : pair.second) {
            buf->release();
        }
        pair.second.clear();
    }
    pools_.clear();
    current_pool_size_ = 0;
}

double BufferPoolManager::hit_rate() const noexcept {
    uint64_t h = hits_.load();
    uint64_t m = misses_.load();
    uint64_t total = h + m;
    return total > 0 ? static_cast<double>(h) / total : 0.0;
}

size_t BufferPoolManager::pool_size() const noexcept {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return current_pool_size_;
}

size_t BufferPoolManager::pooled_count() const noexcept {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    size_t count = 0;
    for (const auto& pair : pools_) {
        count += pair.second.size();
    }
    return count;
}

void BufferPoolManager::evict_to_make_space(size_t needed) {
    // Simple eviction: remove largest pools first
    // This is called with lock held

    while (current_pool_size_ + needed > max_pool_size_ && !pools_.empty()) {
        // Find the largest pool
        size_t largest_size = 0;
        auto largest_it = pools_.end();

        for (auto it = pools_.begin(); it != pools_.end(); ++it) {
            if (!it->second.empty() && it->first > largest_size) {
                largest_size = it->first;
                largest_it = it;
            }
        }

        if (largest_it == pools_.end() || largest_it->second.empty()) {
            break;
        }

        // Release one buffer from the largest pool
        MTL::Buffer* buf = largest_it->second.back();
        largest_it->second.pop_back();
        buf->release();
        current_pool_size_ -= largest_size;

        // Remove empty pool entries
        if (largest_it->second.empty()) {
            pools_.erase(largest_it);
        }
    }
}

// =============================================================================
// HeapBufferAllocator Implementation
// =============================================================================

HeapBufferAllocator::HeapBufferAllocator(MTL::Device* device,
                              size_t heap_size,
                              MTL::ResourceOptions options)
    : device_(device)
    , heap_size_(heap_size)
    , options_(options)
    , allocated_(0)
{
    if (device_) {
        device_->retain();
        create_heap();
    }
}

HeapBufferAllocator::~HeapBufferAllocator() {
    if (heap_) {
        heap_->release();
    }
    if (device_) {
        device_->release();
    }
}

MPSBuffer HeapBufferAllocator::allocate(size_t size) {
    if (!heap_) {
        throw std::runtime_error("Heap not initialized");
    }

    size_t aligned_size = align_buffer_size(size);

    // Check if there's enough space
    if (allocated_.load() + aligned_size > heap_size_) {
        throw std::runtime_error("Heap out of memory");
    }

    MTL::Buffer* buf = heap_->newBuffer(aligned_size, options_);
    if (!buf) {
        throw std::runtime_error("Failed to allocate from heap");
    }

    allocated_.fetch_add(aligned_size);
    return MPSBuffer(buf, size, MPSBuffer::Ownership::Owned);
}

void HeapBufferAllocator::create_heap() {
    MTL::HeapDescriptor* desc = MTL::HeapDescriptor::alloc()->init();
    desc->setSize(heap_size_);
    desc->setResourceOptions(options_);

    heap_ = device_->newHeap(desc);
    desc->release();

    if (!heap_) {
        throw std::runtime_error("Failed to create Metal heap");
    }
}

// =============================================================================
// BufferManager Implementation
// =============================================================================

BufferManager::BufferManager(MTL::Device* device)
    : device_(device)
    , pool_manager_(device)
{
    if (device_) {
        device_->retain();
    }
}

BufferManager::~BufferManager() {
    if (device_) {
        device_->release();
    }
}

MPSBuffer BufferManager::wrap_mps_tensor(uintptr_t data_ptr, size_t size) {
    return pool_manager_.wrap_mps_tensor(data_ptr, size);
}

MPSBuffer BufferManager::allocate(size_t size) {
    return pool_manager_.acquire(size);
}

MPSBuffer BufferManager::allocate_with_options(size_t size, MTL::ResourceOptions options) {
    size_t aligned_size = align_buffer_size(size);
    MTL::Buffer* buf = device_->newBuffer(aligned_size, options);
    if (!buf) {
        throw std::runtime_error("Failed to allocate buffer with options");
    }
    return MPSBuffer(buf, size, MPSBuffer::Ownership::Owned);
}

MPSBuffer BufferManager::create_shared_buffer(size_t size) {
    return allocate_with_options(size, MTL::ResourceStorageModeShared);
}

MPSBuffer BufferManager::create_private_buffer(size_t size) {
    return allocate_with_options(size, MTL::ResourceStorageModePrivate);
}

MPSBuffer BufferManager::create_managed_buffer(size_t size) {
    return allocate_with_options(size, MTL::ResourceStorageModeManaged);
}

// =============================================================================
// C API for Python interop
// =============================================================================

extern "C" {

void* buffer_manager_create(void* device_ptr) {
    try {
        MTL::Device* device = static_cast<MTL::Device*>(device_ptr);
        return new BufferManager(device);
    } catch (...) {
        return nullptr;
    }
}

void buffer_manager_destroy(void* manager_ptr) {
    if (manager_ptr) {
        delete static_cast<BufferManager*>(manager_ptr);
    }
}

void* buffer_manager_wrap_mps_tensor(void* manager_ptr, uintptr_t data_ptr, size_t size) {
    if (!manager_ptr) return nullptr;

    try {
        auto* manager = static_cast<BufferManager*>(manager_ptr);
        MPSBuffer* buffer = new MPSBuffer(manager->wrap_mps_tensor(data_ptr, size));
        return buffer;
    } catch (...) {
        return nullptr;
    }
}

void* buffer_manager_allocate(void* manager_ptr, size_t size) {
    if (!manager_ptr) return nullptr;

    try {
        auto* manager = static_cast<BufferManager*>(manager_ptr);
        MPSBuffer* buffer = new MPSBuffer(manager->allocate(size));
        return buffer;
    } catch (...) {
        return nullptr;
    }
}

void* buffer_manager_create_shared(void* manager_ptr, size_t size) {
    if (!manager_ptr) return nullptr;

    try {
        auto* manager = static_cast<BufferManager*>(manager_ptr);
        MPSBuffer* buffer = new MPSBuffer(manager->create_shared_buffer(size));
        return buffer;
    } catch (...) {
        return nullptr;
    }
}

void mps_buffer_release(void* buffer_ptr) {
    if (buffer_ptr) {
        delete static_cast<MPSBuffer*>(buffer_ptr);
    }
}

void* mps_buffer_get_mtl_buffer(void* buffer_ptr) {
    if (!buffer_ptr) return nullptr;
    auto* buffer = static_cast<MPSBuffer*>(buffer_ptr);
    return buffer->buffer();
}

size_t mps_buffer_get_size(void* buffer_ptr) {
    if (!buffer_ptr) return 0;
    auto* buffer = static_cast<MPSBuffer*>(buffer_ptr);
    return buffer->size();
}

void* mps_buffer_get_contents(void* buffer_ptr) {
    if (!buffer_ptr) return nullptr;
    auto* buffer = static_cast<MPSBuffer*>(buffer_ptr);
    return buffer->contents();
}

uint64_t mps_buffer_get_gpu_address(void* buffer_ptr) {
    if (!buffer_ptr) return 0;
    auto* buffer = static_cast<MPSBuffer*>(buffer_ptr);
    return buffer->gpu_address();
}

void mps_buffer_did_modify_range(void* buffer_ptr, size_t offset, size_t size) {
    if (!buffer_ptr) return;
    auto* buffer = static_cast<MPSBuffer*>(buffer_ptr);
    buffer->did_modify_range(offset, size);
}

double buffer_manager_get_hit_rate(void* manager_ptr) {
    if (!manager_ptr) return 0.0;
    auto* manager = static_cast<BufferManager*>(manager_ptr);
    return manager->pool_hit_rate();
}

void buffer_manager_get_stats(void* manager_ptr,
                               uint64_t* out_hits,
                               uint64_t* out_misses,
                               size_t* out_pool_size,
                               size_t* out_pooled_count) {
    if (!manager_ptr) return;

    auto* manager = static_cast<BufferManager*>(manager_ptr);

    if (out_hits) *out_hits = manager->pool_hits();
    if (out_misses) *out_misses = manager->pool_misses();
    if (out_pool_size) *out_pool_size = manager->pool_size();
    if (out_pooled_count) *out_pooled_count = manager->pooled_count();
}

void buffer_manager_clear_pool(void* manager_ptr) {
    if (!manager_ptr) return;
    auto* manager = static_cast<BufferManager*>(manager_ptr);
    manager->clear_pool();
}

} // extern "C"

} // namespace metal_marlin