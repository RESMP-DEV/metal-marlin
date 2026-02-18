#pragma once

#include "buffer_pool.hpp"

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace metal_marlin {

// Forward declarations
class BufferPoolManager;

/**
 * @brief Represents a zero-copy Metal buffer that may wrap MPS tensor memory.
 */
class MPSBuffer {
public:
    enum class Ownership {
        Owned,      // Buffer allocated by us, we manage lifetime
        Wrapped,    // Wrapping external memory (e.g., MPS tensor), don't free
        Pooled      // From buffer pool, return on release
    };

    MPSBuffer() = default;

    MPSBuffer(MTL::Buffer* buffer,
              size_t size,
              Ownership ownership = Ownership::Owned,
              BufferPoolManager* pool = nullptr);

    ~MPSBuffer();

    MPSBuffer(const MPSBuffer&) = delete;
    MPSBuffer& operator=(const MPSBuffer&) = delete;

    MPSBuffer(MPSBuffer&& other) noexcept;
    MPSBuffer& operator=(MPSBuffer&& other) noexcept;

    MTL::Buffer* buffer() const noexcept { return buffer_; }
    size_t size() const noexcept { return size_; }
    size_t allocated_size() const noexcept;
    void* contents() const noexcept;
    uint64_t gpu_address() const noexcept;
    bool is_valid() const noexcept { return buffer_ != nullptr; }
    Ownership ownership() const noexcept { return ownership_; }
    
    void retain() const;
    void release();
    void did_modify_range(size_t offset, size_t size);

private:
    void return_to_pool();

    MTL::Buffer* buffer_ = nullptr;
    size_t size_ = 0;
    Ownership ownership_ = Ownership::Owned;
    BufferPoolManager* pool_ = nullptr;
};

/**
 * @brief Efficient buffer reuse manager.
 */
class BufferPoolManager {
public:
    explicit BufferPoolManager(MTL::Device* device,
                               MTL::ResourceOptions options = MTL::ResourceStorageModeShared,
                               size_t max_pool_size = 256 * 1024 * 1024);

    ~BufferPoolManager();

    MPSBuffer acquire(size_t size);
    void release_buffer(MTL::Buffer* buffer, size_t size);
    MPSBuffer wrap_external_memory(void* ptr, size_t size);
    MPSBuffer wrap_mps_tensor(uintptr_t data_ptr, size_t size);
    void clear();

    uint64_t hits() const noexcept { return hits_.load(); }
    uint64_t misses() const noexcept { return misses_.load(); }
    double hit_rate() const noexcept;
    size_t pool_size() const noexcept;
    size_t pooled_count() const noexcept;

private:
    void evict_to_make_space(size_t needed);

    MTL::Device* device_;
    MTL::ResourceOptions options_;
    size_t max_pool_size_;
    size_t current_pool_size_;

    mutable std::mutex pool_mutex_;
    std::unordered_map<size_t, std::vector<MTL::Buffer*>> pools_;

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
};

/**
 * @brief Buffer allocator using MTLHeap for improved locality.
 */
class HeapBufferAllocator {
public:
    explicit HeapBufferAllocator(MTL::Device* device,
                                  size_t heap_size = 256 * 1024 * 1024,
                                  MTL::ResourceOptions options = MTL::ResourceStorageModeShared);

    ~HeapBufferAllocator();

    MPSBuffer allocate(size_t size);
    size_t heap_size() const noexcept { return heap_size_; }
    size_t allocated() const noexcept { return allocated_.load(); }
    size_t available() const noexcept { return heap_size_ - allocated_.load(); }

private:
    void create_heap();

    MTL::Device* device_;
    MTL::Heap* heap_ = nullptr;
    size_t heap_size_;
    MTL::ResourceOptions options_;
    std::atomic<size_t> allocated_{0};
};

/**
 * @brief Central manager for all buffer operations.
 */
class BufferManager {
public:
    explicit BufferManager(MTL::Device* device);
    ~BufferManager();

    MPSBuffer wrap_mps_tensor(uintptr_t data_ptr, size_t size);
    MPSBuffer allocate(size_t size);
    MPSBuffer allocate_with_options(size_t size, MTL::ResourceOptions options);
    MPSBuffer create_shared_buffer(size_t size);
    MPSBuffer create_private_buffer(size_t size);
    MPSBuffer create_managed_buffer(size_t size);

    double pool_hit_rate() const { return pool_manager_.hit_rate(); }
    uint64_t pool_hits() const { return pool_manager_.hits(); }
    uint64_t pool_misses() const { return pool_manager_.misses(); }
    size_t pool_size() const { return pool_manager_.pool_size(); }
    size_t pooled_count() const { return pool_manager_.pooled_count(); }
    void clear_pool() { pool_manager_.clear(); }

private:
    MTL::Device* device_;
    BufferPoolManager pool_manager_;
};

} // namespace metal_marlin
