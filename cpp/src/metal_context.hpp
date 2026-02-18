#pragma once

#include <Metal/Metal.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <atomic>

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

constexpr size_t CACHE_LINE_SIZE = 128;         // M3 Max cache line
constexpr size_t MM_PAGE_SIZE = 16 * 1024;         // 16KB page alignment for large buffers
constexpr size_t LARGE_BUFFER_THRESHOLD_CORE = 64 * 1024;  // 64KB
constexpr size_t ASYNC_TRANSFER_THRESHOLD = 1024 * 1024;  // 1MB

// -----------------------------------------------------------------------------
// Buffer alignment utilities
// -----------------------------------------------------------------------------

inline size_t align_to_cache_line(size_t size) {
    return (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
}

inline size_t align_buffer_size(size_t size) {
    if (size >= LARGE_BUFFER_THRESHOLD_CORE) {
        return (size + MM_PAGE_SIZE - 1) & ~(MM_PAGE_SIZE - 1);
    }
    return align_to_cache_line(size);
}

// -----------------------------------------------------------------------------
// BufferPool - Lock-free buffer reuse for transient allocations
// -----------------------------------------------------------------------------

class BufferPool {
public:
    explicit BufferPool(MTL::Device* device, MTL::ResourceOptions options = MTL::ResourceStorageModeShared);
    ~BufferPool();

    MTL::Buffer* get(size_t size);
    void release(MTL::Buffer* buf);

    // Stats for diagnostics
    uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }

    double hit_rate() const;
    size_t pooled_count() const;
    size_t pooled_bytes() const;
    void clear();

private:
    MTL::Device* device_;
    MTL::ResourceOptions options_;
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<MTL::Buffer*>> pools_;
    std::atomic<uint64_t> hits_;
    std::atomic<uint64_t> misses_;
};

// -----------------------------------------------------------------------------
// MetalContext - Core Metal state management
// -----------------------------------------------------------------------------

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    // Non-copyable
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    MTL::Device* device() const { return device_; }
    MTL::CommandQueue* command_queue() const { return command_queue_; }
    MTL::CommandQueue* decode_queue() const { return decode_queue_; }
    BufferPool* buffer_pool() const { return buffer_pool_.get(); }

    // Load precompiled metallib
    void load_metallib(const std::string& path);

    // Get or create pipeline state from a specific library key.
    // If lib is empty, search all loaded libraries.
    MTL::ComputePipelineState* get_or_create_pipeline(
        const std::string& lib,
        const std::string& function_name
    );

    // Backward-compatible pipeline lookup API
    MTL::ComputePipelineState* get_pipeline(
        const std::string& function_name,
        const std::string& metallib_path = ""
    );

    // Compile Metal source at runtime (for testing/development)
    void compile_source(const std::string& name, const std::string& source);

    // Get GPU family (7=M1, 8=M2, 9=M3+)
    int gpu_family() const;

    std::string device_name() const;

private:
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* command_queue_ = nullptr;
    MTL::CommandQueue* decode_queue_ = nullptr;
    std::unique_ptr<BufferPool> buffer_pool_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, MTL::Library*> libraries_;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipeline_cache_;
};

// -----------------------------------------------------------------------------
// Buffer wrapper for Python lifetime management
// -----------------------------------------------------------------------------

class ManagedBuffer {
public:
    explicit ManagedBuffer(MTL::Buffer* buffer, BufferPool* pool = nullptr)
        : buffer_(buffer), pool_(pool) {}

    ~ManagedBuffer() {
        if (buffer_) {
            if (pool_) {
                pool_->release(buffer_);
            } else {
                buffer_->release();
            }
        }
    }

    // Non-copyable, movable
    ManagedBuffer(const ManagedBuffer&) = delete;
    ManagedBuffer& operator=(const ManagedBuffer&) = delete;
    ManagedBuffer(ManagedBuffer&& other) noexcept
        : buffer_(other.buffer_), pool_(other.pool_) {
        other.buffer_ = nullptr;
        other.pool_ = nullptr;
    }

    MTL::Buffer* get() const { return buffer_; }
    size_t length() const { return buffer_ ? buffer_->length() : 0; }
    void* contents() const { return buffer_ ? buffer_->contents() : nullptr; }

    // Direct memory access for numpy interop
    uintptr_t data_ptr() const {
        return buffer_ ? reinterpret_cast<uintptr_t>(buffer_->contents()) : 0;
    }

private:
    MTL::Buffer* buffer_;
    BufferPool* pool_;
};
