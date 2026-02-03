// encoder_cache.hpp - CommandEncoder caching for Metal Marlin
//
// This module provides a high-performance CommandEncoder caching mechanism
// to reduce CPU overhead per dispatch. Command buffer creation in Metal has
// significant overhead (~1-2µs), which becomes the bottleneck in decode
// loops with thousands of small dispatches.
//
// Key optimizations:
// - Object pooling for MTLCommandBuffer (avoid malloc/free in hot path)
// - Reset-and-reuse pattern for command buffers
// - Lock-free fast path for encode operations
// - Thread-safe command buffer rotation
//
// Thread Safety:
// - acquire_encoder() is thread-safe when using separate pools per thread
// - commit_encoder() is thread-safe via atomic buffer rotation
// - Multiple threads can encode concurrently if using separate pools
//
// Usage:
//   EncoderCache cache(queue, 4);  // Pool of 4 buffers
//   auto* encoder = cache.acquire_encoder();
//   // ... encode dispatches ...
//   cache.commit_encoder(encoder);
//   // ... wait for completion ...
//   cache.release_encoder(encoder);

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
// Forward declarations for non-ObjC++ compilation
typedef void* MTLCommandQueue;
typedef void* MTLCommandBuffer;
typedef void* MTLComputeCommandEncoder;
typedef void* MTLComputePipelineState;
typedef void* MTLBuffer;
#endif

namespace metal_marlin {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default pool size for command buffers
constexpr size_t kDefaultPoolSize = 4;

/// Maximum number of command buffers to track for completion
constexpr size_t kMaxInflightBuffers = 8;

// -----------------------------------------------------------------------------
// Encoder entry for the pool
// -----------------------------------------------------------------------------

struct EncoderEntry {
#ifdef __OBJC__
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> compute_encoder = nil;
#else
    void* command_buffer = nullptr;
    void* compute_encoder = nullptr;
#endif
    std::atomic<bool> in_use{false};
    std::atomic<bool> committed{false};
    size_t dispatch_count{0};

    EncoderEntry() = default;

    // Custom move constructor to handle atomics
    EncoderEntry(EncoderEntry&& other) noexcept
        : command_buffer(other.command_buffer),
          compute_encoder(other.compute_encoder),
          in_use(other.in_use.load()),
          committed(other.committed.load()),
          dispatch_count(other.dispatch_count) {
#ifdef __OBJC__
        other.command_buffer = nil;
        other.compute_encoder = nil;
#else
        other.command_buffer = nullptr;
        other.compute_encoder = nullptr;
#endif
    }

    EncoderEntry& operator=(EncoderEntry&& other) noexcept {
        if (this != &other) {
            command_buffer = other.command_buffer;
            compute_encoder = other.compute_encoder;
            in_use.store(other.in_use.load());
            committed.store(other.committed.load());
            dispatch_count = other.dispatch_count;
#ifdef __OBJC__
            other.command_buffer = nil;
            other.compute_encoder = nil;
#else
            other.command_buffer = nullptr;
            other.compute_encoder = nullptr;
#endif
        }
        return *this;
    }

    // Non-copyable
    EncoderEntry(const EncoderEntry&) = delete;
    EncoderEntry& operator=(const EncoderEntry&) = delete;
};

// -----------------------------------------------------------------------------
// Command buffer pool statistics
// -----------------------------------------------------------------------------

struct EncoderCacheStats {
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> total_acquires{0};
    std::atomic<uint64_t> total_commits{0};
    std::atomic<uint64_t> total_releases{0};
    std::atomic<uint64_t> current_in_use{0};
    std::atomic<uint64_t> peak_in_use{0};

    [[nodiscard]] double hit_rate() const;
    void reset();
};

// -----------------------------------------------------------------------------
// Encoder cache - reusable command buffer pool
// -----------------------------------------------------------------------------

class EncoderCache {
public:
    EncoderCache() = default;

#ifdef __OBJC__
    /// Initialize encoder cache with command queue
    /// @param queue Metal command queue for creating command buffers
    /// @param pool_size Number of command buffers in pool (default 4)
    explicit EncoderCache(id<MTLCommandQueue> queue, size_t pool_size = kDefaultPoolSize);
#endif

    ~EncoderCache();

    // Non-copyable, non-movable due to atomics/mutex
    EncoderCache(const EncoderCache&) = delete;
    EncoderCache& operator=(const EncoderCache&) = delete;
    EncoderCache(EncoderCache&&) = delete;
    EncoderCache& operator=(EncoderCache&&) = delete;

    // -------------------------------------------------------------------------
    // Encoder lifecycle
    // -------------------------------------------------------------------------

#ifdef __OBJC__
    /// Acquire a compute command encoder from the pool
    /// Creates a new encoder if needed from a recycled command buffer
    /// @return Compute command encoder pointer, or nullptr if all buffers in use
    [[nodiscard]] id<MTLComputeCommandEncoder> acquire_encoder();

    /// Commit a command buffer from the pool
    /// Ends encoding and commits for async execution
    /// @param encoder The encoder to commit
    void commit_encoder(id<MTLComputeCommandEncoder> encoder);

    /// Release a command buffer back to the pool
    /// Waits for completion before recycling
    /// @param encoder The encoder to release
    void release_encoder(id<MTLComputeCommandEncoder> encoder);

    /// Wait for all in-flight command buffers to complete
    void wait_all();

    /// Wait for a specific number of command buffers to complete
    /// @param count Number of buffers to wait for
    void wait_for(size_t count);

    /// Get a command buffer for direct encoding (without encoder pooling)
    /// Useful for blit encoders or specialized use cases
    [[nodiscard]] id<MTLCommandBuffer> acquire_command_buffer();

    /// Release a manually acquired command buffer
    void release_command_buffer(id<MTLCommandBuffer> buffer);
#endif

    // -------------------------------------------------------------------------
    // Statistics and diagnostics
    // -------------------------------------------------------------------------

    /// Get cache statistics
    [[nodiscard]] const EncoderCacheStats& stats() const noexcept { return stats_; }

    /// Get pool size
    [[nodiscard]] size_t pool_size() const noexcept { return pool_size_; }

    /// Get number of in-use encoders
    [[nodiscard]] size_t in_use_count() const noexcept {
        return stats_.current_in_use.load(std::memory_order_relaxed);
    }

    /// Get number of in-flight command buffers
    [[nodiscard]] size_t inflight_count() const noexcept {
        return inflight_count_.load(std::memory_order_relaxed);
    }

    /// Reset statistics
    void reset_stats() { stats_.reset(); }

    /// Resize the pool (destructive - resets all buffers)
    /// @param new_size New pool size
    void resize(size_t new_size);

    /// Clear all in-flight buffers
    void clear_inflight();

private:
    void update_peak_in_use();

#ifdef __OBJC__
    id<MTLCommandQueue> queue_ = nil;
#else
    void* queue_ = nullptr;
#endif

    size_t pool_size_;
    std::atomic<size_t> next_free_index_;

    std::vector<EncoderEntry> pool_;
    std::vector<void*> inflight_buffers_;  // Track committed buffers
    std::atomic<size_t> inflight_count_;
    std::mutex publish_mutex_;  // Protects inflight_buffers_ tracking

    EncoderCacheStats stats_;
};

// -----------------------------------------------------------------------------
// Per-thread encoder cache for lock-free encoding
// -----------------------------------------------------------------------------

class ThreadLocalEncoderCache {
public:
    ThreadLocalEncoderCache() = default;

#ifdef __OBJC__
    explicit ThreadLocalEncoderCache(id<MTLCommandQueue> queue)
        : cache_(queue, kDefaultPoolSize) {}
#endif

    // Non-copyable, non-movable
    ThreadLocalEncoderCache(const ThreadLocalEncoderCache&) = delete;
    ThreadLocalEncoderCache& operator=(const ThreadLocalEncoderCache&) = delete;
    ThreadLocalEncoderCache(ThreadLocalEncoderCache&&) = delete;
    ThreadLocalEncoderCache& operator=(ThreadLocalEncoderCache&&) = delete;

#ifdef __OBJC__
    [[nodiscard]] id<MTLComputeCommandEncoder> acquire() {
        return cache_.acquire_encoder();
    }

    void commit(id<MTLComputeCommandEncoder> encoder) {
        cache_.commit_encoder(encoder);
    }

    void release(id<MTLComputeCommandEncoder> encoder) {
        cache_.release_encoder(encoder);
    }

    void wait_all() {
        cache_.wait_all();
    }
#endif

    const EncoderCacheStats& stats() const { return cache_.stats(); }

private:
    EncoderCache cache_;
};

}  // namespace metal_marlin
