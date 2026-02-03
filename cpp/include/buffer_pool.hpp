// buffer_pool.hpp - GIL-free Metal buffer pool for Apple Silicon
//
// This C++ implementation releases the Python GIL during Metal allocation,
// allowing other Python threads to run while waiting for buffer allocation.
// This is critical for multi-threaded inference where the GIL would otherwise
// serialize all allocation requests.
//
// Design:
// - MTLHeap-based sub-allocation for reduced allocation overhead
// - Priority-based eviction (PINNED > HIGH > NORMAL > LOW)
// - Lock-free fast path for common allocations
// - GIL released during actual Metal API calls
//
// Thread Safety:
// - All public methods are thread-safe
// - Uses fine-grained locking to minimize contention
// - Allocation from pool is lock-free when buffers are available

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <chrono>
#include <optional>

// Forward declarations for Metal types (defined in Metal/Metal.h)
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLBuffer;
@protocol MTLHeap;
@class MTLHeapDescriptor;
typedef id<MTLDevice> MTLDeviceRef;
typedef id<MTLBuffer> MTLBufferRef;
typedef id<MTLHeap> MTLHeapRef;
#else
// C++ forward declarations
namespace MTL {
    class Device;
    class Buffer;
    class Heap;
}
typedef MTL::Device* MTLDeviceRef;
typedef MTL::Buffer* MTLBufferRef;
typedef MTL::Heap* MTLHeapRef;
#endif

namespace metal_marlin {

// Buffer eviction priority. Lower values = evicted first.
enum class BufferPriority : int {
    LOW = 0,     // Output/temporary buffers, evict first
    NORMAL = 1,  // General purpose buffers
    HIGH = 2,    // Activation buffers, hot path
    PINNED = 3   // Weight buffers, never evict
};

// Metal storage modes
enum class StorageMode : int {
    SHARED = 0,      // CPU + GPU accessible (unified memory)
    MANAGED = 1,     // Automatic sync between CPU/GPU
    PRIVATE = 2,     // GPU only
    MEMORYLESS = 3   // Tile memory only
};

// Alignment constants for Apple Silicon
constexpr size_t CACHE_LINE_BYTES = 128;      // M-series cache line
constexpr size_t PAGE_SIZE_BYTES = 16384;     // 16KB page boundary
constexpr size_t LARGE_BUFFER_THRESHOLD = 65536;

// Buffer metadata for tracking allocations
struct BufferMetadata {
    size_t size;
    BufferPriority priority;
    std::atomic<int> ref_count{0};
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_acquired_at;
    std::atomic<uint64_t> acquisition_count{0};
    std::atomic<uint64_t> total_time_in_pool_ns{0};
    std::chrono::steady_clock::time_point last_release_at;

    BufferMetadata(size_t s, BufferPriority p)
        : size(s), priority(p), created_at(std::chrono::steady_clock::now()) {}
};

// Metrics for pool performance tracking
struct BufferPoolMetrics {
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> allocations{0};
    std::atomic<uint64_t> releases{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> bytes_evicted{0};
    std::atomic<uint64_t> current_allocated{0};
    std::atomic<uint64_t> current_pooled{0};
    std::atomic<uint64_t> peak_allocated{0};
    std::atomic<uint64_t> peak_total{0};

    double hit_rate() const {
        uint64_t total = cache_hits.load() + cache_misses.load();
        return total > 0 ? static_cast<double>(cache_hits.load()) / total : 0.0;
    }

    void reset() {
        cache_hits = 0;
        cache_misses = 0;
        allocations = 0;
        releases = 0;
        evictions = 0;
        bytes_evicted = 0;
        current_allocated = 0;
        current_pooled = 0;
        // Don't reset peak values
    }
};

// Buffer handle returned to Python
// Contains both the buffer reference and metadata
struct BufferHandle {
    MTLBufferRef buffer;
    size_t size;
    size_t aligned_size;
    BufferPriority priority;
    uint64_t buffer_id;  // Unique ID for tracking

    // For Python buffer protocol support
    void* contents() const;
    size_t length() const { return size; }
};

// Main buffer pool class
class BufferPool {
public:
    // Constructor - takes device pointer from Python
    // heap_size: Total heap size in bytes (default 256MB)
    // storage_mode: Metal storage mode (default SHARED for unified memory)
    explicit BufferPool(
        void* device_ptr,
        size_t heap_size = 256 * 1024 * 1024,
        StorageMode storage_mode = StorageMode::SHARED
    );

    ~BufferPool();

    // Non-copyable, non-movable (due to Metal resources)
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = delete;
    BufferPool& operator=(BufferPool&&) = delete;

    // Allocate buffer from pool
    // Returns nullptr on failure
    // GIL is released during Metal allocation
    std::optional<BufferHandle> allocate(
        size_t size,
        BufferPriority priority = BufferPriority::NORMAL
    );

    // Release buffer back to pool
    void release(const BufferHandle& handle);

    // Pin buffer to prevent eviction
    void pin(const BufferHandle& handle);

    // Unpin buffer to allow eviction
    void unpin(const BufferHandle& handle);

    // Evict buffers to free memory
    // Returns bytes actually freed
    size_t evict(size_t target_bytes);

    // Compact pool by merging similar-sized buffers
    void compact();

    // Clear entire pool
    void clear();

    // Get current metrics
    const BufferPoolMetrics& metrics() const { return metrics_; }

    // Get heap statistics
    size_t heap_size() const { return heap_size_; }
    size_t allocated_bytes() const { return metrics_.current_allocated.load(); }
    size_t pooled_bytes() const { return metrics_.current_pooled.load(); }
    size_t available_bytes() const;

    // Convenience allocation methods with priority hints
    std::optional<BufferHandle> allocate_weight(size_t size) {
        return allocate(size, BufferPriority::PINNED);
    }
    std::optional<BufferHandle> allocate_activation(size_t size) {
        return allocate(size, BufferPriority::HIGH);
    }
    std::optional<BufferHandle> allocate_output(size_t size) {
        return allocate(size, BufferPriority::LOW);
    }

private:
    // Align size to appropriate boundary
    size_t align_size(size_t size) const;

    // Try to get buffer from pool (lock-free fast path)
    std::optional<BufferHandle> try_get_from_pool(size_t aligned_size, BufferPriority priority);

    // Allocate new buffer from heap (slow path, releases GIL)
    std::optional<BufferHandle> allocate_new(size_t aligned_size, BufferPriority priority);

    // Update peak metrics
    void update_peak_metrics();

    // Generate unique buffer ID
    uint64_t next_buffer_id() { return buffer_id_counter_.fetch_add(1); }

    // Metal resources
    MTLDeviceRef device_;
    MTLHeapRef heap_;
    size_t heap_size_;
    StorageMode storage_mode_;

    // Pool storage: aligned_size -> vector of (buffer, metadata)
    using PoolEntry = std::pair<MTLBufferRef, std::shared_ptr<BufferMetadata>>;
    std::unordered_map<size_t, std::vector<PoolEntry>> pool_;
    mutable std::shared_mutex pool_mutex_;

    // Active allocations: buffer_id -> metadata
    std::unordered_map<uint64_t, std::pair<MTLBufferRef, std::shared_ptr<BufferMetadata>>> active_;
    mutable std::mutex active_mutex_;

    // Metrics
    BufferPoolMetrics metrics_;

    // Buffer ID counter
    std::atomic<uint64_t> buffer_id_counter_{0};
};

// Ring buffer for transient per-forward-pass allocations
// Provides O(1) reset between forward passes
class TransientRingBuffer {
public:
    explicit TransientRingBuffer(
        void* device_ptr,
        size_t capacity = 100 * 1024 * 1024,  // 100MB default
        StorageMode storage_mode = StorageMode::SHARED
    );

    ~TransientRingBuffer();

    // Allocate from ring buffer
    // Returns (buffer, offset) or nullopt on failure
    std::optional<std::pair<MTLBufferRef, size_t>> allocate(size_t size);

    // Allocate and return raw bytes (for CPU-side memoryview)
    std::optional<std::pair<void*, size_t>> allocate_bytes(size_t size);

    // Reset ring buffer (O(1) operation)
    void reset();

    // Statistics
    size_t capacity() const { return capacity_; }
    size_t used() const { return offset_.load(); }
    size_t available() const { return capacity_ - offset_.load(); }
    double utilization() const {
        return capacity_ > 0 ? static_cast<double>(offset_.load()) / capacity_ : 0.0;
    }

private:
    size_t align_offset(size_t offset) const {
        return (offset + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1);
    }

    MTLDeviceRef device_;
    MTLBufferRef buffer_;
    size_t capacity_;
    StorageMode storage_mode_;
    std::atomic<size_t> offset_{0};
    mutable std::mutex alloc_mutex_;  // For thread-safe allocation
};

// Heap allocator for fine-grained sub-allocation
// Uses MTLHeap for better memory locality
class HeapAllocator {
public:
    explicit HeapAllocator(
        void* device_ptr,
        size_t heap_size = 256 * 1024 * 1024,
        StorageMode storage_mode = StorageMode::SHARED
    );

    ~HeapAllocator();

    // Allocate from heap
    std::optional<std::pair<MTLBufferRef, size_t>> allocate(size_t size);

    // Allocate buffer only (offset tracked internally)
    std::optional<MTLBufferRef> allocate_buffer(size_t size);

    // Get offset for buffer (if allocated from this heap)
    std::optional<size_t> get_offset(MTLBufferRef buffer);

    // Release buffer back to pool
    void release(MTLBufferRef buffer);

    // Compact pool
    void compact();

    // Clear pool
    void clear();

    // Statistics
    size_t heap_size() const { return heap_size_; }
    size_t allocated_bytes() const { return allocated_.load(); }
    size_t available_bytes() const { return heap_size_ - allocated_.load(); }

private:
    MTLDeviceRef device_;
    MTLHeapRef heap_;
    size_t heap_size_;
    StorageMode storage_mode_;

    // Pool for freed buffers: size -> vector of buffers
    std::unordered_map<size_t, std::vector<MTLBufferRef>> pool_;
    mutable std::shared_mutex pool_mutex_;

    // Track buffer offsets
    std::unordered_map<uintptr_t, size_t> offsets_;
    mutable std::mutex offset_mutex_;

    std::atomic<size_t> allocated_{0};
};

}  // namespace metal_marlin
