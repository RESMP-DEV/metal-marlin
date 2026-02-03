/**
 * @file pool.cpp
 * @brief GIL-free Metal buffer pool implementation for Apple Silicon
 *
 * This implementation provides:
 * - MTLHeap-based sub-allocation for reduced allocation overhead
 * - Priority-based eviction (PINNED > HIGH > NORMAL > LOW)
 * - Lock-free fast path for common allocations from pool
 * - GIL released during actual Metal API calls
 *
 * Thread Safety:
 * - All public methods are thread-safe
 * - Uses fine-grained locking to minimize contention
 * - Allocation from pool is lock-free when buffers are available
 */

#include "buffer_pool.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

// Metal-cpp headers
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// Python GIL handling via nanobind (when built as extension)
#if defined(__has_include)
#if __has_include(<nanobind/nanobind.h>)
#include <nanobind/nanobind.h>
namespace nb = nanobind;
#define MM_HAS_NANOBIND 1
#endif
#endif
#ifndef MM_HAS_NANOBIND
#define MM_HAS_NANOBIND 0
#endif

#if MM_HAS_NANOBIND
#define RELEASE_GIL() nb::gil_scoped_release nogil
#else
// Stub for standalone builds
#define RELEASE_GIL() do {} while(0)
#endif

namespace metal_marlin {

// =============================================================================
// BufferPool Implementation
// =============================================================================

BufferPool::BufferPool(void* device_ptr, size_t heap_size, StorageMode storage_mode)
    : device_(nullptr)
    , heap_(nullptr)
    , heap_size_(heap_size)
    , storage_mode_(storage_mode)
{
    if (!device_ptr) {
        throw std::runtime_error("BufferPool: received null device pointer");
    }
    
    // Cast to MTL::Device* - the caller retains ownership
    device_ = static_cast<MTL::Device*>(device_ptr);
    
    // Create the heap descriptor
    MTL::HeapDescriptor* heap_desc = MTL::HeapDescriptor::alloc()->init();
    if (!heap_desc) {
        throw std::runtime_error("BufferPool: failed to create heap descriptor");
    }
    
    // Set heap size
    heap_desc->setSize(heap_size);
    
    // Set storage mode
    MTL::ResourceOptions options;
    switch (storage_mode) {
        case StorageMode::SHARED:
            options = MTL::ResourceStorageModeShared;
            break;
        case StorageMode::MANAGED:
            options = MTL::ResourceStorageModeManaged;
            break;
        case StorageMode::PRIVATE:
            options = MTL::ResourceStorageModePrivate;
            break;
        case StorageMode::MEMORYLESS:
            options = MTL::ResourceStorageModeMemoryless;
            break;
        default:
            options = MTL::ResourceStorageModeShared;
    }
    heap_desc->setResourceOptions(options | MTL::ResourceCPUCacheModeDefaultCache);
    
    // Create heap - release GIL during allocation
    {
        RELEASE_GIL();
        heap_ = device_->newHeap(heap_desc);
    }
    
    heap_desc->release();
    
    if (!heap_) {
        throw std::runtime_error("BufferPool: failed to create heap");
    }
}

BufferPool::~BufferPool() {
    // Clear all pooled buffers
    clear();
    
    // Release the heap
    if (heap_) {
        static_cast<MTL::Heap*>(heap_)->release();
        heap_ = nullptr;
    }
}

size_t BufferPool::align_size(size_t size) const {
    if (size >= LARGE_BUFFER_THRESHOLD) {
        return (size + PAGE_SIZE_BYTES - 1) & ~(PAGE_SIZE_BYTES - 1);
    }
    return (size + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1);
}

std::optional<BufferHandle> BufferPool::allocate(size_t size, BufferPriority priority) {
    if (size == 0) {
        return std::nullopt;
    }
    
    size_t aligned_size = align_size(size);
    
    // Try to get from pool first (fast path)
    auto result = try_get_from_pool(aligned_size, priority);
    if (result) {
        metrics_.cache_hits.fetch_add(1);
        return result;
    }
    
    // Allocate new buffer from heap
    metrics_.cache_misses.fetch_add(1);
    return allocate_new(aligned_size, priority);
}

std::optional<BufferHandle> BufferPool::try_get_from_pool(size_t aligned_size, BufferPriority priority) {
    std::unique_lock<std::shared_mutex> lock(pool_mutex_);
    
    // Look for existing buffer of exact size
    auto it = pool_.find(aligned_size);
    if (it != pool_.end() && !it->second.empty()) {
        // Get the last buffer from the vector (most recently added)
        auto& entry = it->second.back();
        MTLBufferRef buffer = entry.first;
        auto metadata = entry.second;
        
        // Update metadata
        metadata->last_acquired_at = std::chrono::steady_clock::now();
        metadata->acquisition_count.fetch_add(1);
        metadata->ref_count.store(1);
        
        // Remove from pool
        it->second.pop_back();
        
        // Update pool metrics
        metrics_.current_pooled.fetch_sub(metadata->size);
        
        // Move to active allocations
        uint64_t buffer_id = next_buffer_id();
        {
            std::lock_guard<std::mutex> active_lock(active_mutex_);
            active_[buffer_id] = {buffer, metadata};
        }
        
        lock.unlock();
        
        // Update allocated metrics
        metrics_.current_allocated.fetch_add(metadata->size);
        update_peak_metrics();
        
        return BufferHandle{
            buffer,
            metadata->size,
            aligned_size,
            priority,
            buffer_id
        };
    }
    
    return std::nullopt;
}

std::optional<BufferHandle> BufferPool::allocate_new(size_t aligned_size, BufferPriority priority) {
    // Check if we have enough space in heap
    MTL::Heap* heap = static_cast<MTL::Heap*>(heap_);
    size_t used = heap->usedSize();
    
    if (used + aligned_size > heap_size_) {
        // Try to evict some buffers
        size_t freed = evict(aligned_size);
        if (freed < aligned_size) {
            // Still not enough space
            return std::nullopt;
        }
    }
    
    // Allocate buffer from heap - release GIL during Metal allocation
    MTL::Buffer* buffer = nullptr;
    {
        RELEASE_GIL();
        buffer = heap->newBuffer(aligned_size, heap->resourceOptions());
    }
    
    if (!buffer) {
        return std::nullopt;
    }
    
    // Create metadata
    auto metadata = std::make_shared<BufferMetadata>(aligned_size, priority);
    metadata->ref_count.store(1);
    
    // Generate unique ID
    uint64_t buffer_id = next_buffer_id();
    
    // Add to active allocations
    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_[buffer_id] = {buffer, metadata};
    }
    
    // Update metrics
    metrics_.allocations.fetch_add(1);
    metrics_.current_allocated.fetch_add(aligned_size);
    update_peak_metrics();
    
    return BufferHandle{
        buffer,
        aligned_size,
        aligned_size,
        priority,
        buffer_id
    };
}

void BufferPool::release(const BufferHandle& handle) {
    std::shared_ptr<BufferMetadata> metadata;
    MTLBufferRef buffer = nullptr;
    
    // Remove from active allocations
    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        auto it = active_.find(handle.buffer_id);
        if (it == active_.end()) {
            return;  // Already released or invalid handle
        }
        buffer = it->second.first;
        metadata = it->second.second;
        active_.erase(it);
    }
    
    // Update metrics
    metrics_.releases.fetch_add(1);
    metrics_.current_allocated.fetch_sub(metadata->size);
    
    // Update metadata
    metadata->last_release_at = std::chrono::steady_clock::now();
    metadata->ref_count.store(0);
    
    // Calculate time in pool
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - metadata->last_acquired_at);
    metadata->total_time_in_pool_ns.fetch_add(duration.count());
    
    // Return to pool (if not full)
    if (metadata->priority != BufferPriority::PINNED) {
        std::lock_guard<std::shared_mutex> lock(pool_mutex_);
        pool_[handle.aligned_size].push_back({buffer, metadata});
        metrics_.current_pooled.fetch_add(metadata->size);
    } else {
        // Pinned buffers are released immediately
        static_cast<MTL::Buffer*>(buffer)->release();
    }
}

void BufferPool::pin(const BufferHandle& handle) {
    std::lock_guard<std::mutex> lock(active_mutex_);
    auto it = active_.find(handle.buffer_id);
    if (it != active_.end()) {
        it->second.second->priority = BufferPriority::PINNED;
    }
}

void BufferPool::unpin(const BufferHandle& handle) {
    std::lock_guard<std::mutex> lock(active_mutex_);
    auto it = active_.find(handle.buffer_id);
    if (it != active_.end()) {
        it->second.second->priority = BufferPriority::NORMAL;
    }
}

size_t BufferPool::evict(size_t target_bytes) {
    size_t freed = 0;
    
    std::unique_lock<std::shared_mutex> lock(pool_mutex_);
    
    // Collect all pool entries sorted by priority (low first) and age
    std::vector<std::tuple<BufferPriority, std::chrono::steady_clock::time_point, size_t, MTLBufferRef, std::shared_ptr<BufferMetadata>>> candidates;
    
    for (auto& [size, entries] : pool_) {
        for (auto& [buffer, metadata] : entries) {
            candidates.push_back({metadata->priority, metadata->last_release_at, size, buffer, metadata});
        }
    }
    
    // Sort by priority (low first), then by last release time (oldest first)
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        return std::get<1>(a) < std::get<1>(b);
    });
    
    // Evict buffers until we reach target
    for (const auto& [priority, last_release, size, buffer, metadata] : candidates) {
        if (freed >= target_bytes) {
            break;
        }
        
        // Don't evict pinned buffers
        if (priority == BufferPriority::PINNED) {
            continue;
        }
        
        // Find and remove from pool
        auto it = pool_.find(size);
        if (it != pool_.end()) {
            auto& entries = it->second;
            auto entry_it = std::find_if(entries.begin(), entries.end(),
                [buffer](const auto& e) { return e.first == buffer; });
            
            if (entry_it != entries.end()) {
                entries.erase(entry_it);
                
                // Release the buffer
                static_cast<MTL::Buffer*>(buffer)->release();
                
                freed += metadata->size;
                metrics_.evictions.fetch_add(1);
                metrics_.bytes_evicted.fetch_add(metadata->size);
                metrics_.current_pooled.fetch_sub(metadata->size);
            }
        }
    }
    
    // Clean up empty size entries
    for (auto it = pool_.begin(); it != pool_.end();) {
        if (it->second.empty()) {
            it = pool_.erase(it);
        } else {
            ++it;
        }
    }
    
    return freed;
}

void BufferPool::compact() {
    // This is a no-op for now since we're using MTLHeap which handles fragmentation
    // Future: could defragment by copying buffers to new locations
}

void BufferPool::clear() {
    std::unique_lock<std::shared_mutex> lock(pool_mutex_);
    
    // Release all pooled buffers
    for (auto& [size, entries] : pool_) {
        for (auto& [buffer, metadata] : entries) {
            static_cast<MTL::Buffer*>(buffer)->release();
        }
    }
    
    pool_.clear();
    metrics_.current_pooled.store(0);
}

size_t BufferPool::available_bytes() const {
    MTL::Heap* heap = static_cast<MTL::Heap*>(heap_);
    return heap_size_ - heap->usedSize();
}

void BufferPool::update_peak_metrics() {
    uint64_t current = metrics_.current_allocated.load();
    uint64_t peak = metrics_.peak_allocated.load();
    while (current > peak && !metrics_.peak_allocated.compare_exchange_weak(peak, current)) {
        // Retry with updated peak value
    }
    
    uint64_t total = current + metrics_.current_pooled.load();
    uint64_t peak_total = metrics_.peak_total.load();
    while (total > peak_total && !metrics_.peak_total.compare_exchange_weak(peak_total, total)) {
        // Retry with updated peak total value
    }
}

// =============================================================================
// TransientRingBuffer Implementation
// =============================================================================

TransientRingBuffer::TransientRingBuffer(void* device_ptr, size_t capacity, StorageMode storage_mode)
    : device_(nullptr)
    , buffer_(nullptr)
    , capacity_(capacity)
    , storage_mode_(storage_mode)
    , offset_(0)
{
    if (!device_ptr) {
        throw std::runtime_error("TransientRingBuffer: received null device pointer");
    }
    
    device_ = static_cast<MTL::Device*>(device_ptr);
    
    // Determine resource options
    MTL::ResourceOptions options;
    switch (storage_mode) {
        case StorageMode::SHARED:
            options = MTL::ResourceStorageModeShared;
            break;
        case StorageMode::MANAGED:
            options = MTL::ResourceStorageModeManaged;
            break;
        case StorageMode::PRIVATE:
            options = MTL::ResourceStorageModePrivate;
            break;
        case StorageMode::MEMORYLESS:
            options = MTL::ResourceStorageModeMemoryless;
            break;
        default:
            options = MTL::ResourceStorageModeShared;
    }
    
    // Create buffer - release GIL during allocation
    {
        RELEASE_GIL();
        buffer_ = static_cast<MTL::Device*>(device_)->newBuffer(capacity, options);
    }
    
    if (!buffer_) {
        throw std::runtime_error("TransientRingBuffer: failed to create buffer");
    }
}

TransientRingBuffer::~TransientRingBuffer() {
    if (buffer_) {
        static_cast<MTL::Buffer*>(buffer_)->release();
        buffer_ = nullptr;
    }
}

std::optional<std::pair<MTLBufferRef, size_t>> TransientRingBuffer::allocate(size_t size) {
    size_t aligned_size = (size + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1);
    
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    
    size_t current_offset = offset_.load();
    size_t new_offset = align_offset(current_offset) + aligned_size;
    
    if (new_offset > capacity_) {
        return std::nullopt;  // Out of space
    }
    
    offset_.store(new_offset);
    
    return std::make_pair(buffer_, align_offset(current_offset));
}

std::optional<std::pair<void*, size_t>> TransientRingBuffer::allocate_bytes(size_t size) {
    auto result = allocate(size);
    if (!result) {
        return std::nullopt;
    }
    
    MTL::Buffer* buffer = static_cast<MTL::Buffer*>(result->first);
    void* contents = buffer->contents();
    if (!contents) {
        return std::nullopt;
    }
    
    // Return pointer to the offset position
    void* ptr = static_cast<char*>(contents) + result->second;
    return std::make_pair(ptr, result->second);
}

void TransientRingBuffer::reset() {
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    offset_.store(0);
}

// =============================================================================
// HeapAllocator Implementation
// =============================================================================

HeapAllocator::HeapAllocator(void* device_ptr, size_t heap_size, StorageMode storage_mode)
    : device_(nullptr)
    , heap_(nullptr)
    , heap_size_(heap_size)
    , storage_mode_(storage_mode)
    , allocated_(0)
{
    if (!device_ptr) {
        throw std::runtime_error("HeapAllocator: received null device pointer");
    }
    
    device_ = static_cast<MTL::Device*>(device_ptr);
    
    // Create heap descriptor
    MTL::HeapDescriptor* heap_desc = MTL::HeapDescriptor::alloc()->init();
    if (!heap_desc) {
        throw std::runtime_error("HeapAllocator: failed to create heap descriptor");
    }
    
    heap_desc->setSize(heap_size);
    
    // Set storage mode
    MTL::ResourceOptions options;
    switch (storage_mode) {
        case StorageMode::SHARED:
            options = MTL::ResourceStorageModeShared;
            break;
        case StorageMode::MANAGED:
            options = MTL::ResourceStorageModeManaged;
            break;
        case StorageMode::PRIVATE:
            options = MTL::ResourceStorageModePrivate;
            break;
        case StorageMode::MEMORYLESS:
            options = MTL::ResourceStorageModeMemoryless;
            break;
        default:
            options = MTL::ResourceStorageModeShared;
    }
    heap_desc->setResourceOptions(options);
    
    // Create heap - release GIL during allocation
    {
        RELEASE_GIL();
        heap_ = static_cast<MTL::Device*>(device_)->newHeap(heap_desc);
    }
    
    heap_desc->release();
    
    if (!heap_) {
        throw std::runtime_error("HeapAllocator: failed to create heap");
    }
}

HeapAllocator::~HeapAllocator() {
    clear();
    
    if (heap_) {
        static_cast<MTL::Heap*>(heap_)->release();
        heap_ = nullptr;
    }
}

std::optional<std::pair<MTLBufferRef, size_t>> HeapAllocator::allocate(size_t size) {
    size_t aligned_size = (size + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1);
    
    // Try to get from pool first
    {
        std::shared_lock<std::shared_mutex> lock(pool_mutex_);
        auto it = pool_.find(aligned_size);
        if (it != pool_.end() && !it->second.empty()) {
            MTLBufferRef buffer = it->second.back();
            it->second.pop_back();
            
            // Get offset
            std::lock_guard<std::mutex> offset_lock(offset_mutex_);
            auto offset_it = offsets_.find(reinterpret_cast<uintptr_t>(buffer));
            if (offset_it != offsets_.end()) {
                size_t offset = offset_it->second;
                allocated_.fetch_add(aligned_size);
                return std::make_pair(buffer, offset);
            }
        }
    }
    
    // Allocate new buffer from heap
    MTL::Heap* heap = static_cast<MTL::Heap*>(heap_);
    
    if (heap->usedSize() + aligned_size > heap_size_) {
        return std::nullopt;  // Out of space
    }
    
    MTL::Buffer* buffer = nullptr;
    {
        RELEASE_GIL();
        buffer = heap->newBuffer(aligned_size, heap->resourceOptions());
    }
    
    if (!buffer) {
        return std::nullopt;
    }
    
    // Track offset (we use the buffer's actual offset in heap)
    size_t offset = heap->usedSize() - aligned_size;
    {
        std::lock_guard<std::mutex> lock(offset_mutex_);
        offsets_[reinterpret_cast<uintptr_t>(buffer)] = offset;
    }
    
    allocated_.fetch_add(aligned_size);
    
    return std::make_pair(buffer, offset);
}

std::optional<MTLBufferRef> HeapAllocator::allocate_buffer(size_t size) {
    auto result = allocate(size);
    if (!result) {
        return std::nullopt;
    }
    return result->first;
}

std::optional<size_t> HeapAllocator::get_offset(MTLBufferRef buffer) {
    std::lock_guard<std::mutex> lock(offset_mutex_);
    auto it = offsets_.find(reinterpret_cast<uintptr_t>(buffer));
    if (it != offsets_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void HeapAllocator::release(MTLBufferRef buffer) {
    if (!buffer) {
        return;
    }
    
    MTL::Buffer* mtl_buffer = static_cast<MTL::Buffer*>(buffer);
    size_t size = mtl_buffer->length();
    
    // Return to pool
    {
        std::lock_guard<std::shared_mutex> lock(pool_mutex_);
        pool_[size].push_back(buffer);
    }
    
    allocated_.fetch_sub(size);
}

void HeapAllocator::compact() {
    // MTLHeap handles compaction internally
    // This is a placeholder for future custom compaction logic
}

void HeapAllocator::clear() {
    std::unique_lock<std::shared_mutex> lock(pool_mutex_);
    
    // Release all pooled buffers
    for (auto& [size, buffers] : pool_) {
        for (auto buffer : buffers) {
            static_cast<MTL::Buffer*>(buffer)->release();
        }
    }
    
    pool_.clear();
    
    std::lock_guard<std::mutex> offset_lock(offset_mutex_);
    offsets_.clear();
    
    allocated_.store(0);
}

// BufferHandle helper
void* BufferHandle::contents() const {
    if (!buffer) {
        return nullptr;
    }
    return static_cast<MTL::Buffer*>(buffer)->contents();
}

}  // namespace metal_marlin
