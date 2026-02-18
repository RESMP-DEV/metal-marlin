/**
 * @file mtl_buffer_bridge.mm
 * @brief Implementation of optimized MTLBuffer direct pointer access
 *
 * This implementation uses:
 * - Direct property access for minimal objc_msgSend overhead
 * - Single-call initialization pattern
 * - SIMD-aligned operations for vectorized data movement
 * - No method caching overhead for simple property access
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <objc/runtime.h>
#include <objc/message.h>

#include "mtl_buffer_bridge.hpp"

namespace metal_marlin {

// =============================================================================
// DirectBufferPtr Implementation - Optimized for minimal overhead
// =============================================================================

DirectBufferPtr::DirectBufferPtr(id<MTLBuffer> buffer) noexcept
    : buffer_(buffer) {
    if (__builtin_expect(buffer_ != nil, 1)) {
        // Direct property access - single objc_msgSend per property
        contents_ = buffer_.contents;
        length_ = static_cast<size_t>(buffer_.length);
        
        // Retain the buffer
        CFRetain(buffer_);
        
        // Prefetch for immediate access
        __builtin_prefetch(contents_, 0, 3);
    }
}

DirectBufferPtr::DirectBufferPtr(id<MTLBuffer> buffer, void* contents, size_t length) noexcept
    : buffer_(buffer)
    , contents_(contents)
    , length_(length) {
    if (__builtin_expect(buffer_ != nil, 1)) {
        CFRetain(buffer_);
        __builtin_prefetch(contents_, 0, 3);
    }
}

DirectBufferPtr::~DirectBufferPtr() {
    if (buffer_) {
        CFRelease(buffer_);
    }
}

DirectBufferPtr::DirectBufferPtr(DirectBufferPtr&& other) noexcept
    : buffer_(other.buffer_)
    , contents_(other.contents_)
    , length_(other.length_) {
    other.buffer_ = nil;
    other.contents_ = nullptr;
    other.length_ = 0;
}

DirectBufferPtr& DirectBufferPtr::operator=(DirectBufferPtr&& other) noexcept {
    if (this != &other) {
        // Release current buffer
        if (buffer_) {
            CFRelease(buffer_);
        }
        
        buffer_ = other.buffer_;
        contents_ = other.contents_;
        length_ = other.length_;
        
        other.buffer_ = nil;
        other.contents_ = nullptr;
        other.length_ = 0;
    }
    return *this;
}

// =============================================================================
// PooledDirectBuffer Implementation
// =============================================================================

PooledDirectBuffer::PooledDirectBuffer(
    id<MTLBuffer> buffer,
    size_t size,
    std::function<void(id<MTLBuffer>)> deleter) 
    : ptr_(buffer)
    , deleter_(std::move(deleter)) {
    // Validate size matches
    if (ptr_.length() != size) {
        // Size mismatch - clear the pointer
        ptr_ = DirectBufferPtr();
    }
}

PooledDirectBuffer::~PooledDirectBuffer() {
    if (deleter_ && ptr_.buffer()) {
        deleter_(ptr_.buffer());
    }
}

PooledDirectBuffer::PooledDirectBuffer(PooledDirectBuffer&& other) noexcept
    : ptr_(std::move(other.ptr_))
    , deleter_(std::move(other.deleter_)) {
}

PooledDirectBuffer& PooledDirectBuffer::operator=(PooledDirectBuffer&& other) noexcept {
    if (this != &other) {
        // Release current
        if (deleter_ && ptr_.buffer()) {
            deleter_(ptr_.buffer());
        }
        
        ptr_ = std::move(other.ptr_);
        deleter_ = std::move(other.deleter_);
    }
    return *this;
}

// =============================================================================
// MPSTensorWrapper Implementation
// =============================================================================

MPSTensorWrapper MPSTensorWrapper::wrap(id<MTLBuffer> buffer) noexcept {
    MPSTensorWrapper wrapper;
    wrapper.ptr_ = DirectBufferPtr(buffer);
    return wrapper;
}

MPSTensorWrapper MPSTensorWrapper::wrap(void* data_ptr, size_t size) noexcept {
    MPSTensorWrapper wrapper;
    // Create a lightweight wrapper using the fast constructor
    // Note: This creates a wrapper without retaining the buffer
    // Caller must ensure buffer lifetime
    wrapper.ptr_ = DirectBufferPtr{nil, data_ptr, size};
    return wrapper;
}

// =============================================================================
// ThreadLocalBufferCache Implementation
// =============================================================================

DirectBufferPtr* ThreadLocalBufferCache::get(uintptr_t buffer_id) noexcept {
    if (buffer_id == 0) return nullptr;
    
    // Simple hash-based lookup
    size_t index = buffer_id % kMaxCachedBuffers;
    
    if (entries_[index].buffer_id.load(std::memory_order_acquire) == buffer_id) {
        return &entries_[index].ptr;
    }
    
    return nullptr;
}

void ThreadLocalBufferCache::cache(uintptr_t buffer_id, DirectBufferPtr ptr) noexcept {
    if (buffer_id == 0 || !ptr.is_valid()) return;
    
    size_t index = buffer_id % kMaxCachedBuffers;
    
    entries_[index].buffer_id.store(buffer_id, std::memory_order_release);
    entries_[index].ptr = std::move(ptr);
}

void ThreadLocalBufferCache::invalidate(uintptr_t buffer_id) noexcept {
    if (buffer_id == 0) return;
    
    size_t index = buffer_id % kMaxCachedBuffers;
    
    if (entries_[index].buffer_id.load(std::memory_order_acquire) == buffer_id) {
        entries_[index].buffer_id.store(0, std::memory_order_release);
        entries_[index].ptr = DirectBufferPtr();
    }
}

void ThreadLocalBufferCache::clear() noexcept {
    for (size_t i = 0; i < kMaxCachedBuffers; ++i) {
        entries_[i].buffer_id.store(0, std::memory_order_release);
        entries_[i].ptr = DirectBufferPtr();
    }
}

// Thread-local cache instance
ThreadLocalBufferCache& get_thread_local_cache() noexcept {
    static thread_local ThreadLocalBufferCache cache;
    return cache;
}

// =============================================================================
// C API Implementation - Minimal overhead wrapper
// =============================================================================

extern "C" {

void* mtl_buffer_bridge_create(void* mtl_buffer) {
    if (!mtl_buffer) return nullptr;
    
    @try {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mtl_buffer;
        auto* bridge = new DirectBufferPtr(buffer);
        return bridge;
    } @catch (...) {
        return nullptr;
    }
}

void* mtl_buffer_bridge_get_contents(void* bridge) {
    if (!bridge) return nullptr;
    return static_cast<DirectBufferPtr*>(bridge)->contents();
}

size_t mtl_buffer_bridge_get_length(void* bridge) {
    if (!bridge) return 0;
    return static_cast<DirectBufferPtr*>(bridge)->length();
}

void mtl_buffer_bridge_copy_from(void* bridge, const void* src, size_t size, size_t offset) {
    if (!bridge || !src) return;
    static_cast<DirectBufferPtr*>(bridge)->copy_from(src, size, offset);
}

void mtl_buffer_bridge_copy_to(void* bridge, void* dst, size_t size, size_t offset) {
    if (!bridge || !dst) return;
    static_cast<DirectBufferPtr*>(bridge)->copy_to(dst, size, offset);
}

void mtl_buffer_bridge_zero(void* bridge, size_t offset, size_t size) {
    if (!bridge) return;
    static_cast<DirectBufferPtr*>(bridge)->zero(offset, size);
}

void mtl_buffer_bridge_destroy(void* bridge) {
    if (bridge) {
        delete static_cast<DirectBufferPtr*>(bridge);
    }
}

void* mtl_buffer_bridge_wrap_mps(void* mtl_buffer) {
    if (!mtl_buffer) return nullptr;
    
    @try {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mtl_buffer;
        auto* wrapper = new MPSTensorWrapper(MPSTensorWrapper::wrap(buffer));
        return wrapper;
    } @catch (...) {
        return nullptr;
    }
}

void* mtl_buffer_bridge_get_mps_data(void* wrapper) {
    if (!wrapper) return nullptr;
    return static_cast<MPSTensorWrapper*>(wrapper)->data();
}

void mtl_buffer_bridge_destroy_mps_wrapper(void* wrapper) {
    if (wrapper) {
        delete static_cast<MPSTensorWrapper*>(wrapper);
    }
}

} // extern "C"

} // namespace metal_marlin
