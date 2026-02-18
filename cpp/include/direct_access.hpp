/**
 * @file direct_access.hpp
 * @brief Zero-overhead direct MTLBuffer access for Metal Marlin
 *
 * This module provides the fastest possible access to Metal buffer memory
 * by using direct Objective-C property access and pointer caching.
 *
 * Key optimizations:
 * - Direct property access (buffer.contents) instead of methodForSelector
 * - No IMP caching overhead for simple property reads
 * - Inline hot paths for single-instruction access
 * - Zero-copy MPS tensor integration
 */

#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
#include <objc/objc.h>
typedef void* id;
typedef size_t NSUInteger;
typedef NSUInteger MTLResourceOptions;
#endif

#include <memory>
#include <string>
#include <utility>

namespace metal_marlin::direct_access {

// =============================================================================
// Inline helper functions - Fastest access path
// =============================================================================

#ifdef __OBJC__
/**
 * @brief Get buffer contents pointer - inline hot path
 * Uses direct property access for minimal overhead
 */
inline void* buffer_contents(id<MTLBuffer> buffer) noexcept {
    return buffer.contents;
}

/**
 * @brief Get buffer length - inline hot path
 */
inline size_t buffer_length(id<MTLBuffer> buffer) noexcept {
    return static_cast<size_t>(buffer.length);
}

/**
 * @brief Get both pointer and length in one call
 */
inline std::pair<void*, size_t> buffer_ptr_len(id<MTLBuffer> buffer) noexcept {
    return {buffer.contents, static_cast<size_t>(buffer.length)};
}
#endif

// =============================================================================
// BufferPtr - Smart pointer for MTLBuffer with cached contents
// =============================================================================

class BufferPtr {
public:
    class Impl;
    BufferPtr();
    explicit BufferPtr(std::unique_ptr<Impl> impl);
    ~BufferPtr();
    BufferPtr(BufferPtr&&) noexcept;
    BufferPtr& operator=(BufferPtr&&) noexcept;

    // Non-copyable
    BufferPtr(const BufferPtr&) = delete;
    BufferPtr& operator=(const BufferPtr&) = delete;

    void* contents() const noexcept;
    size_t length() const noexcept;
    bool is_valid() const noexcept;
#ifdef __OBJC__
    id<MTLBuffer> buffer() const noexcept;
#else
    id buffer() const noexcept;
#endif

    template<typename T>
    T* as() const noexcept { return reinterpret_cast<T*>(contents()); }
    
    template<typename T>
    T* as_at(size_t byte_offset) const noexcept {
        return reinterpret_cast<T*>(static_cast<uint8_t*>(contents()) + byte_offset);
    }

private:
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// MetalDeviceDirect - Direct device access
// =============================================================================

class MetalDeviceDirect {
public:
    class Impl;
    MetalDeviceDirect();
    explicit MetalDeviceDirect(std::unique_ptr<Impl> impl);
    ~MetalDeviceDirect();
    MetalDeviceDirect(MetalDeviceDirect&&) noexcept;
    MetalDeviceDirect& operator=(MetalDeviceDirect&&) noexcept;

    // Non-copyable
    MetalDeviceDirect(const MetalDeviceDirect&) = delete;
    MetalDeviceDirect& operator=(const MetalDeviceDirect&) = delete;

    bool is_valid() const noexcept;

#ifdef __OBJC__
    id<MTLDevice> device() const noexcept;
    id<MTLCommandQueue> primary_queue() const noexcept;
    id<MTLCommandQueue> decode_queue() const noexcept;
    BufferPtr create_buffer(size_t size, MTLResourceOptions options) const;
    BufferPtr create_buffer_from_bytes(const void* data, size_t size, MTLResourceOptions options) const;
    id<MTLLibrary> load_library(const std::string& path) const;
#else
    id device() const noexcept;
    id primary_queue() const noexcept;
    id decode_queue() const noexcept;
    BufferPtr create_buffer(size_t size, NSUInteger options) const;
    BufferPtr create_buffer_from_bytes(const void* data, size_t size, NSUInteger options) const;
    id load_library(const std::string& path) const;
#endif

    BufferPtr wrap_pointer(void* ptr, size_t size) const;

private:
    std::unique_ptr<Impl> impl_;
};

MetalDeviceDirect& get_default_device();

// =============================================================================
// Fast buffer operations - Inline implementations
// =============================================================================

/**
 * @brief Prefetch buffer for reading
 */
inline void prefetch_read(void* ptr) noexcept {
    __builtin_prefetch(ptr, 0, 3);
}

/**
 * @brief Prefetch buffer for writing
 */
inline void prefetch_write(void* ptr) noexcept {
    __builtin_prefetch(ptr, 1, 3);
}

/**
 * @brief Fast memcpy using builtin
 */
inline void fast_copy(void* dst, const void* src, size_t size) noexcept {
    __builtin_memcpy(dst, src, size);
}

/**
 * @brief Fast memset using builtin
 */
inline void fast_zero(void* ptr, size_t size) noexcept {
    __builtin_memset(ptr, 0, size);
}

} // namespace metal_marlin::direct_access
