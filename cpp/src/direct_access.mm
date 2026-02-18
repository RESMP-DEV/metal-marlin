/**
 * @file direct_access.mm
 * @brief Zero-overhead MTLBuffer pointer access for Metal Marlin
 *
 * This module provides direct Objective-C interop for MTLBuffer access
 * using optimized property access patterns.
 *
 * Key optimizations:
 * - Direct property access (buffer.contents) instead of IMP caching
 * - Single initialization with retained references
 * - Inline hot paths for zero-overhead access
 * - Zero-copy MPS tensor integration via __bridge cast
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "direct_access.hpp"

namespace metal_marlin::direct_access {

// =============================================================================
// BufferPtr Implementation - Simplified, no IMP caching
// =============================================================================

class BufferPtr::Impl {
public:
    explicit Impl(id<MTLBuffer> buffer) : buffer_(buffer) {
        if (__builtin_expect(buffer_ != nil, 1)) {
            // Direct property access - single objc_msgSend
            contents_ = buffer_.contents;
            length_ = static_cast<size_t>(buffer_.length);
            
            // Prefetch first cache line
            __builtin_prefetch(contents_, 0, 3);
        }
    }

    ~Impl() = default;

    // Non-copyable, movable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&& other) noexcept
        : buffer_(other.buffer_)
        , contents_(other.contents_)
        , length_(other.length_) {
        other.buffer_ = nil;
        other.contents_ = nullptr;
        other.length_ = 0;
    }

    [[nodiscard]] inline void* contents() const noexcept { return contents_; }
    [[nodiscard]] inline size_t length() const noexcept { return length_; }
    [[nodiscard]] inline id<MTLBuffer> buffer() const noexcept { return buffer_; }

private:
    id<MTLBuffer> buffer_;
    void* contents_;
    size_t length_;
};

BufferPtr::BufferPtr() : impl_(nullptr) {}

BufferPtr::BufferPtr(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

BufferPtr::~BufferPtr() = default;

BufferPtr::BufferPtr(BufferPtr&&) noexcept = default;
BufferPtr& BufferPtr::operator=(BufferPtr&&) noexcept = default;

void* BufferPtr::contents() const noexcept {
    return __builtin_expect(impl_ != nullptr, 1) ? impl_->contents() : nullptr;
}

size_t BufferPtr::length() const noexcept {
    return __builtin_expect(impl_ != nullptr, 1) ? impl_->length() : 0;
}

bool BufferPtr::is_valid() const noexcept { return impl_ != nullptr; }

id<MTLBuffer> BufferPtr::buffer() const noexcept {
    return __builtin_expect(impl_ != nullptr, 1) ? impl_->buffer() : nil;
}

// =============================================================================
// MetalDeviceDirect Implementation - Direct device access
// =============================================================================

class MetalDeviceDirect::Impl {
public:
    explicit Impl(id<MTLDevice> device) : device_(device) {
        if (device_) {
            // Create command queues directly
            primary_queue_ = [device_ newCommandQueue];
            decode_queue_ = [device_ newCommandQueue];
        }
    }

    ~Impl() = default;

    // Non-copyable, movable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&& other) noexcept
        : device_(other.device_)
        , primary_queue_(other.primary_queue_)
        , decode_queue_(other.decode_queue_) {
        other.device_ = nil;
        other.primary_queue_ = nil;
        other.decode_queue_ = nil;
    }

    [[nodiscard]] inline id<MTLDevice> device() const noexcept { return device_; }
    [[nodiscard]] inline id<MTLCommandQueue> primary_queue() const noexcept {
        return primary_queue_;
    }
    [[nodiscard]] inline id<MTLCommandQueue> decode_queue() const noexcept {
        return decode_queue_;
    }

    // Create buffer with direct API call
    [[nodiscard]] BufferPtr create_buffer(size_t size,
                                          MTLResourceOptions options) const {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:options];

        if (__builtin_expect(buffer == nil, 0)) {
            throw std::runtime_error("Failed to allocate Metal buffer");
        }

        return BufferPtr(std::make_unique<BufferPtr::Impl>(buffer));
    }

    // Create buffer from bytes with direct API call
    [[nodiscard]] BufferPtr
    create_buffer_from_bytes(const void* data, size_t size,
                             MTLResourceOptions options) const {
        __builtin_prefetch(data, 0, 1);

        id<MTLBuffer> buffer = [device_ newBufferWithBytes:data
                                                    length:size
                                                   options:options];

        if (__builtin_expect(buffer == nil, 0)) {
            throw std::runtime_error("Failed to create Metal buffer from bytes");
        }

        return BufferPtr(std::make_unique<BufferPtr::Impl>(buffer));
    }

    // Create buffer wrapping existing pointer (zero-copy for MPS tensors)
    [[nodiscard]] BufferPtr wrap_pointer(void* ptr, size_t size) const {
        __builtin_prefetch(ptr, 0, 1);

        MTLResourceOptions options =
            MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;

        id<MTLBuffer> buffer = [device_ newBufferWithBytesNoCopy:ptr
                                                          length:size
                                                         options:options
                                                      deallocator:nil];

        if (__builtin_expect(buffer == nil, 0)) {
            throw std::runtime_error("Failed to wrap pointer in Metal buffer");
        }

        return BufferPtr(std::make_unique<BufferPtr::Impl>(buffer));
    }

    // Load library from file
    [[nodiscard]] id<MTLLibrary> load_library(const std::string& path) const {
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:ns_path];

        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithURL:url error:&error];

        if (!library) {
            NSString* desc = [error localizedDescription];
            throw std::runtime_error("Failed to load metallib: " +
                                     std::string([desc UTF8String]));
        }

        return library;
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> primary_queue_;
    id<MTLCommandQueue> decode_queue_;
};

MetalDeviceDirect::MetalDeviceDirect() : impl_(nullptr) {}

MetalDeviceDirect::MetalDeviceDirect(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalDeviceDirect::~MetalDeviceDirect() = default;

MetalDeviceDirect::MetalDeviceDirect(MetalDeviceDirect&&) noexcept = default;
MetalDeviceDirect&
MetalDeviceDirect::operator=(MetalDeviceDirect&&) noexcept = default;

bool MetalDeviceDirect::is_valid() const noexcept { return impl_ != nullptr; }

BufferPtr MetalDeviceDirect::create_buffer(size_t size,
                                           MTLResourceOptions options) const {
    if (!impl_)
        throw std::runtime_error("Invalid device");
    return impl_->create_buffer(size, options);
}

BufferPtr
MetalDeviceDirect::create_buffer_from_bytes(const void* data, size_t size,
                                            MTLResourceOptions options) const {
    if (!impl_)
        throw std::runtime_error("Invalid device");
    return impl_->create_buffer_from_bytes(data, size, options);
}

BufferPtr MetalDeviceDirect::wrap_pointer(void* ptr, size_t size) const {
    if (!impl_)
        throw std::runtime_error("Invalid device");
    return impl_->wrap_pointer(ptr, size);
}

id<MTLDevice> MetalDeviceDirect::device() const noexcept {
    return impl_ ? impl_->device() : nil;
}

id<MTLCommandQueue> MetalDeviceDirect::primary_queue() const noexcept {
    return impl_ ? impl_->primary_queue() : nil;
}

id<MTLCommandQueue> MetalDeviceDirect::decode_queue() const noexcept {
    return impl_ ? impl_->decode_queue() : nil;
}

id<MTLLibrary> MetalDeviceDirect::load_library(const std::string& path) const {
    if (!impl_)
        throw std::runtime_error("Invalid device");
    return impl_->load_library(path);
}

// =============================================================================
// Global device singleton
// =============================================================================

namespace {

std::unique_ptr<MetalDeviceDirect> g_device;
std::once_flag g_init_flag;

void init_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("No Metal device available");
    }
    g_device = std::make_unique<MetalDeviceDirect>(
        std::make_unique<MetalDeviceDirect::Impl>(device));
}

} // anonymous namespace

MetalDeviceDirect& get_default_device() {
    std::call_once(g_init_flag, init_device);
    return *g_device;
}

} // namespace metal_marlin::direct_access

// =============================================================================
// C API for interop with other modules
// =============================================================================

extern "C" {

/// Get default Metal device (void* for C interop)
void* metal_direct_get_device() {
    return (__bridge void*)metal_marlin::direct_access::get_default_device()
        .device();
}

/// Get primary command queue
void* metal_direct_get_primary_queue() {
    return (__bridge void*)metal_marlin::direct_access::get_default_device()
        .primary_queue();
}

/// Get decode command queue
void* metal_direct_get_decode_queue() {
    return (__bridge void*)metal_marlin::direct_access::get_default_device()
        .decode_queue();
}

} // extern "C"
