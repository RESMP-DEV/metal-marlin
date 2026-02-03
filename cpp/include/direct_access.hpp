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

namespace metal_marlin::direct_access {

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

private:
    std::unique_ptr<Impl> impl_;
};

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

} // namespace metal_marlin::direct_access
