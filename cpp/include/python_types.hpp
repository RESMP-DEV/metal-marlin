#pragma once

#include "direct_access.hpp"
#include "buffer_manager.hpp"
#include "library_manager.hpp"

#include <nanobind/nanobind.h>
#include <Metal/Metal.hpp>
#include <unordered_map>

namespace metal_marlin {

namespace nb = nanobind;

/**
 * @brief ManagedBuffer wrapper for _cpp_ext compatibility.
 * Wraps direct_access::BufferPtr and provides Python interface.
 */
class ManagedBuffer {
public:
  explicit ManagedBuffer(direct_access::BufferPtr ptr) : ptr_(std::move(ptr)), mps_buffer_() {}
  
  // Constructor from MPSBuffer (for pooled allocations)
  explicit ManagedBuffer(MPSBuffer buf) : ptr_(), mps_buffer_(std::move(buf)) {}

  size_t length() const { 
    if (mps_buffer_.is_valid()) return mps_buffer_.size();
    return ptr_.length(); 
  }
  
  uintptr_t data_ptr() const {
    if (mps_buffer_.is_valid()) return reinterpret_cast<uintptr_t>(mps_buffer_.contents());
    return reinterpret_cast<uintptr_t>(ptr_.contents());
  }

  direct_access::BufferPtr &ptr() { return ptr_; }

  id<MTLBuffer> get_buffer() const {
    if (mps_buffer_.is_valid()) {
       return (__bridge id<MTLBuffer>)mps_buffer_.buffer();
    }
    return ptr_.buffer();
  }

private:
  direct_access::BufferPtr ptr_;
  MPSBuffer mps_buffer_;
};

/**
 * @brief MetalContext wrapper for _cpp_ext compatibility.
 * Consolidates device, queues, and library management.
 */
class MetalContext {
public:
  MetalContext() : device_(direct_access::get_default_device()), buffer_manager_((__bridge MTL::Device*)device_.device()) {}

  ~MetalContext() {
    for (auto& pair : compiled_libs_) {
      if (pair.second) {
        CFRelease(pair.second);
      }
    }
  }

  void load_metallib(const std::string &path) {
    // Use LibraryManager for caching
    LibraryManager::instance().load_metallib(path);
  }

  void compile_source(const std::string& lib_name, const std::string& source) {
    NSError *error = nullptr;
    NSString *ns_source = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions *options = [MTLCompileOptions new];
    
    id<MTLLibrary> library = [device_.device() newLibraryWithSource:ns_source 
                                                          options:options 
                                                            error:&error];
    
    if (!library) {
      std::string msg = "Failed to compile source for " + lib_name;
      if (error) {
        msg += ": " + std::string([[error localizedDescription] UTF8String]);
      }
      throw std::runtime_error(msg);
    }

    // Retain as MTL::Library*
    MTL::Library* cpp_lib = (__bridge_retained MTL::Library*)library;
    
    // Release existing if any
    auto it = compiled_libs_.find(lib_name);
    if (it != compiled_libs_.end()) {
        CFRelease(it->second);
    }
    
    compiled_libs_[lib_name] = cpp_lib;
  }

  nb::object get_pipeline(const std::string &kernel_name,
                          const std::string &metallib_path = "") {
    MTL::Library *lib = nullptr;
    
    // Check compiled libs first
    auto it = compiled_libs_.find(metallib_path);
    if (it != compiled_libs_.end()) {
      lib = it->second;
    } else if (!metallib_path.empty()) {
      lib = LibraryManager::instance().load_metallib(metallib_path);
    }

    MTL::Function *func =
        LibraryManager::instance().get_kernel(kernel_name, lib);
    if (!func) {
      throw std::runtime_error("Kernel function not found: " + kernel_name);
    }

    NSError *error = nullptr;
    id<MTLComputePipelineState> pipeline =
        [device_.device() newComputePipelineStateWithFunction:(__bridge id<MTLFunction>)func
                                                        error:&error];

    if (!pipeline) {
      std::string msg = "Failed to create pipeline";
      if (error) {
        msg += ": " + std::string([[error localizedDescription] UTF8String]);
      }
      throw std::runtime_error(msg);
    }

    void *pipeline_ptr = (__bridge_retained void *)pipeline;
    return nb::capsule(
        pipeline_ptr, "Pipeline", [](void *ptr) noexcept {
          if (ptr != nullptr) {
            (void)CFBridgingRelease(ptr);
          }
        });
  }

  direct_access::MetalDeviceDirect &device() { return device_; }
  
  BufferManager& buffer_pool() { return buffer_manager_; }

  std::string device_name() {
      return std::string([device_.device().name UTF8String]);
  }

  int gpu_family() {
      // Basic check for Apple Silicon
      if ([device_.device() supportsFamily:MTLGPUFamilyApple7]) return 7;
      return 1;
  }

private:
  direct_access::MetalDeviceDirect &device_;
  BufferManager buffer_manager_;
  std::unordered_map<std::string, MTL::Library*> compiled_libs_;
};

} // namespace metal_marlin
