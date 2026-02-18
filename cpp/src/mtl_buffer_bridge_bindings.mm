/**
 * @file mtl_buffer_bridge_bindings.mm
 * @brief nanobind Python bindings for optimized MTLBuffer direct access
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#import <Metal/Metal.h>
#include "mtl_buffer_bridge.hpp"

namespace nb = nanobind;
using namespace metal_marlin;

// =============================================================================
// Helper functions
// =============================================================================

/// Extract MTLBuffer from PyObjC capsule or object
static id<MTLBuffer> extract_mtl_buffer(nb::object obj) {
    // Try capsule first
    if (nb::isinstance<nb::capsule>(obj)) {
        auto cap = nb::cast<nb::capsule>(obj);
        return (__bridge id<MTLBuffer>)cap.data();
    }
    
    // Try to get __pyobjc_id__ attribute (PyObjC objects)
    try {
        uintptr_t ptr_val = nb::cast<uintptr_t>(obj.attr("__pyobjc_id__"));
        return (__bridge id<MTLBuffer>)(void*)ptr_val;
    } catch (...) {
        // Fall through
    }
    
    // Try direct pointer conversion
    try {
        uintptr_t ptr_val = nb::cast<uintptr_t>(obj);
        return (__bridge id<MTLBuffer>)(void*)ptr_val;
    } catch (...) {
        throw std::invalid_argument("Expected MTLBuffer capsule, PyObjC object, or pointer");
    }
}

// =============================================================================
// Module definition
// =============================================================================

NB_MODULE(_mtl_buffer_bridge, m) {
    m.doc() = "Optimized MTLBuffer direct pointer access bridge";
    
    // ========================================================================
    // DirectBufferPtr class
    // ========================================================================
    
    nb::class_<DirectBufferPtr>(m, "DirectBufferPtr",
        "Zero-overhead direct pointer access to Metal buffer contents")
        .def(nb::init<>(), "Create null pointer")
        .def("__init__", [](DirectBufferPtr* self, nb::object buffer_obj) {
            id<MTLBuffer> buffer = extract_mtl_buffer(buffer_obj);
            new (self) DirectBufferPtr(buffer);
        }, nb::arg("buffer"), "Create from MTLBuffer")
        .def_prop_ro("is_valid", &DirectBufferPtr::is_valid,
            "Check if pointer is valid")
        .def_prop_ro("contents", [](DirectBufferPtr& self) -> nb::object {
            void* ptr = self.contents();
            if (!ptr) return nb::none();
            return nb::capsule(ptr, "buffer_contents");
        }, "Get raw contents pointer as capsule")
        .def_prop_ro("ptr", [](DirectBufferPtr& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.contents());
        }, "Get raw pointer as integer")
        .def_prop_ro("length", &DirectBufferPtr::length,
            "Get buffer length in bytes")
        .def("as_float32", [](DirectBufferPtr& self) -> nb::ndarray<float, nb::numpy> {
            size_t num_elements = self.length() / sizeof(float);
            return nb::ndarray<float, nb::numpy>(self.as<float>(), {num_elements});
        }, "Get as float32 numpy array view")
        .def("as_float16", [](DirectBufferPtr& self) -> nb::ndarray<uint16_t, nb::numpy> {
            size_t num_elements = self.length() / sizeof(uint16_t);
            return nb::ndarray<uint16_t, nb::numpy>(self.as<uint16_t>(), {num_elements});
        }, "Get as float16 (uint16) numpy array view")
        .def("as_int32", [](DirectBufferPtr& self) -> nb::ndarray<int32_t, nb::numpy> {
            size_t num_elements = self.length() / sizeof(int32_t);
            return nb::ndarray<int32_t, nb::numpy>(self.as<int32_t>(), {num_elements});
        }, "Get as int32 numpy array view")
        .def("as_uint8", [](DirectBufferPtr& self) -> nb::ndarray<uint8_t, nb::numpy> {
            size_t num_elements = self.length();
            return nb::ndarray<uint8_t, nb::numpy>(self.as<uint8_t>(), {num_elements});
        }, "Get as uint8 numpy array view")
        .def("copy_from", &DirectBufferPtr::copy_from,
            nb::arg("src"), nb::arg("size"), nb::arg("offset") = 0,
            "Copy data from source pointer")
        .def("copy_to", &DirectBufferPtr::copy_to,
            nb::arg("dst"), nb::arg("size"), nb::arg("offset") = 0,
            "Copy data to destination pointer")
        .def("zero", [](DirectBufferPtr& self, size_t offset, size_t size) {
            self.zero(offset, size);
        }, nb::arg("offset") = 0, nb::arg("size"),
            "Zero a region of the buffer")
        .def("zero_all", &DirectBufferPtr::zero_all,
            "Zero entire buffer")
        .def("prefetch_read", &DirectBufferPtr::prefetch_read,
            "Prefetch buffer for reading")
        .def("prefetch_write", &DirectBufferPtr::prefetch_write,
            "Prefetch buffer for writing")
        .def("__len__", &DirectBufferPtr::length,
            "Buffer length")
        .def("__bool__", &DirectBufferPtr::is_valid,
            "Truthiness (True if valid)");
    
    // ========================================================================
    // MPSTensorWrapper class
    // ========================================================================
    
    nb::class_<MPSTensorWrapper>(m, "MPSTensorWrapper",
        "Zero-copy wrapper for MPS tensor data")
        .def_static("wrap", [](nb::object buffer_obj) -> MPSTensorWrapper {
            id<MTLBuffer> buffer = extract_mtl_buffer(buffer_obj);
            return MPSTensorWrapper::wrap(buffer);
        }, nb::arg("buffer"), "Wrap existing MPS tensor buffer")
        .def_prop_ro("is_valid", &MPSTensorWrapper::is_valid,
            "Check if wrapper is valid")
        .def_prop_ro("data", [](MPSTensorWrapper& self) -> nb::object {
            void* ptr = self.data();
            if (!ptr) return nb::none();
            return nb::capsule(ptr, "tensor_data");
        }, "Get tensor data pointer")
        .def_prop_ro("ptr", [](MPSTensorWrapper& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.data());
        }, "Get data pointer as integer")
        .def_prop_ro("size", &MPSTensorWrapper::size,
            "Get tensor size in bytes")
        .def("tensor_data", [](MPSTensorWrapper& self) -> nb::object {
            void* ptr = self.data();
            if (!ptr) return nb::none();
            return nb::capsule(ptr, "tensor_data");
        }, "Get tensor data pointer capsule")
        .def("__len__", &MPSTensorWrapper::size,
            "Tensor size")
        .def("__bool__", &MPSTensorWrapper::is_valid,
            "Truthiness (True if valid)");
    
    // ========================================================================
    // Fast Utility Functions - Direct pointer access
    // ========================================================================
    
    m.def("align_buffer_size", &align_buffer_size, nb::arg("size"),
        "Align buffer size to cache line or page boundary");
    
    m.def("align_to_cache_line", &align_to_cache_line, nb::arg("size"),
        "Align size to cache line boundary (128 bytes)");
    
    // Fast single-shot pointer extraction - optimized path
    m.def("get_buffer_ptr", [](nb::object buffer_obj) -> std::pair<uintptr_t, size_t> {
        id<MTLBuffer> buffer = extract_mtl_buffer(buffer_obj);
        // Direct property access - fastest path
        void* contents = buffer.contents;
        size_t length = static_cast<size_t>(buffer.length);
        return {reinterpret_cast<uintptr_t>(contents), length};
    }, nb::arg("buffer"), 
        "Get (pointer, length) tuple from MTLBuffer using direct property access");
    
    // Batch pointer extraction with prefetch
    m.def("get_buffer_ptr_batch", [](nb::list buffers) -> std::vector<std::pair<uintptr_t, size_t>> {
        std::vector<std::pair<uintptr_t, size_t>> results;
        results.reserve(buffers.size());
        
        for (auto& obj : buffers) {
            try {
                id<MTLBuffer> buffer = extract_mtl_buffer(nb::cast<nb::object>(obj));
                void* contents = buffer.contents;
                size_t length = static_cast<size_t>(buffer.length);
                __builtin_prefetch(contents, 0, 1);  // Prefetch for reading
                results.push_back({reinterpret_cast<uintptr_t>(contents), length});
            } catch (...) {
                results.push_back({0, 0});
            }
        }
        
        return results;
    }, nb::arg("buffers"),
        "Get pointers for multiple buffers with prefetch");
    
    // Fast path for MPS tensor pointer extraction
    m.def("get_mps_tensor_ptr", [](nb::object mps_tensor) -> uintptr_t {
        // MPS tensors have data_ptr() method
        try {
            uintptr_t ptr = nb::cast<uintptr_t>(mps_tensor.attr("data_ptr")());
            return ptr;
        } catch (...) {
            // Fallback: try to extract from buffer
            id<MTLBuffer> buffer = extract_mtl_buffer(mps_tensor);
            return reinterpret_cast<uintptr_t>(buffer.contents);
        }
    }, nb::arg("tensor"), "Get data pointer from MPS tensor");
    
    // ========================================================================
    // Fast memory operations
    // ========================================================================
    
    m.def("fast_copy", [](uintptr_t dst, uintptr_t src, size_t size) {
        __builtin_memcpy(reinterpret_cast<void*>(dst), 
                         reinterpret_cast<void*>(src), size);
    }, nb::arg("dst"), nb::arg("src"), nb::arg("size"),
        "Fast memory copy between pointers using builtin memcpy");
    
    m.def("fast_zero", [](uintptr_t ptr, size_t size) {
        __builtin_memset(reinterpret_cast<void*>(ptr), 0, size);
    }, nb::arg("ptr"), nb::arg("size"),
        "Fast zero fill using builtin memset");
    
    m.def("prefetch_read", [](uintptr_t ptr) {
        __builtin_prefetch(reinterpret_cast<void*>(ptr), 0, 3);
    }, nb::arg("ptr"),
        "Prefetch memory for reading (high temporal locality)");
    
    m.def("prefetch_write", [](uintptr_t ptr) {
        __builtin_prefetch(reinterpret_cast<void*>(ptr), 1, 3);
    }, nb::arg("ptr"),
        "Prefetch memory for writing (high temporal locality)");
    
    // ========================================================================
    // Thread-local cache utilities
    // ========================================================================
    
    m.def("cache_buffer", [](nb::object buffer_obj) -> bool {
        id<MTLBuffer> buffer = extract_mtl_buffer(buffer_obj);
        uintptr_t buffer_id = reinterpret_cast<uintptr_t>(buffer);
        
        auto& cache = get_thread_local_cache();
        if (!cache.get(buffer_id)) {
            cache.cache(buffer_id, DirectBufferPtr(buffer));
            return true;  // Newly cached
        }
        return false;  // Already cached
    }, nb::arg("buffer"), "Cache buffer in thread-local cache, returns true if newly cached");
    
    m.def("get_cached_buffer", [](nb::object buffer_obj) -> std::pair<uintptr_t, size_t> {
        id<MTLBuffer> buffer = extract_mtl_buffer(buffer_obj);
        uintptr_t buffer_id = reinterpret_cast<uintptr_t>(buffer);
        
        auto& cache = get_thread_local_cache();
        auto* cached = cache.get(buffer_id);
        
        if (cached && cached->is_valid()) {
            return {reinterpret_cast<uintptr_t>(cached->contents()), cached->length()};
        }
        
        // Fallback to direct access
        void* contents = buffer.contents;
        size_t length = static_cast<size_t>(buffer.length);
        return {reinterpret_cast<uintptr_t>(contents), length};
    }, nb::arg("buffer"), "Get buffer pointer from cache or direct access");
    
    m.def("invalidate_cached_buffer", [](nb::object buffer_obj) {
        id<MTLBuffer> buffer = extract_mtl_buffer(buffer_obj);
        uintptr_t buffer_id = reinterpret_cast<uintptr_t>(buffer);
        
        auto& cache = get_thread_local_cache();
        cache.invalidate(buffer_id);
    }, nb::arg("buffer"), "Invalidate cached buffer entry");
    
    m.def("clear_buffer_cache", []() {
        auto& cache = get_thread_local_cache();
        cache.clear();
    }, "Clear all thread-local buffer cache entries");
    
    // ========================================================================
    // Constants
    // ========================================================================
    
    m.attr("CACHE_LINE_SIZE") = kCacheLineSize;
    m.attr("PAGE_SIZE") = kPageSize;
    m.attr("LARGE_BUFFER_THRESHOLD") = kLargeBufferThreshold;
}
