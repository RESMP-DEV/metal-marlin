/**
 * @file python_bindings.cpp
 * @brief nanobind Python bindings for Metal Marlin C++ core
 *
 * This module exposes the C++ core classes to Python:
 * - BufferPool: GIL-free Metal buffer pool with priority-based eviction
 * - TransientRingBuffer: O(1) reset ring buffer for transient allocations
 * - HeapAllocator: MTLHeap-based sub-allocation
 * - EncoderCache: Command encoder caching for reduced dispatch overhead
 * - TokenGroupManager: MoE token grouping and dispatch
 * - MetalDevice/QueueManager: Metal device and queue management
 * - LibraryManager: Precompiled metallib loading and caching
 *
 * Usage:
 *     from metal_marlin._core import (
 *         BufferPool, TransientRingBuffer, HeapAllocator,
 *         EncoderCache, TokenGroupManager,
 *         MetalDevice, QueueManager, LibraryManager,
 *         BufferPriority, StorageMode
 *     )
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#import <Metal/Metal.h> // ObjC Metal types for native API calls

#include "buffer_pool.hpp"
#include "direct_access.hpp"
#include "encoder_cache.hpp"
#include "library_manager.hpp"
#include "metal_device.hpp"
#include "moe_manager.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace nb = nanobind;
using namespace metal_marlin;
using namespace metal_marlin::direct_access;

// =============================================================================
// C++ Extension Compatibility Layer (_cpp_ext emulation)
// =============================================================================

/**
 * @brief ManagedBuffer wrapper for _cpp_ext compatibility.
 * Wraps direct_access::BufferPtr and provides Python interface.
 */
class ManagedBuffer {
public:
  explicit ManagedBuffer(BufferPtr ptr) : ptr_(std::move(ptr)) {}

  size_t length() const { return ptr_.length(); }
  uintptr_t data_ptr() const {
    return reinterpret_cast<uintptr_t>(ptr_.contents());
  }

  BufferPtr &ptr() { return ptr_; }

private:
  BufferPtr ptr_;
};

/**
 * @brief MetalContext wrapper for _cpp_ext compatibility.
 * Consolidates device, queues, and library management.
 */
class MetalContext {
public:
  MetalContext() : device_(get_default_device()) {}

  void load_metallib(const std::string &path) {
    // Use LibraryManager for caching, but also expose raw library via
    // direct_access
    LibraryManager::instance().load_metallib(path);
  }

  nb::object get_pipeline(const std::string &kernel_name,
                          const std::string &metallib_path = "") {
    MTL::Library *lib = nullptr;
    if (!metallib_path.empty()) {
      lib = LibraryManager::instance().load_metallib(metallib_path);
    }

    MTL::Function *func =
        LibraryManager::instance().get_kernel(kernel_name, lib);
    if (!func) {
      throw std::runtime_error("Kernel function not found: " + kernel_name);
    }

    NSError *error = nullptr;
    id<MTLComputePipelineState> pipeline =
        [device_.device() newComputePipelineStateWithFunction:(id)func
                                                        error:&error];

    if (!pipeline) {
      std::string msg = "Failed to create pipeline";
      if (error) {
        msg += ": " + std::string([[error localizedDescription] UTF8String]);
      }
      throw std::runtime_error(msg);
    }

    return nb::capsule(pipeline, "Pipeline");
  }

  MetalDeviceDirect &device() { return device_; }

private:
  MetalDeviceDirect &device_;
};

/**
 * @brief BatchDispatch for _cpp_ext compatibility.
 */
class BatchDispatch {
public:
  explicit BatchDispatch(MetalContext &ctx) : ctx_(ctx) {
    cmd_ =
        (id<MTLCommandBuffer>)[(id)ctx_.device().primary_queue() commandBuffer];
    encoder_ = [cmd_ computeCommandEncoder];
  }

  void add_kernel(nb::capsule pipeline_capsule,
                  std::tuple<uint32_t, uint32_t, uint32_t> grid,
                  std::tuple<uint32_t, uint32_t, uint32_t> threadgroup,
                  const std::vector<ManagedBuffer *> &buffers) {
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)pipeline_capsule.data();
    [encoder_ setComputePipelineState:pipeline];

    for (size_t i = 0; i < buffers.size(); ++i) {
      [encoder_ setBuffer:buffers[i]->ptr().buffer() offset:0 atIndex:i];
    }

    MTLSize grid_size =
        MTLSizeMake(std::get<0>(grid), std::get<1>(grid), std::get<2>(grid));
    MTLSize tg_size =
        MTLSizeMake(std::get<0>(threadgroup), std::get<1>(threadgroup),
                    std::get<2>(threadgroup));
    [encoder_ dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
  }

  void commit(bool wait = true) {
    [encoder_ endEncoding];
    [cmd_ commit];
    if (wait) {
      [cmd_ waitUntilCompleted];
    }
  }

private:
  MetalContext &ctx_;
  id<MTLCommandBuffer> cmd_;
  id<MTLComputeCommandEncoder> encoder_;
};

// =============================================================================
// Buffer Pool Bindings
// =============================================================================

void bind_buffer_pool(nb::module_ &m) {
  // BufferPriority enum
  nb::enum_<BufferPriority>(m, "BufferPriority")
      .value("LOW", BufferPriority::LOW)
      .value("NORMAL", BufferPriority::NORMAL)
      .value("HIGH", BufferPriority::HIGH)
      .value("PINNED", BufferPriority::PINNED);

  // StorageMode enum
  nb::enum_<StorageMode>(m, "StorageMode")
      .value("SHARED", StorageMode::SHARED)
      .value("MANAGED", StorageMode::MANAGED)
      .value("PRIVATE", StorageMode::PRIVATE)
      .value("MEMORYLESS", StorageMode::MEMORYLESS);

  // BufferPoolMetrics struct
  nb::class_<BufferPoolMetrics>(m, "BufferPoolMetrics")
      .def_ro("cache_hits", &BufferPoolMetrics::cache_hits)
      .def_ro("cache_misses", &BufferPoolMetrics::cache_misses)
      .def_ro("allocations", &BufferPoolMetrics::allocations)
      .def_ro("releases", &BufferPoolMetrics::releases)
      .def_ro("evictions", &BufferPoolMetrics::evictions)
      .def_ro("bytes_evicted", &BufferPoolMetrics::bytes_evicted)
      .def_ro("current_allocated", &BufferPoolMetrics::current_allocated)
      .def_ro("current_pooled", &BufferPoolMetrics::current_pooled)
      .def_ro("peak_allocated", &BufferPoolMetrics::peak_allocated)
      .def_ro("peak_total", &BufferPoolMetrics::peak_total)
      .def("hit_rate", &BufferPoolMetrics::hit_rate)
      .def("reset", &BufferPoolMetrics::reset);

  // BufferHandle struct - expose as lightweight object
  nb::class_<BufferHandle>(m, "BufferHandle")
      .def_ro("size", &BufferHandle::size)
      .def_ro("aligned_size", &BufferHandle::aligned_size)
      .def_ro("priority", &BufferHandle::priority)
      .def_ro("buffer_id", &BufferHandle::buffer_id)
      .def("length", &BufferHandle::length)
      .def("contents",
           [](BufferHandle &h) -> nb::object {
             void *ptr = h.contents();
             if (!ptr)
               return nb::none();
             return nb::capsule(ptr, "buffer_contents");
           })
      .def("__repr__", [](const BufferHandle &h) {
        return "BufferHandle(size=" + std::to_string(h.size) +
               ", id=" + std::to_string(h.buffer_id) + ")";
      });

  // BufferPool class
  nb::class_<BufferPool>(m, "BufferPool")
      .def(nb::init<void *, size_t, StorageMode>(), nb::arg("device_ptr"),
           nb::arg("heap_size") = 256 * 1024 * 1024,
           nb::arg("storage_mode") = StorageMode::SHARED,
           "Create a buffer pool with given device pointer and heap size")
      .def(
          "allocate",
          [](BufferPool &pool, size_t size,
             BufferPriority priority) -> nb::object {
            auto result = pool.allocate(size, priority);
            if (!result)
              return nb::none();
            return nb::cast(*result);
          },
          nb::arg("size"), nb::arg("priority") = BufferPriority::NORMAL,
          "Allocate a buffer from the pool")
      .def("release", &BufferPool::release, "Release a buffer back to the pool")
      .def("pin", &BufferPool::pin, "Pin buffer to prevent eviction")
      .def("unpin", &BufferPool::unpin, "Unpin buffer to allow eviction")
      .def("evict", &BufferPool::evict, nb::arg("target_bytes"),
           "Evict buffers to free at least target_bytes")
      .def("compact", &BufferPool::compact, "Compact pool by merging buffers")
      .def("clear", &BufferPool::clear, "Clear entire pool")
      .def("metrics", &BufferPool::metrics, nb::rv_policy::reference_internal)
      .def("heap_size", &BufferPool::heap_size, "Get heap size in bytes")
      .def("allocated_bytes", &BufferPool::allocated_bytes,
           "Get currently allocated bytes")
      .def("pooled_bytes", &BufferPool::pooled_bytes,
           "Get pooled but unused bytes")
      .def("available_bytes", &BufferPool::available_bytes,
           "Get available bytes in heap")
      // Convenience allocation methods
      .def(
          "allocate_weight",
          [](BufferPool &pool, size_t size) -> nb::object {
            auto result = pool.allocate_weight(size);
            if (!result)
              return nb::none();
            return nb::cast(*result);
          },
          nb::arg("size"), "Allocate with PINNED priority")
      .def(
          "allocate_activation",
          [](BufferPool &pool, size_t size) -> nb::object {
            auto result = pool.allocate_activation(size);
            if (!result)
              return nb::none();
            return nb::cast(*result);
          },
          nb::arg("size"), "Allocate with HIGH priority")
      .def(
          "allocate_output",
          [](BufferPool &pool, size_t size) -> nb::object {
            auto result = pool.allocate_output(size);
            if (!result)
              return nb::none();
            return nb::cast(*result);
          },
          nb::arg("size"), "Allocate with LOW priority");

  // TransientRingBuffer class
  nb::class_<TransientRingBuffer>(m, "TransientRingBuffer")
      .def(nb::init<void *, size_t, StorageMode>(), nb::arg("device_ptr"),
           nb::arg("capacity") = 100 * 1024 * 1024,
           nb::arg("storage_mode") = StorageMode::SHARED)
      .def(
          "allocate",
          [](TransientRingBuffer &rb, size_t size) -> nb::object {
            auto result = rb.allocate(size);
            if (!result)
              return nb::none();
            return nb::make_tuple(nb::capsule(result->first, "mtlbuffer"),
                                  result->second);
          },
          "Allocate from ring buffer, returns (buffer, offset)")
      .def(
          "allocate_bytes",
          [](TransientRingBuffer &rb, size_t size) -> nb::object {
            auto result = rb.allocate_bytes(size);
            if (!result)
              return nb::none();
            return nb::make_tuple(nb::capsule(result->first, "raw_ptr"),
                                  result->second);
          },
          "Allocate and return raw bytes pointer")
      .def("reset", &TransientRingBuffer::reset, "Reset ring buffer (O(1))")
      .def("capacity", &TransientRingBuffer::capacity)
      .def("used", &TransientRingBuffer::used)
      .def("available", &TransientRingBuffer::available)
      .def("utilization", &TransientRingBuffer::utilization);

  // NOTE: HeapAllocator bindings temporarily disabled due to MTLBufferRef type
  // mismatch between C++ and ObjC++ compilation units. Use pure-Python
  // metal_marlin.heap_allocator.MetalHeapAllocator instead.
#if 0
  // HeapAllocator class
  nb::class_<HeapAllocator>(m, "HeapAllocator")
      .def(nb::init<void *, size_t, StorageMode>(), nb::arg("device_ptr"),
           nb::arg("heap_size") = 256 * 1024 * 1024,
           nb::arg("storage_mode") = StorageMode::SHARED)
      .def(
          "allocate",
          [](HeapAllocator &ha, size_t size) -> nb::object {
            auto result = ha.allocate(size);
            if (!result)
              return nb::none();
            return nb::make_tuple(nb::capsule(result->first, "mtlbuffer"),
                                  result->second);
          },
          "Allocate from heap, returns (buffer, offset)")
      .def(
          "allocate_buffer",
          [](HeapAllocator &ha, size_t size) -> nb::object {
            auto result = ha.allocate_buffer(size);
            if (!result)
              return nb::none();
            return nb::capsule(*result, "mtlbuffer");
          },
          "Allocate buffer only")
      .def(
          "get_offset",
          [](HeapAllocator &ha, nb::capsule buffer) -> nb::object {
            auto result = ha.get_offset((id<MTLBuffer>)buffer.data());
            if (!result)
              return nb::none();
            return nb::cast(*result);
          },
          "Get offset for buffer")
      .def(
          "release",
          [](HeapAllocator &ha, nb::capsule buffer) {
            ha.release((id<MTLBuffer>)buffer.data());
          },
          "Release buffer back to pool")
      .def("compact", &HeapAllocator::compact)
      .def("clear", &HeapAllocator::clear)
      .def("heap_size", &HeapAllocator::heap_size)
      .def("allocated_bytes", &HeapAllocator::allocated_bytes)
      .def("available_bytes", &HeapAllocator::available_bytes);
#endif
}

// =============================================================================
// Encoder Cache Bindings
// =============================================================================

void bind_encoder_cache(nb::module_ &m) {
  // EncoderCacheStats struct
  nb::class_<EncoderCacheStats>(m, "EncoderCacheStats")
      .def_ro("cache_hits", &EncoderCacheStats::cache_hits)
      .def_ro("cache_misses", &EncoderCacheStats::cache_misses)
      .def_ro("total_acquires", &EncoderCacheStats::total_acquires)
      .def_ro("total_commits", &EncoderCacheStats::total_commits)
      .def_ro("total_releases", &EncoderCacheStats::total_releases)
      .def_ro("current_in_use", &EncoderCacheStats::current_in_use)
      .def_ro("peak_in_use", &EncoderCacheStats::peak_in_use)
      .def("hit_rate", &EncoderCacheStats::hit_rate)
      .def("reset", &EncoderCacheStats::reset);

  // Note: EncoderCache requires Objective-C++ and MTLCommandQueue
  // We expose a Python-friendly wrapper
  nb::class_<EncoderCache>(m, "EncoderCache")
      .def(nb::init<>()) // Default constructor for placeholder
      .def("pool_size", &EncoderCache::pool_size)
      .def("in_use_count", &EncoderCache::in_use_count)
      .def("inflight_count", &EncoderCache::inflight_count)
      .def("stats", &EncoderCache::stats, nb::rv_policy::reference_internal)
      .def("reset_stats", &EncoderCache::reset_stats)
      .def("clear_inflight", &EncoderCache::clear_inflight);

  // ThreadLocalEncoderCache
  nb::class_<ThreadLocalEncoderCache>(m, "ThreadLocalEncoderCache")
      .def(nb::init<>())
      .def("stats", [](ThreadLocalEncoderCache &cache) -> nb::object {
        // Returns a dict since stats() returns const reference
        auto &s = cache.stats();
        auto dict = nb::dict();
        dict["cache_hits"] = s.cache_hits.load();
        dict["cache_misses"] = s.cache_misses.load();
        dict["total_acquires"] = s.total_acquires.load();
        dict["total_commits"] = s.total_commits.load();
        dict["current_in_use"] = s.current_in_use.load();
        return dict;
      });
}

// =============================================================================
// MoE Manager Bindings
// =============================================================================

void bind_moe_manager(nb::module_ &m) {
  // DispatchInfo struct
  nb::class_<moe::DispatchInfo>(m, "DispatchInfo")
      .def_ro("sorted_token_indices", &moe::DispatchInfo::sorted_token_indices)
      .def_ro("sorted_expert_indices",
              &moe::DispatchInfo::sorted_expert_indices)
      .def_ro("expert_offsets", &moe::DispatchInfo::expert_offsets)
      .def_ro("inverse_indices", &moe::DispatchInfo::inverse_indices)
      .def_ro("num_tokens", &moe::DispatchInfo::num_tokens)
      .def_ro("top_k", &moe::DispatchInfo::top_k)
      .def_ro("num_experts", &moe::DispatchInfo::num_experts)
      .def("total_assignments", &moe::DispatchInfo::total_assignments)
      .def("expert_batch_size", &moe::DispatchInfo::expert_batch_size)
      .def("is_expert_active", &moe::DispatchInfo::is_expert_active)
      .def("active_expert_count", &moe::DispatchInfo::active_expert_count)
      .def("clear", &moe::DispatchInfo::clear);

  // TokenGroup struct
  nb::class_<moe::TokenGroup>(m, "TokenGroup")
      .def_ro("expert_id", &moe::TokenGroup::expert_id)
      .def_ro("start_idx", &moe::TokenGroup::start_idx)
      .def_ro("end_idx", &moe::TokenGroup::end_idx)
      .def("size", &moe::TokenGroup::size)
      .def("empty", &moe::TokenGroup::empty)
      .def("is_valid", &moe::TokenGroup::is_valid)
      .def("token_index", &moe::TokenGroup::token_index)
      .def("expert_slot", &moe::TokenGroup::expert_slot);

  // TokenGroupManager class
  nb::class_<moe::TokenGroupManager>(m, "TokenGroupManager")
      .def(nb::init<int32_t, int32_t, int32_t>(), nb::arg("num_experts"),
           nb::arg("max_tokens") = moe::kDefaultMaxTokens,
           nb::arg("max_top_k") = moe::kDefaultMaxTopK)
      .def(
          "group_tokens",
          [](moe::TokenGroupManager &mgr,
             const std::vector<int32_t> &expert_ids, int32_t batch_size,
             int32_t top_k) -> moe::DispatchInfo {
            return mgr.group_tokens(expert_ids.data(), batch_size, top_k);
          },
          nb::arg("expert_ids"), nb::arg("batch_size"), nb::arg("top_k"),
          "Group tokens by expert from flattened expert_ids")
      .def(
          "gather_activations_f16",
          [](moe::TokenGroupManager &mgr, nb::bytes activations,
             const moe::DispatchInfo &info, int32_t hidden_dim) -> nb::bytes {
            size_t total_size =
                info.total_assignments() * hidden_dim * sizeof(uint16_t);
            std::vector<uint16_t> gathered(total_size / sizeof(uint16_t));
            mgr.gather_activations(activations.data(), gathered.data(), info,
                                   hidden_dim, sizeof(uint16_t));
            return nb::bytes(reinterpret_cast<const char *>(gathered.data()),
                             gathered.size() * sizeof(uint16_t));
          },
          nb::arg("activations"), nb::arg("info"), nb::arg("hidden_dim"),
          "Gather FP16 activations in expert-sorted order")
      .def(
          "scatter_outputs_f16",
          [](moe::TokenGroupManager &mgr, nb::bytes expert_outputs,
             const std::vector<float> &expert_probs,
             const moe::DispatchInfo &info, int32_t hidden_dim) -> nb::bytes {
            size_t output_size =
                info.num_tokens * hidden_dim * sizeof(uint16_t);
            std::vector<uint16_t> output(output_size / sizeof(uint16_t));
            mgr.scatter_outputs(expert_outputs.data(), expert_probs.data(),
                                output.data(), info, hidden_dim,
                                sizeof(uint16_t));
            return nb::bytes(reinterpret_cast<const char *>(output.data()),
                             output.size() * sizeof(uint16_t));
          },
          nb::arg("expert_outputs"), nb::arg("expert_probs"), nb::arg("info"),
          nb::arg("hidden_dim"),
          "Scatter FP16 expert outputs back to original order")
      .def("get_token_group", &moe::TokenGroupManager::get_token_group)
      .def("compute_expert_loads",
           &moe::TokenGroupManager::compute_expert_loads)
      .def("compute_load_imbalance",
           &moe::TokenGroupManager::compute_load_imbalance)
      .def("is_load_balanced", &moe::TokenGroupManager::is_load_balanced,
           nb::arg("info"), nb::arg("threshold") = 2.0f)
      .def_prop_ro("num_experts", &moe::TokenGroupManager::num_experts)
      .def_prop_ro("max_tokens", &moe::TokenGroupManager::max_tokens)
      .def_prop_ro("max_top_k", &moe::TokenGroupManager::max_top_k);
}

// =============================================================================
// Metal Device Bindings
// =============================================================================

void bind_metal_device(nb::module_ &m) {
  // QueueType enum
  nb::enum_<QueueType>(m, "QueueType")
      .value("PRIMARY", QueueType::Primary)
      .value("DECODE", QueueType::Decode)
      .value("TRANSFER", QueueType::Transfer);

  // MetalDevice class - expose as wrapper
  nb::class_<MetalDevice>(m, "MetalDevice")
      .def_static("default_device", &MetalDevice::default_device)
      .def_static("device_by_name",
                  [](const std::string &name) -> nb::object {
                    auto dev = MetalDevice::device_by_name(name);
                    if (!dev)
                      return nb::none();
                    return nb::cast(std::move(dev));
                  })
      .def("raw",
           [](MetalDevice &dev) { return nb::capsule(dev.raw(), "mtldevice"); })
      .def("name", &MetalDevice::name)
      .def("registry_id", &MetalDevice::registry_id)
      .def("is_low_power", &MetalDevice::is_low_power)
      .def("is_removable", &MetalDevice::is_removable)
      .def("recommended_max_working_set_size",
           &MetalDevice::recommended_max_working_set_size)
      .def("max_buffer_length", &MetalDevice::max_buffer_length)
      .def("max_threadgroup_memory_length",
           &MetalDevice::max_threadgroup_memory_length)
      .def("max_threads_per_threadgroup",
           &MetalDevice::max_threads_per_threadgroup)
      .def("new_buffer",
           [](MetalDevice &dev, size_t length, uint64_t options) -> nb::object {
             auto *buf = dev.new_buffer(length, options);
             if (!buf)
               return nb::none();
             return nb::capsule(buf, "mtlbuffer");
           })
      .def(
          "new_buffer_with_bytes",
          [](MetalDevice &dev, nb::bytes data, uint64_t options) -> nb::object {
            auto *buf =
                dev.new_buffer_with_bytes(data.data(), data.size(), options);
            if (!buf)
              return nb::none();
            return nb::capsule(buf, "mtlbuffer");
          })
      .def("new_command_queue", [](MetalDevice &dev) -> nb::object {
        auto *q = dev.new_command_queue();
        if (!q)
          return nb::none();
        return nb::capsule(q, "mtlcommandqueue");
      });

  // QueueManager class
  nb::class_<QueueManager>(m, "QueueManager")
      .def_static("create",
                  [](nb::capsule device) -> nb::object {
                    auto qm = QueueManager::create(
                        static_cast<MTL::Device *>(device.data()));
                    if (!qm)
                      return nb::none();
                    return nb::cast(std::move(qm));
                  })
      .def_static("create_default", &QueueManager::create_default)
      .def("primary_queue",
           [](QueueManager &qm) {
             return nb::capsule(qm.primary_queue(), "mtlcommandqueue");
           })
      .def("decode_queue",
           [](QueueManager &qm) {
             return nb::capsule(qm.decode_queue(), "mtlcommandqueue");
           })
      .def("transfer_queue",
           [](QueueManager &qm) {
             return nb::capsule(qm.transfer_queue(), "mtlcommandqueue");
           })
      .def("get_queue",
           [](QueueManager &qm, QueueType type) -> nb::object {
             auto *q = qm.get_queue(type);
             if (!q)
               return nb::none();
             return nb::capsule(q, "mtlcommandqueue");
           })
      .def("commit_all", &QueueManager::commit_all)
      .def("wait_for_all", &QueueManager::wait_for_all);
}

// =============================================================================
// Library Manager Bindings
// =============================================================================

void bind_library_manager(nb::module_ &m) {
  // Custom exceptions
  static nb::exception<MetallibNotFoundError> exc_notfound(
      m, "MetallibNotFoundError");
  static nb::exception<MetallibLoadError> exc_load(m, "MetallibLoadError");

  // MetallibVersion struct
  nb::class_<MetallibVersion>(m, "MetallibVersion")
      .def_ro("path", &MetallibVersion::path)
      .def_ro("build_date", &MetallibVersion::build_date)
      .def_ro("size_bytes", &MetallibVersion::size_bytes)
      .def_ro("git_hash", &MetallibVersion::git_hash)
      .def_ro("shader_count", &MetallibVersion::shader_count)
      .def_ro("metal_version", &MetallibVersion::metal_version);

  // StalenessDetails struct
  nb::class_<StalenessDetails>(m, "StalenessDetails")
      .def_ro("metallib_path", &StalenessDetails::metallib_path)
      .def_ro("metallib_exists", &StalenessDetails::metallib_exists)
      .def_ro("is_stale", &StalenessDetails::is_stale)
      .def_ro("reason", &StalenessDetails::reason)
      .def_ro("added_files", &StalenessDetails::added_files)
      .def_ro("removed_files", &StalenessDetails::removed_files)
      .def_ro("modified_files", &StalenessDetails::modified_files)
      .def_ro("has_manifest", &StalenessDetails::has_manifest);

  // LibraryManager singleton
  nb::class_<LibraryManager>(m, "LibraryManager")
      .def_static("instance", &LibraryManager::instance,
                  nb::rv_policy::reference,
                  "Get the singleton LibraryManager instance")
      .def(
          "load_metallib",
          [](LibraryManager &lm, const std::string &path) -> nb::object {
            auto *lib = lm.load_metallib(path);
            if (!lib)
              return nb::none();
            return nb::capsule(lib, "mtllibrary");
          },
          nb::arg("path") = "", "Load a precompiled metallib")
      .def(
          "get_precompiled_library",
          [](LibraryManager &lm, const std::string &path) -> nb::object {
            auto *lib = lm.get_precompiled_library(path);
            if (!lib)
              return nb::none();
            return nb::capsule(lib, "mtllibrary");
          },
          nb::arg("path") = "")
      .def(
          "get_kernel",
          [](LibraryManager &lm, const std::string &kernel_name,
             nb::object library) -> nb::object {
            MTL::Library *lib = nullptr;
            if (!library.is_none()) {
              nb::object ptr_obj = library.attr("__ptr__")();
              if (nb::isinstance<nb::capsule>(ptr_obj)) {
                lib = static_cast<MTL::Library *>(
                    nb::cast<nb::capsule>(ptr_obj).data());
              }
            }
            auto *func = lm.get_kernel(kernel_name, lib);
            if (!func)
              return nb::none();
            return nb::capsule(func, "mtlfunction");
          },
          nb::arg("kernel_name"), nb::arg("library") = nb::none())
      .def("get_metallib_version", &LibraryManager::get_metallib_version,
           nb::arg("path") = "")
      .def("is_metallib_stale", &LibraryManager::is_metallib_stale,
           nb::arg("path") = "")
      .def("get_staleness_details", &LibraryManager::get_staleness_details,
           nb::arg("path") = "")
      .def("clear_cache", &LibraryManager::clear_cache)
      .def("clear_cache_for_path", &LibraryManager::clear_cache_for_path)
      .def("set_default_path", &LibraryManager::set_default_path)
      .def("get_default_path", &LibraryManager::get_default_path)
      .def("compute_file_checksum", &LibraryManager::compute_file_checksum)
      .def("has_metal", &LibraryManager::has_metal)
      .def("require_metal", &LibraryManager::require_metal);
}

// =============================================================================
// Module Definition
// =============================================================================

// =============================================================================
// Module Definition
// =============================================================================

NB_MODULE(_cpp_ext, m) {
  m.doc() = R"doc(
        Metal Marlin C++ Extension - High-performance Metal operations
        (Combined core and extension bindings)
    )doc";

  // Version
  m.attr("__version__") = "0.1.0";

  // Constants
  m.attr("CACHE_LINE_BYTES") = 128;
  m.attr("PAGE_SIZE_BYTES") = 16384;
  m.attr("LARGE_BUFFER_THRESHOLD") = 65536;
  m.attr("DEFAULT_MAX_TOKENS") = moe::kDefaultMaxTokens;
  m.attr("DEFAULT_MAX_TOP_K") = moe::kDefaultMaxTopK;

  // Bind all components
  bind_buffer_pool(m);
  bind_encoder_cache(m);
  bind_moe_manager(m);
  bind_metal_device(m);
  bind_library_manager(m);

  // Bind _cpp_ext compatibility layer
  nb::class_<ManagedBuffer>(m, "ManagedBuffer")
      .def("length", &ManagedBuffer::length)
      .def("data_ptr", &ManagedBuffer::data_ptr);

  nb::class_<MetalContext>(m, "MetalContext")
      .def(nb::init<>())
      .def("load_metallib", &MetalContext::load_metallib)
      .def("get_pipeline", &MetalContext::get_pipeline, nb::arg("kernel_name"),
           nb::arg("metallib_path") = "");

  nb::class_<BatchDispatch>(m, "BatchDispatch")
      .def(nb::init<MetalContext &>())
      .def("add_kernel", &BatchDispatch::add_kernel)
      .def("commit", &BatchDispatch::commit, nb::arg("wait") = true);

  m.def(
      "dispatch_kernel",
      [](MetalContext &ctx, nb::capsule pipeline_capsule,
         std::tuple<uint32_t, uint32_t, uint32_t> grid,
         std::tuple<uint32_t, uint32_t, uint32_t> threadgroup,
         const std::vector<ManagedBuffer *> &buffers, bool wait = true) {
        id<MTLCommandBuffer> cmd =
            [(id)ctx.device().primary_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        id<MTLComputePipelineState> pipeline =
            (id<MTLComputePipelineState>)pipeline_capsule.data();
        [encoder setComputePipelineState:pipeline];

        for (size_t i = 0; i < buffers.size(); ++i) {
          [encoder setBuffer:buffers[i]->ptr().buffer() offset:0 atIndex:i];
        }

        MTLSize grid_size = MTLSizeMake(std::get<0>(grid), std::get<1>(grid),
                                        std::get<2>(grid));
        MTLSize tg_size =
            MTLSizeMake(std::get<0>(threadgroup), std::get<1>(threadgroup),
                        std::get<2>(threadgroup));
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];

        [encoder endEncoding];
        [cmd commit];
        if (wait)
          [cmd waitUntilCompleted];
      },
      nb::arg("ctx"), nb::arg("pipeline"), nb::arg("grid"),
      nb::arg("threadgroup"), nb::arg("buffers"), nb::arg("wait") = true);

  m.def(
      "create_buffer",
      [](MetalContext &ctx, size_t size, bool use_pool) -> ManagedBuffer * {
        // use_pool is ignored for now or simple implementation
        auto ptr =
            ctx.device().create_buffer(size, MTLResourceStorageModeShared);
        return new ManagedBuffer(std::move(ptr));
      },
      nb::arg("ctx"), nb::arg("size"), nb::arg("use_pool") = true);

  m.def("create_buffer_from_numpy", [](MetalContext &ctx, nb::ndarray<> arr) {
      void* data = arr.data();
      size_t size = arr.nbytes();

      // Create shared buffer and copy data using MetalDeviceDirect API
      auto ptr = ctx.device().create_buffer_from_bytes(data, size, MTLResourceStorageModeShared);
      return new ManagedBuffer(std::move(ptr));
  }, nb::arg("ctx"), nb::arg("array"), "Create a Metal buffer from a numpy array");

  m.def(
      "create_buffer_from_ptr",
      [](MetalContext &ctx, uintptr_t ptr, size_t size) -> ManagedBuffer * {
        auto buf_ptr = ctx.device().wrap_pointer((void *)ptr, size);
        return new ManagedBuffer(std::move(buf_ptr));
      },
      nb::arg("ctx"), nb::arg("ptr"), nb::arg("size"));
}
