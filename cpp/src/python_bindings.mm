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
#include "gemm_dispatch.hpp"
#include "library_manager.hpp"
#include "metal_device.hpp"
#include "moe_manager.hpp"
#include "moe_router_dispatch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
        (__bridge id<MTLComputePipelineState>)pipeline_capsule.data();
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
            return nb::make_tuple(
                nb::capsule((__bridge void *)result->first, "mtlbuffer"),
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
// Router Dispatcher Bindings
// =============================================================================

void bind_router_dispatcher(nb::module_ &m) {
  // RouterBatchOutput struct
  nb::class_<moe::RouterBatchOutput>(m, "RouterBatchOutput")
      .def_ro("num_tokens", &moe::RouterBatchOutput::num_tokens)
      .def_ro("top_k", &moe::RouterBatchOutput::top_k)
      .def_ro("num_experts", &moe::RouterBatchOutput::num_experts)
      .def_ro("logits", &moe::RouterBatchOutput::logits)
      .def_ro("topk_expert_ids", &moe::RouterBatchOutput::topk_expert_ids)
      .def_ro("topk_probs", &moe::RouterBatchOutput::topk_probs)
      .def_ro("dispatch_info", &moe::RouterBatchOutput::dispatch_info);

  // RouterBuffer struct
  nb::class_<moe::RouterBuffer>(m, "RouterBuffer")
      .def_ro("sequence_id", &moe::RouterBuffer::sequence_id)
      .def_ro("output", &moe::RouterBuffer::output);

  // FastRouterDispatcher class
  nb::class_<moe::FastRouterDispatcher>(m, "FastRouterDispatcher")
      .def(nb::init<int32_t, int32_t, int32_t, int32_t, size_t, uint32_t>(),
           nb::arg("num_experts"),
           nb::arg("hidden_dim"),
           nb::arg("top_k"),
           nb::arg("max_batch_tokens") = 128,
           nb::arg("hot_pair_cache_capacity") = 256,
           nb::arg("hot_pair_threshold") = 8)
      .def("route_batch",
           [](moe::FastRouterDispatcher &dispatcher,
              nb::bytes token_activations_bf16,
              int32_t num_tokens,
              nb::ndarray<float, nb::c_contig> router_weights,
              nb::object router_bias_obj) -> moe::RouterBatchOutput {
                
             const uint16_t* activations_ptr =
                 reinterpret_cast<const uint16_t*>(token_activations_bf16.data());
             
             const float* weights_ptr = router_weights.data();
             const float* bias_ptr = nullptr;
             
             if (!router_bias_obj.is_none()) {
               auto bias_array = nb::cast<nb::ndarray<float, nb::c_contig>>(router_bias_obj);
               bias_ptr = bias_array.data();
             }
             
             return dispatcher.route_batch(activations_ptr,
                                           num_tokens,
                                           weights_ptr,
                                           bias_ptr);
           },
           nb::arg("token_activations_bf16"),
           nb::arg("num_tokens"),
           nb::arg("router_weights"),
           nb::arg("router_bias") = nb::none(),
           "Synchronous batched router forward + top-k + dispatch packing")
      .def("submit_async",
           [](moe::FastRouterDispatcher &dispatcher,
              nb::bytes token_activations_bf16,
              int32_t num_tokens,
              nb::ndarray<float, nb::c_contig> router_weights,
              nb::object router_bias_obj,
              nb::callable launch_experts) {
             
             const uint16_t* activations_ptr =
                 reinterpret_cast<const uint16_t*>(token_activations_bf16.data());
             
             const float* weights_ptr = router_weights.data();
             const float* bias_ptr = nullptr;
             
             if (!router_bias_obj.is_none()) {
               auto bias_array = nb::cast<nb::ndarray<float, nb::c_contig>>(router_bias_obj);
               bias_ptr = bias_array.data();
             }

             // Wrap the Python callable into ExpertLaunchFn
             auto wrapped_launch = [launch_experts](const moe::RouterBuffer& buffer) -> std::shared_future<void> {
               nb::gil_scoped_acquire acquire;
               launch_experts(nb::cast(buffer));
               
               // For now we return a dummy future as Python async might be handled differently
               std::promise<void> p;
               p.set_value();
               return p.get_future().share();
             };

             {
               nb::gil_scoped_release release;
               dispatcher.submit_async(activations_ptr,
                                       num_tokens,
                                       weights_ptr,
                                       bias_ptr,
                                       wrapped_launch);
             }
           },
           nb::arg("token_activations_bf16"),
           nb::arg("num_tokens"),
           nb::arg("router_weights"),
           nb::arg("router_bias") = nb::none(),
           nb::arg("launch_experts"),
           "Asynchronous overlap path with expert launch callback")
      .def("current_router_buffer", &moe::FastRouterDispatcher::current_router_buffer)
      .def("previous_router_buffer", &moe::FastRouterDispatcher::previous_router_buffer)
      .def("reset_hot_pair_cache", &moe::FastRouterDispatcher::reset_hot_pair_cache)
      .def("num_experts", &moe::FastRouterDispatcher::num_experts)
      .def("hidden_dim", &moe::FastRouterDispatcher::hidden_dim)
      .def("top_k", &moe::FastRouterDispatcher::top_k)
      .def("hot_pair_count", &moe::FastRouterDispatcher::hot_pair_count);
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
// Mixed-BPW Dispatch Bindings
// =============================================================================

void bind_mixed_bpw_dispatch(nb::module_ &m) {
  nb::class_<MoEConfig>(m, "MoEConfig")
      .def(nb::init<>())
      .def_rw("hidden_dim", &MoEConfig::hidden_dim)
      .def_rw("intermediate_dim", &MoEConfig::intermediate_dim)
      .def_rw("num_experts", &MoEConfig::num_experts)
      .def_rw("top_k", &MoEConfig::top_k)
      .def_rw("max_experts_per_batch", &MoEConfig::max_experts_per_batch)
      .def_rw("command_buffers_per_batch_size",
              &MoEConfig::command_buffers_per_batch_size)
      .def_rw("max_inflight_submissions",
              &MoEConfig::max_inflight_submissions)
      .def_rw("threadgroup_size_x", &MoEConfig::threadgroup_size_x)
      .def_rw("common_batch_sizes", &MoEConfig::common_batch_sizes)
      .def_rw("use_indirect_command_buffers",
              &MoEConfig::use_indirect_command_buffers)
      .def_rw("overlap_cpu_encoding", &MoEConfig::overlap_cpu_encoding)
      .def_rw("wait_for_completion", &MoEConfig::wait_for_completion)
      .def_rw("kernel_name", &MoEConfig::kernel_name)
      .def_rw("metallib_path", &MoEConfig::metallib_path);

  nb::class_<MixedBPWBatchPlan>(m, "MixedBPWBatchPlan")
      .def_ro("bit_width", &MixedBPWBatchPlan::bit_width)
      .def_ro("expert_ids", &MixedBPWBatchPlan::expert_ids)
      .def_ro("expert_token_counts", &MixedBPWBatchPlan::expert_token_counts)
      .def_ro("token_count", &MixedBPWBatchPlan::token_count);

  nb::class_<MixedBPWDispatchStats>(m, "MixedBPWDispatchStats")
      .def_ro("queued_experts", &MixedBPWDispatchStats::queued_experts)
      .def_ro("routed_experts", &MixedBPWDispatchStats::routed_experts)
      .def_ro("grouped_batches", &MixedBPWDispatchStats::grouped_batches)
      .def_ro("command_buffer_submissions",
              &MixedBPWDispatchStats::command_buffer_submissions)
      .def_ro("indirect_command_batches",
              &MixedBPWDispatchStats::indirect_command_batches);

  nb::class_<BatchDispatchMixedBPW>(m, "BatchDispatchMixedBPW")
      .def(nb::init<>())
      .def("reset", &BatchDispatchMixedBPW::reset)
      .def("add_expert_bits", &BatchDispatchMixedBPW::add_expert_bits,
           nb::arg("expert_bits"))
      .def("set_active_experts", &BatchDispatchMixedBPW::set_active_experts,
           nb::arg("active_expert_ids"))
      .def("clear_active_experts", &BatchDispatchMixedBPW::clear_active_experts)
      .def("reserve_command_buffers",
           &BatchDispatchMixedBPW::reserve_command_buffers,
           nb::arg("common_batch_sizes"),
           nb::arg("command_buffers_per_size") = 2)
      .def("build_batches", &BatchDispatchMixedBPW::build_batches,
           nb::arg("max_experts_per_batch") = 0)
      .def(
          "build_batches_for_routing",
          [](const BatchDispatchMixedBPW &dispatcher,
             nb::ndarray<nb::numpy, int32_t, nb::ndim<2>, nb::c_contig>
                 expert_indices,
             uint32_t max_experts_per_batch) {
            return dispatcher.build_batches_for_routing(
                expert_indices.data(), static_cast<uint32_t>(expert_indices.shape(0)),
                static_cast<uint32_t>(expert_indices.shape(1)),
                max_experts_per_batch);
          },
          nb::arg("expert_indices"), nb::arg("max_experts_per_batch") = 0)
      .def("try_acquire_command_buffer_slot",
           &BatchDispatchMixedBPW::try_acquire_command_buffer_slot,
           nb::arg("batch_size"))
      .def("release_command_buffer_slot",
           &BatchDispatchMixedBPW::release_command_buffer_slot,
           nb::arg("batch_size"))
      .def("command_buffer_slot_key",
           &BatchDispatchMixedBPW::command_buffer_slot_key,
           nb::arg("batch_size"))
      .def("stats",
           [](const BatchDispatchMixedBPW &dispatcher) {
             return dispatcher.stats_snapshot();
           });

  m.def(
      "dispatch_mixed_bpw_moe",
      [](nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig> hidden_states,
         std::vector<nb::ndarray<nb::numpy, nb::c_contig>> expert_weights_packed,
         const std::vector<int> &expert_bits,
         std::vector<nb::ndarray<nb::numpy, nb::c_contig>> expert_scales,
         nb::ndarray<nb::numpy, int32_t, nb::ndim<2>, nb::c_contig>
             expert_indices,
         nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig> expert_probs,
         MoEConfig config) {
        const size_t num_experts = expert_bits.size();
        if (num_experts == 0) {
          return;
        }
        if (hidden_states.shape(0) != expert_indices.shape(0)) {
          throw std::invalid_argument(
              "dispatch_mixed_bpw_moe: hidden_states batch dimension must match "
              "expert_indices");
        }
        if (expert_probs.shape(0) != expert_indices.shape(0)) {
          throw std::invalid_argument(
              "dispatch_mixed_bpw_moe: expert_probs batch dimension must match "
              "expert_indices");
        }
        if (expert_weights_packed.size() != num_experts ||
            expert_scales.size() != num_experts) {
          throw std::invalid_argument(
              "dispatch_mixed_bpw_moe: expert weight/scale array lengths must "
              "match expert_bits length");
        }

        const uint32_t num_tokens = static_cast<uint32_t>(hidden_states.shape(0));
        const uint32_t hidden_dim = static_cast<uint32_t>(hidden_states.shape(1));
        const uint32_t top_k = static_cast<uint32_t>(expert_indices.shape(1));

        if (config.hidden_dim == 0) {
          config.hidden_dim = hidden_dim;
        } else if (config.hidden_dim != hidden_dim) {
          throw std::invalid_argument(
              "dispatch_mixed_bpw_moe: config.hidden_dim must match hidden_states");
        }
        if (config.num_experts == 0) {
          config.num_experts = static_cast<uint32_t>(num_experts);
        } else if (config.num_experts != static_cast<uint32_t>(num_experts)) {
          throw std::invalid_argument(
              "dispatch_mixed_bpw_moe: config.num_experts must match "
              "expert_bits/expert_weights/expert_scales length");
        }
        if (config.top_k == 0) {
          config.top_k = top_k;
        } else if (config.top_k != top_k) {
          throw std::invalid_argument(
              "dispatch_mixed_bpw_moe: config.top_k must match "
              "expert_indices.shape[1]");
        }

        std::vector<const void *> expert_weight_ptrs;
        std::vector<size_t> expert_weight_sizes;
        std::vector<const void *> expert_scale_ptrs;
        std::vector<size_t> expert_scale_sizes;
        expert_weight_ptrs.reserve(num_experts);
        expert_weight_sizes.reserve(num_experts);
        expert_scale_ptrs.reserve(num_experts);
        expert_scale_sizes.reserve(num_experts);

        for (size_t i = 0; i < num_experts; ++i) {
          auto weight_dtype = expert_weights_packed[i].dtype();
          if (weight_dtype.code != static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) ||
              weight_dtype.bits != 8) {
            throw std::invalid_argument(
                "dispatch_mixed_bpw_moe: expert_weights_packed entries must be uint8 arrays");
          }

          auto scale_dtype = expert_scales[i].dtype();
          if (scale_dtype.code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) ||
              scale_dtype.bits != 16) {
            throw std::invalid_argument(
                "dispatch_mixed_bpw_moe: expert_scales entries must be float16 arrays");
          }
          expert_weight_ptrs.push_back(expert_weights_packed[i].data());
          expert_weight_sizes.push_back(expert_weights_packed[i].nbytes());
          expert_scale_ptrs.push_back(expert_scales[i].data());
          expert_scale_sizes.push_back(expert_scales[i].nbytes());
        }

        const float *expert_probs_ptr = nullptr;
        std::vector<float> gathered_topk_probs;
        const int64_t probs_width = expert_probs.shape(1);
        if (probs_width == expert_indices.shape(1)) {
          expert_probs_ptr = expert_probs.data();
        } else if (probs_width == static_cast<int64_t>(num_experts)) {
          gathered_topk_probs.resize(
              static_cast<size_t>(num_tokens) * static_cast<size_t>(top_k));
          const float *all_probs = expert_probs.data();
          const int32_t *idx_ptr = expert_indices.data();
          for (uint32_t token = 0; token < num_tokens; ++token) {
            const size_t token_base = static_cast<size_t>(token) * static_cast<size_t>(top_k);
            const size_t probs_base = static_cast<size_t>(token) * static_cast<size_t>(num_experts);
            for (uint32_t k = 0; k < top_k; ++k) {
              const int32_t expert_id = idx_ptr[token_base + k];
              if (expert_id < 0 || static_cast<size_t>(expert_id) >= num_experts) {
                throw std::invalid_argument(
                    "dispatch_mixed_bpw_moe: expert_indices contains out-of-range "
                    "expert id");
              }
              gathered_topk_probs[token_base + k] =
                  all_probs[probs_base + static_cast<size_t>(expert_id)];
            }
          }
          expert_probs_ptr = gathered_topk_probs.data();
        } else {
          throw std::invalid_argument(
              "dispatch_mixed_bpw_moe: expert_probs shape must be [tokens, top_k] "
              "or [tokens, num_experts]");
        }

        {
          nb::gil_scoped_release release;
          dispatch_mixed_bpw_moe(
              hidden_states.data(),
              expert_weight_ptrs,
              expert_weight_sizes,
              expert_bits,
              expert_scale_ptrs,
              expert_scale_sizes,
              expert_indices.data(),
              expert_probs_ptr,
              num_tokens,
              top_k,
              config);
        }
      },
      nb::arg("hidden_states"), nb::arg("expert_weights_packed"),
      nb::arg("expert_bits"), nb::arg("expert_scales"),
      nb::arg("expert_indices"), nb::arg("expert_probs"), nb::arg("config"));
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
  bind_router_dispatcher(m);
  bind_metal_device(m);
  bind_library_manager(m);
  bind_mixed_bpw_dispatch(m);

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
            (__bridge id<MTLComputePipelineState>)pipeline_capsule.data();
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
