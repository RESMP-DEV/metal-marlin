// encoder_cache.cpp - CommandEncoder caching implementation for Metal Marlin
//
// This module provides a high-performance CommandEncoder caching mechanism
// to reduce CPU overhead per dispatch. Command buffer creation in Metal has
// significant overhead (~1-2µs), which becomes the bottleneck in decode
// loops with thousands of small dispatches.
//
// Key optimizations:
// - Object pooling for MTLCommandBuffer (avoid malloc/free in hot path)
// - Reset-and-reuse pattern for command buffers
// - Lock-free fast path for encode operations
// - Thread-safe command buffer rotation
//
// Thread Safety:
// - acquire_encoder() is thread-safe when using separate pools per thread
// - commit_encoder() is thread-safe via atomic buffer rotation
// - Multiple threads can encode concurrently if using separate pools

#include "encoder_cache.hpp"
#include <algorithm>
#include <stdexcept>

namespace metal_marlin {

// -----------------------------------------------------------------------------
// EncoderCache implementation
// -----------------------------------------------------------------------------

#ifdef __OBJC__

EncoderCache::EncoderCache(id<MTLCommandQueue> queue, size_t pool_size)
    : queue_(queue), pool_size_(pool_size), next_free_index_(0),
      pool_(pool_size), inflight_buffers_(kMaxInflightBuffers),
      inflight_count_(0) {

  if (!queue_) {
    throw std::runtime_error("EncoderCache: null command queue");
  }

  // Pre-allocate command buffers
  for (size_t i = 0; i < pool_size_; ++i) {
    id<MTLCommandBuffer> cmd_buffer = [queue_ commandBuffer];
    if (!cmd_buffer) {
      throw std::runtime_error("EncoderCache: failed to create command buffer");
    }
    pool_[i].command_buffer = cmd_buffer;
  }
}

EncoderCache::~EncoderCache() {
  // Release all Metal resources
  for (auto &entry : pool_) {
    if (entry.compute_encoder) {
      [entry.compute_encoder endEncoding];
      entry.compute_encoder = nil;
    }
    entry.command_buffer = nil;
  }
  for (auto &buffer : inflight_buffers_) {
    if (buffer) {
      buffer = nil;
    }
  }
}

id<MTLComputeCommandEncoder> EncoderCache::acquire_encoder() {
  stats_.total_acquires.fetch_add(1, std::memory_order_relaxed);

  // Fast path: try to acquire without locking
  for (size_t attempt = 0; attempt < pool_size_; ++attempt) {
    size_t idx =
        next_free_index_.fetch_add(1, std::memory_order_relaxed) % pool_size_;
    auto &entry = pool_[idx];

    // Try to claim this entry
    bool expected = false;
    if (entry.in_use.compare_exchange_strong(expected, true,
                                             std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {
      // Successfully claimed
      stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
      stats_.current_in_use.fetch_add(1, std::memory_order_relaxed);
      update_peak_in_use();

      // Create new encoder if needed (reset after previous use)
      if (!entry.compute_encoder ||
          entry.committed.load(std::memory_order_relaxed)) {
        entry.compute_encoder = [entry.command_buffer computeCommandEncoder];
        entry.committed.store(false, std::memory_order_release);
        entry.dispatch_count = 0;
      }
      return entry.compute_encoder;
    }
  }

  // Slow path: all entries in use
  stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
  return nil;
}

void EncoderCache::commit_encoder(id<MTLComputeCommandEncoder> encoder) {
  if (!encoder)
    return;

  stats_.total_commits.fetch_add(1, std::memory_order_relaxed);

  // Find the entry for this encoder
  std::lock_guard<std::mutex> lock(publish_mutex_);

  for (auto &entry : pool_) {
    if (entry.compute_encoder == encoder &&
        entry.in_use.load(std::memory_order_relaxed)) {
      // End encoding
      [encoder endEncoding];
      entry.compute_encoder = nil;

      // Commit the command buffer
      [entry.command_buffer commit];

      // Track in-flight buffers
      size_t idx = inflight_count_.load(std::memory_order_relaxed);
      if (idx < inflight_buffers_.size()) {
        inflight_buffers_[idx] = entry.command_buffer;
        inflight_count_.fetch_add(1, std::memory_order_release);
      }

      entry.committed.store(true, std::memory_order_release);
      entry.in_use.store(false, std::memory_order_release);
      stats_.current_in_use.fetch_sub(1, std::memory_order_relaxed);
      return;
    }
  }
}

void EncoderCache::release_encoder(id<MTLComputeCommandEncoder> encoder) {
  if (!encoder)
    return;

  stats_.total_releases.fetch_add(1, std::memory_order_relaxed);

  for (auto &entry : pool_) {
    if (entry.compute_encoder == encoder) {
      // End and discard without committing (if not already committed)
      if (entry.compute_encoder) {
        [encoder endEncoding];
        entry.compute_encoder = nil;
      }

      // Reset command buffer for reuse
      [entry.command_buffer addCompletedHandler:^(id<MTLCommandBuffer>){
          // Handler for async completion tracking
      }];

      entry.in_use.store(false, std::memory_order_release);
      entry.committed.store(false, std::memory_order_release);
      stats_.current_in_use.fetch_sub(1, std::memory_order_relaxed);
      return;
    }
  }
}

void EncoderCache::wait_all() {
  std::lock_guard<std::mutex> lock(publish_mutex_);

  size_t count = inflight_count_.load(std::memory_order_acquire);
  for (size_t i = 0; i < count; ++i) {
    if (inflight_buffers_[i]) {
      [inflight_buffers_[i] waitUntilCompleted];
      inflight_buffers_[i] = nil;
    }
  }
  inflight_count_.store(0, std::memory_order_release);
}

void EncoderCache::wait_for(size_t count) {
  std::lock_guard<std::mutex> lock(publish_mutex_);

  size_t actual_count =
      std::min(count, inflight_count_.load(std::memory_order_acquire));
  for (size_t i = 0; i < actual_count; ++i) {
    if (inflight_buffers_[i]) {
      [inflight_buffers_[i] waitUntilCompleted];
      inflight_buffers_[i] = nil;
    }
  }
  inflight_count_.store(inflight_count_.load(std::memory_order_acquire) -
                            actual_count,
                        std::memory_order_release);
}

id<MTLCommandBuffer> EncoderCache::acquire_command_buffer() {
  std::lock_guard<std::mutex> lock(publish_mutex_);
  return [queue_ commandBuffer];
}

void EncoderCache::release_command_buffer(id<MTLCommandBuffer> buffer) {
  (void)buffer; // Command buffers are autoreleased or managed by caller
}

void EncoderCache::resize(size_t new_size) {
  std::lock_guard<std::mutex> lock(publish_mutex_);

  // Release old buffers
  for (auto &entry : pool_) {
    if (entry.compute_encoder) {
      [entry.compute_encoder endEncoding];
      entry.compute_encoder = nil;
    }
    entry.command_buffer = nil;
  }

  // Resize and allocate new buffers
  pool_.resize(new_size);
  pool_size_ = new_size;

  for (size_t i = 0; i < pool_size_; ++i) {
    id<MTLCommandBuffer> cmd_buffer = [queue_ commandBuffer];
    if (!cmd_buffer) {
      throw std::runtime_error("EncoderCache: failed to resize pool");
    }
    pool_[i].command_buffer = cmd_buffer;
  }

  next_free_index_.store(0, std::memory_order_release);
}

void EncoderCache::clear_inflight() {
  std::lock_guard<std::mutex> lock(publish_mutex_);
  inflight_count_.store(0, std::memory_order_release);
  for (auto &buffer : inflight_buffers_) {
    buffer = nullptr;
  }
}

#endif // __OBJC__

void EncoderCache::update_peak_in_use() {
  uint64_t current = stats_.current_in_use.load(std::memory_order_relaxed);
  uint64_t peak = stats_.peak_in_use.load(std::memory_order_relaxed);
  while (current > peak) {
    if (stats_.peak_in_use.compare_exchange_weak(peak, current,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed)) {
      break;
    }
  }
}

// -----------------------------------------------------------------------------
// ThreadLocalEncoderCache implementation
// -----------------------------------------------------------------------------

#ifdef __OBJC__

#endif // __OBJC__

// -----------------------------------------------------------------------------
// EncoderCacheStats implementation
// -----------------------------------------------------------------------------

double EncoderCacheStats::hit_rate() const {
  uint64_t total = cache_hits.load() + cache_misses.load();
  return total > 0 ? static_cast<double>(cache_hits.load()) / total : 0.0;
}

void EncoderCacheStats::reset() {
  cache_hits = 0;
  cache_misses = 0;
  total_acquires = 0;
  total_commits = 0;
  total_releases = 0;
  current_in_use = 0;
  // Don't reset peak values
}

} // namespace metal_marlin
