#pragma once

#include "buffer_pool.hpp"
#include <memory>
#include <mutex>
#include <optional>

namespace metal_marlin {

/**
 * @brief Singleton buffer pool specifically for expert weights.
 *
 * This provides a global access point for managing expert weights in the C++ extension,
 * allowing them to be shared across different components and efficiently managed.
 *
 * Usage:
 *   ExpertBufferPool::instance().initialize(device_ptr);
 *   auto buffer = ExpertBufferPool::instance().allocate_weight(size);
 */
class ExpertBufferPool {
public:
    static ExpertBufferPool& instance();

    // Delete copy/move constructors/assignments
    ExpertBufferPool(const ExpertBufferPool&) = delete;
    ExpertBufferPool& operator=(const ExpertBufferPool&) = delete;
    ExpertBufferPool(ExpertBufferPool&&) = delete;
    ExpertBufferPool& operator=(ExpertBufferPool&&) = delete;

    /**
     * @brief Initialize the global expert buffer pool.
     * @param device_ptr Pointer to the Metal device.
     * @param heap_size Size of the heap in bytes (default 1GB for experts).
     */
    void initialize(void* device_ptr, size_t heap_size = 1024 * 1024 * 1024);

    /**
     * @brief Check if the pool is initialized.
     */
    bool is_initialized() const;

    /**
     * @brief Get the underlying BufferPool instance.
     * @throws std::runtime_error if not initialized.
     */
    BufferPool& get_pool();

    /**
     * @brief Allocate memory for expert weights (PINNED priority).
     * @param size Size in bytes.
     * @return BufferHandle or nullopt if allocation failed.
     */
    std::optional<BufferHandle> allocate_weight(size_t size);

    /**
     * @brief Clear the pool.
     */
    void clear();

    /**
     * @brief Get pool metrics.
     */
    const BufferPoolMetrics* metrics() const;

private:
    ExpertBufferPool() = default;
    ~ExpertBufferPool() = default;

    std::unique_ptr<BufferPool> pool_;
    mutable std::mutex mutex_;
};

} // namespace metal_marlin
