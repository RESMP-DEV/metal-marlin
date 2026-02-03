#pragma once

// Metal-cpp: Apple's C++ interface for Metal
// https://developer.apple.com/metal/cpp/

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace metal_marlin {

/**
 * @brief RAII wrapper for MTLDevice with automatic resource management.
 *
 * MetalDevice provides a C++ interface to Metal GPU devices with:
 * - Automatic device selection (default system GPU)
 * - Device capability queries
 * - Buffer and library creation helpers
 * - Thread-safe singleton access for the default device
 *
 * Usage:
 *   auto device = MetalDevice::default_device();
 *   auto buffer = device->new_buffer(1024, MTL::ResourceStorageModeShared);
 */
class MetalDevice {
public:
    /**
     * @brief Construct a MetalDevice wrapping an existing MTLDevice.
     * @param device Raw MTLDevice pointer. Takes ownership.
     */
    explicit MetalDevice(MTL::Device* device);

    /**
     * @brief Destructor. Releases the MTLDevice.
     */
    ~MetalDevice();

    // Non-copyable
    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;

    // Movable
    MetalDevice(MetalDevice&& other) noexcept;
    MetalDevice& operator=(MetalDevice&& other) noexcept;

    /**
     * @brief Get the default system Metal device (singleton).
     *
     * Thread-safe. Returns nullptr if no Metal device is available.
     */
    static std::shared_ptr<MetalDevice> default_device();

    /**
     * @brief Create a MetalDevice for a specific device by name.
     * @param name Substring to match against device names.
     * @return MetalDevice or nullptr if no match found.
     */
    static std::unique_ptr<MetalDevice> device_by_name(const std::string& name);

    /**
     * @brief Get a device by index.
     * @param index Device index (0-based).
     * @return MetalDevice or nullptr if index is invalid.
     */
    static std::unique_ptr<MetalDevice> device_by_index(size_t index);

    /**
     * @brief Get all available Metal devices.
     * @return Vector of all devices.
     */
    static std::vector<std::unique_ptr<MetalDevice>> all_devices();

    /**
     * @brief Get the number of available Metal devices.
     * @return Device count.
     */
    static size_t device_count();

    /**
     * @brief Get the underlying MTLDevice pointer.
     *
     * The returned pointer is valid for the lifetime of this MetalDevice.
     */
    MTL::Device* raw() const noexcept { return _device; }

    /**
     * @brief Get device name (e.g., "Apple M3 Max").
     */
    std::string name() const;

    /**
     * @brief Get the registry ID for this device.
     */
    uint64_t registry_id() const;

    /**
     * @brief Check if device supports a GPU family.
     * @param family GPU family to check (e.g., MTL::GPUFamilyApple9).
     */
    bool supports_family(MTL::GPUFamily family) const;

    /**
     * @brief Check if device is low-power (integrated GPU).
     */
    bool is_low_power() const;

    /**
     * @brief Check if device is removable (external GPU).
     */
    bool is_removable() const;

    /**
     * @brief Get recommended maximum working set size in bytes.
     */
    size_t recommended_max_working_set_size() const;

    /**
     * @brief Get maximum buffer length in bytes.
     */
    size_t max_buffer_length() const;

    /**
     * @brief Get maximum threadgroup memory length in bytes.
     */
    size_t max_threadgroup_memory_length() const;

    /**
     * @brief Get maximum threads per threadgroup.
     */
    NS::UInteger max_threads_per_threadgroup() const;

    // -------------------------------------------------------------------------
    // Resource Creation
    // -------------------------------------------------------------------------

    /**
     * @brief Create a new buffer.
     * @param length Buffer size in bytes.
     * @param options Resource options (storage mode, CPU cache mode, etc.).
     * @return MTLBuffer pointer. Caller is responsible for release.
     */
    MTL::Buffer* new_buffer(size_t length, MTL::ResourceOptions options) const;

    /**
     * @brief Create a buffer with initial data.
     * @param data Pointer to source data.
     * @param length Data size in bytes.
     * @param options Resource options.
     * @return MTLBuffer pointer. Caller is responsible for release.
     */
    MTL::Buffer* new_buffer_with_bytes(
        const void* data,
        size_t length,
        MTL::ResourceOptions options
    ) const;

    /**
     * @brief Create a new command queue.
     * @return MTLCommandQueue pointer. Caller is responsible for release.
     */
    MTL::CommandQueue* new_command_queue() const;

    /**
     * @brief Create a command queue with a maximum number of command buffers.
     * @param max_command_buffers Maximum concurrent command buffers.
     * @return MTLCommandQueue pointer. Caller is responsible for release.
     */
    MTL::CommandQueue* new_command_queue(NS::UInteger max_command_buffers) const;

    /**
     * @brief Create a library from Metal source code.
     * @param source Metal shader source code.
     * @param options Compile options (can be nullptr for defaults).
     * @param error Output error if compilation fails.
     * @return MTLLibrary pointer or nullptr on failure.
     */
    MTL::Library* new_library_with_source(
        const std::string& source,
        MTL::CompileOptions* options,
        NS::Error** error
    ) const;

    /**
     * @brief Load a precompiled library from a metallib file.
     * @param path Path to the .metallib file.
     * @param error Output error if loading fails.
     * @return MTLLibrary pointer or nullptr on failure.
     */
    MTL::Library* new_library_with_file(
        const std::string& path,
        NS::Error** error
    ) const;

    /**
     * @brief Create a compute pipeline state for a function.
     * @param function Compute function from a library.
     * @param error Output error if creation fails.
     * @return MTLComputePipelineState pointer or nullptr on failure.
     */
    MTL::ComputePipelineState* new_compute_pipeline_state(
        MTL::Function* function,
        NS::Error** error
    ) const;

    /**
     * @brief Create a heap for sub-allocating buffers.
     * @param descriptor Heap descriptor specifying size and options.
     * @return MTLHeap pointer. Caller is responsible for release.
     */
    MTL::Heap* new_heap(MTL::HeapDescriptor* descriptor) const;

private:
    MTL::Device* _device;

    // Singleton state for default device
    static std::shared_ptr<MetalDevice> _default_device;
    static std::once_flag _default_device_init;
};

/**
 * @brief Queue type enumeration for QueueManager.
 */
enum class QueueType {
    Primary,   ///< General compute work
    Decode,    ///< Low-latency inference work
    Transfer   ///< Async data transfers
};

/**
 * @brief Manages multiple command queues for different workload types.
 *
 * QueueManager provides a unified interface for managing multiple Metal
 * command queues, enabling workload separation and parallel execution:
 * - Primary queue: General GPU compute operations
 * - Decode queue: Low-latency inference (overlaps with primary)
 * - Transfer queue: Async memory transfers (when available)
 *
 * Usage:
 *   auto queues = QueueManager::create(device->raw());
 *   auto buffer = queues->primary_queue()->commandBuffer();
 *   // ... encode work ...
 *   buffer->commit();
 */
class QueueManager {
public:
    /**
     * @brief Construct a QueueManager for the given device.
     * @param device MTLDevice to create queues on.
     * @throws std::runtime_error if queue creation fails.
     */
    explicit QueueManager(MTL::Device* device);

    /**
     * @brief Destructor. Releases all queues and device reference.
     */
    ~QueueManager();

    // Non-copyable
    QueueManager(const QueueManager&) = delete;
    QueueManager& operator=(const QueueManager&) = delete;

    // Movable
    QueueManager(QueueManager&& other) noexcept;
    QueueManager& operator=(QueueManager&& other) noexcept;

    /**
     * @brief Get the primary compute queue.
     * @return MTLCommandQueue for general compute work.
     */
    MTL::CommandQueue* primary_queue() const noexcept;

    /**
     * @brief Get the decode queue for low-latency work.
     * @return MTLCommandQueue for inference/decode operations.
     */
    MTL::CommandQueue* decode_queue() const noexcept;

    /**
     * @brief Get the transfer queue for async data movement.
     * @return MTLCommandQueue for memory transfers.
     */
    MTL::CommandQueue* transfer_queue() const noexcept;

    /**
     * @brief Get a queue by type.
     * @param type Queue type to retrieve.
     * @return MTLCommandQueue pointer or nullptr if invalid type.
     */
    MTL::CommandQueue* get_queue(QueueType type) const;

    /**
     * @brief Signal that all pending work should be submitted.
     *
     * Note: Actual commit happens on individual command buffers.
     * This method is for API completeness.
     */
    void commit_all();

    /**
     * @brief Wait for all queues to complete pending work.
     *
     * Creates synchronization points on all queues and blocks until complete.
     * Use sparingly as it stalls the GPU pipeline.
     */
    void wait_for_all();

    /**
     * @brief Factory method to create a QueueManager.
     * @param device MTLDevice to create queues on.
     * @return Unique pointer to QueueManager or nullptr on failure.
     */
    static std::unique_ptr<QueueManager> create(MTL::Device* device);

    /**
     * @brief Factory method using the default Metal device.
     * @return Unique pointer to QueueManager or nullptr on failure.
     */
    static std::unique_ptr<QueueManager> create_default();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_marlin
