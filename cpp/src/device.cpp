#include "metal_device.hpp"

#include <stdexcept>
#include <vector>

namespace metal_marlin {

// =============================================================================
// MetalDevice Implementation
// =============================================================================

// Static members
std::shared_ptr<MetalDevice> MetalDevice::_default_device = nullptr;
std::once_flag MetalDevice::_default_device_init;

MetalDevice::MetalDevice(MTL::Device* device)
    : _device(device)
{
    if (!_device) {
        throw std::runtime_error("MetalDevice: received null device pointer");
    }
    // Retain the device since we own it
    _device->retain();
}

MetalDevice::~MetalDevice() {
    if (_device) {
        _device->release();
        _device = nullptr;
    }
}

MetalDevice::MetalDevice(MetalDevice&& other) noexcept
    : _device(other._device)
{
    other._device = nullptr;
}

MetalDevice& MetalDevice::operator=(MetalDevice&& other) noexcept {
    if (this != &other) {
        if (_device) {
            _device->release();
        }
        _device = other._device;
        other._device = nullptr;
    }
    return *this;
}

std::shared_ptr<MetalDevice> MetalDevice::default_device() {
    std::call_once(_default_device_init, []() {
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        if (device) {
            _default_device = std::make_shared<MetalDevice>(device);
            // CreateSystemDefaultDevice returns a retained device, but our
            // constructor also retains, so release the extra retain count
            device->release();
        }
    });
    return _default_device;
}

std::unique_ptr<MetalDevice> MetalDevice::device_by_name(const std::string& name) {
    NS::Array* devices = MTL::CopyAllDevices();
    if (!devices) {
        return nullptr;
    }

    std::unique_ptr<MetalDevice> result = nullptr;

    for (NS::UInteger i = 0; i < devices->count(); ++i) {
        MTL::Device* device = static_cast<MTL::Device*>(devices->object(i));
        const char* device_name = device->name()->utf8String();
        if (device_name && std::string(device_name).find(name) != std::string::npos) {
            result = std::make_unique<MetalDevice>(device);
            break;
        }
    }

    devices->release();
    return result;
}

std::unique_ptr<MetalDevice> MetalDevice::device_by_index(size_t index) {
    NS::Array* devices = MTL::CopyAllDevices();
    if (!devices || index >= static_cast<size_t>(devices->count())) {
        if (devices) {
            devices->release();
        }
        return nullptr;
    }

    MTL::Device* device = static_cast<MTL::Device*>(devices->object(index));
    auto result = std::make_unique<MetalDevice>(device);
    devices->release();
    return result;
}

std::vector<std::unique_ptr<MetalDevice>> MetalDevice::all_devices() {
    NS::Array* devices = MTL::CopyAllDevices();
    if (!devices) {
        return {};
    }

    std::vector<std::unique_ptr<MetalDevice>> result;
    result.reserve(devices->count());

    for (NS::UInteger i = 0; i < devices->count(); ++i) {
        MTL::Device* device = static_cast<MTL::Device*>(devices->object(i));
        result.push_back(std::make_unique<MetalDevice>(device));
    }

    devices->release();
    return result;
}

size_t MetalDevice::device_count() {
    NS::Array* devices = MTL::CopyAllDevices();
    if (!devices) {
        return 0;
    }

    size_t count = static_cast<size_t>(devices->count());
    devices->release();
    return count;
}

std::string MetalDevice::name() const {
    const char* n = _device->name()->utf8String();
    return n ? std::string(n) : "";
}

uint64_t MetalDevice::registry_id() const {
    return _device->registryID();
}

bool MetalDevice::supports_family(MTL::GPUFamily family) const {
    return _device->supportsFamily(family);
}

bool MetalDevice::is_low_power() const {
    return _device->lowPower();
}

bool MetalDevice::is_removable() const {
    return _device->removable();
}

size_t MetalDevice::recommended_max_working_set_size() const {
    return _device->recommendedMaxWorkingSetSize();
}

size_t MetalDevice::max_buffer_length() const {
    return _device->maxBufferLength();
}

size_t MetalDevice::max_threadgroup_memory_length() const {
    return _device->maxThreadgroupMemoryLength();
}

NS::UInteger MetalDevice::max_threads_per_threadgroup() const {
    return _device->maxThreadsPerThreadgroup().width;
}

// Resource Creation

MTL::Buffer* MetalDevice::new_buffer(size_t length, MTL::ResourceOptions options) const {
    return _device->newBuffer(length, options);
}

MTL::Buffer* MetalDevice::new_buffer_with_bytes(
    const void* data,
    size_t length,
    MTL::ResourceOptions options
) const {
    return _device->newBuffer(data, length, options);
}

MTL::CommandQueue* MetalDevice::new_command_queue() const {
    return _device->newCommandQueue();
}

MTL::CommandQueue* MetalDevice::new_command_queue(NS::UInteger max_command_buffers) const {
    return _device->newCommandQueue(max_command_buffers);
}

MTL::Library* MetalDevice::new_library_with_source(
    const std::string& source,
    MTL::CompileOptions* options,
    NS::Error** error
) const {
    NS::String* ns_source = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    return _device->newLibrary(ns_source, options, error);
}

MTL::Library* MetalDevice::new_library_with_file(
    const std::string& path,
    NS::Error** error
) const {
    NS::String* ns_path = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    NS::URL* url = NS::URL::fileURLWithPath(ns_path);
    return _device->newLibrary(url, error);
}

MTL::ComputePipelineState* MetalDevice::new_compute_pipeline_state(
    MTL::Function* function,
    NS::Error** error
) const {
    return _device->newComputePipelineState(function, error);
}

MTL::Heap* MetalDevice::new_heap(MTL::HeapDescriptor* descriptor) const {
    return _device->newHeap(descriptor);
}

// =============================================================================
// QueueManager Implementation
// =============================================================================

/**
 * @brief Internal implementation of QueueManager using Metal-cpp.
 * 
 * QueueManager manages multiple command queues for different workload types:
 * - Primary queue: General compute work
 * - Decode queue: Low-latency inference work (can overlap with primary)
 * - Transfer queue: Async data transfers (when available)
 */
class QueueManager::Impl {
public:
    explicit Impl(MTL::Device* device) 
        : device_(device)
        , primary_queue_(nullptr)
        , decode_queue_(nullptr)
        , transfer_queue_(nullptr)
    {
        if (!device_) {
            throw std::runtime_error("QueueManager: received null device pointer");
        }
        
        // Retain the device reference
        device_->retain();
        
        // Create primary command queue
        primary_queue_ = device_->newCommandQueue();
        if (!primary_queue_) {
            throw std::runtime_error("QueueManager: failed to create primary command queue");
        }
        
        // Create decode queue for overlapping work
        decode_queue_ = device_->newCommandQueue();
        if (!decode_queue_) {
            throw std::runtime_error("QueueManager: failed to create decode command queue");
        }
        
        // Create transfer queue for async data movement
        transfer_queue_ = device_->newCommandQueue();
        if (!transfer_queue_) {
            throw std::runtime_error("QueueManager: failed to create transfer command queue");
        }
    }
    
    ~Impl() {
        // Release all queues first
        if (transfer_queue_) {
            transfer_queue_->release();
            transfer_queue_ = nullptr;
        }
        if (decode_queue_) {
            decode_queue_->release();
            decode_queue_ = nullptr;
        }
        if (primary_queue_) {
            primary_queue_->release();
            primary_queue_ = nullptr;
        }
        // Release device reference
        if (device_) {
            device_->release();
            device_ = nullptr;
        }
    }
    
    // Non-copyable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    
    MTL::CommandQueue* primary_queue() const noexcept {
        return primary_queue_;
    }
    
    MTL::CommandQueue* decode_queue() const noexcept {
        return decode_queue_;
    }
    
    MTL::CommandQueue* transfer_queue() const noexcept {
        return transfer_queue_;
    }
    
    MTL::CommandQueue* get_queue(QueueType type) const {
        switch (type) {
            case QueueType::Primary:
                return primary_queue_;
            case QueueType::Decode:
                return decode_queue_;
            case QueueType::Transfer:
                return transfer_queue_;
            default:
                return nullptr;
        }
    }
    
    void commit_all() {
        // Note: Command buffers must be committed, not queues
        // This is a helper to signal that all pending work should be submitted
        // Actual commit happens on command buffers created from these queues
    }
    
    void wait_for_all() {
        // Create temporary command buffers to insert completion markers
        if (primary_queue_) {
            MTL::CommandBuffer* buffer = primary_queue_->commandBuffer();
            if (buffer) {
                buffer->commit();
                buffer->waitUntilCompleted();
                buffer->release();
            }
        }
        if (decode_queue_) {
            MTL::CommandBuffer* buffer = decode_queue_->commandBuffer();
            if (buffer) {
                buffer->commit();
                buffer->waitUntilCompleted();
                buffer->release();
            }
        }
        if (transfer_queue_) {
            MTL::CommandBuffer* buffer = transfer_queue_->commandBuffer();
            if (buffer) {
                buffer->commit();
                buffer->waitUntilCompleted();
                buffer->release();
            }
        }
    }
    
private:
    MTL::Device* device_;
    MTL::CommandQueue* primary_queue_;
    MTL::CommandQueue* decode_queue_;
    MTL::CommandQueue* transfer_queue_;
};

// QueueManager public interface

QueueManager::QueueManager(MTL::Device* device)
    : impl_(std::make_unique<Impl>(device))
{
}

QueueManager::~QueueManager() = default;

// Movable
QueueManager::QueueManager(QueueManager&& other) noexcept = default;
QueueManager& QueueManager::operator=(QueueManager&& other) noexcept = default;

MTL::CommandQueue* QueueManager::primary_queue() const noexcept {
    return impl_->primary_queue();
}

MTL::CommandQueue* QueueManager::decode_queue() const noexcept {
    return impl_->decode_queue();
}

MTL::CommandQueue* QueueManager::transfer_queue() const noexcept {
    return impl_->transfer_queue();
}

MTL::CommandQueue* QueueManager::get_queue(QueueType type) const {
    return impl_->get_queue(type);
}

void QueueManager::commit_all() {
    impl_->commit_all();
}

void QueueManager::wait_for_all() {
    impl_->wait_for_all();
}

std::unique_ptr<QueueManager> QueueManager::create(MTL::Device* device) {
    if (!device) {
        return nullptr;
    }
    try {
        return std::make_unique<QueueManager>(device);
    } catch (const std::exception&) {
        return nullptr;
    }
}

std::unique_ptr<QueueManager> QueueManager::create_default() {
    auto device = MetalDevice::default_device();
    if (!device) {
        return nullptr;
    }
    return create(device->raw());
}

} // namespace metal_marlin
