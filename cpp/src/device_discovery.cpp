#include "metal_device.hpp"
#include <iostream>
#include <vector>
#include <string>

namespace metal_marlin {

struct DeviceInfo {
    size_t index;
    std::string name;
    uint64_t registry_id;
    bool is_low_power;
    bool is_removable;
    size_t max_buffer_length;
    size_t recommended_max_working_set_size;
};

std::vector<DeviceInfo> discover_all_devices() {
    std::vector<DeviceInfo> infos;
    
    NS::Array* devices = MTL::CopyAllDevices();
    if (!devices) {
        return infos;
    }

    infos.reserve(devices->count());
    
    for (NS::UInteger i = 0; i < devices->count(); ++i) {
        MTL::Device* device = static_cast<MTL::Device*>(devices->object(i));
        
        DeviceInfo info;
        info.index = static_cast<size_t>(i);
        info.registry_id = device->registryID();
        
        const char* device_name = device->name()->utf8String();
        info.name = device_name ? std::string(device_name) : "Unknown";
        
        info.is_low_power = device->lowPower();
        info.is_removable = device->removable();
        info.max_buffer_length = device->maxBufferLength();
        info.recommended_max_working_set_size = device->recommendedMaxWorkingSetSize();
        
        infos.push_back(info);
    }
    
    devices->release();
    return infos;
}

void print_device_info(const DeviceInfo& info) {
    std::cout << "Device " << info.index << ": " << info.name << "\n";
    std::cout << "  Registry ID: " << info.registry_id << "\n";
    std::cout << "  Low Power: " << (info.is_low_power ? "Yes" : "No") << "\n";
    std::cout << "  Removable: " << (info.is_removable ? "Yes" : "No") << "\n";
    std::cout << "  Max Buffer: " << (info.max_buffer_length / (1024 * 1024)) << " MB\n";
    std::cout << "  Recommended Working Set: " 
              << (info.recommended_max_working_set_size / (1024 * 1024)) << " MB\n";
}

void print_all_devices() {
    auto devices = discover_all_devices();
    
    if (devices.empty()) {
        std::cout << "No Metal devices found.\n";
        return;
    }
    
    std::cout << "Found " << devices.size() << " Metal device(s):\n\n";
    
    for (const auto& info : devices) {
        print_device_info(info);
        std::cout << "\n";
    }
}

std::vector<size_t> get_high_performance_device_indices() {
    std::vector<size_t> indices;
    
    auto devices = discover_all_devices();
    for (const auto& info : devices) {
        if (!info.is_low_power) {
            indices.push_back(info.index);
        }
    }
    
    return indices;
}

size_t get_default_device_index() {
    auto device = MetalDevice::default_device();
    if (!device) {
        return 0;
    }
    
    uint64_t default_id = device->registry_id();
    auto devices = discover_all_devices();
    
    for (const auto& info : devices) {
        if (info.registry_id == default_id) {
            return info.index;
        }
    }
    
    return 0;
}

std::vector<std::unique_ptr<MetalDevice>> create_all_devices() {
    return MetalDevice::all_devices();
}

std::vector<std::unique_ptr<MetalDevice>> create_high_performance_devices() {
    std::vector<std::unique_ptr<MetalDevice>> devices;
    
    auto indices = get_high_performance_device_indices();
    for (size_t idx : indices) {
        auto device = MetalDevice::device_by_index(idx);
        if (device) {
            devices.push_back(std::move(device));
        }
    }
    
    return devices;
}

} // namespace metal_marlin
