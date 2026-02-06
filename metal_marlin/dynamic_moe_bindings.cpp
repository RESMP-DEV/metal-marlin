// dynamic_moe_bindings.cpp - Python bindings for DynamicMoEDispatcher
#include "cpp/dynamic_moe_dispatch.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

// For MPS tensor buffer access
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace py = pybind11;
using namespace metal_marlin;

// Helper to get MTLBuffer from PyTorch MPS tensor
// This is the standard PyTorch approach for accessing the underlying Metal buffer
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_mps(), "Tensor must be on MPS device");
    // The storage().data() contains the id<MTLBuffer> for MPS tensors
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Helper to get MTL::Buffer* from PyTorch MPS tensor (C++ wrapper version)
inline MTL::Buffer* tensor_to_buffer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_mps(), "Tensor must be on MPS device, got device: ", tensor.device().str());
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    // Synchronize to ensure any pending operations on the tensor are complete
    at::mps::synchronize();

    // Get the Objective-C MTLBuffer
    id<MTLBuffer> mtl_buffer = getMTLBufferStorage(tensor);

    // Cast to C++ Metal-cpp type
    // Note: MTL::Buffer is toll-free bridged with id<MTLBuffer>
    return reinterpret_cast<MTL::Buffer*>(mtl_buffer);
}

PYBIND11_MODULE(_dynamic_moe, m) {
    m.doc() = "Dynamic mixed-precision MoE dispatch for Metal";

    py::class_<ExpertBitConfig>(m, "ExpertBitConfig")
        .def(py::init<>())
        .def_readwrite("gate_bits", &ExpertBitConfig::gate_bits)
        .def_readwrite("up_bits", &ExpertBitConfig::up_bits)
        .def_readwrite("down_bits", &ExpertBitConfig::down_bits);

    py::class_<DynamicMoEParams>(m, "DynamicMoEParams")
        .def(py::init<>())
        .def_readwrite("batch_size", &DynamicMoEParams::batch_size)
        .def_readwrite("hidden_dim", &DynamicMoEParams::hidden_dim)
        .def_readwrite("intermediate_dim", &DynamicMoEParams::intermediate_dim)
        .def_readwrite("num_experts", &DynamicMoEParams::num_experts)
        .def_readwrite("top_k", &DynamicMoEParams::top_k)
        .def_readwrite("tile_size", &DynamicMoEParams::tile_size);

    py::class_<DynamicMoEDispatcher>(m, "DynamicMoEDispatcher")
        .def(py::init([](py::object device, py::object library) {
            // Extract Metal pointers from Python objects
            // This requires the Metal device/library to be passed from metal_lib
            auto* dev = reinterpret_cast<MTL::Device*>(
                py::cast<uintptr_t>(device.attr("_ptr")));
            auto* lib = reinterpret_cast<MTL::Library*>(
                py::cast<uintptr_t>(library.attr("_ptr")));
            return new DynamicMoEDispatcher(dev, lib);
        }))
        .def("set_expert_bits", &DynamicMoEDispatcher::set_expert_bits)
        .def("dispatch", [](DynamicMoEDispatcher& self,
                            torch::Tensor activations,
                            torch::Tensor expert_ids,
                            torch::Tensor expert_probs,
                            torch::Tensor gate_weights,
                            torch::Tensor gate_scales,
                            torch::Tensor up_weights,
                            torch::Tensor up_scales,
                            torch::Tensor down_weights,
                            torch::Tensor down_scales,
                            torch::Tensor gate_su,
                            torch::Tensor gate_sv,
                            torch::Tensor up_su,
                            torch::Tensor up_sv,
                            torch::Tensor down_su,
                            torch::Tensor down_sv,
                            DynamicMoEParams params) {
            auto* output = self.dispatch(
                tensor_to_buffer(activations),
                tensor_to_buffer(expert_ids),
                tensor_to_buffer(expert_probs),
                tensor_to_buffer(gate_weights),
                tensor_to_buffer(gate_scales),
                tensor_to_buffer(up_weights),
                tensor_to_buffer(up_scales),
                tensor_to_buffer(down_weights),
                tensor_to_buffer(down_scales),
                tensor_to_buffer(gate_su),
                tensor_to_buffer(gate_sv),
                tensor_to_buffer(up_su),
                tensor_to_buffer(up_sv),
                tensor_to_buffer(down_su),
                tensor_to_buffer(down_sv),
                params);
            return reinterpret_cast<uintptr_t>(output);
        });
}
