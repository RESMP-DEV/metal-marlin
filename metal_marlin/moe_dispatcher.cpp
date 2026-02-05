/**
 * Pybind11 bindings for MoEDispatcher.
 *
 * This file provides Python bindings for the MoEDispatcher C++ class.
 * The actual implementation is in cpp/moe_dispatcher.mm (Objective-C++).
 */

#include "cpp/moe_dispatcher.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

// Wrapper to bridge Python objects to Metal-cpp
// Note: This requires the caller to pass Metal device/library pointers

PYBIND11_MODULE(_moe_dispatcher, m) {
    m.doc() = "MoE Dispatcher bindings for Metal Marlin - High-performance MoE layer dispatch";

    py::class_<metal_marlin::MoEDispatcher>(m, "MoEDispatcher")
        .def(py::init<MTL::Device*, MTL::Library*>(),
             py::arg("device"),
             py::arg("library"),
             "Create MoE dispatcher with Metal device and library")
        .def("dispatch", &metal_marlin::MoEDispatcher::dispatch,
             py::arg("activations"),
             py::arg("expert_ids"),
             py::arg("expert_probs"),
             py::arg("expert_weights"),
             py::arg("hidden_dim"),
             py::arg("intermediate_dim"),
             py::arg("num_experts"),
             py::arg("top_k"),
             R"doc(
Dispatch a complete MoE layer computation.

Args:
    activations: Input tensor [batch, hidden_dim] on MPS device
    expert_ids: Expert assignments [batch, top_k] (int32)
    expert_probs: Routing probabilities [batch, top_k] (float)
    expert_weights: List of expert weight tensors
    hidden_dim: Hidden dimension size
    intermediate_dim: FFN intermediate dimension
    num_experts: Total number of experts
    top_k: Number of experts per token

Returns:
    Output tensor [batch, hidden_dim]

This method implements the full MoE forward pass with the following optimizations:
- Single command buffer for all expert dispatches
- Buffer reuse via internal pool
- Indirect dispatch for variable token counts
)doc")
        .def("prepare_dispatch", &metal_marlin::MoEDispatcher::prepare_dispatch,
             py::arg("batch_size"),
             py::arg("hidden_dim"),
             py::arg("num_experts"),
             py::arg("top_k"),
             R"doc(
Pre-encode a command buffer for repeated dispatch.

This optimizes for repeated calls with the same dimensions by pre-encoding
the command buffer structure. Use execute_prepared() to run.
)doc")
        .def("execute_prepared", &metal_marlin::MoEDispatcher::execute_prepared,
             "Execute a pre-encoded dispatch");

    py::class_<metal_marlin::BufferPool>(m, "BufferPool")
        .def(py::init<MTL::Device*>(), py::arg("device"))
        .def("acquire", &metal_marlin::BufferPool::acquire,
             py::arg("size"),
             py::arg("options") = MTL::ResourceStorageModeShared,
             "Acquire a buffer from the pool")
        .def("release", &metal_marlin::BufferPool::release,
             py::arg("buffer"),
             "Return a buffer to the pool")
        .def("clear", &metal_marlin::BufferPool::clear,
             "Clear all pooled buffers and release memory");
}
