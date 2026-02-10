/*
 * No-op kernel for dispatch overhead measurement.
 * This kernel returns immediately to measure pure dispatch latency.
 */

#include <metal_stdlib>
using namespace metal;

// Empty kernel - returns immediately
kernel void noop_kernel_empty()
{
    // No operation - measures pure dispatch overhead
}

// Kernel with single buffer argument (still does nothing)
kernel void noop_kernel_with_buffer(device float* buffer [[buffer(0)]])
{
    (void)buffer;
    // No operation - measures dispatch overhead with buffer binding
}
