#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "python_types.hpp"

namespace metal_marlin {

void mla_proj_fp4(
    MetalContext& ctx,
    nb::bytes A,
    nb::bytes B_packed,
    nb::bytes scales,
    nb::bytes C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t group_size,
    bool wait = true
);

void mla_decode_proj_fp4(
    MetalContext& ctx,
    nb::bytes x,
    nb::bytes W_packed,
    nb::bytes scales,
    nb::bytes out,
    uint32_t K,
    uint32_t N,
    uint32_t group_size,
    bool wait = true
);

void mla_fused_kv_proj_fp4(
    MetalContext& ctx,
    nb::bytes hidden,
    nb::bytes W_a_packed,
    nb::bytes scales_a,
    nb::bytes W_b_packed,
    nb::bytes scales_b,
    nb::bytes out,
    uint32_t M,
    uint32_t K_hidden,
    uint32_t K_latent,
    uint32_t N_out,
    uint32_t group_size_a,
    uint32_t group_size_b,
    bool wait = true
);

} // namespace metal_marlin
