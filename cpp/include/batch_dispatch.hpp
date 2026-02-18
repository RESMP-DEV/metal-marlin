#pragma once

#include "python_types.hpp"
#include <nanobind/nanobind.h>
#include <vector>
#include <tuple>

namespace metal_marlin {

namespace nb = nanobind;

class BatchDispatchCPP {
public:
  explicit BatchDispatchCPP(MetalContext &ctx);

  void add_kernel(nb::capsule pipeline_capsule,
                  std::tuple<uint32_t, uint32_t, uint32_t> grid,
                  std::tuple<uint32_t, uint32_t, uint32_t> threadgroup,
                  const std::vector<ManagedBuffer *> &buffers);

  void add_mmfp4_gemm(nb::capsule pipeline_capsule, ManagedBuffer *A,
                      ManagedBuffer *B, ManagedBuffer *S, ManagedBuffer *C,
                      uint32_t M, uint32_t N, uint32_t K, uint32_t group_size);

  void add_int4_gemm(nb::capsule pipeline_capsule, ManagedBuffer *A,
                     ManagedBuffer *B, ManagedBuffer *S, ManagedBuffer *Z,
                     ManagedBuffer *C, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t group_size);

  void commit(bool wait = true);

private:
  MetalContext &ctx_;
  id<MTLCommandBuffer> cmd_;
  id<MTLComputeCommandEncoder> encoder_;
};

} // namespace metal_marlin