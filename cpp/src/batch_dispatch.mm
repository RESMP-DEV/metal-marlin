#include "batch_dispatch.hpp"
#include <stdexcept>

namespace metal_marlin {

BatchDispatchCPP::BatchDispatchCPP(MetalContext &ctx) : ctx_(ctx) {
    cmd_ = (id<MTLCommandBuffer>)[(id)ctx_.device().primary_queue() commandBuffer];
    encoder_ = [cmd_ computeCommandEncoder];
}

void BatchDispatchCPP::add_kernel(nb::capsule pipeline_capsule,
                  std::tuple<uint32_t, uint32_t, uint32_t> grid,
                  std::tuple<uint32_t, uint32_t, uint32_t> threadgroup,
                  const std::vector<ManagedBuffer *> &buffers) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_capsule.data();
    [encoder_ setComputePipelineState:pipeline];

    for (size_t i = 0; i < buffers.size(); ++i) {
        // Use get_buffer() which handles both BufferPtr and MPSBuffer
        [encoder_ setBuffer:buffers[i]->get_buffer() offset:0 atIndex:i];
    }

    MTLSize grid_size =
        MTLSizeMake(std::get<0>(grid), std::get<1>(grid), std::get<2>(grid));
    MTLSize tg_size =
        MTLSizeMake(std::get<0>(threadgroup), std::get<1>(threadgroup),
                    std::get<2>(threadgroup));
    [encoder_ dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void BatchDispatchCPP::add_mmfp4_gemm(nb::capsule pipeline_capsule, ManagedBuffer *A,
                      ManagedBuffer *B, ManagedBuffer *S, ManagedBuffer *C,
                      uint32_t M, uint32_t N, uint32_t K, uint32_t group_size) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_capsule.data();
    [encoder_ setComputePipelineState:pipeline];
    
    [encoder_ setBuffer:A->get_buffer() offset:0 atIndex:0];
    [encoder_ setBuffer:B->get_buffer() offset:0 atIndex:1];
    [encoder_ setBuffer:S->get_buffer() offset:0 atIndex:2];
    [encoder_ setBuffer:C->get_buffer() offset:0 atIndex:3];

    [encoder_ setBytes:&M length:sizeof(uint32_t) atIndex:4];
    [encoder_ setBytes:&N length:sizeof(uint32_t) atIndex:5];
    [encoder_ setBytes:&K length:sizeof(uint32_t) atIndex:6];
    [encoder_ setBytes:&group_size length:sizeof(uint32_t) atIndex:7];

    MTLSize grid_size = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    MTLSize tg_size = MTLSizeMake(128, 1, 1);
    [encoder_ dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void BatchDispatchCPP::add_int4_gemm(nb::capsule pipeline_capsule, ManagedBuffer *A,
                     ManagedBuffer *B, ManagedBuffer *S, ManagedBuffer *Z,
                     ManagedBuffer *C, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t group_size) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_capsule.data();
    [encoder_ setComputePipelineState:pipeline];
    
    [encoder_ setBuffer:A->get_buffer() offset:0 atIndex:0];
    [encoder_ setBuffer:B->get_buffer() offset:0 atIndex:1];
    [encoder_ setBuffer:S->get_buffer() offset:0 atIndex:2];
    [encoder_ setBuffer:Z->get_buffer() offset:0 atIndex:3];
    [encoder_ setBuffer:C->get_buffer() offset:0 atIndex:4];

    [encoder_ setBytes:&M length:sizeof(uint32_t) atIndex:5];
    [encoder_ setBytes:&N length:sizeof(uint32_t) atIndex:6];
    [encoder_ setBytes:&K length:sizeof(uint32_t) atIndex:7];
    [encoder_ setBytes:&group_size length:sizeof(uint32_t) atIndex:8];

    MTLSize grid_size = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    MTLSize tg_size = MTLSizeMake(128, 1, 1);
    [encoder_ dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void BatchDispatchCPP::commit(bool wait) {
    [encoder_ endEncoding];
    [cmd_ commit];
    if (wait) {
      [cmd_ waitUntilCompleted];
    }
}

} // namespace metal_marlin