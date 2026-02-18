#include "mla_attention.hpp"
#include <Metal/Metal.hpp>
#include <vector>
#include <tuple>

namespace metal_marlin {

// Constants
constexpr uint32_t TILE_N_MLA = 64;
constexpr uint32_t TILE_M_MLA = 64;
constexpr uint32_t THREADS_PER_TG_MLA = 128;
constexpr uint32_t THREADS_PER_TG_DECODE = 128;

// Local dispatch helper
static void dispatch_kernel(
    MetalContext& ctx,
    const std::string& kernel_name,
    std::tuple<uint32_t, uint32_t, uint32_t> grid,
    std::tuple<uint32_t, uint32_t, uint32_t> threadgroup,
    const std::vector<id<MTLBuffer>>& buffers,
    bool wait
) {
    auto pipeline_obj = ctx.get_pipeline(kernel_name);
    nb::capsule pipeline_capsule = nb::cast<nb::capsule>(pipeline_obj);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_capsule.data();
    
    id<MTLCommandBuffer> cmd = [ctx.device().primary_queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    for (size_t i = 0; i < buffers.size(); ++i) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    
    MTLSize grid_size = MTLSizeMake(std::get<0>(grid), std::get<1>(grid), std::get<2>(grid));
    MTLSize tg_size = MTLSizeMake(std::get<0>(threadgroup), std::get<1>(threadgroup), std::get<2>(threadgroup));
    
    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
    [encoder endEncoding];
    [cmd commit];
    
    if (wait) {
        [cmd waitUntilCompleted];
    }
}

// Helper to create a Metal buffer from raw bytes.
static id<MTLBuffer> newBufferWithBytes(id<MTLDevice> device, const void* bytes, NSUInteger length) {
    return [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

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
    bool wait)
{
    id<MTLDevice> device = ctx.device().device();

    id<MTLBuffer> A_buf = newBufferWithBytes(device, A.data(), A.size());
    id<MTLBuffer> B_buf = newBufferWithBytes(device, B_packed.data(), B_packed.size());
    id<MTLBuffer> S_buf = newBufferWithBytes(device, scales.data(), scales.size());
    id<MTLBuffer> C_buf = newBufferWithBytes(device, C.data(), C.size());

    id<MTLBuffer> M_buf = newBufferWithBytes(device, &M, sizeof(M));
    id<MTLBuffer> N_buf = newBufferWithBytes(device, &N, sizeof(N));
    id<MTLBuffer> K_buf = newBufferWithBytes(device, &K, sizeof(K));
    id<MTLBuffer> gs_buf = newBufferWithBytes(device, &group_size, sizeof(group_size));

    const char* kernel_name = (K <= 1024) ? "mla_proj_fp4_k16" : "mla_proj_fp4_k32";

    uint32_t grid_x = (N + TILE_N_MLA - 1) / TILE_N_MLA;
    uint32_t grid_y = (M + TILE_M_MLA - 1) / TILE_M_MLA;

    std::vector<id<MTLBuffer>> buffers = {A_buf, B_buf, S_buf, C_buf, M_buf, N_buf, K_buf, gs_buf};
    
    dispatch_kernel(
        ctx,
        kernel_name,
        {grid_x, grid_y, 1},
        {THREADS_PER_TG_MLA, 1, 1},
        buffers,
        wait
    );
}

void mla_decode_proj_fp4(
    MetalContext& ctx,
    nb::bytes x,
    nb::bytes W_packed,
    nb::bytes scales,
    nb::bytes out,
    uint32_t K,
    uint32_t N,
    uint32_t group_size,
    bool wait)
{
    id<MTLDevice> device = ctx.device().device();

    id<MTLBuffer> x_buf = newBufferWithBytes(device, x.data(), x.size());
    id<MTLBuffer> W_buf = newBufferWithBytes(device, W_packed.data(), W_packed.size());
    id<MTLBuffer> S_buf = newBufferWithBytes(device, scales.data(), scales.size());
    id<MTLBuffer> out_buf = newBufferWithBytes(device, out.data(), out.size());

    id<MTLBuffer> K_buf = newBufferWithBytes(device, &K, sizeof(K));
    id<MTLBuffer> N_buf = newBufferWithBytes(device, &N, sizeof(N));
    id<MTLBuffer> gs_buf = newBufferWithBytes(device, &group_size, sizeof(group_size));

    uint32_t num_tgs = (N + THREADS_PER_TG_DECODE - 1) / THREADS_PER_TG_DECODE;

    std::vector<id<MTLBuffer>> buffers = {x_buf, W_buf, S_buf, out_buf, K_buf, N_buf, gs_buf};

    dispatch_kernel(
        ctx,
        "mla_decode_proj_fp4",
        {num_tgs, 1, 1},
        {THREADS_PER_TG_DECODE, 1, 1},
        buffers,
        wait
    );
}

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
    bool wait)
{
    id<MTLDevice> device = ctx.device().device();

    id<MTLBuffer> hidden_buf = newBufferWithBytes(device, hidden.data(), hidden.size());
    id<MTLBuffer> Wa_buf = newBufferWithBytes(device, W_a_packed.data(), W_a_packed.size());
    id<MTLBuffer> Sa_buf = newBufferWithBytes(device, scales_a.data(), scales_a.size());
    id<MTLBuffer> Wb_buf = newBufferWithBytes(device, W_b_packed.data(), W_b_packed.size());
    id<MTLBuffer> Sb_buf = newBufferWithBytes(device, scales_b.data(), scales_b.size());
    id<MTLBuffer> out_buf = newBufferWithBytes(device, out.data(), out.size());

    id<MTLBuffer> M_buf = newBufferWithBytes(device, &M, sizeof(M));
    id<MTLBuffer> Kh_buf = newBufferWithBytes(device, &K_hidden, sizeof(K_hidden));
    id<MTLBuffer> Kl_buf = newBufferWithBytes(device, &K_latent, sizeof(K_latent));
    id<MTLBuffer> Nout_buf = newBufferWithBytes(device, &N_out, sizeof(N_out));
    id<MTLBuffer> gsa_buf = newBufferWithBytes(device, &group_size_a, sizeof(group_size_a));
    id<MTLBuffer> gsb_buf = newBufferWithBytes(device, &group_size_b, sizeof(group_size_b));

    uint32_t grid_x = (N_out + TILE_N_MLA - 1) / TILE_N_MLA;
    uint32_t grid_y = (M + TILE_M_MLA - 1) / TILE_M_MLA;

    std::vector<id<MTLBuffer>> buffers = {
        hidden_buf, Wa_buf, Sa_buf, Wb_buf, Sb_buf, out_buf,
        M_buf, Kh_buf, Kl_buf, Nout_buf, gsa_buf, gsb_buf
    };

    dispatch_kernel(
        ctx,
        "mla_fused_kv_proj_fp4",
        {grid_x, grid_y, 1},
        {THREADS_PER_TG_MLA, 1, 1},
        buffers,
        wait
    );
}

} // namespace metal_marlin
