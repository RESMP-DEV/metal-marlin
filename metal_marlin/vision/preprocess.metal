#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------------
// Metal-accelerated vision preprocessing kernels.
// - Resize (bilinear/bicubic) via texture sampling.
// - Normalize with per-channel mean/std.
// - Patch extraction for ViT-style encoders.
// - Dynamic resolution resize (Qwen2-VL style).
// - Supports RGB/BGR/Grayscale input formats and batched images.
// ----------------------------------------------------------------------------

enum Layout : uint {
    kNCHW = 0,
    kNHWC = 1
};

enum ChannelOrder : uint {
    kRGB = 0,
    kBGR = 1,
    kGRAY = 2
};

struct ResizeParams {
    uint batch_size;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint channels;
    uint layout;
    uint channel_order;
};

struct PatchParams {
    uint batch_size;
    uint height;
    uint width;
    uint channels;
    uint patch_size;
};

inline float3 canonicalize(float4 sample, uint channel_order) {
    if (channel_order == kBGR) {
        return float3(sample.b, sample.g, sample.r);
    }
    if (channel_order == kGRAY) {
        return float3(sample.r, sample.r, sample.r);
    }
    return float3(sample.r, sample.g, sample.b);
}

// Catmull-Rom cubic weight.
inline float cubic_weight(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    if (t < 1.0f) {
        return 1.5f * t3 - 2.5f * t2 + 1.0f;
    }
    return -0.5f * t3 + 2.5f * t2 - 4.0f * t + 2.0f;
}

inline float4 sample_bicubic_texture(
    texture2d_array<float, access::sample> tex,
    sampler pixel_sampler,
    float2 px_coord,
    uint slice
) {
    float x = px_coord.x;
    float y = px_coord.y;
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    float fx = x - float(x0);
    float fy = y - float(y0);

    float4 accum = float4(0.0f);
    for (int j = -1; j <= 2; ++j) {
        float wy = cubic_weight(abs(float(j) - fy));
        for (int i = -1; i <= 2; ++i) {
            float wx = cubic_weight(abs(float(i) - fx));
            float2 coord = float2(float(x0 + i), float(y0 + j));
            float4 val = tex.sample(pixel_sampler, coord, slice);
            accum += val * (wx * wy);
        }
    }
    return accum;
}

// ----------------------------------------------------------------------------
// Texture-based resize (bilinear) - device function implementation
// ----------------------------------------------------------------------------
inline void image_resize_bilinear_texture_impl(
    texture2d_array<float, access::sample> input,
    device float* output,
    constant ResizeParams& params,
    uint3 gid
) {
    if (gid.z >= params.batch_size || gid.y >= params.out_h || gid.x >= params.out_w) {
        return;
    }

    sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid.x + 0.5f, gid.y + 0.5f) /
                 float2(params.out_w, params.out_h));

    float4 sample = input.sample(linear_sampler, uv, gid.z);
    float3 rgb = canonicalize(sample, params.channel_order);

    uint out_base = gid.z * params.out_h * params.out_w * params.channels;
    if (params.layout == kNHWC) {
        uint idx = out_base + gid.y * params.out_w * params.channels + gid.x * params.channels;
        if (params.channels == 1) {
            output[idx] = rgb.r;
        } else {
            output[idx + 0] = rgb.r;
            output[idx + 1] = rgb.g;
            output[idx + 2] = rgb.b;
        }
    } else {
        uint plane = params.out_h * params.out_w;
        if (params.channels == 1) {
            output[out_base + gid.y * params.out_w + gid.x] = rgb.r;
        } else {
            output[out_base + 0 * plane + gid.y * params.out_w + gid.x] = rgb.r;
            output[out_base + 1 * plane + gid.y * params.out_w + gid.x] = rgb.g;
            output[out_base + 2 * plane + gid.y * params.out_w + gid.x] = rgb.b;
        }
    }
}

// ----------------------------------------------------------------------------
// Texture-based resize (bilinear) - kernel wrapper
// ----------------------------------------------------------------------------
kernel void image_resize_bilinear_texture(
    texture2d_array<float, access::sample> input [[texture(0)]],
    device float* output                         [[buffer(0)]],
    constant ResizeParams& params                [[buffer(1)]],
    uint3 gid                                    [[thread_position_in_grid]]
) {
    image_resize_bilinear_texture_impl(input, output, params, gid);
}

// ----------------------------------------------------------------------------
// Texture-based resize (bicubic)
// ----------------------------------------------------------------------------
kernel void image_resize_bicubic_texture(
    texture2d_array<float, access::sample> input [[texture(0)]],
    device float* output                         [[buffer(0)]],
    constant ResizeParams& params                [[buffer(1)]],
    uint3 gid                                    [[thread_position_in_grid]]
) {
    if (gid.z >= params.batch_size || gid.y >= params.out_h || gid.x >= params.out_w) {
        return;
    }

    sampler pixel_sampler(coord::pixel, address::clamp_to_edge, filter::nearest);
    float2 px = (float2(gid.x + 0.5f, gid.y + 0.5f) *
                 float2(params.in_w, params.in_h) /
                 float2(params.out_w, params.out_h));

    float4 sample = sample_bicubic_texture(input, pixel_sampler, px, gid.z);
    float3 rgb = canonicalize(sample, params.channel_order);

    uint out_base = gid.z * params.out_h * params.out_w * params.channels;
    if (params.layout == kNHWC) {
        uint idx = out_base + gid.y * params.out_w * params.channels + gid.x * params.channels;
        if (params.channels == 1) {
            output[idx] = rgb.r;
        } else {
            output[idx + 0] = rgb.r;
            output[idx + 1] = rgb.g;
            output[idx + 2] = rgb.b;
        }
    } else {
        uint plane = params.out_h * params.out_w;
        if (params.channels == 1) {
            output[out_base + gid.y * params.out_w + gid.x] = rgb.r;
        } else {
            output[out_base + 0 * plane + gid.y * params.out_w + gid.x] = rgb.r;
            output[out_base + 1 * plane + gid.y * params.out_w + gid.x] = rgb.g;
            output[out_base + 2 * plane + gid.y * params.out_w + gid.x] = rgb.b;
        }
    }
}

// ----------------------------------------------------------------------------
// Fused resize + normalize (texture input)
// ----------------------------------------------------------------------------
kernel void image_resize_normalize_texture(
    texture2d_array<float, access::sample> input [[texture(0)]],
    device float* output                         [[buffer(0)]],
    constant float* mean                         [[buffer(1)]],
    constant float* std_inv                      [[buffer(2)]],
    constant ResizeParams& params                [[buffer(3)]],
    uint3 gid                                    [[thread_position_in_grid]]
) {
    if (gid.z >= params.batch_size || gid.y >= params.out_h || gid.x >= params.out_w) {
        return;
    }

    sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid.x + 0.5f, gid.y + 0.5f) /
                 float2(params.out_w, params.out_h));

    float4 sample = input.sample(linear_sampler, uv, gid.z);
    float3 rgb = canonicalize(sample, params.channel_order);

    uint out_base = gid.z * params.out_h * params.out_w * params.channels;
    if (params.layout == kNHWC) {
        uint idx = out_base + gid.y * params.out_w * params.channels + gid.x * params.channels;
        if (params.channels == 1) {
            output[idx] = (rgb.r - mean[0]) * std_inv[0];
        } else {
            output[idx + 0] = (rgb.r - mean[0]) * std_inv[0];
            output[idx + 1] = (rgb.g - mean[1]) * std_inv[1];
            output[idx + 2] = (rgb.b - mean[2]) * std_inv[2];
        }
    } else {
        uint plane = params.out_h * params.out_w;
        if (params.channels == 1) {
            output[out_base + gid.y * params.out_w + gid.x] = (rgb.r - mean[0]) * std_inv[0];
        } else {
            output[out_base + 0 * plane + gid.y * params.out_w + gid.x] = (rgb.r - mean[0]) * std_inv[0];
            output[out_base + 1 * plane + gid.y * params.out_w + gid.x] = (rgb.g - mean[1]) * std_inv[1];
            output[out_base + 2 * plane + gid.y * params.out_w + gid.x] = (rgb.b - mean[2]) * std_inv[2];
        }
    }
}

// ----------------------------------------------------------------------------
// Dynamic resize (Qwen2-VL style; sizes computed host-side)
// ----------------------------------------------------------------------------
kernel void dynamic_resize_qwen2vl_texture(
    texture2d_array<float, access::sample> input [[texture(0)]],
    device float* output                         [[buffer(0)]],
    constant ResizeParams& params                [[buffer(1)]],
    uint3 gid                                    [[thread_position_in_grid]]
) {
    image_resize_bilinear_texture_impl(input, output, params, gid);
}

// ----------------------------------------------------------------------------
// ViT patch extraction (NHWC only) - struct-based params variant
// ----------------------------------------------------------------------------
kernel void vit_patch_extract_struct(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant PatchParams& params[[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint patches_h = params.height / params.patch_size;
    uint patches_w = params.width / params.patch_size;
    uint patch_dim = params.patch_size * params.patch_size * params.channels;

    uint n = gid.z;
    uint patch_y = gid.y;
    uint patch_x = gid.x;
    if (n >= params.batch_size || patch_y >= patches_h || patch_x >= patches_w) {
        return;
    }

    uint patch_idx = patch_y * patches_w + patch_x;
    uint y_start = patch_y * params.patch_size;
    uint x_start = patch_x * params.patch_size;

    uint input_offset = n * params.height * params.width * params.channels;
    uint output_offset = n * (patches_h * patches_w) * patch_dim + patch_idx * patch_dim;

    uint out_idx = 0;
    for (uint py = 0; py < params.patch_size; ++py) {
        for (uint px = 0; px < params.patch_size; ++px) {
            uint y = y_start + py;
            uint x = x_start + px;
            uint in_base = input_offset + y * params.width * params.channels + x * params.channels;
            for (uint c = 0; c < params.channels; ++c) {
                output[output_offset + out_idx++] = input[in_base + c];
            }
        }
    }
}
