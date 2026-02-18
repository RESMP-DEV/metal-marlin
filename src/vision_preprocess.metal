#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Vision Preprocessing Kernels for Apple Silicon
// ============================================================================
//
// GPU-accelerated image preprocessing for vision models (ViT, CLIP, Qwen2-VL).
// Replaces CPU-bound PIL/torchvision resize/normalize pipeline.
//
// Kernels:
//   1. image_resize_bilinear   - Bilinear interpolation resize
//   2. image_resize_bicubic    - Bicubic interpolation resize (higher quality)
//   3. image_normalize         - Channel-wise mean/std normalization
//   4. dynamic_resize_patches  - Qwen2-VL style dynamic resolution
//
// Memory layout:
//   Input:  [N, H, W, C] or [N, C, H, W] (configurable)
//   Output: [N, H', W', C] or patches [N, num_patches, patch_size^2 * C]
//
// All kernels process batches in parallel. Each threadgroup handles one
// output pixel/patch to maximize occupancy on Apple Silicon.
//
// ============================================================================

// ============================================================================
// Constants
// ============================================================================

// Maximum patch size for ViT models (224/14 = 16 typical, up to 32 for some)
constant constexpr uint MAX_PATCH_SIZE = 32;

// ============================================================================
// Texture sampling helpers
// ============================================================================

/// Bilinear interpolation for texture coordinate (u, v) in [0, 1] range.
/// Uses hardware texture sampling when available, falls back to manual.
inline float4 sample_bilinear(
    device const float* image,
    float u,
    float v,
    uint width,
    uint height,
    uint channels,
    bool nhwc  // True if NHWC layout, false if NCHW
) {
    // Convert normalized coords to pixel coords
    float px = u * float(width - 1);
    float py = v * float(height - 1);

    // Get integer and fractional parts
    uint x0 = uint(px);
    uint y0 = uint(py);
    uint x1 = min(x0 + 1, width - 1);
    uint y1 = min(y0 + 1, height - 1);

    float fx = px - float(x0);
    float fy = py - float(y0);

    float4 result = float4(0.0f);

    // Sample 4 corners and interpolate
    for (uint c = 0; c < min(channels, 4u); ++c) {
        float v00, v01, v10, v11;

        if (nhwc) {
            // NHWC: image[y * W * C + x * C + c]
            v00 = image[y0 * width * channels + x0 * channels + c];
            v01 = image[y0 * width * channels + x1 * channels + c];
            v10 = image[y1 * width * channels + x0 * channels + c];
            v11 = image[y1 * width * channels + x1 * channels + c];
        } else {
            // NCHW: image[c * H * W + y * W + x]
            v00 = image[c * height * width + y0 * width + x0];
            v01 = image[c * height * width + y0 * width + x1];
            v10 = image[c * height * width + y1 * width + x0];
            v11 = image[c * height * width + y1 * width + x1];
        }

        // Bilinear interpolation
        float top = mix(v00, v01, fx);
        float bot = mix(v10, v11, fx);
        result[c] = mix(top, bot, fy);
    }

    return result;
}

/// Cubic interpolation weight (Catmull-Rom spline)
inline float cubic_weight(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    if (t < 1.0f) {
        return 1.5f * t3 - 2.5f * t2 + 1.0f;
    } else {
        return -0.5f * t3 + 2.5f * t2 - 4.0f * t + 2.0f;
    }
}

/// Sample with bicubic interpolation (4x4 neighborhood)
inline float4 sample_bicubic(
    device const float* image,
    float u,
    float v,
    uint width,
    uint height,
    uint channels,
    bool nhwc
) {
    float px = u * float(width - 1);
    float py = v * float(height - 1);

    int x0 = int(px);
    int y0 = int(py);
    float fx = px - float(x0);
    float fy = py - float(y0);

    float4 result = float4(0.0f);

    for (uint c = 0; c < min(channels, 4u); ++c) {
        float col_sum = 0.0f;

        for (int j = -1; j <= 2; ++j) {
            float row_sum = 0.0f;
            int y = clamp(y0 + j, 0, int(height - 1));

            for (int i = -1; i <= 2; ++i) {
                int x = clamp(x0 + i, 0, int(width - 1));

                float val;
                if (nhwc) {
                    val = image[y * int(width * channels) + x * int(channels) + int(c)];
                } else {
                    val = image[int(c) * int(height * width) + y * int(width) + x];
                }

                float wx = cubic_weight(abs(float(i) - fx));
                row_sum += val * wx;
            }

            float wy = cubic_weight(abs(float(j) - fy));
            col_sum += row_sum * wy;
        }

        result[c] = col_sum;
    }

    return result;
}

// ============================================================================
// Image Resize Kernel - Bilinear
// ============================================================================

/// Resize image using bilinear interpolation.
/// Each thread computes one output pixel.
///
/// @param input       Input image [N, H_in, W_in, C] or [N, C, H_in, W_in]
/// @param output      Output image [N, H_out, W_out, C] or [N, C, H_out, W_out]
/// @param params      [batch_size, H_in, W_in, H_out, W_out, channels, nhwc]
kernel void image_resize_bilinear(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    // Compute source coordinates (center of output pixel -> source coord)
    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    // Offset to input image for this batch element
    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    float4 sampled = sample_bilinear(
        input + input_offset,
        u, v,
        W_in, H_in, channels, nhwc
    );

    // Write output
    for (uint c = 0; c < channels; ++c) {
        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = sampled[c];
    }
}

// ============================================================================
// Image Resize Kernel - Bicubic
// ============================================================================

/// Resize image using bicubic interpolation (higher quality, slower).
/// Each thread computes one output pixel.
kernel void image_resize_bicubic(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    float4 sampled = sample_bicubic(
        input + input_offset,
        u, v,
        W_in, H_in, channels, nhwc
    );

    // Write output
    for (uint c = 0; c < channels; ++c) {
        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = sampled[c];
    }
}

// ============================================================================
// Image Normalize Kernel
// ============================================================================

/// Apply channel-wise normalization: output = (input - mean) / std
/// Supports both NHWC and NCHW layouts.
///
/// @param input       Input image [N, H, W, C] or [N, C, H, W]
/// @param output      Output image (same shape)
/// @param mean        Per-channel mean [C]
/// @param std_inv     Per-channel 1/std [C] (precomputed for efficiency)
/// @param params      [batch_size, height, width, channels, nhwc]
kernel void image_normalize(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant float* mean        [[buffer(2)]],
    constant float* std_inv     [[buffer(3)]],
    constant uint* params       [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    uint channels = params[3];
    bool nhwc = params[4] != 0;

    uint n = gid.z;
    uint y = gid.y;
    uint x = gid.x;

    if (n >= batch_size || y >= height || x >= width) return;

    uint base_offset = n * height * width * channels;

    for (uint c = 0; c < channels; ++c) {
        uint idx;
        if (nhwc) {
            idx = base_offset + y * width * channels + x * channels + c;
        } else {
            idx = base_offset + c * height * width + y * width + x;
        }

        output[idx] = (input[idx] - mean[c]) * std_inv[c];
    }
}

// ============================================================================
// Half-precision variants (FP16)
// ============================================================================

/// Bilinear resize with half precision I/O (reduced memory bandwidth)
kernel void image_resize_bilinear_f16(
    device const half* input    [[buffer(0)]],
    device half* output         [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    // Convert normalized coords to pixel coords
    float px = u * float(W_in - 1);
    float py = v * float(H_in - 1);

    uint x0 = uint(px);
    uint y0 = uint(py);
    uint x1 = min(x0 + 1, W_in - 1);
    uint y1 = min(y0 + 1, H_in - 1);

    float fx = px - float(x0);
    float fy = py - float(y0);

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    for (uint c = 0; c < channels; ++c) {
        float v00, v01, v10, v11;

        if (nhwc) {
            v00 = float(input[input_offset + y0 * W_in * channels + x0 * channels + c]);
            v01 = float(input[input_offset + y0 * W_in * channels + x1 * channels + c]);
            v10 = float(input[input_offset + y1 * W_in * channels + x0 * channels + c]);
            v11 = float(input[input_offset + y1 * W_in * channels + x1 * channels + c]);
        } else {
            v00 = float(input[input_offset + c * H_in * W_in + y0 * W_in + x0]);
            v01 = float(input[input_offset + c * H_in * W_in + y0 * W_in + x1]);
            v10 = float(input[input_offset + c * H_in * W_in + y1 * W_in + x0]);
            v11 = float(input[input_offset + c * H_in * W_in + y1 * W_in + x1]);
        }

        float top = mix(v00, v01, fx);
        float bot = mix(v10, v11, fx);
        float result = mix(top, bot, fy);

        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = half(result);
    }
}

/// Normalize with half precision I/O
kernel void image_normalize_f16(
    device const half* input    [[buffer(0)]],
    device half* output         [[buffer(1)]],
    constant float* mean        [[buffer(2)]],
    constant float* std_inv     [[buffer(3)]],
    constant uint* params       [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    uint channels = params[3];
    bool nhwc = params[4] != 0;

    uint n = gid.z;
    uint y = gid.y;
    uint x = gid.x;

    if (n >= batch_size || y >= height || x >= width) return;

    uint base_offset = n * height * width * channels;

    for (uint c = 0; c < channels; ++c) {
        uint idx;
        if (nhwc) {
            idx = base_offset + y * width * channels + x * channels + c;
        } else {
            idx = base_offset + c * height * width + y * width + x;
        }

        float val = float(input[idx]);
        output[idx] = half((val - mean[c]) * std_inv[c]);
    }
}

// ============================================================================
// Fused Resize + Normalize Kernel
// ============================================================================

/// Fused bilinear resize and normalize in single pass.
/// Eliminates intermediate buffer allocation and memory round-trip.
///
/// @param input       Input image [N, H_in, W_in, C]
/// @param output      Output image [N, H_out, W_out, C]
/// @param mean        Per-channel mean [C]
/// @param std_inv     Per-channel 1/std [C]
/// @param params      [batch_size, H_in, W_in, H_out, W_out, channels, nhwc]
kernel void image_resize_normalize_fused(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant float* mean        [[buffer(2)]],
    constant float* std_inv     [[buffer(3)]],
    constant uint* params       [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    float4 sampled = sample_bilinear(
        input + input_offset,
        u, v,
        W_in, H_in, channels, nhwc
    );

    // Apply normalization and write output
    for (uint c = 0; c < channels; ++c) {
        float normalized = (sampled[c] - mean[c]) * std_inv[c];

        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = normalized;
    }
}

/// Half-precision fused resize + normalize
kernel void image_resize_normalize_fused_f16(
    device const half* input    [[buffer(0)]],
    device half* output         [[buffer(1)]],
    constant float* mean        [[buffer(2)]],
    constant float* std_inv     [[buffer(3)]],
    constant uint* params       [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    // Convert normalized coords to pixel coords
    float px = u * float(W_in - 1);
    float py = v * float(H_in - 1);

    uint x0 = uint(px);
    uint y0 = uint(py);
    uint x1 = min(x0 + 1, W_in - 1);
    uint y1 = min(y0 + 1, H_in - 1);

    float fx = px - float(x0);
    float fy = py - float(y0);

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    for (uint c = 0; c < channels; ++c) {
        float v00, v01, v10, v11;

        if (nhwc) {
            v00 = float(input[input_offset + y0 * W_in * channels + x0 * channels + c]);
            v01 = float(input[input_offset + y0 * W_in * channels + x1 * channels + c]);
            v10 = float(input[input_offset + y1 * W_in * channels + x0 * channels + c]);
            v11 = float(input[input_offset + y1 * W_in * channels + x1 * channels + c]);
        } else {
            v00 = float(input[input_offset + c * H_in * W_in + y0 * W_in + x0]);
            v01 = float(input[input_offset + c * H_in * W_in + y0 * W_in + x1]);
            v10 = float(input[input_offset + c * H_in * W_in + y1 * W_in + x0]);
            v11 = float(input[input_offset + c * H_in * W_in + y1 * W_in + x1]);
        }

        float top = mix(v00, v01, fx);
        float bot = mix(v10, v11, fx);
        float sampled = mix(top, bot, fy);

        // Apply normalization
        float normalized = (sampled - mean[c]) * std_inv[c];

        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = half(normalized);
    }
}

// ============================================================================
// Qwen2-VL Dynamic Resolution Kernel
// ============================================================================

/// Dynamic resolution preprocessing for Qwen2-VL style models.
/// Processes images at native resolution with adaptive patch sizes.
///
/// Qwen2-VL approach:
///   1. Keep aspect ratio, resize to fit max_pixels while being divisible by patch_size
///   2. Extract patches at multiple scales
///   3. Add position embeddings based on actual (h, w) not fixed 224x224
///
/// This kernel handles step 1: resize to nearest valid resolution.
///
/// @param input       Input image [N, H_in, W_in, C]
/// @param output      Output image [N, H_out, W_out, C]
/// @param params      [batch_size, H_in, W_in, H_out, W_out, channels, nhwc]
kernel void dynamic_resize_qwen2vl(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    // Same as bilinear resize - the dynamic part is handled in Python
    // by computing H_out, W_out based on aspect ratio and constraints
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    float4 sampled = sample_bilinear(
        input + input_offset,
        u, v,
        W_in, H_in, channels, nhwc
    );

    for (uint c = 0; c < channels; ++c) {
        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = sampled[c];
    }
}

// ============================================================================
// Batched Processing with Coalesced Memory Access
// ============================================================================

/// Optimized batch resize with coalesced memory access pattern.
/// Processes 4 pixels per thread using vectorized loads/stores.
kernel void image_resize_bilinear_batch_opt(
    device const float4* input  [[buffer(0)]],
    device float4* output       [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    // Assumes channels = 4 for this optimized path (RGBA)
    uint channels = 4;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    float px = u * float(W_in - 1);
    float py = v * float(H_in - 1);

    uint x0 = uint(px);
    uint y0 = uint(py);
    uint x1 = min(x0 + 1, W_in - 1);
    uint y1 = min(y0 + 1, H_in - 1);

    float fx = px - float(x0);
    float fy = py - float(y0);

    uint input_stride = H_in * W_in;
    uint output_stride = H_out * W_out;
    uint input_offset = n * input_stride;
    uint output_offset = n * output_stride;

    // Vectorized load (all 4 channels at once)
    float4 v00 = input[input_offset + y0 * W_in + x0];
    float4 v01 = input[input_offset + y0 * W_in + x1];
    float4 v10 = input[input_offset + y1 * W_in + x0];
    float4 v11 = input[input_offset + y1 * W_in + x1];

    // Bilinear interpolation on all 4 channels
    float4 top = mix(v00, v01, fx);
    float4 bot = mix(v10, v11, fx);
    float4 result = mix(top, bot, fy);

    // Vectorized store
    output[output_offset + y_out * W_out + x_out] = result;
}

// ============================================================================
// RGB <-> BGR Conversion (for OpenCV/PIL interop)
// ============================================================================

/// Convert RGB to BGR or vice versa (channel swap).
/// Operates in-place if input == output.
kernel void channel_swap_rgb_bgr(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    uint channels = params[3];  // Must be 3 for RGB/BGR
    bool nhwc = params[4] != 0;

    uint n = gid.z;
    uint y = gid.y;
    uint x = gid.x;

    if (n >= batch_size || y >= height || x >= width) return;

    uint base = n * height * width * channels;

    float r, g, b;
    if (nhwc) {
        uint idx = base + y * width * channels + x * channels;
        r = input[idx];
        g = input[idx + 1];
        b = input[idx + 2];
        output[idx] = b;
        output[idx + 1] = g;
        output[idx + 2] = r;
    } else {
        uint idx_r = base + 0 * height * width + y * width + x;
        uint idx_g = base + 1 * height * width + y * width + x;
        uint idx_b = base + 2 * height * width + y * width + x;
        r = input[idx_r];
        g = input[idx_g];
        b = input[idx_b];
        output[idx_r] = b;
        output[idx_g] = g;
        output[idx_b] = r;
    }
}

// ============================================================================
// Grayscale Conversion
// ============================================================================

/// Convert RGB to grayscale using ITU-R BT.601 weights.
/// Y = 0.299*R + 0.587*G + 0.114*B
kernel void rgb_to_grayscale(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    bool nhwc = params[3] != 0;

    uint n = gid.z;
    uint y = gid.y;
    uint x = gid.x;

    if (n >= batch_size || y >= height || x >= width) return;

    float r, g, b;
    if (nhwc) {
        uint base = n * height * width * 3 + y * width * 3 + x * 3;
        r = input[base];
        g = input[base + 1];
        b = input[base + 2];
    } else {
        uint base = n * 3 * height * width;
        r = input[base + 0 * height * width + y * width + x];
        g = input[base + 1 * height * width + y * width + x];
        b = input[base + 2 * height * width + y * width + x];
    }

    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    output[n * height * width + y * width + x] = gray;
}

// ============================================================================
// Center Crop Kernel
// ============================================================================

/// Center crop an image to target size.
/// Crops from the center, discarding border pixels.
kernel void center_crop(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    // Compute crop offsets (center the crop region)
    uint y_offset = (H_in - H_out) / 2;
    uint x_offset = (W_in - W_out) / 2;

    uint y_in = y_offset + y_out;
    uint x_in = x_offset + x_out;

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    for (uint c = 0; c < channels; ++c) {
        uint in_idx, out_idx;
        if (nhwc) {
            in_idx = input_offset + y_in * W_in * channels + x_in * channels + c;
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            in_idx = input_offset + c * H_in * W_in + y_in * W_in + x_in;
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = input[in_idx];
    }
}

// ============================================================================
// Uint8 to Float Conversion with Normalization
// ============================================================================

/// Convert uint8 [0, 255] to float [0, 1] range.
/// Common preprocessing step before resize/normalize.
kernel void uint8_to_float(
    device const uchar* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    uint total_elements = params[0];

    if (gid >= total_elements) return;

    output[gid] = float(input[gid]) / 255.0f;
}

/// Vectorized uint8 to float (4 elements per thread)
kernel void uint8_to_float_vec4(
    device const uchar4* input  [[buffer(0)]],
    device float4* output       [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    uint total_vec4 = params[0] / 4;

    if (gid >= total_vec4) return;

    uchar4 in_val = input[gid];
    output[gid] = float4(in_val) / 255.0f;
}

// ============================================================================
// Large Resolution Optimizations (1024x1024+)
// ============================================================================

/// Tile size for tile-based processing (optimized for 1024x1024+ images)
/// Each tile is 16x16 pixels = 256 pixels, fits in threadgroup memory
constant constexpr uint TILE_SIZE = 16;
constant constexpr uint TILE_PIXELS = TILE_SIZE * TILE_SIZE;

// ============================================================================
// Tile-based Bilinear Resize (optimized for large images)
// ============================================================================

/// Tile-based bilinear resize for large images (1024x1024+).
/// Each threadgroup processes a TILE_SIZE x TILE_SIZE tile of output.
/// Uses thread ID to determine pixel within tile, avoiding redundant loops.
///
/// @param input       Input image [N, H_in, W_in, C]
/// @param output      Output image [N, H_out, W_out, C]
/// @param params      [batch_size, H_in, W_in, H_out, W_out, channels, nhwc]
kernel void image_resize_bilinear_tiled(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint3 lid                   [[thread_position_in_threadgroup]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint tile_x = tgid.x * TILE_SIZE;
    uint tile_y = tgid.y * TILE_SIZE;
    uint n = tgid.z;

    if (n >= batch_size) return;

    // Use local thread ID for pixel within tile
    uint ty = lid.y;
    uint tx = lid.x;

    uint y_out = tile_y + ty;
    uint x_out = tile_x + tx;

    if (y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    float4 sampled = sample_bilinear(
        input + input_offset,
        u, v,
        W_in, H_in, channels, nhwc
    );

    for (uint c = 0; c < channels; ++c) {
        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = sampled[c];
    }
}

/// Threadgroup memory variant of tile-based resize.
/// Optimizes for cases where input reuse is high (e.g. upscaling or slight downscaling).
/// Limits: Only supports up to 4 channels (RGBA).
kernel void image_resize_bilinear_tiled_shared(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint3 lid                   [[thread_position_in_threadgroup]]
) {
    // Support up to 4 channels in shared memory (float4)
    threadgroup float4 shared_input[TILE_SIZE + 2][TILE_SIZE + 2];

    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    if (channels > 4) return; // Fallback or fail for >4 channels

    uint tile_x = tgid.x * TILE_SIZE;
    uint tile_y = tgid.y * TILE_SIZE;
    uint n = tgid.z;

    if (n >= batch_size) return;

    float scale_x = float(W_in) / float(W_out);
    float scale_y = float(H_in) / float(H_out);

    // Calculate input tile boundaries
    // We map the output tile to input coordinates to find what to load
    int in_tile_x = int(float(tile_x) * scale_x);
    int in_tile_y = int(float(tile_y) * scale_y);

    // Load input tile into shared memory (with 1-pixel boundary)
    // Parallel loading: each thread loads one pixel if possible
    // Since TILE_SIZE is small (16), we can just loop or map threads
    for (uint i = lid.y; i < TILE_SIZE + 2; i += TILE_SIZE) {
        for (uint j = lid.x; j < TILE_SIZE + 2; j += TILE_SIZE) {
            int y_in = in_tile_y + int(i) - 1;
            int x_in = in_tile_x + int(j) - 1;
            
            y_in = clamp(y_in, 0, int(H_in) - 1);
            x_in = clamp(x_in, 0, int(W_in) - 1);

            uint input_offset = n * H_in * W_in * channels;
            
            float4 val = float4(0.0f);
            if (nhwc) {
                // Load contiguous channels
                uint idx = input_offset + y_in * W_in * channels + x_in * channels;
                for (uint c = 0; c < channels; ++c) {
                    val[c] = input[idx + c];
                }
            } else {
                // Load planar channels
                for (uint c = 0; c < channels; ++c) {
                    val[c] = input[input_offset + c * H_in * W_in + y_in * W_in + x_in];
                }
            }
            shared_input[i][j] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process output pixels
    uint local_y = lid.y;
    uint local_x = lid.x;
    
    if (local_y < TILE_SIZE && local_x < TILE_SIZE) {
        uint y_out = tile_y + local_y;
        uint x_out = tile_x + local_x;

        if (y_out < H_out && x_out < W_out) {
            float u = (float(x_out) + 0.5f) / float(W_out);
            float v = (float(y_out) + 0.5f) / float(H_out);

            float px = u * float(W_in - 1);
            float py = v * float(H_in - 1);
            
            // Map to shared memory coordinates relative to loaded tile
            float sx = px - float(in_tile_x) + 1.0f;
            float sy = py - float(in_tile_y) + 1.0f;
            
            int x0 = int(sx);
            int y0 = int(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float fx = sx - float(x0);
            float fy = sy - float(y0);
            
            // Clamp to shared memory bounds
            x0 = clamp(x0, 0, int(TILE_SIZE + 1));
            y0 = clamp(y0, 0, int(TILE_SIZE + 1));
            x1 = clamp(x1, 0, int(TILE_SIZE + 1));
            y1 = clamp(y1, 0, int(TILE_SIZE + 1));

            float4 v00 = shared_input[y0][x0];
            float4 v01 = shared_input[y0][x1];
            float4 v10 = shared_input[y1][x0];
            float4 v11 = shared_input[y1][x1];

            float4 top = mix(v00, v01, fx);
            float4 bot = mix(v10, v11, fx);
            float4 result = mix(top, bot, fy);

            uint output_offset = n * H_out * W_out * channels;

            for (uint c = 0; c < channels; ++c) {
                uint out_idx;
                if (nhwc) {
                    out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
                } else {
                    out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
                }
                output[out_idx] = result[c];
            }
        }
    }
}

// ============================================================================
// Multi-pixel Processing Kernels (for large resolutions)
// ============================================================================

/// Process 4 pixels per thread for large image resize.
/// Reduces kernel launch overhead and improves instruction throughput.
kernel void image_resize_bilinear_4pixel(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    // Each thread processes 4 pixels in a 2x2 pattern
    uint n = gid.z;
    uint y_base = gid.y * 2;
    uint x_base = gid.x * 2;

    if (n >= batch_size || y_base >= H_out || x_base >= W_out) return;

    float scale_x = float(W_in) / float(W_out);
    float scale_y = float(H_in) / float(H_out);

    uint input_offset = n * H_in * W_in * channels;

    for (uint dy = 0; dy < 2; ++dy) {
        uint y_out = y_base + dy;
        if (y_out >= H_out) break;

        float v = (float(y_out) + 0.5f) / float(H_out);
        float py = v * float(H_in - 1);
        uint y0 = uint(py);
        uint y1 = min(y0 + 1, H_in - 1);
        float fy = py - float(y0);

        for (uint dx = 0; dx < 2; ++dx) {
            uint x_out = x_base + dx;
            if (x_out >= W_out) break;

            float u = (float(x_out) + 0.5f) / float(W_out);
            float px = u * float(W_in - 1);
            uint x0 = uint(px);
            uint x1 = min(x0 + 1, W_in - 1);
            float fx = px - float(x0);

            uint output_offset = n * H_out * W_out * channels;

            for (uint c = 0; c < channels; ++c) {
                float v00, v01, v10, v11;

                if (nhwc) {
                    v00 = input[input_offset + y0 * W_in * channels + x0 * channels + c];
                    v01 = input[input_offset + y0 * W_in * channels + x1 * channels + c];
                    v10 = input[input_offset + y1 * W_in * channels + x0 * channels + c];
                    v11 = input[input_offset + y1 * W_in * channels + x1 * channels + c];
                } else {
                    v00 = input[input_offset + c * H_in * W_in + y0 * W_in + x0];
                    v01 = input[input_offset + c * H_in * W_in + y0 * W_in + x1];
                    v10 = input[input_offset + c * H_in * W_in + y1 * W_in + x0];
                    v11 = input[input_offset + c * H_in * W_in + y1 * W_in + x1];
                }

                float top = mix(v00, v01, fx);
                float bot = mix(v10, v11, fx);
                float result = mix(top, bot, fy);

                uint out_idx;
                if (nhwc) {
                    out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
                } else {
                    out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
                }
                output[out_idx] = result;
            }
        }
    }
}

// ============================================================================
// Pyramid Resize for Multi-scale Vision Models
// ============================================================================

/// Generate an image pyramid at multiple scales for multi-scale processing.
/// Common in vision transformers that benefit from multiple resolution views.
///
/// @param input       Input image [N, H_in, W_in, C]
/// @param output      Output pyramid concatenated [N, sum(H_outs[i] * W_outs[i]), C]
/// @param scales      Array of scale factors [0.5, 0.25, 0.125, ...]
/// @param params      [batch_size, H_in, W_in, num_scales, channels, nhwc]
kernel void image_resize_pyramid(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    device const float* scales  [[buffer(2)]],
    constant uint* params       [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint num_scales = params[3];
    uint channels = params[4];
    bool nhwc = params[5] != 0;

    // Determine which scale and pixel this thread handles
    uint total_pixels = gid.x;  // Flattened pixel index across all scales

    // Compute scale index and pixel within that scale
    uint scale_idx = 0;
    uint pixel_offset = 0;

    for (uint s = 0; s < num_scales; ++s) {
        float scale = scales[s];
        uint H_scale = uint(float(H_in) * scale + 0.5f);
        uint W_scale = uint(float(W_in) * scale + 0.5f);
        uint scale_pixels = H_scale * W_scale;

        if (total_pixels < pixel_offset + scale_pixels) {
            scale_idx = s;
            break;
        }
        pixel_offset += scale_pixels;
    }

    if (scale_idx >= num_scales) return;

    float scale = scales[scale_idx];
    uint H_scale = uint(float(H_in) * scale + 0.5f);
    uint W_scale = uint(float(W_in) * scale + 0.5f);

    uint pixel_in_scale = total_pixels - pixel_offset;
    uint y_scale = pixel_in_scale / W_scale;
    uint x_scale = pixel_in_scale % W_scale;

    // Map to input coordinates
    float u = (float(x_scale) + 0.5f) / float(W_scale);
    float v = (float(y_scale) + 0.5f) / float(H_scale);

    uint n = gid.z;  // Batch index
    if (n >= batch_size) return;

    uint input_offset = n * H_in * W_in * channels;

    // Compute offsets for pyramid output
    // Each scale's pixels are concatenated
    uint pyramid_offset = n * pixel_offset;  // Simplified - actual offset calc needed
    // For proper pyramid, need to compute cumulative offset per scale

    float4 sampled = sample_bilinear(
        input + input_offset,
        u, v,
        W_in, H_in, channels, nhwc
    );

    // Write to pyramid output
    for (uint c = 0; c < channels; ++c) {
        uint out_idx = pyramid_offset + y_scale * W_scale * channels + x_scale * channels + c;
        output[out_idx] = sampled[c];
    }
}

// ============================================================================
// Large Image Center Crop (1024x1024+)
// ============================================================================

/// Optimized center crop for large images.
/// Crops from center to specified size, common in vision model preprocessing.
kernel void center_crop_large(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    // Compute crop offsets (center crop region)
    uint y_offset = (H_in - H_out) / 2;
    uint x_offset = (W_in - W_out) / 2;

    uint y_in = y_offset + y_out;
    uint x_in = x_offset + x_out;

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    for (uint c = 0; c < channels; ++c) {
        uint in_idx, out_idx;
        if (nhwc) {
            in_idx = input_offset + y_in * W_in * channels + x_in * channels + c;
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            in_idx = input_offset + c * H_in * W_in + y_in * W_in + x_in;
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = input[in_idx];
    }
}

// ============================================================================
// Fused Large Image Pipeline (Crop + Resize + Normalize)
// ============================================================================

/// Complete preprocessing pipeline for large images in a single kernel.
/// Performs: center crop -> bilinear resize -> normalize
/// Eliminates intermediate buffers for memory efficiency.
kernel void preprocess_large_image_fused(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant float* mean        [[buffer(2)]],
    constant float* std_inv     [[buffer(3)]],
    constant uint* params       [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    // params: [batch_size, H_in, W_in, crop_h, crop_w, H_out, W_out, channels, nhwc]
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint crop_h = params[3];
    uint crop_w = params[4];
    uint H_out = params[5];
    uint W_out = params[6];
    uint channels = params[7];
    bool nhwc = params[8] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    // Crop offsets (center of input)
    uint y_offset = (H_in - crop_h) / 2;
    uint x_offset = (W_in - crop_w) / 2;

    // Map output pixel to cropped input coordinates
    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    // Map to full input coordinates (accounting for crop)
    float px = u * float(crop_w - 1) + float(x_offset);
    float py = v * float(crop_h - 1) + float(y_offset);

    px = clamp(px, 0.0f, float(W_in - 1));
    py = clamp(py, 0.0f, float(H_in - 1));

    uint x0 = uint(px);
    uint y0 = uint(py);
    uint x1 = min(x0 + 1, W_in - 1);
    uint y1 = min(y0 + 1, H_in - 1);

    float fx = px - float(x0);
    float fy = py - float(y0);

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    for (uint c = 0; c < channels; ++c) {
        float v00, v01, v10, v11;

        if (nhwc) {
            v00 = input[input_offset + y0 * W_in * channels + x0 * channels + c];
            v01 = input[input_offset + y0 * W_in * channels + x1 * channels + c];
            v10 = input[input_offset + y1 * W_in * channels + x0 * channels + c];
            v11 = input[input_offset + y1 * W_in * channels + x1 * channels + c];
        } else {
            v00 = input[input_offset + c * H_in * W_in + y0 * W_in + x0];
            v01 = input[input_offset + c * H_in * W_in + y0 * W_in + x1];
            v10 = input[input_offset + c * H_in * W_in + y1 * W_in + x0];
            v11 = input[input_offset + c * H_in * W_in + y1 * W_in + x1];
        }

        float top = mix(v00, v01, fx);
        float bot = mix(v10, v11, fx);
        float sampled = mix(top, bot, fy);

        // Apply normalization
        float normalized = (sampled - mean[c]) * std_inv[c];

        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = normalized;
    }
}

// ============================================================================
// Adaptive Patch Extraction for Vision Transformers
// ============================================================================

/// Extract patches from large image for ViT-style processing.
/// Optimized for 1024x1024 images with 16x16 or 32x32 patches.
///
/// @param input       Input image [N, H_in, W_in, C]
/// @param output      Output patches [N, num_patches, patch_size, patch_size, C]
/// @param params      [batch_size, H_in, W_in, patch_size, channels, nhwc]
kernel void extract_patches(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint patch_size = params[3];
    uint channels = params[4];
    bool nhwc = params[5] != 0;

    // Grid layout: x=patch_x, y=patch_y, z=batch
    uint patch_x = gid.x;
    uint patch_y = gid.y;
    uint n = gid.z;

    uint num_patches_x = (W_in + patch_size - 1) / patch_size;
    uint num_patches_y = (H_in + patch_size - 1) / patch_size;

    if (n >= batch_size || patch_x >= num_patches_x || patch_y >= num_patches_y) return;

    // Patch index in flat array
    uint patch_idx = patch_y * num_patches_x + patch_x;

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * num_patches_y * num_patches_x * patch_size * patch_size * channels;

    // Copy all pixels in this patch
    for (uint py = 0; py < patch_size; ++py) {
        uint y_in = patch_y * patch_size + py;
        if (y_in >= H_in) break;

        for (uint px = 0; px < patch_size; ++px) {
            uint x_in = patch_x * patch_size + px;
            if (x_in >= W_in) break;

            for (uint c = 0; c < channels; ++c) {
                uint in_idx, out_idx;
                if (nhwc) {
                    in_idx = input_offset + y_in * W_in * channels + x_in * channels + c;
                    out_idx = output_offset + patch_idx * patch_size * patch_size * channels +
                              py * patch_size * channels + px * channels + c;
                } else {
                    in_idx = input_offset + c * H_in * W_in + y_in * W_in + x_in;
                    out_idx = output_offset + patch_idx * patch_size * patch_size * channels +
                              c * patch_size * patch_size + py * patch_size + px;
                }
                output[out_idx] = input[in_idx];
            }
        }
    }
}

/// Vectorized patch extraction for 3-channel (RGB) images.
/// Processes 4 patches per thread for better utilization.
kernel void extract_patches_vec4(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint patch_size = params[3];
    uint channels = params[4];
    bool nhwc = params[5] != 0;

    // Assumes channels = 3 or 4 for vectorization
    if (channels < 3 || channels > 4) return;

    // Each thread processes a 2x2 grid of patches
    uint n = gid.z;
    uint patch_y_base = gid.y * 2;
    uint patch_x_base = gid.x * 2;

    uint num_patches_x = (W_in + patch_size - 1) / patch_size;
    uint num_patches_y = (H_in + patch_size - 1) / patch_size;

    if (n >= batch_size) return;

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * num_patches_y * num_patches_x * patch_size * patch_size * channels;

    for (uint dy = 0; dy < 2; ++dy) {
        uint patch_y = patch_y_base + dy;
        if (patch_y >= num_patches_y) break;

        for (uint dx = 0; dx < 2; ++dx) {
            uint patch_x = patch_x_base + dx;
            if (patch_x >= num_patches_x) break;

            uint patch_idx = patch_y * num_patches_x + patch_x;

            for (uint py = 0; py < patch_size; ++py) {
                uint y_in = patch_y * patch_size + py;
                if (y_in >= H_in) break;

                for (uint px = 0; px < patch_size; ++px) {
                    uint x_in = patch_x * patch_size + px;
                    if (x_in >= W_in) break;

                    // Vectorized load/store for channels
                    if (nhwc) {
                        uint in_idx = input_offset + y_in * W_in * channels + x_in * channels;
                        uint out_idx = output_offset + patch_idx * patch_size * patch_size * channels +
                                      py * patch_size * channels + px * channels;

                        if (channels >= 4) {
                            float4 val = *((device float4*)(input + in_idx));
                            *((device float4*)(output + out_idx)) = val;
                        } else {
                            for (uint c = 0; c < channels; ++c) {
                                output[out_idx + c] = input[in_idx + c];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Large Resolution Bicubic with Lanczos-style Window
// ============================================================================

/// Bicubic resize with 8x8 neighborhood (Lanczos-like) for high-quality downscaling.
/// Uses larger support window for better results when downscaling significantly
/// (e.g., 2048x2048 -> 1024x1024 or larger downscale factors).
kernel void image_resize_bicubic_8x8(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    float u = (float(x_out) + 0.5f) / float(W_out);
    float v = (float(y_out) + 0.5f) / float(H_out);

    float px = u * float(W_in - 1);
    float py = v * float(H_in - 1);

    int x0 = int(px);
    int y0 = int(py);
    float fx = px - float(x0);
    float fy = py - float(y0);

    // Lanczos window of 8x8 (4 pixels each side)
    float result = 0.0f;

    uint input_offset = n * H_in * W_in * channels;
    uint output_offset = n * H_out * W_out * channels;

    for (uint c = 0; c < channels; ++c) {
        float col_sum = 0.0f;
        float weight_sum = 0.0f;

        for (int j = -3; j <= 4; ++j) {
            float row_sum = 0.0f;
            float row_weight = 0.0f;
            int y = clamp(y0 + j, 0, int(H_in - 1));

            for (int i = -3; i <= 4; ++i) {
                int x = clamp(x0 + i, 0, int(W_in - 1));

                float val;
                if (nhwc) {
                    val = input[input_offset + y * int(W_in * channels) + x * int(channels) + int(c)];
                } else {
                    val = input[input_offset + int(c) * int(H_in * W_in) + y * int(W_in) + x];
                }

                // Lanczos-3 window function
                float dx = abs(float(i) - fx);
                float dy = abs(float(j) - fy);
                float wx = (dx < 3.0f) ? (3.0f * sin(M_PI_F * dx) * sin(M_PI_F * dx / 3.0f) /
                                          (M_PI_F * M_PI_F * dx * dx + 1e-6f)) : 0.0f;
                float wy = (dy < 3.0f) ? (3.0f * sin(M_PI_F * dy) * sin(M_PI_F * dy / 3.0f) /
                                          (M_PI_F * M_PI_F * dy * dy + 1e-6f)) : 0.0f;
                float w = wx * wy;

                row_sum += val * w;
                row_weight += w;
            }

            col_sum += row_sum;
            weight_sum += row_weight;
        }

        float sampled = (weight_sum > 1e-6f) ? col_sum / weight_sum : 0.0f;

        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = sampled;
    }
}

// ============================================================================
// Aspect Ratio Preserving Resize (for Qwen2-VL style models)
// ============================================================================

/// Resize image while preserving aspect ratio, padding to maintain square output.
/// Common preprocessing for vision models that expect square inputs.
///
/// @param input       Input image [N, H_in, W_in, C]
/// @param output      Output image [N, H_out, W_out, C] with padding
/// @param params      [batch_size, H_in, W_in, H_out, W_out, channels, nhwc, pad_value]
kernel void resize_aspect_ratio_preserve(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint H_in = params[1];
    uint W_in = params[2];
    uint H_out = params[3];
    uint W_out = params[4];
    uint channels = params[5];
    bool nhwc = params[6] != 0;
    float pad_value = as_type<float>(params[7]);

    uint n = gid.z;
    uint y_out = gid.y;
    uint x_out = gid.x;

    if (n >= batch_size || y_out >= H_out || x_out >= W_out) return;

    // Compute scale to fit within output bounds
    float scale = min(float(H_out) / float(H_in), float(W_out) / float(W_in));
    uint H_scaled = uint(float(H_in) * scale + 0.5f);
    uint W_scaled = uint(float(W_in) * scale + 0.5f);

    // Center the scaled image in output
    uint y_offset = (H_out - H_scaled) / 2;
    uint x_offset = (W_out - W_scaled) / 2;

    // Check if this output pixel is in the padded region
    if (y_out < y_offset || y_out >= y_offset + H_scaled ||
        x_out < x_offset || x_out >= x_offset + W_scaled) {
        // Padding region
        uint output_offset = n * H_out * W_out * channels;
        for (uint c = 0; c < channels; ++c) {
            uint out_idx;
            if (nhwc) {
                out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
            } else {
                out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
            }
            output[out_idx] = pad_value;
        }
        return;
    }

    // Map output pixel to input coordinates (accounting for padding)
    uint y_scaled = y_out - y_offset;
    uint x_scaled = x_out - x_offset;

    float u = float(x_scaled) / float(W_scaled - 1);
    float v = float(y_scaled) / float(H_scaled - 1);

    uint input_offset = n * H_in * W_in * channels;

    float4 sampled = sample_bilinear(
        input + input_offset,
        u, v,
        W_in, H_in, channels, nhwc
    );

    uint output_offset = n * H_out * W_out * channels;
    for (uint c = 0; c < channels; ++c) {
        uint out_idx;
        if (nhwc) {
            out_idx = output_offset + y_out * W_out * channels + x_out * channels + c;
        } else {
            out_idx = output_offset + c * H_out * W_out + y_out * W_out + x_out;
        }
        output[out_idx] = sampled[c];
    }
}

// ============================================================================
// Memory-Aware Batch Normalization (for large batches of large images)
// ============================================================================

/// Compute channel-wise mean for normalization in a single pass.
/// Reduces all pixel values per channel using atomic operations.
kernel void compute_channel_mean(
    device const float* input   [[buffer(0)]],
    device atomic<float>* mean  [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    uint channels = params[3];
    bool nhwc = params[4] != 0;

    uint n = gid.z;
    uint y = gid.y;
    uint x = gid.x;

    if (n >= batch_size || y >= height || x >= width) return;

    uint base_offset = n * height * width * channels;

    for (uint c = 0; c < channels; ++c) {
        uint idx;
        if (nhwc) {
            idx = base_offset + y * width * channels + x * channels + c;
        } else {
            idx = base_offset + c * height * width + y * width + x;
        }

        atomic_fetch_add_explicit(&mean[c], input[idx], memory_order_relaxed);
    }
}

/// Compute channel-wise std for normalization.
/// Uses precomputed mean to compute variance in a single pass.
kernel void compute_channel_std(
    device const float* input   [[buffer(0)]],
    device atomic<float>* var   [[buffer(1)]],
    constant float* mean        [[buffer(2)]],
    constant uint* params       [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    uint channels = params[3];
    bool nhwc = params[4] != 0;

    uint n = gid.z;
    uint y = gid.y;
    uint x = gid.x;

    if (n >= batch_size || y >= height || x >= width) return;

    uint base_offset = n * height * width * channels;

    for (uint c = 0; c < channels; ++c) {
        uint idx;
        if (nhwc) {
            idx = base_offset + y * width * channels + x * channels + c;
        } else {
            idx = base_offset + c * height * width + y * width + x;
        }

        float diff = input[idx] - mean[c];
        atomic_fetch_add_explicit(&var[c], diff * diff, memory_order_relaxed);
    }
}
