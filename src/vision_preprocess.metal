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
//   4. vit_patch_extract       - Extract fixed-size patches for ViT
//   5. dynamic_resize_patches  - Qwen2-VL style dynamic resolution
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
// ViT Patch Extraction Kernel
// ============================================================================

/// Extract non-overlapping patches for Vision Transformer models.
/// Converts image [N, H, W, C] -> patches [N, num_patches, patch_size * patch_size * C]
///
/// For a 224x224 image with patch_size=16:
///   num_patches = (224/16) * (224/16) = 196
///   patch_dim = 16 * 16 * 3 = 768
///
/// @param input       Input image [N, H, W, C] (NHWC only for patches)
/// @param output      Output patches [N, num_patches, patch_dim]
/// @param params      [batch_size, height, width, channels, patch_size]
kernel void vit_patch_extract(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    uint channels = params[3];
    uint patch_size = params[4];

    uint patches_h = height / patch_size;
    uint patches_w = width / patch_size;
    uint num_patches = patches_h * patches_w;
    uint patch_dim = patch_size * patch_size * channels;

    // gid.z = batch index
    // gid.y = patch row (patch_y)
    // gid.x = patch col (patch_x)
    uint n = gid.z;
    uint patch_y = gid.y;
    uint patch_x = gid.x;

    if (n >= batch_size || patch_y >= patches_h || patch_x >= patches_w) return;

    uint patch_idx = patch_y * patches_w + patch_x;

    // Starting pixel in input image
    uint y_start = patch_y * patch_size;
    uint x_start = patch_x * patch_size;

    // Copy patch to output (flattened)
    uint input_offset = n * height * width * channels;
    uint output_offset = n * num_patches * patch_dim + patch_idx * patch_dim;

    uint out_idx = 0;
    for (uint py = 0; py < patch_size; ++py) {
        for (uint px = 0; px < patch_size; ++px) {
            uint y = y_start + py;
            uint x = x_start + px;
            uint in_base = input_offset + y * width * channels + x * channels;

            for (uint c = 0; c < channels; ++c) {
                output[output_offset + out_idx++] = input[in_base + c];
            }
        }
    }
}

/// Half-precision patch extraction
kernel void vit_patch_extract_f16(
    device const half* input    [[buffer(0)]],
    device half* output         [[buffer(1)]],
    constant uint* params       [[buffer(2)]],
    uint3 gid                   [[thread_position_in_grid]]
) {
    uint batch_size = params[0];
    uint height = params[1];
    uint width = params[2];
    uint channels = params[3];
    uint patch_size = params[4];

    uint patches_h = height / patch_size;
    uint patches_w = width / patch_size;
    uint num_patches = patches_h * patches_w;
    uint patch_dim = patch_size * patch_size * channels;

    uint n = gid.z;
    uint patch_y = gid.y;
    uint patch_x = gid.x;

    if (n >= batch_size || patch_y >= patches_h || patch_x >= patches_w) return;

    uint patch_idx = patch_y * patches_w + patch_x;
    uint y_start = patch_y * patch_size;
    uint x_start = patch_x * patch_size;

    uint input_offset = n * height * width * channels;
    uint output_offset = n * num_patches * patch_dim + patch_idx * patch_dim;

    uint out_idx = 0;
    for (uint py = 0; py < patch_size; ++py) {
        for (uint px = 0; px < patch_size; ++px) {
            uint y = y_start + py;
            uint x = x_start + px;
            uint in_base = input_offset + y * width * channels + x * channels;

            for (uint c = 0; c < channels; ++c) {
                output[output_offset + out_idx++] = input[in_base + c];
            }
        }
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
