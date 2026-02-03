// ARM NEON-optimized CPU fallback dequantizers
// Provides vectorized dequantization for quantized weights when Metal is unavailable

#include <cstdint>
#include <cstring>
#include <algorithm>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace metal_marlin {
namespace cpu_dequant {

// E2M1 FP4 codebook (16 values)
static const float E2M1_CODEBOOK[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// NF4 codebook (QLoRA)
static const float NF4_CODEBOOK[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

#ifdef __ARM_NEON

// Unpack 8 nibbles from uint32 using NEON
inline void unpack_nibbles_neon(const uint32_t word, uint8_t* out) {
    // Extract nibbles: word contains 8 4-bit values
    // nibble[i] = (word >> (i*4)) & 0xF
    uint32x4_t words = vdupq_n_u32(word);
    
    // Shifts for extracting nibbles 0-3 and 4-7
    const uint32x4_t shifts_lo = {0, 4, 8, 12};
    const uint32x4_t shifts_hi = {16, 20, 24, 28};
    
    uint32x4_t nibbles_lo = vshlq_u32(words, vnegq_s32(vreinterpretq_s32_u32(shifts_lo)));
    uint32x4_t nibbles_hi = vshlq_u32(words, vnegq_s32(vreinterpretq_s32_u32(shifts_hi)));
    
    uint32x4_t mask = vdupq_n_u32(0xF);
    nibbles_lo = vandq_u32(nibbles_lo, mask);
    nibbles_hi = vandq_u32(nibbles_hi, mask);
    
    // Narrow to uint8
    uint16x4_t narrow_lo = vmovn_u32(nibbles_lo);
    uint16x4_t narrow_hi = vmovn_u32(nibbles_hi);
    uint8x8_t result = vmovn_u16(vcombine_u16(narrow_lo, narrow_hi));
    
    vst1_u8(out, result);
}

// FP4 E2M1 dequantization with NEON
void dequant_fp4_e2m1_neon(
    const uint32_t* packed,      // [K/8, N]
    const float* scales,         // [K/group_size, N]
    float* output,               // [K, N]
    int K,
    int N,
    int group_size
) {
    const int k_blocks = K / 8;
    
    for (int n = 0; n < N; n += 4) {
        const int n_lanes = std::min(4, N - n);
        
        for (int kb = 0; kb < k_blocks; kb++) {
            const int k_base = kb * 8;
            const int group_idx = k_base / group_size;
            
            // Load 4 packed words (4 columns)
            uint32_t words[4];
            for (int lane = 0; lane < n_lanes; lane++) {
                words[lane] = packed[kb * N + (n + lane)];
            }
            
            // Process 8 rows per block
            for (int i = 0; i < 8; i++) {
                const int k_idx = k_base + i;
                if (k_idx >= K) break;
                
                // Extract nibbles for this row
                uint8_t nibbles[4];
                for (int lane = 0; lane < n_lanes; lane++) {
                    nibbles[lane] = (words[lane] >> (i * 4)) & 0xF;
                }
                
                // Load scales (vectorized)
                float32x4_t vscales = vld1q_f32(&scales[group_idx * N + n]);
                
                // Codebook lookup
                float values[4];
                for (int lane = 0; lane < n_lanes; lane++) {
                    values[lane] = E2M1_CODEBOOK[nibbles[lane]];
                }
                float32x4_t vvalues = vld1q_f32(values);
                
                // Multiply: output = values * scales
                float32x4_t result = vmulq_f32(vvalues, vscales);
                
                // Store result
                vst1q_f32(&output[k_idx * N + n], result);
            }
        }
    }
}

// INT4 symmetric dequantization with NEON
void dequant_int4_symmetric_neon(
    const uint32_t* packed,      // [K/8, N]
    const float* scales,         // [K/group_size, N]
    float* output,               // [K, N]
    int K,
    int N,
    int group_size
) {
    const int k_blocks = K / 8;
    const float32x4_t vzero_offset = vdupq_n_f32(-8.0f);
    
    for (int n = 0; n < N; n += 4) {
        const int n_lanes = std::min(4, N - n);
        
        for (int kb = 0; kb < k_blocks; kb++) {
            const int k_base = kb * 8;
            const int group_idx = k_base / group_size;
            
            uint32_t words[4];
            for (int lane = 0; lane < n_lanes; lane++) {
                words[lane] = packed[kb * N + (n + lane)];
            }
            
            float32x4_t vscales = vld1q_f32(&scales[group_idx * N + n]);
            
            for (int i = 0; i < 8; i++) {
                const int k_idx = k_base + i;
                if (k_idx >= K) break;
                
                // Extract nibbles
                uint8_t nibbles[4];
                for (int lane = 0; lane < n_lanes; lane++) {
                    nibbles[lane] = (words[lane] >> (i * 4)) & 0xF;
                }
                
                // Convert to float and offset: (nibble - 8)
                float values[4];
                for (int lane = 0; lane < n_lanes; lane++) {
                    values[lane] = static_cast<float>(nibbles[lane]);
                }
                float32x4_t vvalues = vld1q_f32(values);
                vvalues = vaddq_f32(vvalues, vzero_offset);
                
                // Multiply by scale
                float32x4_t result = vmulq_f32(vvalues, vscales);
                
                vst1q_f32(&output[k_idx * N + n], result);
            }
        }
    }
}

// NF4 dequantization with NEON
void dequant_nf4_neon(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    const int k_blocks = K / 8;
    
    for (int n = 0; n < N; n += 4) {
        const int n_lanes = std::min(4, N - n);
        
        for (int kb = 0; kb < k_blocks; kb++) {
            const int k_base = kb * 8;
            const int group_idx = k_base / group_size;
            
            uint32_t words[4];
            for (int lane = 0; lane < n_lanes; lane++) {
                words[lane] = packed[kb * N + (n + lane)];
            }
            
            float32x4_t vscales = vld1q_f32(&scales[group_idx * N + n]);
            
            for (int i = 0; i < 8; i++) {
                const int k_idx = k_base + i;
                if (k_idx >= K) break;
                
                uint8_t nibbles[4];
                for (int lane = 0; lane < n_lanes; lane++) {
                    nibbles[lane] = (words[lane] >> (i * 4)) & 0xF;
                }
                
                // NF4 codebook lookup
                float values[4];
                for (int lane = 0; lane < n_lanes; lane++) {
                    values[lane] = NF4_CODEBOOK[nibbles[lane]];
                }
                float32x4_t vvalues = vld1q_f32(values);
                
                float32x4_t result = vmulq_f32(vvalues, vscales);
                vst1q_f32(&output[k_idx * N + n], result);
            }
        }
    }
}

// INT8 dequantization with NEON
void dequant_int8_neon(
    const int8_t* data,          // [K, N]
    const float* scales,         // [K/group_size, N]
    float* output,               // [K, N]
    int K,
    int N,
    int group_size
) {
    for (int k = 0; k < K; k += 8) {
        const int group_idx = k / group_size;
        const int k_lanes = std::min(8, K - k);
        
        for (int n = 0; n < N; n += 4) {
            const int n_lanes = std::min(4, N - n);
            
            // Load scale
            float32x4_t vscales = vld1q_f32(&scales[group_idx * N + n]);
            
            for (int i = 0; i < k_lanes; i++) {
                const int k_idx = k + i;
                
                // Load int8 values
                int8_t values_i8[4];
                for (int lane = 0; lane < n_lanes; lane++) {
                    values_i8[lane] = data[k_idx * N + n + lane];
                }
                
                // Convert int8 -> int32 -> float32
                int16x4_t values_i16 = vget_low_s16(vmovl_s8(vld1_s8(values_i8)));
                int32x4_t values_i32 = vmovl_s16(values_i16);
                float32x4_t values_f32 = vcvtq_f32_s32(values_i32);
                
                // Multiply by scale
                float32x4_t result = vmulq_f32(values_f32, vscales);
                
                vst1q_f32(&output[k_idx * N + n], result);
            }
        }
    }
}

// FP8 E4M3 dequantization with NEON
void dequant_fp8_e4m3_neon(
    const uint8_t* data,         // [K, N]
    const float* scales,         // [K/group_size, N]
    float* output,               // [K, N]
    int K,
    int N,
    int group_size
) {
    // FP8 E4M3: sign(1) | exp(4) | mantissa(3), bias=7
    
    for (int k = 0; k < K; k++) {
        const int group_idx = k / group_size;
        
        for (int n = 0; n < N; n += 4) {
            const int n_lanes = std::min(4, N - n);
            
            // Load FP8 codes
            uint8_t codes[4];
            for (int lane = 0; lane < n_lanes; lane++) {
                codes[lane] = data[k * N + n + lane];
            }
            
            // Decode FP8 E4M3 to float32
            float values[4];
            for (int lane = 0; lane < n_lanes; lane++) {
                const uint8_t code = codes[lane];
                const uint8_t s = (code >> 7) & 1;
                const uint8_t e = (code >> 3) & 0xF;
                const uint8_t m = code & 0x7;
                
                float value;
                if (e == 15) {
                    value = NAN;  // Invalid
                } else if (e == 0) {
                    if (m == 0) {
                        value = s ? -0.0f : 0.0f;
                    } else {
                        // Subnormal
                        value = (s ? -1.0f : 1.0f) * 0.015625f * (m / 8.0f);
                    }
                } else {
                    // Normal
                    float exp_val = static_cast<float>(1 << (e - 7));
                    value = (s ? -1.0f : 1.0f) * exp_val * (1.0f + m / 8.0f);
                }
                values[lane] = value;
            }
            
            // Load scale and multiply
            float32x4_t vvalues = vld1q_f32(values);
            float32x4_t vscales = vld1q_f32(&scales[group_idx * N + n]);
            float32x4_t result = vmulq_f32(vvalues, vscales);
            
            vst1q_f32(&output[k * N + n], result);
        }
    }
}

// FP8 E5M2 dequantization with NEON
void dequant_fp8_e5m2_neon(
    const uint8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    // FP8 E5M2: sign(1) | exp(5) | mantissa(2), bias=15
    
    for (int k = 0; k < K; k++) {
        const int group_idx = k / group_size;
        
        for (int n = 0; n < N; n += 4) {
            const int n_lanes = std::min(4, N - n);
            
            uint8_t codes[4];
            for (int lane = 0; lane < n_lanes; lane++) {
                codes[lane] = data[k * N + n + lane];
            }
            
            float values[4];
            for (int lane = 0; lane < n_lanes; lane++) {
                const uint8_t code = codes[lane];
                const uint8_t s = (code >> 7) & 1;
                const uint8_t e = (code >> 2) & 0x1F;
                const uint8_t m = code & 0x3;
                
                float value;
                if (e == 31) {
                    value = (m == 0) ? ((s ? -1.0f : 1.0f) * INFINITY) : NAN;
                } else if (e == 0) {
                    if (m == 0) {
                        value = s ? -0.0f : 0.0f;
                    } else {
                        // Subnormal
                        value = (s ? -1.0f : 1.0f) * 6.103515625e-05f * (m / 4.0f);
                    }
                } else {
                    // Normal
                    float exp_val = static_cast<float>(1 << (e - 15));
                    value = (s ? -1.0f : 1.0f) * exp_val * (1.0f + m / 4.0f);
                }
                values[lane] = value;
            }
            
            float32x4_t vvalues = vld1q_f32(values);
            float32x4_t vscales = vld1q_f32(&scales[group_idx * N + n]);
            float32x4_t result = vmulq_f32(vvalues, vscales);
            
            vst1q_f32(&output[k * N + n], result);
        }
    }
}

#else // Scalar fallback implementations

void dequant_fp4_e2m1_scalar(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    const int k_blocks = K / 8;
    
    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * 8;
        const int group_idx = k_base / group_size;
        
        for (int i = 0; i < 8; i++) {
            const int k_idx = k_base + i;
            if (k_idx >= K) break;
            
            for (int n = 0; n < N; n++) {
                const uint8_t nibble = (packed[kb * N + n] >> (i * 4)) & 0xF;
                const float scale = scales[group_idx * N + n];
                output[k_idx * N + n] = E2M1_CODEBOOK[nibble] * scale;
            }
        }
    }
}

void dequant_int4_symmetric_scalar(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    const int k_blocks = K / 8;
    
    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * 8;
        const int group_idx = k_base / group_size;
        
        for (int i = 0; i < 8; i++) {
            const int k_idx = k_base + i;
            if (k_idx >= K) break;
            
            for (int n = 0; n < N; n++) {
                const uint8_t nibble = (packed[kb * N + n] >> (i * 4)) & 0xF;
                const float scale = scales[group_idx * N + n];
                output[k_idx * N + n] = (static_cast<float>(nibble) - 8.0f) * scale;
            }
        }
    }
}

void dequant_nf4_scalar(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    const int k_blocks = K / 8;
    
    for (int kb = 0; kb < k_blocks; kb++) {
        const int k_base = kb * 8;
        const int group_idx = k_base / group_size;
        
        for (int i = 0; i < 8; i++) {
            const int k_idx = k_base + i;
            if (k_idx >= K) break;
            
            for (int n = 0; n < N; n++) {
                const uint8_t nibble = (packed[kb * N + n] >> (i * 4)) & 0xF;
                const float scale = scales[group_idx * N + n];
                output[k_idx * N + n] = NF4_CODEBOOK[nibble] * scale;
            }
        }
    }
}

void dequant_int8_scalar(
    const int8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    for (int k = 0; k < K; k++) {
        const int group_idx = k / group_size;
        
        for (int n = 0; n < N; n++) {
            const float scale = scales[group_idx * N + n];
            output[k * N + n] = static_cast<float>(data[k * N + n]) * scale;
        }
    }
}

#endif // __ARM_NEON

// Public API - auto-dispatches to NEON or scalar
void dequant_fp4_e2m1(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
#ifdef __ARM_NEON
    dequant_fp4_e2m1_neon(packed, scales, output, K, N, group_size);
#else
    dequant_fp4_e2m1_scalar(packed, scales, output, K, N, group_size);
#endif
}

void dequant_int4_symmetric(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
#ifdef __ARM_NEON
    dequant_int4_symmetric_neon(packed, scales, output, K, N, group_size);
#else
    dequant_int4_symmetric_scalar(packed, scales, output, K, N, group_size);
#endif
}

void dequant_nf4(
    const uint32_t* packed,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
#ifdef __ARM_NEON
    dequant_nf4_neon(packed, scales, output, K, N, group_size);
#else
    dequant_nf4_scalar(packed, scales, output, K, N, group_size);
#endif
}

void dequant_int8(
    const int8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
#ifdef __ARM_NEON
    dequant_int8_neon(data, scales, output, K, N, group_size);
#else
    dequant_int8_scalar(data, scales, output, K, N, group_size);
#endif
}

#ifdef __ARM_NEON
void dequant_fp8_e4m3(
    const uint8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    dequant_fp8_e4m3_neon(data, scales, output, K, N, group_size);
}

void dequant_fp8_e5m2(
    const uint8_t* data,
    const float* scales,
    float* output,
    int K,
    int N,
    int group_size
) {
    dequant_fp8_e5m2_neon(data, scales, output, K, N, group_size);
}
#endif

} // namespace cpu_dequant
} // namespace metal_marlin
