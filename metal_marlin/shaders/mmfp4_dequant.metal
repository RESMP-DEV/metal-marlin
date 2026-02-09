#include <metal_stdlib>
using namespace metal;

inline half mmfp4_e2m1_value(uint nibble) {
    switch (nibble & 0xFu) {
        case 0u: return half(0.0h);
        case 1u: return half(0.5h);
        case 2u: return half(1.0h);
        case 3u: return half(1.5h);
        case 4u: return half(2.0h);
        case 5u: return half(3.0h);
        case 6u: return half(4.0h);
        case 7u: return half(6.0h);
        case 8u: return half(-0.0h);
        case 9u: return half(-0.5h);
        case 10u: return half(-1.0h);
        case 11u: return half(-1.5h);
        case 12u: return half(-2.0h);
        case 13u: return half(-3.0h);
        case 14u: return half(-4.0h);
        case 15u: return half(-6.0h);
        default: return half(0.0h);
    }
}

kernel void dequantize_mmfp4(
    device const uint* B_packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device half* out [[buffer(2)]],
    device const uint* K_p [[buffer(3)]],
    device const uint* N_p [[buffer(4)]],
    device const uint* group_size_p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint K = K_p[0];
    const uint N = N_p[0];
    const uint group_size = group_size_p[0];
    if (group_size == 0u) {
        return;
    }

    const uint n = gid.x;
    const uint k = gid.y;
    if (k >= K || n >= N) {
        return;
    }

    const uint n_packed = (N + 7u) >> 3;
    const uint packed_word = B_packed[k * n_packed + (n >> 3)];
    const uint nibble = (packed_word >> ((n & 7u) * 4u)) & 0xFu;
    const uint group_idx = k / group_size;
    const half scale = scales[group_idx * N + n];
    out[k * N + n] = mmfp4_e2m1_value(nibble) * scale;
}
