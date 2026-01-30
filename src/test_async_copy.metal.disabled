#include <metal_stdlib>
using namespace metal;

// Experimental: private async copy instruction (undocumented, use at own risk).
struct _simdgroup_event_t;
thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d(
    ulong element_size,
    ulong element_alignment,
    threadgroup void* dst,
    ulong dst_elements_per_row,
    ulong dst_element_stride,
    ulong2 dst_tile_dimensions,
    const device void* src,
    ulong src_elements_per_row,
    ulong src_element_stride,
    ulong2 src_tile_dimensions,
    long2 offset_in_src_tile,
    int clamp_mode)
    __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

void __metal_wait_simdgroup_events(int count, thread _simdgroup_event_t** events)
    __asm("air.wait_simdgroup_events");

constexpr ushort kTileM = 8;
constexpr ushort kTileN = 8;
constexpr ushort kTileK = 8;

inline thread _simdgroup_event_t* simdgroup_async_copy_2d_float(
    const device float* src,
    uint src_elements_per_row,
    ushort2 tile_size,
    threadgroup float* dst) {
    return __metal_simdgroup_async_copy_2d(
        sizeof(float),
        alignof(float),
        reinterpret_cast<threadgroup void*>(dst),
        ulong(tile_size.x),
        1,
        ulong2(tile_size.x, tile_size.y),
        reinterpret_cast<const device void*>(src),
        ulong(src_elements_per_row),
        1,
        ulong2(tile_size.x, tile_size.y),
        long2(0),
        0);
}

// Asynchronous copy GEMM (device -> threadgroup staging).
// Assumes M, N, K are multiples of 8 and the threadgroup size is (8, 8, 1).
kernel void test_async_copy_gemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    device uint* stats [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float A_tile[kTileM][kTileK];
    threadgroup float B_tile[kTileK][kTileN];

    uint row = tgid.y * kTileM + tid.y;
    uint col = tgid.x * kTileN + tid.x;

    float acc = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += kTileK) {
        if (tid.x == 0 && tid.y == 0) {
            const device float* A_src = A + (tgid.y * kTileM) * K + k0;
            const device float* B_src = B + k0 * N + (tgid.x * kTileN);

            thread _simdgroup_event_t* events[2];
            events[0] = simdgroup_async_copy_2d_float(
                A_src,
                K,
                ushort2(kTileK, kTileM),
                &A_tile[0][0]);
            events[1] = simdgroup_async_copy_2d_float(
                B_src,
                N,
                ushort2(kTileN, kTileK),
                &B_tile[0][0]);
            __metal_wait_simdgroup_events(2, events);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            for (uint kk = 0; kk < kTileK; kk++) {
                acc += A_tile[tid.y][kk] * B_tile[kk][tid.x];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }

    if (tgid.x == 0 && tgid.y == 0 && tid.x == 0 && tid.y == 0) {
        stats[0] = M * K * sizeof(float); // A bytes read
        stats[1] = K * N * sizeof(float); // B bytes read
        stats[2] = M * N * sizeof(float); // C bytes written
        stats[3] = 2u * M * N * K;        // FLOPs
    }
}

// Synchronous copy GEMM baseline.
kernel void test_sync_copy_gemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    device uint* stats [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float A_tile[kTileM][kTileK];
    threadgroup float B_tile[kTileK][kTileN];

    uint row = tgid.y * kTileM + tid.y;
    uint col = tgid.x * kTileN + tid.x;

    float acc = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += kTileK) {
        uint a_row = tgid.y * kTileM + tid.y;
        uint a_col = k0 + tid.x;
        uint b_row = k0 + tid.y;
        uint b_col = tgid.x * kTileN + tid.x;

        A_tile[tid.y][tid.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        B_tile[tid.y][tid.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M && col < N) {
            for (uint kk = 0; kk < kTileK; kk++) {
                acc += A_tile[tid.y][kk] * B_tile[kk][tid.x];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }

    if (tgid.x == 0 && tgid.y == 0 && tid.x == 0 && tid.y == 0) {
        stats[0] = M * K * sizeof(float);
        stats[1] = K * N * sizeof(float);
        stats[2] = M * N * sizeof(float);
        stats[3] = 2u * M * N * K;
    }
}
