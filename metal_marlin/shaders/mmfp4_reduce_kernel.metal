#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// -----------------------------------------------------------------------------
// MMFP4 Reduction Kernels
// -----------------------------------------------------------------------------
//
// This file implements row-wise and column-wise reductions (Sum/Mean)
// using Metal's simdgroup_matrix (AMX) for high throughput.
//
// Concepts:
// - Row Reduction: Treated as A * Ones_Vec.
//   We accumulate 8x8 chunks of A multiplied by an 8x8 matrix of 1s.
//   The result contains the row partial sums repeated in every column.
//
// - Column Reduction: Treated as Ones_Vec * A.
//   We accumulate an 8x8 matrix of 1s multiplied by 8x8 chunks of A.
//   The result contains the column partial sums repeated in every row.
//
// - Bank Conflicts: Threadgroup memory uses padded strides (e.g., +4 or +1)
//   to avoid bank conflicts when accessing columns/rows.
//
// -----------------------------------------------------------------------------

constant uint SIMDGROUP_SIZE = 32;
constant uint TILE_DIM = 8;

// Helper: Fill 8x8 simdgroup matrix with 1.0h (half)
// Used as the identity accumulator for reductions.
// Note: We perform accumulation in float for precision, but inputs are half.
// The "Ones" matrix is loaded as half.
inline simdgroup_matrix<half, 8, 8> make_ones_matrix() {
    return make_filled_simdgroup_matrix<half, 8, 8>(1.0h);
}

// -----------------------------------------------------------------------------
// Kernel: reduce_rows (Sum/Mean)
// -----------------------------------------------------------------------------
// Reduces matrix of size MxN along the N dimension (rows).
// Output: Mx1 vector.
//
// Parameters:
// - src: Input matrix (MxN)
// - dst: Output vector (M)
// - M_p, N_p: Dimensions
// - scale_p: Scalar to multiply result by (1.0 for sum, 1.0/N for mean)
//
// Grid:
// - Threads: 32 per threadgroup (1 simdgroup)
// - Threadgroups: (1, (M + 7) / 8, 1) -> Each simdgroup handles 8 rows.
kernel void reduce_rows(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    device const uint* M_p [[buffer(2)]],
    device const uint* N_p [[buffer(3)]],
    device const float* scale_p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint M = *M_p;
    const uint N = *N_p;
    const float scale = *scale_p;

    // Each simdgroup processes 8 rows.
    const uint row_base = tgid.y * 8;
    
    // Boundary check for rows
    if (row_base >= M) return;

    // Accumulator: 8x8 float matrix
    // We want to calculate Sum(A[row, :]).
    // We compute Accumulator += A_chunk * Ones.
    // Resulting Accumulator[r][c] will contain sum(A_chunk[r][:]) in every column c.
    simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<half, 8, 8> ones = make_ones_matrix();

    // Iterate over N in chunks of 8
    const uint num_blocks = N / 8;
    for (uint k = 0; k < num_blocks; ++k) {
        simdgroup_matrix<half, 8, 8> a_frag;
        // Load 8x8 block from src
        // src is row-major. simdgroup_load handles stride.
        ulong src_offset = (ulong)row_base * N + k * 8;
        simdgroup_load(a_frag, src + src_offset, N, ulong2(M - row_base, N - k * 8));
        
        // Multiply accumulate: acc += a_frag * ones
        simdgroup_multiply_accumulate(acc, a_frag, ones, acc);
    }

    // Handle tail (remaining columns < 8)
    // We can just load with bounds checking handled by simdgroup_load (hardware clipping or 0-padding?)
    // Metal simdgroup_load safely handles out-of-bounds by reading 0 if 'coord' is provided?
    // Actually, `simdgroup_load` with pointer and stride doesn't auto-pad safely without coord check or careful buffer sizing.
    // For simplicity in this kernel, we use a scalar cleanup loop if N is not multiple of 8,
    // OR we mask the tail.
    // Given the constraints and typical use, let's assume padding or handle with scalar loop.
    // But since we are using matrix ops, we can just load the last partial tile if we are careful.
    // Let's rely on standard practice: process remainders.
    
    // For the remainder, since we are in a simdgroup, we can collaboratively load.
    // However, mixing matrix ops and scalar ops is tricky.
    // Let's assume N is multiple of 8 for the matrix path, or handle remainder manually.
    if (N % 8 != 0) {
        uint k_base = num_blocks * 8;
        for (uint i = 0; i < (N % 8); ++i) {
             // For each of the 8 rows in this simdgroup
             // But simdgroup threads are mapped 32 threads -> 8x8 matrix loosely.
             // It is better to treat the threads as linear.
             // simd_lane 0..31.
             // Lane layout for 8x8 is complex.
             // Let's just do a naive load for the tail into the accumulator?
             // No, cannot inject into opaque object easily.
             // Instead, accumulate tail into a separate register and add later.
             // Or simpler: Just loop the remaining elements scalar-wise after storing the matrix result.
        }
    }

    // Store result to threadgroup memory to extract the first column.
    // 8 rows, 8 columns.
    // Pad the stride to avoid bank conflicts (though 8 is small, 32 banks usually fine with stride 8 if accessing distinct banks).
    // Stride 8 + padding. Using 8 floats = 32 bytes.
    // Using 8+1 floats stride.
    threadgroup float temp_storage[8 * 9]; // 8 rows * 9 cols
    
    simdgroup_store(acc, temp_storage, 9);
    
    // Only the first column of the result matrix is needed (it contains the row sums).
    // Actually, all columns contain the row sums.
    // We need to write 8 values to dst (for the 8 rows).
    // Threads 0..7 can handle this.
    
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 8) {
        uint r = tid;
        if (row_base + r < M) {
            float row_sum = temp_storage[r * 9]; // First column
            
            // Add tail if N % 8 != 0
            if (N % 8 != 0) {
                for (uint c = num_blocks * 8; c < N; ++c) {
                    row_sum += (float)src[(row_base + r) * N + c];
                }
            }
            
            dst[row_base + r] = (half)(row_sum * scale);
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: reduce_cols (Sum/Mean)
// -----------------------------------------------------------------------------
// Reduces matrix of size MxN along the M dimension (columns).
// Output: 1xN vector.
//
// Strategy:
// We iterate over M in chunks of 8.
// We maintain an accumulator of size 8x8 (representing 8 columns, reduced over M rows).
// But wait, if we process 8 rows at a time, we reduce 8 rows into 1 row?
// Yes. Ones * A_chunk = Row_Vector_of_Sums.
// We can accumulate these Row Vectors.
//
// Grid:
// - Threadgroups cover N dimension (chunks of 8 cols).
// - Each simdgroup handles 8 columns.
kernel void reduce_cols(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    device const uint* M_p [[buffer(2)]],
    device const uint* N_p [[buffer(3)]],
    device const float* scale_p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint M = *M_p;
    const uint N = *N_p;
    const float scale = *scale_p;

    // Each simdgroup processes 8 columns.
    const uint col_base = tgid.x * 8;
    
    if (col_base >= N) return;

    simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<half, 8, 8> ones = make_ones_matrix();

    // Iterate over M in chunks of 8
    const uint num_blocks = M / 8;
    for (uint k = 0; k < num_blocks; ++k) {
        simdgroup_matrix<half, 8, 8> a_frag;
        ulong src_offset = (ulong)k * 8 * N + col_base;
        
        // Load 8x8 block
        simdgroup_load(a_frag, src + src_offset, N, ulong2(M - k * 8, N - col_base));
        
        // Multiply accumulate: acc += ones * a_frag
        // ones (8x8) * a_frag (8x8) -> result (8x8)
        // Rows of 'ones' are all 1s.
        // res[r][c] = sum_k(ones[r][k] * a_frag[k][c]) = sum_k(1 * a_frag[k][c])
        // So every row of res contains the column sums of a_frag.
        simdgroup_multiply_accumulate(acc, ones, a_frag, acc);
    }

    // Storage for extraction
    threadgroup float temp_storage[8 * 9];
    simdgroup_store(acc, temp_storage, 9);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Threads 0..7 extract the results for the 8 columns
    // We take the first row (temp_storage[0][c]).
    // Note: acc accumulated the sums into *every* row. So row 0 is sufficient.
    
    if (tid < 8) {
        uint c = tid;
        if (col_base + c < N) {
            float col_sum = temp_storage[c]; // Row 0, Col c
            
            // Handle tail M
            if (M % 8 != 0) {
                for (uint r = num_blocks * 8; r < M; ++r) {
                    col_sum += (float)src[r * N + col_base + c];
                }
            }
            
            dst[col_base + c] = (half)(col_sum * scale);
        }
    }
}
