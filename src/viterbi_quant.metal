// viterbi_quant.metal - Parallel trellis quantization using Viterbi algorithm
// ============================================================================
//
// Metal kernel for parallel Viterbi-based quantization of weight tiles.
// Each threadgroup processes one 16x16 tile (256 elements) independently.
//
// The Viterbi algorithm finds the optimal quantization path through a trellis
// where each state represents a quantized value. For 4-bit quantization,
// we have 16 states (0-15), and for each element we compute the minimum-cost
// path from all previous states.
//
// Reference: CUDA kernel structure from ExllamaV3 ext.quantize_tiles()
//
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant constexpr uint TILE_SIZE = 256;      // 16x16 tile
constant constexpr uint TILE_DIM = 16;        // 16x16 dimensions
constant constexpr uint MAX_STATES = 16;      // 4-bit = 16 states
constant constexpr float INF = 1e20f;         // Infinity for Viterbi costs

// ============================================================================
// Helper Structures
// ============================================================================

/// Return type for simd_reduce_min
struct MinResult {
    float error;
    short state;
};

// ============================================================================
// Viterbi Path Finding - Optimized with Shared Memory Trellis States
// ============================================================================

/// Compute squared error between original value and quantized value.
///
/// @param original   Original float value from the tile
/// @param state      Quantized state (0-15)
/// @param scale      Per-tile scale factor
/// @param grid_val   Grid value for this state (dequantized center)
/// @return           Squared quantization error
inline float quant_error(float original, uint state, float scale, float grid_val) {
    float dequant = grid_val * scale;
    float diff = original - dequant;
    return diff * diff;
}

/// SIMD-based parallel min reduction for finding best state.
///
/// Uses quad_shuffle_x4 to reduce across 4-thread groups, then
/// simd_shuffle across the entire wave to find the global minimum.
///
/// @param err       Array of 4 errors (one per thread in quad)
/// @param state     Array of 4 corresponding states
/// @param lane_id   Thread index in wave [0-63]
/// @return          MinResult with (min_error, best_state) across the wave
inline MinResult simd_reduce_min(
    float4 err4,
    short4 state4,
    uint lane_id
) {
    // Step 1: Reduce within each 4-thread quad
    uint quad_id = lane_id & 0x3;  // lane_id % 4
    uint quad_base = lane_id & ~0x3;

    // Quad shuffle: get min across 4 threads
    float err_min = err4.x;
    short state_min = state4.x;
    for (uint i = 1; i < 4; i++) {
        if (err4[i] < err_min) {
            err_min = err4[i];
            state_min = state4[i];
        }
    }

    // Broadcast the quad minimum to all threads in the quad
    // Note: In Metal, we'd use simd_broadcast here

    // Step 2: Reduce across the wave using shuffle-based tree
    // Wave size is 32 or 64 on Apple Silicon
    // We'll use simd_shuffle for this

    float wave_min = err_min;
    short wave_state = state_min;

    // Tree reduction: compare with lane ^ 1, ^ 2, ^ 4, ^ 8, ^ 16
    for (uint offset = 1; offset < 32; offset <<= 1) {
        float other_min = simd_shuffle(wave_min, lane_id ^ offset);
        short other_state = simd_shuffle(wave_state, lane_id ^ offset);
        if (other_min < wave_min) {
            wave_min = other_min;
            wave_state = other_state;
        }
    }

    MinResult result;
    result.error = wave_min;
    result.state = wave_state;
    return result;
}

/// Optimized Viterbi forward pass using shared memory trellis states.
///
/// This version:
/// 1. Uses double-buffered shared memory for trellis states [2][MAX_STATES]
/// 2. Stores backpointers in compact shared memory edges[TILE_SIZE][MAX_STATES]
/// 3. Parallelizes state computation across threads in each position
///
/// @param tile_shared    Shared memory copy of tile data [TILE_SIZE]
/// @param grid_shared    Shared memory copy of grid values [MAX_STATES]
/// @param scale          Per-tile scale factor
/// @param n_states       Number of quantization states
/// @param costs          Double-buffered trellis costs [2][MAX_STATES]
/// @param edges          Backtracking edges [TILE_SIZE][MAX_STATES]
/// @param lane_id        Thread index within threadgroup [0, 255]
void viterbi_forward_optimized(
    threadgroup const float* tile_shared,
    threadgroup const float* grid_shared,
    float scale,
    uint n_states,
    threadgroup float costs[2][MAX_STATES],
    threadgroup short edges[TILE_SIZE][MAX_STATES],
    uint lane_id
) {
    // Lane 0 initializes the first position (pos=0)
    if (lane_id == 0) {
        float first_val = tile_shared[0];
        for (uint s = 0; s < n_states; s++) {
            float dequant = grid_shared[s] * scale;
            float diff = first_val - dequant;
            costs[0][s] = diff * diff;
            edges[0][s] = -1;  // No predecessor for first position
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward pass for positions 1 to TILE_SIZE-1
    // Use one thread per state (only first n_states lanes are active).
    uint state = lane_id;
    float state_dequant = 0.0f;
    if (state < n_states) {
        state_dequant = grid_shared[state] * scale;
    }

    for (uint pos = 1; pos < TILE_SIZE; pos++) {
        uint curr_buf = pos & 1;
        uint prev_buf = 1 - curr_buf;

        if (state < n_states) {
            float min_cost = INF;
            short best_prev = 0;
            float val = tile_shared[pos];
            float diff = val - state_dequant;
            float err = diff * diff;

            // Load previous costs from shared memory
            threadgroup const float* prev_costs = costs[prev_buf];
            for (uint prev_s = 0; prev_s < n_states; prev_s++) {
                float prev_cost = prev_costs[prev_s];
                float trans_cost = (prev_s == state) ? 0.0f : 0.01f;
                float total = prev_cost + err + trans_cost;

                if (total < min_cost) {
                    min_cost = total;
                    best_prev = short(prev_s);
                }
            }

            costs[curr_buf][state] = min_cost;
            edges[pos][state] = best_prev;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// Optimized Viterbi backward pass with parallel final state search.
///
/// After the forward pass, backtrack from the minimum-cost final state.
/// Uses shared memory to find the best final state efficiently.
///
/// @param edges         Backtracking edges from forward pass
/// @param costs         Final costs from forward pass
/// @param n_states      Number of quantization states
/// @param path          Output: optimal state for each position [TILE_SIZE]
/// @param reduce_costs  Temporary storage for reduction [TILE_SIZE]
/// @param reduce_states Temporary storage for reduction [TILE_SIZE]
/// @param lane_id       Thread index within threadgroup [0, 255]
void viterbi_backward_parallel(
    threadgroup short edges[TILE_SIZE][MAX_STATES],
    threadgroup float costs[2][MAX_STATES],
    uint n_states,
    threadgroup short path[TILE_SIZE],
    threadgroup float reduce_costs[TILE_SIZE],
    threadgroup short reduce_states[TILE_SIZE],
    uint lane_id
) {
    // Use threadgroup reduction to find best final state
    float local_min = INF;
    short local_state = 0;
    uint final_buf = (TILE_SIZE - 1) & 1;

    // Each thread checks some states (load balancing)
    for (uint s = lane_id; s < n_states; s += TILE_SIZE) {
        float cost = costs[final_buf][s];
        if (cost < local_min) {
            local_min = cost;
            local_state = short(s);
        }
    }

    reduce_costs[lane_id] = local_min;
    reduce_states[lane_id] = local_state;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane_id < stride) {
            float other_min = reduce_costs[lane_id + stride];
            short other_state = reduce_states[lane_id + stride];
            if (other_min < reduce_costs[lane_id]) {
                reduce_costs[lane_id] = other_min;
                reduce_states[lane_id] = other_state;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane 0 performs the backtracking
    if (lane_id == 0) {
        short curr_state = reduce_states[0];
        for (int pos = TILE_SIZE - 1; pos >= 0; pos--) {
            path[pos] = curr_state;
            if (pos > 0) {
                curr_state = edges[pos][curr_state];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// Main Kernel
// ============================================================================

/// Quantize tiles using parallel Viterbi algorithm.
///
/// Each threadgroup processes one 16x16 tile independently. The Viterbi
/// algorithm finds the optimal quantization path that minimizes total
/// quantization error while encouraging smooth state transitions.
///
/// Buffer layout:
///   - tiles: [n_tiles, 256] float input tiles
///   - scales: [n_tiles] per-tile scale factors
///   - grid: [n_states] quantization grid values
///   - indices: [n_tiles, 256] output quantized indices (short)
///   - dequantized: [n_tiles, 256] output dequantized values
///
/// @param tiles         Input tiles [n_tiles, TILE_SIZE]
/// @param scales        Per-tile scale factors [n_tiles]
/// @param grid          Quantization grid values [n_states]
/// @param indices       Output quantized indices [n_tiles, TILE_SIZE]
/// @param dequantized   Output dequantized values [n_tiles, TILE_SIZE]
/// @param n_tiles       Total number of tiles
/// @param n_states      Number of quantization states
/// @param tile_id       Threadgroup index (which tile to process)
/// @param lane_id       Thread index within threadgroup [0, 255]
kernel void quantize_tiles_viterbi(
    device const float* tiles [[buffer(0)]],       // [n_tiles, 256]
    device const float* scales [[buffer(1)]],      // [n_tiles]
    device const float* grid [[buffer(2)]],        // [n_states]
    device short* indices [[buffer(3)]],           // [n_tiles, 256]
    device float* dequantized [[buffer(4)]],       // [n_tiles, 256]
    constant uint& n_tiles [[buffer(5)]],
    constant uint& n_states [[buffer(6)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    if (tile_id >= n_tiles) return;

    // Shared memory: tile data, grid, Viterbi costs, edges, and path
    threadgroup float tile_shared[TILE_SIZE];
    threadgroup float grid_shared[MAX_STATES];
    threadgroup float costs[2][MAX_STATES];
    threadgroup short edges[TILE_SIZE][MAX_STATES];
    threadgroup short path[TILE_SIZE];
    threadgroup float reduce_costs[TILE_SIZE];
    threadgroup short reduce_states[TILE_SIZE];

    // Load tile data into shared memory (coalesced loads)
    device const float* tile_device = tiles + tile_id * TILE_SIZE;
    for (uint i = lane_id; i < TILE_SIZE; i += 256) {
        tile_shared[i] = tile_device[i];
    }

    // Load grid into shared memory (each thread loads what it can)
    for (uint i = lane_id; i < n_states; i += 256) {
        grid_shared[i] = grid[i];
    }

    // Synchronize after loading shared data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load scale
    float scale = scales[tile_id];

    // Run Viterbi forward pass with shared memory data
    viterbi_forward_optimized(tile_shared, grid_shared, scale, n_states, costs, edges, lane_id);

    // Run Viterbi backward pass to find optimal path
    viterbi_backward_parallel(edges, costs, n_states, path, reduce_costs, reduce_states, lane_id);

    // Write output: each thread handles its own positions
    // Lane 0-255 each write their corresponding element
    if (lane_id < TILE_SIZE) {
        short best_state = path[lane_id];
        indices[tile_id * TILE_SIZE + lane_id] = best_state;
        dequantized[tile_id * TILE_SIZE + lane_id] = grid_shared[best_state] * scale;
    }
}

// ============================================================================
// Optimized Variants
// ============================================================================

/// Simplified Viterbi quantization for 16-state (4-bit) tiles.
///
/// Optimized version that assumes exactly 16 states and uses
/// fixed-size loops for better compiler optimization.
///
/// @param tiles         Input tiles [n_tiles, TILE_SIZE]
/// @param scales        Per-tile scale factors [n_tiles]
/// @param grid          Quantization grid values [16]
/// @param indices       Output quantized indices [n_tiles, TILE_SIZE]
/// @param dequantized   Output dequantized values [n_tiles, TILE_SIZE]
/// @param n_tiles       Total number of tiles
/// @param tile_id       Threadgroup index
/// @param lane_id       Thread index within threadgroup
kernel void quantize_tiles_viterbi_u4(
    device const float* tiles [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device short* indices [[buffer(3)]],
    device float* dequantized [[buffer(4)]],
    constant uint& n_tiles [[buffer(5)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    if (tile_id >= n_tiles) return;

    const uint N_STATES = 16;

    // Shared memory for tile, grid, costs, edges, and path
    threadgroup float tile_shared[TILE_SIZE];
    threadgroup float grid_shared[N_STATES];
    threadgroup float costs[2][N_STATES];
    threadgroup short edges[TILE_SIZE][N_STATES];
    threadgroup short path[TILE_SIZE];

    // Load tile data into shared memory (coalesced loads)
    device const float* tile_device = tiles + tile_id * TILE_SIZE;
    for (uint i = lane_id; i < TILE_SIZE; i += 256) {
        tile_shared[i] = tile_device[i];
    }

    // Load grid into shared memory (only first 16 threads needed)
    if (lane_id < N_STATES) {
        grid_shared[lane_id] = grid[lane_id];
    }

    // Synchronize after loading shared data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = scales[tile_id];

    // Initialize using shared memory
    if (lane_id == 0) {
        float first_val = tile_shared[0];
        for (uint s = 0; s < N_STATES; s++) {
            float dequant = grid_shared[s] * scale;
            float diff = first_val - dequant;
            costs[0][s] = diff * diff;
            edges[0][s] = -1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward pass - each thread handles one state
    uint state = lane_id & 15;  // lane_id % 16

    // Pre-compute this state's dequantized value
    float state_dequant = grid_shared[state] * scale;

    for (uint pos = 1; pos < TILE_SIZE; pos++) {
        uint curr_buf = pos & 1;
        uint prev_buf = 1 - curr_buf;

        if (state < N_STATES && lane_id < N_STATES) {
            float min_cost = INF;
            short best_prev = 0;
            float val = tile_shared[pos];
            float diff = val - state_dequant;
            float err = diff * diff;

            // Unrolled loop for 16 states using shared memory costs
            #pragma unroll
            for (uint prev_s = 0; prev_s < N_STATES; prev_s++) {
                float prev_cost = costs[prev_buf][prev_s];
                float trans_cost = (prev_s == state) ? 0.0f : 0.01f;
                float total = prev_cost + err + trans_cost;

                if (total < min_cost) {
                    min_cost = total;
                    best_prev = short(prev_s);
                }
            }

            costs[curr_buf][state] = min_cost;
            edges[pos][state] = best_prev;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Backward pass
    if (lane_id == 0) {
        uint final_buf = (TILE_SIZE - 1) & 1;
        float min_cost = INF;
        short best_state = 0;

        for (uint s = 0; s < N_STATES; s++) {
            if (costs[final_buf][s] < min_cost) {
                min_cost = costs[final_buf][s];
                best_state = short(s);
            }
        }

        short curr_state = best_state;
        for (int pos = TILE_SIZE - 1; pos >= 0; pos--) {
            path[pos] = curr_state;
            if (pos > 0) {
                curr_state = edges[pos][curr_state];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output using shared grid
    if (lane_id < TILE_SIZE) {
        short s = path[lane_id];
        indices[tile_id * TILE_SIZE + lane_id] = s;
        dequantized[tile_id * TILE_SIZE + lane_id] = grid_shared[s] * scale;
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================

/// Simple per-element quantization (for comparison/baseline).
///
/// Quantizes each element independently using round-to-nearest.
/// This is faster but doesn't consider inter-element correlations.
///
/// @param tiles         Input tiles [n_tiles, TILE_SIZE]
/// @param scales        Per-tile scale factors [n_tiles]
/// @param grid          Quantization grid values [n_states]
/// @param indices       Output quantized indices [n_tiles, TILE_SIZE]
/// @param dequantized   Output dequantized values [n_tiles, TILE_SIZE]
/// @param n_tiles       Total number of tiles
/// @param n_states      Number of quantization states
/// @param tile_id       Threadgroup index
/// @param lane_id       Thread index within threadgroup
kernel void quantize_tiles_naive(
    device const float* tiles [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device short* indices [[buffer(3)]],
    device float* dequantized [[buffer(4)]],
    constant uint& n_tiles [[buffer(5)]],
    constant uint& n_states [[buffer(6)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    if (tile_id >= n_tiles || lane_id >= TILE_SIZE) return;

    // Shared memory for grid (shared across threads in threadgroup)
    threadgroup float grid_shared[MAX_STATES];

    // Load grid into shared memory (all threads participate for efficiency)
    for (uint i = lane_id; i < n_states; i += 256) {
        grid_shared[i] = grid[i];
    }

    // Synchronize to ensure grid is fully loaded
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = scales[tile_id];
    float val = tiles[tile_id * TILE_SIZE + lane_id] / scale;

    // Find nearest grid point using shared memory
    float min_dist = INF;
    short best_idx = 0;

    for (uint s = 0; s < n_states; s++) {
        float dist = abs(val - grid_shared[s]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = short(s);
        }
    }

    indices[tile_id * TILE_SIZE + lane_id] = best_idx;
    dequantized[tile_id * TILE_SIZE + lane_id] = grid_shared[best_idx] * scale;
}

/// Compute quantization error for a tile.
///
/// Useful for evaluating quantization quality.
///
/// @param original      Original tile values [n_tiles, TILE_SIZE]
/// @param dequantized   Dequantized values [n_tiles, TILE_SIZE]
/// @param errors        Output: per-element squared errors [n_tiles, TILE_SIZE]
/// @param n_tiles       Total number of tiles
/// @param tile_id       Threadgroup index
/// @param lane_id       Thread index within threadgroup
kernel void compute_quant_error(
    device const float* original [[buffer(0)]],
    device const float* dequantized [[buffer(1)]],
    device float* errors [[buffer(2)]],
    constant uint& n_tiles [[buffer(3)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_threadgroup]]
) {
    if (tile_id >= n_tiles || lane_id >= TILE_SIZE) return;
    
    uint idx = tile_id * TILE_SIZE + lane_id;
    float diff = original[idx] - dequantized[idx];
    errors[idx] = diff * diff;
}
