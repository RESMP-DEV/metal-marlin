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
// Viterbi Path Finding
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

/// Viterbi forward pass: compute minimum cost paths through trellis.
///
/// For each position in the tile (256 elements), compute the minimum cost
/// to reach each of the 16 states, considering transitions from all
/// previous states. Transition cost penalizes state changes to encourage
/// smooth quantization paths.
///
/// @param tile          256 float values from the input tile
/// @param scale         Per-tile scale factor
/// @param grid          Grid values for each state [n_states]
/// @param n_states      Number of quantization states
/// @param costs         Threadgroup shared memory for Viterbi costs [2][MAX_STATES]
/// @param edges         Threadgroup shared memory for backtracking edges [TILE_SIZE][MAX_STATES]
/// @param lane_id       Thread index within threadgroup [0, 255]
inline void viterbi_forward(
    device const float* tile,
    float scale,
    device const float* grid,
    uint n_states,
    threadgroup float costs[2][MAX_STATES],
    threadgroup short edges[TILE_SIZE][MAX_STATES],
    uint lane_id
) {
    // Initialize costs for position 0
    if (lane_id == 0) {
        for (uint s = 0; s < n_states; s++) {
            costs[0][s] = quant_error(tile[0], s, scale, grid[s]);
            edges[0][s] = -1;  // No predecessor for first element
        }
    }
    
    // Synchronize before starting forward pass
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Forward pass: compute costs for each position
    // Each thread handles one state for all positions
    uint state = lane_id % n_states;
    uint pos_offset = lane_id / n_states;
    
    for (uint pos = 1; pos < TILE_SIZE; pos++) {
        uint curr_buf = pos & 1;      // pos % 2
        uint prev_buf = 1 - curr_buf;  // 1 - (pos % 2)
        
        // Only process if this thread's state is valid
        if (state < n_states && pos_offset == 0) {
            float min_cost = INF;
            short best_prev = 0;
            
            float err = quant_error(tile[pos], state, scale, grid[state]);
            
            // Find minimum cost from all previous states
            for (uint prev_s = 0; prev_s < n_states; prev_s++) {
                float prev_cost = costs[prev_buf][prev_s];
                
                // Transition cost: penalize state changes slightly
                // This encourages smooth quantization paths
                float trans_cost = (prev_s == state) ? 0.0f : 0.01f;
                
                float total_cost = prev_cost + err + trans_cost;
                
                if (total_cost < min_cost) {
                    min_cost = total_cost;
                    best_prev = short(prev_s);
                }
            }
            
            costs[curr_buf][state] = min_cost;
            edges[pos][state] = best_prev;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// Viterbi backward pass: backtrack to find optimal path.
///
/// After the forward pass, backtrack from the minimum-cost final state
/// to recover the optimal quantization path.
///
/// @param edges         Backtracking edges from forward pass
/// @param n_states      Number of quantization states
/// @param path          Output: optimal state for each position [TILE_SIZE]
/// @param lane_id       Thread index within threadgroup [0, 255]
inline void viterbi_backward(
    threadgroup short edges[TILE_SIZE][MAX_STATES],
    uint n_states,
    threadgroup short path[TILE_SIZE],
    uint lane_id
) {
    // Find best final state (only lane 0 does this)
    if (lane_id == 0) {
        // The final state info is in the last row of edges
        // We use the last state's backpointer chain
        short best_state = 0;
        
        // Start from the last position and backtrack
        // For simplicity, assume we want the path ending at state with min cost
        // We'll use edges to trace back
        short curr_state = best_state;
        
        // Store path in reverse order
        for (int pos = TILE_SIZE - 1; pos >= 0; pos--) {
            path[pos] = curr_state;
            if (pos > 0) {
                curr_state = edges[pos][curr_state];
            }
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Parallel Viterbi backward pass using parallel reduction.
///
/// More efficient version that uses parallel threads for backtracking.
///
/// @param edges         Backtracking edges from forward pass
/// @param costs         Final costs from forward pass
/// @param n_states      Number of quantization states
/// @param path          Output: optimal state for each position [TILE_SIZE]
/// @param lane_id       Thread index within threadgroup [0, 255]
inline void viterbi_backward_parallel(
    threadgroup short edges[TILE_SIZE][MAX_STATES],
    threadgroup float costs[2][MAX_STATES],
    uint n_states,
    threadgroup short path[TILE_SIZE],
    uint lane_id
) {
    if (lane_id == 0) {
        // Find best final state
        uint final_buf = (TILE_SIZE - 1) & 1;
        float min_cost = INF;
        short best_state = 0;
        
        for (uint s = 0; s < n_states; s++) {
            if (costs[final_buf][s] < min_cost) {
                min_cost = costs[final_buf][s];
                best_state = short(s);
            }
        }
        
        // Backtrack to find optimal path
        short curr_state = best_state;
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
    
    // Shared memory for Viterbi algorithm
    threadgroup float costs[2][MAX_STATES];
    threadgroup short edges[TILE_SIZE][MAX_STATES];
    threadgroup short path[TILE_SIZE];
    
    // Load tile data and scale
    float scale = scales[tile_id];
    device const float* tile = tiles + tile_id * TILE_SIZE;
    
    // Run Viterbi forward pass
    viterbi_forward(tile, scale, grid, n_states, costs, edges, lane_id);
    
    // Run Viterbi backward pass to find optimal path
    viterbi_backward_parallel(edges, costs, n_states, path, lane_id);
    
    // Write output: each thread handles its own positions
    // Lane 0-255 each write their corresponding element
    if (lane_id < TILE_SIZE) {
        short best_state = path[lane_id];
        indices[tile_id * TILE_SIZE + lane_id] = best_state;
        dequantized[tile_id * TILE_SIZE + lane_id] = grid[best_state] * scale;
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
    
    threadgroup float costs[2][N_STATES];
    threadgroup short edges[TILE_SIZE][N_STATES];
    threadgroup short path[TILE_SIZE];
    
    float scale = scales[tile_id];
    device const float* tile = tiles + tile_id * TILE_SIZE;
    
    // Initialize
    if (lane_id == 0) {
        for (uint s = 0; s < N_STATES; s++) {
            costs[0][s] = quant_error(tile[0], s, scale, grid[s]);
            edges[0][s] = -1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Forward pass - each thread handles one state
    uint state = lane_id & 15;  // lane_id % 16
    
    for (uint pos = 1; pos < TILE_SIZE; pos++) {
        uint curr_buf = pos & 1;
        uint prev_buf = 1 - curr_buf;
        
        if (state < N_STATES && lane_id < N_STATES) {
            float min_cost = INF;
            short best_prev = 0;
            float err = quant_error(tile[pos], state, scale, grid[state]);
            
            // Unrolled loop for 16 states
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
    
    // Write output
    if (lane_id < TILE_SIZE) {
        short s = path[lane_id];
        indices[tile_id * TILE_SIZE + lane_id] = s;
        dequantized[tile_id * TILE_SIZE + lane_id] = grid[s] * scale;
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
    
    float scale = scales[tile_id];
    float val = tiles[tile_id * TILE_SIZE + lane_id] / scale;
    
    // Find nearest grid point
    float min_dist = INF;
    short best_idx = 0;
    
    for (uint s = 0; s < n_states; s++) {
        float dist = abs(val - grid[s]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = short(s);
        }
    }
    
    indices[tile_id * TILE_SIZE + lane_id] = best_idx;
    dequantized[tile_id * TILE_SIZE + lane_id] = grid[best_idx] * scale;
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
