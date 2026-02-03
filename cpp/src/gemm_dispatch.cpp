// gemm_dispatch.cpp
// Port of marlin_gemm dispatch logic from Python to C++

#include <cstdint>
#include <stdexcept>
#include <string>

// Constants matching Python implementation
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int THREADS_PER_TG = 128;
constexpr int PAD_MULTIPLE = 64;
constexpr int FP32_ACCUM_K_THRESHOLD = 8192;

namespace metal_marlin {

// GPU family detection (simplified - would need actual Metal device query)
inline int get_gpu_family(void* device) {
    // Placeholder: Real implementation would query Metal device properties
    // Apple M1/M2 = family 7-8, M3+ = family 9+
    return 9; // Assume modern GPU for now
}

// Round up to multiple
inline int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// Kernel name selection based on parameters
std::string select_kernel_fp4(int K, int gpu_family) {
    if (K > FP32_ACCUM_K_THRESHOLD) {
        return gpu_family >= 9 ? "marlin_gemm_fused_fp4_fp32acc" : "marlin_gemm_fp4_fp32acc";
    }
    return gpu_family >= 9 ? "marlin_gemm_fused_fp4" : "marlin_gemm_fp4";
}

// Padding configuration
struct PaddingConfig {
    int pad_m_multiple;
    int pad_n_multiple;
    int pad_k_multiple;
};

PaddingConfig get_padding_config(const std::string& kernel_name, int group_size) {
    PaddingConfig config;
    config.pad_k_multiple = std::max(PAD_MULTIPLE, group_size);
    
    if (kernel_name == "marlin_gemm_fused_fp4") {
        // Fused kernel requires tile-aligned padding
        config.pad_m_multiple = std::max(PAD_MULTIPLE, TILE_M);
        config.pad_n_multiple = std::max(PAD_MULTIPLE, TILE_N);
    } else {
        config.pad_m_multiple = PAD_MULTIPLE;
        config.pad_n_multiple = PAD_MULTIPLE;
    }
    
    return config;
}

// Compute grid dimensions
struct GridDims {
    int grid_m;
    int grid_n;
    int grid_z;
};

GridDims compute_grid(int M, int N) {
    GridDims grid;
    grid.grid_m = (M + TILE_M - 1) / TILE_M;
    grid.grid_n = (N + TILE_N - 1) / TILE_N;
    grid.grid_z = 1;
    return grid;
}

// FP4 GEMM dispatch configuration
struct GemmFp4Config {
    std::string kernel_name;
    int M_padded;
    int N_padded;
    int K_padded;
    int pad_m;
    int pad_n;
    GridDims grid;
    
    GemmFp4Config(int M, int N, int K, int group_size, int gpu_family, bool enable_padding) {
        // Select kernel
        kernel_name = select_kernel_fp4(K, gpu_family);
        
        // Get padding config
        PaddingConfig pad_cfg = get_padding_config(kernel_name, group_size);
        
        // Compute padded dimensions
        if (enable_padding) {
            M_padded = round_up(M, pad_cfg.pad_m_multiple);
            N_padded = round_up(N, pad_cfg.pad_n_multiple);
            K_padded = round_up(K, pad_cfg.pad_k_multiple);
            pad_m = M_padded - M;
            pad_n = N_padded - N;
        } else {
            M_padded = M;
            N_padded = N;
            K_padded = K;
            pad_m = 0;
            pad_n = 0;
        }
        
        // Compute grid
        grid = compute_grid(M_padded, N_padded);
    }
};

// FP8 GEMM dispatch configuration
struct GemmFp8Config {
    std::string kernel_name;
    int M_padded;
    int N_padded;
    int K_padded;
    int pad_m;
    int pad_n;
    GridDims grid;
    
    GemmFp8Config(int M, int N, int K, int group_size, bool enable_padding) {
        kernel_name = "marlin_gemm_fp8_e4m3";
        
        if (enable_padding) {
            M_padded = round_up(M, PAD_MULTIPLE);
            N_padded = round_up(N, std::max(PAD_MULTIPLE, 4));
            K_padded = round_up(K, std::max(PAD_MULTIPLE, group_size));
            pad_m = M_padded - M;
            pad_n = N_padded - N;
        } else {
            M_padded = M;
            N_padded = N;
            K_padded = K;
            pad_m = 0;
            pad_n = 0;
        }
        
        grid = compute_grid(M_padded, N_padded);
    }
};

// INT2 GEMM dispatch configuration
struct GemmInt2Config {
    std::string kernel_name;
    int M_padded;
    int N_padded;
    int K_padded;
    int pad_m;
    int pad_n;
    GridDims grid;
    
    GemmInt2Config(int M, int N, int K, int group_size, bool enable_padding) {
        kernel_name = "marlin_gemm_int2";
        
        if (enable_padding) {
            M_padded = round_up(M, PAD_MULTIPLE);
            N_padded = round_up(N, std::max(PAD_MULTIPLE, 16));
            K_padded = round_up(K, std::max(PAD_MULTIPLE, group_size));
            pad_m = M_padded - M;
            pad_n = N_padded - N;
        } else {
            M_padded = M;
            N_padded = N;
            K_padded = K;
            pad_m = 0;
            pad_n = 0;
        }
        
        grid = compute_grid(M_padded, N_padded);
    }
};

// Validate tensor shapes
void validate_fp4_shapes(int M, int N, int K, int B_rows, int B_cols, int S_rows, int S_cols, int group_size) {
    // B_packed is [K/8, N] for FP4 (8 values per uint32)
    int expected_B_rows = (K + 7) / 8;
    if (B_rows != expected_B_rows) {
        throw std::runtime_error("FP4 packed tensor K dimension mismatch: expected " + 
                                 std::to_string(expected_B_rows) + ", got " + std::to_string(B_rows));
    }
    if (B_cols != N) {
        throw std::runtime_error("FP4 packed tensor N dimension mismatch: expected " + 
                                 std::to_string(N) + ", got " + std::to_string(B_cols));
    }
    
    // Scales is [K/group_size, N]
    int expected_S_rows = (K + group_size - 1) / group_size;
    if (S_rows != expected_S_rows) {
        throw std::runtime_error("Scales K dimension mismatch: expected " + 
                                 std::to_string(expected_S_rows) + ", got " + std::to_string(S_rows));
    }
    if (S_cols != N) {
        throw std::runtime_error("Scales N dimension mismatch: expected " + 
                                 std::to_string(N) + ", got " + std::to_string(S_cols));
    }
}

} // namespace metal_marlin
