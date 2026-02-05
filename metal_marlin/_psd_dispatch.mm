// _psd_dispatch.cpp - C++ Metal dispatch for PSD projection with prefetch

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <dispatch/dispatch.h>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <queue>
#include <thread>

namespace py = pybind11;

// Embedded Metal shader source (avoids path resolution issues)
static const char *PSD_SHADER_SOURCE = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint PSD_TILE = 32;

kernel void cholesky_inplace(
    device float* H             [[buffer(0)]],
    device atomic_int* success  [[buffer(1)]],
    constant uint& dim          [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    
    atomic_store_explicit(success, 1, memory_order_relaxed);
    
    for (uint j = 0; j < dim; ++j) {
        float sum = 0.0f;
        for (uint k = 0; k < j; ++k) {
            float val = H[j * dim + k];
            sum += val * val;
        }
        float diag = H[j * dim + j] - sum;
        
        if (diag <= 0.0f) {
            atomic_store_explicit(success, 0, memory_order_relaxed);
            return;
        }
        
        H[j * dim + j] = sqrt(diag);
        float inv_ljj = 1.0f / H[j * dim + j];
        
        for (uint i = j + 1; i < dim; ++i) {
            float sum_ij = 0.0f;
            for (uint k = 0; k < j; ++k) {
                sum_ij += H[i * dim + k] * H[j * dim + k];
            }
            H[i * dim + j] = (H[i * dim + j] - sum_ij) * inv_ljj;
        }
        
        for (uint i = 0; i < j; ++i) {
            H[i * dim + j] = 0.0f;
        }
    }
}

kernel void add_diagonal(
    device float* H          [[buffer(0)]],
    constant float& lambda   [[buffer(1)]],
    constant uint& dim       [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    if (tid < dim) {
        H[tid * dim + tid] += lambda;
    }
}

kernel void reconstruct_from_cholesky(
    device const float* L    [[buffer(0)]],
    device float* H_psd      [[buffer(1)]],
    constant uint& dim       [[buffer(2)]],
    uint2 tid                [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;
    
    if (row >= dim || col >= dim) return;
    
    float sum = 0.0f;
    uint max_k = min(row, col) + 1;
    
    for (uint k = 0; k < max_k; ++k) {
        sum += L[row * dim + k] * L[col * dim + k];
    }
    
    H_psd[row * dim + col] = sum;
}

kernel void matrix_copy(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& size      [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    if (tid < size) {
        dst[tid] = src[tid];
    }
}
)metal";

// Prefetch buffer for next layer's activations
struct PrefetchState {
  std::unique_ptr<float[]> buffer;
  size_t size = 0;
  bool ready = false;
  std::mutex mtx;
  std::condition_variable cv;
};

static PrefetchState g_prefetch;

// Background prefetch thread (works on raw buffer, no Python objects)
void prefetch_copy_thread(std::unique_ptr<float[]> data, size_t n_elements) {
  std::lock_guard<std::mutex> lock(g_prefetch.mtx);
  g_prefetch.buffer = std::move(data);
  g_prefetch.size = n_elements;
  g_prefetch.ready = true;
  g_prefetch.cv.notify_all();
}

void start_prefetch(py::array_t<float> arr) {
  // Reset state
  {
    std::lock_guard<std::mutex> lock(g_prefetch.mtx);
    g_prefetch.ready = false;
  }

  // Copy data to raw buffer BEFORE releasing GIL
  auto info = arr.request();
  size_t n_elements = info.size;
  auto data = std::make_unique<float[]>(n_elements);
  std::memcpy(data.get(), info.ptr, n_elements * sizeof(float));

  // Launch background thread with raw data (no Python objects)
  std::thread(prefetch_copy_thread, std::move(data), n_elements).detach();
}

py::array_t<float> get_prefetched() {
  std::unique_lock<std::mutex> lock(g_prefetch.mtx);
  g_prefetch.cv.wait(lock, [] { return g_prefetch.ready; });

  // Transfer ownership
  auto result = py::array_t<float>(g_prefetch.size);
  std::memcpy(result.mutable_data(), g_prefetch.buffer.get(),
              g_prefetch.size * sizeof(float));

  g_prefetch.buffer.reset();
  g_prefetch.ready = false;

  return result;
}

// Metal PSD projection dispatch
py::array_t<float> psd_project_metal(py::array_t<float, py::array::c_style> H,
                                     float sigma_reg, int max_iters) {
  auto info = H.request();
  if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
    throw std::runtime_error("H must be square matrix");
  }

  uint32_t dim = static_cast<uint32_t>(info.shape[0]);
  size_t matrix_bytes = dim * dim * sizeof(float);

  // Get Metal device
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      throw std::runtime_error("Metal not available");
    }

    // Compile shader from embedded source (no file path issues)
    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:PSD_SHADER_SOURCE];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:options
                                                    error:&error];
    if (!library) {
      std::string err_msg = "Shader compilation failed";
      if (error) {
        err_msg += ": ";
        err_msg += [[error localizedDescription] UTF8String];
      }
      throw std::runtime_error(err_msg);
    }

    // Get kernels
    id<MTLFunction> fn_cholesky =
        [library newFunctionWithName:@"cholesky_inplace"];
    id<MTLFunction> fn_add_diag = [library newFunctionWithName:@"add_diagonal"];
    id<MTLFunction> fn_reconstruct =
        [library newFunctionWithName:@"reconstruct_from_cholesky"];
    id<MTLFunction> fn_copy = [library newFunctionWithName:@"matrix_copy"];

    id<MTLComputePipelineState> pso_cholesky =
        [device newComputePipelineStateWithFunction:fn_cholesky error:&error];
    id<MTLComputePipelineState> pso_add_diag =
        [device newComputePipelineStateWithFunction:fn_add_diag error:&error];
    id<MTLComputePipelineState> pso_reconstruct =
        [device newComputePipelineStateWithFunction:fn_reconstruct
                                              error:&error];
    id<MTLComputePipelineState> pso_copy =
        [device newComputePipelineStateWithFunction:fn_copy error:&error];

    // Allocate buffers
    id<MTLBuffer> buf_H =
        [device newBufferWithLength:matrix_bytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_H_work =
        [device newBufferWithLength:matrix_bytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_success =
        [device newBufferWithLength:sizeof(int32_t)
                            options:MTLResourceStorageModeShared];

    // Copy input
    std::memcpy(buf_H.contents, info.ptr, matrix_bytes);

    id<MTLCommandQueue> queue = [device newCommandQueue];

    float lambda = sigma_reg;
    bool success = false;

    for (int iter = 0; iter < max_iters && !success; ++iter) {
      // Copy H to H_work
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      [enc setComputePipelineState:pso_copy];
      [enc setBuffer:buf_H offset:0 atIndex:0];
      [enc setBuffer:buf_H_work offset:0 atIndex:1];
      uint32_t total_elems = dim * dim;
      [enc setBytes:&total_elems length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(total_elems, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];

      // Add diagonal if iter > 0
      if (iter > 0) {
        cmd = [queue commandBuffer];
        enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_add_diag];
        [enc setBuffer:buf_H_work offset:0 atIndex:0];
        [enc setBytes:&lambda length:sizeof(float) atIndex:1];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
      }

      // Reset success flag
      *(int32_t *)buf_success.contents = 1;

      // Try Cholesky
      cmd = [queue commandBuffer];
      enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pso_cholesky];
      [enc setBuffer:buf_H_work offset:0 atIndex:0];
      [enc setBuffer:buf_success offset:0 atIndex:1];
      [enc setBytes:&dim length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];

      success = (*(int32_t *)buf_success.contents == 1);
      lambda *= 2.0f; // Exponential backoff
    }

    if (!success) {
      throw std::runtime_error("PSD projection failed after max iterations");
    }

    // Reconstruct H_psd from L
    id<MTLBuffer> buf_H_psd =
        [device newBufferWithLength:matrix_bytes
                            options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso_reconstruct];
    [enc setBuffer:buf_H_work offset:0 atIndex:0]; // L
    [enc setBuffer:buf_H_psd offset:0 atIndex:1];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:2];

    // Dispatch one thread per matrix element (2D grid)
    [enc dispatchThreads:MTLSizeMake(dim, dim, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Copy result
    auto result = py::array_t<float>({dim, dim});
    std::memcpy(result.mutable_data(), buf_H_psd.contents, matrix_bytes);

    return result;
  }
}

// Fast check if matrix is diagonally dominant (likely PSD)
// Returns true if H[i,i] >= sum(|H[i,j]| for j!=i) for all i
bool is_diagonally_dominant(py::array_t<float, py::array::c_style> H) {
  auto info = H.request();
  if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
    return false;
  }

  const float *data = static_cast<const float *>(info.ptr);
  size_t dim = info.shape[0];

  for (size_t i = 0; i < dim; ++i) {
    float diag = std::abs(data[i * dim + i]);
    float off_diag_sum = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
      if (i != j) {
        off_diag_sum += std::abs(data[i * dim + j]);
      }
    }
    if (diag < off_diag_sum) {
      return false;
    }
  }
  return true;
}

// Estimate minimum eigenvalue via Gershgorin circles (fast, O(N^2))
// Returns lower bound on smallest eigenvalue
float gershgorin_min_eigenvalue(py::array_t<float, py::array::c_style> H) {
  auto info = H.request();
  if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
    return -1e10f;
  }

  const float *data = static_cast<const float *>(info.ptr);
  size_t dim = info.shape[0];

  float min_bound = std::numeric_limits<float>::max();
  for (size_t i = 0; i < dim; ++i) {
    float diag = data[i * dim + i];
    float radius = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
      if (i != j) {
        radius += std::abs(data[i * dim + j]);
      }
    }
    float lower_bound = diag - radius;
    min_bound = std::min(min_bound, lower_bound);
  }
  return min_bound;
}

// Quick PSD check: if Gershgorin lower bound > 0, matrix is PSD
// This is O(N^2) instead of O(N^3) eigendecomposition
bool is_likely_psd(py::array_t<float, py::array::c_style> H,
                   float tolerance = 0.0f) {
  return gershgorin_min_eigenvalue(H) > tolerance;
}

PYBIND11_MODULE(_psd_dispatch, m) {
  m.def("psd_project_metal", &psd_project_metal,
        "PSD projection via iterative Cholesky", py::arg("H"),
        py::arg("sigma_reg") = 0.01f, py::arg("max_iters") = 10);
  m.def("start_prefetch", &start_prefetch,
        "Start background prefetch of activations");
  m.def("get_prefetched", &get_prefetched,
        "Get prefetched activations (blocks until ready)");
  m.def("is_diagonally_dominant", &is_diagonally_dominant,
        "Check if matrix is diagonally dominant");
  m.def("gershgorin_min_eigenvalue", &gershgorin_min_eigenvalue,
        "Estimate minimum eigenvalue via Gershgorin circles");
  m.def("is_likely_psd", &is_likely_psd,
        "Quick O(N^2) check if matrix is likely PSD", py::arg("H"),
        py::arg("tolerance") = 0.0f);
}
