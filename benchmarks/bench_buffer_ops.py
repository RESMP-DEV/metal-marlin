
import time

import Metal
import torch

from metal_marlin._buffer_pool import MetalBufferPool


def benchmark_buffer_ops():
    print("Benchmarking Buffer Operations...")

    # Setup
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        print("Error: Metal device not available.")
        return

    pool = MetalBufferPool(device)

    # Sizes to benchmark
    sizes = [
        (1024, "1KB"),
        (1024 * 1024, "1MB"),
        (100 * 1024 * 1024, "100MB")
    ]

    # Warmup
    torch.empty(1024, device='mps')
    pool.get(1024)
    pool.release(pool.get(1024))

    for size_bytes, size_label in sizes:
        print(f"\n--- Size: {size_label} ---")

        # 1. Creation Time
        print("Creation Time:")

        # Torch MPS
        start = time.perf_counter()
        t = torch.empty(size_bytes, dtype=torch.uint8, device='mps')
        torch.mps.synchronize() # Ensure it's allocated
        end = time.perf_counter()
        print(f"  torch.empty(mps): {(end - start) * 1e6:.2f} us")

        # Raw Metal
        start = time.perf_counter()
        buf = device.newBufferWithLength_options_(size_bytes, Metal.MTLResourceStorageModeShared)
        end = time.perf_counter()
        print(f"  Metal.newBuffer:  {(end - start) * 1e6:.2f} us")

        # Pool (First alloc - miss)
        # Ensure we are testing a miss by asking for a size we haven't released yet (or just flush pool)
        # But for simplicity, we just measure 'get' which might be hit or miss.
        # To test miss, we should clear pool or ensure unique size/fresh pool.
        # Let's use a fresh pool for 'Miss' test.
        temp_pool = MetalBufferPool(device)
        start = time.perf_counter()
        p_buf = temp_pool.get(size_bytes)
        end = time.perf_counter()
        print(f"  Pool.get (Miss):  {(end - start) * 1e6:.2f} us")

        # Pool (Second alloc - hit)
        temp_pool.release(p_buf)
        start = time.perf_counter()
        p_buf_hit = temp_pool.get(size_bytes)
        end = time.perf_counter()
        print(f"  Pool.get (Hit):   {(end - start) * 1e6:.2f} us")


        # 2. Copy/Transfer Time (Host -> Device)
        print("Copy Time (Host -> Device):")
        data = b'\x00' * size_bytes

        # Torch
        cpu_tensor = torch.zeros(size_bytes, dtype=torch.uint8)
        # Sync before
        torch.mps.synchronize()
        start = time.perf_counter()
        gpu_tensor = cpu_tensor.to('mps')
        torch.mps.synchronize()
        end = time.perf_counter()
        print(f"  torch.to(mps):    {(end - start) * 1e6:.2f} us")

        # Metal (newBufferWithBytes) - effectively creation + copy
        start = time.perf_counter()
        m_buf = device.newBufferWithBytes_length_options_(data, size_bytes, Metal.MTLResourceStorageModeShared)
        end = time.perf_counter()
        print(f"  Metal.newWithBytes: {(end - start) * 1e6:.2f} us")

        # Metal (memcpy to contents)
        # We use the buffer created above
        contents = m_buf.contents()
        # Create a memoryview for efficient copy
        view = contents.as_buffer(size_bytes)
        start = time.perf_counter()
        view[:] = data
        end = time.perf_counter()
        print(f"  Metal memcpy:       {(end - start) * 1e6:.2f} us")


        # Cleanup
        del t
        del buf
        del p_buf
        del p_buf_hit
        del m_buf
        del gpu_tensor

if __name__ == "__main__":
    try:
        benchmark_buffer_ops()
    except ImportError:
        print("Skipping benchmark: metal_marlin or dependencies not found.")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
