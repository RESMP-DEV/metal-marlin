# Fast Dispatch (C++ Extension)

Metal Marlin can use a C++ extension for 5-10x faster kernel dispatch.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cp _cpp_ext.cpython-312-darwin.so ../metal_marlin/
```

## Performance

| Backend | Overhead per call |
|---------|-------------------|
| C++ extension | ~5-15μs |
| PyObjC fallback | ~80-150μs |

For MoE layers with 2-4 expert dispatches, the C++ path saves 200-600μs per layer.

## Usage

```python
from metal_marlin import fast_dispatch_available

if fast_dispatch_available():
    print("Using C++ fast path (~5-15μs overhead)")
else:
    print("Using PyObjC fallback (~80-150μs overhead)")
```

## Available Exports

```python
from metal_marlin._cpp_ext import (
    BatchDispatch,      # Batched kernel dispatch
    BufferPool,         # Reusable buffer management
    EncoderCache,       # MTLCommandEncoder caching
    LibraryManager,     # Precompiled metallib loading
    ManagedBuffer,      # RAII buffer wrapper
    MetalContext,       # Device + queue singleton
    MetalDevice,        # MTLDevice wrapper
    QueueManager,       # Command queue management
    dispatch_kernel,    # Low-level dispatch
    create_buffer,      # Buffer creation helper
)
```

## Fallback

The pure PyObjC path remains fully functional if you don't build the extension. Metal Marlin automatically detects and uses the fastest available dispatch path.
