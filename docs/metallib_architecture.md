# Metal Shader Precompilation Architecture

## Overview

metal_marlin uses precompiled Metal shaders (.metallib) for fast kernel dispatch.
This provides 100-1000x faster kernel lookup compared to runtime JIT compilation.

## Directory Structure

```
metal_marlin/
├── src/                    # Metal shader sources (.metal)
│   ├── fusion/             # Fused kernel sources
│   └── *.metal             # Individual shaders
├── metal_marlin/
│   ├── lib/
│   │   ├── metal_marlin.metallib  # Precompiled library
│   │   ├── air/                   # Intermediate .air files
│   │   └── .metallib_version      # Build metadata
│   ├── metallib_loader.py         # Python loader
│   └── metal_dispatch.py          # Kernel dispatch (uses metallib)
└── scripts/
    └── build_metallib.sh          # Compilation script
```

## Build Process

1. `build_metallib.sh` compiles all .metal files to .air (intermediate)
2. Links .air files into single .metallib
3. Writes version metadata

## Runtime Behavior

1. `get_kernel()` first checks precompiled metallib
2. Falls back to JIT compilation if kernel not found
3. Logging indicates which path was used ([metallib] vs [jit])

## Performance

| Operation | Time |
|-----------|------|
| Metallib lookup | ~0.01 ms |
| JIT compile (first time) | ~50-100 ms |
| JIT lookup (cached) | ~0.1 ms |
