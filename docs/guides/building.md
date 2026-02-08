# Building metal_marlin

## Prerequisites

- macOS 13+ (Ventura or later)
- Xcode Command Line Tools
- Python 3.12+
- uv package manager

## Building the Metallib

```bash
cd contrib/metal_marlin
./scripts/build_metallib.sh
```

Options:
- `OUTPUT_DIR`: Custom output directory (default: `metal_marlin/lib`)

## Build Process

The build script compiles all Metal shaders to a single `.metallib`:

```bash
# First build: compiles all ~67 shaders
./scripts/build_metallib.sh

# Output: metal_marlin/lib/metal_marlin.metallib
```

The script:
1. Finds all `.metal` files in `src/` and `metal_marlin/` subdirectories
2. Excludes header-only files (`bf16_compat.metal`, `reduction_helpers.metal`, `dequant_helpers.metal`)
3. Compiles each to `.air` (intermediate representation) with Metal 3.0 and optimization flags
4. Links all `.air` files into a single `metal_marlin.metallib`

## Using Make

```bash
make metallib    # Build
make clean       # Remove build artifacts
make rebuild     # Clean + build
make test        # Build + verify library loads
```

## Troubleshooting

### "Multiple symbols" error

Duplicate kernel definitions exist. Run:
```bash
grep -rh "^kernel void" src/ | awk '{print $3}' | sort | uniq -c | awk '$1>1'
```

### Syntax errors

Check specific file:
```bash
xcrun -sdk macosx metal -std=metal3.0 -fsyntax-only -Isrc/ src/FILE.metal
```

### Shader compilation failures

The build script attempts relaxed compilation (`-O0`) for shaders that fail with default flags. Check the build output for "FAILED (skipped)" messages and investigate those files individually.

### Verify library loads

```bash
uv run python -c "from metal_marlin.metal_dispatch import get_precompiled_library; print('OK' if get_precompiled_library() else 'FAIL')"
```
