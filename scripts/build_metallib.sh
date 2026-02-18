#!/bin/bash
# Precompile all Metal shaders to metallib
# Usage: ./build_metallib.sh [--force|-f] [--sequential|-s] [output_dir]
#
# Options:
#   --force, -f       Force full rebuild (delete all cached .air files)
#   --sequential, -s  Disable parallel compilation (for debugging)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# Parse arguments
FORCE_REBUILD=false
PARALLEL=true
OUTPUT_DIR=""

for arg in "$@"; do
    case $arg in
        --force|-f)
            FORCE_REBUILD=true
            ;;
        --sequential|-s)
            PARALLEL=false
            ;;
        *)
            OUTPUT_DIR="$arg"
            ;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/metal_marlin/lib}"

# Handle --force flag
if [ "$FORCE_REBUILD" = true ]; then
    echo "Force rebuild: removing cached .air files"
    rm -rf "$OUTPUT_DIR/air"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/air"

echo "=== Metal Shader Precompilation ==="
echo "Source: $SRC_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Find all .metal files
METAL_FILES=$(find "$SRC_DIR" -name "*.metal" -type f)
AIR_FILES=""

# Also include metal files from metal_marlin subdirectories
METAL_MARLIN_DIR="$PROJECT_ROOT/metal_marlin"
for subdir in distributed vision shaders src; do
    if [ -d "$METAL_MARLIN_DIR/$subdir" ]; then
        EXTRA_METALS=$(find "$METAL_MARLIN_DIR/$subdir" -name "*.metal" -type f 2>/dev/null || true)
        if [ -n "$EXTRA_METALS" ]; then
            METAL_FILES="$METAL_FILES $EXTRA_METALS"
        fi
    fi
done

# Exclude header-only files (no kernels, only inline helpers)
# These are #included by other shaders
if [ -f "$SCRIPT_DIR/header_only_files.txt" ]; then
    while read -r header; do
        # Skip empty lines or comments
        [[ -z "$header" || "$header" =~ ^# ]] && continue
        header_base=$(basename "$header")
        METAL_FILES=$(echo "$METAL_FILES" | tr ' ' '\n' | grep -v "$header_base" | tr '\n' ' ')
    done < "$SCRIPT_DIR/header_only_files.txt"
fi

# Exclude header-only files that cause duplicate symbol errors
# bf16_kernels.metal contains test-only utilities; compile in tests to avoid duplicates
# dequant_helpers.metal is #included by other shaders
DUPLICATES="bf16_kernels.metal$|dequant_helpers.metal$|reduction_helpers.metal$"

METAL_FILES=$(echo "$METAL_FILES" | tr ' ' '\n' | grep -Ev "$DUPLICATES" | tr '\n' ' ')

# Count files
TOTAL=$(echo "$METAL_FILES" | wc -w | tr -d ' ')
echo "Found $TOTAL Metal shader files"
echo ""

# Common include paths
INCLUDE_PATHS="-I$SRC_DIR -I$SRC_DIR/fusion -I$METAL_MARLIN_DIR/src"

# Metal compiler flags
# -std=metal3.0: Metal 3.0 for simdgroup_matrix
# -O2: Optimization level
# -ffast-math: Enable fast math
# -freciprocal-math: Use reciprocal approximations
# -fno-signed-zeros: Ignore signed zeros
METAL_FLAGS="-std=metal3.0 -O2 -ffast-math -freciprocal-math -fno-signed-zeros"

# Number of parallel jobs (all CPU cores)
NCPU=$(sysctl -n hw.ncpu)
if [ "$PARALLEL" = true ]; then
    echo "Using $NCPU parallel compilation jobs"
else
    echo "Sequential compilation mode"
fi
echo ""

# Track status via temp files (parallel processes can't update shell vars)
FAILED_LOG="$OUTPUT_DIR/air/.failed"
CACHED_LOG="$OUTPUT_DIR/air/.cached"
COMPILED_LOG="$OUTPUT_DIR/air/.compiled"
rm -f "$FAILED_LOG" "$CACHED_LOG" "$COMPILED_LOG"
touch "$FAILED_LOG" "$CACHED_LOG" "$COMPILED_LOG"

# Compile a single .metal file to .air
# Arguments: metal_file output_dir include_paths metal_flags
compile_one() {
    local metal_file="$1"
    local output_dir="$2"
    local include_paths="$3"
    local metal_flags="$4"

    # Create unique name using parent dir + basename to avoid collisions
    local parent_dir
    parent_dir=$(basename "$(dirname "$metal_file")")
    local base
    base=$(basename "$metal_file" .metal)
    local unique_name="${parent_dir}_${base}"
    local air_file="$output_dir/air/${unique_name}.air"

    # Check if rebuild is needed (incremental build support)
    if [ -f "$air_file" ] && [ ! "$metal_file" -nt "$air_file" ]; then
        echo "CACHED: $base.metal"
        echo "$base.metal" >> "$output_dir/air/.cached"
        return 0
    fi

    # shellcheck disable=SC2086
    if xcrun -sdk macosx metal $metal_flags $include_paths -c "$metal_file" -o "$air_file" 2>/dev/null; then
        echo "OK: $base.metal"
        echo "$base.metal" >> "$output_dir/air/.compiled"
        return 0
    else
        # Try with relaxed flags
        # shellcheck disable=SC2086
        if xcrun -sdk macosx metal -std=metal3.0 -O0 $include_paths -c "$metal_file" -o "$air_file" 2>/dev/null; then
            echo "OK (relaxed): $base.metal"
            echo "$base.metal" >> "$output_dir/air/.compiled"
            return 0
        else
            echo "FAILED: $base.metal"
            echo "$base.metal" >> "$output_dir/air/.failed"
            return 1
        fi
    fi
}
export -f compile_one

# Export variables for subprocesses
export OUTPUT_DIR INCLUDE_PATHS METAL_FLAGS

# Compile all .metal files
echo "Compiling shaders..."
if [ "$PARALLEL" = true ]; then
    # Parallel compilation using xargs -P with exported env vars
    # Use -n 1 to process one file at a time, avoiding command line length issues
    echo "$METAL_FILES" | tr ' ' '\n' | grep -v '^$' | \
        xargs -n 1 -P "$NCPU" -I {} bash -c 'compile_one "$1" "$OUTPUT_DIR" "$INCLUDE_PATHS" "$METAL_FLAGS"' _ {} || true
else
    # Sequential compilation (for debugging)
    for metal_file in $METAL_FILES; do
        compile_one "$metal_file" "$OUTPUT_DIR" "$INCLUDE_PATHS" "$METAL_FLAGS" || true
    done
fi

# Collect all successfully compiled .air files
AIR_FILES=$(find "$OUTPUT_DIR/air" -name "*.air" -type f 2>/dev/null | tr '\n' ' ')

# Count results from logs
FAILED=0
CACHED=0
COMPILED=0
FAILED_FILES=""
if [ -s "$FAILED_LOG" ]; then
    FAILED=$(wc -l < "$FAILED_LOG" | tr -d ' ')
    FAILED_FILES=$(tr '\n' ' ' < "$FAILED_LOG")
fi
if [ -s "$CACHED_LOG" ]; then
    CACHED=$(wc -l < "$CACHED_LOG" | tr -d ' ')
fi
if [ -s "$COMPILED_LOG" ]; then
    COMPILED=$(wc -l < "$COMPILED_LOG" | tr -d ' ')
fi
rm -f "$FAILED_LOG" "$CACHED_LOG" "$COMPILED_LOG"

echo ""

if [ -z "$AIR_FILES" ]; then
    echo "ERROR: No shaders compiled successfully"
    exit 1
fi

# Link all .air files into a single .metallib
METALLIB_FILE="$OUTPUT_DIR/metal_marlin.metallib"
echo "Linking to $METALLIB_FILE..."

# shellcheck disable=SC2086
if xcrun -sdk macosx metallib $AIR_FILES -o "$METALLIB_FILE"; then
    echo "SUCCESS: Created $METALLIB_FILE"
    ls -lh "$METALLIB_FILE"
else
    echo "ERROR: Failed to create metallib"
    exit 1
fi

echo ""
echo "Validating metallib..."

# List all functions in metallib
kernel_count=$(xcrun -sdk macosx metal-objdump --disassemble-symbols="" "$METALLIB_FILE" 2>/dev/null | grep -c "^[a-zA-Z_]" || echo 0)

# Verify we can load it
if xcrun -sdk macosx metal-objdump -t "$METALLIB_FILE" >/dev/null 2>&1; then
    echo "Validation: PASSED"
    echo "Kernels in metallib: $kernel_count"
else
    echo "Validation: FAILED"
    exit 1
fi

# Cleanup intermediate .air files (optional)
# rm -rf "$OUTPUT_DIR/air"

echo ""
echo "=== Build Summary ==="
echo "Total shaders: $TOTAL"
echo "Compiled: $COMPILED"
echo "Cached: $CACHED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed files:"
    for f in $FAILED_FILES; do
        echo "  - $f"
    done
fi

if [ -f "$METALLIB_FILE" ]; then
    echo ""
    echo "Output: $METALLIB_FILE"
    echo "Size: $(ls -lh "$METALLIB_FILE" | awk '{print $5}')"
    echo "Kernels: $(xcrun -sdk macosx metal-objdump -t "$METALLIB_FILE" 2>/dev/null | grep -c "kernel" || echo "N/A")"

    # Write version info for Python validation
    VERSION_FILE="$OUTPUT_DIR/.metallib_version"
    cat > "$VERSION_FILE" << EOF
build_date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
shader_count=$((TOTAL - FAILED))
metal_version=3.0
EOF
    echo "Version: $VERSION_FILE"

    # Generate checksum manifest for staleness detection
    echo ""
    echo "Generating checksum manifest..."
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from metal_marlin.metallib_loader import save_checksum_manifest
from pathlib import Path
manifest_path = save_checksum_manifest(Path('$METALLIB_FILE'))
print(f'Checksum manifest: {manifest_path}')
" 2>/dev/null || echo "Warning: Could not generate checksum manifest (Python not available or module error)"
    else
        echo "Warning: Python3 not available, skipping checksum manifest generation"
    fi
fi
