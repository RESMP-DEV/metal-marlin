"""
Build script for Cython extension.

This is used alongside hatchling via a custom build hook.
Run directly for development:
    python setup.py build_ext --inplace
"""

import os
import sys
from pathlib import Path

# Check for Cython and numpy before importing setuptools
try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options

    Options.fast_fail = True
except ImportError:
    print("Cython not found. Install with: uv add cython", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None

from setuptools import Extension, setup

# Extension configuration
HERE = Path(__file__).parent
EXT_DIR = HERE / "metal_marlin"

# macOS-specific flags for Objective-C runtime
extra_compile_args = [
    "-O3",
    "-ffast-math",
    "-std=c11",
    "-fobjc-arc",  # Automatic reference counting
]

extra_link_args = [
    "-framework", "Foundation",
    "-framework", "Metal",
    "-lobjc",
]

# Include paths
include_dirs = [str(EXT_DIR)]
if np is not None:
    include_dirs.append(np.get_include())

extensions = [
    Extension(
        "metal_marlin._metal_buffer_bridge",
        sources=["metal_marlin/_metal_buffer_bridge.pyx"],  # Relative path
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
]

# Cythonize with optimization settings
cython_directives = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
    "embedsignature": True,
}

if __name__ == "__main__":
    setup(
        name="metal-marlin-ext",
        ext_modules=cythonize(
            extensions,
            compiler_directives=cython_directives,
            annotate=os.environ.get("CYTHON_ANNOTATE", "0") == "1",
        ),
    )
