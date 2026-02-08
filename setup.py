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

try:
    import pybind11
    PYBIND11_INCLUDE = pybind11.get_include()
except ImportError:
    PYBIND11_INCLUDE = None

try:
    import nanobind
    NANOBIND_INCLUDE = nanobind.get_include()
    USE_NANOBIND = True
except ImportError:
    NANOBIND_INCLUDE = None
    USE_NANOBIND = False

try:
    import torch
    TORCH_INCLUDE = torch.utils.cpp_extension.include_paths()
    TORCH_LIB_PATH = [torch.utils.cpp_extension.library_paths(
    )[0]] if torch.utils.cpp_extension.library_paths() else []
    HAS_TORCH = True
except ImportError:
    TORCH_INCLUDE = []
    TORCH_LIB_PATH = []
    HAS_TORCH = False

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExtWithMM(build_ext):
    """Custom build_ext that handles .mm (Objective-C++) files."""

    def build_extension(self, ext):
        # Ensure .mm is recognized as a source file
        if hasattr(self.compiler, 'src_extensions'):
            if '.mm' not in self.compiler.src_extensions:
                self.compiler.src_extensions.append('.mm')
        if any(str(src).endswith(".mm") for src in ext.sources):
            ext.extra_compile_args = ext.extra_compile_args or []
            if "-fobjc-arc" not in ext.extra_compile_args:
                ext.extra_compile_args.append("-fobjc-arc")
        super().build_extension(ext)


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
include_dirs = [str(EXT_DIR), str(HERE / "include"), str(HERE / "cpp/include")]
if np is not None:
    include_dirs.append(np.get_include())
if PYBIND11_INCLUDE is not None:
    include_dirs.append(PYBIND11_INCLUDE)
if NANOBIND_INCLUDE is not None:
    include_dirs.append(NANOBIND_INCLUDE)
if TORCH_INCLUDE:
    include_dirs.extend(TORCH_INCLUDE)

extensions = [
    Extension(
        "metal_marlin._metal_buffer_bridge",
        sources=["metal_marlin/_metal_buffer_bridge.pyx"],  # Relative path
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
    Extension(
        "metal_marlin.quantization._ldl_fast",
        sources=["metal_marlin/quantization/_ldl_fast.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math", "-std=c11"],
        extra_link_args=["-framework", "Accelerate"],
        language="c",
    ),
    Extension(
        "metal_marlin.quantization._eigh_fast",
        sources=["metal_marlin/quantization/_eigh_fast.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-ffast-math", "-std=c11"],
        extra_link_args=["-framework", "Accelerate"],
        language="c",
    ),
    Extension(
        "metal_marlin._psd_dispatch",
        sources=["metal_marlin/_psd_dispatch.mm"],
        include_dirs=include_dirs,
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-fvisibility=hidden",
            "-fobjc-arc",  # Automatic reference counting
        ],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "Foundation",
            "-lobjc",
        ],
        language="objc++",
    ),
]

# Add torch-dependent extensions only if torch is available at build time
if HAS_TORCH:
    extensions.extend([
        Extension(
            "metal_marlin._moe_dispatcher",
            sources=["metal_marlin/cpp/moe_dispatcher.mm"],
            include_dirs=include_dirs,
            extra_compile_args=[
                "-std=c++17",
                "-O3",
                "-fvisibility=hidden",
                "-fobjc-arc",
            ],
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
                "-lobjc",
            ],
            language="objc++",
        ),
    ])

    # _cpp_ext extension: use nanobind version if available, otherwise pybind11
    if USE_NANOBIND:
        print("Building _cpp_ext with nanobind (full mixed-bpw dispatch support)", file=sys.stderr)
        extensions.append(
            Extension(
                "metal_marlin._cpp_ext",
                sources=[
                    "cpp/src/python_bindings.mm",
                    "metal_marlin/cpp/moe_dispatcher.mm",
                    "cpp/src/gemm_dispatch.cpp",
                    "cpp/src/device.cpp",
                    "cpp/src/device_discovery.cpp",
                    "cpp/src/library_manager.mm",
                    "cpp/src/events.mm",
                    "cpp/src/encoder_cache.mm",
                    "cpp/src/buffer_manager.cpp",
                    "cpp/src/pool.cpp",
                    "cpp/src/moe_manager.mm",
                    "cpp/src/moe_router_dispatch.cpp",
                ],
                include_dirs=include_dirs,
                extra_compile_args=[
                    "-std=c++17",
                    "-O3",
                    "-fvisibility=hidden",
                    "-fobjc-arc",
                ],
                extra_link_args=[
                    "-framework", "Metal",
                    "-framework", "Foundation",
                    "-lobjc",
                ],
                language="objc++",
            ),
        )
    else:
        print("Building _cpp_ext with pybind11 (basic MoE dispatch, no mixed-bpw)", file=sys.stderr)
        extensions.append(
            Extension(
                "metal_marlin._cpp_ext",
                sources=[
                    "metal_marlin/cpp_extension.cpp",
                    "metal_marlin/cpp/moe_dispatcher.mm",
                    "cpp/src/gemm_dispatch.cpp",
                ],
                include_dirs=include_dirs,
                extra_compile_args=[
                    "-std=c++17",
                    "-O3",
                    "-fvisibility=hidden",
                    "-fobjc-arc",
                ],
                extra_link_args=[
                    "-framework", "Metal",
                    "-framework", "Foundation",
                    "-lobjc",
                ],
                language="objc++",
            ),
        )
else:
    print("Note: torch not found at build time, skipping C++ MoE dispatcher extensions", file=sys.stderr)

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
        cmdclass={"build_ext": BuildExtWithMM},
    )
