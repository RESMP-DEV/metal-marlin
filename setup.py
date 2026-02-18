"""
Build script for Cython extension.

This is used alongside hatchling via a custom build hook.
Run directly for development:
    python setup.py build_ext --inplace

On Linux, no native extensions are built (pure Python only).
Metal extensions are macOS-only.
"""

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

IS_MACOS = sys.platform == "darwin"

# ============================================================================
# Non-macOS: Pure Python package, no native extensions
# ============================================================================
if not IS_MACOS:
    if __name__ == "__main__":
        setup(
            name="metal-marlin-ext",
            ext_modules=[],
        )
else:
    # ========================================================================
    # macOS: Build Cython and Objective-C++ extensions
    # ========================================================================
    
    # Check for Cython and numpy before building
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
        import shutil
        NANOBIND_INCLUDE = nanobind.include_dir()
        USE_NANOBIND = True
        NANOBIND_ROOT = Path(NANOBIND_INCLUDE).parent
        
        # Copy nanobind src to local dir to avoid absolute path issues
        NANOBIND_SRC_DIR = NANOBIND_ROOT / "src"
        LOCAL_NB_DIR = Path("cpp/nanobind_src")
        
        if NANOBIND_SRC_DIR.exists():
            if not LOCAL_NB_DIR.exists():
                print(f"Copying {NANOBIND_SRC_DIR} to {LOCAL_NB_DIR}")
                shutil.copytree(NANOBIND_SRC_DIR, LOCAL_NB_DIR)
            
            NANOBIND_SRC = LOCAL_NB_DIR / "nb_combined.cpp"
        else:
            NANOBIND_SRC = None
            
        NANOBIND_EXT_INCLUDE = NANOBIND_ROOT / "ext" / "robin_map" / "include"
    except ImportError:
        NANOBIND_INCLUDE = None
        USE_NANOBIND = False
        NANOBIND_SRC = None
        NANOBIND_EXT_INCLUDE = None

    try:
        import torch
        import torch.utils.cpp_extension
        TORCH_INCLUDE = torch.utils.cpp_extension.include_paths()
        TORCH_LIB_PATH = [torch.utils.cpp_extension.library_paths()[0]] if torch.utils.cpp_extension.library_paths() else []
        HAS_TORCH = True
    except ImportError:
        TORCH_INCLUDE = []
        TORCH_LIB_PATH = []
        HAS_TORCH = False

    class BuildExtWithMM(build_ext):
        """Custom build_ext that handles .mm (Objective-C++) files."""

        def build_extension(self, ext):
            if hasattr(self.compiler, 'src_extensions'):
                if '.mm' not in self.compiler.src_extensions:
                    self.compiler.src_extensions.append('.mm')
            if any(str(src).endswith(".mm") for src in ext.sources):
                ext.extra_compile_args = ext.extra_compile_args or []
                if "-fobjc-arc" not in ext.extra_compile_args:
                    ext.extra_compile_args.append("-fobjc-arc")
            super().build_extension(ext)

    HERE = Path(__file__).parent
    EXT_DIR = HERE / "metal_marlin"

    extra_compile_args = ["-O3", "-ffast-math", "-std=c11", "-fobjc-arc", "-march=armv8.2-a+fp16+dotprod"]
    extra_link_args = ["-framework", "Foundation", "-framework", "Metal", "-lobjc"]

    include_dirs = [str(EXT_DIR), str(HERE / "include"), str(HERE / "cpp/include")]
    if np is not None:
        include_dirs.append(np.get_include())
    if PYBIND11_INCLUDE is not None:
        include_dirs.append(PYBIND11_INCLUDE)
    if NANOBIND_INCLUDE is not None:
        include_dirs.append(NANOBIND_INCLUDE)
        if NANOBIND_EXT_INCLUDE and NANOBIND_EXT_INCLUDE.exists():
            include_dirs.append(str(NANOBIND_EXT_INCLUDE))
    if TORCH_INCLUDE:
        include_dirs.extend(TORCH_INCLUDE)

    extensions = [
        Extension(
            "metal_marlin._metal_buffer_bridge",
            sources=["metal_marlin/_metal_buffer_bridge.pyx"],
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
            extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden", "-fobjc-arc"],
            extra_link_args=["-framework", "Metal", "-framework", "Foundation", "-lobjc"],
            language="objc++",
        ),
    ]

    if HAS_TORCH:
        extensions.append(
            Extension(
                "metal_marlin._moe_dispatcher",
                sources=["metal_marlin/cpp/moe_dispatcher.mm"],
                include_dirs=include_dirs,
                extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden", "-fobjc-arc"],
                extra_link_args=["-framework", "Metal", "-framework", "Foundation", "-lobjc"],
                language="objc++",
            )
        )
        if USE_NANOBIND:
            sources = [
                "cpp/src/python_bindings.mm", "metal_marlin/cpp/moe_dispatcher.mm",
                "cpp/src/gemm_dispatch.cpp", "cpp/src/device.cpp", "cpp/src/device_discovery.cpp",
                "cpp/src/library_manager.mm", "cpp/src/events.mm", "cpp/src/encoder_cache.mm",
                "cpp/src/buffer_manager.cpp", "cpp/src/pool.cpp", "cpp/src/expert_buffer_pool.cpp",
                "cpp/src/moe_manager.mm", "cpp/src/pipeline.mm",
                "cpp/src/moe_router_dispatch.cpp", "cpp/src/metal_impl.cpp",
                "cpp/src/direct_access.mm", "cpp/src/norm_ops.cpp",
                "cpp/src/batch_dispatch.mm", "cpp/src/mla_attention.mm",
                "cpp/src/weights_ops.cpp",
            ]
            if NANOBIND_SRC and NANOBIND_SRC.exists():
                sources.append(str(NANOBIND_SRC))

            extensions.append(
                Extension(
                    "metal_marlin._cpp_ext",
                    sources=sources,
                    include_dirs=include_dirs,
                    extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden", "-fobjc-arc"],
                    extra_link_args=["-framework", "Metal", "-framework", "Foundation", "-lobjc"],
                    language="objc++",
                )
            )
            
            # Optimized MTLBuffer direct access bridge
            mtl_bridge_sources = [
                "cpp/src/mtl_buffer_bridge.mm",
                "cpp/src/mtl_buffer_bridge_bindings.mm",
            ]
            if NANOBIND_SRC and NANOBIND_SRC.exists():
                mtl_bridge_sources.append(str(NANOBIND_SRC))
            
            extensions.append(
                Extension(
                    "metal_marlin._mtl_buffer_bridge",
                    sources=mtl_bridge_sources,
                    include_dirs=include_dirs,
                    extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden", "-fobjc-arc"],
                    extra_link_args=["-framework", "Metal", "-framework", "Foundation", "-lobjc"],
                    language="objc++",
                )
            )
        else:
            extensions.append(
                Extension(
                    "metal_marlin._cpp_ext",
                    sources=["metal_marlin/cpp_extension.cpp", "metal_marlin/cpp/moe_dispatcher.mm", "cpp/src/gemm_dispatch.cpp", "cpp/src/moe_router_dispatch.cpp"],
                    include_dirs=include_dirs,
                    extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden", "-fobjc-arc"],
                    extra_link_args=["-framework", "Metal", "-framework", "Foundation", "-lobjc"],
                    language="objc++",
                )
            )
    else:
        print("Note: torch not found at build time, skipping C++ MoE dispatcher extensions", file=sys.stderr)

    cython_directives = {
        "language_level": "3", "boundscheck": False, "wraparound": False,
        "cdivision": True, "initializedcheck": False, "nonecheck": False, "embedsignature": True,
    }

    if __name__ == "__main__":
        setup(
            name="metal-marlin-ext",
            ext_modules=cythonize(extensions, compiler_directives=cython_directives, annotate=os.environ.get("CYTHON_ANNOTATE", "0") == "1"),
            cmdclass={"build_ext": BuildExtWithMM},
        )
