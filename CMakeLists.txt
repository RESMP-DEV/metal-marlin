# Metal Marlin C++ Extension Build Configuration
#
# Requirements:
#   - macOS 13.0+ with Metal 3 support
#   - Xcode Command Line Tools (for metal-cpp)
#   - Python 3.11+ with nanobind
#
# Build:
#   pip install nanobind
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j
#
# Or via scikit-build-core:
#   pip install .

cmake_minimum_required(VERSION 3.18)

project(metal_marlin_ext LANGUAGES CXX OBJCXX)

# Require macOS
if(NOT APPLE)
    message(FATAL_ERROR "Metal Marlin C++ extension requires macOS")
endif()

# C++20 for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find Python and nanobind
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind via Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_RESULT
)

if(NOT NB_RESULT EQUAL 0)
    message(FATAL_ERROR "nanobind not found. Install with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Metal-cpp headers (included with Xcode)
# These are header-only and live in the macOS SDK
set(METAL_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp")

# Download metal-cpp if not present
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
    message(STATUS "Downloading metal-cpp headers...")
    file(DOWNLOAD
        "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
        "${CMAKE_BINARY_DIR}/metal-cpp.zip"
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download metal-cpp")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/metal-cpp.zip"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

# macOS frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

# Source files
file(GLOB CPP_SOURCES "cpp/src/*.cpp" "cpp/src/*.mm")
set(ALL_SOURCES ${CPP_SOURCES})

# Create the nanobind module
nanobind_add_module(
    _cpp_ext
    STABLE_ABI
    NB_STATIC
    ${ALL_SOURCES}
)

# Include directories
target_include_directories(_cpp_ext PRIVATE
    ${METAL_CPP_DIR}
    cpp/include
)

# Link frameworks
target_link_libraries(_cpp_ext PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
)

# macOS deployment target
set_target_properties(_cpp_ext PROPERTIES
    MACOSX_DEPLOYMENT_TARGET "13.0"
)

# Compiler flags for Objective-C++ interop
target_compile_options(_cpp_ext PRIVATE
    -fno-objc-arc  # metal-cpp manages its own memory
    -Wno-deprecated-declarations
)

# Install to the metal_marlin package
install(TARGETS _cpp_ext LIBRARY DESTINATION metal_marlin)
