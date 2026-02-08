cmake_minimum_required(VERSION 3.18)

# Test executable for FastRouterDispatcher
project(test_router_dispatch LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Test executable - only uses C++ files (no Metal/objc needed for pure router tests)
add_executable(test_router_dispatch
    tests/test_router_dispatch.cpp
    src/moe_router_dispatch.cpp
)

target_include_directories(test_router_dispatch PRIVATE
    include
)

# Standard library linkage for std::bit_cast
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(test_router_dispatch PRIVATE c++abi)
endif()

# Benchmark executable (same as above)
add_executable(bench_router_dispatch
    tests/bench_router_dispatch.cpp
    src/moe_router_dispatch.cpp
)

target_include_directories(bench_router_dispatch PRIVATE
    include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(bench_router_dispatch PRIVATE c++abi)
endif()
