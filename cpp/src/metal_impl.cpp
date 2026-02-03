/**
 * @file metal_impl.cpp
 * @brief Translation unit for Metal-cpp implementation symbols.
 * 
 * This file defines the implementation macros for Metal-cpp (Apple's C++ interface).
 * These must be defined in exactly one translation unit in the whole project
 * to avoid redefinition errors at link time.
 */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
