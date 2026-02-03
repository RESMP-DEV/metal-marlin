#!/usr/bin/env python3
"""Diagnose the grid issue in MoE dispatch."""

batch_size = 128
top_k = 8
hidden_dim = 2048
tile_n = 64

# Current implementation
n = batch_size * top_k  # 1024
grid_x = (hidden_dim + tile_n - 1) // tile_n  # 33
grid_y = n  # 1024

print("=== Current Grid Calculation ===")
print(f"batch_size = {batch_size}")
print(f"top_k = {top_k}")
print(f"n = batch_size * top_k = {n}")
print(f"grid_x = (2048 + 64 - 1) // 64 = {grid_x}")
print(f"grid_y = n = {grid_y}")
print(f"Total threadgroups = {grid_x * grid_y:,}")
print()

# What should it be
print("=== What it should be ===")
print("Each token should be processed ONCE, not top_k times")
print("So grid_y should be batch_size, not n")
grid_y_fixed = batch_size
print(f"grid_y (fixed) = batch_size = {grid_y_fixed}")
print(f"Total threadgroups (fixed) = {grid_x * grid_y_fixed:,}")
print()

# Memory analysis
print("=== Memory Traffic Analysis ===")
activation_size_bytes = batch_size * hidden_dim * 2  # fp16 = 2 bytes per element
print(f"Activation size: {activation_size_bytes / 1024:.1f} KB")

# Current: each activation loaded top_k times
current_memory = activation_size_bytes * top_k
print(f"Memory traffic (current): {current_memory / 1024:.1f} KB (loaded {top_k}x)")

# Fixed: each activation loaded once
fixed_memory = activation_size_bytes
print(f"Memory traffic (fixed): {fixed_memory / 1024:.1f} KB (loaded once)")
print()

print(f"Memory reduction: {current_memory / fixed_memory:.1f}x")
print("This explains the 13x slowdown!")
