"""Debug U4 scale indexing to find divergence between Python and Metal."""

import numpy as np

# Same parameters as the failing test
M, K, N = 8, 512, 512
group_size = 32
num_groups = K // group_size  # 16 groups

print(f"Matrix dimensions: M={M}, K={K}, N={N}")
print(f"Group size: {group_size}")
print(f"Number of groups: {num_groups}")
print()

# Generate simple scales - easy to identify which group was used
scales = np.zeros((num_groups, N), dtype=np.float16)
for g in range(num_groups):
    scales[g, :] = (g + 1) * 1.0  # Group 0: 1.0, Group 1: 2.0, etc.

print(f"Scales shape: {scales.shape}")
print(f"Scales array is C-contiguous (row-major): {scales.flags['C_CONTIGUOUS']}")
print()

# Show the memory layout
print("=== Memory Layout Analysis ===")
print("Python flattened scales[:2, :4]:")
for g in range(min(2, num_groups)):
    for n in range(min(4, N)):
        print(f"  scales[{g},{n}] = {scales[g, n]:.1f} at flat index {g * N + n}")

print()
print("=== Metal Kernel Indexing ===")
print("Metal kernel uses: scales[group_idx * N + b_col]")
print("This is row-major indexing (row * num_cols + col)")
print()

# Show what the Metal kernel would access
print("=== What Metal sees ===")
scales_flat = scales.ravel()
print(f"Flattened scales shape: {scales_flat.shape}")

# For k_idx=0, group_idx=0, col=0: should get scales[0,0] = 1.0
# For k_idx=32, group_idx=1, col=0: should get scales[1,0] = 2.0
test_cases = [
    (0, 0),    # k=0, col=0 -> group 0
    (31, 0),   # k=31, col=0 -> group 0
    (32, 0),   # k=32, col=0 -> group 1
    (63, 0),   # k=63, col=0 -> group 1
    (64, 0),   # k=64, col=0 -> group 2
    (0, 1),    # k=0, col=1 -> group 0
    (32, 1),   # k=32, col=1 -> group 1
]

print("\nChecking scale access for various (k, col) pairs:")
for k_idx, col in test_cases:
    group_idx = k_idx // group_size
    # Row-major indexing (what Metal uses)
    flat_idx_rowmajor = group_idx * N + col
    # Column-major indexing (if Metal was wrong)
    flat_idx_colmajor = col * num_groups + group_idx

    scale_rowmajor = scales_flat[flat_idx_rowmajor] if flat_idx_rowmajor < len(scales_flat) else -999
    scale_colmajor = scales_flat[flat_idx_colmajor] if flat_idx_colmajor < len(scales_flat) else -999
    expected = scales[group_idx, col]

    print(f"  k={k_idx:3d}, col={col}: group={group_idx:2d}")
    print(f"    row-major index: {flat_idx_rowmajor:5d} -> {scale_rowmajor:.1f} (expected: {expected:.1f})")
    print(f"    col-major index: {flat_idx_colmajor:5d} -> {scale_colmajor:.1f}")
    if scale_rowmajor != expected:
        print("    *** MISMATCH! row-major doesn't match expected ***")

print()
print("=== Buffer Layout Check ===")

# Check if the buffer being passed to Metal might have different layout
scales_cont = np.ascontiguousarray(scales.astype(np.float16))
print(f"After ascontiguousarray: {scales_cont.flags['C_CONTIGUOUS']}")
print(f"Shape preserved: {scales_cont.shape}")

# Simulate what tobytes() would produce
scales_bytes = scales_cont.tobytes()
print(f"Total bytes: {len(scales_bytes)}")

# Read back as flat array
scales_readback = np.frombuffer(scales_bytes, dtype=np.float16)
print(f"Readback shape: {scales_readback.shape}")

# Check first few values
print("\nFirst 8 values in memory order (should be scales[0,0..7]):")
print(f"  {scales_readback[:8]}")
print(f"  Expected (scales[0,:8]): {scales[0,:8]}")

print("\nValues at positions N..N+7 (should be scales[1,0..7]):")
print(f"  {scales_readback[N:N+8]}")
print(f"  Expected (scales[1,:8]): {scales[1,:8]}")

# The issue might be in how the packed buffer is created/accessed
# Let's check the packed weight access pattern too
print()
print("=== Packed Weight B Layout ===")
print("Packed as [K/8, N] where each uint32 holds 8 consecutive K-values at one column")
print("Metal accesses: B[k_pack_idx * N + b_col]")
print()
print(f"For K={K}, k_packs = K/8 = {K//8}")
print(f"For k_idx=0..7, col=0: k_pack_idx=0, access B[0*{N} + 0] = B[0]")
print(f"For k_idx=8..15, col=0: k_pack_idx=1, access B[1*{N} + 0] = B[{N}]")
print(f"For k_idx=0..7, col=1: k_pack_idx=0, access B[0*{N} + 1] = B[1]")

print()
print("=== Potential Issue: fused_dequant_u4x8 nibble ordering ===")
print("The kernel unpacks 8 nibbles from a uint32:")
print("  nibble 0 = bits [3:0]   -> k_idx + 0")
print("  nibble 1 = bits [7:4]   -> k_idx + 1")
print("  etc...")
print()
print("But fused_dequant_u4x8 uses a magic bias trick and outputs:")
print("  out[0] = v0.x -> nibble from (packed & LO_MASK).x")
print("  out[1] = v1.x -> nibble from ((packed >> 4) & LO_MASK).x")
print("  ...")
print("  out[4] = v0.y -> nibble from (packed & LO_MASK).y (bits [19:16])")
print()
print("Wait, the LO_NIBBLE_MASK = 0x000F000F extracts bits [3:0] and [19:16]!")
print("So the .x component gets the LOW nibble and .y gets a HIGH nibble")
print()
print("Looking at the kernel output ordering:")
print("  out[0..3] are from .x components")
print("  out[4..7] are from .y components")
print("This means k indices are NOT consecutive! They are interleaved!")

print()
print("=== Let's trace through fused_dequant_u4x8 ===")
packed = 0x76543210  # nibbles: 0,1,2,3,4,5,6,7 from LSB to MSB
print(f"Example packed = 0x{packed:08X}")

# Python simulation of what Metal does
FUSED_MAGIC_BIAS = 0x64006400
FUSED_LO_MASK = 0x000F000F

def unpack_half2(val):
    """Extract two values from uint32 treated as half2."""
    # In Metal, bits [15:0] are .x and bits [31:16] are .y
    lo = val & 0xFFFF
    hi = (val >> 16) & 0xFFFF
    return lo, hi

n0 = (packed & FUSED_LO_MASK) | FUSED_MAGIC_BIAS
n1 = ((packed >> 4) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS
n2 = ((packed >> 8) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS
n3 = ((packed >> 12) & FUSED_LO_MASK) | FUSED_MAGIC_BIAS

print("\nn0 = (packed & 0x000F000F) | 0x64006400")
print(f"   packed & LO_MASK = 0x{packed & FUSED_LO_MASK:08X}")
print(f"   n0 = 0x{n0:08X}")
lo0, hi0 = unpack_half2(n0)
print(f"   .x = 0x{lo0:04X} (nibble value: {(packed >> 0) & 0xF})")
print(f"   .y = 0x{hi0:04X} (nibble value: {(packed >> 16) & 0xF})")

print("\nn1 = ((packed >> 4) & 0x000F000F) | 0x64006400")
n1_masked = (packed >> 4) & FUSED_LO_MASK
print(f"   (packed >> 4) & LO_MASK = 0x{n1_masked:08X}")
print(f"   n1 = 0x{n1:08X}")
lo1, hi1 = unpack_half2(n1)
print(f"   .x = 0x{lo1:04X} (nibble value: {(packed >> 4) & 0xF})")
print(f"   .y = 0x{hi1:04X} (nibble value: {(packed >> 20) & 0xF})")

print("\nn2 = ((packed >> 8) & 0x000F000F) | 0x64006400")
n2_masked = (packed >> 8) & FUSED_LO_MASK
print(f"   (packed >> 8) & LO_MASK = 0x{n2_masked:08X}")
lo2, hi2 = unpack_half2(n2)
print(f"   .x = 0x{lo2:04X} (nibble value: {(packed >> 8) & 0xF})")
print(f"   .y = 0x{hi2:04X} (nibble value: {(packed >> 24) & 0xF})")

print("\nn3 = ((packed >> 12) & 0x000F000F) | 0x64006400")
n3_masked = (packed >> 12) & FUSED_LO_MASK
print(f"   (packed >> 12) & LO_MASK = 0x{n3_masked:08X}")
lo3, hi3 = unpack_half2(n3)
print(f"   .x = 0x{lo3:04X} (nibble value: {(packed >> 12) & 0xF})")
print(f"   .y = 0x{hi3:04X} (nibble value: {(packed >> 28) & 0xF})")

print("\n=== ACTUAL OUTPUT ORDERING FROM fused_dequant_u4x8 ===")
nibble_order = []
# out[0] = v0.x, out[1] = v1.x, out[2] = v2.x, out[3] = v3.x
# out[4] = v0.y, out[5] = v1.y, out[6] = v2.y, out[7] = v3.y
nibble_positions = [0, 4, 8, 12, 16, 20, 24, 28]
for i in range(8):
    npos = nibble_positions[i]
    nibble_val = (packed >> npos) & 0xF
    print(f"  out[{i}] comes from nibble at bit position {npos} (nibble index {npos//4}), value = {nibble_val}")
    nibble_order.append(npos // 4)

print(f"\nSo out[0..7] correspond to nibble indices: {nibble_order}")
print("Expected for consecutive k values: [0,1,2,3,4,5,6,7]")
print()
if nibble_order != [0,1,2,3,4,5,6,7]:
    print("*** MISMATCH! The nibble unpacking order is wrong! ***")
    print("This means k values are being read in the wrong order!")
else:
    print("Nibble order is correct!")
