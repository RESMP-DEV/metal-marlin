#include <metal_stdlib>
using namespace metal;

// ============================================================================
// General N:M Structured Sparsity for Apple Metal
// ============================================================================
//
// Supports arbitrary N:M patterns where N non-zero values are kept per
// M-element block. Common patterns:
//
//   Pattern  | Sparsity | Bits/block | Blocks/uint32 | Dense/uint32
//   ---------|----------|------------|---------------|-------------
//   1:4      | 75%      | 2          | 16            | 64
//   2:4      | 50%      | 4          | 8             | 32
//   2:8      | 75%      | 6          | 5 (+2 waste)  | 40
//   4:8      | 50%      | 7          | 4 (+4 waste)  | 32
//
// Metadata encoding:
//   - 1:4, 2:4, 2:8 use packed position indices (LSB-first).
//   - 4:8 uses a compact 7-bit combination index (C(8,4)=70).
//
//   M=4 → 2-bit indices (mask 0x3)
//   M=8 → 3-bit indices (mask 0x7)
//
// ============================================================================

// ============================================================================
// Compile-time constants for N:M patterns
// ============================================================================

// Bits needed to represent a position within an M-element block
constant constexpr uint BITS_PER_POS_M4 = 2;  // ceil(log2(4))
constant constexpr uint BITS_PER_POS_M8 = 3;  // ceil(log2(8))

// Position masks
constant constexpr uint POS_MASK_M4 = 0x3u;   // (1 << 2) - 1
constant constexpr uint POS_MASK_M8 = 0x7u;   // (1 << 3) - 1

// Bits per block for each pattern
constant constexpr uint BITS_PER_BLOCK_1_4 = 2;   // 1 * 2
constant constexpr uint BITS_PER_BLOCK_2_4 = 4;   // 2 * 2
constant constexpr uint BITS_PER_BLOCK_2_8 = 6;   // 2 * 3
constant constexpr uint BITS_PER_BLOCK_4_8 = 7;   // compact: ceil(log2(C(8,4)))
constant constexpr uint BITS_PER_BLOCK_4_8_PACKED = 12;  // legacy packed indices

// Blocks that fit in a single uint32
constant constexpr uint BLOCKS_PER_U32_1_4 = 16;  // 32 / 2
constant constexpr uint BLOCKS_PER_U32_2_4 = 8;   // 32 / 4
constant constexpr uint BLOCKS_PER_U32_2_8 = 5;   // floor(32 / 6)
constant constexpr uint BLOCKS_PER_U32_4_8 = 4;   // floor(32 / 7)

// Tile dimensions for dequant_sparse_fp4_tile
constant constexpr uint TILE_K = 8;              // K dimension per tile
constant constexpr uint TILE_N = 64;              // N dimension per tile
constant constexpr uint THREADS_PER_TG = 256;     // Threads per threadgroup


// ============================================================================
// 1:4 Pattern (75% sparse) - 1 value per 4-element block
// ============================================================================

/// Decode 1:4 metadata for one block.
/// Input: 2-bit field (position 0-3 of the single non-zero value).
inline uint decode_sparse_1in4(uint metadata_2bit) {
    return metadata_2bit & POS_MASK_M4;
}

/// Decode 16 blocks of 1:4 metadata from a packed uint32.
/// Each block uses 2 bits (one 2-bit index).
/// Covers 64 dense elements → 16 sparse values.
inline void decode_sparse_1in4_x16(uint32_t metadata,
                                    thread uint positions[16]) {
    for (uint i = 0; i < 16; ++i) {
        positions[i] = (metadata >> (i * BITS_PER_POS_M4)) & POS_MASK_M4;
    }
}

/// Unrolled variant for 1:4 x16 decode.
inline void decode_sparse_1in4_x16_unrolled(uint32_t metadata,
                                             thread uint positions[16]) {
    positions[0]  =  metadata        & POS_MASK_M4;
    positions[1]  = (metadata >>  2) & POS_MASK_M4;
    positions[2]  = (metadata >>  4) & POS_MASK_M4;
    positions[3]  = (metadata >>  6) & POS_MASK_M4;
    positions[4]  = (metadata >>  8) & POS_MASK_M4;
    positions[5]  = (metadata >> 10) & POS_MASK_M4;
    positions[6]  = (metadata >> 12) & POS_MASK_M4;
    positions[7]  = (metadata >> 14) & POS_MASK_M4;
    positions[8]  = (metadata >> 16) & POS_MASK_M4;
    positions[9]  = (metadata >> 18) & POS_MASK_M4;
    positions[10] = (metadata >> 20) & POS_MASK_M4;
    positions[11] = (metadata >> 22) & POS_MASK_M4;
    positions[12] = (metadata >> 24) & POS_MASK_M4;
    positions[13] = (metadata >> 26) & POS_MASK_M4;
    positions[14] = (metadata >> 28) & POS_MASK_M4;
    positions[15] = (metadata >> 30) & POS_MASK_M4;
}

/// Scatter 1 sparse value into a 4-element dense block.
inline void scatter_1in4(half val, uint pos,
                          thread half (&block)[4]) {
    block[0] = 0.0h;
    block[1] = 0.0h;
    block[2] = 0.0h;
    block[3] = 0.0h;
    block[pos] = val;
}

/// Encode a single position (0-3) into a 2-bit metadata field.
inline uint encode_sparse_1in4(uint pos) {
    return pos & POS_MASK_M4;
}

/// Pack 16 block metadata (1:4) into a single uint32.
inline uint32_t pack_metadata_1in4_x16(thread uint fields[16]) {
    uint32_t packed = 0;
    for (uint i = 0; i < 16; ++i) {
        packed |= (fields[i] & POS_MASK_M4) << (i * BITS_PER_POS_M4);
    }
    return packed;
}


// ============================================================================
// 2:4 Pattern (50% sparse) - 2 values per 4-element block (NVIDIA standard)
// ============================================================================

/// FP4 E2M1 dequantization helper (matches sparse_gemm.metal semantics).
inline half dequant_fp4_e2m1(uint nibble) {
    uint sign_bit = (nibble >> 3) & 1u;
    uint exp_bits = (nibble >> 1) & 0x3u;
    uint man_bit  = nibble & 1u;

    half sub_mag = half(man_bit) * half(0.25h);
    half norm_mag = half(1u << (exp_bits - 1u))
                    * (half(1.0h) + half(man_bit) * half(0.5h));

    half magnitude = select(norm_mag, sub_mag, exp_bits == 0u);
    return select(magnitude, -magnitude, bool(sign_bit));
}

/// Decode 2:4 metadata for one block.
/// Input: 4-bit metadata nibble (bits [3:0] of metadata_4bit).
/// Output: pos0, pos1 are the two positions (0-3) containing non-zero values.
inline void decode_sparse_2in4(uint metadata_4bit,
                                thread uint& pos0,
                                thread uint& pos1) {
    pos0 = metadata_4bit & POS_MASK_M4;
    pos1 = (metadata_4bit >> BITS_PER_POS_M4) & POS_MASK_M4;
}

/// Decode 8 blocks of 2:4 metadata from a packed uint32.
/// Each block occupies 4 bits (two 2-bit position indices).
/// Covers 32 dense elements → 16 sparse (non-zero) values.
inline void decode_sparse_metadata_x8(uint32_t metadata,
                                       thread uint2 positions[8]) {
    for (uint i = 0; i < 8; ++i) {
        uint nibble = (metadata >> (i * BITS_PER_BLOCK_2_4)) & 0xFu;
        positions[i] = uint2(nibble & POS_MASK_M4,
                             (nibble >> BITS_PER_POS_M4) & POS_MASK_M4);
    }
}

/// Decode 8 blocks with unrolled bit extraction (branchless, no loop).
inline void decode_sparse_metadata_x8_unrolled(uint32_t metadata,
                                                thread uint2 positions[8]) {
    positions[0] = uint2( metadata        & POS_MASK_M4, (metadata >>  2) & POS_MASK_M4);
    positions[1] = uint2((metadata >>  4) & POS_MASK_M4, (metadata >>  6) & POS_MASK_M4);
    positions[2] = uint2((metadata >>  8) & POS_MASK_M4, (metadata >> 10) & POS_MASK_M4);
    positions[3] = uint2((metadata >> 12) & POS_MASK_M4, (metadata >> 14) & POS_MASK_M4);
    positions[4] = uint2((metadata >> 16) & POS_MASK_M4, (metadata >> 18) & POS_MASK_M4);
    positions[5] = uint2((metadata >> 20) & POS_MASK_M4, (metadata >> 22) & POS_MASK_M4);
    positions[6] = uint2((metadata >> 24) & POS_MASK_M4, (metadata >> 26) & POS_MASK_M4);
    positions[7] = uint2((metadata >> 28) & POS_MASK_M4, (metadata >> 30) & POS_MASK_M4);
}

/// Scatter 2 sparse values into a 4-element dense block.
inline void scatter_2in4(half val0, half val1,
                          uint pos0, uint pos1,
                          thread half (&block)[4]) {
    block[0] = 0.0h;
    block[1] = 0.0h;
    block[2] = 0.0h;
    block[3] = 0.0h;
    block[pos0] = val0;
    block[pos1] = val1;
}

/// Encode two positions (0-3) into a 4-bit metadata nibble.
inline uint encode_sparse_2in4(uint pos0, uint pos1) {
    return (pos1 << BITS_PER_POS_M4) | pos0;
}

/// Pack 8 block metadata nibbles (2:4) into a single uint32.
inline uint32_t pack_metadata_x8(thread uint nibbles[8]) {
    uint32_t packed = 0;
    for (uint i = 0; i < 8; ++i) {
        packed |= (nibbles[i] & 0xFu) << (i * BITS_PER_BLOCK_2_4);
    }
    return packed;
}

// Dequant sparse FP4 values and scatter to dense tile
// Input: packed sparse values + metadata
// Output: dense TILE_K × TILE_N tile with zeros in pruned positions

inline void dequant_sparse_fp4_tile(
    device const uint32_t* sparse_values,  // [TILE_K/16, TILE_N] packed
    device const uint32_t* metadata,       // [TILE_K/32, TILE_N]
    device const half* scales,
    threadgroup half (&B_tile)[TILE_K][TILE_N],
    uint thread_idx,
    uint group_size
) {
    // Zero-fill dense tile first.
    const uint dense_elems = TILE_K * TILE_N;
    for (uint idx = thread_idx; idx < dense_elems; idx += THREADS_PER_TG) {
        uint row = idx / TILE_N;
        uint col = idx - row * TILE_N;
        B_tile[row][col] = 0.0h;
    }

    const uint groups_per_col = TILE_K / 4u;
    const uint total_groups = groups_per_col * TILE_N;

    // Each thread handles a strided subset of sparsity groups.
    for (uint g = thread_idx; g < total_groups; g += THREADS_PER_TG) {
        uint col = g / groups_per_col;
        uint group_idx = g - col * groups_per_col;
        uint dense_k_base = group_idx * 4u;

        // --- Decode metadata ---
        uint meta_word_idx = group_idx / 8u;       // 8 groups per uint32
        uint meta_nibble_idx = group_idx & 7u;
        uint32_t meta_word = metadata[meta_word_idx * TILE_N + col];
        uint meta_nibble = (meta_word >> (meta_nibble_idx * 4u)) & 0xFu;

        uint pos0, pos1;
        decode_sparse_2in4(meta_nibble, pos0, pos1);

        // --- Load packed sparse values ---
        uint sparse_val_base = group_idx * 2u;  // 2 values per group
        uint value_word_idx = sparse_val_base / 8u;
        uint value_nibble_offset = sparse_val_base & 7u;
        uint32_t packed = sparse_values[value_word_idx * TILE_N + col];

        uint nibble0 = (packed >> (value_nibble_offset * 4u)) & 0xFu;
        uint nibble1 = (packed >> ((value_nibble_offset + 1u) * 4u)) & 0xFu;

        uint scale_group = dense_k_base / group_size;
        half scale = scales[scale_group * TILE_N + col];

        half val0 = dequant_fp4_e2m1(nibble0) * scale;
        half val1 = dequant_fp4_e2m1(nibble1) * scale;

        // --- Scatter to dense positions ---
        B_tile[dense_k_base + pos0][col] = val0;
        B_tile[dense_k_base + pos1][col] = val1;
    }
}


// ============================================================================
// 2:8 Pattern (75% sparse) - 2 values per 8-element block
// ============================================================================
//
// Metadata: two 3-bit position indices per block = 6 bits/block.
// A uint32 fits 5 blocks (30 bits used, 2 wasted).
// Covers 40 dense elements → 10 sparse values per uint32.
//
// Valid patterns: C(8,2) = 28 combinations.
// ============================================================================

/// Decode 2:8 metadata for one block.
/// Input: 6-bit field containing two 3-bit position indices.
inline void decode_sparse_2in8(uint metadata_6bit,
                                thread uint& pos0,
                                thread uint& pos1) {
    pos0 = metadata_6bit & POS_MASK_M8;
    pos1 = (metadata_6bit >> BITS_PER_POS_M8) & POS_MASK_M8;
}

/// Decode 5 blocks of 2:8 metadata from a packed uint32.
/// Each block uses 6 bits (two 3-bit indices). 30 bits used, 2 wasted.
/// Covers 40 dense elements → 10 sparse values.
inline void decode_sparse_2in8_x5(uint32_t metadata,
                                   thread uint2 positions[5]) {
    for (uint i = 0; i < 5; ++i) {
        uint field = (metadata >> (i * BITS_PER_BLOCK_2_8)) & 0x3Fu;
        positions[i] = uint2(field & POS_MASK_M8,
                             (field >> BITS_PER_POS_M8) & POS_MASK_M8);
    }
}

/// Unrolled 2:8 x5 decode.
inline void decode_sparse_2in8_x5_unrolled(uint32_t metadata,
                                            thread uint2 positions[5]) {
    positions[0] = uint2( metadata        & POS_MASK_M8, (metadata >>  3) & POS_MASK_M8);
    positions[1] = uint2((metadata >>  6) & POS_MASK_M8, (metadata >>  9) & POS_MASK_M8);
    positions[2] = uint2((metadata >> 12) & POS_MASK_M8, (metadata >> 15) & POS_MASK_M8);
    positions[3] = uint2((metadata >> 18) & POS_MASK_M8, (metadata >> 21) & POS_MASK_M8);
    positions[4] = uint2((metadata >> 24) & POS_MASK_M8, (metadata >> 27) & POS_MASK_M8);
}

/// Scatter 2 sparse values into an 8-element dense block.
inline void scatter_2in8(half val0, half val1,
                          uint pos0, uint pos1,
                          thread half (&block)[8]) {
    for (uint i = 0; i < 8; ++i) block[i] = 0.0h;
    block[pos0] = val0;
    block[pos1] = val1;
}

/// Encode two positions (0-7) into a 6-bit metadata field.
inline uint encode_sparse_2in8(uint pos0, uint pos1) {
    return (pos1 << BITS_PER_POS_M8) | pos0;
}

/// Pack 5 block metadata fields (2:8) into a single uint32.
inline uint32_t pack_metadata_2in8_x5(thread uint fields[5]) {
    uint32_t packed = 0;
    for (uint i = 0; i < 5; ++i) {
        packed |= (fields[i] & 0x3Fu) << (i * BITS_PER_BLOCK_2_8);
    }
    return packed;
}


// ============================================================================
// 4:8 Pattern (50% sparse) - 4 values per 8-element block
// ============================================================================
//
// Metadata: legacy packed indices (4 * 3-bit = 12 bits/block).
// A uint32 fits 2 blocks (24 bits used, 8 wasted).
// Covers 16 dense elements → 8 sparse values per uint32.
// NOTE: The generic N:M path supports a compact 7-bit combination index.
//
// Valid patterns: C(8,4) = 70 combinations.
// ============================================================================

/// Decode 4:8 metadata for one block.
/// Input: 12-bit field containing four 3-bit position indices.
/// Output: positions[0..3] are the 4 non-zero positions (0-7).
inline void decode_sparse_4in8(uint metadata_12bit,
                                thread uint positions[4]) {
    positions[0] =  metadata_12bit        & POS_MASK_M8;
    positions[1] = (metadata_12bit >>  3) & POS_MASK_M8;
    positions[2] = (metadata_12bit >>  6) & POS_MASK_M8;
    positions[3] = (metadata_12bit >>  9) & POS_MASK_M8;
}

/// Decode 2 blocks of 4:8 metadata from a packed uint32.
/// Each block uses 12 bits (four 3-bit indices). 24 bits used, 8 wasted.
/// Covers 16 dense elements → 8 sparse values.
inline void decode_sparse_4in8_x2(uint32_t metadata,
                                   thread uint positions[2][4]) {
    // Block 0: bits [11:0]
    uint field0 = metadata & 0xFFFu;
    positions[0][0] =  field0        & POS_MASK_M8;
    positions[0][1] = (field0 >>  3) & POS_MASK_M8;
    positions[0][2] = (field0 >>  6) & POS_MASK_M8;
    positions[0][3] = (field0 >>  9) & POS_MASK_M8;

    // Block 1: bits [23:12]
    uint field1 = (metadata >> BITS_PER_BLOCK_4_8_PACKED) & 0xFFFu;
    positions[1][0] =  field1        & POS_MASK_M8;
    positions[1][1] = (field1 >>  3) & POS_MASK_M8;
    positions[1][2] = (field1 >>  6) & POS_MASK_M8;
    positions[1][3] = (field1 >>  9) & POS_MASK_M8;
}

/// Scatter 4 sparse values into an 8-element dense block.
inline void scatter_4in8(thread half (&values)[4],
                          thread uint (&pos)[4],
                          thread half (&block)[8]) {
    for (uint i = 0; i < 8; ++i) block[i] = 0.0h;
    block[pos[0]] = values[0];
    block[pos[1]] = values[1];
    block[pos[2]] = values[2];
    block[pos[3]] = values[3];
}

/// Overload taking device pointer source values for GEMM integration.
inline void scatter_4in8(device const half* values,
                          thread uint (&pos)[4],
                          thread half (&block)[8]) {
    for (uint i = 0; i < 8; ++i) block[i] = 0.0h;
    block[pos[0]] = values[0];
    block[pos[1]] = values[1];
    block[pos[2]] = values[2];
    block[pos[3]] = values[3];
}

/// Encode four positions (0-7) into a 12-bit metadata field.
inline uint encode_sparse_4in8(uint pos0, uint pos1, uint pos2, uint pos3) {
    return pos0 | (pos1 << 3) | (pos2 << 6) | (pos3 << 9);
}

/// Pack 2 block metadata fields (4:8) into a single uint32.
inline uint32_t pack_metadata_4in8_x2(uint field0, uint field1) {
    return (field0 & 0xFFFu) | ((field1 & 0xFFFu) << BITS_PER_BLOCK_4_8_PACKED);
}


// ============================================================================
// Generic N:M decode template
// ============================================================================
//
// For patterns not covered by the specialized functions above, this template
// handles arbitrary N:M with runtime parameters. Less efficient than the
// compile-time specialized versions but useful for experimentation.
// ============================================================================

// Small combinatorics helper for 8 choose k (k <= 4).
constant constexpr uint COMB_8[9][5] = {
    {1, 0, 0, 0, 0},
    {1, 1, 0, 0, 0},
    {1, 2, 1, 0, 0},
    {1, 3, 3, 1, 0},
    {1, 4, 6, 4, 1},
    {1, 5, 10, 10, 5},
    {1, 6, 15, 20, 15},
    {1, 7, 21, 35, 35},
    {1, 8, 28, 56, 70},
};

inline uint comb8(uint n, uint k) {
    return (n < 9 && k < 5) ? COMB_8[n][k] : 0;
}

/// Decode 4:8 combination index (metadata is 7-bit index in [0, 69]).
inline void decode_sparse_comb_4in8(uint metadata_7bit,
                                     thread uint positions[4]) {
    uint idx = min(metadata_7bit, 69u);
    uint k = 4;
    uint start = 0;
    for (uint i = 0; i < 4; ++i) {
        for (uint v = start; v <= 8 - k; ++v) {
            uint count = comb8(8 - (v + 1), k - 1);
            if (idx >= count) {
                idx -= count;
            } else {
                positions[i] = v;
                start = v + 1;
                k -= 1;
                break;
            }
        }
    }
}

/// Decode packed N:M metadata field into positions.
/// Uses compact combination encoding for 4:8 and packed indices elsewhere.
template <uint N_VALS, uint M_BLOCK>
inline void decode_sparse_nm(uint metadata,
                              thread uint positions[N_VALS]) {
    if (M_BLOCK == 4 && N_VALS == 1) {
        positions[0] = metadata & POS_MASK_M4;
        return;
    }
    if (M_BLOCK == 4 && N_VALS == 2) {
        positions[0] = metadata & POS_MASK_M4;
        positions[1] = (metadata >> BITS_PER_POS_M4) & POS_MASK_M4;
        return;
    }
    if (M_BLOCK == 8 && N_VALS == 2) {
        positions[0] = metadata & POS_MASK_M8;
        positions[1] = (metadata >> BITS_PER_POS_M8) & POS_MASK_M8;
        return;
    }
    if (M_BLOCK == 8 && N_VALS == 4) {
        decode_sparse_comb_4in8(metadata, positions);
        return;
    }

    // Fallback: packed indices based on ceil(log2(M)).
    uint bpp = (M_BLOCK <= 2) ? 1 :
               (M_BLOCK <= 4) ? 2 :
               (M_BLOCK <= 8) ? 3 : 4;
    uint mask = (1u << bpp) - 1u;
    uint shifted = metadata;
    for (uint i = 0; i < N_VALS; ++i) {
        positions[i] = shifted & mask;
        shifted >>= bpp;
    }
}

/// Generic decode: extract N position indices of BPP bits each from a metadata
/// word starting at bit offset `base_bit`.
///
/// Template parameters:
///   N_VALS:  Number of non-zero values per block
///   BPP:    Bits per position index (ceil(log2(M_BLOCK)))
///
/// The metadata word is shifted right by base_bit before extraction.
template <uint N_VALS, uint BPP>
inline void decode_sparse_nm(uint32_t metadata,
                              uint base_bit,
                              thread uint positions[N_VALS]) {
    const uint mask = (1u << BPP) - 1u;
    uint shifted = metadata >> base_bit;
    for (uint i = 0; i < N_VALS; ++i) {
        positions[i] = shifted & mask;
        shifted >>= BPP;
    }
}

/// Generic scatter: place N sparse values into an M-element dense block.
template <uint N_VALS, uint M_BLOCK>
inline void scatter_nm(thread half (&values)[N_VALS],
                        thread uint (&positions)[N_VALS],
                        thread half (&block)[M_BLOCK]) {
    for (uint i = 0; i < M_BLOCK; ++i) block[i] = 0.0h;
    for (uint i = 0; i < N_VALS; ++i) block[positions[i]] = values[i];
}

/// Generic encode: pack N position indices into a single metadata field.
template <uint N_VALS, uint BPP>
inline uint encode_sparse_nm(thread uint positions[N_VALS]) {
    const uint mask = (1u << BPP) - 1u;
    uint encoded = 0;
    for (uint i = 0; i < N_VALS; ++i) {
        encoded |= (positions[i] & mask) << (i * BPP);
    }
    return encoded;
}


// ============================================================================
// Sparsity pattern descriptor (runtime dispatch)
// ============================================================================

struct SparsityPattern {
    uint n_vals;          // N: non-zero values per block
    uint m_block;         // M: block size
    uint bits_per_pos;    // ceil(log2(M))
    uint bits_per_block;  // packed/compact bits per block
    uint blocks_per_u32;  // floor(32 / bits_per_block)
    uint pos_mask;        // (1 << bits_per_pos) - 1
};

inline uint bits_per_block_nm(uint n_vals, uint m_block) {
    if (m_block == 4 && n_vals == 1) return BITS_PER_BLOCK_1_4;
    if (m_block == 4 && n_vals == 2) return BITS_PER_BLOCK_2_4;
    if (m_block == 8 && n_vals == 2) return BITS_PER_BLOCK_2_8;
    if (m_block == 8 && n_vals == 4) return BITS_PER_BLOCK_4_8;
    return 0;
}

/// Build a SparsityPattern descriptor from N and M parameters.
inline SparsityPattern make_sparsity_pattern(uint n_vals, uint m_block) {
    // ceil(log2(M)) computed via leading zeros
    // For M=4: bpp=2, for M=8: bpp=3, for M=16: bpp=4
    uint bpp = (m_block <= 2) ? 1 :
               (m_block <= 4) ? 2 :
               (m_block <= 8) ? 3 : 4;
    SparsityPattern p;
    p.n_vals = n_vals;
    p.m_block = m_block;
    p.bits_per_pos = bpp;
    uint bits_block = bits_per_block_nm(n_vals, m_block);
    p.bits_per_block = (bits_block > 0) ? bits_block : (n_vals * bpp);
    p.blocks_per_u32 = 32 / p.bits_per_block;
    p.pos_mask = (1u << bpp) - 1u;
    return p;
}

/// Runtime decode using SparsityPattern descriptor.
/// Decodes block_idx-th block from a packed uint32.
/// Writes N position indices to positions[].
/// Returns false if block_idx exceeds blocks_per_u32.
inline bool decode_sparse_runtime(SparsityPattern p,
                                   uint32_t metadata,
                                   uint block_idx,
                                   thread uint* positions) {
    if (block_idx >= p.blocks_per_u32) return false;
    uint base = block_idx * p.bits_per_block;
    uint field_mask = (p.bits_per_block >= 32) ? 0xFFFFFFFFu : ((1u << p.bits_per_block) - 1u);
    uint field = (metadata >> base) & field_mask;

    if (p.m_block == 4 && p.n_vals == 1) {
        decode_sparse_nm<1, 4>(field, positions);
        return true;
    }
    if (p.m_block == 4 && p.n_vals == 2) {
        decode_sparse_nm<2, 4>(field, positions);
        return true;
    }
    if (p.m_block == 8 && p.n_vals == 2) {
        decode_sparse_nm<2, 8>(field, positions);
        return true;
    }
    if (p.m_block == 8 && p.n_vals == 4) {
        decode_sparse_nm<4, 8>(field, positions);
        return true;
    }

    uint shifted = field;
    for (uint i = 0; i < p.n_vals; ++i) {
        positions[i] = shifted & p.pos_mask;
        shifted >>= p.bits_per_pos;
    }
    return true;
}

/// Runtime scatter: place n_vals sparse values into an m_block-element dense block.
/// Caller must ensure block[] has at least m_block elements.
inline void scatter_sparse_runtime(SparsityPattern p,
                                    device const half* values,
                                    thread uint* positions,
                                    thread half* block) {
    for (uint i = 0; i < p.m_block; ++i) block[i] = 0.0h;
    for (uint i = 0; i < p.n_vals; ++i) block[positions[i]] = values[i];
}


// ============================================================================
// Sparse GEMM kernel with runtime N:M pattern dispatch
// ============================================================================
//
// This kernel performs C = A @ decompress(B_sparse, metadata) where B_sparse
// contains only the N non-zero values per M-element block, and metadata
// encodes their positions.
//
// The kernel dynamically interprets the metadata based on the N:M pattern
// specified via buffer parameters. For maximum performance with known patterns,
// use the specialized 2:4 or 4:8 paths in the main GEMM kernel.
// ============================================================================

kernel void marlin_gemm_sparse_nm(
    device const half* A                  [[buffer(0)]],   // [M_dim, K]
    device const half* B_sparse           [[buffer(1)]],   // [K_sparse, N_dim]
    device const uint32_t* metadata       [[buffer(2)]],   // packed position indices
    device const half* scales             [[buffer(3)]],   // per-group dequant scales
    device half* C                        [[buffer(4)]],   // [M_dim, N_dim] output
    constant uint& M_dim                  [[buffer(5)]],   // rows of A / rows of C
    constant uint& N_dim                  [[buffer(6)]],   // cols of B / cols of C
    constant uint& K_dim                  [[buffer(7)]],   // cols of A (dense dimension)
    constant uint& group_size             [[buffer(8)]],   // quantization group size
    constant uint& n_vals                 [[buffer(9)]],   // N in N:M
    constant uint& m_block                [[buffer(10)]],  // M in N:M
    uint2 tgid                            [[threadgroup_position_in_grid]],
    uint2 tid                             [[thread_position_in_threadgroup]],
    uint simd_lane                        [[thread_index_in_simdgroup]],
    uint simd_id                          [[simdgroup_index_in_threadgroup]]
) {
    // Build pattern descriptor
    SparsityPattern pat = make_sparsity_pattern(n_vals, m_block);

    // K dimension in sparse storage: K_dim * n_vals / m_block
    uint K_sparse = K_dim * pat.n_vals / pat.m_block;

    // Tile assignment: each threadgroup handles a TILE_M x TILE_N output tile
    const uint TILE = 8;  // Sub-tile size matching simdgroup_matrix
    uint row_base = tgid.y * TILE;
    uint col_base = tgid.x * TILE;

    if (row_base >= M_dim || col_base >= N_dim) return;

    // Accumulator for one 8x8 output sub-tile
    half acc[TILE][TILE];
    for (uint r = 0; r < TILE; ++r)
        for (uint c = 0; c < TILE; ++c)
            acc[r][c] = 0.0h;

    // Walk along K dimension in units of m_block (one sparse block per step)
    uint num_blocks_k = K_dim / pat.m_block;
    uint meta_blocks_per_u32 = pat.blocks_per_u32;

    // Temporary buffers for position decoding and value expansion
    // Max N is 4 for supported patterns, max M is 8
    uint pos_buf[4];
    half dense_col[8];

    for (uint kb = 0; kb < num_blocks_k; ++kb) {
        // Which uint32 in the metadata stream and which sub-block within it
        uint meta_word_idx = kb / meta_blocks_per_u32;
        uint meta_sub_idx  = kb % meta_blocks_per_u32;

        // Metadata is laid out as: for each column, sequential blocks along K.
        // meta_word_idx advances along K, one word per blocks_per_u32 blocks.
        // Actual layout depends on the packing convention; here we use row-major
        // metadata where each column's metadata is contiguous.

        uint k_dense_base = kb * pat.m_block;  // Dense K position for this block

        // For each output column in our tile
        for (uint nc = 0; nc < TILE && (col_base + nc) < N_dim; ++nc) {
            uint col = col_base + nc;

            // Read metadata for this column's block
            // Layout: metadata[col * meta_words_per_col + meta_word_idx]
            uint meta_words_per_col = (num_blocks_k + meta_blocks_per_u32 - 1) / meta_blocks_per_u32;
            uint32_t meta_word = metadata[col * meta_words_per_col + meta_word_idx];

            // Decode positions for this sub-block
            decode_sparse_runtime(pat, meta_word, meta_sub_idx, pos_buf);

            // Read the N sparse values for this block from B_sparse
            // B_sparse layout: [K_sparse, N_dim], column-major sparse values
            uint sparse_k_base = kb * pat.n_vals;

            // Expand to dense: place values at decoded positions
            for (uint i = 0; i < pat.m_block; ++i) dense_col[i] = 0.0h;
            for (uint i = 0; i < pat.n_vals; ++i) {
                half sv = B_sparse[(sparse_k_base + i) * N_dim + col];
                // Apply dequant scale if quantized
                uint group_idx = (k_dense_base + pos_buf[i]) / group_size;
                half s = scales[group_idx * N_dim + col];
                dense_col[pos_buf[i]] = sv * s;
            }

            // Accumulate: for each row in our tile, dot with A row
            for (uint mr = 0; mr < TILE && (row_base + mr) < M_dim; ++mr) {
                uint row = row_base + mr;
                half dot = 0.0h;
                for (uint ki = 0; ki < pat.m_block; ++ki) {
                    dot += A[row * K_dim + k_dense_base + ki] * dense_col[ki];
                }
                acc[mr][nc] += dot;
            }
        }
    }

    // Write output tile
    for (uint mr = 0; mr < TILE && (row_base + mr) < M_dim; ++mr) {
        for (uint nc = 0; nc < TILE && (col_base + nc) < N_dim; ++nc) {
            C[(row_base + mr) * N_dim + (col_base + nc)] = acc[mr][nc];
        }
    }
}


// ============================================================================
// Test kernels
// ============================================================================

// --- 2:4 pattern tests (preserved from original) ---

kernel void test_decode_all_patterns(
    device uint* output          [[buffer(0)]],
    uint tid                     [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    const uint patterns[6] = {
        encode_sparse_2in4(0, 1),
        encode_sparse_2in4(0, 2),
        encode_sparse_2in4(0, 3),
        encode_sparse_2in4(1, 2),
        encode_sparse_2in4(1, 3),
        encode_sparse_2in4(2, 3),
    };

    for (uint i = 0; i < 6; ++i) {
        uint pos0, pos1;
        decode_sparse_2in4(patterns[i], pos0, pos1);
        output[i * 2]     = pos0;
        output[i * 2 + 1] = pos1;
    }
}

kernel void test_decode_packed_x8(
    device const uint32_t* input_metadata  [[buffer(0)]],
    device uint* output                    [[buffer(1)]],
    uint tid                               [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint2 positions[8];
    decode_sparse_metadata_x8(input_metadata[0], positions);

    for (uint i = 0; i < 8; ++i) {
        output[i * 2]     = positions[i].x;
        output[i * 2 + 1] = positions[i].y;
    }
}

kernel void test_scatter_2in4(
    device const half* sparse_values      [[buffer(0)]],
    device const uint* metadata_nibble    [[buffer(1)]],
    device half* dense_output             [[buffer(2)]],
    uint tid                              [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint pos0, pos1;
    decode_sparse_2in4(metadata_nibble[0], pos0, pos1);

    half block[4];
    scatter_2in4(sparse_values[0], sparse_values[1], pos0, pos1, block);

    for (uint i = 0; i < 4; ++i) {
        dense_output[i] = block[i];
    }
}

// --- 1:4 pattern test ---

/// Test kernel: decode 16 blocks of 1:4 metadata and write positions.
/// Output: 16 uint values (1 position per block).
kernel void test_decode_1in4_x16(
    device const uint32_t* input_metadata  [[buffer(0)]],
    device uint* output                    [[buffer(1)]],
    uint tid                               [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint positions[16];
    decode_sparse_1in4_x16(input_metadata[0], positions);

    for (uint i = 0; i < 16; ++i) {
        output[i] = positions[i];
    }
}

/// Test kernel: scatter 1 sparse value into a 4-element block.
kernel void test_scatter_1in4(
    device const half* sparse_value       [[buffer(0)]],
    device const uint* metadata_field     [[buffer(1)]],
    device half* dense_output             [[buffer(2)]],
    uint tid                              [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint pos = decode_sparse_1in4(metadata_field[0]);

    half block[4];
    scatter_1in4(sparse_value[0], pos, block);

    for (uint i = 0; i < 4; ++i) {
        dense_output[i] = block[i];
    }
}

// --- 2:8 pattern test ---

/// Test kernel: decode 5 blocks of 2:8 metadata.
/// Output: 10 uint values (5 blocks * 2 positions).
kernel void test_decode_2in8_x5(
    device const uint32_t* input_metadata  [[buffer(0)]],
    device uint* output                    [[buffer(1)]],
    uint tid                               [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint2 positions[5];
    decode_sparse_2in8_x5(input_metadata[0], positions);

    for (uint i = 0; i < 5; ++i) {
        output[i * 2]     = positions[i].x;
        output[i * 2 + 1] = positions[i].y;
    }
}

/// Test kernel: scatter 2 sparse values into an 8-element block using 2:8.
kernel void test_scatter_2in8(
    device const half* sparse_values      [[buffer(0)]],
    device const uint* metadata_field     [[buffer(1)]],
    device half* dense_output             [[buffer(2)]],
    uint tid                              [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint pos0, pos1;
    decode_sparse_2in8(metadata_field[0], pos0, pos1);

    half block[8];
    scatter_2in8(sparse_values[0], sparse_values[1], pos0, pos1, block);

    for (uint i = 0; i < 8; ++i) {
        dense_output[i] = block[i];
    }
}

// --- 4:8 pattern test ---

/// Test kernel: decode 2 blocks of 4:8 metadata from a uint32.
/// Output: 8 uint values (2 blocks * 4 positions).
kernel void test_decode_4in8_x2(
    device const uint32_t* input_metadata  [[buffer(0)]],
    device uint* output                    [[buffer(1)]],
    uint tid                               [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint positions[2][4];
    decode_sparse_4in8_x2(input_metadata[0], positions);

    for (uint b = 0; b < 2; ++b) {
        for (uint i = 0; i < 4; ++i) {
            output[b * 4 + i] = positions[b][i];
        }
    }
}

/// Test kernel: scatter 4 sparse values into an 8-element block using 4:8.
kernel void test_scatter_4in8(
    device const half* sparse_values      [[buffer(0)]],
    device const uint* metadata_field     [[buffer(1)]],
    device half* dense_output             [[buffer(2)]],
    uint tid                              [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint positions[4];
    decode_sparse_4in8(metadata_field[0], positions);

    half vals[4] = { sparse_values[0], sparse_values[1],
                     sparse_values[2], sparse_values[3] };
    half block[8];
    scatter_4in8(vals, positions, block);

    for (uint i = 0; i < 8; ++i) {
        dense_output[i] = block[i];
    }
}

// --- Generic N:M template test ---

/// Test kernel: generic N:M decode/scatter via template instantiation.
/// Tests 2:4 and 4:8 via the template path for correctness comparison.
kernel void test_generic_nm_decode(
    device const uint32_t* input_metadata  [[buffer(0)]],
    device uint* output                    [[buffer(1)]],
    constant uint& pattern_id              [[buffer(2)]],  // 0=2:4, 1=4:8
    uint tid                               [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    if (pattern_id == 0) {
        // 2:4 via template
        uint positions[2];
        decode_sparse_nm<2, 2>(input_metadata[0], 0, positions);
        output[0] = positions[0];
        output[1] = positions[1];
    } else if (pattern_id == 1) {
        // 4:8 via template
        uint positions[4];
        decode_sparse_nm<4, 3>(input_metadata[0], 0, positions);
        output[0] = positions[0];
        output[1] = positions[1];
        output[2] = positions[2];
        output[3] = positions[3];
    }
}

/// Test kernel: runtime N:M decode via SparsityPattern descriptor.
kernel void test_runtime_nm_decode(
    device const uint32_t* input_metadata  [[buffer(0)]],
    device uint* output                    [[buffer(1)]],
    constant uint& n_vals                  [[buffer(2)]],
    constant uint& m_block                 [[buffer(3)]],
    constant uint& block_idx               [[buffer(4)]],
    uint tid                               [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    SparsityPattern pat = make_sparsity_pattern(n_vals, m_block);
    uint positions[4];  // max N=4 for supported patterns
    decode_sparse_runtime(pat, input_metadata[0], block_idx, positions);

    for (uint i = 0; i < n_vals; ++i) {
        output[i] = positions[i];
    }
}

/// Test kernel: roundtrip encode/decode for all supported patterns.
/// Encodes known positions, decodes, writes to output for host verification.
kernel void test_roundtrip_all_patterns(
    device uint* output          [[buffer(0)]],
    uint tid                     [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint out_idx = 0;

    // --- 1:4 roundtrip ---
    {
        uint fields[16];
        for (uint i = 0; i < 16; ++i) fields[i] = encode_sparse_1in4(i % 4);
        uint32_t packed = pack_metadata_1in4_x16(fields);
        uint decoded[16];
        decode_sparse_1in4_x16(packed, decoded);
        for (uint i = 0; i < 16; ++i) output[out_idx++] = decoded[i];
    }

    // --- 2:4 roundtrip ---
    {
        uint nibbles[8];
        nibbles[0] = encode_sparse_2in4(0, 1);
        nibbles[1] = encode_sparse_2in4(0, 2);
        nibbles[2] = encode_sparse_2in4(0, 3);
        nibbles[3] = encode_sparse_2in4(1, 2);
        nibbles[4] = encode_sparse_2in4(1, 3);
        nibbles[5] = encode_sparse_2in4(2, 3);
        nibbles[6] = encode_sparse_2in4(0, 1);
        nibbles[7] = encode_sparse_2in4(2, 3);
        uint32_t packed = pack_metadata_x8(nibbles);
        uint2 positions[8];
        decode_sparse_metadata_x8(packed, positions);
        for (uint i = 0; i < 8; ++i) {
            output[out_idx++] = positions[i].x;
            output[out_idx++] = positions[i].y;
        }
    }

    // --- 2:8 roundtrip ---
    {
        uint fields[5];
        fields[0] = encode_sparse_2in8(0, 7);
        fields[1] = encode_sparse_2in8(1, 6);
        fields[2] = encode_sparse_2in8(2, 5);
        fields[3] = encode_sparse_2in8(3, 4);
        fields[4] = encode_sparse_2in8(0, 4);
        uint32_t packed = pack_metadata_2in8_x5(fields);
        uint2 positions[5];
        decode_sparse_2in8_x5(packed, positions);
        for (uint i = 0; i < 5; ++i) {
            output[out_idx++] = positions[i].x;
            output[out_idx++] = positions[i].y;
        }
    }

    // --- 4:8 roundtrip ---
    {
        uint field0 = encode_sparse_4in8(0, 2, 5, 7);
        uint field1 = encode_sparse_4in8(1, 3, 4, 6);
        uint32_t packed = pack_metadata_4in8_x2(field0, field1);
        uint positions[2][4];
        decode_sparse_4in8_x2(packed, positions);
        for (uint b = 0; b < 2; ++b)
            for (uint i = 0; i < 4; ++i)
                output[out_idx++] = positions[b][i];
    }
}
