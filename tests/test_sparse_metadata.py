"""Unit tests for 2:4 structured sparsity metadata encoding/decoding.

Validates:
  1. All 6 valid 2:4 patterns (C(4,2) = 6 ways to choose 2 positions from 4)
  2. Encode/decode roundtrip for single blocks
  3. Packed x8 encode/decode roundtrip (8 blocks in one uint32)
  4. Scatter correctness for all patterns
  5. Edge cases (identical encode values, boundary positions)

Usage:
    cd metal_marlin
    uv run pytest tests/test_sparse_metadata.py -v
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

# ============================================================================
# Reference implementations (matching sparse.metal logic exactly)
# ============================================================================


def ref_encode_sparse_2in4(pos0: int, pos1: int) -> int:
    """Encode two positions into a 4-bit metadata nibble.

    Format: bits [1:0] = pos0, bits [3:2] = pos1.
    """
    assert 0 <= pos0 <= 3
    assert 0 <= pos1 <= 3
    assert pos0 != pos1
    return (pos1 << 2) | pos0


def ref_decode_sparse_2in4(nibble: int) -> tuple[int, int]:
    """Decode a 4-bit metadata nibble into (pos0, pos1).

    Returns the two positions (0-3) of non-zero values.
    """
    pos0 = nibble & 0x3
    pos1 = (nibble >> 2) & 0x3
    return pos0, pos1


def ref_pack_metadata_x8(nibbles: list[int]) -> int:
    """Pack 8 metadata nibbles into a uint32."""
    assert len(nibbles) == 8
    packed = 0
    for i, n in enumerate(nibbles):
        packed |= (n & 0xF) << (i * 4)
    return packed & 0xFFFFFFFF


def ref_decode_metadata_x8(packed: int) -> list[tuple[int, int]]:
    """Decode a packed uint32 into 8 (pos0, pos1) pairs."""
    positions = []
    for i in range(8):
        nibble = (packed >> (i * 4)) & 0xF
        positions.append(ref_decode_sparse_2in4(nibble))
    return positions


def ref_scatter_2in4(val0: float, val1: float,
                     pos0: int, pos1: int) -> list[float]:
    """Scatter two sparse values into a 4-element dense block."""
    block = [0.0, 0.0, 0.0, 0.0]
    block[pos0] = val0
    block[pos1] = val1
    return block


# ============================================================================
# All 6 valid 2:4 patterns
# ============================================================================

ALL_2IN4_PATTERNS: list[tuple[int, int]] = list(combinations(range(4), 2))
# [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]


# ============================================================================
# Tests: encode/decode single block
# ============================================================================


class TestEncodeDecodeSingle:
    """Test single-block 2:4 metadata encode/decode."""

    @pytest.mark.parametrize("pos0,pos1", ALL_2IN4_PATTERNS)
    def test_roundtrip_all_patterns(self, pos0: int, pos1: int) -> None:
        """Encode then decode must recover original positions."""
        nibble = ref_encode_sparse_2in4(pos0, pos1)
        decoded_pos0, decoded_pos1 = ref_decode_sparse_2in4(nibble)
        assert decoded_pos0 == pos0
        assert decoded_pos1 == pos1

    @pytest.mark.parametrize("pos0,pos1", ALL_2IN4_PATTERNS)
    def test_nibble_range(self, pos0: int, pos1: int) -> None:
        """Encoded nibble must fit in 4 bits."""
        nibble = ref_encode_sparse_2in4(pos0, pos1)
        assert 0 <= nibble <= 15

    @pytest.mark.parametrize("pos0,pos1", ALL_2IN4_PATTERNS)
    def test_nibble_values_distinct(self, pos0: int, pos1: int) -> None:
        """Each valid pattern must produce a unique nibble."""
        nibbles = [ref_encode_sparse_2in4(p0, p1) for p0, p1 in ALL_2IN4_PATTERNS]
        assert len(set(nibbles)) == 6

    def test_expected_nibble_values(self) -> None:
        """Verify exact encoded values match Metal shader comments."""
        expected = {
            (0, 1): 0b0100,   # 4
            (0, 2): 0b1000,   # 8
            (0, 3): 0b1100,   # 12
            (1, 2): 0b1001,   # 9
            (1, 3): 0b1101,   # 13
            (2, 3): 0b1110,   # 14
        }
        for (pos0, pos1), expected_nibble in expected.items():
            assert ref_encode_sparse_2in4(pos0, pos1) == expected_nibble


# ============================================================================
# Tests: packed x8 encode/decode
# ============================================================================


class TestPackedX8:
    """Test vectorized 8-block packed metadata encoding/decoding."""

    def test_roundtrip_all_same_pattern(self) -> None:
        """8 identical blocks encode/decode correctly."""
        for pos0, pos1 in ALL_2IN4_PATTERNS:
            nibble = ref_encode_sparse_2in4(pos0, pos1)
            nibbles = [nibble] * 8
            packed = ref_pack_metadata_x8(nibbles)
            decoded = ref_decode_metadata_x8(packed)
            for i in range(8):
                assert decoded[i] == (pos0, pos1), f"block {i} failed"

    def test_roundtrip_mixed_patterns(self) -> None:
        """8 different blocks (cycling through patterns) roundtrip correctly."""
        nibbles = []
        patterns_used = []
        for i in range(8):
            pos0, pos1 = ALL_2IN4_PATTERNS[i % 6]
            patterns_used.append((pos0, pos1))
            nibbles.append(ref_encode_sparse_2in4(pos0, pos1))

        packed = ref_pack_metadata_x8(nibbles)
        decoded = ref_decode_metadata_x8(packed)

        for i in range(8):
            assert decoded[i] == patterns_used[i], f"block {i}: expected {patterns_used[i]}, got {decoded[i]}"

    def test_packed_fits_uint32(self) -> None:
        """Packed result fits in 32 bits."""
        nibbles = [ref_encode_sparse_2in4(p0, p1) for p0, p1 in ALL_2IN4_PATTERNS]
        nibbles.extend([nibbles[0], nibbles[1]])  # pad to 8
        packed = ref_pack_metadata_x8(nibbles)
        assert 0 <= packed <= 0xFFFFFFFF

    def test_packed_bit_layout(self) -> None:
        """Verify that nibble i occupies bits [4i+3 : 4i]."""
        nibbles = [0xA, 0xB, 0xC, 0xD, 0x1, 0x2, 0x3, 0x4]
        packed = ref_pack_metadata_x8(nibbles)
        for i, expected_nibble in enumerate(nibbles):
            extracted = (packed >> (i * 4)) & 0xF
            assert extracted == expected_nibble


# ============================================================================
# Tests: scatter correctness
# ============================================================================


class TestScatter:
    """Test sparse value scatter to dense block."""

    @pytest.mark.parametrize("pos0,pos1", ALL_2IN4_PATTERNS)
    def test_scatter_values_in_correct_positions(self, pos0: int, pos1: int) -> None:
        """Non-zero values appear at exactly the encoded positions."""
        val0, val1 = 1.5, -2.25
        block = ref_scatter_2in4(val0, val1, pos0, pos1)

        assert block[pos0] == val0
        assert block[pos1] == val1

    @pytest.mark.parametrize("pos0,pos1", ALL_2IN4_PATTERNS)
    def test_scatter_zeros_elsewhere(self, pos0: int, pos1: int) -> None:
        """Positions not encoded are zero."""
        val0, val1 = 3.0, 7.0
        block = ref_scatter_2in4(val0, val1, pos0, pos1)

        zero_positions = set(range(4)) - {pos0, pos1}
        for zp in zero_positions:
            assert block[zp] == 0.0

    @pytest.mark.parametrize("pos0,pos1", ALL_2IN4_PATTERNS)
    def test_scatter_produces_2in4_sparsity(self, pos0: int, pos1: int) -> None:
        """Output block has exactly 2 non-zero values out of 4."""
        block = ref_scatter_2in4(1.0, 2.0, pos0, pos1)
        nonzero_count = sum(1 for v in block if v != 0.0)
        assert nonzero_count == 2


# ============================================================================
# Tests: end-to-end metadata → scatter pipeline
# ============================================================================


class TestEndToEnd:
    """Test full encode → decode → scatter pipeline."""

    @pytest.mark.parametrize("pos0,pos1", ALL_2IN4_PATTERNS)
    def test_encode_decode_scatter(self, pos0: int, pos1: int) -> None:
        """Full pipeline: encode positions, decode, scatter values."""
        val0 = np.float16(1.5)
        val1 = np.float16(-3.0)

        nibble = ref_encode_sparse_2in4(pos0, pos1)
        dec_pos0, dec_pos1 = ref_decode_sparse_2in4(nibble)
        block = ref_scatter_2in4(float(val0), float(val1), dec_pos0, dec_pos1)

        expected = [0.0] * 4
        expected[pos0] = float(val0)
        expected[pos1] = float(val1)
        assert block == expected

    def test_packed_decode_scatter_multiple_blocks(self) -> None:
        """Decode 8 blocks from packed uint32 and scatter each."""
        rng = np.random.default_rng(42)
        patterns = [ALL_2IN4_PATTERNS[i % 6] for i in range(8)]
        nibbles = [ref_encode_sparse_2in4(p0, p1) for p0, p1 in patterns]
        packed = ref_pack_metadata_x8(nibbles)

        sparse_values = rng.standard_normal(16).astype(np.float16)

        decoded = ref_decode_metadata_x8(packed)
        dense_output = np.zeros(32, dtype=np.float16)

        for block_idx in range(8):
            pos0, pos1 = decoded[block_idx]
            v0 = sparse_values[block_idx * 2]
            v1 = sparse_values[block_idx * 2 + 1]
            dense_offset = block_idx * 4
            dense_output[dense_offset + pos0] = v0
            dense_output[dense_offset + pos1] = v1

        # Verify sparsity pattern
        for block_idx in range(8):
            dense_offset = block_idx * 4
            block = dense_output[dense_offset:dense_offset + 4]
            nonzero_count = np.count_nonzero(block)
            assert nonzero_count == 2, f"block {block_idx}: {nonzero_count} nonzero"

        # Verify values match
        for block_idx in range(8):
            pos0, pos1 = patterns[block_idx]
            dense_offset = block_idx * 4
            np.testing.assert_equal(
                dense_output[dense_offset + pos0],
                sparse_values[block_idx * 2]
            )
            np.testing.assert_equal(
                dense_output[dense_offset + pos1],
                sparse_values[block_idx * 2 + 1]
            )

    def test_random_patterns_stress(self) -> None:
        """Stress test with 1000 random blocks."""
        rng = np.random.default_rng(123)

        for _ in range(1000):
            pattern_idx = rng.integers(0, 6)
            pos0, pos1 = ALL_2IN4_PATTERNS[pattern_idx]
            val0 = float(rng.standard_normal())
            val1 = float(rng.standard_normal())

            nibble = ref_encode_sparse_2in4(pos0, pos1)
            dec_pos0, dec_pos1 = ref_decode_sparse_2in4(nibble)
            block = ref_scatter_2in4(val0, val1, dec_pos0, dec_pos1)

            assert block[pos0] == val0
            assert block[pos1] == val1
            zero_positions = set(range(4)) - {pos0, pos1}
            for zp in zero_positions:
                assert block[zp] == 0.0


# ============================================================================
# Tests: pattern enumeration completeness
# ============================================================================


class TestPatternCompleteness:
    """Verify all valid 2:4 patterns are accounted for."""

    def test_exactly_6_patterns(self) -> None:
        """C(4,2) = 6 valid 2:4 patterns exist."""
        assert len(ALL_2IN4_PATTERNS) == 6

    def test_all_pairs_covered(self) -> None:
        """Every pair of distinct positions in [0,3] is represented."""
        expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        actual = set(ALL_2IN4_PATTERNS)
        assert actual == expected

    def test_all_nibbles_unique(self) -> None:
        """Each pattern maps to a distinct nibble value."""
        nibbles = [ref_encode_sparse_2in4(p0, p1) for p0, p1 in ALL_2IN4_PATTERNS]
        assert len(set(nibbles)) == 6

    def test_all_nibbles_decodable(self) -> None:
        """Every encoded nibble decodes back to a valid (distinct) pair."""
        for pos0, pos1 in ALL_2IN4_PATTERNS:
            nibble = ref_encode_sparse_2in4(pos0, pos1)
            dec0, dec1 = ref_decode_sparse_2in4(nibble)
            assert dec0 != dec1, f"nibble {nibble:#06b} decoded to same position"
            assert 0 <= dec0 <= 3
            assert 0 <= dec1 <= 3
