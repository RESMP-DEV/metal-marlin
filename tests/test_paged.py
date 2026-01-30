"""Tests for paged KV cache block."""

import numpy as np
import pytest

from metal_marlin.paged import KVBlock, KVBlockConfig


class TestKVBlockConfig:
    def test_defaults(self):
        cfg = KVBlockConfig()
        assert cfg.block_size == 16
        assert cfg.num_heads == 32
        assert cfg.head_dim == 128
        assert cfg.dtype == np.dtype(np.float16)

    def test_memory_bytes_fp16(self):
        cfg = KVBlockConfig(block_size=16, num_heads=32, head_dim=128)
        # 2 * 16 * 32 * 128 * 2 = 262144
        assert cfg.memory_bytes == 2 * 16 * 32 * 128 * 2

    def test_memory_bytes_fp32(self):
        cfg = KVBlockConfig(block_size=16, num_heads=32, head_dim=128, dtype=np.dtype(np.float32))
        assert cfg.memory_bytes == 2 * 16 * 32 * 128 * 4

    def test_shape(self):
        cfg = KVBlockConfig(block_size=8, num_heads=4, head_dim=64)
        assert cfg.shape == (2, 8, 4, 64)

    def test_frozen(self):
        cfg = KVBlockConfig()
        with pytest.raises(Exception):
            cfg.block_size = 32  # type: ignore[misc]


class TestKVBlockLifecycle:
    def test_init_unallocated(self):
        block = KVBlock()
        assert block.data is None
        assert block.token_count == 0
        assert block.is_empty
        assert not block.is_full

    def test_allocate(self):
        cfg = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        assert block.data is not None
        assert block.data.shape == (2, 4, 2, 8)
        assert block.data.dtype == np.float16

    def test_allocate_resets_count(self):
        cfg = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        k = np.ones((2, 8), dtype=np.float16)
        v = np.ones((2, 8), dtype=np.float16)
        block.append_kv(k, v)
        block.allocate()
        assert block.token_count == 0

    def test_reset(self):
        cfg = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        block.acquire()
        k = np.ones((2, 8), dtype=np.float16)
        v = np.ones((2, 8), dtype=np.float16)
        block.append_kv(k, v)
        block.reset()
        assert block.token_count == 0
        assert block.ref_count == 0
        assert block.data is not None  # Still allocated


class TestKVBlockAppend:
    @pytest.fixture
    def small_block(self):
        cfg = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        return block

    def test_append_single(self, small_block: KVBlock):
        k = np.ones((2, 8), dtype=np.float16)
        v = np.full((2, 8), 2.0, dtype=np.float16)
        remaining = small_block.append_kv(k, v)
        assert remaining == 3
        assert small_block.token_count == 1
        assert not small_block.is_full

    def test_append_until_full(self, small_block: KVBlock):
        for i in range(4):
            k = np.full((2, 8), float(i), dtype=np.float16)
            v = np.full((2, 8), float(i + 10), dtype=np.float16)
            remaining = small_block.append_kv(k, v)
            assert remaining == 3 - i
        assert small_block.is_full
        assert small_block.remaining == 0

    def test_append_overflow_raises(self, small_block: KVBlock):
        for _ in range(4):
            k = np.ones((2, 8), dtype=np.float16)
            v = np.ones((2, 8), dtype=np.float16)
            small_block.append_kv(k, v)
        with pytest.raises(RuntimeError, match="Block is full"):
            small_block.append_kv(
                np.ones((2, 8), dtype=np.float16),
                np.ones((2, 8), dtype=np.float16),
            )

    def test_append_unallocated_raises(self):
        block = KVBlock()
        with pytest.raises(RuntimeError, match="not allocated"):
            block.append_kv(
                np.ones((2, 8), dtype=np.float16),
                np.ones((2, 8), dtype=np.float16),
            )

    def test_append_preserves_values(self, small_block: KVBlock):
        k0 = np.full((2, 8), 1.0, dtype=np.float16)
        v0 = np.full((2, 8), 2.0, dtype=np.float16)
        k1 = np.full((2, 8), 3.0, dtype=np.float16)
        v1 = np.full((2, 8), 4.0, dtype=np.float16)

        small_block.append_kv(k0, v0)
        small_block.append_kv(k1, v1)

        keys, values = small_block.get_kv()

        assert keys.shape == (2, 2, 8)
        assert values.shape == (2, 2, 8)
        assert np.allclose(keys[0], k0)
        assert np.allclose(keys[1], k1)
        assert np.allclose(values[0], v0)
        assert np.allclose(values[1], v1)


class TestKVBlockBatchAppend:
    @pytest.fixture
    def block(self):
        cfg = KVBlockConfig(block_size=8, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        return block

    def test_batch_append(self, block: KVBlock):
        keys = np.ones((3, 2, 8), dtype=np.float16)
        values = np.full((3, 2, 8), 2.0, dtype=np.float16)
        remaining = block.append_kv_batch(keys, values)
        assert remaining == 5
        assert block.token_count == 3

    def test_batch_overflow_raises(self, block: KVBlock):
        keys = np.ones((9, 2, 8), dtype=np.float16)
        values = np.ones((9, 2, 8), dtype=np.float16)
        with pytest.raises(RuntimeError, match="exceeds remaining"):
            block.append_kv_batch(keys, values)

    def test_batch_preserves_values(self, block: KVBlock):
        keys = np.arange(3 * 2 * 8, dtype=np.float16).reshape(3, 2, 8)
        values = np.arange(3 * 2 * 8, dtype=np.float16).reshape(3, 2, 8) + 100

        block.append_kv_batch(keys, values)
        got_k, got_v = block.get_kv()

        assert got_k.shape == (3, 2, 8)
        assert np.allclose(got_k, keys)
        assert np.allclose(got_v, values)

    def test_batch_after_single(self, block: KVBlock):
        k0 = np.full((2, 8), 99.0, dtype=np.float16)
        v0 = np.full((2, 8), 88.0, dtype=np.float16)
        block.append_kv(k0, v0)

        keys = np.ones((3, 2, 8), dtype=np.float16)
        values = np.full((3, 2, 8), 2.0, dtype=np.float16)
        block.append_kv_batch(keys, values)

        assert block.token_count == 4
        got_k, got_v = block.get_kv()
        assert np.allclose(got_k[0], k0)
        assert np.allclose(got_k[1:], keys)


class TestKVBlockGetKV:
    def test_get_kv_empty(self):
        cfg = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        keys, values = block.get_kv()
        assert keys.shape == (0, 2, 8)
        assert values.shape == (0, 2, 8)

    def test_get_kv_unallocated_raises(self):
        block = KVBlock()
        with pytest.raises(RuntimeError, match="not allocated"):
            block.get_kv()


class TestKVBlockRefCount:
    def test_acquire_release(self):
        block = KVBlock()
        assert block.ref_count == 0
        block.acquire()
        assert block.ref_count == 1
        block.acquire()
        assert block.ref_count == 2
        remaining = block.release()
        assert remaining == 1
        assert block.ref_count == 1

    def test_release_floor_zero(self):
        block = KVBlock()
        remaining = block.release()
        assert remaining == 0
        assert block.ref_count == 0


class TestKVBlockCopy:
    def test_copy_shares_initial_data(self):
        cfg = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        k = np.ones((2, 8), dtype=np.float16)
        v = np.full((2, 8), 2.0, dtype=np.float16)
        block.append_kv(k, v)

        clone = block.copy()
        assert clone.token_count == 1
        assert clone.ref_count == 0  # Fresh ref count

        # Mutating clone doesn't affect original
        clone.append_kv(
            np.full((2, 8), 9.0, dtype=np.float16), np.full((2, 8), 9.0, dtype=np.float16)
        )
        assert clone.token_count == 2
        assert block.token_count == 1

    def test_copy_unallocated(self):
        block = KVBlock()
        clone = block.copy()
        assert clone.data is None
        assert clone.token_count == 0


class TestKVBlockRepr:
    def test_repr_unallocated(self):
        block = KVBlock()
        r = repr(block)
        assert "allocated=no" in r
        assert "tokens=0/16" in r

    def test_repr_allocated(self):
        cfg = KVBlockConfig(block_size=4, num_heads=2, head_dim=8)
        block = KVBlock(config=cfg)
        block.allocate()
        r = repr(block)
        assert "allocated=yes" in r
        assert "tokens=0/4" in r
        assert "heads=2" in r
        assert "dim=8" in r
