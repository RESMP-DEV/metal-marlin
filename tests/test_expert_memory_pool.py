import pytest
import torch
from metal_marlin.moe.expert_memory_pool import ExpertMemoryPool, PoolConfig

def test_pool_allocation():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        device = "cpu"
    else:
        # Use CPU for deterministic logic testing without GPU dependency
        device = "cpu"

    # Create a small pool
    # expert_dim=64, hidden_dim=64 -> small experts
    # params = 3 * 64 * 64 = 12288 elements
    # 4-bit size = 12288 * 4 / 8 = 6144 bytes
    # 2-bit size = 3072 bytes
    # 8-bit size = 12288 bytes
    
    config = PoolConfig(
        pool_size_mb=1,  # 1MB is plenty for small tests
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    
    pool = ExpertMemoryPool(config)
    
    # Test allocation
    t1 = pool.allocate(layer_idx=0, expert_idx=0, bit_width=4)
    assert isinstance(t1, torch.Tensor)
    assert t1.numel() == 6144
    assert t1.dtype == torch.uint8
    
    # Test persistence (should return same slot)
    t2 = pool.allocate(layer_idx=0, expert_idx=0, bit_width=4)
    assert t1.data_ptr() == t2.data_ptr()
    
    # Test different expert
    t3 = pool.allocate(layer_idx=0, expert_idx=1, bit_width=4)
    assert t1.data_ptr() != t3.data_ptr()

def test_pool_eviction():
    device = "cpu"
    
    # Calculate sizes to force eviction
    # params = 3 * 64 * 64 = 12288 elements
    # 8-bit size = 12288 bytes
    # Pool size = 1MB. 8-bit allocation is 25% = 256KB
    # Max slots = 256KB / 12KB ~= 21 slots.
    
    # Let's make it tighter.
    # pool_size_mb = 1. 
    # 8-bit gets 0.25MB = 262144 bytes.
    # expert size = 12288 bytes.
    # max slots = 21.
    
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=100,
        expert_dim=64,
        hidden_dim=64
    )
    
    pool = ExpertMemoryPool(config)
    pool_8bit = pool.pools[8]
    
    # Fill up the pool
    allocated = []
    for i in range(pool_8bit.max_slots):
        t = pool.allocate(layer_idx=0, expert_idx=i, bit_width=8)
        allocated.append(t)
        
    assert len(pool_8bit.free_slots) == 0
    assert len(pool_8bit.used_slots) == pool_8bit.max_slots
    
    # Allocate one more - should evict LRU (index 0)
    # Access index 1 to make it recently used, so 0 remains LRU
    pool.allocate(layer_idx=0, expert_idx=1, bit_width=8)
    
    # Now allocate new one, should evict 0
    t_new = pool.allocate(layer_idx=0, expert_idx=100, bit_width=8)
    
    assert (0, 0) not in pool_8bit.used_slots
    assert (0, 100) in pool_8bit.used_slots

def test_mixed_precision():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Allocate in different pools
    t2 = pool.allocate(0, 1, 2)
    t4 = pool.allocate(0, 1, 4)
    t8 = pool.allocate(0, 1, 8)
    
    assert t2.numel() < t4.numel() < t8.numel()

def test_cpu_cache():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Add a tensor to CPU cache
    cpu_tensor = torch.zeros(6144, dtype=torch.uint8, device="cpu")
    pool.add_to_cpu_cache(layer_idx=0, expert_idx=0, cpu_tensor=cpu_tensor)
    
    # Register loader that returns cached data
    def make_loader(tensor):
        def loader():
            return tensor.clone()
        return loader
    
    pool.register_expert(0, 0, 4, make_loader(cpu_tensor))
    
    # Load should use loader
    gpu_buffer = pool.load_expert(0, 0)
    assert gpu_buffer is not None
    assert gpu_buffer.numel() == 6144

def test_expert_locking():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Allocate and lock an expert
    pool.allocate(0, 0, 4)
    pool.lock_expert(0, 0)
    
    # Should be in active_experts
    assert (0, 0) in pool.active_experts
    assert pool.active_experts[(0, 0)] == 1
    
    # Lock again (nested)
    pool.lock_expert(0, 0)
    assert pool.active_experts[(0, 0)] == 2
    
    # Unlock once
    pool.unlock_expert(0, 0)
    assert pool.active_experts[(0, 0)] == 1
    
    # Unlock completely
    pool.unlock_expert(0, 0)
    assert (0, 0) not in pool.active_experts

def test_statistics():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Make some allocations
    pool.allocate(0, 0, 4)
    pool.allocate(0, 1, 4)
    
    # Get stats
    stats = pool.get_stats()
    
    assert "pools" in stats
    assert "allocations" in stats
    assert stats["allocations"] >= 2
    assert "bits_4" in stats["pools"]

def test_get_expert_buffer():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Expert not loaded yet
    assert pool.get_expert_buffer(0, 0) is None
    
    # Allocate expert
    buffer = pool.allocate(0, 0, 4)
    
    # Should return same buffer
    retrieved = pool.get_expert_buffer(0, 0)
    assert retrieved is not None
    assert retrieved.data_ptr() == buffer.data_ptr()

def test_prefetch_integration():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Register some experts with loaders
    def make_loader(expert_id):
        def loader():
            return torch.zeros(6144, dtype=torch.uint8, device="cpu")
        return loader
    
    for eid in range(5):
        pool.register_expert(0, eid, 4, make_loader(eid))
    
    # Prefetch some experts
    pool.prefetch_experts(0, [0, 1, 2])
    
    # Check prefetcher recorded routing
    # (This tests that prefetch integration works)
    stats = pool.get_stats()
    assert "prefetcher" in stats

def test_defragmentation():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=100,
        expert_dim=64,
        hidden_dim=64,
        enable_defrag=True
    )
    pool = ExpertMemoryPool(config)
    
    # Allocate some experts
    for i in range(10):
        pool.allocate(0, i, 4)
    
    # Defrag should not raise
    pool.defragment()
    
    # Pool should still work
    buffer = pool.allocate(0, 10, 4)
    assert buffer is not None

def test_pool_stats_detailed():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Get initial stats
    initial_stats = pool.get_stats()
    
    # Allocate across different bit-widths
    pool.allocate(0, 0, 2)
    pool.allocate(0, 1, 4)
    pool.allocate(0, 2, 8)
    
    # Get stats again
    final_stats = pool.get_stats()
    
    # Check that stats changed
    assert final_stats["allocations"] > initial_stats["allocations"]
    
    # Check utilization
    pool_2bit = final_stats["pools"]["bits_2"]
    assert pool_2bit["used_slots"] > 0
    assert pool_2bit["utilization"] > 0

def test_bit_width_default():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=1,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64,
        default_bit_width=4
    )
    pool = ExpertMemoryPool(config)
    
    # Allocate without specifying bit width
    buffer = pool.allocate(0, 0)
    
    # Should use default (4-bit)
    expected_size = (3 * 64 * 64 * 4) // 8  # 6144 bytes
    assert buffer.numel() == expected_size

def test_multiple_pools():
    device = "cpu"
    config = PoolConfig(
        pool_size_mb=2,
        device=device,
        num_experts=10,
        expert_dim=64,
        hidden_dim=64
    )
    pool = ExpertMemoryPool(config)
    
    # Verify all three pools exist
    assert 2 in pool.pools
    assert 4 in pool.pools
    assert 8 in pool.pools
    
    # Each pool should have some slots
    assert pool.pools[2].max_slots > 0
    assert pool.pools[4].max_slots > 0
    assert pool.pools[8].max_slots > 0
    
    # Slot sizes should be different
    assert pool.pools[2].slot_size_bytes < pool.pools[4].slot_size_bytes
    assert pool.pools[4].slot_size_bytes < pool.pools[8].slot_size_bytes
