
import unittest
from contrib.metal_marlin.metal_marlin.paged.allocator import MultimodalBlockAllocator, TokenModality

class TestMultimodalBlockAllocatorOptimization(unittest.TestCase):
    def test_fragmentation_and_contiguous_allocation(self):
        # Initialize with 100 blocks
        allocator = MultimodalBlockAllocator(num_blocks=100)
        
        # Allocate some scattered blocks to create fragmentation
        # Allocate 0, 2, 4, 6...
        allocated_indices = []
        for i in range(0, 50, 2):
            # We want to allocate block i.
            # Since free list is sorted (0, 1, 2...), we can manipulate it
            # But here we just allocate everything then free odds
            pass
            
        # Strategy: allocate 50 blocks. Then free every other one.
        indices = []
        for _ in range(50):
            idx = allocator.allocate()
            indices.append(idx)
            
        # Free odd indices: 1, 3, 5...
        for i in range(1, 50, 2):
            allocator.free(indices[i])
            
        # Now free list should have holes.
        # Current free list size: 50 (initial) + 25 (freed) = 75.
        # Holes at 0, 2, 4... are taken. 1, 3, 5... are free. 50-99 are free.
        
        # Try to allocate 4 image blocks.
        # Ideally we want them contiguous.
        # With current implementation (popleft), it will pick the first available: 1, 3, 5, 7.
        # These are NOT contiguous.
        
        image_hash = "hash1"
        block_indices = allocator.allocate_image_blocks(num_tokens=16*4, image_hash=image_hash)
        
        print(f"Allocated indices: {block_indices}")
        
        # Check contiguity
        is_contiguous = True
        if block_indices:
            sorted_indices = sorted(block_indices)
            for i in range(len(sorted_indices) - 1):
                if sorted_indices[i+1] != sorted_indices[i] + 1:
                    is_contiguous = False
                    break
        
        # We expect it to FAIL contiguity currently, because it just pops from free list.
        # But for optimization, we might prefer contiguous blocks if available (e.g. from 50-99).
        
        # Also check unused constants logic
        # allocator._COMPACT_THRESHOLD
        
    def test_unused_compaction_logic(self):
        allocator = MultimodalBlockAllocator(num_blocks=10)
        # Check if _rebuild_free_runs exists but is not used
        self.assertTrue(hasattr(allocator, '_free_runs'))
        # It is initialized to None
        
if __name__ == '__main__':
    unittest.main()
