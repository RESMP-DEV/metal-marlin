
import pytest
from metal_marlin.paged.allocator import MultimodalBlockAllocator, TokenModality

class TestMultimodalBlockAllocator:
    def test_allocate_basic(self):
        allocator = MultimodalBlockAllocator(num_blocks=10)
        idx = allocator.allocate()
        assert idx == 0
        assert allocator.num_free == 9
        assert allocator.num_allocated == 1

    def test_allocate_image_blocks_contiguous(self):
        allocator = MultimodalBlockAllocator(num_blocks=10, block_size=16)
        # Allocate 32 tokens -> 2 blocks
        indices = allocator.allocate_image_blocks(num_tokens=32, image_hash="hash1")
        assert len(indices) == 2
        assert indices == [0, 1]
        assert allocator.num_free == 8

    def test_fragmentation_behavior(self):
        allocator = MultimodalBlockAllocator(num_blocks=10, block_size=16)
        
        # Create fragmentation: allocate 0, 1, 2, 3
        b0 = allocator.allocate() # 0
        b1 = allocator.allocate() # 1
        b2 = allocator.allocate() # 2
        b3 = allocator.allocate() # 3
        
        # Free 1 and 3 to create holes
        allocator.free(1)
        allocator.free(3)
        
        # Now free list has [1, 3] (sorted order)
        # Allocate 2 blocks for an image
        # Current behavior: pop(0) gives 1, then next allocate gives 3.
        # So we get [1, 3] which are NOT contiguous.
        
        indices = allocator.allocate_image_blocks(num_tokens=32, image_hash="hash2")
        assert indices == [1, 3] # This confirms current fragmentation behavior

    def test_allocate_contiguous_implementation(self):
        # This test is for the NEW functionality we are about to add.
        # It might fail or error before implementation.
        allocator = MultimodalBlockAllocator(num_blocks=10, block_size=16)
        
        if not hasattr(allocator, "allocate_contiguous"):
            pytest.skip("allocate_contiguous not implemented yet")

        # Create fragmentation: allocate 0, 1, 2, 3, 4
        ids = [allocator.allocate() for _ in range(5)]
        
        # Free 1 and 3 -> holes at 1 and 3.
        allocator.free(1)
        allocator.free(3)
        
        # Free 4 -> hole at 4.
        # Free list: 1, 3, 4. 
        # Contiguous runs: [1], [3, 4].
        
        # Allocate 2 blocks. Should skip 1 and take [3, 4].
        indices = allocator.allocate_contiguous(2)
        assert indices == [3, 4]
