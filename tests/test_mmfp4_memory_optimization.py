
import unittest
import torch
import time
from metal_marlin.memory.mmfp4_memory import MMFP4MemoryManager, ExpertMetadata

class TestMMFP4MemoryOptimization(unittest.TestCase):
    def setUp(self):
        # Setup basic params
        self.num_layers = 2
        self.num_experts = 4
        self.cache_size = 2
        self.memory_manager = MMFP4MemoryManager(
            model_path="dummy_path",
            max_memory_gb=1.0,
            num_layers=self.num_layers,
            num_experts_per_layer=self.num_experts,
            expert_cache_size=self.cache_size,
            unified_memory=False # Force manual management
        )

    def test_expert_weight_cache_existence(self):
        """Test that _expert_weight_cache exists."""
        self.assertTrue(hasattr(self.memory_manager, "_expert_weight_cache"), 
                        "MMFP4MemoryManager should have _expert_weight_cache attribute")
        
    def test_smart_caching_behavior(self):
        """Test that expert weights are cached and evicted correctly."""
        # Mock loader function if needed, or we rely on internal logic we are about to add.
        # Since I'm adding the logic, I'll assume I add a method to get weights.
        
        # We need to simulate loading experts.
        # If the API doesn't have load_expert, I might need to add it or use an internal method.
        # Let's assume I add get_expert_weights(layer_idx, expert_idx)
        
        if not hasattr(self.memory_manager, "get_expert_weights"):
             self.skipTest("get_expert_weights not implemented yet")

        # Load expert (0, 0)
        w00 = self.memory_manager.get_expert_weights(0, 0)
        self.assertIsNotNone(w00)
        self.assertIn((0, 0), self.memory_manager._expert_weight_cache)
        
        # Load expert (0, 1)
        w01 = self.memory_manager.get_expert_weights(0, 1)
        self.assertIn((0, 1), self.memory_manager._expert_weight_cache)
        
        # Cache is full (size 2). 
        # Cache state: (0, 0) -> (0, 1) (most recent)
        
        # Load expert (0, 2). Should evict (0, 0)
        w02 = self.memory_manager.get_expert_weights(0, 2)
        self.assertIn((0, 2), self.memory_manager._expert_weight_cache)
        self.assertNotIn((0, 0), self.memory_manager._expert_weight_cache)
        
        # Access (0, 1) again. Should make it most recent.
        w01_again = self.memory_manager.get_expert_weights(0, 1)
        # Cache state: (0, 2) -> (0, 1)
        
        # Load expert (0, 3). Should evict (0, 2)
        w03 = self.memory_manager.get_expert_weights(0, 3)
        self.assertIn((0, 3), self.memory_manager._expert_weight_cache)
        self.assertNotIn((0, 2), self.memory_manager._expert_weight_cache)
        self.assertIn((0, 1), self.memory_manager._expert_weight_cache)

if __name__ == "__main__":
    unittest.main()
