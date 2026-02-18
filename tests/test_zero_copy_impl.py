import torch
import unittest
from unittest.mock import MagicMock, patch
from metal_marlin.memory.mmfp4_memory import MMFP4MemoryManager, MLACompressionRatio

class TestZeroCopy(unittest.TestCase):
    def setUp(self):
        # Mocking init to avoid complex setup
        with patch('metal_marlin.memory.mmfp4_memory.MMFP4ModelLoader'):
            self.manager = MMFP4MemoryManager(
                model_path=".", 
                max_memory_gb=1.0, 
                unified_memory=True
            )
        # Mock _has_unified_memory to ensure we test the path regardless of hardware
        self.manager._unified_memory = True
        self.manager._device = "mps"

    def test_same_device(self):
        # Create a tensor on the same device (mocked)
        t = MagicMock(spec=torch.Tensor)
        t.device = MagicMock()
        t.device.type = "mps"
        
        result = self.manager._zero_copy(t)
        self.assertIs(result, t, "Should return same tensor if on correct device")

    def test_cpu_unified(self):
        # Create a tensor on CPU
        t = MagicMock(spec=torch.Tensor)
        t.device = MagicMock()
        t.device.type = "cpu"
        t.is_pinned.return_value = False
        
        pinned_t = MagicMock(spec=torch.Tensor)
        t.pin_memory.return_value = pinned_t
        
        mps_t = MagicMock(spec=torch.Tensor)
        pinned_t.to.return_value = mps_t
        pinned_t.device = MagicMock()
        pinned_t.device.type = "cpu" # Still cpu after pin_memory
        
        result = self.manager._zero_copy(t)
        
        t.pin_memory.assert_called_once()
        pinned_t.to.assert_called_with("mps", non_blocking=True)
        self.assertIs(result, mps_t)

    def test_cpu_unified_already_pinned(self):
        # Create a tensor on CPU, already pinned
        t = MagicMock(spec=torch.Tensor)
        t.device = MagicMock()
        t.device.type = "cpu"
        t.is_pinned.return_value = True
        
        mps_t = MagicMock(spec=torch.Tensor)
        t.to.return_value = mps_t
        
        result = self.manager._zero_copy(t)
        
        t.pin_memory.assert_not_called()
        t.to.assert_called_with("mps", non_blocking=True)
        self.assertIs(result, mps_t)

if __name__ == "__main__":
    unittest.main()
