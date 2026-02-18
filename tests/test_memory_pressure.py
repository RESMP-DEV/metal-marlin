import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add contrib/metal_marlin to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from metal_marlin.mmfp4_loader import WeightPrefetcher, MMFP4ModelLoader
from metal_marlin.memory.mmfp4_memory import MemoryPressureConfig, get_global_memory_pressure_monitor, _global_monitor_lock

class TestMemoryPressure(unittest.TestCase):
    def setUp(self):
        self.loader = MagicMock(spec=MMFP4ModelLoader)
        self.loader.load_tensor.return_value = torch.zeros(1)
        
        # Reset global monitor
        import metal_marlin.memory.mmfp4_memory as mm
        with mm._global_monitor_lock:
            mm._global_pressure_monitor = None

    @patch('metal_marlin.memory.mmfp4_memory.psutil.virtual_memory')
    def test_pressure_critical(self, mock_vm):
        # Mock critical memory (e.g. 100MB free)
        mock_vm.return_value.available = 100 * 1024 * 1024
        mock_vm.return_value.total = 16 * 1024 * 1024 * 1024
        
        config = MemoryPressureConfig(critical_threshold_mb=1024, check_interval_seconds=0.0)
        monitor = get_global_memory_pressure_monitor(config)
        monitor._update_stats() # Force update
        
        prefetcher = WeightPrefetcher(self.loader)
        # Override monitor to use our configured one (though global should be set)
        prefetcher._pressure_monitor = monitor
        
        # Prefetch should be blocked
        futures = prefetcher.prefetch(['tensor1'])
        self.assertEqual(len(futures), 0)
        
    @patch('metal_marlin.memory.mmfp4_memory.psutil.virtual_memory')
    def test_pressure_normal(self, mock_vm):
        # Mock normal memory (e.g. 8GB free)
        mock_vm.return_value.available = 8 * 1024 * 1024 * 1024
        mock_vm.return_value.total = 16 * 1024 * 1024 * 1024
        
        config = MemoryPressureConfig(critical_threshold_mb=1024, check_interval_seconds=0.0)
        monitor = get_global_memory_pressure_monitor(config)
        monitor._update_stats()
        
        prefetcher = WeightPrefetcher(self.loader)
        prefetcher._pressure_monitor = monitor
        
        # Prefetch should be allowed
        futures = prefetcher.prefetch(['tensor1'])
        self.assertEqual(len(futures), 1)

if __name__ == '__main__':
    unittest.main()
