
import threading
import time
import pytest
from unittest.mock import MagicMock
from metal_marlin.mmfp4_loader import MMFP4ModelLoader
from metal_marlin.memory.mmfp4_memory import MMFP4MemoryManager

class MockLoader:
    def load_layer(self, layer_idx, device):
        time.sleep(0.1)  # Simulate slow load
        return {"weight": MagicMock(numel=lambda: 1000, element_size=lambda: 4)}

def test_concurrent_layer_loading():
    # Setup
    manager = MMFP4MemoryManager(
        model_path="dummy",
        num_layers=10,
        unified_memory=False  # Avoid platform checks
    )
    # Mock the loader
    manager._loader = MockLoader()
    # Mock device to avoid torch issues if not present
    manager._device = "cpu"

    start_time = time.time()
    
    futures = []
    # Launch 5 loads
    for i in range(5):
        futures.append(manager.load_layer_async(i))
    
    # Wait for all
    for f in futures:
        f.result()
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Duration: {duration:.4f}s")
    
    # With global lock: 5 * 0.1 = 0.5s minimum
    # With per-layer lock: ~0.1s (parallel)
    # Allowing some overhead, if it's < 0.3s it's likely parallel
    assert duration < 0.4, f"Loading took too long ({duration:.4f}s), likely sequential"

if __name__ == "__main__":
    test_concurrent_layer_loading()
