
import pytest
import torch
from unittest.mock import MagicMock, patch
from metal_marlin.memory.mmfp4_memory import MMFP4MemoryManager

class TestMMFP4MemoryZeroCopy:
    
    @patch('metal_marlin.memory.mmfp4_memory.MMFP4ModelLoader')
    @patch('platform.system', return_value="Darwin")
    @patch('platform.machine', return_value="arm64")
    def test_load_layer_uses_zero_copy(self, mock_machine, mock_system, mock_loader_cls):
        """Test that load_layer is called with zero_copy=True on unified memory systems."""
        # Setup mock loader - make the class return our mock instance
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_layer.return_value = {"weight": torch.zeros(1)}
        mock_loader_cls.return_value = mock_loader_instance
        
        # Also need to mock Path.exists() so loader gets created
        with patch('pathlib.Path.exists', return_value=True):
            # Initialize manager with unified_memory=True
            manager = MMFP4MemoryManager(
                model_path="dummy_path",
                unified_memory=True
            )
            
            # Disable prefetcher to test the load_layer path
            manager._weight_prefetcher = None
            
            # Verify unified memory is enabled
            assert manager._unified_memory is True
            
            # Verify loader was created (it should use our mock)
            assert manager._loader is not None
            
            # Load layer
            future = manager.load_layer_async(0)
            result = future.result()
            
            # Verify load_layer was called with zero_copy=True
            # Use assert_any_call because prefetching may trigger multiple calls
            mock_loader_instance.load_layer.assert_any_call(
                0, 
                device=manager._device, 
                zero_copy=True
            )
            
            # Also verify the first call has zero_copy=True
            first_call = mock_loader_instance.load_layer.call_args_list[0]
            assert first_call.kwargs.get('zero_copy') is True
        
    @patch('platform.system', return_value="Darwin")
    @patch('platform.machine', return_value="arm64")
    def test_zero_copy_method(self, mock_machine, mock_system):
        """Test that _zero_copy method works correctly."""
        # Initialize manager (no loader needed)
        with patch('pathlib.Path.exists', return_value=False):
            manager = MMFP4MemoryManager(
                model_path="dummy_path",
                unified_memory=True
            )
            
            # Mock tensor on CPU
            cpu_tensor = torch.zeros(10)
            
            # Check if _zero_copy exists
            assert hasattr(manager, '_zero_copy')
            
            # Call _zero_copy and verify it returns a tensor
            res = manager._zero_copy(cpu_tensor)
            assert isinstance(res, torch.Tensor)
            
            # If device is available, it should be on device
            if torch.backends.mps.is_available():
                assert res.device.type == "mps"
        
    def test_zero_copy_mock_path(self):
        """Test the _zero_copy path when loader is not available."""
        # Test the else branch in _load_layer_impl where loader is None
        with patch('metal_marlin.memory.mmfp4_memory.MMFP4ModelLoader', side_effect=Exception("No loader")):
            with patch('platform.system', return_value="Darwin"), \
                 patch('platform.machine', return_value="arm64"), \
                 patch('pathlib.Path.exists', return_value=False):
                
                manager = MMFP4MemoryManager(
                    model_path="dummy_path",
                    unified_memory=True
                )
                
                # Spy on _zero_copy
                manager._zero_copy = MagicMock(wraps=manager._zero_copy)
                
                future = manager.load_layer_async(0)
                result = future.result()
                
                # Check that _zero_copy was called
                assert manager._zero_copy.called

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
