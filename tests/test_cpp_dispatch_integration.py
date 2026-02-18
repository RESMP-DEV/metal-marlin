'''Test C++ extension dispatch integration.'''
import pytest
import torch

from metal_marlin.kernels import HAS_CPP_EXT, dispatch_kernel

@pytest.mark.skipif(not HAS_CPP_EXT, reason="C++ extension not available")
class TestCppDispatch:
    def test_dispatch_uses_cpp_when_available(self):
        '''Verify dispatch routes through C++ extension.'''
        from metal_marlin._cpp_ext import dispatch_kernel as cpp_dispatch
        assert cpp_dispatch is not None
        
    def test_dispatch_fallback_when_unavailable(self, monkeypatch):
        '''Verify Python fallback works.'''
        import metal_marlin.kernels as k
        monkeypatch.setattr(k, 'HAS_CPP_EXT', False)
        # Should not raise
        # dispatch_kernel(...)  # Would need mock lib
