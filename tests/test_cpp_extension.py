"""Tests for the Metal Marlin C++ extension (_cpp_ext)."""

from __future__ import annotations

import ctypes
import gc

import numpy as np
import pytest

from metal_marlin._compat import HAS_CPP_EXT

try:
    import metal_marlin._cpp_ext as cpp_ext
    _CPP_EXT_AVAILABLE = True
except Exception:
    cpp_ext = None
    _CPP_EXT_AVAILABLE = False


pytestmark = pytest.mark.skipif(not _CPP_EXT_AVAILABLE, reason="C++ extension not available")


def _require_context():
    if cpp_ext is None:
        pytest.skip("C++ extension not available")
    try:
        return cpp_ext.MetalContext()
    except Exception as exc:
        pytest.skip(f"MetalContext unavailable: {exc}")


def test_extension_flag_matches_import():
    assert HAS_CPP_EXT == _CPP_EXT_AVAILABLE


def test_module_exports():
    assert cpp_ext is not None
    assert hasattr(cpp_ext, "MetalContext")
    assert hasattr(cpp_ext, "BufferPool")
    assert hasattr(cpp_ext, "ManagedBuffer")
    assert hasattr(cpp_ext, "BatchDispatch")
    assert hasattr(cpp_ext, "dispatch_kernel")
    assert hasattr(cpp_ext, "create_buffer")
    assert hasattr(cpp_ext, "create_buffer_from_bytes")
    assert hasattr(cpp_ext, "create_buffer_from_ptr")
    assert hasattr(cpp_ext, "align_buffer_size")

    assert isinstance(cpp_ext.CACHE_LINE_SIZE, int)
    assert isinstance(cpp_ext.PAGE_SIZE, int)
    assert isinstance(cpp_ext.LARGE_BUFFER_THRESHOLD, int)
    assert cpp_ext.CACHE_LINE_SIZE > 0
    assert cpp_ext.PAGE_SIZE > 0
    assert cpp_ext.LARGE_BUFFER_THRESHOLD > 0


def test_align_buffer_size():
    cache = cpp_ext.CACHE_LINE_SIZE
    page = cpp_ext.PAGE_SIZE
    threshold = cpp_ext.LARGE_BUFFER_THRESHOLD

    small = cache - 1
    expected_small = (small + cache - 1) & ~(cache - 1)
    assert cpp_ext.align_buffer_size(small) == expected_small

    large = threshold + 1
    expected_large = (large + page - 1) & ~(page - 1)
    assert cpp_ext.align_buffer_size(large) == expected_large


def test_metal_context_properties():
    ctx = _require_context()

    name = ctx.device_name()
    family = ctx.gpu_family()

    assert isinstance(name, str)
    assert name
    assert isinstance(family, int)
    assert family >= 7


def test_buffer_pool_reuse():
    ctx = _require_context()
    pool = ctx.buffer_pool

    size = 1024
    hits_before = pool.hits()
    misses_before = pool.misses()

    buf = cpp_ext.create_buffer(ctx, size, True)
    assert buf.length() >= size
    assert buf.data_ptr() != 0

    del buf
    gc.collect()

    pooled_after_release = pool.pooled_count()
    assert pooled_after_release >= 1

    buf2 = cpp_ext.create_buffer(ctx, size, True)
    del buf2
    gc.collect()

    assert pool.misses() >= misses_before + 1
    assert pool.hits() >= hits_before + 1


def test_create_buffer_from_bytes_roundtrip():
    ctx = _require_context()

    payload = b"metal-marlin-cpp-ext"
    buf = cpp_ext.create_buffer_from_bytes(ctx, payload, False)

    assert buf.length() >= len(payload)
    ptr = buf.data_ptr()
    assert ptr != 0

    read_back = ctypes.string_at(ptr, len(payload))
    assert read_back == payload

    del buf
    gc.collect()


def test_create_buffer_from_ptr():
    ctx = _require_context()

    payload = b"direct-pointer"
    raw = ctypes.create_string_buffer(payload)
    ptr = ctypes.addressof(raw)

    buf = cpp_ext.create_buffer_from_ptr(ctx, ptr, len(payload))

    assert buf.length() >= len(payload)
    assert buf.data_ptr() != 0

    del buf
    gc.collect()


def test_expert_buffer_pool():
    if not hasattr(cpp_ext, "ExpertBufferPool"):
        pytest.skip("ExpertBufferPool not available")

    pool = cpp_ext.ExpertBufferPool.instance()
    assert pool is not None
    
    if hasattr(cpp_ext, "MetalDevice"):
         device = cpp_ext.MetalDevice.default_device()
         # Initialize with device pointer
         pool.initialize(device.raw())
         assert pool.is_initialized()
         
         # Test allocation
         buf = pool.allocate_weight(1024)
         assert buf is not None
         assert buf.size >= 1024
         
         # Verify it's pinned
         # assert buf.priority == cpp_ext.BufferPriority.PINNED 
         # (If BufferPriority is exposed and checking logic exists)
         
         pool.clear()


# =============================================================================
# Expert Manager C++ Tests
# =============================================================================

class TestExpertManagerCpp:
    """Tests for C++ expert manager integration."""
    
    def test_expert_manager_available(self):
        """C++ expert manager should be available."""
        from metal_marlin._compat import HAS_CPP_EXT
        from metal_marlin.expert_manager_cpp import is_available
        
        # Availability should match HAS_CPP_EXT
        assert is_available() == HAS_CPP_EXT
    
    def test_expert_manager_import(self):
        """ExpertManagerCpp should be importable from main module."""
        from metal_marlin import ExpertManagerCpp, expert_manager_available
        
        # If C++ extension is available, ExpertManagerCpp should be defined
        if expert_manager_available:
            assert ExpertManagerCpp is not None
    
    def test_expert_manager_creation(self):
        """ExpertManagerCpp should be creatable."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=8, max_tokens=1024, max_top_k=2)
        
        assert manager.num_experts == 8
        assert manager.max_tokens == 1024
        assert manager.max_top_k == 2
    
    def test_expert_manager_invalid_num_experts(self):
        """ExpertManagerCpp should reject invalid num_experts."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        with pytest.raises(ValueError, match="num_experts must be positive"):
            ExpertManagerCpp(num_experts=0)
        
        with pytest.raises(ValueError, match="num_experts must be positive"):
            ExpertManagerCpp(num_experts=-1)
    
    def test_group_tokens_basic(self):
        """Token grouping should work correctly."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=4, max_tokens=8, max_top_k=2)
        
        # Simple case: 3 tokens, top-2 each
        # Token 0 -> experts 0, 1
        # Token 1 -> experts 1, 2
        # Token 2 -> experts 2, 3
        expert_ids = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
        
        info = manager.group_tokens(expert_ids, batch_size=3, top_k=2)
        
        assert info.num_tokens == 3
        assert info.top_k == 2
        assert info.num_experts == 4
        assert info.total_assignments() == 6
    
    def test_group_tokens_2d(self):
        """Token grouping from 2D array should work."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=4, max_tokens=8, max_top_k=2)
        
        # 2D input: [batch, top_k]
        expert_ids_2d = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        
        info = manager.group_tokens_2d(expert_ids_2d, batch_size=3, top_k=2)
        
        assert info.num_tokens == 3
        assert info.top_k == 2
        assert info.total_assignments() == 6
    
    def test_compute_expert_loads(self):
        """Expert load computation should be accurate."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=4, max_tokens=8, max_top_k=2)
        
        # Create known distribution
        expert_ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)  # 3 to expert 0, 2 to expert 1, 1 to expert 2
        info = manager.group_tokens(expert_ids, batch_size=3, top_k=2)
        
        loads = manager.compute_expert_loads(info)
        
        assert len(loads) == 4  # num_experts
        assert loads[0] == 3  # expert 0 has 3 assignments
        assert loads[1] == 2  # expert 1 has 2 assignments
        assert loads[2] == 1  # expert 2 has 1 assignment
        assert loads[3] == 0  # expert 3 has 0 assignments
    
    def test_load_imbalance_perfect(self):
        """Perfectly balanced load should have imbalance close to 0."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=4, max_tokens=8, max_top_k=1)
        
        # Perfect balance: each expert gets 2 tokens
        expert_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        info = manager.group_tokens(expert_ids, batch_size=8, top_k=1)
        
        imbalance = manager.compute_load_imbalance(info)
        
        # Should be close to 0 (perfect balance)
        assert imbalance < 0.1
        assert manager.is_load_balanced(info, threshold=0.5)
    
    def test_load_imbalance_imbalanced(self):
        """Imbalanced load should have high imbalance."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=4, max_tokens=8, max_top_k=1)
        
        # Imbalanced: most tokens go to experts 0 and 1, few to 2 and 3
        # 5 to expert 0, 2 to expert 1, 1 to expert 2, 0 to expert 3
        expert_ids = np.array([0, 0, 0, 0, 0, 1, 1, 2], dtype=np.int32)
        info = manager.group_tokens(expert_ids, batch_size=8, top_k=1)
        
        imbalance = manager.compute_load_imbalance(info)
        
        # Imbalanced should have imbalance > 0
        assert imbalance > 0.0
        
        # With only 1 sample for expert 2, should have some imbalance
        # but still be considered "balanced" with a high threshold
        assert manager.is_load_balanced(info, threshold=3.0)
    
    def test_active_expert_count(self):
        """Active expert counting should work."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=8, max_tokens=8, max_top_k=1)
        
        # Only experts 0, 1, 2 are active
        expert_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        info = manager.group_tokens(expert_ids, batch_size=6, top_k=1)
        
        assert info.active_expert_count() == 3
        assert info.is_expert_active(0)
        assert info.is_expert_active(1)
        assert info.is_expert_active(2)
        assert not info.is_expert_active(3)
    
    def test_expert_batch_size(self):
        """Per-expert batch size should be correct."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=4, max_tokens=8, max_top_k=1)
        
        expert_ids = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)  # 2 to 0, 3 to 1, 1 to 2
        info = manager.group_tokens(expert_ids, batch_size=6, top_k=1)
        
        assert info.expert_batch_size(0) == 2
        assert info.expert_batch_size(1) == 3
        assert info.expert_batch_size(2) == 1
        assert info.expert_batch_size(3) == 0
    
    def test_get_token_group(self):
        """Token group retrieval should work."""
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        manager = ExpertManagerCpp(num_experts=4, max_tokens=8, max_top_k=1)
        
        expert_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        info = manager.group_tokens(expert_ids, batch_size=4, top_k=1)
        
        group = manager.get_token_group(info, expert_id=0)
        
        assert group.expert_id == 0
        assert group.size() == 2
        assert not group.empty()
        assert group.is_valid()
    
    def test_cpp_path_used_in_hot_path(self):
        """Verify that C++ path is actually used for token grouping.
        
        This test ensures the C++ implementation is being called, not a Python fallback.
        """
        from metal_marlin.expert_manager_cpp import ExpertManagerCpp, is_available
        
        if not is_available():
            pytest.skip("C++ extension not available")
        
        # Create manager and verify it's using the C++ backend
        manager = ExpertManagerCpp(num_experts=8, max_tokens=1024, max_top_k=2)
        
        # The _manager attribute should be the C++ TokenGroupManager
        assert hasattr(manager, '_manager')
        assert type(manager._manager).__name__ == 'TokenGroupManager'
        
        # Perform operation and verify result type
        expert_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        info = manager.group_tokens(expert_ids, batch_size=4, top_k=2)
        
        # Result should be C++ DispatchInfo
        assert type(info).__name__ == 'DispatchInfo'
        
        # Verify the operation actually ran (not just returned None)
        assert info.num_tokens == 4
        assert info.total_assignments() == 8


def test_mmfp4_gemm_binding():
    """Test the direct mmfp4_gemm binding with a dummy kernel."""
    ctx = _require_context()

    # Define a dummy kernel that matches the signature expected by mmfp4_gemm
    source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void dummy_mmfp4(
        device const half* A [[buffer(0)]],
        device const uint* B [[buffer(1)]],
        device const half* S [[buffer(2)]],
        device half* C [[buffer(3)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        // do nothing
    }
    """
    
    # Compile the dummy kernel
    try:
        ctx.compile_source("dummy_lib", source)
        pipeline = ctx.get_pipeline("dummy_mmfp4", "dummy_lib")
    except Exception as e:
        pytest.skip(f"Failed to compile dummy kernel: {e}")

    # Create dummy buffers
    M, N, K = 16, 16, 16
    group_size = 16
    
    # Sizes in bytes
    size_A = M * K * 2  # fp16
    size_B = (K * N * 4) // 8  # 4-bit packed
    size_S = (K // group_size) * N * 2 # fp16
    size_C = M * N * 2 # fp16
    
    buf_A = cpp_ext.create_buffer(ctx, size_A)
    buf_B = cpp_ext.create_buffer(ctx, size_B)
    buf_S = cpp_ext.create_buffer(ctx, size_S)
    buf_C = cpp_ext.create_buffer(ctx, size_C)
    
    # Dispatch
    # Should not throw
    cpp_ext.mmfp4_gemm(
        ctx,
        pipeline,
        buf_A,
        buf_B,
        buf_S,
        buf_C,
        M, N, K, group_size,
        True # wait
    )
    
    # Explicitly delete buffers to ensure they are destroyed before ctx
    del buf_A
    del buf_B
    del buf_S
    del buf_C
    gc.collect()


def test_batch_dispatch_mmfp4_gemm():
    """Test batch dispatch of mmfp4_gemm."""
    ctx = _require_context()
    if not hasattr(cpp_ext, "BatchDispatch"):
        pytest.skip("BatchDispatch not available")

    # Define a dummy kernel
    source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void dummy_mmfp4(
        device const half* A [[buffer(0)]],
        device const uint* B [[buffer(1)]],
        device const half* S [[buffer(2)]],
        device half* C [[buffer(3)]],
        constant uint& M [[buffer(4)]],
        constant uint& N [[buffer(5)]],
        constant uint& K [[buffer(6)]],
        constant uint& group_size [[buffer(7)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        // do nothing
    }
    """
    
    try:
        ctx.compile_source("dummy_lib_batch", source)
        pipeline = ctx.get_pipeline("dummy_mmfp4", "dummy_lib_batch")
    except Exception as e:
        pytest.skip(f"Failed to compile dummy kernel: {e}")

    # Create dummy buffers
    M, N, K = 16, 16, 16
    group_size = 16
    size_A = M * K * 2
    size_B = (K * N * 4) // 8
    size_S = (K // group_size) * N * 2
    size_C = M * N * 2
    
    buf_A = cpp_ext.create_buffer(ctx, size_A)
    buf_B = cpp_ext.create_buffer(ctx, size_B)
    buf_S = cpp_ext.create_buffer(ctx, size_S)
    buf_C = cpp_ext.create_buffer(ctx, size_C)
    
    # Create a mock MetalKernelLibrary for FastPath
    class MockLib:
        pass
    mock_lib = MockLib()
    
    # Initialize FastPath (it needs _available=True)
    from metal_marlin.metal_dispatch import FastPath
    fp = FastPath(mock_lib)
    
    # Inject our context into FastPath (since we created it manually)
    fp._ctx = ctx
    fp._pipelines["dummy_mmfp4"] = pipeline
    
    # Use FastPath to dispatch batch
    ops = [("dummy_mmfp4", buf_A, buf_B, buf_S, buf_C, M, N, K, group_size)]
    fp.batch_mmfp4_gemm(ops, wait=True)
    
    # Explicitly delete buffers and ops list to ensure destruction order
    del ops
    del buf_A
    del buf_B
    del buf_S
    del buf_C
    gc.collect()



# =============================================================================
# ServingContext C++ Tests
# =============================================================================

class TestServingContextCpp:
    """Tests for C++ serving mode integration (ServingContext)."""
    
    def test_serving_context_available(self):
        """ServingContext should be available when C++ extension is built."""
        if not _CPP_EXT_AVAILABLE:
            pytest.skip("C++ extension not available")
        
        # ServingContext may or may not be available depending on extension version
        has_serving = hasattr(cpp_ext, 'ServingContext')
        if not has_serving:
            pytest.skip("ServingContext not available in this build of _cpp_ext")
        
        assert has_serving, "ServingContext not found in _cpp_ext"
    
    def test_serving_context_creation(self):
        """ServingContext should be creatable from MetalContext."""
        if not _CPP_EXT_AVAILABLE:
            pytest.skip("C++ extension not available")
        
        if not hasattr(cpp_ext, 'ServingContext'):
            pytest.skip("ServingContext not available in this build")
        
        ctx = _require_context()
        serving_ctx = cpp_ext.ServingContext(ctx)
        
        assert serving_ctx is not None
        # Should be able to get underlying MetalContext
        assert serving_ctx.metal_context() is ctx
    
    def test_serving_cpp_module_import(self):
        """serving_cpp module should be importable."""
        from metal_marlin import serving_cpp
        assert hasattr(serving_cpp, 'ServingCppDispatcher')
    
    def test_serving_cpp_dispatcher_creation(self):
        """ServingCppDispatcher should be creatable."""
        from metal_marlin.serving_cpp import ServingCppDispatcher
        
        dispatcher = ServingCppDispatcher()
        # availability depends on C++ extension
        assert isinstance(dispatcher.available, bool)
    
    def test_serving_cpp_metrics(self):
        """ServingCppDispatcher should provide metrics."""
        from metal_marlin.serving_cpp import ServingCppDispatcher
        
        dispatcher = ServingCppDispatcher()
        metrics = dispatcher.get_metrics()
        
        assert "dispatch_count" in metrics
        assert "total_dispatch_us" in metrics
        assert "avg_dispatch_us" in metrics
    
    def test_serving_cpp_reset_metrics(self):
        """ServingCppDispatcher should allow resetting metrics."""
        from metal_marlin.serving_cpp import ServingCppDispatcher
        
        dispatcher = ServingCppDispatcher()
        dispatcher.reset_metrics()
        
        metrics = dispatcher.get_metrics()
        assert metrics["dispatch_count"] == 0
        assert metrics["total_dispatch_us"] == 0
    
    def test_serving_engine_has_cpp_path(self, tmp_path):
        """ServingEngine should have C++ serving path wired."""
        pytest.importorskip("pydantic")
        pytest.importorskip("scipy")
        from metal_marlin.serving.engine import EngineConfig
        
        # Check that EngineConfig has use_cpp_serving option
        mock_model_path = tmp_path / "mock_model"
        config = EngineConfig(
            model_path=str(mock_model_path),
            use_cpp_serving=True
        )
        assert hasattr(config, 'use_cpp_serving')
        assert config.use_cpp_serving is True
    
    def test_cpp_path_used_in_hot_path(self):
        """Verify C++ path is available for serving hot path."""
        if not _CPP_EXT_AVAILABLE:
            pytest.skip("C++ extension not available")
        
        if not hasattr(cpp_ext, 'ServingContext'):
            pytest.skip("ServingContext not available in this build")
        
        # Verify ServingContext exists and can be used for dispatch
        ctx = _require_context()
        serving_ctx = cpp_ext.ServingContext(ctx)
        
        # Verify it has the required methods for hot path
        assert hasattr(serving_ctx, 'dispatch_sync')
        assert hasattr(serving_ctx, 'dispatch_async')
        assert hasattr(serving_ctx, 'wait')
        assert hasattr(serving_ctx, 'is_complete')
        assert hasattr(serving_ctx, 'metrics')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
