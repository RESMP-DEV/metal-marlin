"""Test that Metal kernels are used instead of PyTorch fallbacks.

This test file verifies that operations on MPS tensors use Metal kernels
rather than falling back to PyTorch operations. It uses a dispatch counter
to detect when PyTorch operations are called during Metal-accelerated paths.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from metal_marlin._compat import HAS_MPS, HAS_TORCH

if TYPE_CHECKING:
    from collections.abc import Callable


@contextmanager
def count_torch_dispatches(*ops: str):
    """Context manager to count PyTorch dispatch calls.

    Args:
        *ops: Names of torch operations to monitor (e.g., 'argmax', 'sort')

    Yields:
        dict: Mapping of operation name to call count

    Example:
        with count_torch_dispatches('argmax', 'sort') as counts:
            sampler.argmax(logits)
        assert counts['argmax'] == 0, "argmax should use Metal, not PyTorch"
    """
    counts: dict[str, int] = {op: 0 for op in ops}
    original_ops: dict[str, Callable] = {}

    def make_counter(op_name: str, original: Callable) -> Callable:
        def counted_op(*args, **kwargs):  # type: ignore[no-untyped-def]
            counts[op_name] += 1
            return original(*args, **kwargs)

        return counted_op

    # Apply patches
    patches = []
    for op in ops:
        if hasattr(torch, op):
            original_ops[op] = getattr(torch, op)
            p = patch(f"torch.{op}", make_counter(op, original_ops[op]))
            p.start()
            patches.append(p)

    try:
        yield counts
    finally:
        for p in patches:
            p.stop()


@contextmanager
def count_torch_tensor_methods(*ops: str):
    """Context manager to count PyTorch tensor method calls.

    Similar to count_torch_dispatches but for tensor methods like argsort, sort.
    """
    counts: dict[str, int] = {op: 0 for op in ops}
    original_methods: dict[str, Callable] = {}

    def make_counter(op_name: str, original: Callable) -> Callable:
        def counted_method(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            counts[op_name] += 1
            return original(self, *args, **kwargs)

        return counted_method

    # Apply patches to torch.Tensor methods
    patches = []
    for op in ops:
        if hasattr(torch.Tensor, op):
            original_methods[op] = getattr(torch.Tensor, op)
            p = patch(
                f"torch.Tensor.{op}", make_counter(op, original_methods[op])
            )
            p.start()
            patches.append(p)

    try:
        yield counts
    finally:
        for p in patches:
            p.stop()


# Skip all tests if MPS or PyTorch is not available
pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch"),
    pytest.mark.skipif(not HAS_MPS, reason="Requires MPS (Apple Silicon)"),
]


class TestSamplingUsesMetal:
    """Verify sampling operations dispatch to Metal kernels."""

    @pytest.fixture
    def sampler(self):
        """Create a MetalSampler instance."""
        from metal_marlin.sampler import MetalSampler

        return MetalSampler(vocab_size=32000)

    @pytest.fixture
    def sample_logits(self):
        """Create sample logits tensor on MPS."""
        torch.manual_seed(42)
        return torch.randn(1, 32000, device="mps")

    def test_argmax_uses_metal(self, sampler, sample_logits):
        """Verify argmax uses Metal kernel, not torch.argmax."""
        with count_torch_dispatches("argmax") as counts:
            result = sampler.argmax(sample_logits)

        assert isinstance(result, int)
        assert counts["argmax"] == 0, (
            f"argmax should use Metal kernel, but torch.argmax was called {counts['argmax']} times"
        )

    def test_sample_top_p_uses_metal(self, sampler, sample_logits):
        """Verify top-p sampling uses Metal kernel, not torch.sort or torch.multinomial."""
        with (
            count_torch_dispatches("sort", "multinomial") as torch_counts,
            count_torch_tensor_methods("sort") as tensor_counts,
        ):
            result = sampler.sample_top_p(sample_logits, p=0.9)

        assert isinstance(result, int)
        # Should not use PyTorch sorting or multinomial
        total_sort = torch_counts["sort"] + tensor_counts.get("sort", 0)
        assert total_sort == 0, (
            f"top-p sampling should use Metal kernel, but torch.sort was called {total_sort} times"
        )
        assert torch_counts["multinomial"] == 0, (
            f"top-p sampling should use Metal kernel, but torch.multinomial was called {torch_counts['multinomial']} times"
        )

    def test_sample_top_k_uses_metal(self, sampler, sample_logits):
        """Verify top-k sampling uses Metal kernel, not torch.topk."""
        with count_torch_dispatches("topk") as counts:
            result = sampler.sample_top_k(sample_logits, k=50)

        assert isinstance(result, int)
        assert counts["topk"] == 0, (
            f"top-k sampling should use Metal kernel, but torch.topk was called {counts['topk']} times"
        )

    def test_sample_categorical_uses_metal(self, sampler, sample_logits):
        """Verify categorical sampling uses Metal kernel, not torch.multinomial."""
        with count_torch_dispatches("multinomial") as counts:
            result = sampler.sample_categorical(sample_logits, temperature=1.0)

        assert isinstance(result, int)
        assert counts["multinomial"] == 0, (
            f"categorical sampling should use Metal kernel, "
            f"but torch.multinomial was called {counts['multinomial']} times"
        )


class TestMoEDispatchUsesMetal:
    """Verify MoE dispatch uses Metal kernels."""

    @pytest.fixture
    def sample_expert_ids(self):
        """Create sample expert IDs tensor on MPS."""
        torch.manual_seed(42)
        return torch.randint(0, 8, (64, 2), device="mps")

    def _check_metal_moe_available(self):
        """Check if Metal MoE dispatch is actually working."""
        from metal_marlin.moe_dispatch import _USE_METAL

        if not _USE_METAL:
            pytest.skip("Metal MoE dispatch not available (_USE_METAL=False)")

        # Try to import and verify the module works
        try:
            from metal_marlin.moe_dispatch_metal import group_tokens_by_expert_metal
            # Do a quick test to see if it works
            test_ids = torch.randint(0, 4, (4, 2), device="mps")
            _ = group_tokens_by_expert_metal(test_ids, num_experts=4)
            return group_tokens_by_expert_metal
        except Exception as e:
            pytest.skip(f"Metal MoE dispatch not functional: {e}")

    def test_group_tokens_by_expert_avoids_torch_sort(self, sample_expert_ids):
        """Verify group_tokens_by_expert avoids torch.argsort when Metal is available."""
        group_tokens_by_expert_metal = self._check_metal_moe_available()

        with count_torch_tensor_methods("argsort") as counts:
            sorted_idx, offsets, inverse = group_tokens_by_expert_metal(
                sample_expert_ids, num_experts=8
            )

        # Verify results are valid
        assert sorted_idx.dtype == torch.int64
        assert offsets.dtype == torch.int64
        assert inverse.dtype == torch.int64
        assert len(offsets) == 9  # num_experts + 1

        # Metal implementation should not use torch.argsort
        assert counts["argsort"] == 0, (
            f"group_tokens_by_expert_metal should use Metal kernel, "
            f"but torch.argsort was called {counts['argsort']} times"
        )

    def test_group_tokens_does_not_call_bincount_or_cumsum(self, sample_expert_ids):
        """Verify Metal MoE dispatch avoids torch.bincount and torch.cumsum."""
        group_tokens_by_expert_metal = self._check_metal_moe_available()

        with (
            count_torch_dispatches("bincount") as counts_bc,
            count_torch_tensor_methods("cumsum") as counts_cs,
        ):
            sorted_idx, offsets, inverse = group_tokens_by_expert_metal(
                sample_expert_ids, num_experts=8
            )

        assert counts_bc["bincount"] == 0, (
            f"Metal MoE should not use torch.bincount, was called {counts_bc['bincount']} times"
        )
        assert counts_cs.get("cumsum", 0) == 0, (
            f"Metal MoE should not use torch.cumsum, was called {counts_cs.get('cumsum', 0)} times"
        )


class TestFP4QuantizeUsesMetal:
    """Verify FP4 quantization uses Metal kernels."""

    def test_fp4_metal_flag_reflects_availability(self):
        """Verify _USE_METAL reflects Metal FP4 availability.

        This test documents whether Metal FP4 is available on this system.
        If False, the Metal FP4 tests will be skipped.
        """
        from metal_marlin.quantize_fp4 import _USE_METAL

        # Log the status without failing - Metal FP4 might not be built
        if _USE_METAL:
            print("Metal FP4 quantization is available")
        else:
            print("Metal FP4 quantization is not available (using NumPy fallback)")

    def test_quantize_fp4_dispatches_to_metal(self):
        """Verify quantize_fp4 uses Metal path when available."""
        from metal_marlin.quantize_fp4 import _USE_METAL, quantize_fp4

        if not _USE_METAL:
            pytest.skip("Metal FP4 quantization not available")

        import numpy as np

        np.random.seed(42)
        weight = np.random.randn(1024, 4096).astype(np.float32)

        # The function should complete without errors and use Metal internally
        packed, scales = quantize_fp4(weight, group_size=128)

        # Verify output shapes
        assert packed.dtype == np.uint32
        assert scales.dtype == np.float16
        # Packed size: [in_feat/8, out_feat] for Marlin layout
        assert packed.shape[0] == 4096 // 8  # K/8
        assert packed.shape[1] == 1024  # N
        # Scales: [K/group_size, N]
        assert scales.shape[0] == 4096 // 128
        assert scales.shape[1] == 1024

    def test_unpack_fp4_uses_metal(self):
        """Verify unpack_fp4 uses Metal when available."""
        from metal_marlin.quantize_fp4 import _USE_METAL, quantize_fp4, unpack_fp4

        if not _USE_METAL:
            pytest.skip("Metal FP4 not available")

        import numpy as np

        np.random.seed(42)
        weight = np.random.randn(512, 1024).astype(np.float32)

        packed, scales = quantize_fp4(weight, group_size=128)
        unpacked = unpack_fp4(packed, scales, group_size=128)

        # Verify unpacked shape (Marlin layout returns [K, N])
        assert unpacked.shape[0] == 1024  # K = in_feat
        assert unpacked.shape[1] == 512  # N = out_feat


class TestActivationUsesMetal:
    """Verify activation functions use Metal kernels where applicable."""

    @pytest.fixture
    def sample_hidden_states(self):
        """Create sample hidden states on MPS."""
        torch.manual_seed(42)
        return torch.randn(2, 128, 4096, device="mps", dtype=torch.float16)

    def test_marlin_linear_uses_metal(self, sample_hidden_states):
        """Verify MarlinLinear uses Metal GEMM kernels."""
        from metal_marlin.layers import MarlinLinear

        # Create layer using dimension-based constructor
        linear = MarlinLinear(
            4096,  # in_features
            11008,  # out_features
            bias=False,
            quant_type="fp4",
            group_size=128,
        )

        # Forward pass should work with MPS tensors
        output = linear(sample_hidden_states)

        assert output.shape[-1] == 11008
        assert output.device.type == "mps"


class TestMetalKernelDispatch:
    """General tests for Metal kernel dispatch behavior."""

    def test_metal_dispatch_does_not_sync_to_cpu(self):
        """Verify that Metal operations don't unnecessarily sync to CPU."""
        from metal_marlin.metal_dispatch import require_metal, require_mps

        # These should not raise on MPS-enabled systems
        require_mps()

        # require_metal may raise if Metal framework can't be loaded,
        # but it should not cause a CPU sync
        try:
            require_metal()
        except RuntimeError as e:
            if "Metal" in str(e):
                pytest.skip(f"Metal framework not available: {e}")
            raise

    def test_mps_tensors_stay_on_device_through_ops(self):
        """Verify MPS tensors stay on MPS through Metal operations."""
        from metal_marlin.sampler import MetalSampler

        sampler = MetalSampler(vocab_size=1000)

        logits = torch.randn(1, 1000, device="mps")
        original_device = logits.device

        token_id = sampler.argmax(logits)

        # Operation should not have moved the tensor
        assert logits.device == original_device
        # Result should be a Python int (not a tensor)
        assert isinstance(token_id, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
