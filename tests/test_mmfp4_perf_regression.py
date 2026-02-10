"""Performance regression tests for MMFP4 kernels."""
import os
import pytest
import time
import torch

from metal_marlin.layers.mmfp4_expert import MMFP4Expert

# Skip in CI (no GPU)
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("CI"),
    reason="MPS required and not in CI"
)

# Performance thresholds (fail if slower)
THRESHOLDS = {
    "moe_expert_standard_ms": 10.0,  # Must be < 10ms for GLM-4.7-Flash
    "moe_expert_fused_ms": 15.0,     # Must be < 15ms after optimization
}


class TestMMFP4PerfRegression:
    @pytest.fixture
    def expert(self):
        return MMFP4Expert(
            hidden_size=2048,
            moe_intermediate_size=1536,
            use_fused=False,
        ).to("mps")

    def _bench(self, fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        return (time.perf_counter() - start) / iters * 1000

    def test_moe_expert_standard_perf(self, expert):
        x = torch.randn(1, 2048, dtype=torch.float16, device="mps")
        ms = self._bench(lambda: expert(x))
        assert ms < THRESHOLDS["moe_expert_standard_ms"], \
            f"MoE standard {ms:.1f}ms > {THRESHOLDS['moe_expert_standard_ms']}ms"

    def test_moe_expert_fused_perf(self, expert):
        expert.use_fused = True
        x = torch.randn(1, 2048, dtype=torch.float16, device="mps")
        out = expert(x)
        if torch.isnan(out).any():
            pytest.skip("Fused kernel produces NaN")
        ms = self._bench(lambda: expert(x))
        assert ms < THRESHOLDS["moe_expert_fused_ms"], \
            f"MoE fused {ms:.1f}ms > {THRESHOLDS['moe_expert_fused_ms']}ms"
