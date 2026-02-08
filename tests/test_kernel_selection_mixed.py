import unittest
from unittest.mock import patch

from metal_marlin.trellis.kernel_selection_mixed import (
    FAST_2BIT_FAMILY,
    MIXED_BPW_FAMILY,
    STANDARD_FAMILY,
    MixedKernelSelector,
)


class TestMixedKernelSelector(unittest.TestCase):
    def setUp(self):
        self.selector = MixedKernelSelector(history_len=10, initial_exploration_rate=0.0)

    def test_heuristics_dominant_2bit(self):
        bits = [2, 2, 2, 4]  # >50% 2-bit

        kernel, info = self.selector.select_kernel(batch_size=1, active_expert_bits=bits)
        self.assertEqual(kernel, "moe_trellis_swiglu_decode")
        self.assertEqual(info["heuristic_family"], FAST_2BIT_FAMILY)

        kernel, info = self.selector.select_kernel(batch_size=20, active_expert_bits=bits)
        self.assertEqual(kernel, "moe_trellis_swiglu")
        self.assertEqual(info["heuristic_family"], FAST_2BIT_FAMILY)

    def test_heuristics_mixed_2_3_4(self):
        bits = [2, 3, 4, 2]

        kernel, info = self.selector.select_kernel(batch_size=1, active_expert_bits=bits)
        self.assertEqual(kernel, "moe_trellis_mixed_swiglu_decode")
        self.assertEqual(info["heuristic_family"], MIXED_BPW_FAMILY)

        kernel, info = self.selector.select_kernel(batch_size=20, active_expert_bits=bits)
        self.assertEqual(kernel, "moe_trellis_mixed_swiglu")
        self.assertEqual(info["heuristic_family"], MIXED_BPW_FAMILY)

    def test_heuristics_standard_mostly_4bit(self):
        bits = [4, 4, 4, 3]  # mostly 4-bit
        kernel, info = self.selector.select_kernel(batch_size=40, active_expert_bits=bits)
        self.assertEqual(kernel, "moe_trellis_swiglu_large_batch")
        self.assertEqual(info["heuristic_family"], STANDARD_FAMILY)

    def test_decode_fp32acc_uses_decode_kernel(self):
        kernel, info = self.selector.select_kernel(
            batch_size=1,
            active_expert_bits=[4, 4, 4, 4],
            use_fp32_acc=True,
        )
        self.assertEqual(kernel, "moe_trellis_swiglu_decode")
        self.assertEqual(info["baseline_kernel"], "moe_trellis_swiglu_decode")

    def test_memory_pressure_override(self):
        bits = [2, 3, 4, 2]
        kernel, info = self.selector.select_kernel(
            batch_size=1,
            active_expert_bits=bits,
            gpu_memory_pressure=0.95,
        )
        self.assertEqual(kernel, "moe_trellis_swiglu")
        self.assertEqual(info["reason"], "memory_pressure")

    def test_feedback_loop_exploitation(self):
        bits = [4, 4, 4, 4]
        decode = "moe_trellis_swiglu_decode"
        base = "moe_trellis_swiglu"

        for _ in range(5):
            self.selector.record_timing(decode, 1, 10.0, "heuristic")
        for _ in range(5):
            self.selector.record_timing(base, 1, 5.0, "heuristic")

        kernel, info = self.selector.select_kernel(
            batch_size=1,
            active_expert_bits=bits,
            available_kernels={decode, base},
        )
        self.assertEqual(kernel, base)
        self.assertEqual(info["reason"], "feedback_optimization")

    @patch("random.random")
    @patch("random.choice")
    def test_ab_testing_runtime_selects_faster(self, mock_choice, mock_random):
        selector = MixedKernelSelector(history_len=10, initial_exploration_rate=1.0)
        mock_random.return_value = 0.0
        mock_choice.return_value = "moe_trellis_swiglu"

        def benchmark(kernel_name: str) -> float:
            if kernel_name == "moe_trellis_swiglu_decode":
                return 4.0
            if kernel_name == "moe_trellis_swiglu":
                return 2.0
            return 10.0

        kernel, info = selector.select_kernel(
            batch_size=1,
            active_expert_bits=[4, 4, 4, 4],
            available_kernels={"moe_trellis_swiglu_decode", "moe_trellis_swiglu"},
            benchmark_kernel=benchmark,
        )
        self.assertEqual(kernel, "moe_trellis_swiglu")
        self.assertEqual(info["reason"], "ab_testing_runtime")
        self.assertEqual(info["ab_test"]["winner"], "moe_trellis_swiglu")

        stats = selector.get_statistics()
        self.assertEqual(stats["ab_tests_run"], 1)
        self.assertEqual(stats["selection_accuracy"], 0.0)

    @patch("random.random")
    @patch("random.choice")
    def test_ab_testing_accuracy_tracking(self, mock_choice, mock_random):
        selector = MixedKernelSelector(history_len=10, initial_exploration_rate=1.0)
        mock_random.return_value = 0.0
        mock_choice.return_value = "moe_trellis_swiglu"

        def benchmark(kernel_name: str) -> float:
            if kernel_name == "moe_trellis_swiglu_decode":
                return 2.0
            if kernel_name == "moe_trellis_swiglu":
                return 4.0
            return 10.0

        kernel, _ = selector.select_kernel(
            batch_size=1,
            active_expert_bits=[4, 4, 4, 4],
            available_kernels={"moe_trellis_swiglu_decode", "moe_trellis_swiglu"},
            benchmark_kernel=benchmark,
        )
        self.assertEqual(kernel, "moe_trellis_swiglu_decode")

        stats = selector.get_statistics()
        self.assertEqual(stats["ab_tests_run"], 1)
        self.assertEqual(stats["selection_accuracy"], 1.0)

    def test_activation_pattern_adjustment(self):
        bits = [2, 3, 4, 2]
        kernel, info = self.selector.select_kernel(
            batch_size=1,
            active_expert_bits=bits,
            expert_activation_pattern={"top_expert_share": 0.9},
        )
        self.assertEqual(kernel, "moe_trellis_swiglu_decode")
        self.assertEqual(info["reason"], "activation_pattern_skewed")

    def test_availability_fallback(self):
        bits = [2, 3, 4, 2]
        kernel, info = self.selector.select_kernel(
            batch_size=1,
            active_expert_bits=bits,
            available_kernels={"moe_trellis_swiglu"},
        )
        self.assertEqual(kernel, "moe_trellis_swiglu")
        self.assertEqual(info["reason"], "availability_fallback")

    def test_statistics_and_reset(self):
        for _ in range(3):
            self.selector.select_kernel(batch_size=1, active_expert_bits=[4, 4, 4, 4])

        stats = self.selector.get_statistics()
        self.assertEqual(stats["total_selections"], 3)
        self.assertIn("selection_accuracy", stats)
        self.assertIn("tracked_kernels", stats)

        self.selector.reset_history()
        stats = self.selector.get_statistics()
        self.assertEqual(stats["total_selections"], 0)
        self.assertEqual(stats["selection_accuracy"], 0.0)
        self.assertEqual(len(self.selector.timings), 0)

    def test_exploration_rate_update(self):
        selector = MixedKernelSelector(history_len=10, initial_exploration_rate=0.3)
        selector.set_exploration_rate(0.2)
        self.assertEqual(selector.exploration_rate, 0.2)

        with self.assertRaises(ValueError):
            selector.set_exploration_rate(1.5)


if __name__ == "__main__":
    unittest.main()
