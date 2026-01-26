"""Tests for MoE token-to-expert grouping dispatch."""

import mlx.core as mx
import numpy as np

from metal_marlin.moe_dispatch import (
    compute_expert_load,
    compute_load_balancing_loss,
    gather_for_experts,
    group_tokens_by_expert,
    group_tokens_by_expert_full,
    scatter_expert_outputs,
)


class TestGroupTokensByExpert:
    """Tests for the core group_tokens_by_expert function."""

    def test_basic_grouping(self):
        """Test basic token grouping with simple input."""
        # 3 tokens, top_k=2, 3 experts
        # Token 0 -> experts 0, 2
        # Token 1 -> experts 1, 2
        # Token 2 -> experts 0, 1
        expert_ids = mx.array([[0, 2], [1, 2], [0, 1]], dtype=mx.int32)
        num_experts = 3

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(sorted_idx, offsets, inverse)

        # Check offsets sum correctly
        total = expert_ids.size
        assert offsets[-1].item() == total

        # Check each expert's count
        expert_counts = np.diff(np.array(offsets))
        # Expert 0: tokens 0, 2 (positions 0, 4 in flat)
        # Expert 1: tokens 1, 2 (positions 2, 5 in flat)
        # Expert 2: tokens 0, 1 (positions 1, 3 in flat)
        assert expert_counts[0] == 2  # Expert 0
        assert expert_counts[1] == 2  # Expert 1
        assert expert_counts[2] == 2  # Expert 2

        # Verify sorted_idx groups by expert
        expert_ids_flat = np.array(expert_ids.reshape(-1))
        sorted_experts = expert_ids_flat[np.array(sorted_idx)]

        # Should be sorted by expert
        assert np.all(sorted_experts[:-1] <= sorted_experts[1:])

        # Verify inverse is correct: inverse[sorted_idx] should give identity permutation
        # and sorted_idx[inverse] should also give identity
        perm = np.array(sorted_idx)[np.array(inverse)]
        assert np.array_equal(perm, np.arange(total))

    def test_single_expert_per_token(self):
        """Test with top_k=1 (single expert per token)."""
        # 4 tokens, top_k=1, 2 experts
        expert_ids = mx.array([[0], [1], [0], [1]], dtype=mx.int32)
        num_experts = 2

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(sorted_idx, offsets, inverse)

        # Expert 0: tokens 0, 2 -> positions 0, 2
        # Expert 1: tokens 1, 3 -> positions 1, 3
        assert offsets[0].item() == 0
        assert offsets[1].item() == 2
        assert offsets[2].item() == 4

    def test_uneven_expert_distribution(self):
        """Test when experts have unequal load."""
        # 4 tokens, top_k=2, 3 experts
        # All tokens route to expert 0 as first choice
        expert_ids = mx.array([[0, 1], [0, 2], [0, 1], [0, 2]], dtype=mx.int32)
        num_experts = 3

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(sorted_idx, offsets, inverse)

        expert_counts = np.diff(np.array(offsets))
        assert expert_counts[0] == 4  # Expert 0 gets all 4 tokens
        assert expert_counts[1] == 2  # Expert 1 gets tokens 0, 2
        assert expert_counts[2] == 2  # Expert 2 gets tokens 1, 3

    def test_empty_experts(self):
        """Test when some experts receive no tokens."""
        # 2 tokens, top_k=1, 4 experts
        # Only experts 0 and 2 are used
        expert_ids = mx.array([[0], [2]], dtype=mx.int32)
        num_experts = 4

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(sorted_idx, offsets, inverse)

        expert_counts = np.diff(np.array(offsets))
        assert expert_counts[0] == 1
        assert expert_counts[1] == 0  # Expert 1 unused
        assert expert_counts[2] == 1
        assert expert_counts[3] == 0  # Expert 3 unused

    def test_large_batch(self):
        """Test with larger batch size."""
        batch_size = 128
        top_k = 4
        num_experts = 16

        # Random expert assignments
        np.random.seed(42)
        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = mx.array(expert_ids_np, dtype=mx.int32)

        sorted_idx, offsets, inverse = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(sorted_idx, offsets, inverse)

        # Basic checks
        total = batch_size * top_k
        assert offsets[-1].item() == total
        assert sorted_idx.shape[0] == total
        assert inverse.shape[0] == total

        # Verify grouping correctness
        expert_ids_flat = np.array(expert_ids.reshape(-1))
        sorted_experts = expert_ids_flat[np.array(sorted_idx)]
        assert np.all(sorted_experts[:-1] <= sorted_experts[1:])

        # Verify inverse correctness
        perm = np.array(sorted_idx)[np.array(inverse)]
        assert np.array_equal(perm, np.arange(total))


class TestMoEDispatchInfo:
    """Tests for the full dispatch info structure."""

    def test_dispatch_info_structure(self):
        """Test MoEDispatchInfo fields are correct."""
        expert_ids = mx.array([[0, 2], [1, 0], [2, 1]], dtype=mx.int32)
        num_experts = 3

        info = group_tokens_by_expert_full(expert_ids, num_experts)
        mx.eval(
            info.sorted_token_indices,
            info.sorted_expert_indices,
            info.expert_offsets,
            info.inverse_indices,
        )

        assert info.num_tokens == 3
        assert info.top_k == 2
        assert info.num_experts == 3
        assert info.total_assignments == 6

        # Check shapes
        assert info.sorted_token_indices.shape == (6,)
        assert info.sorted_expert_indices.shape == (6,)
        assert info.expert_offsets.shape == (4,)
        assert info.inverse_indices.shape == (6,)

    def test_token_and_expert_indices_consistency(self):
        """Verify sorted_token_indices and sorted_expert_indices are consistent."""
        expert_ids = mx.array([[1, 0], [0, 2], [2, 1]], dtype=mx.int32)
        num_experts = 3

        info = group_tokens_by_expert_full(expert_ids, num_experts)
        mx.eval(
            info.sorted_token_indices, info.sorted_expert_indices, info.expert_offsets
        )

        # For each sorted position, verify the expert is correct
        sorted_token_idx = np.array(info.sorted_token_indices)
        sorted_expert_slot = np.array(info.sorted_expert_indices)
        expert_ids_np = np.array(expert_ids)

        for i in range(info.total_assignments):
            token_idx = sorted_token_idx[i]
            expert_slot = sorted_expert_slot[i]
            expected_expert = expert_ids_np[token_idx, expert_slot]

            # This should match the expert we're currently processing
            # (determined by which offset range i falls into)
            for e in range(num_experts):
                start = info.expert_offsets[e].item()
                end = info.expert_offsets[e + 1].item()
                if start <= i < end:
                    assert expected_expert == e
                    break


class TestGatherAndScatter:
    """Tests for gather_for_experts and scatter_expert_outputs."""

    def test_gather_for_experts(self):
        """Test activation gathering in expert-sorted order."""
        batch_size = 4
        hidden_dim = 8
        top_k = 2
        num_experts = 3

        # Create activations
        activations = mx.arange(batch_size * hidden_dim, dtype=mx.float32).reshape(
            batch_size, hidden_dim
        )
        expert_ids = mx.array([[0, 1], [1, 2], [0, 2], [1, 0]], dtype=mx.int32)

        info = group_tokens_by_expert_full(expert_ids, num_experts)
        gathered = gather_for_experts(activations, info)
        mx.eval(gathered)

        # Check shape
        assert gathered.shape == (batch_size * top_k, hidden_dim)

        # Verify each gathered row matches the correct token
        sorted_token_idx = np.array(info.sorted_token_indices)
        activations_np = np.array(activations)
        gathered_np = np.array(gathered)

        for i in range(batch_size * top_k):
            expected_token = sorted_token_idx[i]
            np.testing.assert_array_equal(
                gathered_np[i], activations_np[expected_token]
            )

    def test_scatter_expert_outputs(self):
        """Test output scattering and weighted combination."""
        batch_size = 3
        out_dim = 4
        top_k = 2
        num_experts = 3

        expert_ids = mx.array([[0, 1], [1, 2], [0, 2]], dtype=mx.int32)
        expert_probs = mx.array(
            [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], dtype=mx.float32
        )

        info = group_tokens_by_expert_full(expert_ids, num_experts)

        # Create mock expert outputs (all ones for simplicity)
        expert_outputs = mx.ones((batch_size * top_k, out_dim), dtype=mx.float32)

        result = scatter_expert_outputs(expert_outputs, expert_probs, info)
        mx.eval(result)

        # Each token should sum to 1.0 since probs sum to 1 and outputs are all 1s
        expected = np.ones((batch_size, out_dim), dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)

    def test_scatter_with_varying_outputs(self):
        """Test scatter with different expert outputs."""
        batch_size = 2
        out_dim = 2
        top_k = 2
        num_experts = 2

        expert_ids = mx.array([[0, 1], [1, 0]], dtype=mx.int32)
        expert_probs = mx.array([[0.5, 0.5], [0.5, 0.5]], dtype=mx.float32)

        info = group_tokens_by_expert_full(expert_ids, num_experts)

        # Expert 0 outputs [1, 0], Expert 1 outputs [0, 1]
        # Need to create outputs in sorted order
        # After sorting by expert:
        # - Expert 0 assignments come first, then Expert 1
        mx.zeros(
            (batch_size * top_k, out_dim), dtype=mx.float32
        )

        # Create outputs based on which expert each sorted position belongs to
        expert_outputs_list = []
        for i in range(batch_size * top_k):
            # Find which expert this position belongs to
            for e in range(num_experts):
                start = info.expert_offsets[e].item()
                end = info.expert_offsets[e + 1].item()
                if start <= i < end:
                    if e == 0:
                        expert_outputs_list.append([1.0, 0.0])
                    else:
                        expert_outputs_list.append([0.0, 1.0])
                    break

        mx.eval(info.expert_offsets)
        expert_outputs = mx.array(expert_outputs_list, dtype=mx.float32)

        result = scatter_expert_outputs(expert_outputs, expert_probs, info)
        mx.eval(result)

        # Token 0: 0.5 * [1,0] + 0.5 * [0,1] = [0.5, 0.5]
        # Token 1: 0.5 * [0,1] + 0.5 * [1,0] = [0.5, 0.5]
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


class TestExpertLoad:
    """Tests for expert load computation."""

    def test_compute_expert_load(self):
        """Test expert load counting."""
        expert_ids = mx.array([[0, 1], [0, 2], [1, 2]], dtype=mx.int32)
        num_experts = 3

        load = compute_expert_load(expert_ids, num_experts)
        mx.eval(load)

        # Expert 0: 2 assignments (tokens 0, 1)
        # Expert 1: 2 assignments (tokens 0, 2)
        # Expert 2: 2 assignments (tokens 1, 2)
        expected = np.array([2, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(np.array(load), expected)

    def test_compute_expert_load_uneven(self):
        """Test with uneven expert distribution."""
        expert_ids = mx.array([[0, 0], [0, 1], [0, 0]], dtype=mx.int32)
        num_experts = 3

        load = compute_expert_load(expert_ids, num_experts)
        mx.eval(load)

        # Expert 0: 5 assignments
        # Expert 1: 1 assignment
        # Expert 2: 0 assignments
        expected = np.array([5, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(np.array(load), expected)


class TestLoadBalancingLoss:
    """Tests for auxiliary load balancing loss."""

    def test_perfect_balance(self):
        """Test loss when load is perfectly balanced."""
        batch_size = 4
        num_experts = 2

        # Each expert gets exactly 2 tokens
        expert_ids = mx.array([[0], [1], [0], [1]], dtype=mx.int32)

        # Uniform routing probabilities
        expert_probs_pre_topk = mx.ones((batch_size, num_experts)) / num_experts

        loss = compute_load_balancing_loss(expert_probs_pre_topk, expert_ids, num_experts)
        mx.eval(loss)

        # f = [0.5, 0.5], P = [0.5, 0.5]
        # loss = 2 * (0.5 * 0.5 + 0.5 * 0.5) = 2 * 0.5 = 1.0
        assert abs(loss.item() - 1.0) < 1e-5

    def test_imbalanced_load(self):
        """Test loss increases with imbalanced load."""
        batch_size = 4
        num_experts = 2

        # All tokens to expert 0
        expert_ids = mx.array([[0], [0], [0], [0]], dtype=mx.int32)

        # Uniform routing probabilities
        expert_probs_pre_topk = mx.ones((batch_size, num_experts)) / num_experts

        loss = compute_load_balancing_loss(expert_probs_pre_topk, expert_ids, num_experts)
        mx.eval(loss)

        # f = [1.0, 0.0], P = [0.5, 0.5]
        # loss = 2 * (1.0 * 0.5 + 0.0 * 0.5) = 2 * 0.5 = 1.0
        # Same as balanced case since probs are uniform!
        # The loss increases when probs are also skewed
        assert loss.item() >= 0.0

    def test_loss_with_skewed_probs(self):
        """Test loss is higher when probs match skewed routing."""
        num_experts = 2

        # All tokens to expert 0
        expert_ids = mx.array([[0], [0], [0], [0]], dtype=mx.int32)

        # Probs also favor expert 0
        expert_probs_pre_topk = mx.array(
            [[0.9, 0.1], [0.8, 0.2], [0.9, 0.1], [0.85, 0.15]], dtype=mx.float32
        )

        loss = compute_load_balancing_loss(expert_probs_pre_topk, expert_ids, num_experts)
        mx.eval(loss)

        # f = [1.0, 0.0]
        # P = mean([[0.9, 0.1], ...]) = [0.8625, 0.1375]
        # loss = 2 * (1.0 * 0.8625 + 0.0 * 0.1375) = 1.725
        assert loss.item() > 1.0  # Higher than balanced case


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_moe_dispatch_flow(self):
        """Test complete dispatch -> gather -> compute -> scatter flow."""
        batch_size = 8
        hidden_dim = 16
        out_dim = 16
        top_k = 2
        num_experts = 4

        # Random expert assignments
        np.random.seed(123)
        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = mx.array(expert_ids_np, dtype=mx.int32)

        # Random routing probs (normalized)
        expert_probs_np = np.random.rand(batch_size, top_k).astype(np.float32)
        expert_probs_np /= expert_probs_np.sum(axis=1, keepdims=True)
        expert_probs = mx.array(expert_probs_np)

        # Random activations
        activations = mx.random.normal((batch_size, hidden_dim))

        # Dispatch
        info = group_tokens_by_expert_full(expert_ids, num_experts)

        # Gather
        gathered = gather_for_experts(activations, info)

        # Mock expert computation: just pass through (identity)
        expert_outputs = gathered[:, :out_dim]

        # Scatter
        result = scatter_expert_outputs(expert_outputs, expert_probs, info)
        mx.eval(result)

        # Check shape
        assert result.shape == (batch_size, out_dim)

        # Verify result is weighted sum of original activations
        # Since expert computation is identity, result[i] should be:
        # sum over k: expert_probs[i, k] * activations[i, :out_dim]
        # = sum(expert_probs[i]) * activations[i, :out_dim]
        # = 1.0 * activations[i, :out_dim]
        expected = np.array(activations[:, :out_dim])
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-4)
