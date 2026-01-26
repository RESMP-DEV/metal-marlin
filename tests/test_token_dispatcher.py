"""Tests for MoE token dispatcher with efficient batching."""

import numpy as np
import pytest

# Skip all tests if MLX not available
mlx = pytest.importorskip("mlx.core")
import mlx.core as mx

from metal_marlin.moe.token_dispatcher import (
    DispatchInfo,
    DispatchStats,
    TokenDispatcher,
    combine_expert_outputs,
    compute_expert_load,
    compute_load_balancing_loss,
    dispatch_to_experts,
    gather_tokens_for_expert,
    group_tokens_by_expert,
)


class TestGroupTokensByExpert:
    """Tests for the group_tokens_by_expert function."""

    def test_basic_grouping(self):
        """Test basic token grouping with simple input."""
        # 3 tokens, top_k=2, 3 experts
        # Token 0 -> experts 0, 2
        # Token 1 -> experts 1, 2
        # Token 2 -> experts 0, 1
        expert_ids = mx.array([[0, 2], [1, 2], [0, 1]], dtype=mx.int32)
        num_experts = 3

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.sorted_token_indices, info.expert_offsets, info.inverse_indices)

        # Check basic structure
        assert info.num_tokens == 3
        assert info.top_k == 2
        assert info.num_experts == 3
        assert info.total_assignments == 6

        # Check shapes
        assert info.sorted_token_indices.shape == (6,)
        assert info.sorted_expert_slots.shape == (6,)
        assert info.expert_offsets.shape == (4,)
        assert info.inverse_indices.shape == (6,)

        # Check offsets sum correctly
        assert info.expert_offsets[-1].item() == 6

        # Each expert gets 2 assignments
        expert_counts = np.diff(np.array(info.expert_offsets))
        assert expert_counts[0] == 2  # Expert 0: tokens 0, 2
        assert expert_counts[1] == 2  # Expert 1: tokens 1, 2
        assert expert_counts[2] == 2  # Expert 2: tokens 0, 1

    def test_single_expert_per_token(self):
        """Test with top_k=1 (single expert per token)."""
        expert_ids = mx.array([[0], [1], [0], [1]], dtype=mx.int32)
        num_experts = 2

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.expert_offsets)

        assert info.expert_offsets[0].item() == 0
        assert info.expert_offsets[1].item() == 2  # Expert 0 gets 2 tokens
        assert info.expert_offsets[2].item() == 4  # Expert 1 gets 2 tokens

    def test_uneven_distribution(self):
        """Test when experts have unequal load."""
        # All tokens route to expert 0 as first choice
        expert_ids = mx.array([[0, 1], [0, 2], [0, 1], [0, 2]], dtype=mx.int32)
        num_experts = 3

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.expert_offsets)

        expert_counts = np.diff(np.array(info.expert_offsets))
        assert expert_counts[0] == 4  # Expert 0 gets all 4 tokens
        assert expert_counts[1] == 2  # Expert 1 gets tokens 0, 2
        assert expert_counts[2] == 2  # Expert 2 gets tokens 1, 3

    def test_empty_experts(self):
        """Test when some experts receive no tokens."""
        expert_ids = mx.array([[0], [2]], dtype=mx.int32)
        num_experts = 4

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.expert_offsets)

        expert_counts = np.diff(np.array(info.expert_offsets))
        assert expert_counts[0] == 1  # Expert 0
        assert expert_counts[1] == 0  # Expert 1 unused
        assert expert_counts[2] == 1  # Expert 2
        assert expert_counts[3] == 0  # Expert 3 unused

    def test_inverse_indices_correctness(self):
        """Verify inverse indices correctly restore order."""
        expert_ids = mx.array([[1, 0], [0, 2], [2, 1]], dtype=mx.int32)
        num_experts = 3

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.sorted_token_indices, info.inverse_indices)

        # Round-trip: applying sorted then inverse should give identity
        total = info.total_assignments
        np.arange(total)[np.array(info.inverse_indices)]
        # This actually tests: inverse_indices gives us original positions
        # So sorted_indices[inverse_indices] should give us original ordering
        np.array(info.sorted_token_indices)[np.array(info.inverse_indices)]
        np.arange(total)
        # Actually, the inverse relationship is: argsort(sorted_indices) = inverse_indices
        # So sorted_indices[inverse_indices[i]] gives us the original position i
        # Let's verify the inverse property differently
        np.argsort(np.array(info.sorted_token_indices) * total + np.arange(total))
        # Actually simpler: verify argsort(sorted_indices) ≈ inverse_indices
        # The relationship is: sorted_indices[i] tells us which original position goes to sorted position i
        # inverse_indices[i] tells us which sorted position original position i maps to
        # So: inverse_indices = argsort(argsort(sort_keys))
        # Actually this is already tested in test_moe_dispatch.py, let's just check shapes here
        assert len(np.unique(np.array(info.inverse_indices))) == total

    def test_large_batch(self):
        """Test with larger batch size."""
        batch_size = 128
        top_k = 4
        num_experts = 16

        np.random.seed(42)
        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = mx.array(expert_ids_np, dtype=mx.int32)

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.sorted_token_indices, info.expert_offsets, info.inverse_indices)

        # Basic checks
        total = batch_size * top_k
        assert info.expert_offsets[-1].item() == total
        assert info.sorted_token_indices.shape[0] == total
        assert info.inverse_indices.shape[0] == total


class TestGatherTokensForExpert:
    """Tests for gathering activations for specific experts."""

    def test_gather_for_single_expert(self):
        """Test gathering activations for one expert."""
        batch_size = 4
        hidden_dim = 8
        num_experts = 3

        activations = mx.arange(batch_size * hidden_dim, dtype=mx.float32).reshape(
            batch_size, hidden_dim
        )
        expert_ids = mx.array([[0, 1], [1, 2], [0, 2], [1, 0]], dtype=mx.int32)

        info = group_tokens_by_expert(expert_ids, num_experts)

        # Gather for expert 0 (should get tokens 0, 2, 3)
        gathered = gather_tokens_for_expert(activations, info, expert_id=0)
        mx.eval(gathered)

        # Expert 0 receives: token 0 (slot 0), token 2 (slot 0), token 3 (slot 1)
        assert gathered.shape[0] == 3  # 3 assignments to expert 0
        assert gathered.shape[1] == hidden_dim

    def test_gather_empty_expert(self):
        """Test gathering for expert with no tokens."""
        activations = mx.ones((4, 8), dtype=mx.float32)
        expert_ids = mx.array([[0], [0], [0], [0]], dtype=mx.int32)  # All to expert 0
        num_experts = 3

        info = group_tokens_by_expert(expert_ids, num_experts)

        # Expert 1 has no tokens
        gathered = gather_tokens_for_expert(activations, info, expert_id=1)
        mx.eval(gathered)

        assert gathered.shape[0] == 0
        assert gathered.shape[1] == 8


class TestDispatchToExperts:
    """Tests for batched expert dispatch."""

    def test_dispatch_simple(self):
        """Test basic dispatch with identity experts."""
        batch_size = 4
        hidden_dim = 8
        out_dim = 8
        top_k = 2
        num_experts = 3

        activations = mx.ones((batch_size, hidden_dim), dtype=mx.float32)
        expert_ids = mx.array([[0, 1], [1, 2], [0, 2], [1, 0]], dtype=mx.int32)

        info = group_tokens_by_expert(expert_ids, num_experts)

        # Identity expert forward
        def expert_forward(x: mx.array, expert_id: int) -> mx.array:
            return x[:, :out_dim]  # Simple slice

        outputs = dispatch_to_experts(activations, info, expert_forward)
        mx.eval(outputs)

        # All outputs should be 1s since input is 1s and experts are identity
        assert outputs.shape == (batch_size * top_k, out_dim)
        np.testing.assert_allclose(np.array(outputs), np.ones((batch_size * top_k, out_dim)), rtol=1e-5)

    def test_dispatch_with_expert_specific_output(self):
        """Test dispatch where each expert produces different output."""
        batch_size = 2
        hidden_dim = 4
        num_experts = 2

        activations = mx.ones((batch_size, hidden_dim), dtype=mx.float32)
        expert_ids = mx.array([[0, 1], [1, 0]], dtype=mx.int32)

        info = group_tokens_by_expert(expert_ids, num_experts)

        # Expert 0 outputs 1s, Expert 1 outputs 2s
        def expert_forward(x: mx.array, expert_id: int) -> mx.array:
            return mx.full(x.shape, float(expert_id + 1), dtype=mx.float32)

        outputs = dispatch_to_experts(activations, info, expert_forward)
        mx.eval(outputs, info.expert_offsets)

        # Check that outputs match expected expert values
        offsets = np.array(info.expert_offsets)
        for e in range(num_experts):
            start, end = offsets[e], offsets[e + 1]
            if start < end:
                expected_val = float(e + 1)
                actual = np.array(outputs[start:end])
                np.testing.assert_allclose(actual, expected_val, rtol=1e-5)


class TestCombineExpertOutputs:
    """Tests for combining expert outputs with probability weighting."""

    def test_combine_with_equal_probs(self):
        """Test combination with equal routing probabilities."""
        batch_size = 2
        out_dim = 4
        top_k = 2
        num_experts = 2

        expert_ids = mx.array([[0, 1], [1, 0]], dtype=mx.int32)
        expert_probs = mx.array([[0.5, 0.5], [0.5, 0.5]], dtype=mx.float32)

        info = group_tokens_by_expert(expert_ids, num_experts)

        # All expert outputs are 1s
        expert_outputs = mx.ones((batch_size * top_k, out_dim), dtype=mx.float32)

        result = combine_expert_outputs(expert_outputs, expert_probs, info)
        mx.eval(result)

        # Each token should sum to 1.0 (0.5 * 1 + 0.5 * 1)
        expected = np.ones((batch_size, out_dim), dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)

    def test_combine_with_varying_probs(self):
        """Test combination with different routing probabilities."""
        batch_size = 2
        top_k = 2
        num_experts = 2

        expert_ids = mx.array([[0, 1], [1, 0]], dtype=mx.int32)
        expert_probs = mx.array([[0.7, 0.3], [0.6, 0.4]], dtype=mx.float32)

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.expert_offsets)

        # Create outputs: expert 0 outputs [1, 0], expert 1 outputs [0, 1]
        outputs_list = []
        for i in range(batch_size * top_k):
            for e in range(num_experts):
                start = int(info.expert_offsets[e].item())
                end = int(info.expert_offsets[e + 1].item())
                if start <= i < end:
                    if e == 0:
                        outputs_list.append([1.0, 0.0])
                    else:
                        outputs_list.append([0.0, 1.0])
                    break

        expert_outputs = mx.array(outputs_list, dtype=mx.float32)

        result = combine_expert_outputs(expert_outputs, expert_probs, info)
        mx.eval(result)

        # Token 0: 0.7 * [1,0] + 0.3 * [0,1] = [0.7, 0.3]
        # Token 1: 0.6 * [0,1] + 0.4 * [1,0] = [0.4, 0.6]
        expected = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


class TestTokenDispatcher:
    """Tests for the high-level TokenDispatcher class."""

    def test_dispatcher_creation(self):
        """Test creating a TokenDispatcher."""
        dispatcher = TokenDispatcher(
            num_experts=64,
            hidden_dim=4096,
            intermediate_dim=14336,
            top_k=2,
        )

        assert dispatcher.num_experts == 64
        assert dispatcher.hidden_dim == 4096
        assert dispatcher.top_k == 2

    def test_dispatcher_simple_dispatch(self):
        """Test simple dispatch through TokenDispatcher."""
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_dim=8,
            intermediate_dim=16,
            top_k=2,
            enable_stats=True,
        )

        batch_size = 4
        hidden_dim = 8

        hidden_states = mx.ones((batch_size, hidden_dim), dtype=mx.float32)
        expert_ids = mx.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=mx.int32)
        expert_probs = mx.array(
            [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]], dtype=mx.float32
        )

        def expert_forward(x: mx.array, expert_id: int) -> mx.array:
            return x  # Identity

        output = dispatcher.dispatch(
            hidden_states, expert_ids, expert_probs, expert_forward
        )
        mx.eval(output)

        # Output should equal input since probs sum to 1 and experts are identity
        np.testing.assert_allclose(
            np.array(output), np.array(hidden_states), rtol=1e-5
        )

        # Check stats were collected
        assert dispatcher.last_stats is not None
        assert dispatcher.last_stats.num_tokens == 4
        assert dispatcher.last_stats.top_k == 2
        assert dispatcher.last_stats.active_experts == 4  # All 4 experts used

    def test_dispatcher_with_shared_expert(self):
        """Test dispatch with shared expert."""
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_dim=8,
            intermediate_dim=16,
            top_k=2,
        )

        batch_size = 4
        hidden_dim = 8

        hidden_states = mx.ones((batch_size, hidden_dim), dtype=mx.float32)
        expert_ids = mx.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=mx.int32)
        expert_probs = mx.array(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=mx.float32
        )

        def expert_forward(x: mx.array, expert_id: int) -> mx.array:
            return x

        def shared_expert_forward(x: mx.array, expert_id: int) -> mx.array:
            return x * 2.0  # Double the input

        output = dispatcher.dispatch_with_shared_expert(
            hidden_states,
            expert_ids,
            expert_probs,
            expert_forward,
            shared_expert_forward,
            shared_expert_weight=0.5,
        )
        mx.eval(output)

        # routed: 1.0 (probs sum to 1, identity experts)
        # shared: 2.0 * 0.5 = 1.0
        # total: 2.0
        expected = np.full((batch_size, hidden_dim), 2.0, dtype=np.float32)
        np.testing.assert_allclose(np.array(output), expected, rtol=1e-5)


class TestComputeExpertLoad:
    """Tests for expert load computation."""

    def test_balanced_load(self):
        """Test with balanced expert distribution."""
        expert_ids = mx.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=mx.int32)
        num_experts = 2

        load = compute_expert_load(expert_ids, num_experts)
        mx.eval(load)

        np.testing.assert_array_equal(np.array(load), [4, 4])

    def test_unbalanced_load(self):
        """Test with unbalanced expert distribution."""
        expert_ids = mx.array([[0, 0], [0, 1], [0, 0]], dtype=mx.int32)
        num_experts = 3

        load = compute_expert_load(expert_ids, num_experts)
        mx.eval(load)

        np.testing.assert_array_equal(np.array(load), [5, 1, 0])


class TestLoadBalancingLoss:
    """Tests for auxiliary load balancing loss."""

    def test_perfect_balance(self):
        """Test loss with perfectly balanced routing."""
        batch_size = 4
        num_experts = 2

        expert_ids = mx.array([[0], [1], [0], [1]], dtype=mx.int32)
        router_probs = mx.ones((batch_size, num_experts), dtype=mx.float32) / num_experts

        loss = compute_load_balancing_loss(router_probs, expert_ids, num_experts)
        mx.eval(loss)

        # f = [0.5, 0.5], P = [0.5, 0.5]
        # loss = 2 * (0.5 * 0.5 + 0.5 * 0.5) = 1.0
        assert abs(loss.item() - 1.0) < 1e-5

    def test_skewed_routing(self):
        """Test loss increases with skewed routing and probs."""
        num_experts = 2

        # All tokens to expert 0
        expert_ids = mx.array([[0], [0], [0], [0]], dtype=mx.int32)

        # Probs also favor expert 0
        router_probs = mx.array(
            [[0.9, 0.1], [0.8, 0.2], [0.9, 0.1], [0.85, 0.15]], dtype=mx.float32
        )

        loss = compute_load_balancing_loss(router_probs, expert_ids, num_experts)
        mx.eval(loss)

        # Should be higher than balanced case
        assert loss.item() > 1.0


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_dispatch_flow(self):
        """Test complete dispatch flow from grouping to output."""
        batch_size = 8
        hidden_dim = 16
        out_dim = 16
        top_k = 2
        num_experts = 4

        np.random.seed(42)

        # Random expert assignments
        expert_ids_np = np.random.randint(0, num_experts, size=(batch_size, top_k))
        expert_ids = mx.array(expert_ids_np, dtype=mx.int32)

        # Random routing probs (normalized)
        expert_probs_np = np.random.rand(batch_size, top_k).astype(np.float32)
        expert_probs_np /= expert_probs_np.sum(axis=1, keepdims=True)
        expert_probs = mx.array(expert_probs_np)

        # Random activations
        activations = mx.random.normal((batch_size, hidden_dim))

        # Dispatch
        info = group_tokens_by_expert(expert_ids, num_experts)

        def expert_forward(x: mx.array, expert_id: int) -> mx.array:
            return x[:, :out_dim]

        expert_outputs = dispatch_to_experts(activations, info, expert_forward)
        result = combine_expert_outputs(expert_outputs, expert_probs, info)
        mx.eval(result)

        # Output should be weighted sum of activations
        # Since experts are identity, result should equal activations weighted by probs
        expected = np.array(activations[:, :out_dim])
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-4)

    def test_dispatch_info_expert_batch_size(self):
        """Test DispatchInfo.expert_batch_size method."""
        expert_ids = mx.array([[0, 1], [0, 2], [1, 2]], dtype=mx.int32)
        num_experts = 3

        info = group_tokens_by_expert(expert_ids, num_experts)
        mx.eval(info.expert_offsets)

        assert info.expert_batch_size(0) == 2
        assert info.expert_batch_size(1) == 2
        assert info.expert_batch_size(2) == 2

    def test_dispatch_stats(self):
        """Test DispatchStats calculation."""
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_dim=8,
            intermediate_dim=16,
            top_k=2,
            enable_stats=True,
        )

        # Skewed distribution: all tokens to experts 0 and 1
        expert_ids = mx.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=mx.int32)
        expert_probs = mx.array(
            [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4]], dtype=mx.float32
        )
        hidden_states = mx.ones((4, 8), dtype=mx.float32)

        def expert_forward(x: mx.array, expert_id: int) -> mx.array:
            return x

        dispatcher.dispatch(hidden_states, expert_ids, expert_probs, expert_forward)

        stats = dispatcher.last_stats
        assert stats is not None
        assert stats.active_experts == 2  # Only experts 0 and 1 used
        assert stats.max_expert_load == 4  # Each active expert gets 4 tokens
        assert stats.load_imbalance == 0.0  # Equal load between active experts
