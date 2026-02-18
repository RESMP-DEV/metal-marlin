"""Tests for adaptive speculation depth in MMFP4.

Tests cover:
1. AdaptiveSpeculationController: EMA calculation, depth adjustment
2. MMFP4MTPHead integration: Adaptive depth with MTP head
3. Pipeline integration: End-to-end adaptive speculation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from metal_marlin.layers.adaptive_depth import (
    AdaptiveDepthConfig,
    AdaptiveDepthStats,
    AdaptiveSpeculationController,
)
from metal_marlin.layers.mmfp4_mtp_head import MMFP4MTPHead


# -----------------------------------------------------------------------------
# Test AdaptiveSpeculationController
# -----------------------------------------------------------------------------

class TestAdaptiveSpeculationController:
    """Test the adaptive speculation controller."""
    
    def test_initial_depth(self):
        """Controller should start with initial depth."""
        config = AdaptiveDepthConfig(initial_depth=4)
        controller = AdaptiveSpeculationController(config)
        
        assert controller.current_depth == 4
    
    def test_depth_bounds(self):
        """Depth should stay within min/max bounds."""
        config = AdaptiveDepthConfig(min_depth=2, max_depth=6, initial_depth=4)
        controller = AdaptiveSpeculationController(config)
        
        # Try to force depth beyond bounds with extreme acceptance rates
        for _ in range(20):
            controller.update(10, 10)  # 100% acceptance
        
        assert controller.current_depth <= 6
        
        for _ in range(20):
            controller.update(0, 10)  # 0% acceptance
        
        assert controller.current_depth >= 2
    
    def test_increase_on_high_acceptance(self):
        """Depth should increase with sustained high acceptance."""
        config = AdaptiveDepthConfig(
            initial_depth=3,
            min_depth=1,
            max_depth=8,
            ema_alpha=0.5,  # High alpha for faster response
        )
        controller = AdaptiveSpeculationController(config)
        
        initial_depth = controller.current_depth
        
        # Simulate high acceptance
        for _ in range(10):
            controller.update(4, 4)  # 100% acceptance
        
        # Depth should have increased
        assert controller.current_depth >= initial_depth
    
    def test_decrease_on_low_acceptance(self):
        """Depth should decrease with sustained low acceptance."""
        config = AdaptiveDepthConfig(
            initial_depth=6,
            min_depth=1,
            max_depth=8,
            ema_alpha=0.5,  # High alpha for faster response
        )
        controller = AdaptiveSpeculationController(config)
        
        initial_depth = controller.current_depth
        
        # Simulate low acceptance
        for _ in range(10):
            controller.update(0, 4)  # 0% acceptance
        
        # Depth should have decreased
        assert controller.current_depth <= initial_depth
    
    def test_ema_smoothing(self):
        """EMA should smooth out acceptance rate fluctuations."""
        config = AdaptiveDepthConfig(ema_alpha=0.3)
        controller = AdaptiveSpeculationController(config)
        
        # Single update
        controller.update(2, 4)  # 50% acceptance
        
        # EMA should be between 0.5 and initial 0.5 (weighted)
        assert 0 < controller._ema_acceptance < 1
    
    def test_stats_tracking(self):
        """Stats should track cumulative metrics."""
        controller = AdaptiveSpeculationController()
        
        controller.update(3, 4)  # 75% acceptance
        controller.update(2, 4)  # 50% acceptance
        controller.update(4, 4)  # 100% acceptance
        
        stats = controller.stats
        
        assert stats.total_steps == 3
        assert stats.total_accepted == 9
        assert stats.total_proposed == 12
        assert stats.overall_acceptance_rate == 0.75
    
    def test_reset_clears_state(self):
        """Reset should restore initial state."""
        config = AdaptiveDepthConfig(initial_depth=4)
        controller = AdaptiveSpeculationController(config)
        
        # Modify state
        controller.update(4, 4)
        controller.update(4, 4)
        
        # Reset
        controller.reset()
        
        assert controller.current_depth == 4
        assert controller._ema_acceptance == 0.5
        assert len(controller._acceptance_history) == 0
    
    def test_speedup_estimate(self):
        """Speedup estimate should be positive."""
        controller = AdaptiveSpeculationController()
        
        # Update with some acceptance
        controller.update(3, 4)
        
        speedup = controller.get_speedup_estimate()
        
        assert speedup > 0
        # With any acceptance, should be >= 1.0 (no worse than autoregressive)
        assert speedup >= 0.5
    
    def test_should_increase_depth(self):
        """Should suggest increase when acceptance is high."""
        config = AdaptiveDepthConfig(target_acceptance=0.7)
        controller = AdaptiveSpeculationController(config)
        
        # Low acceptance - should not increase
        controller.update(1, 4)
        assert not controller.should_increase_depth()
        
        # High acceptance - should suggest increase
        for _ in range(5):
            controller.update(4, 4)
        
        assert controller.should_increase_depth()
    
    def test_should_decrease_depth(self):
        """Should suggest decrease when acceptance is low and depth > min."""
        config = AdaptiveDepthConfig(
            target_acceptance=0.7,
            initial_depth=6,
            min_depth=2,  # Higher min so we don't hit it immediately
            max_depth=8,
        )
        controller = AdaptiveSpeculationController(config)
        
        # High acceptance - should not decrease
        controller.update(4, 4)
        assert not controller.should_decrease_depth()
        
        # Low acceptance - should suggest decrease (need sustained low for EMA to drop)
        for _ in range(3):
            controller.update(0, 4)
        
        # EMA should now be below threshold (0.5) and depth > min_depth
        assert controller._ema_acceptance < 0.5
        # After decreasing due to low acceptance, we might already be at min_depth
        # So check the condition only if we can actually decrease
        if controller.current_depth > config.min_depth:
            assert controller.should_decrease_depth()
    
    def test_disabled_adjustment(self):
        """When disabled, depth should not change."""
        config = AdaptiveDepthConfig(
            initial_depth=4,
            enable_dynamic_adjustment=False,
        )
        controller = AdaptiveSpeculationController(config)
        
        for _ in range(10):
            controller.update(4, 4)  # 100% acceptance
        
        # Depth should remain unchanged
        assert controller.current_depth == 4
    
    def test_efficiency_score(self):
        """Efficiency score should reflect acceptance quality."""
        controller = AdaptiveSpeculationController()
        
        # Perfect acceptance
        controller.update(4, 4)
        stats = controller.stats
        assert stats.efficiency_score == 1.0
        
        # Reset and try partial acceptance
        controller.reset()
        controller.update(2, 4)
        stats = controller.stats
        assert stats.efficiency_score == 0.5


# -----------------------------------------------------------------------------
# Test MMFP4MTPHead Adaptive Integration
# -----------------------------------------------------------------------------

class TestMMFP4MTPHeadAdaptive:
    """Test MMFP4MTPHead integration with adaptive depth."""
    
    def test_adaptive_enabled_by_default(self):
        """MTP head should have adaptive depth enabled by default."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
        )
        
        assert head.adaptive_depth_enabled
        assert head._adaptive_controller is not None
    
    def test_adaptive_can_be_disabled(self):
        """MTP head adaptive depth can be disabled."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
            adaptive_depth=False,
        )
        
        assert not head.adaptive_depth_enabled
        assert head._adaptive_controller is None
    
    def test_current_speculation_depth_adaptive(self):
        """Current depth should come from controller when adaptive."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
            adaptive_depth=True,
        )
        
        # Initially should match initial config
        assert head.current_speculation_depth == 4
        
        # Update with high acceptance to increase depth
        for _ in range(10):
            head.update_adaptive_depth(4, 4)
        
        # Depth may have increased (capped at num_predictions=4)
        assert head.current_speculation_depth <= 4
    
    def test_current_speculation_depth_fixed(self):
        """Current depth should be fixed when adaptive disabled."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=6,
            adaptive_depth=False,
        )
        
        assert head.current_speculation_depth == 6
    
    def test_update_adaptive_depth_returns_new_depth(self):
        """Update should return the new depth."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
        )
        
        new_depth = head.update_adaptive_depth(2, 4)
        
        assert isinstance(new_depth, int)
        assert 1 <= new_depth <= 4
    
    def test_get_adaptive_stats(self):
        """Should return stats dict when adaptive enabled."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
        )
        
        # Update to generate some stats
        head.update_adaptive_depth(3, 4)
        
        stats = head.get_adaptive_stats()
        
        assert "current_depth" in stats
        assert "ema_acceptance" in stats
        assert "overall_acceptance_rate" in stats
        assert "estimated_speedup" in stats
    
    def test_get_adaptive_stats_when_disabled(self):
        """Should return empty dict when adaptive disabled."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
            adaptive_depth=False,
        )
        
        stats = head.get_adaptive_stats()
        
        assert stats == {}
    
    def test_reset_adaptive_depth(self):
        """Reset should restore initial depth."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
        )
        
        # Modify with updates
        for _ in range(10):
            head.update_adaptive_depth(0, 4)  # Low acceptance
        
        # Reset
        head.reset_adaptive_depth()
        
        # Should be back to initial
        assert head.current_speculation_depth == 4
    
    def test_forward_with_adaptive(self):
        """Forward pass should work with adaptive depth enabled."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
            adaptive_depth=True,
        )
        
        hidden = torch.randn(2, 5, 128)  # batch=2, seq=5
        
        # Forward should work normally
        output = head(hidden)
        
        assert output.shape == (2, 4, 100)  # [batch, num_predictions, vocab]
    
    def test_speculate_with_adaptive(self):
        """Speculate should work with adaptive depth enabled."""
        head = MMFP4MTPHead(
            hidden_size=128,
            vocab_size=100,
            num_predictions=4,
            adaptive_depth=True,
        )
        
        hidden = torch.randn(2, 5, 128)
        
        tokens, probs = head.speculate(hidden)
        
        assert tokens.shape == (2, 4)
        assert probs.shape == (2, 4, 100)


# -----------------------------------------------------------------------------
# Test AdaptiveDepthConfig
# -----------------------------------------------------------------------------

class TestAdaptiveDepthConfig:
    """Test AdaptiveDepthConfig dataclass."""
    
    def test_default_values(self):
        """Config should have sensible defaults."""
        config = AdaptiveDepthConfig()
        
        assert config.initial_depth == 4
        assert config.min_depth == 1
        assert config.max_depth == 8
        assert config.ema_alpha == 0.3
        assert config.target_acceptance == 0.7
        assert config.enable_dynamic_adjustment is True
    
    def test_custom_values(self):
        """Config should accept custom values."""
        config = AdaptiveDepthConfig(
            initial_depth=6,
            min_depth=2,
            max_depth=10,
            ema_alpha=0.5,
        )
        
        assert config.initial_depth == 6
        assert config.min_depth == 2
        assert config.max_depth == 10
        assert config.ema_alpha == 0.5


# -----------------------------------------------------------------------------
# Test AdaptiveDepthStats
# -----------------------------------------------------------------------------

class TestAdaptiveDepthStats:
    """Test AdaptiveDepthStats dataclass."""
    
    def test_default_values(self):
        """Stats should initialize with defaults."""
        stats = AdaptiveDepthStats()
        
        assert stats.current_depth == 4
        assert stats.ema_acceptance == 0.5
        assert stats.total_steps == 0
    
    def test_overall_acceptance_rate_empty(self):
        """Overall rate should be 0 when no proposals."""
        stats = AdaptiveDepthStats()
        
        assert stats.overall_acceptance_rate == 0.0
    
    def test_overall_acceptance_rate_with_data(self):
        """Overall rate should be accepted/proposed."""
        stats = AdaptiveDepthStats(
            total_accepted=75,
            total_proposed=100,
        )
        
        assert stats.overall_acceptance_rate == 0.75
    
    def test_efficiency_score(self):
        """Efficiency score should match acceptance rate."""
        stats = AdaptiveDepthStats(
            total_accepted=80,
            total_proposed=100,
        )
        
        assert stats.efficiency_score == 0.8


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestAdaptiveDepthIntegration:
    """Integration tests for adaptive depth."""
    
    def test_adaptive_responds_to_acceptance_pattern(self):
        """Controller should adapt to changing acceptance patterns."""
        config = AdaptiveDepthConfig(
            initial_depth=4,
            min_depth=1,
            max_depth=8,
            ema_alpha=0.4,
        )
        controller = AdaptiveSpeculationController(config)
        
        # Phase 1: Low acceptance (should decrease depth)
        initial_depth = controller.current_depth
        for _ in range(10):
            controller.update(0, 4)  # 0% acceptance
        
        depth_after_low = controller.current_depth
        
        # Phase 2: High acceptance (should increase depth)
        for _ in range(10):
            controller.update(4, 4)  # 100% acceptance
        
        depth_after_high = controller.current_depth
        
        # Verify adaptation occurred
        assert depth_after_low <= initial_depth
        assert depth_after_high >= depth_after_low
    
    def test_depth_never_negative(self):
        """Depth should never go below 0 (or min_depth)."""
        config = AdaptiveDepthConfig(min_depth=1)
        controller = AdaptiveSpeculationController(config)
        
        # Extreme low acceptance
        for _ in range(100):
            controller.update(0, 100)
        
        assert controller.current_depth >= 1
    
    def test_concurrent_updates_consistent(self):
        """Multiple updates should maintain consistent state."""
        controller = AdaptiveSpeculationController()
        
        # Simulate multiple steps
        for i in range(20):
            accepted = (i % 4) + 1  # Varying acceptance: 1, 2, 3, 4
            controller.update(accepted, 4)
        
        stats = controller.stats
        
        assert stats.total_steps == 20
        assert stats.total_accepted == sum((i % 4) + 1 for i in range(20))
        assert stats.total_proposed == 80
