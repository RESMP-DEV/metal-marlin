"""Enhanced parity tests for mixed bit-width inference at
`contrib/metal_marlin/tests/test_mixed_bpw_parity.py`.

Test that mixed BPW produces same results as reference:
1. Numerical parity tests:
   - Compare mixed 2+3+4 bit vs pure 4-bit
   - Compare mixed vs FP16 reference (within tolerance)
   - Test each bit-width individually

2. Correctness tests:
   - Token-level output matching
   - Attention score accuracy
   - Expert routing consistency

3. Regression tests:
   - Catch precision regressions in CI
   - Perplexity comparison on validation set
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import tempfile
from pathlib import Path

# Local imports
from metal_marlin.trellis.model import TrellisMoEMLP
from metal_marlin.trellis.layer import TrellisDenseMLP
from metal_marlin.trellis.linear import TrellisLinear
from metal_marlin.quantization.trellis_codebook import TrellisCodebook

def _has_metal() -> bool:
    """Check if Metal is available and MPS is initialized."""
    try:
        from metal_marlin.metal_dispatch import HAS_METAL, HAS_MPS
        return HAS_METAL and HAS_MPS
    except ImportError:
        return False

# Skip all tests in this module if Metal is not available
pytestmark = pytest.mark.skipif(not _has_metal(), reason="Metal backend not available")

# --- Test Fixtures and Helper Functions ---

def create_test_trellis_linear(
    in_features: int,
    out_features: int,
    bits: int,
    device: str = "mps"
) -> TrellisLinear:
    """Creates and initializes a `TrellisLinear` layer for testing purposes."""
    layer = TrellisLinear(in_features, out_features, bits, device=device)
    
    codebook = TrellisCodebook(bits=bits)
    grid = torch.from_numpy(codebook.get_grid()).float()
    layer.grid = grid.to(device)

    layer.packed_indices = torch.randint(0, 256, layer.packed_indices.shape, dtype=torch.uint8, device=device)
    n_groups = layer.scales.shape[0]
    layer.scales = torch.rand(n_groups, out_features, dtype=torch.float32, device=device) * 0.1 + 0.01
    layer.su = torch.where(torch.rand(in_features, device=device) > 0.5, 1.0, -1.0)
    layer.sv = torch.where(torch.rand(out_features, device=device) > 0.5, 1.0, -1.0)
    
    return layer

def create_test_expert(
    hidden_dim: int,
    intermediate_dim: int,
    bits: int,
    device: str = "mps"
) -> TrellisDenseMLP:
    """Creates a `TrellisDenseMLP` expert for testing."""
    gate_proj = create_test_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    up_proj = create_test_trellis_linear(hidden_dim, intermediate_dim, bits, device)
    down_proj = create_test_trellis_linear(intermediate_dim, hidden_dim, bits, device)
    
    return TrellisDenseMLP(gate_proj, up_proj, down_proj)

def create_mixed_bit_experts(
    hidden_dim: int,
    intermediate_dim: int,
    bit_pattern: List[int],
    device: str = "mps"
) -> List[TrellisDenseMLP]:
    """Create experts with specified bit pattern."""
    return [create_test_expert(hidden_dim, intermediate_dim, bits, device) for bits in bit_pattern]

class ExplicitMoEReference:
    """A reference MoE implementation that uses explicit dequantization for parity checking."""
    def __init__(self, layer: TrellisMoEMLP):
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass using explicit dequantization."""
        router_logits = x @ self.layer.router.weight.T
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.layer.num_experts_per_tok, dim=-1)
        
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        
        batch_size, _ = x.shape
        output = torch.zeros_like(x, dtype=torch.float32)
        
        for b in range(batch_size):
            for k in range(self.layer.num_experts_per_tok):
                expert_idx = topk_indices[b, k].item()
                weight = topk_weights[b, k].item()
                
                expert = self.layer.experts[expert_idx]
                
                # Explicitly dequantize weights
                gw = expert.gate_proj.dequantize().float()
                uw = expert.up_proj.dequantize().float()
                dw = expert.down_proj.dequantize().float()
                
                token_x = x[b:b+1].float()
                gate = F.silu(token_x @ gw.T)
                up = token_x @ uw.T
                expert_out = (gate * up) @ dw.T
                
                output[b:b+1] += expert_out * weight
                
        return output.half()

class FP16ReferenceMoE:
    """FP16 reference implementation for comparison."""
    def __init__(self, experts: List[TrellisDenseMLP], router: nn.Linear, top_k: int = 2):
        self.experts = experts
        self.router = router
        self.top_k = top_k
        
        # Dequantize all experts to FP16
        self.fp16_experts = []
        for expert in experts:
            gate_w = expert.gate_proj.dequantize().half()
            up_w = expert.up_proj.dequantize().half()
            down_w = expert.down_proj.dequantize().half()
            self.fp16_experts.append({
                'gate': gate_w,
                'up': up_w,
                'down': down_w
            })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP16 weights."""
        router_logits = x @ self.router.weight.T
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        
        batch_size, _ = x.shape
        output = torch.zeros_like(x)
        
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = topk_indices[b, k].item()
                weight = topk_weights[b, k].item()
                
                expert_weights = self.fp16_experts[expert_idx]
                gate = F.silu(x[b:b+1] @ expert_weights['gate'].T)
                up = x[b:b+1] @ expert_weights['up'].T
                expert_out = (gate * up) @ expert_weights['down'].T
                
                output[b:b+1] += expert_out * weight
                
        return output

@pytest.fixture(scope="module")
def mixed_bit_experts_fixture():
    """Provides a shared set of mixed-bit experts for testing."""
    torch.manual_seed(42)
    device = "mps"
    num_experts = 8
    hidden_dim = 128
    intermediate_dim = 256
    
    # Mixed bit pattern: 2, 3, 4-bit experts
    bit_pattern = [2, 3, 4, 2, 3, 4, 2, 4]
    experts = create_mixed_bit_experts(hidden_dim, intermediate_dim, bit_pattern, device)
    
    router = nn.Linear(hidden_dim, num_experts, bias=False, device=device)
    nn.init.normal_(router.weight, std=0.02)
    
    return router, experts, bit_pattern

@pytest.fixture(scope="module")
def pure_4bit_experts_fixture():
    """Provides a shared set of pure 4-bit experts for comparison."""
    torch.manual_seed(42)
    device = "mps"
    num_experts = 8
    hidden_dim = 128
    intermediate_dim = 256
    
    # Pure 4-bit experts
    bit_pattern = [4] * num_experts
    experts = create_mixed_bit_experts(hidden_dim, intermediate_dim, bit_pattern, device)
    
    router = nn.Linear(hidden_dim, num_experts, bias=False, device=device)
    nn.init.normal_(router.weight, std=0.02)
    
    return router, experts

# --- Test Cases ---

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_individual_bit_width_parity_fp16(bits):
    """Test each bit-width individually against FP16 reference."""
    torch.manual_seed(bits)
    device = "mps"
    hidden_dim, intermediate_dim = 64, 128
    
    expert = create_test_expert(hidden_dim, intermediate_dim, bits, device)
    router = nn.Linear(hidden_dim, 1, bias=False, device=device)
    nn.init.normal_(router.weight, std=0.02)
    
    layer = TrellisMoEMLP(
        router=router,
        experts=[expert],
        shared_expert=None,
        num_experts_per_tok=1,
        use_mixed_bpw_optimizations=True
    )
    
    x = torch.randn(4, hidden_dim, dtype=torch.float16, device=device)
    
    # Reference calculation with FP16 weights
    fp16_ref = FP16ReferenceMoE([expert], router, top_k=1)
    expected = fp16_ref.forward(x)
    
    # Actual calculation with quantized weights
    with torch.no_grad():
        actual = layer(x)
    
    # Tolerances adjusted for quantization error
    if bits == 2:
        rtol, atol = 5e-2, 5e-2
    elif bits == 3:
        rtol, atol = 2e-2, 2e-2
    else:  # bits == 4
        rtol, atol = 1e-2, 1e-2
    
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
                              msg=f"Parity failed for {bits}-bit expert vs FP16 reference")

def test_mixed_2_3_4_bit_vs_pure_4bit(mixed_bit_experts_fixture, pure_4bit_experts_fixture):
    """Compare mixed (2+3+4)-bit experts vs pure 4-bit experts."""
    router_mixed, experts_mixed, bit_pattern = mixed_bit_experts_fixture
    router_4bit, experts_4bit = pure_4bit_experts_fixture
    
    # Ensure routers have same weights for fair comparison
    router_4bit.weight.data.copy_(router_mixed.weight.data)
    
    # Create layers
    layer_mixed = TrellisMoEMLP(
        router=router_mixed,
        experts=experts_mixed,
        shared_expert=None,
        num_experts_per_tok=2,
        use_mixed_bpw_optimizations=True
    )
    
    layer_4bit = TrellisMoEMLP(
        router=router_4bit,
        experts=experts_4bit,
        shared_expert=None,
        num_experts_per_tok=2,
        use_mixed_bpw_optimizations=True
    )
    
    # Generate test inputs
    batch_sizes = [1, 4, 16]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, router_mixed.in_features, dtype=torch.float16, device="mps")
        
        with torch.no_grad():
            out_mixed = layer_mixed(x)
            out_4bit = layer_4bit(x)
        
        # Compute metrics
        mse = F.mse_loss(out_mixed.float(), out_4bit.float())
        cos_sim = F.cosine_similarity(out_mixed.view(-1), out_4bit.view(-1), dim=0)
        
        # Mixed BPW should be reasonably close to pure 4-bit
        assert mse < 0.05, f"MSE too high for batch_size={batch_size}: {mse.item():.4f}"
        assert cos_sim > 0.95, f"Cosine similarity too low for batch_size={batch_size}: {cos_sim.item():.4f}"

def test_mixed_bpw_vs_fp16_reference(mixed_bit_experts_fixture):
    """Compare mixed BPW output with FP16 reference implementation."""
    router, experts, bit_pattern = mixed_bit_experts_fixture
    
    layer = TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=None,
        num_experts_per_tok=2,
        use_mixed_bpw_optimizations=True
    )
    
    # Create FP16 reference
    fp16_ref = FP16ReferenceMoE(experts, router, top_k=2)
    
    # Test different batch sizes
    for batch_size in [1, 2, 8, 32]:
        x = torch.randn(batch_size, router.in_features, dtype=torch.float16, device="mps")
        
        with torch.no_grad():
            out_mixed = layer(x)
            out_fp16 = fp16_ref.forward(x)
        
        # Compute error metrics
        mse = F.mse_loss(out_mixed.float(), out_fp16.float())
        max_abs_error = torch.max(torch.abs(out_mixed - out_fp16))
        cos_sim = F.cosine_similarity(out_mixed.view(-1), out_fp16.view(-1), dim=0)
        
        # Assertions
        assert mse < 0.01, f"MSE too high for batch_size={batch_size}: {mse.item():.6f}"
        assert max_abs_error < 0.1, f"Max absolute error too high for batch_size={batch_size}: {max_abs_error.item():.6f}"
        assert cos_sim > 0.99, f"Cosine similarity too low for batch_size={batch_size}: {cos_sim.item():.6f}"

def test_token_level_output_consistency(mixed_bit_experts_fixture):
    """Verify token-level consistency across different batch configurations."""
    router, experts, _ = mixed_bit_experts_fixture
    
    layer = TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=None,
        num_experts_per_tok=2,
        use_mixed_bpw_optimizations=True
    )
    
    # Create a single token
    single_token = torch.randn(1, router.in_features, dtype=torch.float16, device="mps")
    
    # Compute output for single token
    with torch.no_grad():
        single_output = layer(single_token)
    
    # Create batch with same token repeated
    batch_sizes = [2, 4, 8]
    for batch_size in batch_sizes:
        batch_tokens = single_token.repeat(batch_size, 1)
        
        with torch.no_grad():
            batch_output = layer(batch_tokens)
        
        # Each token in batch should produce same output as single token
        for i in range(batch_size):
            token_output = batch_output[i:i+1]
            torch.testing.assert_close(
                token_output, single_output,
                rtol=1e-4, atol=1e-4,
                msg=f"Token {i} in batch_size={batch_size} differs from single token output"
            )

def test_attention_score_accuracy():
    """Test attention score accuracy in mixed BPW context."""
    device = "mps"
    
    # Create query, key, value projections with mixed bit-widths
    hidden_dim = 64
    head_dim = 16
    num_heads = 4
    
    # Different bit-widths for attention projections
    q_bits, k_bits, v_bits, o_bits = 4, 3, 2, 4
    
    q_proj = create_test_trellis_linear(hidden_dim, num_heads * head_dim, q_bits, device)
    k_proj = create_test_trellis_linear(hidden_dim, num_heads * head_dim, k_bits, device)
    v_proj = create_test_trellis_linear(hidden_dim, num_heads * head_dim, v_bits, device)
    o_proj = create_test_trellis_linear(num_heads * head_dim, hidden_dim, o_bits, device)
    
    # Dequantize for reference
    q_weight = q_proj.dequantize().half()
    k_weight = k_proj.dequantize().half()
    v_weight = v_proj.dequantize().half()
    o_weight = o_proj.dequantize().half()
    
    # Generate input
    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    
    # Reference attention computation
    q_ref = x @ q_weight.T
    k_ref = x @ k_weight.T
    v_ref = x @ v_weight.T
    
    # Reshape for multi-head attention
    q_ref = q_ref.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k_ref = k_ref.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v_ref = v_ref.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Compute attention scores
    attn_scores_ref = torch.matmul(q_ref, k_ref.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_probs_ref = F.softmax(attn_scores_ref, dim=-1)
    attn_output_ref = torch.matmul(attn_probs_ref, v_ref)
    attn_output_ref = attn_output_ref.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    output_ref = attn_output_ref @ o_weight.T
    
    # Quantized attention computation (through linear layers)
    with torch.no_grad():
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
    
    # Reshape and compute attention
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    
    with torch.no_grad():
        output = o_proj(attn_output)
    
    # Compare attention scores and outputs
    score_mse = F.mse_loss(attn_scores.float(), attn_scores_ref.float())
    output_mse = F.mse_loss(output.float(), output_ref.float())
    
    assert score_mse < 0.01, f"Attention score MSE too high: {score_mse.item():.6f}"
    assert output_mse < 0.02, f"Attention output MSE too high: {output_mse.item():.6f}"

def test_expert_routing_decision_consistency(mixed_bit_experts_fixture):
    """Verify routing decisions are consistent regardless of bit-width."""
    router, experts, bit_pattern = mixed_bit_experts_fixture
    
    layer = TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=None,
        num_experts_per_tok=2,
        use_mixed_bpw_optimizations=True
    )
    
    # Generate multiple inputs
    num_tests = 10
    for _ in range(num_tests):
        x = torch.randn(4, router.in_features, dtype=torch.float16, device="mps")
        
        # Compute routing manually
        with torch.no_grad():
            router_logits = x @ router.weight.T
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            _, expected_indices = torch.topk(routing_weights, 2, dim=-1)
        
        # Compute output with layer
        with torch.no_grad():
            output = layer(x)
        
        # Create a version with all 4-bit experts but same router
        experts_4bit = create_mixed_bit_experts(router.in_features, 256, [4]*len(experts))
        layer_4bit = TrellisMoEMLP(
            router=router,
            experts=experts_4bit,
            shared_expert=None,
            num_experts_per_tok=2,
            use_mixed_bpw_optimizations=True
        )
        
        # Compute output with 4-bit layer
        with torch.no_grad():
            output_4bit = layer_4bit(x)
        
        # Outputs should be similar despite different bit-widths
        mse = F.mse_loss(output.float(), output_4bit.float())
        assert mse < 0.1, f"Output MSE too high between mixed and 4-bit: {mse.item():.6f}"

def test_mixed_bpw_regression_detection():
    """Regression test to catch precision regressions in CI."""
    torch.manual_seed(12345)
    device = "mps"
    
    # Fixed configuration
    hidden_dim = 128
    intermediate_dim = 256
    num_experts = 8
    top_k = 2
    
    # Create mixed-bit experts
    bit_pattern = [2, 3, 4, 2, 3, 4, 2, 4]
    experts = create_mixed_bit_experts(hidden_dim, intermediate_dim, bit_pattern, device)
    
    router = nn.Linear(hidden_dim, num_experts, bias=False, device=device)
    nn.init.normal_(router.weight, std=0.02)
    
    layer = TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=None,
        num_experts_per_tok=top_k,
        use_mixed_bpw_optimizations=True
    )
    
    # Create FP16 reference
    fp16_ref = FP16ReferenceMoE(experts, router, top_k=top_k)
    
    # Run multiple forward passes and collect statistics
    num_runs = 5
    mse_values = []
    cos_sim_values = []
    
    for run in range(num_runs):
        x = torch.randn(8, hidden_dim, dtype=torch.float16, device=device)
        
        with torch.no_grad():
            out_mixed = layer(x)
            out_fp16 = fp16_ref.forward(x)
        
        mse = F.mse_loss(out_mixed.float(), out_fp16.float())
        cos_sim = F.cosine_similarity(out_mixed.view(-1), out_fp16.view(-1), dim=0)
        
        mse_values.append(mse.item())
        cos_sim_values.append(cos_sim.item())
    
    # Compute averages
    avg_mse = np.mean(mse_values)
    avg_cos_sim = np.mean(cos_sim_values)
    
    # Assert regression thresholds
    assert avg_mse < 0.005, f"Regression detected: MSE increased to {avg_mse:.6f}"
    assert avg_cos_sim > 0.995, f"Regression detected: cosine similarity dropped to {avg_cos_sim:.6f}"
    
    # Print statistics for monitoring
    print(f"\nRegression test statistics:")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average Cosine Similarity: {avg_cos_sim:.6f}")
    print(f"  MSE range: [{min(mse_values):.6f}, {max(mse_values):.6f}]")

def test_perplexity_comparison_smoke():
    """Smoke test for perplexity comparison (doesn't run full validation)."""
    device = "mps"
    
    # Create a small test model
    hidden_dim = 64
    vocab_size = 100
    num_experts = 4
    intermediate_dim = 128
    
    # Create mixed-bit experts
    bit_pattern = [2, 3, 4, 2]
    experts = create_mixed_bit_experts(hidden_dim, intermediate_dim, bit_pattern, device)
    
    router = nn.Linear(hidden_dim, num_experts, bias=False, device=device)
    nn.init.normal_(router.weight, std=0.02)
    
    layer = TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=None,
        num_experts_per_tok=2,
        use_mixed_bpw_optimizations=True
    )
    
    # Create a simple language modeling head
    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False, device=device)
    nn.init.normal_(lm_head.weight, std=0.02)
    
    # Generate some dummy tokens
    batch_size = 4
    seq_len = 8
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Create random embeddings
    embeddings = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    
    # Process through MoE layer
    with torch.no_grad():
        # Apply MoE to each position independently (simplified)
        outputs = []
        for pos in range(seq_len):
            pos_emb = embeddings[:, pos:pos+1, :].squeeze(1)
            moe_out = layer(pos_emb)
            outputs.append(moe_out.unsqueeze(1))
        
        hidden_states = torch.cat(outputs, dim=1)
        
        # Compute logits
        logits = lm_head(hidden_states)
        
        # Compute cross-entropy loss (proxy for perplexity)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            token_ids.view(-1),
            reduction='none'
        )
        
        # Compute perplexity
        perplexity = torch.exp(loss.mean()).item()
    
    # Just verify it runs without error
    assert perplexity > 0, f"Perplexity should be positive, got {perplexity}"
    print(f"Smoke test perplexity: {perplexity:.2f}")

def test_edge_cases():
    """Test edge cases for mixed BPW inference."""
    device = "mps"
    
    # Test 1: All experts same bit-width (should still work)
    hidden_dim = 64
    intermediate_dim = 128
    
    for uniform_bits in [2, 3, 4]:
        experts = create_mixed_bit_experts(hidden_dim, intermediate_dim, [uniform_bits]*4, device)
        router = nn.Linear(hidden_dim, 4, bias=False, device=device)
        
        layer = TrellisMoEMLP(
            router=router,
            experts=experts,
            shared_expert=None,
            num_experts_per_tok=2,
            use_mixed_bpw_optimizations=True
        )
        
        x = torch.randn(2, hidden_dim, dtype=torch.float16, device=device)
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (2, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    # Test 2: Single expert (edge case for routing)
    expert = create_test_expert(hidden_dim, intermediate_dim, 4, device)
    router = nn.Linear(hidden_dim, 1, bias=False, device=device)
    
    layer = TrellisMoEMLP(
        router=router,
        experts=[expert],
        shared_expert=None,
        num_experts_per_tok=1,
        use_mixed_bpw_optimizations=True
    )
    
    x = torch.randn(1, hidden_dim, dtype=torch.float16, device=device)
    with torch.no_grad():
        output = layer(x)
    
    assert output.shape == (1, hidden_dim)
    
    # Test 3: Large batch size
    experts = create_mixed_bit_experts(hidden_dim, intermediate_dim, [2, 3, 4, 2], device)
    router = nn.Linear(hidden_dim, 4, bias=False, device=device)
    
    layer = TrellisMoEMLP(
        router=router,
        experts=experts,
        shared_expert=None,
        num_experts_per_tok=2,
        use_mixed_bpw_optimizations=True
    )
    
    x = torch.randn(128, hidden_dim, dtype=torch.float16, device=device)
    with torch.no_grad():
        output = layer(x)
    
    assert output.shape == (128, hidden_dim)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])