"""Parity tests for mixed bit-width (BPW) inference.

This module validates numerical and behavioral parity for mixed 2/3/4-bit MoE
inference using deterministic synthetic fixtures that run on CPU.

Coverage targets:
1. Numerical parity:
   - Mixed (2+3+4)-bit vs pure 4-bit.
   - Mixed vs FP16 reference.
   - Individual 2/3/4-bit expert quality.
2. Correctness:
   - Token-level output matching.
   - Attention-score accuracy.
   - Expert-routing consistency.
3. Regression:
   - CI-friendly precision guardrails.
   - Perplexity comparison on a synthetic validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

from metal_marlin._compat import HAS_TORCH, torch

if not HAS_TORCH or torch is None:
    pytest.skip("PyTorch required", allow_module_level=True)

import torch.nn as nn
import torch.nn.functional as F

try:
    from metal_marlin.quantization.trellis_codebook import TrellisCodebook
    from metal_marlin.trellis.linear import TrellisLinear

    _HAS_TRELLIS = True
except Exception:
    _HAS_TRELLIS = False

pytestmark = pytest.mark.skipif(
    not _HAS_TRELLIS,
    reason="Trellis quantization modules are unavailable",
)


@dataclass
class _MockTrellisWeight:
    packed_indices: torch.Tensor
    scales: torch.Tensor
    su: torch.Tensor
    sv: torch.Tensor
    bits: int
    original_shape: tuple[int, int]


@dataclass(frozen=True)
class _PreparedExpert:
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    bits: int


@dataclass(frozen=True)
class _ParityFixture:
    hidden_dim: int
    top_k: int
    router: nn.Linear
    mixed_bits: tuple[int, ...]
    reference_experts: tuple[_PreparedExpert, ...]
    mixed_experts: tuple[_PreparedExpert, ...]
    pure_4bit_experts: tuple[_PreparedExpert, ...]
    token_embedding: torch.Tensor
    lm_head: torch.Tensor
    validation_tokens: tuple[torch.Tensor, ...]


def _pack_tile_indices(tile_indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack a 16x16 tile of quant indices into Trellis byte layout."""
    flat = np.asarray(tile_indices, dtype=np.uint8).reshape(-1)

    if bits == 4:
        out = np.zeros(128, dtype=np.uint8)
        for i in range(128):
            out[i] = int(flat[2 * i] | (flat[2 * i + 1] << 4))
        return out

    if bits == 2:
        out = np.zeros(64, dtype=np.uint8)
        for i in range(64):
            out[i] = int(
                flat[4 * i]
                | (flat[4 * i + 1] << 2)
                | (flat[4 * i + 2] << 4)
                | (flat[4 * i + 3] << 6)
            )
        return out

    if bits == 3:
        out = np.zeros(96, dtype=np.uint8)
        dst = 0
        for i in range(0, 256, 8):
            a0, a1, a2, a3, a4, a5, a6, a7 = [int(v) for v in flat[i : i + 8]]
            out[dst] = (a0 & 0x7) | ((a1 & 0x7) << 3) | ((a2 & 0x3) << 6)
            out[dst + 1] = (
                ((a2 >> 2) & 0x1)
                | ((a3 & 0x7) << 1)
                | ((a4 & 0x7) << 4)
                | ((a5 & 0x1) << 7)
            )
            out[dst + 2] = ((a5 >> 1) & 0x3) | ((a6 & 0x7) << 2) | ((a7 & 0x7) << 5)
            dst += 3
        return out

    raise ValueError(f"Unsupported bit-width for parity tests: {bits}")


def _quantize_weight_to_trellis(weight: torch.Tensor, bits: int) -> _MockTrellisWeight:
    """Quantize an FP32 matrix into Trellis packed representation."""
    matrix = weight.detach().cpu().float().numpy()
    out_features, in_features = matrix.shape

    codebook = TrellisCodebook(bits=bits)
    grid = codebook.get_grid().astype(np.float32)
    zero_idx = int(np.argmin(np.abs(grid)))
    grid_abs_max = max(float(np.max(np.abs(grid))), 1e-6)

    n_groups = (in_features + 127) // 128
    scales = np.zeros((n_groups, out_features), dtype=np.float32)
    indices = np.full((out_features, in_features), zero_idx, dtype=np.uint8)

    for group_idx in range(n_groups):
        start = group_idx * 128
        end = min((group_idx + 1) * 128, in_features)
        block = matrix[:, start:end]
        scale = np.maximum(np.max(np.abs(block), axis=1) / grid_abs_max, 1e-6)
        scales[group_idx, :] = scale
        normalized = block / scale[:, None]
        distances = np.abs(normalized[..., None] - grid[None, None, :])
        indices[:, start:end] = np.argmin(distances, axis=-1).astype(np.uint8)

    tile_dim = 16
    tiles_k = (out_features + tile_dim - 1) // tile_dim
    tiles_n = (in_features + tile_dim - 1) // tile_dim
    packed_bytes = {2: 64, 3: 96, 4: 128}[bits]
    packed = np.zeros((tiles_k, tiles_n, packed_bytes), dtype=np.uint8)

    for tile_k in range(tiles_k):
        for tile_n in range(tiles_n):
            tile = np.full((tile_dim, tile_dim), zero_idx, dtype=np.uint8)
            for local_k in range(tile_dim):
                k = tile_k * tile_dim + local_k
                if k >= out_features:
                    continue
                for local_n in range(tile_dim):
                    n = tile_n * tile_dim + local_n
                    if n >= in_features:
                        continue
                    tile[local_k, local_n] = indices[k, n]
            packed[tile_k, tile_n] = _pack_tile_indices(tile.reshape(-1), bits)

    return _MockTrellisWeight(
        packed_indices=torch.from_numpy(packed),
        scales=torch.from_numpy(scales),
        su=torch.ones(in_features, dtype=torch.float32),
        sv=torch.ones(out_features, dtype=torch.float32),
        bits=bits,
        original_shape=(out_features, in_features),
    )


def _prepare_quantized_expert(base: _PreparedExpert, bits: int) -> _PreparedExpert:
    """Create a quantized expert by dequantizing TrellisLinear weights once."""
    gate_linear = TrellisLinear.from_trellis_weight(
        _quantize_weight_to_trellis(base.gate_weight, bits),
        device="cpu",
    )
    up_linear = TrellisLinear.from_trellis_weight(
        _quantize_weight_to_trellis(base.up_weight, bits),
        device="cpu",
    )
    down_linear = TrellisLinear.from_trellis_weight(
        _quantize_weight_to_trellis(base.down_weight, bits),
        device="cpu",
    )

    return _PreparedExpert(
        gate_weight=gate_linear.dequantize().float(),
        up_weight=up_linear.dequantize().float(),
        down_weight=down_linear.dequantize().float(),
        bits=bits,
    )


def _run_expert(expert: _PreparedExpert, x: torch.Tensor) -> torch.Tensor:
    gate = F.silu(x @ expert.gate_weight.t())
    up = x @ expert.up_weight.t()
    return (gate * up) @ expert.down_weight.t()


def _route_tokens(
    router: nn.Linear,
    x: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = x @ router.weight.t()
    routing_weights, selected_experts = torch.topk(
        torch.softmax(logits, dim=-1, dtype=torch.float32),
        k=top_k,
        dim=-1,
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return selected_experts, routing_weights


def _moe_forward(
    x: torch.Tensor,
    router: nn.Linear,
    experts: tuple[_PreparedExpert, ...],
    top_k: int,
    *,
    return_routing: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    selected_experts, routing_weights = _route_tokens(router, x, top_k)
    output = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)

    for token_idx in range(x.shape[0]):
        token = x[token_idx : token_idx + 1]
        for slot_idx in range(top_k):
            expert_id = int(selected_experts[token_idx, slot_idx].item())
            weight = routing_weights[token_idx, slot_idx]
            output[token_idx : token_idx + 1] += _run_expert(experts[expert_id], token) * weight

    if return_routing:
        return output, selected_experts, routing_weights
    return output


def _attention_scores(hidden_states: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(hidden_states.shape[-1])
    return (hidden_states @ hidden_states.t()) * scale


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = actual - expected
    return {
        "mse": float(torch.mean(diff.pow(2)).item()),
        "max_abs": float(diff.abs().max().item()),
        "cosine": float(
            F.cosine_similarity(actual.reshape(-1), expected.reshape(-1), dim=0).item()
        ),
    }


def _compute_perplexity(
    validation_tokens: tuple[torch.Tensor, ...],
    token_embedding: torch.Tensor,
    lm_head: torch.Tensor,
    router: nn.Linear,
    experts: tuple[_PreparedExpert, ...],
    top_k: int,
) -> float:
    total_nll = 0.0
    total_tokens = 0

    for token_seq in validation_tokens:
        if token_seq.numel() < 2:
            continue
        x = token_embedding[token_seq[:-1]].float()
        hidden = _moe_forward(x, router, experts, top_k)
        logits = hidden @ lm_head
        log_probs = F.log_softmax(logits, dim=-1)
        targets = token_seq[1:]
        token_nll = -log_probs[torch.arange(targets.numel()), targets]
        total_nll += float(token_nll.sum().item())
        total_tokens += int(targets.numel())

    if total_tokens == 0:
        raise ValueError("Validation split produced zero scored tokens")

    return math.exp(total_nll / total_tokens)


@pytest.fixture(scope="module")
def parity_fixture() -> _ParityFixture:
    torch.manual_seed(17)

    hidden_dim = 32
    intermediate_dim = 32
    num_experts = 6
    top_k = 2
    mixed_bits = (2, 3, 4, 2, 3, 4)

    router = nn.Linear(hidden_dim, num_experts, bias=False, dtype=torch.float32)
    with torch.no_grad():
        router.weight.copy_(torch.randn_like(router.weight) * 0.05)

    reference_experts: list[_PreparedExpert] = []
    for _ in range(num_experts):
        reference_experts.append(
            _PreparedExpert(
                gate_weight=torch.randn(intermediate_dim, hidden_dim) * 0.12,
                up_weight=torch.randn(intermediate_dim, hidden_dim) * 0.12,
                down_weight=torch.randn(hidden_dim, intermediate_dim) * 0.12,
                bits=16,
            )
        )

    mixed_experts = [
        _prepare_quantized_expert(expert, bits)
        for expert, bits in zip(reference_experts, mixed_bits)
    ]
    pure_4bit_experts = [
        _prepare_quantized_expert(expert, 4)
        for expert in reference_experts
    ]

    vocab_size = 64
    token_embedding = torch.randn(vocab_size, hidden_dim, dtype=torch.float32) * 0.1
    lm_head = torch.randn(hidden_dim, vocab_size, dtype=torch.float32) * 0.1
    validation_tokens = (
        torch.tensor([1, 4, 6, 3, 2, 9, 8, 7, 10, 2, 3, 4], dtype=torch.long),
        torch.tensor([5, 2, 1, 4, 9, 6, 11, 12, 4, 3], dtype=torch.long),
        torch.tensor([8, 8, 2, 2, 5, 5, 1, 1, 7], dtype=torch.long),
    )

    return _ParityFixture(
        hidden_dim=hidden_dim,
        top_k=top_k,
        router=router,
        mixed_bits=mixed_bits,
        reference_experts=tuple(reference_experts),
        mixed_experts=tuple(mixed_experts),
        pure_4bit_experts=tuple(pure_4bit_experts),
        token_embedding=token_embedding,
        lm_head=lm_head,
        validation_tokens=validation_tokens,
    )


@pytest.mark.smoke
def test_mixed_bpw_smoke(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(2, parity_fixture.hidden_dim, dtype=torch.float32)
    out = _moe_forward(
        x,
        parity_fixture.router,
        parity_fixture.mixed_experts,
        parity_fixture.top_k,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_individual_bit_width_accuracy(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(8, parity_fixture.hidden_dim, dtype=torch.float32)
    reference = parity_fixture.reference_experts[0]
    expected = _run_expert(reference, x)

    thresholds = {
        2: {"mse": 0.03, "max_abs": 0.80, "cosine": 0.70},
        3: {"mse": 0.006, "max_abs": 0.30, "cosine": 0.90},
        4: {"mse": 0.002, "max_abs": 0.15, "cosine": 0.96},
    }

    for bits in (2, 3, 4):
        quantized = _prepare_quantized_expert(reference, bits)
        actual = _run_expert(quantized, x)
        metrics = _error_metrics(actual, expected)
        limit = thresholds[bits]

        assert metrics["mse"] < limit["mse"], (
            f"{bits}-bit MSE too high: {metrics['mse']:.6f} >= {limit['mse']:.6f}"
        )
        assert metrics["max_abs"] < limit["max_abs"], (
            f"{bits}-bit max_abs too high: {metrics['max_abs']:.6f} >= {limit['max_abs']:.6f}"
        )
        assert metrics["cosine"] > limit["cosine"], (
            f"{bits}-bit cosine too low: {metrics['cosine']:.6f} <= {limit['cosine']:.6f}"
        )


def test_mixed_bpw_vs_fp16_reference(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(12, parity_fixture.hidden_dim, dtype=torch.float32)

    mixed = _moe_forward(
        x, parity_fixture.router, parity_fixture.mixed_experts, parity_fixture.top_k
    )
    reference = _moe_forward(
        x, parity_fixture.router, parity_fixture.reference_experts, parity_fixture.top_k
    )
    metrics = _error_metrics(mixed, reference)

    assert metrics["mse"] < 0.01, f"Mixed vs FP16 MSE regression: {metrics['mse']:.6f}"
    assert metrics["cosine"] > 0.85, f"Mixed vs FP16 cosine too low: {metrics['cosine']:.6f}"
    assert metrics["max_abs"] < 0.40, f"Mixed vs FP16 max_abs too high: {metrics['max_abs']:.6f}"


def test_mixed_bpw_vs_pure_4bit_parity(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(12, parity_fixture.hidden_dim, dtype=torch.float32)

    mixed = _moe_forward(
        x, parity_fixture.router, parity_fixture.mixed_experts, parity_fixture.top_k
    )
    pure_4bit = _moe_forward(
        x, parity_fixture.router, parity_fixture.pure_4bit_experts, parity_fixture.top_k
    )
    metrics = _error_metrics(mixed, pure_4bit)

    assert metrics["mse"] < 0.01, f"Mixed vs 4-bit MSE regression: {metrics['mse']:.6f}"
    assert metrics["cosine"] > 0.85, f"Mixed vs 4-bit cosine too low: {metrics['cosine']:.6f}"
    assert metrics["max_abs"] < 0.40, f"Mixed vs 4-bit max_abs too high: {metrics['max_abs']:.6f}"


def test_token_level_output_matching(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(1, parity_fixture.hidden_dim, dtype=torch.float32)

    mixed = _moe_forward(
        x, parity_fixture.router, parity_fixture.mixed_experts, parity_fixture.top_k
    )
    reference = _moe_forward(
        x, parity_fixture.router, parity_fixture.reference_experts, parity_fixture.top_k
    )

    torch.testing.assert_close(mixed, reference, rtol=0.25, atol=0.12)


def test_sequence_level_consistency(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(14, parity_fixture.hidden_dim, dtype=torch.float32)
    full = _moe_forward(
        x, parity_fixture.router, parity_fixture.mixed_experts, parity_fixture.top_k
    )
    chunked = torch.cat(
        [
            _moe_forward(
                x[:7],
                parity_fixture.router,
                parity_fixture.mixed_experts,
                parity_fixture.top_k,
            ),
            _moe_forward(
                x[7:],
                parity_fixture.router,
                parity_fixture.mixed_experts,
                parity_fixture.top_k,
            ),
        ],
        dim=0,
    )
    torch.testing.assert_close(full, chunked, rtol=1e-6, atol=1e-6)


def test_batch_processing_consistency(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(8, parity_fixture.hidden_dim, dtype=torch.float32)
    batch_out = _moe_forward(
        x, parity_fixture.router, parity_fixture.mixed_experts, parity_fixture.top_k
    )
    token_out = torch.cat(
        [
            _moe_forward(
                x[idx : idx + 1],
                parity_fixture.router,
                parity_fixture.mixed_experts,
                parity_fixture.top_k,
            )
            for idx in range(x.shape[0])
        ],
        dim=0,
    )
    torch.testing.assert_close(batch_out, token_out, rtol=1e-6, atol=1e-6)


def test_attention_like_computation_accuracy(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(10, parity_fixture.hidden_dim, dtype=torch.float32)

    hidden_mixed = _moe_forward(
        x, parity_fixture.router, parity_fixture.mixed_experts, parity_fixture.top_k
    )
    hidden_ref = _moe_forward(
        x, parity_fixture.router, parity_fixture.reference_experts, parity_fixture.top_k
    )

    mixed_scores = _attention_scores(hidden_mixed)
    ref_scores = _attention_scores(hidden_ref)
    diff = (mixed_scores - ref_scores).abs()

    assert float(diff.mean().item()) < 0.03, "Attention-score MAE regression"
    assert float(diff.max().item()) < 0.25, "Attention-score max error regression"


def test_moe_routing_consistency_mixed_bpw(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(16, parity_fixture.hidden_dim, dtype=torch.float32)

    _, idx_mixed, weights_mixed = _moe_forward(
        x,
        parity_fixture.router,
        parity_fixture.mixed_experts,
        parity_fixture.top_k,
        return_routing=True,
    )
    _, idx_4bit, weights_4bit = _moe_forward(
        x,
        parity_fixture.router,
        parity_fixture.pure_4bit_experts,
        parity_fixture.top_k,
        return_routing=True,
    )
    _, idx_ref, weights_ref = _moe_forward(
        x,
        parity_fixture.router,
        parity_fixture.reference_experts,
        parity_fixture.top_k,
        return_routing=True,
    )

    torch.testing.assert_close(idx_mixed, idx_4bit)
    torch.testing.assert_close(idx_mixed, idx_ref)
    torch.testing.assert_close(weights_mixed, weights_4bit, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weights_mixed, weights_ref, rtol=0.0, atol=0.0)


def test_mixed_bpw_perplexity_approximation(parity_fixture: _ParityFixture) -> None:
    ppl_ref = _compute_perplexity(
        parity_fixture.validation_tokens,
        parity_fixture.token_embedding,
        parity_fixture.lm_head,
        parity_fixture.router,
        parity_fixture.reference_experts,
        parity_fixture.top_k,
    )
    ppl_mixed = _compute_perplexity(
        parity_fixture.validation_tokens,
        parity_fixture.token_embedding,
        parity_fixture.lm_head,
        parity_fixture.router,
        parity_fixture.mixed_experts,
        parity_fixture.top_k,
    )
    ppl_4bit = _compute_perplexity(
        parity_fixture.validation_tokens,
        parity_fixture.token_embedding,
        parity_fixture.lm_head,
        parity_fixture.router,
        parity_fixture.pure_4bit_experts,
        parity_fixture.top_k,
    )

    delta_ref_pct = abs(ppl_mixed - ppl_ref) / max(ppl_ref, 1e-6) * 100.0
    delta_4bit_pct = abs(ppl_mixed - ppl_4bit) / max(ppl_4bit, 1e-6) * 100.0

    assert delta_ref_pct < 2.0, (
        f"Mixed BPW perplexity diverged from FP16 reference by {delta_ref_pct:.3f}%"
    )
    assert delta_4bit_pct < 1.0, (
        f"Mixed BPW perplexity diverged from pure 4-bit by {delta_4bit_pct:.3f}%"
    )


def test_mixed_bpw_regression_detection(parity_fixture: _ParityFixture) -> None:
    x = torch.randn(12, parity_fixture.hidden_dim, dtype=torch.float32)

    mixed = _moe_forward(
        x, parity_fixture.router, parity_fixture.mixed_experts, parity_fixture.top_k
    )
    ref = _moe_forward(
        x, parity_fixture.router, parity_fixture.reference_experts, parity_fixture.top_k
    )
    pure_4 = _moe_forward(
        x, parity_fixture.router, parity_fixture.pure_4bit_experts, parity_fixture.top_k
    )

    mixed_vs_ref = _error_metrics(mixed, ref)
    mixed_vs_4 = _error_metrics(mixed, pure_4)

    ppl_ref = _compute_perplexity(
        parity_fixture.validation_tokens,
        parity_fixture.token_embedding,
        parity_fixture.lm_head,
        parity_fixture.router,
        parity_fixture.reference_experts,
        parity_fixture.top_k,
    )
    ppl_mixed = _compute_perplexity(
        parity_fixture.validation_tokens,
        parity_fixture.token_embedding,
        parity_fixture.lm_head,
        parity_fixture.router,
        parity_fixture.mixed_experts,
        parity_fixture.top_k,
    )

    ppl_delta_pct = abs(ppl_mixed - ppl_ref) / max(ppl_ref, 1e-6) * 100.0

    assert mixed_vs_ref["mse"] < 0.01, "Regression guard: mixed-vs-ref MSE exceeded threshold"
    assert mixed_vs_ref["cosine"] > 0.85, "Regression guard: mixed-vs-ref cosine below threshold"
    assert mixed_vs_4["mse"] < 0.01, "Regression guard: mixed-vs-4bit MSE exceeded threshold"
    assert ppl_delta_pct < 2.0, "Regression guard: mixed-vs-ref perplexity delta exceeded threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
