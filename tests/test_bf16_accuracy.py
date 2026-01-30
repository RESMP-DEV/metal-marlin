"""BF16 accuracy validation tests for Metal Marlin.

Validates:
  1) BF16 conversion accuracy (RTZ vs RNE, round-trip error, edge cases)
  2) GEMM accuracy (old accum vs FP32 accum)
  3) Attention softmax accuracy (precision-sensitive, long sequence stress)
  4) End-to-end perplexity on a calibration set (old vs new path)
  5) Gradient checks when training support exists (PyTorch)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from metal_marlin._compat import HAS_TORCH, torch
from metal_marlin.eval import compute_perplexity_from_logits, load_wikitext2

from .conftest import requires_torch


def _float32_to_bits(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32).view(np.uint32)


def _bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint32).view(np.float32)


def bf16_rtz_bits(values: np.ndarray) -> np.ndarray:
    """Convert fp32 -> bf16 bits using round-toward-zero (truncation)."""
    bits = _float32_to_bits(values)
    return (bits >> np.uint32(16)).astype(np.uint16)


def bf16_rne_bits(values: np.ndarray) -> np.ndarray:
    """Convert fp32 -> bf16 bits using round-to-nearest-even."""
    bits = _float32_to_bits(values)
    upper = bits >> np.uint32(16)
    lower = bits & np.uint32(0xFFFF)

    exp_all_ones = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    round_bit = (lower & np.uint32(0x8000)) != 0
    sticky = (lower & np.uint32(0x7FFF)) != 0
    lsb = (upper & np.uint32(1)) != 0

    increment = round_bit & (sticky | lsb)
    rounded = upper + increment.astype(np.uint32)

    return np.where(exp_all_ones, upper, rounded).astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return _bits_to_float32(bits.astype(np.uint32) << np.uint32(16))


def bf16_rtz(values: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(bf16_rtz_bits(values))


def bf16_rne(values: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(bf16_rne_bits(values))


def _ordered_int_bf16(bits: np.ndarray) -> np.ndarray:
    signed = bits.astype(np.int16).astype(np.int32)
    ordered = signed.copy()
    neg_mask = signed < 0
    ordered[neg_mask] = 0x8000 - ordered[neg_mask]
    return ordered


def ulp_distance_bf16(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    ordered_a = _ordered_int_bf16(a_bits)
    ordered_b = _ordered_int_bf16(b_bits)
    return np.abs(ordered_a - ordered_b).astype(np.int32)


def _ordered_int_fp32(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.int32)
    ordered = bits.copy()
    neg_mask = bits < 0
    ordered[neg_mask] = np.int32(-2147483648) - ordered[neg_mask]
    return ordered


def ulp_distance_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ordered_a = _ordered_int_fp32(a)
    ordered_b = _ordered_int_fp32(b)
    return np.abs(ordered_a - ordered_b).astype(np.int64)


def gemm_accumulate(
    a: np.ndarray, b: np.ndarray, *, acc_dtype: np.dtype, tile_k: int = 32
) -> np.ndarray:
    """GEMM with explicit accumulation dtype to simulate old/new paths."""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2

    if acc_dtype == np.float16:
        out = np.zeros((m, n), dtype=np.float16)
    else:
        out = np.zeros((m, n), dtype=np.float32)

    for k0 in range(0, k, tile_k):
        k1 = min(k0 + tile_k, k)
        prod = a[:, k0:k1].astype(np.float32) @ b[k0:k1, :].astype(np.float32)
        if acc_dtype == np.float16:
            out = (out.astype(np.float32) + prod).astype(np.float16)
        else:
            out = out + prod.astype(np.float32)

    return out.astype(np.float32)


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_output(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    use_fp32_acc: bool,
    is_causal: bool = False,
) -> np.ndarray:
    batch, heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape
    scale = 1.0 / np.sqrt(head_dim)

    scores = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float32), k.astype(np.float32))
    scores = scores * scale

    if is_causal:
        q_idx = np.arange(seq_q)[:, None]
        k_idx = np.arange(seq_k)[None, :]
        mask = np.where(k_idx > q_idx, -np.inf, 0.0).astype(np.float32)
        scores = scores + mask

    if not use_fp32_acc:
        scores = scores.astype(np.float16).astype(np.float32)

    weights = stable_softmax(scores, axis=-1)
    output = np.einsum("bhqk,bhkd->bhqd", weights, v.astype(np.float32))
    return output.astype(np.float32)


def compute_error_metrics(ref: np.ndarray, out: np.ndarray) -> dict[str, float]:
    diff = out.astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff**2))
    ulp = float(np.max(ulp_distance_fp32(ref.astype(np.float32), out.astype(np.float32))))
    return {"max_abs": max_abs, "mse": mse, "ulp": ulp}


class TestBF16Conversion:
    def test_roundtrip_and_rtz_vs_rne(self, rng: np.random.Generator) -> None:
        values = rng.standard_normal(20000).astype(np.float32) * np.float32(10.0)

        rtz_bits = bf16_rtz_bits(values)
        rne_bits = bf16_rne_bits(values)

        rtz = bf16_bits_to_float32(rtz_bits)
        rne = bf16_bits_to_float32(rne_bits)

        max_err_rtz = float(np.max(np.abs(rtz - values)))
        max_err_rne = float(np.max(np.abs(rne - values)))

        mse_rtz = float(np.mean((rtz - values) ** 2))
        mse_rne = float(np.mean((rne - values) ** 2))

        ulp = int(np.max(ulp_distance_bf16(rtz_bits, rne_bits)))

        print(
            f"BF16 round-trip max_err_rtz={max_err_rtz:.6e} max_err_rne={max_err_rne:.6e} "
            f"mse_rtz={mse_rtz:.6e} mse_rne={mse_rne:.6e} max_ulp={ulp}"
        )

        assert ulp < 2, f"Max ULP error too large: {ulp}"
        assert mse_rne <= mse_rtz * 1.01, (
            f"RNE should be at least as good as RTZ (mse_rne={mse_rne}, mse_rtz={mse_rtz})"
        )
        assert max_err_rne <= max_err_rtz * 1.01, (
            f"RNE max error should be <= RTZ max error ({max_err_rne} vs {max_err_rtz})"
        )

    def test_edge_cases(self) -> None:
        smallest = np.nextafter(np.float32(0.0), np.float32(1.0))
        values = np.array(
            [
                np.float32(0.0),
                np.float32(-0.0),
                np.float32(np.inf),
                np.float32(-np.inf),
                np.float32(np.nan),
                np.float32(1e-40),
                np.float32(1e-45),
                smallest,
            ],
            dtype=np.float32,
        )

        rtz = bf16_rtz(values)
        rne = bf16_rne(values)

        assert np.isposinf(rtz[2]) and np.isposinf(rne[2])
        assert np.isneginf(rtz[3]) and np.isneginf(rne[3])
        assert np.isnan(rtz[4]) and np.isnan(rne[4])

        finite_mask = np.isfinite(values)
        assert np.all(np.isfinite(rtz[finite_mask]))
        assert np.all(np.isfinite(rne[finite_mask]))


class TestBF16GEMMAccuracy:
    @pytest.mark.parametrize("k_dim", [128, 512, 1024])
    def test_fp32_accumulator_improves_gemm(self, rng: np.random.Generator, k_dim: int) -> None:
        m, n = 16, 32
        a = rng.standard_normal((m, k_dim)).astype(np.float32)
        b = rng.standard_normal((k_dim, n)).astype(np.float32)

        a_bf16 = bf16_rtz(a)
        b_bf16 = bf16_rtz(b)

        ref = a_bf16.astype(np.float64) @ b_bf16.astype(np.float64)
        old_out = gemm_accumulate(a_bf16, b_bf16, acc_dtype=np.float16)
        new_out = gemm_accumulate(a_bf16, b_bf16, acc_dtype=np.float32)

        old_metrics = compute_error_metrics(ref, old_out)
        new_metrics = compute_error_metrics(ref, new_out)

        print(
            f"GEMM k={k_dim} old max_abs={old_metrics['max_abs']:.6e} "
            f"mse={old_metrics['mse']:.6e} ulp={old_metrics['ulp']:.0f} | "
            f"new max_abs={new_metrics['max_abs']:.6e} "
            f"mse={new_metrics['mse']:.6e} ulp={new_metrics['ulp']:.0f}"
        )

        assert np.all(np.isfinite(old_out))
        assert np.all(np.isfinite(new_out))

        assert new_metrics["max_abs"] <= old_metrics["max_abs"] * 1.01
        assert new_metrics["mse"] <= old_metrics["mse"] * 1.01


class TestBF16AttentionAccuracy:
    def test_softmax_accuracy(self, rng: np.random.Generator) -> None:
        batch, heads, seq, head_dim = 1, 2, 64, 32

        q = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
        k = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
        v = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)

        q_bf16 = bf16_rtz(q)
        k_bf16 = bf16_rtz(k)
        v_bf16 = bf16_rtz(v)

        ref = attention_output(q, k, v, use_fp32_acc=True)
        old_out = attention_output(q_bf16, k_bf16, v_bf16, use_fp32_acc=False)
        new_out = attention_output(q_bf16, k_bf16, v_bf16, use_fp32_acc=True)

        old_metrics = compute_error_metrics(ref, old_out)
        new_metrics = compute_error_metrics(ref, new_out)

        assert np.all(np.isfinite(old_out))
        assert np.all(np.isfinite(new_out))
        assert new_metrics["max_abs"] <= old_metrics["max_abs"] * 1.05

    @pytest.mark.slow
    def test_softmax_long_sequence_stability(self, rng: np.random.Generator) -> None:
        batch, heads, seq, head_dim = 1, 2, 512, 32

        q = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
        k = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
        v = rng.standard_normal((batch, heads, seq, head_dim)).astype(np.float32)

        q_bf16 = bf16_rtz(q)
        k_bf16 = bf16_rtz(k)
        v_bf16 = bf16_rtz(v)

        out = attention_output(q_bf16, k_bf16, v_bf16, use_fp32_acc=True, is_causal=True)

        assert np.all(np.isfinite(out)), "NaN/Inf in attention output"


@dataclass
class ToyBf16LM:
    vocab_size: int
    hidden_size: int
    embed: np.ndarray
    proj: np.ndarray

    @classmethod
    def create(cls, vocab_size: int = 256, hidden_size: int = 64, seed: int = 123) -> ToyBf16LM:
        rng = np.random.default_rng(seed)
        embed = rng.standard_normal((vocab_size, hidden_size)).astype(np.float32) * np.float32(0.02)
        proj = rng.standard_normal((hidden_size, vocab_size)).astype(np.float32) * np.float32(0.02)
        return cls(vocab_size=vocab_size, hidden_size=hidden_size, embed=embed, proj=proj)

    def logits(self, input_ids: np.ndarray, *, acc_dtype: np.dtype) -> np.ndarray:
        seq_len = input_ids.shape[1]
        tokens = input_ids.reshape(-1)
        embed = bf16_rtz(self.embed)[tokens].reshape(seq_len, self.hidden_size)
        proj = bf16_rtz(self.proj)
        logits = gemm_accumulate(embed, proj, acc_dtype=acc_dtype)
        return logits.reshape(1, seq_len, self.vocab_size)


def _tokenize_bytes(text: str, vocab_size: int) -> list[int]:
    return [b % vocab_size for b in text.encode("utf-8", errors="ignore")]


@dataclass
class ByteTokenizer:
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        return _tokenize_bytes(text, self.vocab_size)


class TestBF16Perplexity:
    @pytest.mark.slow
    def test_perplexity_old_vs_new(self) -> None:
        try:
            texts = load_wikitext2(max_samples=10)
        except Exception as exc:
            pytest.skip(f"Could not load wikitext2: {exc}")

        model = ToyBf16LM.create()
        vocab_size = model.vocab_size

        def logits_old(input_ids: np.ndarray) -> np.ndarray:
            return model.logits(input_ids, acc_dtype=np.float16)

        def logits_new(input_ids: np.ndarray) -> np.ndarray:
            return model.logits(input_ids, acc_dtype=np.float32)

        tokenizer = ByteTokenizer(vocab_size=vocab_size)

        ppl_old = compute_perplexity_from_logits(
            logits_old, tokenizer, texts, max_length=128, verbose=False
        )
        ppl_new = compute_perplexity_from_logits(
            logits_new, tokenizer, texts, max_length=128, verbose=False
        )

        assert np.isfinite(ppl_old)
        assert np.isfinite(ppl_new)
        assert abs(ppl_new - ppl_old) < 0.01, (
            f"Perplexity delta too large: old={ppl_old:.6f}, new={ppl_new:.6f}"
        )


class TestBF16Gradients:
    @requires_torch
    def test_gradient_matches_fp32_reference(self) -> None:
        if not HAS_TORCH or torch is None:
            pytest.skip("torch not available")

        torch.manual_seed(123)
        x = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
        w = torch.randn(16, 12, dtype=torch.float32, requires_grad=True)

        x_bf16 = x.detach().to(torch.bfloat16).requires_grad_(True)
        w_bf16 = w.detach().to(torch.bfloat16).requires_grad_(True)

        ref = (x @ w).sum()
        grad_x_ref, grad_w_ref = torch.autograd.grad(ref, (x, w))

        old = (x_bf16 @ w_bf16).float().sum()
        grad_x_old, grad_w_old = torch.autograd.grad(old, (x_bf16, w_bf16))

        new = (x_bf16.float() @ w_bf16.float()).sum()
        grad_x_new, grad_w_new = torch.autograd.grad(new, (x_bf16, w_bf16))

        diff_old = (grad_x_old.float() - grad_x_ref).abs().mean().item()
        diff_new = (grad_x_new.float() - grad_x_ref).abs().mean().item()

        assert np.isfinite(diff_old)
        assert np.isfinite(diff_new)
        assert diff_new <= diff_old * 1.05
