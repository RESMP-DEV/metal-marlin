# Qwen3.5 / Qwen3.6 / Qwen3-Coder-Next DeltaNet Inventory

**Date:** 2026-04-21  
**Auditor:** automated inventory from local HF cache snapshots  
**Snapshots inspected:**

| Model | Snapshot SHA |
|---|---|
| Qwen3.5-35B-A3B | `ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307` |
| Qwen3.6-35B-A3B | `7da1103448ba36029c34ce1a9a741dfe93ee0c50` |
| Qwen3-Coder-Next | `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` |

---

## 1. Architecture & `model_type` Comparison

| Field | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Qwen3-Coder-Next |
|---|---|---|---|
| `architectures[0]` | `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` | `Qwen3NextForCausalLM` |
| `model_type` (top) | `qwen3_5_moe` | `qwen3_5_moe` | `qwen3_next` |
| `text_config.model_type` | `qwen3_5_moe_text` | `qwen3_5_moe_text` | *(no `text_config`)* |
| Multimodal | Yes (vision_config) | Yes (vision_config) | No (text-only causal LM) |
| `transformers_version` | `4.57.0.dev0` | `4.57.1` | `4.57.0.dev0` |

**Key insight:** Qwen3.5 and Qwen3.6 share the same `Qwen3_5MoeForConditionalGeneration` architecture class. Qwen3-Coder-Next is a **completely different architecture** (`Qwen3NextForCausalLM`, model_type `qwen3_next`) with no `text_config` nesting and no vision encoder.

---

## 2. `text_config` vs Top-Level Config Layout

### Qwen3.5 & Qwen3.6 (nested layout)

All text-model fields are nested under `text_config`:

```json
{
  "model_type": "qwen3_5_moe",
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_experts": 256,
    "full_attention_interval": 4,
    "layer_types": [...],
    "linear_key_head_dim": 128,
    ...
  },
  "vision_config": { ... }
}
```

### Qwen3-Coder-Next (flat layout)

All fields sit at the top level—no `text_config`, no `vision_config`:

```json
{
  "model_type": "qwen3_next",
  "hidden_size": 2048,
  "num_hidden_layers": 48,
  "num_experts": 512,
  "full_attention_interval": 4,
  "linear_key_head_dim": 128,
  ...
}
```

**Implication:** Any config reader must handle both layouts. When `text_config` exists, merge or delegate into it; when absent, read top-level directly.

---

## 3. `layer_types` / `full_attention_interval`

| Field | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Qwen3-Coder-Next |
|---|---|---|---|
| `full_attention_interval` | 4 | 4 | 4 |
| `layer_types` (explicit array) | ✅ 40 entries | ✅ 40 entries | ❌ absent |
| Pattern | 3×`linear_attention` + 1×`full_attention` repeating | Same | Inferred from `full_attention_interval` |

Qwen3.5/3.6 ship an explicit `layer_types` array of length 40:
```
[linear, linear, linear, full,   # layers 0-3
 linear, linear, linear, full,   # layers 4-7
 ... ]                            # ×10 blocks
```
Full-attention layers: **3, 7, 11, 15, 19, 23, 27, 31, 35, 39** (i.e. `i % 4 == 3`).

Qwen3-Coder-Next has **no** `layer_types` field. With 48 layers and `full_attention_interval=4`, the same pattern applies but must be computed at runtime: `layer_type = "full_attention" if i % full_attention_interval == full_attention_interval - 1 else "linear_attention"`.

Additionally, Qwen3-Coder-Next has `decoder_sparse_step=1`, meaning every layer is MoE (no dense-MLP layers). There is no `mlp_only_layers` list (it is `[]` in Qwen3.5/3.6 and absent in Coder-Next, but functionally empty).

---

## 4. DeltaNet Fields

| Field | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Qwen3-Coder-Next |
|---|---|---|---|
| `linear_key_head_dim` | 128 | 128 | 128 |
| `linear_value_head_dim` | 128 | 128 | 128 |
| `linear_num_key_heads` | 16 | 16 | 16 |
| `linear_num_value_heads` | 32 | 32 | 32 |
| `linear_conv_kernel_dim` | 4 | 4 | 4 |
| `mamba_ssm_dtype` | `float32` | `float32` | *(absent)* |

All three models share identical DeltaNet dimension values. Qwen3.5/3.6 place `mamba_ssm_dtype` inside `text_config`; Qwen3-Coder-Next omits it (default `float32` assumed).

---

## 5. MoE Fields

| Field | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Qwen3-Coder-Next |
|---|---|---|---|
| `num_experts` | 256 | 256 | **512** |
| `num_experts_per_tok` | 8 | 8 | **10** |
| `moe_intermediate_size` | 512 | 512 | 512 |
| `shared_expert_intermediate_size` | 512 | 512 | 512 |
| `router_aux_loss_coef` | 0.001 | 0.001 | 0.001 |
| `decoder_sparse_step` | *(absent)* | *(absent)* | 1 |
| `norm_topk_prob` | *(absent)* | *(absent)* | true |

Qwen3-Coder-Next doubles the expert count (512 vs 256) and increases top-k (10 vs 8), making its expert weight footprint substantially larger. The `decoder_sparse_step=1` field confirms all 48 layers are MoE. `norm_topk_prob=true` means router probabilities are normalized before dispatch—this affects the gating numerics and must be honoured by any custom MoE kernel.

---

## 6. Other Divergent Config Fields

| Field | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Qwen3-Coder-Next |
|---|---|---|---|
| `num_hidden_layers` | 40 | 40 | **48** |
| `vocab_size` | 248320 | 248320 | **151936** |
| `rope_theta` | 10 000 000 | 10 000 000 | **5 000 000** |
| `bos_token_id` | *(absent)* | 248044 | 151643 |
| `eos_token_id` | 248044 | 248044 | 151645 |
| `intermediate_size` | *(absent)* | *(absent)* | 5120 |
| `use_sliding_window` | *(absent)* | *(absent)* | false |
| `mtp_num_hidden_layers` | 1 | 1 | *(absent)* |
| `mtp_use_dedicated_embeddings` | false | false | *(absent)* |
| `attn_output_gate` | true | true | *(absent)* |
| `partial_rotary_factor` | 0.25 | 0.25 | 0.25 |
| `rope_parameters.mrope_interleaved` | true | true | *(absent)* |
| `rope_parameters.mrope_section` | [11,11,10] | [11,11,10] | *(absent)* |

Qwen3.5/3.6 use mRoPE (multi-dimensional RoPE with interleaved layout and a `[11,11,10]` section split for temporal/spatial dimensions—needed for vision). Qwen3-Coder-Next is text-only and uses standard 1D RoPE with a shorter theta.

---

## 7. Checkpoint Tensor Prefixes

### Qwen3.5 & Qwen3.6

```
model.language_model.embed_tokens.weight
model.language_model.layers.{N}.{submodule}.{param}
model.language_model.norm.weight
model.visual.blocks.{B}.{submodule}.{param}
model.visual.merger.{param}
model.visual.patch_embed.proj.{param}
model.visual.pos_embed.weight
```

The extra `language_model` nesting reflects the `ForConditionalGeneration` wrapper: `model` → `language_model` → `layers`.

### Qwen3-Coder-Next

```
model.embed_tokens.weight
model.layers.{N}.{submodule}.{param}
model.norm.weight
```

No `language_model` intermediate. No `visual.*` tensors at all.

**Implication:** A weight loader must strip or add the `language_model.` prefix depending on model_type. Currently `hf_loader.py`'s `_extract_layer_index` regex `r"layers\.(\d+)\."` works for both because it matches within either prefix—but downstream code that constructs tensor names (e.g. for quantized output) must be aware of the full prefix.

---

## 8. DeltaNet Tensor Names

This is the most critical divergence for kernel integration.

### Qwen3.5 & Qwen3.6 — 4-tensor layout

```
model.language_model.layers.N.linear_attn.in_proj_qkv.weight   # QKV projection
model.language_model.layers.N.linear_attn.in_proj_z.weight     # Z gate
model.language_model.layers.N.linear_attn.in_proj_a.weight     # A (decay) projection
model.language_model.layers.N.linear_attn.in_proj_b.weight     # B (recurrence) projection
```

Four separate weight tensors per linear-attention layer. `in_proj_qkv` fuses Q, K, V; `in_proj_z` is the output gate; `in_proj_a` and `in_proj_b` are the DeltaNet recurrence projections.

Additional per-layer DeltaNet tensors (shared across all three models):
```
.linear_attn.A_log           # log-diagonal decay matrix
.linear_attn.dt_bias         # timestep bias
.linear_attn.conv1d.weight   # causal conv1d
.linear_attn.norm.weight     # RMSNorm inside linear attention
.linear_attn.out_proj.weight # output projection
```

### Qwen3-Coder-Next — 2-tensor fused layout

```
model.layers.N.linear_attn.in_proj_qkvz.weight   # QKV + Z gate fused
model.layers.N.linear_attn.in_proj_ba.weight      # B + A fused
```

Only **two** weight tensors per linear-attention layer. `in_proj_qkvz` fuses Q, K, V, and Z into a single matmul; `in_proj_ba` fuses B and A. The remaining auxiliary tensors (`A_log`, `dt_bias`, `conv1d.weight`, `norm.weight`, `out_proj.weight`) are identical in naming.

### Summary Table

| Tensor | Qwen3.5 / Qwen3.6 | Qwen3-Coder-Next |
|---|---|---|
| QKV projection | `in_proj_qkv.weight` | `in_proj_qkvz.weight` (includes Z) |
| Z gate | `in_proj_z.weight` | *(fused into `in_proj_qkvz`)* |
| A projection | `in_proj_a.weight` | *(fused into `in_proj_ba`)* |
| B projection | `in_proj_b.weight` | *(fused into `in_proj_ba`)* |
| A log-diagonal | `A_log` | `A_log` (same) |
| dt bias | `dt_bias` | `dt_bias` (same) |
| Causal conv1d | `conv1d.weight` | `conv1d.weight` (same) |
| Internal norm | `norm.weight` | `norm.weight` (same) |
| Output projection | `out_proj.weight` | `out_proj.weight` (same) |

**Implication:** Any linear-attention kernel or weight-quantization path must branch on model_type to know whether it receives 4 separate or 2 fused projection tensors. The column split logic is also different: `in_proj_qkvz` has 4 logical heads per head instead of 3 (QKV) + 1 (Z separate).

---

## 9. Expert Tensor Naming

### Qwen3.5 & Qwen3.6 — fused `gate_up_proj`, no `.weight` suffix on experts

```
model.language_model.layers.N.mlp.experts.Edown_proj
model.language_model.layers.N.mlp.experts.Egate_up_proj
model.language_model.layers.N.mlp.gate.weight
model.language_model.layers.N.mlp.shared_expert_gate.weight
model.language_model.layers.N.mlp.shared_expert.gate_proj.weight
model.language_model.layers.N.mlp.shared_expert.up_proj.weight
model.language_model.layers.N.mlp.shared_expert.down_proj.weight
```

Notable:
- Expert projections use **fused** `gate_up_proj` (gate + up combined).
- Expert `down_proj` and `gate_up_proj` **omit** the `.weight` suffix in the index key (they are stored as tensors without the suffix in the weight_map, likely a serialization artifact—actual safetensors metadata may differ).
- `E` is a numeric expert index (0–255).

### Qwen3-Coder-Next — separate `gate_proj`/`up_proj`, always `.weight`

```
model.layers.N.mlp.experts.E.gate_proj.weight
model.layers.N.mlp.experts.E.up_proj.weight
model.layers.N.mlp.experts.E.down_proj.weight
model.layers.N.mlp.gate.weight
model.layers.N.mlp.shared_expert_gate.weight
model.layers.N.mlp.shared_expert.gate_proj.weight
model.layers.N.mlp.shared_expert.up_proj.weight
model.layers.N.mlp.shared_expert.down_proj.weight
```

Notable:
- Expert projections are **separate** `gate_proj` and `up_proj`.
- All tensors carry `.weight` suffix consistently.
- `E` ranges 0–511 (512 experts).

### Comparison Table

| Tensor | Qwen3.5 / Qwen3.6 | Qwen3-Coder-Next |
|---|---|---|
| Expert gate | fused `experts.Egate_up_proj` | separate `experts.E.gate_proj.weight` |
| Expert up | *(fused into gate_up_proj)* | `experts.E.up_proj.weight` |
| Expert down | `experts.Edown_proj` | `experts.E.down_proj.weight` |
| Router gate | `mlp.gate.weight` | `mlp.gate.weight` (same) |
| `.weight` suffix on experts | missing (index artifact) | present |

**Implication:** Weight-splitting logic for quantization must handle the fused `gate_up_proj` for Qwen3.5/3.6 vs. separate `gate_proj`/`up_proj` for Coder-Next. The missing `.weight` suffix on Qwen3.5/3.6 expert tensors is a naming irregularity that can cause key-miss errors if not handled.

---

## 10. `shared_expert` vs `shared_experts` Naming Mismatch

All three models consistently use **`shared_expert`** (singular) in their tensor names:

```
mlp.shared_expert.gate_proj.weight
mlp.shared_expert.up_proj.weight
mlp.shared_expert.down_proj.weight
mlp.shared_expert_gate.weight
```

However, the HuggingFace Transformers model code for some Qwen variants (notably Qwen2.5-MoE and early Qwen3 checkpoints) uses the **plural** `shared_experts` in its Python `nn.Module` attribute names:

```python
# In some HF model code:
self.shared_experts = Qwen2_5MoeMLP(...)   # plural 's'
# vs. checkpoint key:
model.layers.N.mlp.shared_expert.gate_proj.weight  # singular
```

This mismatch means:
1. **State dict loading:** HF's `_load_pretrained` must apply a key-remapping from `shared_expert` → `shared_experts` (or vice versa) when loading these checkpoints.
2. **Custom loaders:** Our `hf_loader.py` must not assume the module attribute name matches the checkpoint key. When constructing tensor names for quantized output or for a re-serialized checkpoint, use the **checkpoint convention** (`shared_expert`, singular).
3. **Future models:** Any new Qwen release may standardize one way or the other; always verify against the actual `model.safetensors.index.json`.

---

## 11. Full-Attention Layer Tensors (all three models, same naming)

```
{prefix}.layers.N.self_attn.q_proj.weight
{prefix}.layers.N.self_attn.k_proj.weight
{prefix}.layers.N.self_attn.v_proj.weight
{prefix}.layers.N.self_attn.o_proj.weight
{prefix}.layers.N.self_attn.q_norm.weight
{prefix}.layers.N.self_attn.k_norm.weight
```

These are standard GQA self-attention tensors present on full-attention layers only (indices where `i % 4 == 3`). Linear-attention layers have `linear_attn.*` instead of `self_attn.*`. No QKNorm is present on linear-attention layers.

---

## 12. Concrete Implementation Implications

### `metal_marlin/hf_loader.py`

- **Prefix normalization:** `_extract_layer_index` regex works for both prefixes, but code that *constructs* tensor names (e.g., for quantized output naming) must be aware of `model.language_model.` vs `model.`.
- **Expert key normalization:** Add logic to handle the missing `.weight` suffix on Qwen3.5/3.6 expert tensors. When doing `weight_map` lookups, try both `name.weight` and `name` variants.
- **Fused gate_up_proj handling:** When loading individual expert sub-tensors, detect `gate_up_proj` and split into `gate_proj` + `up_proj` if needed by downstream kernels.
- **`shared_expert` vs `shared_experts`:** Never assume the HF module attribute name; always use the checkpoint key convention (`shared_expert`, singular).

### `metal_marlin/trellis/config.py`

- **Flat vs nested config:** `_from_dict` must handle both layouts. When `text_config` is present, recurse into it; when absent, read top-level directly.
- **DeltaNet fields:** Add fields for `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`, `linear_num_value_heads`, `linear_conv_kernel_dim`, and `full_attention_interval`. Currently absent from `TrellisModelConfig`.
- **`layer_types` computation:** When `layer_types` is absent (Coder-Next), compute it from `full_attention_interval` and `num_hidden_layers`.
- **New model_type:** Add `qwen3_next` to the supported model types alongside `qwen3_5_moe`.
- **`norm_topk_prob`:** Add this field; it affects MoE dispatch numerics.
- **`decoder_sparse_step`:** Add this field; determines whether all layers are MoE or only a subset.
- **`intermediate_size`:** Coder-Next exposes this at top level (5120) — this is the dense-MLP intermediate size, different from `moe_intermediate_size` (512).

### `metal_marlin/mmfp4_loader.py`

- **DeltaNet weight packing:** Must branch on model_type to handle 4-tensor (Qwen3.5/3.6: `in_proj_qkv` + `in_proj_z` + `in_proj_a` + `in_proj_b`) vs. 2-tensor (Coder-Next: `in_proj_qkvz` + `in_proj_ba`) layouts. FP4 quantization of these projections requires knowing the column partition for each logical sub-projection.
- **Expert weight splitting:** For Qwen3.5/3.6, split `gate_up_proj` into separate gate and up projections before quantization. For Coder-Next, they are already separate.

### `metal_marlin/_quantized_weights.py`

- **Quantized key naming convention:** Decide whether quantized output uses checkpoint-style keys (singular `shared_expert`, no `.weight` on experts for Qwen3.5/3.6) or normalized keys. Recommend normalizing to always include `.weight` for consistency.
- **DeltaNet projection quantization:** The `in_proj_qkv` / `in_proj_qkvz` tensors have large column dimensions (fusing multiple heads). Quantization group boundaries must not cross sub-projection boundaries.

### `metal_marlin/serving/engine.py`

- **Layer dispatch:** For Qwen3-Coder-Next, every layer is both MoE and DeltaNet/full-attention (no dense-MLP layers). The engine must support dispatching to the correct kernel per layer based on the computed `layer_types`.
- **Expert count scaling:** Coder-Next has 512 experts × 48 layers = 24 576 expert blocks (vs. 256 × 40 = 10 240 for Qwen3.5/3.6). Memory planning must account for this ~2.4× increase in expert parameter count.
- **Vocabulary difference:** Tokenizer setup for Coder-Next uses `vocab_size=151936` vs. `248320` for Qwen3.5/3.6. Embedding and LM-head tensors have different shapes.

### `scripts/quantize_qwen35_35b_a3b_mmfp4.py`

- **Current state:** Assumes `model.language_model.layers.*` prefix, 4-tensor DeltaNet layout, and fused `gate_up_proj` expert format. This is correct for Qwen3.5/3.6.
- **Fix needed:** The missing `.weight` suffix on expert keys may cause output-tensor naming issues. Verify that quantized output keys match what `hf_loader.py` expects on re-load.

### Future `scripts/quantize_qwen36_35b_a3b_mmfp4.py`

- **Can largely clone `quantize_qwen35_35b_a3b_mmfp4.py`** since Qwen3.6 shares the same architecture class, checkpoint prefix, and tensor layout.
- **Minor differences to port:** Qwen3.6 config has `bos_token_id` explicitly set and a slightly different `transformers_version`. These are irrelevant for quantization.
- **No structural tensor changes between 3.5 and 3.6.**

### Future `scripts/quantize_qwen3_coder_next_mmfp4.py`

- **Cannot be a simple clone.** Must handle:
  1. Flat prefix (`model.layers.*` not `model.language_model.layers.*`).
  2. 2-tensor fused DeltaNet layout (`in_proj_qkvz` + `in_proj_ba`).
  3. Separate expert `gate_proj`/`up_proj` (not fused `gate_up_proj`).
  4. 512 experts per layer (2× the count, different shard layout).
  5. 48 layers instead of 40.
  6. No vision encoder tensors to skip/ignore.
  7. `norm_topk_prob=true` affects router weight handling.
  8. Different `vocab_size` (151936) and `rope_theta` (5M).

---

## Appendix A: Complete Tensor Name Patterns (normalized)

### Qwen3.5 / Qwen3.6 (text model only)

```
model.language_model.embed_tokens.weight
model.language_model.layers.N.input_layernorm.weight
model.language_model.layers.N.linear_attn.A_log
model.language_model.layers.N.linear_attn.conv1d.weight
model.language_model.layers.N.linear_attn.dt_bias
model.language_model.layers.N.linear_attn.in_proj_a.weight
model.language_model.layers.N.linear_attn.in_proj_b.weight
model.language_model.layers.N.linear_attn.in_proj_qkv.weight
model.language_model.layers.N.linear_attn.in_proj_z.weight
model.language_model.layers.N.linear_attn.norm.weight
model.language_model.layers.N.linear_attn.out_proj.weight
model.language_model.layers.N.self_attn.q_proj.weight
model.language_model.layers.N.self_attn.k_proj.weight
model.language_model.layers.N.self_attn.v_proj.weight
model.language_model.layers.N.self_attn.o_proj.weight
model.language_model.layers.N.self_attn.q_norm.weight
model.language_model.layers.N.self_attn.k_norm.weight
model.language_model.layers.N.mlp.experts.Edown_proj
model.language_model.layers.N.mlp.experts.Egate_up_proj
model.language_model.layers.N.mlp.gate.weight
model.language_model.layers.N.mlp.shared_expert_gate.weight
model.language_model.layers.N.mlp.shared_expert.gate_proj.weight
model.language_model.layers.N.mlp.shared_expert.up_proj.weight
model.language_model.layers.N.mlp.shared_expert.down_proj.weight
model.language_model.layers.N.post_attention_layernorm.weight
model.language_model.norm.weight
```

### Qwen3-Coder-Next

```
model.embed_tokens.weight
model.layers.N.input_layernorm.weight
model.layers.N.linear_attn.A_log
model.layers.N.linear_attn.conv1d.weight
model.layers.N.linear_attn.dt_bias
model.layers.N.linear_attn.in_proj_ba.weight
model.layers.N.linear_attn.in_proj_qkvz.weight
model.layers.N.linear_attn.norm.weight
model.layers.N.linear_attn.out_proj.weight
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.self_attn.q_norm.weight
model.layers.N.self_attn.k_norm.weight
model.layers.N.mlp.experts.E.gate_proj.weight
model.layers.N.mlp.experts.E.up_proj.weight
model.layers.N.mlp.experts.E.down_proj.weight
model.layers.N.mlp.gate.weight
model.layers.N.mlp.shared_expert_gate.weight
model.layers.N.mlp.shared_expert.gate_proj.weight
model.layers.N.mlp.shared_expert.up_proj.weight
model.layers.N.mlp.shared_expert.down_proj.weight
model.layers.N.post_attention_layernorm.weight
model.norm.weight
```
