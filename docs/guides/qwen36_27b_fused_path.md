# Qwen3.6-27B Fused Metal Path

Qwen3.6-27B is integrated as a dense hybrid DeltaNet/full-attention profile,
not as the standalone `qwenmetal` runtime.  The tracked profile lives in
`metal_marlin/qwen36_27b_profile.py`; ignored workspace evidence can be
refreshed with:

```bash
uv run python scripts/refresh_qwen36_27b_shape_contract.py
```

The fused runtime remains opt-in:

```bash
METAL_MARLIN_QWEN36_27B_MEGAKERNEL=1 uv run pytest tests/test_qwen36_27b_fused_path.py -q
```

## Artifact Layout

The fused path accepts the qwenmetal int4 layout only through
`metal_marlin.qwen36_27b_artifact`:

- `qweight[(K / 8), N]` as packed unsigned 4-bit values in `uint32` lanes.
- `scales[ceil(K / group_size), N]` as FP16.
- `zeros[ceil(K / group_size), N]` as FP16.
- default `group_size = 128`.

Each tensor records role, layer index, source checkpoint, calibration metadata,
and expected `(in_features, out_features)` so the dense 27B path cannot silently
consume the 35B-A3B MoE shape.

Prototype `.qmi4` files can be imported without the qwenmetal C++ runtime:

```bash
uv run python scripts/import_qwen36_27b_qmi4_artifacts.py \
  /path/to/qwenmetal/build/qwen36-awq-layer0 \
  --out agent_workspace/qwen36_27b/artifacts/layer0
```

The importer writes raw `qweight`, `scales`, and `zeros` files plus a
Metal Marlin manifest.  Keep those generated artifacts under `models/` or
`agent_workspace/qwen36_27b/`; do not add prototype `build/` outputs to the
tracked package.

Runtime code can load one manifest tensor with
`metal_marlin.kernels.qwen36_27b.load_packed_int4_matrix(...)` and pass the
result into the fused wrapper calls.

For full-checkpoint packing, manifests can contain multiple layers of the same
role.  Use `manifest_tensor_index(...)` to reject duplicate `(layer_index,
role)` entries, `expected_roles_for_layer(...)` to follow the real hybrid
cadence, `tensors_for_layer(...)` to fetch the layer-local role map, and
`validate_required_roles_for_layer(...)` to gate per-layer artifacts before
wiring them into the fused decode scheduler.  `artifact_coverage(...)` accepts
either the current template artifact or a full-layer manifest; the older
`validate_required_roles(...)` helper remains intentionally strict for code
that must consume exactly one template tensor per role.

When local layer-0 artifacts are present, validate the imported qmi4 data
against the fused Q/K/V/Beta projection kernel:

```bash
uv run python scripts/validate_qwen36_27b_layer0_artifacts.py
```

For the full imported layer-0 block artifact set, validate the dense MLP
`gate/up/down` path with:

```bash
uv run python scripts/validate_qwen36_27b_layer0_artifacts.py --mode mlp
```

Validate the full-attention `q/k/v/o` projection artifacts with:

```bash
uv run python scripts/validate_qwen36_27b_layer0_artifacts.py --mode attn
```

Validate the remaining linear-attention block artifacts and LM head with:

```bash
uv run python scripts/validate_qwen36_27b_layer0_artifacts.py --mode linear
uv run python scripts/validate_qwen36_27b_layer0_artifacts.py --mode lm-head
```

The same full block manifest can also be used as the Q/K/V/Beta source:

```bash
uv run python scripts/validate_qwen36_27b_layer0_artifacts.py \
  --manifest agent_workspace/qwen36_27b/artifacts/prototype_block_layer0/manifest.json
```

Use `--mode both` for the small Q/K/V/Beta fixture plus full-block MLP, or
`--mode all` to include the full-attention, linear-tail, and LM-head artifact
checks as well.

The optional pytest entrypoint is gated to avoid requiring large generated
artifacts on every machine:

```bash
METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS=1 \
  uv run pytest tests/test_qwen36_27b_local_artifacts.py -q
```

For launch evidence, use the repo-native block-skeleton benchmark rather than
the prototype `qwen36_e2e_bench` binary:

```bash
uv run python benchmarks/bench_qwen36_27b_block_skeleton.py --dry-run
uv run python benchmarks/bench_qwen36_27b_block_skeleton.py \
  --manifest agent_workspace/qwen36_27b/artifacts/prototype_block_layer0/manifest.json \
  --runner direct \
  --max-command-buffers-per-token 1 \
  --max-dispatches-per-token 563 \
  --decode-tokens 4 \
  --warmup-tokens 1
```

The benchmark reports wrapper-level dispatch counts, kernel counts, decode
timing, and manifest coverage fields (`coverage_kind`, `coverage_layers`, and
`coverage_tensors`) for the current fused skeleton.  It keeps
`quality_claim=false` until generation or perplexity validation exists.  Template
manifests report `template_weight_reuse=true`; full-layer manifests resolve
concrete per-layer tensors and report `template_weight_reuse=false`.

`--runner direct` is the default path for launch evidence.  It loads the
manifest into Metal buffers, keeps intermediate activations out of Python
`torch.Tensor` copy-back boundaries, and encodes one command buffer per token.
`--runner wrapper` is retained only as a comparison against the public Python
wrapper surface.

Use the direct-only profiling switches to keep performance work honest:

```bash
uv run python benchmarks/bench_qwen36_27b_block_skeleton.py \
  --runner direct --decode-tokens 4 --warmup-tokens 1 --max-context 5
uv run python benchmarks/bench_qwen36_27b_block_skeleton.py \
  --runner direct --decode-tokens 4 --warmup-tokens 1 --max-context 5 \
  --skip-lm-head --skip-mlp
```

On this local run after porting the lane-parallel prototype GEMV pattern, the
full template-weight skeleton measured about 8.9 tok/s steady-state, while the
no-MLP/no-LM-head profile measured about 23 tok/s.  That makes dense MLP GEMV
the next optimization target; the launch path is already at one command buffer
per token.

The serving stack accepts the same manifest as an opt-in status surface:

```bash
METAL_MARLIN_QWEN36_27B_MEGAKERNEL=1 metal-marlin serve /path/to/model \
  --qwen36-artifact-manifest agent_workspace/qwen36_27b/artifacts/prototype_block_layer0/manifest.json
```

`/v1/models/{model_id}` reports `qwen36_27b_fused.enabled`, the fallback
reason, and structured coverage fields (`coverage_kind`, `coverage_layers`,
and `coverage_tensors`) so callers do not need to parse the reason string.  The
unfused/Trellis path remains the active correctness reference unless that
status is eligible and later generation wiring explicitly consumes the fused
wrappers.

## Runtime Boundary

`metal_marlin.kernels.qwen36_27b` exposes the typed Python wrapper.  It dispatches
symbols from the normal `metal_marlin.metallib` and uses existing buffer bridge,
launch tracing, and feature-flag selection.  Artifact-backed execution should
enter through `decide_fused_artifact_path(...)`, which rejects the fused path
unless all of the following are true:

- `METAL_MARLIN_QWEN36_27B_MEGAKERNEL=1` is set.
- The config is dense `Qwen/Qwen3.6-27B`, not the 35B-A3B MoE shape.
- The manifest uses `qwen36_27b_int4_v1`, model id `Qwen/Qwen3.6-27B`,
  `uint4_asym`, `packed_k_major_u32`, `group_major_f16`, and group size 128.
- The manifest satisfies either template coverage, where every required role is
  present exactly once, or full-layer coverage, where each of the 64 layers has
  exactly the layer-local roles required by its linear/full-attention cadence
  and the global roles are present once.
- The metallib checksum manifest is fresh and every Qwen3.6-27B kernel symbol
  is available.

The first supported hot path is the single-token linear-attention block:

1. `qwen36_27b_int4_qkvb`
2. `qwen36_27b_deltanet_update`

The first-wave shader symbols also include `qwen36_27b_deltanet_interval4`,
`qwen36_27b_int4_linear_az`, `qwen36_27b_linear_o_residual`,
`qwen36_27b_dense_gate_up_silu`, `qwen36_27b_dense_down_residual`,
`qwen36_27b_rmsnorm_hidden`, `qwen36_27b_linear_rmsnorm_gated`,
`qwen36_27b_int4_attention_qkv`, `qwen36_27b_attention_decode`,
`qwen36_27b_attention_cache_write`, `qwen36_27b_attention_o_residual`,
`qwen36_27b_lm_head_logits`, and `qwen36_27b_argmax_f16`.  These are gated
behind the same feature flag and must remain subordinate to the unfused
reference path until the local metallib is fresh and parity evidence exists.

The existing unfused/Trellis/MMFP4 path remains the default and the correctness
reference until launch-budget and parity checks are green.

Current no-weight validation includes live MPS parity for Q/K/V/Beta projection,
DeltaNet update/readout, interval4, dense MLP gate/up/down, RMSNorm, gated
RMSNorm, full-attention Q/K/V and output projection, attention cache write,
token-0 attention decode, linear-tail A/Z/output projection, LM-head logits,
argmax, and the two-dispatch linear-attention wrapper when a fresh metallib is
available.
