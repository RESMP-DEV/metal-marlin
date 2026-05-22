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

The same full block manifest can also be used as the Q/K/V/Beta source:

```bash
uv run python scripts/validate_qwen36_27b_layer0_artifacts.py \
  --manifest agent_workspace/qwen36_27b/artifacts/prototype_block_layer0/manifest.json
```

Use `--mode both` when both the small Q/K/V/Beta fixture and the full block MLP
manifest are available locally.

The optional pytest entrypoint is gated to avoid requiring large generated
artifacts on every machine:

```bash
METAL_MARLIN_QWEN36_27B_VALIDATE_LOCAL_ARTIFACTS=1 \
  uv run pytest tests/test_qwen36_27b_local_artifacts.py -q
```

## Runtime Boundary

`metal_marlin.kernels.qwen36_27b` exposes the typed Python wrapper.  It dispatches
symbols from the normal `metal_marlin.metallib` and uses existing buffer bridge,
launch tracing, and feature-flag selection.  The first supported hot path is the
single-token linear-attention block:

1. `qwen36_27b_int4_qkvb`
2. `qwen36_27b_deltanet_update`

The first-wave shader symbols also include `qwen36_27b_deltanet_interval4`,
`qwen36_27b_dense_gate_up_silu`, `qwen36_27b_dense_down_residual`,
`qwen36_27b_rmsnorm_hidden`, `qwen36_27b_linear_rmsnorm_gated`,
`qwen36_27b_attention_decode`, `qwen36_27b_attention_cache_write`, and
`qwen36_27b_argmax_f16`.  These are gated behind the same feature flag and must
remain subordinate to the unfused reference path until the local metallib is
fresh and parity evidence exists.

The existing unfused/Trellis/MMFP4 path remains the default and the correctness
reference until launch-budget and parity checks are green.

Current no-weight validation includes live MPS parity for Q/K/V/Beta projection,
DeltaNet update/readout, interval4, dense MLP gate/up/down, RMSNorm, gated
RMSNorm, attention cache write, token-0 attention decode, argmax, and the
two-dispatch linear-attention wrapper when a fresh metallib is available.
