# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Platform Reality

This package targets **macOS + Apple Silicon (MPS/Metal)**. The AlphaHENG parent repo runs on Linux, so most kernels, shaders, and the MMFP4/Trellis pipeline **cannot execute here** — changes must be validated on an M-series Mac (the M4 worker per AlphaHENG topology). Treat Linux runs as compile/lint/typecheck only.

## Commands

Python is managed by `uv` against Python 3.12 (required by parent AlphaHENG guard — bare `python`/`pip` are blocked).

- `uv sync --extra all` — install with dev extras
- `uv run pytest tests/ -v` — full Python suite
- `uv run pytest tests/test_gemm.py -v` — single file
- `uv run pytest -k "moe" tests/ -v` — filter by keyword
- `uv run pytest tests/ -v -m smoke` — quick smoke marker
- `uv run ruff check .` / `uv run ruff format .` — lint/format (line length 100)
- `uv run pyright metal_marlin/` — type check
- `./tests/validation/run_all_validation.sh` — end-to-end TPS + perplexity validation (macOS only)
- `mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j` — optional C++ fast-dispatch extension; copy `_cpp_ext.cpython-312-darwin.so` into `metal_marlin/`

Per parent AlphaHENG rules: `cmake --build`, `make`, `cargo build`, `pip install`, and `uv venv` are **blocked by `build_guard.py`** when invoked by agents. The cmake/make step above is for humans on macOS; do not run it from an agent task.

## Architecture

Metal Marlin is an LLM inference engine for Apple Silicon. The package is `metal_marlin/` with these major subsystems (important to understand before editing across files):

- `inference/` — top-level pipelines. `MMFP4Pipeline` (see `mmfp4_pipeline.py`) is the canonical entry point used by README + serving scripts. Trellis decoding lives in `trellis/`.
- `trellis/` — Trellis-v2 quantization codec + GLM-4.7-Flash inference path (the 35 tok/s flagship).
- `quantization/` + `awq.py` + `calibration/` — model conversion (FP4, MMFP4, Trellis-v2, MR-GPTQ). CLI entry is `metal-marlin quantize ...` (see `cli.py`).
- `layers/`, `attention.py`, `activation_metal.py`, `cache_metal.py` — model building blocks backed by Metal kernels.
- `kernels/` + `shaders/` — Metal shader sources and precompiled `.metallib`. See `docs/internals/metallib_architecture.md` for the staleness detection (recent commit `8058396` added that).
- `paged/` — PagedAttention KV cache (used by the OpenAI server). There is an in-flight investigation in `docs/paged_attention_metal_correctness_investigation.md` — read it before touching paged attention correctness.
- `moe/` + `expert_cache.py` + `expert_manager_cpp.py` — Mixture-of-Experts routing, expert caching, and dynamic bit allocation for MoE quantization.
- `memory/` + `buffer_pool.py` / `buffer_ring.py` / `buffer_bridge.py` — unified-memory buffer management; `docs/memory_optimization.md` is the design doc.
- `serving/` + `scripts/serve_glm47.py` — OpenAI-compatible FastAPI server (chat/completions, perplexity, Prometheus `/metrics`, streaming, batching up to 32).
- `cpp/` and `metal_marlin/cpp*` — optional C++ native extension for faster dispatch (5–10× vs pure-Python path). Availability gated by `fast_dispatch_available()`.
- `continuous_batching.py`, `async_dispatch.py`, `early_exit.py` — serving-side throughput features layered over the pipelines.

Ad-hoc test scripts live in `developer_tests/` and `tests/validation/`; canonical pytest suite is `tests/`. Large model artifacts go in `models/` (gitignored), never inside the package.

## Conventions

- Python 3.11+ syntax targeted, 3.12 runtime. 4-space indent, snake_case modules, matrix-math uppercase variables allowed.
- Tests mirror feature names: `tests/test_paged_attention.py` for `paged/`.
- Conventional commit prefixes: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `refactor:`.
- Add/update tests with every behavior change, especially kernels, quantization, and serving flows.

## Parent Repo Integration

This is a submodule under `contrib/metal_marlin` of AlphaHENG. Inherit all parent `CLAUDE.md` rules: Python 3.12 via `uv`, Redis-based control plane, `search_code` before cross-cutting edits, no markdown tables in agent-facing docs, and direct commits (not PRs) for repo-owner fixes unless the user asks otherwise.
