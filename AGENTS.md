# Metal Marlin — Repository Guide

## Project Scope

Metal Marlin is an Apple Silicon inference stack for fast local LLM serving. It is more than an MPS wrapper: the repository includes a Trellis inference stack, Metal shaders, optional native extensions, and a large benchmark corpus.

Before implementing anything new, check whether it already exists in:

- `benchmarks/` — there are already many benchmark drivers,
- the Trellis stack (`TrellisForCausalLM`, `TrellisMLAttention`, `TrellisGenerator`),
- `kernels/`, `shaders/`, or the serving code.

## Environment and Setup

Use the project’s `uv` workflow.

- `uv sync --extra all` — install runtime and dev extras.
- `uv run pytest tests/ -v` — run the test suite.
- `uv run ruff check .` and `uv run ruff format .` — lint and format.
- `uv run pyright metal_marlin/` — type-check the main package.
- `./tests/validation/run_all_validation.sh` — run the broader validation path.

The optional native extension can be built with the repo’s CMake flow when needed.

## Repository Layout

Important areas:

- `metal_marlin/` — main package.
- `inference/`, `trellis/`, `quantization/`, `layers/`, `paged/`, `moe/`, `memory/`, and `serving/` — core runtime surfaces.
- `kernels/` and `shaders/` — GPU compute paths.
- `cpp/` and `metal_marlin/cpp/` — optional native acceleration.
- `tests/` and `tests/validation/` — tests and validation scripts.
- `developer_tests/` — ad hoc diagnostics and performance probes.
- `docs/`, `examples/`, and `scripts/` — documentation and operator helpers.
- `models/` — local model artifacts; do not bury them inside package code.

## Model and Tokenizer Rules

For GLM-4.7-Flash work:

- use the tokenizer from `metal_marlin.trellis_config.GLM4_TOKENIZER_ID`,
- do not use `THUDM/glm-4-9b-chat`, which has the wrong vocabulary size,
- and keep Trellis config, tokenizer, and serving behavior aligned.

## Working Rules

Keep these invariants intact:

- Prefer existing kernels, Trellis abstractions, or benchmark harnesses over adding a parallel implementation path.
- Follow scientific naming conventions where matrix math benefits from established notation.
- Use type hints for public Python APIs.
- Keep new filenames snake_case and test filenames aligned with the affected feature.
- When performance-sensitive behavior changes, document the relevant hardware or macOS assumption in the change notes.

## Validation Expectations

Run the smallest relevant test first, then the wider suite when a change crosses subsystem boundaries.

- Kernel or quantization changes should get targeted tests plus the relevant validation script.
- Serving changes should get CLI or API-focused coverage if the behavior is externally visible.
- Performance claims should be grounded in existing benchmark drivers instead of improvised one-off snippets.
