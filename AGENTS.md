# Repository Guidelines

## Project Structure & Module Organization
`metal_marlin/` is the main package. Core runtime code lives in `inference/`, `trellis/`, `quantization/`, `layers/`, `paged/`, `moe/`, `memory/`, and `serving/`. GPU code is split across `kernels/` and `shaders/`, with optional native acceleration under `cpp/` and `metal_marlin/cpp/`. Tests are in `tests/`, with validation scripts under `tests/validation/` and extra ad hoc checks in `developer_tests/`. Docs live in `docs/`; runnable examples and helper scripts are in `examples/` and `scripts/`. Store large local model artifacts under `models/`, not inside package code.

## Build, Test, and Development Commands
Use `uv` for environment management.

- `uv sync --extra all`: install the project and dev extras.
- `uv run pytest tests/ -v`: run the full Python test suite.
- `uv run pytest tests/test_gemm.py -v` or `uv run pytest -k "moe" tests/ -v`: run a focused subset.
- `./tests/validation/run_all_validation.sh`: run end-to-end validation and performance checks.
- `uv run ruff check .` and `uv run ruff format .`: lint and format Python.
- `uv run pyright metal_marlin/`: run type checking on the package.
- `mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j`: build the optional C++ extension.

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation. Ruff is the source of truth; line length is 100. Use type hints for public APIs and concise docstrings where behavior is not obvious. Follow existing scientific naming where needed for matrix math, even when variables are uppercase. Keep new files and modules snake_case; tests should mirror the feature name, for example `tests/test_paged_attention.py`.

## Testing Guidelines
Pytest is the primary test framework. Add or update tests with every behavior change, especially for kernels, quantization paths, and serving flows. Prefer small targeted tests first, then broader integration coverage when interfaces cross modules. Run the smallest relevant file locally before the full suite.

## Commit & Pull Request Guidelines
History follows Conventional Commit-style prefixes such as `feat:`, `fix:`, `perf:`, `docs:`, and `test:`. Use imperative, scoped summaries, for example `perf: fuse dequant path for decode`. PRs should explain the change, list validation run (`pytest`, `ruff`, `pyright`, benchmarks if relevant), and note any model, hardware, or macOS assumptions for performance-sensitive work.
