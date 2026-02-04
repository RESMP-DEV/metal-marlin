# Contributing to Metal Marlin

Thanks for your interest in contributing! Metal Marlin provides quantized GEMM kernels for Apple Silicon, enabling fast LLM inference on Mac. We welcome improvements that make it faster, more accurate, or easier to use.

## Quick Start

```bash
# Clone and install
git clone https://github.com/RESMP-DEV/metal-marlin.git
cd metal-marlin
uv sync --extra all

# Run tests
uv run pytest tests/ -v -m smoke

# Build C++ extension (5-10x faster dispatch)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cp _cpp_ext.cpython-312-darwin.so ../metal_marlin/
```

## Development Setup

### Prerequisites

- **macOS 13.0+** (Ventura or later)
- **Apple Silicon** (M1/M2/M3/M4)
- **Python 3.12** via `uv`
- **CMake 3.20+** for C++ extension

### Building

```bash
# Python package only
uv sync --extra all

# With C++ extension
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Copy extension to package
cp _cpp_ext.cpython-312-darwin.so ../metal_marlin/

# Verify installation
uv run python -c "from metal_marlin import fast_dispatch_available; print(f'Fast dispatch: {fast_dispatch_available()}')"
```

### Testing

```bash
# Smoke tests (quick, ~30s)
uv run pytest tests/ -v -m smoke

# Full test suite (~4 min, 1565 tests)
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_gemm.py -v

# Pattern matching
uv run pytest -k "moe" tests/ -v

# With coverage
uv run pytest tests/ -v --cov=metal_marlin --cov-report=html
```

### Linting

```bash
# Auto-fix style issues
uv run ruff check . --fix
uv run ruff format .

# Type checking
uv run pyright metal_marlin/
```

## Code Organization

```
metal_marlin/
├── __init__.py          # Public API exports
├── trellis/             # Trellis inference (main use case)
│   ├── trellis_model.py     # TrellisForCausalLM model
│   ├── trellis_attention.py # MLA attention implementation
│   ├── trellis_linear.py    # Quantized linear layers
│   ├── trellis_loader.py    # Layer-wise model loading
│   └── trellis_generate.py  # Text generation
├── kernels/             # Metal shaders
│   ├── gemm_*.metal         # GEMM kernels
│   └── moe_*.metal          # MoE dispatch kernels
├── dispatch/            # Kernel dispatch (PyObjC + C++)
├── quantize/            # Quantization algorithms
└── serve/               # OpenAI-compatible server
```

### Key Types

| Type | Purpose |
|------|---------|
| `TrellisForCausalLM` | Main model class with `from_pretrained()` and `generate()` |
| `MetalQuantizedLinear` | Quantized linear layer with fused dequant+matmul |
| `MetalContext` | Metal device/queue management |
| `BatchDispatch` | Batched kernel dispatch for MoE |

## Adding Features

### Adding a New Metal Kernel

1. Create the kernel in `metal_marlin/kernels/`:

```metal
// metal_marlin/kernels/my_kernel.metal
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < N) {
        output[tid] = input[tid] * 2.0;
    }
}
```

2. Add dispatch function in `metal_marlin/dispatch/`:

```python
def dispatch_my_kernel(input_buffer, output_buffer, N):
    """Dispatch my_kernel on Metal."""
    ...
```

3. Add tests in `tests/test_my_kernel.py`:

```python
def test_my_kernel_basic():
    """Test my_kernel with basic input."""
    ...
```

### Adding a New Quantization Format

1. Define format in `metal_marlin/quantize/formats.py`
2. Implement packing in `metal_marlin/quantize/pack.py`
3. Add Metal dequant shader in `metal_marlin/kernels/dequant_*.metal`
4. Add integration test

## Code Style

### Formatting

```bash
uv run ruff format .
uv run ruff check . --fix
```

### Conventions

- Use type hints for all public functions
- Prefer `torch.Tensor` for GPU tensors, `np.ndarray` for CPU
- Use `tracing`-style logging (`import logging; log = logging.getLogger(__name__)`)
- Keep functions under ~50 lines when practical
- Add docstrings for public APIs

### Commit Messages

```
feat: add FP8 quantization support
fix: handle empty expert batch in MoE dispatch
perf: fuse dequant with matmul in decode kernel
docs: update installation instructions for M4
test: add benchmark for prefill throughput
```

## Pull Request Process

1. **Fork** the repo and create a feature branch
2. **Write tests** for new functionality
3. **Run the test suite** and fix any failures
4. **Update documentation** if adding user-facing features
5. **Open a PR** with a clear description

### PR Checklist

- [ ] `uv run pytest tests/ -v -m smoke` passes
- [ ] `uv run ruff check .` has no errors
- [ ] `uv run pyright metal_marlin/` has no errors
- [ ] New features have tests
- [ ] README updated if needed

## Reporting Issues

When reporting bugs, please include:

1. **macOS version** and chip (e.g., macOS 14.2, M4 Max)
2. **Metal Marlin version** (`uv run python -c "import metal_marlin; print(metal_marlin.__version__)"`)
3. **Steps to reproduce**
4. **Expected vs actual behavior**
5. **Error messages** or logs

For performance issues, include:

- Model being used
- Context length and batch size
- Memory usage
- tok/s measurements

## Areas We'd Love Help With

- **New quantization formats**: INT3, FP6, etc.
- **Model support**: Testing with more models beyond GLM-4.7-Flash
- **Performance**: Metal shader optimizations
- **Documentation**: Examples, troubleshooting guides
- **CI**: Automated benchmarking

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
