# Metal Marlin Documentation

## Overview
Metal Marlin provides quantized GEMM kernels and a lightweight integration layer to run large language models efficiently on Apple Silicon, focusing on fast dequantization and matrix multiplication while relying on upstream model architectures for correctness and feature coverage.

## Quick Links
- [Getting Started](getting_started.md)
- [CLI Reference](cli.md)
- [Troubleshooting](troubleshooting.md)

## API Reference
- [API Reference](api.md)

## Technical Deep Dives
- [Architecture](architecture.md)
- [Inference Architecture](inference_architecture.md)
- [MoE Architecture](moe_architecture.md)

## Contributing
If you are adding kernel support, start by identifying the target operator and data types, then add or extend the Metal shader, wire it into the kernel registry, and verify numerical parity against a reference implementation. Add targeted benchmarks for the new kernel and document any constraints (alignment, tile sizes, supported layouts). Testing requirements: run the unit tests that cover your kernel path, add a focused regression test for edge cases, and run the relevant performance or integration tests before submitting changes.
