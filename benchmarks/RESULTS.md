# Metal Marlin Benchmark Results

*This file is auto-generated. Run `python -m metal_marlin.benchmark_report generate` to update.*

## Summary

No benchmark results found yet.

To generate results:

1. **Run comparison benchmarks:**
   ```bash
   cd metal_marlin
   uv run python -m benchmarks.bench_comparison
   ```

2. **Run perplexity evaluation:**
   ```bash
   uv run python -m metal_marlin.eval.perplexity ./path/to/model --samples 100
   ```

3. **Regenerate this report:**
   ```bash
   uv run python -m metal_marlin.benchmark_report generate ./benchmarks/results/
   ```

## Expected Content

Once benchmarks are run, this report will include:

### Summary Table
Best configuration per model with PPL delta, compression ratio, and throughput.

### Per-Model Detailed Breakdown
All tested configurations for each model, sorted by perplexity delta.

### Uniform vs Mixed-Precision Comparison
Side-by-side comparison showing the quality improvement from mixed-precision
quantization (using different bit-widths for sensitive vs. insensitive layers).

### WikiText-2 vs Bartowski v3 Calibration Comparison
Impact of calibration dataset choice on quantization quality.

## Plots

When matplotlib is available, the report generator creates:

- `plots/ppl_vs_compression.png` - Scatter plot of quality vs compression tradeoff
- `plots/throughput_comparison.png` - Bar chart of throughput by model and config

## Methodology

- **Perplexity**: WikiText-2 test set, 512 token context, cross-entropy based
- **Throughput**: Mean of 100 iterations after 15 warmup iterations, 2-sigma outlier removal
- **Compression ratio**: Original FP16 model size / Quantized model size
- **Hardware**: Apple M4 Max (32 TFLOPS FP16 peak, 546 GB/s memory bandwidth)

## Benchmark Configurations

| Config ID | Quant Type | Group Size | Description |
|-----------|------------|------------|-------------|
| fp4-g32 | FP4 E2M1 | 32 | Tightest FP4, best quality |
| fp4-g64 | FP4 E2M1 | 64 | Balanced FP4 |
| fp4-g128 | FP4 E2M1 | 128 | Standard FP4 (default) |
| fp4-g256 | FP4 E2M1 | 256 | Aggressive FP4 |
| int4-g64 | INT4 asymm | 64 | Integer quantization |
| int4-g128 | INT4 asymm | 128 | Standard INT4 |
| uniform | mixed | varies | Same precision all layers |
| mixed | mixed | varies | Layer-sensitive precision |
