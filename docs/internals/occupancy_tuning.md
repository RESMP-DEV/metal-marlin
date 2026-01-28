# Occupancy tuning for M4-class GPUs

This note documents the expected threadgroup occupancy for Metal Marlin kernels and provides a simple tuning plan for num_tgs. The target is M4 Max, with variants for M4 Pro and M4.

## Hardware assumptions

- M4 Max GPU cores: 40
- Threadgroup memory budget: 32 KB per core
- SIMDgroups per threadgroup: 4 (128 threads total)

## Threadgroup memory footprint

Current shared-memory usage is ~16 KB per threadgroup, dominated by double-buffered A and B tiles. With a 32 KB budget per core, this leaves room for two concurrent threadgroups per core on memory alone.

## Register pressure

Accumulator footprint (per threadgroup):

- 4 simdgroups x 8 tiles x 64 elements x 2 B = 4 KB

Per-thread registers are typically ~128 registers for this kernel. Register pressure is not expected to reduce occupancy below the two-TG-by-memory limit, but it may prevent going beyond two TGs/core in practice.

## Optimal threadgroup count (M4 Max)

Target total threadgroups in flight:

- 40 cores x 4 waves = 160 threadgroups

This gives a high-latency hiding target across the chip and matches the assumption of two threadgroups per core with multiple waves per TG. For striped kernels, set num_tgs to 160 or a multiple of 160 to keep distribution even.

## Variants for other M4 GPUs

- M4 Pro (18 cores): target num_tgs = 72 (18 x 4)
- M4 (10 cores): target num_tgs = 40 (10 x 4)

Use multiples of these targets for large problem sizes.

## Benchmark plan

Sweep num_tgs and measure throughput (tokens/s or GFLOP/s) to confirm the saturation point. Suggested values:

- num_tgs in {40, 80, 120, 160, 200, 240}

Report throughput vs. num_tgs and note where it plateaus. On M4 Max, expect performance to peak around 160 and flatten or dip beyond that if register pressure or memory bandwidth becomes dominant.
