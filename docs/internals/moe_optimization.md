# MoE Architecture and Optimization Guide

This document details the Mixture-of-Experts (MoE) implementation in Metal Marlin, focusing on the memory layout, dispatch strategies, and optimization techniques for Apple Silicon GPUs.

## 1. Architecture Overview

The Metal Marlin MoE architecture implements a sparse Top-K gating mechanism with support for optional shared experts. It is designed to minimize memory bandwidth usage, which is the primary bottleneck for MoE inference.

### Core Components

*   **Router:** A linear layer computes gate logits for each token. A Softmax followed by Top-K selection determines the `k` expert indices and their routing weights.
*   **Experts:**
    *   **Routed Experts:** A set of FFN (Feed-Forward Network) blocks. Only `k` (typically 2-8) out of `N` (e.g., 64) are active per token.
    *   **Shared Expert:** An optional, always-active expert that processes all tokens. Its output is weighted and added to the routed experts' output.
*   **Combine:** The outputs from selected experts are weighted by their routing probabilities and summed.

### Data Flow

1.  **Input:** `[batch, sequence, hidden_dim]`
2.  **Routing:** Compute `top_k_indices` and `top_k_probs`.
3.  **Dispatch & Compute:**
    *   **Fused Path:** Single kernel handles routing, computation, and combination.
    *   **Grouped Path:** Tokens are physically or logically reordered (grouped) by expert assignment to maximize weight reuse.
4.  **Shared Expert (Optional):** Computed in parallel or fused into the dispatch kernel.
5.  **Output:** `[batch, sequence, hidden_dim]`

---

## 2. Memory Layout and Dispatch Strategy

Efficient memory access is critical for MoE. We use specific layouts to enable coalesced access and maximize cache hits.

### Memory Layouts

*   **Activations:** `[batch, hidden_dim]` (Half Precision/FP16) - Row-major.
*   **Expert Weights:** `[num_experts, hidden_dim/8, out_dim]` (Packed FP4) - Optimized for 4-bit quantization.
*   **Scales:** `[num_experts, num_groups, out_dim]` (Half Precision) - Per-group scales for dequantization.
*   **Routing Info:**
    *   `expert_ids`: `[batch, top_k]` (uint32)
    *   `expert_probs`: `[batch, top_k]` (Half Precision)

### Dispatch Strategies

We employ three distinct dispatch strategies depending on the workload and hardware characteristics:

#### A. Fused Dispatch (`moe_dispatch_fused`)
*   **Description:** A single kernel performs dispatch, GEMM, and weighted combination.
*   **Mechanism:**
    *   Loads routing table into shared memory (Threadgroup memory).
    *   Each threadgroup processes a tile of the output.
    *   Accumulates results in shared memory before writing once to global memory.
*   **Benefit:** Eliminates intermediate memory traffic (writing sorted tokens or partial expert results), saving ~57KB per token per layer (for hidden=7168).

#### B. Grouped Dispatch (`moe_dispatch_grouped`)
*   **Description:** Tokens assigned to the same expert are logically grouped to execute as a single large batch.
*   **Mechanism:**
    1.  **Histogram:** Compute counts of tokens per expert using threadgroup-local atomics (reduces global contention).
    2.  **Prefix Sum:** Compute offsets for each expert.
    3.  **Scatter:** Generate a `sorted_indices` mapping.
    4.  **Compute:** The kernel processes tokens in contiguous blocks defined by the sorted indices, ensuring expert weights are loaded once and reused across many tokens.

#### C. Parallel Dispatch (`moe_dispatch_parallel`)
*   **Description:** Fully parallel execution of all `top_k` experts for a token.
*   **Mechanism:**
    *   Uses a 3D grid: `[output_blocks, batch_blocks, top_k_slots]`.
    *   Each of the `k` selected experts for a token is processed by a separate threadgroup concurrently.
    *   Results are accumulated into the final output buffer using **FP32 Atomic CAS** (Compare-And-Swap) to handle concurrent writes safely.
*   **Benefit:** Lowest latency by maximizing GPU occupancy, especially for small batch sizes.

---

## 3. Optimization Techniques

### Quantization & Memory Compression
*   **FP4/INT4 Weights:** Expert weights are stored in 4-bit packed formats, reducing memory footprint by 4x compared to FP16.
*   **Trellis Quantization:** Utilizes trellis-coded quantization for higher accuracy at low bitrates (2-bit/3-bit).
*   **On-the-fly Dequantization:** Weights are dequantized in registers during the GEMM computation, keeping data compressed in cache.

### Kernel Optimizations
*   **Coalesced Access:** Kernels are designed such that consecutive threads access consecutive memory addresses for both activations and output, fully utilizing memory bus width.
*   **Shared Memory Tiling:**
    *   Activations and Weights are double-buffered in Threadgroup memory.
    *   Hides memory latency by prefetching the next tile while computing the current one.
*   **SIMD Group Matrix (Matmul):** Utilizes hardware-accelerated matrix multiplication instructions (SIMD-scope) available on Apple Silicon.

### Expert Caching (`ExpertCache`)
For systems with limited RAM (or massive models), not all experts may fit on GPU.
*   **Frequency Tracking:** Tracks expert usage statistics over a sliding window.
*   **Prefetching:** Uses router logits to predict and prefetch likely next experts.
*   **Hybrid Execution:**
    *   **Resident Experts:** Kept in GPU memory for immediate execution.
    *   **Streamed Experts:** Rarely used experts are kept in system RAM and streamed to GPU on-demand asynchronously.
    *   **Pipeline:** Overlaps computation of resident experts with the data transfer of streamed experts.

### Fused Shared Expert
The shared expert computation is fused into the dispatch kernel where possible. Instead of:
`Output = Dispatch(X) + Shared(X)` (3 RW operations: Write Dispatch, Write Shared, Read Both & Sum)
We do:
`Output = FusedDispatchAndShared(X)` (1 Write operation)

---

## 4. Performance Characteristics

*   **Memory Bandwidth Bound:** Like most LLM inference, performance is primarily limited by memory bandwidth. The FP4 quantization provides a linear speedup (approx 2x-3x vs FP16) by reducing data movement.
*   **Batch Size Sensitivity:**
    *   **Small Batch (1-16):** `moe_dispatch_parallel` excels by maximizing parallelism across expert slots.
    *   **Large Batch (64+):** `moe_dispatch_grouped` becomes more efficient as the probability of "expert collision" (multiple tokens needing the same expert) increases, improving weight reuse.
*   **Expert Reuse:** The "Grouped" strategy effectiveness depends on entropy. Low entropy (tokens choosing same experts) leads to high reuse and peak performance. High entropy (random access) degrades to standard GEMV performance.
*   **Atomic Contention:** The "Parallel" strategy's use of atomics is optimized for Apple Silicon's L2 cache hierarchy but can bottleneck if `top_k` is very large (>16) and all experts write to the same output address simultaneously (rare in practice).
