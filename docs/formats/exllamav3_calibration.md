# ExllamaV3 Calibration Methodology

This document analyzes the calibration and quantization methodology from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) and identifies techniques that should be adopted for Metal Marlin's MR-GPTQ implementation.

**Key Insight:** ExllamaV3 uses a completely different quantization algorithm from ExllamaV2. While ExllamaV2 used AdaptiveGPTQ with the EXL2 format, ExllamaV3 introduces the **EXL3 format** with **trellis encoding** and **integrated Hadamard rotation**.

## Core Architecture

ExllamaV3's quantization is a single-pass calibration-driven process:

```
Phase 1: Calibration Capture (convert_model.py)
    │
    │   For each module:
    │     1. Forward pass with calibration data
    │     2. Capture H = X^T @ X (Hessian) during forward
    │     3. Apply random sign flips and Hadamard to H
    │     4. LDL decomposition of regularized H
    │
    ▼
Phase 2: LDLQ Quantization (quantize.py)
    │
    │   For each linear layer:
    │     1. Apply Hadamard rotation to weights
    │     2. Quantize tiles using trellis encoding
    │     3. Error compensation via LDL decomposition
    │     4. Pack into EXL3 format with tensor core layout
    │
    ▼
Output: EXL3 quantized model

Optional: Measure + Optimize (separate workflow)
    │   Compare multiple pre-quantized models at different bitrates
    │   Use KL-divergence to measure quality
    │   Greedy selection of high-quality components
    │
    ▼
Output: Mixed-bitrate EXL3 model
```

## EXL3 Quantization Algorithm

### Hessian Capture and Pre-processing

From `quantize.py`, the Hessian processing includes integrated Hadamard rotation:

```python
def finalize_capture_H(H_data: dict, quant_args: dict, verbose: bool):
    H = H_data["H"]
    
    # Mean of samples summed up during forward pass
    H /= H_data["count"]

    # Regularize diagonal
    diag_mean = torch.diag(H).mean()
    idx = torch.arange(H.shape[0])
    H[idx, idx] += quant_args.get("sigma_reg", 0.025) * diag_mean

    # Random sign flips for input channel
    k = H.shape[0]
    su = (torch.randn(k, device = H.device).sign() + 1e-5).sign()
    
    # Apply Hadamard rotation to H (128-dim blocks)
    H *= su.T
    blockwise_preapply_had_r_(H, had_k)  # had_k = 128
    H *= su
    blockwise_preapply_had_l_(H, had_k)

    # Block LDL decomposition
    L, H = block_ldl(H, 16, verbose)
    dr = torch.arange(k)
    L[dr, dr] = 0  # Zero diagonal for error compensation
    
    return H, L, su, diag
```

Key insights:
1. **Random sign flips**: Applied to input channels before Hadamard
2. **Block Hadamard**: Applied in 128-dim blocks (both left and right)
3. **LDL decomposition**: 16×16 block size for efficient GPU execution
4. **Diagonal regularization**: 2.5% of mean diagonal (configurable)

### LDLQ: Block-wise Quantization with Error Compensation

The core quantization uses LDL decomposition for error compensation:

```python
def ldlq(weight, L, quant_args, pb=None):
    """
    LDLQ: LDL-based quantization with block error compensation
    
    Processes in reverse row order (bottom-up) for proper error propagation.
    """
    size_k, size_n = weight.shape
    buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
    
    prod_cache = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
    weight_q = torch.zeros((size_k, size_n), dtype=torch.float, device=buffer_device)
    encoded = torch.zeros((tiles_k, tiles_n, 256), dtype=torch.short, device=buffer_device)

    # Process rows in reverse order (bottom-up)
    for j in range(size_k, 0, -buf_size_k):
        i = j - buf_size_k
        
        b_weight = weight[i:j]
        b_L = L[i:j]
        
        # Iterate over 16-row blocks within current span
        for bj in range(buf_size_k, 0, -16):
            bi = bj - 16
            
            # Error from already-quantized rows
            bb_err = b_weight[bj:] - b_weight_q[bj:]
            bb_L = b_L[bj:, i + bi:i + bj]
            
            # Compensation term: accumulate error weighted by L
            compensation_term = b_prod_cache[bi:bj]
            compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
            rows = b_weight[bi:bj] + compensation_term
            
            # Reshape to tiles (16×16 blocks)
            tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
            
            # Pre-permute to tensor core layout
            tiles = tiles[:, tensor_core_perm(device)]
            
            # Quantize tiles using trellis encoding
            quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)
            
            # Store results (indices stay in tensor core layout)
            quant_w = quant_w[:, tensor_core_perm_i(device)]
            b_weight_q[bi:bj] = quant_w.reshape(16, size_n)
            b_encoded[bi // 16 : bj // 16] = quant_i.unsqueeze(0)
        
        # Cache error for remaining rows
        b_err = b_weight - b_weight_q
        prod_cache.addmm_(b_L.T, b_err, alpha=1.0, beta=1.0)
    
    return weight_q, encoded
```

This is similar to GPTQ's column-wise error compensation but operates row-wise in 16-row blocks, optimized for tensor core execution.

### Trellis Encoding and Codebooks

ExllamaV3 uses specialized codebooks for quantization:

```python
# Constants from quantize.py
had_k, had_n = 128, 128  # Hadamard block sizes
codebook_scale = 1.24371088

# Codebook options (magic constants for CUDA kernels)
codebook_mcg_mult = 0xCBAC1FED   # Multi-component grid
codebook_mul1_mult = 0x83DCD12D  # Multiplicative codebook

def quantize_tiles(tiles, quant_args):
    """Quantize 256-element tiles using trellis encoding"""
    K = quant_args["K"]       # Bits per weight
    mcg = "mcg" in quant_args
    mul1 = "mul1" in quant_args
    
    # CUDA kernel handles trellis path search
    ext.quantize_tiles(
        tiles,
        quantized_tiles,
        quantized_idx,
        temp_costs,      # Viterbi costs buffer
        temp_edges,      # Edge tracking for backtracking
        K,
        mcg,
        mul1,
    )
    return quantized_tiles, quantized_idx
```

The trellis encoding:
- Operates on 16×16 tiles (256 elements)
- Uses Viterbi algorithm for optimal path search
- Pre-permuted for tensor core memory layout
- Supports 1-8 bits per weight

### Multi-GPU Parallel Quantization

ExllamaV3 supports multi-GPU quantization with automatic load balancing:

```python
def quantize_tiles_multigpu(tiles, quant_args):
    devices = quant_args["devices"]
    if len(devices) == 1:
        return quantize_tiles(tiles, quant_args)
    
    # Split workload across GPUs
    ratios = quant_args.get("device_ratios")
    if ratios:
        split_sizes = [tiles.shape[0] * r / sum(ratios) for r in ratios]
    else:
        split_sizes = [tiles.shape[0] // len(devices)] * len(devices)
    
    # Use pinned memory for efficient CPU↔GPU transfers
    pin_tiles, pin_q_tiles, pin_q_idx = get_pinned(tiles.shape[0])
    
    # Launch quantization on each GPU asynchronously
    for i, device in enumerate(devices):
        stream = get_quant_stream(device)
        with torch.cuda.stream(stream):
            dev_tiles = pin_split_tiles[i].to(device, non_blocking=True)
            ext.quantize_tiles(dev_tiles, ...)
            pin_split_q_tiles[i].copy_(dev_q_tiles, non_blocking=True)
    
    # Gather results
    return q_tiles, q_idx
```

## Optional: Measure and Optimize Workflow

ExllamaV3 provides an optional two-step workflow for creating mixed-bitrate models:

### Measurement Phase (measure_model.py)

Unlike ExllamaV2, ExllamaV3's measurement doesn't try different quantization configs. Instead, it compares **pre-quantized models at different bitrates**:

```python
def main(args, job_state):
    # Load reference model and multiple quantized versions
    dir_ref = args["ref_dir"]        # Unquantized reference
    dir_q = args["in_dir"]           # List of quantized models (increasing bitrate)
    
    model_ref = Model.from_config(Config.from_directory(dir_ref))
    model_q = [Model.from_config(Config.from_directory(d)) for d in dir_q]
    
    # For each module, measure KL-divergence delta
    for idx, module in enumerate(model_ref.modules):
        # Reference forward pass
        new_states_ref = module.forward(states_ref, params)
        
        # Base quantized forward pass
        new_states_q = modules[0].forward(states_q, params)
        base_kld = kldiv(new_states_q, new_states_ref)
        
        # For each candidate upgrade
        for k in range(num_cand):
            # Forward with specific layers from higher-bitrate model
            params["ovr"] = {key: model_q[k + 1].find_module(key) for key in targets}
            s = modules[0].forward(states_q, params)
            cand_kld[k].append(kldiv(s, new_states_ref) - base_kld)
            cand_costs[k].append(cost_delta)
```

Key insight: This measures the **delta KL-divergence** from upgrading specific layers to higher bitrate.

### Optimization Phase (optimize_model.py)

Uses a **greedy algorithm** (not simulated annealing like ExllamaV2):

```python
def optimize(meas, base_numel, base_cost, target_cost, base_kld, num_q):
    groups = meas["groups"]
    num_groups = len(groups)
    solution = [0] * num_groups  # Start with lowest bitrate for all
    budget = target_cost - base_cost

    def adjust(dkld):
        # Non-linear adjustment: penalize positive (worse) more than reward negative
        if dkld > 0:
            return dkld
        return -((-dkld) ** 0.69)

    while True:
        best = None
        best_r = 0.0
        
        # Find best upgrade (most quality per bit)
        for i, g in enumerate(groups):
            for j, c in enumerate(g["candidates"]):
                if j < solution[i]: continue
                
                dk = adjust(c["dkld"])
                db = c["dbits"]
                
                # Ratio: quality improvement per bit spent
                r = 1e10 * dk / (db + 1)
                
                if r < best_r and budget > db:
                    best = i, j, db
                    best_r = r
        
        if best is None:
            break
        
        i, j, db = best
        solution[i] = j + 1
        budget -= db
    
    return solution
```

The adjustment function `(-dkld)^0.69` means:
- Quality improvements are sub-linearly rewarded
- Quality degradations are fully penalized
- This biases toward conservative upgrades

## EXL3 LinearEXL3 Module

The quantized layer representation from `exl3.py`:

```python
class LinearEXL3:
    quant_type: str = "exl3"

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        suh: torch.Tensor,        # Unpacked sign flips (input Hadamard scale)
        svh: torch.Tensor,        # Unpacked sign flips (output Hadamard scale)
        trellis: torch.Tensor,    # Quantized weights in trellis format
        mcg: torch.Tensor | None, # MCG codebook flag
        mul1: torch.Tensor | None,# Mul1 codebook flag
        bias: torch.Tensor | None,
        ...
    ):
        self.K = trellis.shape[-1] // 16  # Bits per weight
        self.in_features = in_features
        self.out_features = out_features
        
        # Create CUDA kernel binding
        self.bc = ext.BC_LinearEXL3(
            self.trellis, self.suh, self.svh,
            self.K, self.bias, self.mcg, self.mul1,
            g_tensor_cache.get(...)
        )

    def forward(self, x, params, out_dtype=None):
        bsz = x.numel() // x.shape[-1]
        torch_mode = params.get("reconstruct", bsz > 32)
        
        if torch_mode:  # Large batch: use Torch matmul
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)  # Input Hadamard
            w = self.get_inner_weight_tensor()         # Reconstruct weights
            ext.hgemm(xh, w, y)                        # GEMM
            ext.had_r_128(y, y, None, self.svh, 1.0)   # Output Hadamard
        else:  # Small batch: fused kernel
            self.bc.run(x, y)
        
        return y

    def get_weight_tensor(self):
        """Reconstruct full weight tensor"""
        suh = self.suh.unsqueeze(1)
        svh = self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)  # Left Hadamard
        w *= suh                       # Input sign flips
        w = preapply_had_r(w, had_n)  # Right Hadamard
        w *= svh                       # Output sign flips
        return w
```

Key features:
1. **Trellis tensor**: `shape = (K*16, out_features//16, in_features)` packed indices
2. **Integrated Hadamard**: `suh` and `svh` are ±1 sign patterns for rotation
3. **Adaptive dispatch**: Fused kernel for small batches, Torch path for large
4. **On-the-fly reconstruction**: Weights reconstructed from trellis when needed

## MoE Expert Handling

From `convert_model.py`, MoE layers get special calibration treatment:

```python
# Capture calibration input states during forward pass
# For block-sparse models, all expert layers are activated
params = {
    "attn_mode": "flash_attn_nc",
    "capture": capture_H,
    "activate_all_experts": model.calibration_all_experts,  # Force all experts
}
rs = module.forward(rs, params)
```

Key insight: ExllamaV3 **activates all experts** during calibration capture to ensure every expert's down projection gets calibration data. This avoids the under-calibration problem endemic to MoE models.

For inference reference states (error measurement), only selected experts are used:
```python
if model.calibration_all_experts:
    # Do not activate all experts for reference state
    params = {"attn_mode": "flash_attn_nc"}
    rs = module.forward(state[i], params)
    ref_states.append(rs.cpu())
```

### Inf/NaN Detection

ExllamaV3 checks for numerical issues in captured Hessians:

```python
for k, v in capture_H.items():
    infs, nans = v["inf_nan"][0].item(), v["inf_nan"][1].item()
    if infs or nans:
        numel = v["num_total"]
        print(f" !! Warning: {k} has {infs:,} inf and {nans:,} NaN (of {numel:,})")
```

## Row-Wise vs Column-Wise Error Propagation

### ExllamaV3's Row-Wise Block Approach

ExllamaV3 processes weights in **row-wise 16-row blocks** rather than column-wise:

```
Weight matrix W: [in_features, out_features] (row-major)
                      ↓
    Iterate over rows in reverse (bottom-up):

    Row 255-240   Row 239-224   ...   Row 15-0
    ───────────   ───────────         ────────
         ↓             ↓                  ↓
     Quantize      Quantize    ...    Quantize
         ↓             ↓                  ↓
     Error →→→→→ Compensate via L^T @ error
```

The error propagation formula in LDLQ:
```python
# For rows bi:bj (16 rows at a time):
bb_err = b_weight[bj:] - b_weight_q[bj:]           # Error from quantized rows
bb_L = b_L[bj:, i + bi:i + bj]                     # L block for compensation
compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
rows = b_weight[bi:bj] + compensation_term         # Compensated input
```

### Why Row-Wise in EXL3?

1. **Tensor core alignment**: 16×16 tiles match GPU tensor core layout
2. **Memory efficiency**: Process in chunks that fit L2 cache
3. **Parallel quantization**: Each tile can be quantized independently after compensation

### Block LDL Decomposition

ExllamaV3 uses **block LDL** (not standard Cholesky):

```python
def block_ldl(H: torch.Tensor, b: int, verbose: bool):
    n, _ = H.shape
    m = n // b  # Number of blocks
    
    # Cholesky: H = L @ L.T
    L = torch.linalg.cholesky(H)
    
    # Get blocks along diagonal: DL.shape = (m, b, b)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    
    # Invert each diagonal block
    DL = torch.linalg.inv(DL)
    
    # Multiply each column by its inverse diagonal block
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]
    
    return L, H
```

This produces a block-normalized L where diagonal blocks are identity matrices.

## Comparison Table

| Feature | GPTQ | AWQ | ExllamaV2/EXL2 | ExllamaV3/EXL3 | Metal Marlin MR-GPTQ |
|---------|------|-----|----------------|----------------|----------------------|
| **Quantization Method** | Column-wise GPTQ | Activation-aware scales | AdaptiveGPTQ | LDLQ + Trellis | Column-wise GPTQ |
| **Error Metric** | MSE per layer | MSE per group | Relative Frobenius | Proxy error + KL-div | MSE per layer |
| **Hessian Processing** | Cholesky inverse | Not used | Cholesky inverse | Block LDL decomposition | Cholesky inverse |
| **Hadamard Rotation** | Optional (QuaRot) | No | No | **Mandatory** (128-dim) | Optional (Hadamard pre-rotate) |
| **Bit Allocation** | Fixed | Fixed | Simulated annealing | Greedy (measure+optimize) | Fixed per layer type |
| **Mixed Precision** | No | No | Yes (bits_prop) | Via measure+optimize | No (single format) |
| **MoE Handling** | Per-expert Hessian | Per-expert activation | Per-expert + warnings | Activate all experts | Per-expert Hessian |
| **Packing Format** | INT4 groups | INT4 groups | Variable-bit groups | Trellis (tensor core) | FP4/INT4 groups |
| **Codebook** | Uniform grid | Uniform grid | Adaptive per-group | MCG / Mul1 | FP4 E2M1 grid |
| **Multi-GPU** | Sequential | Sequential | Sequential | Parallel tiles | Sequential |
| **Group Size** | 128 typical | 128 typical | Variable (32-128) | Implicit (16×16 tiles) | 64-128 |
| **Processing Order** | Column order | Channel order | Column order | Row order (reverse) | Column order |

### Key Differences

**GPTQ**: Classic algorithm, column-wise with Cholesky inverse, fixed bit width.

**AWQ**: Activation-aware scaling without Hessian. Simpler but no error propagation.

**ExllamaV2/EXL2**: AdaptiveGPTQ with simulated annealing for bit allocation. Mixed precision within layers, quantized scale storage.

**ExllamaV3/EXL3**: Complete redesign with mandatory Hadamard rotation, trellis encoding, row-wise LDLQ, and specialized codebooks. More GPU-friendly with tensor core layout and multi-GPU parallelization.

**MR-GPTQ (Current)**: GPTQ + optional Hadamard rotation. Standard column-wise error compensation with FP16 scales.

## Techniques to Adopt for Metal Marlin

### High Priority

1. **Mandatory Hadamard Rotation**

   ExllamaV3 applies Hadamard unconditionally to both weights and Hessian:
   ```python
   # Apply Hadamard to H before LDL decomposition
   H *= su.T
   blockwise_preapply_had_r_(H, 128)
   H *= su
   blockwise_preapply_had_l_(H, 128)
   
   # Apply Hadamard during inference
   ext.had_r_128(x, xh, self.suh, None, 1.0)  # Input
   ext.had_r_128(y, y, None, self.svh, 1.0)   # Output
   ```

   **Why**: Outlier dispersal is always beneficial. Making it mandatory simplifies the pipeline and ensures consistent quality.

2. **Random Sign Flips**

   ExllamaV3 adds random ±1 sign flips before Hadamard:
   ```python
   su = (torch.randn(k, device = H.device).sign() + 1e-5).sign()
   ```

   **Why**: Breaks any remaining structure that Hadamard alone doesn't handle. Low storage cost (1 bit per channel, stored as half).

3. **Block LDL Decomposition**

   Use 16×16 block LDL instead of full Cholesky:
   ```python
   L, H = block_ldl(H, 16, verbose)
   ```

   **Why**: Better numerical stability on GPU, aligns with Metal's M1/M2/M3/M4 SIMD group sizes.

### Medium Priority

4. **Activate All Experts During Calibration**

   For MoE models, force all expert activations during Hessian capture:
   ```python
   params = {"activate_all_experts": model.calibration_all_experts}
   ```

   **Why**: Ensures every expert gets calibration data. Avoids under-calibrated experts.

5. **Row-Wise Block Processing**

   ExllamaV3's reverse row-order processing:
   ```python
   for j in range(size_k, 0, -buf_size_k):
       for bj in range(buf_size_k, 0, -16):
           # Process 16 rows at a time, bottom-up
   ```

   **Why**: Better memory locality for GPU caches. Consider adapting for Metal's unified memory.

6. **Proxy Error Metric**

   Track Hessian diagonal for quality estimation:
   ```python
   diag = H[idx, idx].clone()
   proxy_err = linear.convert_exl3(capture_H[linear.qmap], ...)
   print(f"proxy_err: {proxy_err:.6f}")
   ```

   **Why**: Cheaper than full forward pass for quick quality assessment.

### Lower Priority

7. **Multi-Device Parallel Quantization**

   ExllamaV3's tile-parallel quantization across GPUs:
   ```python
   for i, device in enumerate(devices):
       stream = get_quant_stream(device)
       with torch.cuda.stream(stream):
           ext.quantize_tiles(dev_tiles, ...)
   ```

   **Why**: Accelerates quantization for large models. Metal equivalent would use multiple GPU compute pipelines.

8. **KL-Divergence Based Optimization**

   For mixed-bitrate models:
   ```python
   def kldiv(s, ref):
       ref_probs = torch.softmax(ref, dim=-1)
       s_probs = torch.softmax(s, dim=-1)
       return F.kl_div(torch.log(s_probs + 1e-10), ref_probs, reduction="sum")
   ```

   **Why**: Measures actual output distribution shift, not just reconstruction error.

9. **Checkpoint/Resume Support**

   ExllamaV3's checkpoint interval:
   ```python
   cpi = args.get("checkpoint_interval", 120)  # 2 minutes default
   if time.time() - last_checkpoint_time > cpi:
       save_tensor("ckpt/state.safetensors", state, args)
   ```

   **Why**: Large models take hours. Crash recovery is essential.

## Implementation Roadmap

### Phase 1: Core Improvements
- [ ] Make Hadamard rotation mandatory (always apply)
- [ ] Add random sign flips (±1 per input channel)
- [ ] Implement block LDL decomposition (16×16 blocks)
- [ ] Add proxy error metric logging

### Phase 2: MoE Support
- [ ] Implement "activate all experts" for MoE calibration
- [ ] Add under-calibrated expert detection
- [ ] Consider expert-specific quantization strategies

### Phase 3: Advanced Features
- [ ] Evaluate row-wise LDLQ for Metal (vs column-wise)
- [ ] Implement KL-divergence measurement for optimization
- [ ] Add checkpoint/resume for long calibrations
- [ ] Explore MCG/Mul1 codebook adaptations for FP4

## References

1. **ExllamaV3 Source**: https://github.com/turboderp-org/exllamav3
   - `exllamav3/conversion/convert_model.py`: Main conversion pipeline
   - `exllamav3/conversion/measure_model.py`: KL-divergence measurement
   - `exllamav3/conversion/optimize_model.py`: Greedy bit allocation
   - `exllamav3/modules/quant/exl3_lib/quantize.py`: LDLQ algorithm
   - `exllamav3/modules/quant/exl3.py`: LinearEXL3 inference module

2. **ExllamaV2 Source**: https://github.com/turboderp/exllamav2
   - `exllamav2/conversion/measure.py`: Per-layer sensitivity measurement
   - `exllamav2/conversion/optimize.py`: Simulated annealing bit allocation
   - `exllamav2/conversion/adaptivegptq.py`: AdaptiveGPTQ + scale search

3. **GPTQ Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", Frantar et al., 2022

4. **AWQ Paper**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", Lin et al., 2023

5. **QuaRot Paper**: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", Ashkboos et al., 2024
