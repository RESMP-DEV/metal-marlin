
from pathlib import Path

import yaml

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_METAL_MARLIN_REL = _THIS_FILE.parents[1].relative_to(_REPO_ROOT).as_posix()
_USER_HOME_PATH_RE = r"/(Users|home)/[^/]+/"


def generate_optimization_tasks():
    tasks = []

    # --- P0: Infrastructure & Critical Fixes (10 tasks) ---
    infra_tasks = [
        ("fix-kv-cache-torch-attr", "Fix the AttributeError with `torch.float32` in `contrib/metal_marlin/metal_marlin/kv_cache.py` by ensuring torch is fully initialized before use.",
         "P0", "grep -c 'torch.float32' contrib/metal_marlin/metal_marlin/kv_cache.py"),
        ("auto-detect-metallib-staleness", "Implement a checksum-based staleness check in `contrib/metal_marlin/metal_marlin/metallib_loader.py` that compares source `.metal` files with the compiled `.metallib`.",
         "P0", "grep -q 'checksum' contrib/metal_marlin/metal_marlin/metallib_loader.py"),
        ("add-build-metallib-to-makefile", "Add a `build-metallib` target to `contrib/metal_marlin/Makefile` that calls `scripts/build_metallib.sh`.",
         "P0", "grep -q 'build-metallib:' contrib/metal_marlin/Makefile"),
        ("fix-bf16-kernels-separation", "Ensure `contrib/metal_marlin/src/bf16_kernels.metal` is properly excluded from main library builds if it causes duplicates.",
         "P0", "grep -q 'bf16_kernels' contrib/metal_marlin/scripts/build_metallib.sh"),
        ("ci-metallib-validation", "Add a validation step to `contrib/metal_marlin/scripts/verify_kernels.py` that checks if the precompiled library exists and matches the expected version.",
         "P0", "grep -q 'check_library_exists' contrib/metal_marlin/scripts/verify_kernels.py"),
        ("centralize-model-configs", "Create `contrib/metal_marlin/metal_marlin/models/registry.py` to centralize model IDs and configurations (GQA heads, dimensions).",
         "P0", "test -f contrib/metal_marlin/metal_marlin/models/registry.py"),
        ("rm-absolute-paths-check", "Scan all files in `contrib/metal_marlin` for hardcoded absolute paths and replace them with relative `Path(__file__)` logic.",
         "P0", f"! grep -rE '{_USER_HOME_PATH_RE}' {_METAL_MARLIN_REL}"),
        ("update-ruff-config", "Update `contrib/metal_marlin/pyproject.toml` to globally suppress W293 and F841 warnings.",
         "P0", "grep -q 'W293' contrib/metal_marlin/pyproject.toml"),
        ("add-profiling-decorator", "Implement a `@profile_kernel` decorator in `contrib/metal_marlin/metal_marlin/utils/profiling.py` for Metal timing.",
         "P0", "test -f contrib/metal_marlin/metal_marlin/utils/profiling.py"),
        ("fix-pyright-warnings", "Resolve common Pyright warnings in `contrib/metal_marlin/metal_marlin/` through type annotations.",
         "P0", "uv run pyright contrib/metal_marlin/metal_marlin/ | grep -v 'error'"),
    ]

    for name, prompt, priority, *verify in infra_tasks:
        task = {
            "name": name,
            "prompt": prompt,
            "priority": priority,
        }
        if verify:
            task["verify_command"] = verify[0]
        tasks.append(task)

    # --- P1: Core Performance - Metal Shaders & Fusion (45 tasks) ---
    # Flash Attention v3 (8 tasks)
    fa3_tasks = [
        "Implement chunked QK calculation in `contrib/metal_marlin/src/flash_attention_v3.metal` to support long sequence prefill (>4K tokens).",
        "Implement online softmax partial sum accumulation for chunked attention blocks in `contrib/metal_marlin/src/flash_attention_v3.metal`.",
        "Implement causal mask application within the tiled loop of `contrib/metal_marlin/src/flash_attention_v3.metal`.",
        "Optimize shared memory bank conflicts in `contrib/metal_marlin/src/flash_attention_v3.metal` by using padded shared memory arrays.",
        "Implement GQA (Grouped Query Attention) specialized variants for Flash Attention v3 in `contrib/metal_marlin/src/flash_attention_v3.metal`.",
        "Add SIMDgroup-level reduction (simd_sum) for faster normalization in `contrib/metal_marlin/src/flash_attention_v3.metal`.",
        "Implement `flash_attention_v3_decode` optimized for single-token M=1 case in `contrib/metal_marlin/src/flash_attention_v3.metal`.",
        "Create `contrib/metal_marlin/scripts/benchmark_fa3.py` to benchmark Flash Attention v3 vs v2 across context lengths 1K to 32K.",
    ]
    for i, p in enumerate(fa3_tasks):
        tasks.append({
            "name": f"fa3-optimization-{i+1}",
            "prompt": p,
            "priority": "P1",
            "dependencies": ["auto-detect-metallib-staleness"],
            "verify_command": "grep -ri 'flash_attention_v3' contrib/metal_marlin/src/"
        })

    # MoE Fusion (10 tasks)
    moe_fusion_tasks = [
        "Fuse MoE router selection into a single Metal kernel in `contrib/metal_marlin/src/moe_fused_router.metal` that outputs sorted expert indices.",
        "Implement `moe_fused_expert_swiglu` kernel in `contrib/metal_marlin/src/moe_fused_expert.metal` that combines GEMM + activation in one pass.",
        "Add support for 'top-k' expert grouping on GPU to avoid CPU-based token sorting by implementing `moe_gpu_sort` in `contrib/metal_marlin/src/moe_dispatch.metal`.",
        "Implement asynchronous command buffer submission for parallel expert execution in `contrib/metal_marlin/metal_marlin/moe/moe_dispatch_metal.py`.",
        "Optimize `moe_dispatch` in `contrib/metal_marlin/src/moe_dispatch.metal` to use atomic increments for expert counts on GPU.",
        "Create `contrib/metal_marlin/scripts/benchmark_moe_fusion.py` to benchmark Fused MoE vs Sequential Dispatch for GLM-4.7-Flash (64 experts).",
        "Implement 'expert-prefetch' logic in `contrib/metal_marlin/metal_marlin/moe/expert_weight_manager.py` to overlap weight loading with computation.",
        "Optimize shared experts execution in `contrib/metal_marlin/src/moe_fused_shared_expert.metal` by fusing it with the router kernel.",
        "Add a 'sparse-expert' path for tokens that only active one expert in `contrib/metal_marlin/metal_marlin/moe/moe_dispatch_metal.py`.",
        "Implement 2:4 sparsity support in MoE expert GEMMs in `contrib/metal_marlin/src/sparse_gemm.metal`.",
    ]
    for i, p in enumerate(moe_fusion_tasks):
        tasks.append({
            "name": f"moe-fusion-task-{i+1}",
            "prompt": p,
            "priority": "P1",
            "verify_command": "grep -ri 'moe' contrib/metal_marlin/src/ || grep -ri 'moe' contrib/metal_marlin/metal_marlin/"
        })

    # C++ Dispatch Bridge (15 tasks)
    cpp_tasks = [
        "Create `contrib/metal_marlin/cpp/src/core.cpp` with a high-performance Metal dispatch loop.",
        "Implement `MetalDevice` and `QueueManager` in `contrib/metal_marlin/cpp/src/device.cpp` using the Metal-cpp library.",
        "Expose C++ core to Python using `nanobind` in `contrib/metal_marlin/cpp/src/python_bindings.cpp`.",
        "Implement `CommandEncoder` caching in `contrib/metal_marlin/cpp/src/encoder_cache.cpp` to reduce CPU overhead per dispatch.",
        "Port `marlin_gemm` dispatch logic from Python to `contrib/metal_marlin/cpp/src/gemm_dispatch.cpp`.",
        "Implement zero-copy Buffer sharing between MPS and the C++ extension in `contrib/metal_marlin/cpp/src/buffer_manager.cpp`.",
        "Add asynchronous event waiting in `contrib/metal_marlin/cpp/src/events.cpp` to support overlapping CPU/GPU work.",
        "Implement a 'FastPath' in `contrib/metal_marlin/metal_marlin/metal_dispatch.py` that uses the C++ extension if available.",
        "Create `contrib/metal_marlin/scripts/benchmark_cpp_dispatch.py` to benchmark C++ Dispatch Latency vs Python dispatch.",
        "Add unit tests for the C++ extension in `contrib/metal_marlin/tests/test_cpp_extension.py`.",
        "Implement shader library management in `contrib/metal_marlin/cpp/src/library_manager.cpp` to replace `metallib_loader.py` logic.",
        "Implement a C++ based `BufferPool` in `contrib/metal_marlin/cpp/src/pool.cpp` to avoid GIL during allocation.",
        "Add support for multiple GPU devices in `contrib/metal_marlin/cpp/src/device_discovery.cpp`.",
        "Implement a C++ based `TokenGroup` manager in `contrib/metal_marlin/cpp/src/moe_manager.cpp` for MoE dispatch.",
        "Optimize the C++ bridge to use direct pointers for MTLBuffer access in `contrib/metal_marlin/cpp/src/direct_access.cpp`.",
    ]
    for i, p in enumerate(cpp_tasks):
        tasks.append({
            "name": f"cpp-bridge-task-{i+1}",
            "prompt": p,
            "priority": "P1",
            "verify_command": "test -d contrib/metal_marlin/cpp/src"
        })

    # Dequantization Specialization (12 tasks)
    dequant_tasks = [
        "Implement specialized FP4 dequantization for M=1 (decode) in `contrib/metal_marlin/src/dequant_fp4_fast.metal`.",
        "Add ARM-specific SIMD optimizations (neon) for fallback CPU dequantizers in `contrib/metal_marlin/metal_marlin/utils/cpu_dequant.cpp`.",
        "Implement INT8 quantization support in the `TrellisLinear` fused kernel in `contrib/metal_marlin/src/gemm_trellis.metal`.",
        "Specializing `gemm_trellis` in `contrib/metal_marlin/src/gemm_trellis.metal` for 2-bit weights (4 values per byte).",
        "Add support for 'sub-channel' scales in FP8 dequantization kernels in `contrib/metal_marlin/src/dequant.metal`.",
        "Optimize `viterbi_quant` search in `contrib/metal_marlin/src/viterbi_quant.metal` by using shared memory for trellis states.",
        "Create `contrib/metal_marlin/scripts/benchmark_dequant.py` to benchmark dequantization performance for M=1, 32, 128 shapes.",
        "Implement 'lazy-dequant' that only unpacks weights needed for active experts in `contrib/metal_marlin/src/moe_dispatch.metal`.",
        "Add support for 'group-size=32' in all dequantization kernels in `contrib/metal_marlin/src/dequant.metal`.",
        "Optimize SIMD-shuffled bit unpacking in `contrib/metal_marlin/src/dequant.metal` to avoid memory stalls.",
        "Implement a 'MixedPrecisionLinear' in `contrib/metal_marlin/metal_marlin/layers.py` that handles both BF16 and FP4 inputs.",
        "Add validation tests for dequantization accuracy across all formats in `contrib/metal_marlin/tests/test_dequant_accuracy.py`.",
    ]
    for i, p in enumerate(dequant_tasks):
        tasks.append({
            "name": f"dequant-opt-{i+1}",
            "prompt": p,
            "priority": "P1",
            "verify_command": "grep -ri 'dequant' contrib/metal_marlin/src/ || grep -ri 'dequant' contrib/metal_marlin/metal_marlin/"
        })

    # --- P2: Scaling & Production (30 tasks) ---
    # Continuous Batching (15 tasks)
    cb_tasks = [
        "Implement `BatchScheduler` in `contrib/metal_marlin/metal_marlin/serving/scheduler.py` that handles dynamic insertion of requests.",
        "Implement `PagedKVCache` tracker in `contrib/metal_marlin/metal_marlin/paged/cache_manager.py` for block-level memory management.",
        "Create `contrib/metal_marlin/src/paged_attention.metal` that handles non-contiguous KV blocks.",
        "Add support for variable sequence lengths in a single batch dispatch in `contrib/metal_marlin/metal_marlin/paged/runner.py`.",
        "Implement a 'RunningRequest' state machine in `contrib/metal_marlin/metal_marlin/serving/request.py` for tracking completion.",
        "Add a 'Pre-fill Prioritization' strategy to the batch scheduler in `contrib/metal_marlin/metal_marlin/serving/scheduler.py`.",
        "Optimize block allocation in `contrib/metal_marlin/metal_marlin/paged/block_allocator.py` to minimize memory fragmentation.",
        "Implement 'Speculative Decoding' draft model generation loop in `contrib/metal_marlin/metal_marlin/speculative/engine.py`.",
        "Add validation for Paged Attention vs Linear Attention parity in `contrib/metal_marlin/tests/test_paged_attention.py`.",
        "Implement 'Copy-on-Write' blocks for prompt sharing in `contrib/metal_marlin/metal_marlin/paged/cache_manager.py`.",
        "Add per-request latency tracking in the serving layer in `contrib/metal_marlin/metal_marlin/serving/pipeline.py`.",
        "Optimize the 'KV Cache Eviction' policy in `contrib/metal_marlin/metal_marlin/paged/cache_manager.py` for long-running servers.",
        "Implement 'Batch-Aware LoRA' dispatch in `contrib/metal_marlin/metal_marlin/layers.py` (if LoRA is added later).",
        "Add support for prefix-caching across multiple requests in `contrib/metal_marlin/metal_marlin/serving/scheduler.py`.",
        "Create `contrib/metal_marlin/scripts/benchmark_continuous_batching.py` to benchmark throughput (Req/sec) vs Context Length.",
    ]
    for i, p in enumerate(cb_tasks):
        tasks.append({
            "name": f"cb-optimization-{i+1}",
            "prompt": p,
            "priority": "P2"
        })

    # Advanced Kernels & Formats (15 tasks)
    adv_tasks = [
        "Implement `contrib/metal_marlin/src/rwkv_wkv_v6.metal` optimized kernel for RWKV models.",
        "Add `contrib/metal_marlin/src/mamba_ssm.metal` scan kernel for state-space model support.",
        "Implement `contrib/metal_marlin/src/diff_attention.metal` kernel for Differential Attention models.",
        "Add support for 'Block-Sparse' attention masks in `contrib/metal_marlin/src/attention.metal`.",
        "Implement a 'Search-based' Top-P sampler on GPU in `contrib/metal_marlin/src/sampling.metal`.",
        "Optimize `contrib/metal_marlin/src/sampling.metal` for large vocabulary sizes (>100K).",
        "Add support for 'Quantized KV Cache' in INT8 format in `contrib/metal_marlin/metal_marlin/kv_cache.py`.",
        "Implement a 'Streaming Quantizer' in `contrib/metal_marlin/metal_marlin/quantization/streaming.py` for real-time model loading.",
        "Add 'Weight-Only' dequantization for MLP layers with large hidden dims in `contrib/metal_marlin/src/dequant.metal`.",
        "Optimize `contrib/metal_marlin/src/layernorm.metal` to use multi-pass reduction for larger dims.",
        "Implement `contrib/metal_marlin/src/vision_preprocess.metal` kernels for larger image resolutions (1024x1024).",
        "Add support for 'FP8-E5M2' format in `contrib/metal_marlin/metal_marlin/dtypes.py` for better training compatibility.",
        "Optimize `contrib/metal_marlin/src/hadamard.metal` transforms for non-power-of-2 sizes.",
        "Implement 'Multi-GPU' P2P transfer kernels in `contrib/metal_marlin/src/distributed_gptq.metal` (using Metal Peer-to-Peer).",
        "Add kernel support for 'MoE-Gate' entropy regularization in `contrib/metal_marlin/src/moe_dispatch.metal`.",
    ]
    for i, p in enumerate(adv_tasks):
        tasks.append({
            "name": f"adv-kernel-task-{i+1}",
            "prompt": p,
            "priority": "P2"
        })

    # --- P3: Metrics, Docs & DX (15 tasks) ---
    dx_tasks = [
        "Implement a terminal-based 'Inference Monitor' dashboard in `contrib/metal_marlin/scripts/inference_monitor.py`.",
        "Add Prometheus metrics exporter to the `metal-marlin serve` command in `contrib/metal_marlin/metal_marlin/serving/cli.py`.",
        "Document the C++ extension architecture in `contrib/metal_marlin/docs/cpp_integration.md`.",
        "Create `contrib/metal_marlin/docs/metal_error_guide.md` for common Metal compiler errors.",
        "Add a 'Performance Regression Test' in `contrib/metal_marlin/tests/test_perf_regression.py` that fails if latency increases >5%.",
        "Implement a `metal_marlin.profile` module in `contrib/metal_marlin/metal_marlin/utils/profile_ops.py` for per-layer FLOPs calculation.",
        "Add `API Reference` documentation in `contrib/metal_marlin/docs/api/` using pdoc or sphinx.",
        "Create a 'Quickstart' notebook in `contrib/metal_marlin/examples/GLM-4.7-Flash-MoE.ipynb`.",
        "Implement a 'Model Downloader' script in `contrib/metal_marlin/scripts/hf_download.py` for easy setup of Trellis models.",
        "Add support for 'GGUF' metadata extraction in `contrib/metal_marlin/metal_marlin/gguf_loader.py` for better config parity.",
        "Implement a 'Weight Visualization' tool in `contrib/metal_marlin/scripts/viz_quant.py` for analyzing quantization errors.",
        "Add per-kernel memory bandwidth utilization metrics in `contrib/metal_marlin/scripts/perf_summary.py`.",
        "Create a 'Compatibility Matrix' in `contrib/metal_marlin/docs/compatibility.md` for different Apple Silicon generations.",
        "Document the 'Flash Attention v3' tiling strategies in `contrib/metal_marlin/docs/fa3_architecture.md`.",
        "Clean up all temporary files and logs generated during benchmarks in `contrib/metal_marlin/scripts/cleanup_benchmarks.sh`.",
    ]
    for i, p in enumerate(dx_tasks):
        tasks.append({
            "name": f"dx-task-{i+1}",
            "prompt": p,
            "priority": "P3"
        })

    # Total check
    print(f"Generated {len(tasks)} tasks.")

    output = {
        "tasks": tasks
    }

    with open("agent_workspace/optimization_swarm_v1.yaml", "w") as f:
        f.write("# yaml-language-server: $schema=\n")
        yaml.dump(output, f, sort_keys=False, indent=2)


if __name__ == "__main__":
    generate_optimization_tasks()
