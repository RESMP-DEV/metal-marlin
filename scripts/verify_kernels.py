from metal_marlin.metal_dispatch import MetalKernelLibrary

lib = MetalKernelLibrary.from_source_dir()

required_kernels = [
    "marlin_gemm_fp4",
    "moe_dispatch_optimized",
    "flash_attention_v2",
    "simdgroup_attention",
    "dense_gemm",
]

for kernel in required_kernels:
    try:
        pipeline = lib.get_pipeline(kernel)
        print(f"✓ {kernel}")
    except Exception as e:
        print(f"✗ {kernel}: {e}")
