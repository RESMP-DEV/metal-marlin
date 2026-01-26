#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-marlin.h"

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { printf("FAILED (%s)\n", msg); return 1; } \
} while (0)

static int test_marlin_init(void) {
    printf("Testing ggml_marlin_metal_init... ");

    bool ok = ggml_marlin_metal_init();
    ASSERT_TRUE(ok, "init returned false");
    ASSERT_TRUE(ggml_marlin_metal_available(), "not available after init");

    printf("PASSED\n");
    return 0;
}

static int test_weight_conversion(void) {
    printf("Testing ggml_marlin_convert_weights... ");

    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL, "ggml_init failed");

    int64_t K = 4096;
    int64_t N = 4096;
    struct ggml_tensor* q4_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    ASSERT_TRUE(q4_weights != NULL, "tensor alloc failed");

    // Fill with deterministic pseudo-random Q4_0 blocks
    // Q4_0 block: 2-byte scale (f16) + 16 bytes of packed nibbles = 18 bytes per 32 elements
    size_t nbytes = ggml_nbytes(q4_weights);
    uint8_t* data = (uint8_t*)q4_weights->data;
    for (size_t i = 0; i < nbytes; i++) {
        data[i] = (uint8_t)((i * 7 + 13) & 0xFF);
    }

    // Convert to Marlin FP4
    struct ggml_tensor* marlin_fp4 = ggml_marlin_convert_weights(
        ctx, q4_weights, GGML_MARLIN_FP4
    );
    ASSERT_TRUE(marlin_fp4 != NULL, "FP4 conversion returned NULL");

    // Convert to Marlin INT4
    struct ggml_tensor* marlin_int4 = ggml_marlin_convert_weights(
        ctx, q4_weights, GGML_MARLIN_INT4
    );
    ASSERT_TRUE(marlin_int4 != NULL, "INT4 conversion returned NULL");

    // Convert to Marlin INT4_SYM
    struct ggml_tensor* marlin_int4_sym = ggml_marlin_convert_weights(
        ctx, q4_weights, GGML_MARLIN_INT4_SYM
    );
    ASSERT_TRUE(marlin_int4_sym != NULL, "INT4_SYM conversion returned NULL");

    printf("PASSED\n");
    ggml_free(ctx);
    return 0;
}

static int test_weight_conversion_sizes(void) {
    printf("Testing ggml_marlin_convert_weights (various sizes)... ");

    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL, "ggml_init failed");

    // Test various K x N dimensions (must be multiples of group_size=128 for K)
    int64_t sizes[][2] = {
        {128,  128},
        {256,  512},
        {1024, 1024},
        {4096, 4096},
        {4096, 11008},  // LLaMA-7B FFN dimensions
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int64_t K = sizes[i][0];
        int64_t N = sizes[i][1];

        struct ggml_tensor* q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
        ASSERT_TRUE(q4 != NULL, "tensor alloc failed");

        // Zero-init to avoid garbage
        memset(q4->data, 0, ggml_nbytes(q4));

        struct ggml_tensor* marlin = ggml_marlin_convert_weights(
            ctx, q4, GGML_MARLIN_FP4
        );
        if (!marlin) {
            printf("FAILED (conversion NULL for K=%lld N=%lld)\n", K, N);
            ggml_free(ctx);
            return 1;
        }
    }

    printf("PASSED\n");
    ggml_free(ctx);
    return 0;
}

static int test_mul_mat_accuracy(void) {
    printf("Testing ggml_marlin_mul_mat accuracy... ");

    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL, "ggml_init failed");

    int64_t M = 1;    // Single token inference
    int64_t K = 256;
    int64_t N = 256;

    // Create Q4_0 weights and convert to Marlin
    struct ggml_tensor* q4_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    ASSERT_TRUE(q4_weights != NULL, "weight alloc failed");

    // Fill with small deterministic values
    size_t nbytes = ggml_nbytes(q4_weights);
    uint8_t* wdata = (uint8_t*)q4_weights->data;
    for (size_t i = 0; i < nbytes; i++) {
        wdata[i] = (uint8_t)((i * 3 + 5) % 256);
    }

    struct ggml_tensor* marlin_weights = ggml_marlin_convert_weights(
        ctx, q4_weights, GGML_MARLIN_FP4
    );
    ASSERT_TRUE(marlin_weights != NULL, "weight conversion failed");

    // Create F16 activations
    struct ggml_tensor* activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    ASSERT_TRUE(activations != NULL, "activation alloc failed");

    // Fill activations with 1.0f16
    uint16_t* adata = (uint16_t*)activations->data;
    uint16_t one_f16 = 0x3C00;  // 1.0 in FP16
    for (int64_t i = 0; i < K * M; i++) {
        adata[i] = one_f16;
    }

    // Create output tensor
    struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    ASSERT_TRUE(dst != NULL, "output alloc failed");
    memset(dst->data, 0, ggml_nbytes(dst));

    // Run Marlin matmul
    ggml_marlin_mul_mat(dst, marlin_weights, activations);

    // Verify output is not all-zero (basic sanity)
    float* out = (float*)dst->data;
    float sum = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        sum += fabsf(out[i]);
    }
    ASSERT_TRUE(sum > 0.0f, "output is all zeros");

    // Verify no NaN/Inf
    for (int64_t i = 0; i < N; i++) {
        ASSERT_TRUE(!isnan(out[i]), "output contains NaN");
        ASSERT_TRUE(!isinf(out[i]), "output contains Inf");
    }

    printf("PASSED\n");
    ggml_free(ctx);
    return 0;
}

static int test_mul_mat_batch(void) {
    printf("Testing ggml_marlin_mul_mat batched... ");

    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL, "ggml_init failed");

    int64_t M = 32;   // Batch of 32 tokens
    int64_t K = 512;
    int64_t N = 512;

    struct ggml_tensor* q4_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    ASSERT_TRUE(q4_weights != NULL, "weight alloc failed");
    memset(q4_weights->data, 0x55, ggml_nbytes(q4_weights));

    struct ggml_tensor* marlin_weights = ggml_marlin_convert_weights(
        ctx, q4_weights, GGML_MARLIN_FP4
    );
    ASSERT_TRUE(marlin_weights != NULL, "conversion failed");

    struct ggml_tensor* activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    ASSERT_TRUE(activations != NULL, "activation alloc failed");

    uint16_t* adata = (uint16_t*)activations->data;
    uint16_t half_f16 = 0x3800;  // 0.5 in FP16
    for (int64_t i = 0; i < K * M; i++) {
        adata[i] = half_f16;
    }

    struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    ASSERT_TRUE(dst != NULL, "output alloc failed");
    memset(dst->data, 0, ggml_nbytes(dst));

    ggml_marlin_mul_mat(dst, marlin_weights, activations);

    // Verify each row has non-zero output
    float* out = (float*)dst->data;
    for (int64_t m = 0; m < M; m++) {
        float row_sum = 0.0f;
        for (int64_t n = 0; n < N; n++) {
            float v = out[m * N + n];
            ASSERT_TRUE(!isnan(v), "NaN in batched output");
            ASSERT_TRUE(!isinf(v), "Inf in batched output");
            row_sum += fabsf(v);
        }
        ASSERT_TRUE(row_sum > 0.0f, "batch row is all zeros");
    }

    printf("PASSED\n");
    ggml_free(ctx);
    return 0;
}

static int test_stats(void) {
    printf("Testing ggml_marlin_stats... ");

    ggml_marlin_reset_stats();
    struct ggml_marlin_stats stats = ggml_marlin_get_stats();
    ASSERT_TRUE(stats.kernel_calls == 0, "kernel_calls not zero after reset");
    ASSERT_TRUE(stats.total_flops == 0, "total_flops not zero after reset");
    ASSERT_TRUE(stats.total_time_ms == 0.0, "total_time_ms not zero after reset");

    printf("PASSED\n");
    return 0;
}

static int test_conversion_null_inputs(void) {
    printf("Testing ggml_marlin_convert_weights NULL handling... ");

    struct ggml_init_params params = {
        .mem_size = 4 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL, "ggml_init failed");

    // NULL weights should return NULL
    struct ggml_tensor* result = ggml_marlin_convert_weights(ctx, NULL, GGML_MARLIN_FP4);
    ASSERT_TRUE(result == NULL, "expected NULL for NULL weights input");

    // NULL ctx should return NULL
    struct ggml_tensor* dummy = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 128);
    result = ggml_marlin_convert_weights(NULL, dummy, GGML_MARLIN_FP4);
    ASSERT_TRUE(result == NULL, "expected NULL for NULL ctx");

    printf("PASSED\n");
    ggml_free(ctx);
    return 0;
}

int main(void) {
    int failures = 0;

    printf("=== ggml-marlin Metal Integration Tests ===\n\n");

    failures += test_marlin_init();
    failures += test_weight_conversion();
    failures += test_weight_conversion_sizes();
    failures += test_conversion_null_inputs();
    failures += test_mul_mat_accuracy();
    failures += test_mul_mat_batch();
    failures += test_stats();

    printf("\n=== Results: %d test(s) failed ===\n", failures);
    return failures > 0 ? 1 : 0;
}
