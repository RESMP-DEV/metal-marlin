#include <metal_stdlib>
using namespace metal;

constant uint Q36_HIDDEN = 5120u;
constant uint Q36_DELTA_KEY_HEADS = 16u;
constant uint Q36_DELTA_VALUE_HEADS = 48u;
constant uint Q36_DELTA_KEY_DIM = 128u;
constant uint Q36_DELTA_VALUE_DIM = 128u;
constant uint Q36_DELTA_Q_FEATURES = Q36_DELTA_KEY_HEADS * Q36_DELTA_KEY_DIM;
constant uint Q36_DELTA_K_FEATURES = Q36_DELTA_KEY_HEADS * Q36_DELTA_KEY_DIM;
constant uint Q36_DELTA_V_FEATURES = Q36_DELTA_VALUE_HEADS * Q36_DELTA_VALUE_DIM;
constant uint Q36_DELTA_BETA_FEATURES = Q36_DELTA_VALUE_HEADS;
constant uint Q36_VALUE_HEADS_PER_KEY_HEAD = 3u;
constant uint Q36_MLP_INTERMEDIATE = 17408u;
constant uint Q36_ATTN_HEADS = 24u;
constant uint Q36_ATTN_KV_HEADS = 4u;
constant uint Q36_ATTN_HEAD_DIM = 256u;
constant uint Q36_ATTN_ROTARY_DIM = 64u;
constant uint Q36_ATTN_KV_GROUPS = Q36_ATTN_HEADS / Q36_ATTN_KV_HEADS;
constant uint Q36_ATTN_Q_FEATURES = Q36_ATTN_HEADS * Q36_ATTN_HEAD_DIM;
constant uint Q36_ATTN_O_FEATURES = Q36_ATTN_Q_FEATURES;
constant uint Q36_ATTN_KV_FEATURES = 1024u;
constant uint Q36_VOCAB = 248320u;
constant float Q36_RMS_EPS = 0.000001f;

inline float q36_dequant_u4(uint nibble, float scale, float zero) {
  return (float(nibble) - zero) * scale;
}

inline float q36_silu(float x) {
  return x / (1.0f + exp(-x));
}

inline float q36_sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

inline float q36_int4_dot(device const half *x,
                          device const uint *qweight,
                          device const half *scales,
                          device const half *zeros,
                          uint in_features,
                          uint out_features,
                          uint out_col,
                          uint group_size) {
  float acc = 0.0f;
  const uint packed_rows = in_features >> 3u;
  for (uint packed_k = 0u; packed_k < packed_rows; ++packed_k) {
    const uint packed = qweight[packed_k * out_features + out_col];
    const uint base_k = packed_k << 3u;
    for (uint lane = 0u; lane < 8u; ++lane) {
      const uint k = base_k + lane;
      const uint group = k / group_size;
      const uint meta_index = group * out_features + out_col;
      const float scale = float(scales[meta_index]);
      const float zero = float(zeros[meta_index]);
      const uint nibble = (packed >> (lane * 4u)) & 0xFu;
      acc += float(x[k]) * q36_dequant_u4(nibble, scale, zero);
    }
  }
  return acc;
}

kernel void qwen36_27b_int4_qkvb(device const half *x [[buffer(0)]],
                                 device const uint *q_qweight [[buffer(1)]],
                                 device const half *q_scales [[buffer(2)]],
                                 device const half *q_zeros [[buffer(3)]],
                                 device const uint *k_qweight [[buffer(4)]],
                                 device const half *k_scales [[buffer(5)]],
                                 device const half *k_zeros [[buffer(6)]],
                                 device const uint *v_qweight [[buffer(7)]],
                                 device const half *v_scales [[buffer(8)]],
                                 device const half *v_zeros [[buffer(9)]],
                                 device const uint *b_qweight [[buffer(10)]],
                                 device const half *b_scales [[buffer(11)]],
                                 device const half *b_zeros [[buffer(12)]],
                                 device half *q_out [[buffer(13)]],
                                 device half *k_out [[buffer(14)]],
                                 device half *v_out [[buffer(15)]],
                                 device half *beta_out [[buffer(16)]],
                                 device const uint *params [[buffer(17)]],
                                 uint gid [[thread_position_in_grid]]) {
  const uint group_size = params[0];
  const uint total_cols = params[1];
  if (gid >= total_cols) {
    return;
  }

  if (gid < Q36_DELTA_Q_FEATURES) {
    q_out[gid] = half(q36_int4_dot(x, q_qweight, q_scales, q_zeros,
                                   Q36_HIDDEN, Q36_DELTA_Q_FEATURES, gid,
                                   group_size));
    return;
  }

  uint col = gid - Q36_DELTA_Q_FEATURES;
  if (col < Q36_DELTA_K_FEATURES) {
    k_out[col] = half(q36_int4_dot(x, k_qweight, k_scales, k_zeros,
                                   Q36_HIDDEN, Q36_DELTA_K_FEATURES, col,
                                   group_size));
    return;
  }

  col -= Q36_DELTA_K_FEATURES;
  if (col < Q36_DELTA_V_FEATURES) {
    v_out[col] = half(q36_int4_dot(x, v_qweight, v_scales, v_zeros,
                                   Q36_HIDDEN, Q36_DELTA_V_FEATURES, col,
                                   group_size));
    return;
  }

  col -= Q36_DELTA_V_FEATURES;
  if (col < Q36_DELTA_BETA_FEATURES) {
    beta_out[col] = half(q36_int4_dot(x, b_qweight, b_scales, b_zeros,
                                      Q36_HIDDEN, Q36_DELTA_BETA_FEATURES, col,
                                      group_size));
  }
}

kernel void qwen36_27b_int4_attention_qkv(device const half *x [[buffer(0)]],
                                          device const uint *q_qweight [[buffer(1)]],
                                          device const half *q_scales [[buffer(2)]],
                                          device const half *q_zeros [[buffer(3)]],
                                          device const uint *k_qweight [[buffer(4)]],
                                          device const half *k_scales [[buffer(5)]],
                                          device const half *k_zeros [[buffer(6)]],
                                          device const uint *v_qweight [[buffer(7)]],
                                          device const half *v_scales [[buffer(8)]],
                                          device const half *v_zeros [[buffer(9)]],
                                          device half *q_out [[buffer(10)]],
                                          device half *k_out [[buffer(11)]],
                                          device half *v_out [[buffer(12)]],
                                          device const uint *params [[buffer(13)]],
                                          uint gid [[thread_position_in_grid]]) {
  const uint group_size = params[0];
  const uint total_cols = params[1];
  if (gid >= total_cols) {
    return;
  }

  if (gid < Q36_ATTN_Q_FEATURES * 2u) {
    q_out[gid] = half(q36_int4_dot(x, q_qweight, q_scales, q_zeros,
                                   Q36_HIDDEN, Q36_ATTN_Q_FEATURES * 2u, gid,
                                   group_size));
    return;
  }

  uint col = gid - Q36_ATTN_Q_FEATURES * 2u;
  if (col < Q36_ATTN_KV_FEATURES) {
    k_out[col] = half(q36_int4_dot(x, k_qweight, k_scales, k_zeros,
                                   Q36_HIDDEN, Q36_ATTN_KV_FEATURES, col,
                                   group_size));
    return;
  }

  col -= Q36_ATTN_KV_FEATURES;
  if (col < Q36_ATTN_KV_FEATURES) {
    v_out[col] = half(q36_int4_dot(x, v_qweight, v_scales, v_zeros,
                                   Q36_HIDDEN, Q36_ATTN_KV_FEATURES, col,
                                   group_size));
  }
}

kernel void qwen36_27b_deltanet_update(device const half *q [[buffer(0)]],
                                       device const half *k [[buffer(1)]],
                                       device const half *v [[buffer(2)]],
                                       device const half *beta [[buffer(3)]],
                                       device half *state [[buffer(4)]],
                                       device half *y [[buffer(5)]],
                                       uint gid [[thread_position_in_grid]]) {
  if (gid >= Q36_DELTA_V_FEATURES) {
    return;
  }

  const uint value_head = gid / Q36_DELTA_VALUE_DIM;
  const uint value_col = gid % Q36_DELTA_VALUE_DIM;
  const uint key_head = value_head / Q36_VALUE_HEADS_PER_KEY_HEAD;
  const uint qk_base = key_head * Q36_DELTA_KEY_DIM;
  const uint value_base = value_head * Q36_DELTA_VALUE_DIM;
  const uint state_base =
      value_head * Q36_DELTA_KEY_DIM * Q36_DELTA_VALUE_DIM + value_col;

  float prediction = 0.0f;
  for (uint key_col = 0u; key_col < Q36_DELTA_KEY_DIM; ++key_col) {
    const uint state_index = state_base + key_col * Q36_DELTA_VALUE_DIM;
    prediction += float(k[qk_base + key_col]) * float(state[state_index]);
  }

  const float beta_value = float(beta[value_head]);
  const float delta = float(v[value_base + value_col]) - prediction;
  float output = 0.0f;
  for (uint key_col = 0u; key_col < Q36_DELTA_KEY_DIM; ++key_col) {
    const uint state_index = state_base + key_col * Q36_DELTA_VALUE_DIM;
    const float next_state =
        float(state[state_index]) + beta_value * float(k[qk_base + key_col]) * delta;
    state[state_index] = half(next_state);
    output += float(q[qk_base + key_col]) * next_state;
  }
  y[gid] = half(output);
}

kernel void qwen36_27b_deltanet_interval4(device const half *q [[buffer(0)]],
                                          device const half *k [[buffer(1)]],
                                          device const half *v [[buffer(2)]],
                                          device const half *beta [[buffer(3)]],
                                          device half *state [[buffer(4)]],
                                          device half *y [[buffer(5)]],
                                          uint gid [[thread_position_in_grid]]) {
  const uint layer_size = Q36_DELTA_V_FEATURES;
  if (gid >= 3u * layer_size) {
    return;
  }
  const uint local_layer = gid / layer_size;
  const uint local_gid = gid - local_layer * layer_size;
  const uint value_head = local_gid / Q36_DELTA_VALUE_DIM;
  const uint value_col = local_gid % Q36_DELTA_VALUE_DIM;
  const uint key_head = value_head / Q36_VALUE_HEADS_PER_KEY_HEAD;
  const uint qk_base = (local_layer * Q36_DELTA_KEY_HEADS + key_head) * Q36_DELTA_KEY_DIM;
  const uint value_base =
      (local_layer * Q36_DELTA_VALUE_HEADS + value_head) * Q36_DELTA_VALUE_DIM;
  const uint state_base =
      ((local_layer * Q36_DELTA_VALUE_HEADS + value_head) *
       Q36_DELTA_KEY_DIM * Q36_DELTA_VALUE_DIM) + value_col;

  float prediction = 0.0f;
  for (uint key_col = 0u; key_col < Q36_DELTA_KEY_DIM; ++key_col) {
    prediction += float(k[qk_base + key_col]) *
                  float(state[state_base + key_col * Q36_DELTA_VALUE_DIM]);
  }

  const float beta_value = float(beta[local_layer * Q36_DELTA_BETA_FEATURES + value_head]);
  const float delta = float(v[value_base + value_col]) - prediction;
  float output = 0.0f;
  for (uint key_col = 0u; key_col < Q36_DELTA_KEY_DIM; ++key_col) {
    const uint state_index = state_base + key_col * Q36_DELTA_VALUE_DIM;
    const float next_state =
        float(state[state_index]) + beta_value * float(k[qk_base + key_col]) * delta;
    state[state_index] = half(next_state);
    output += float(q[qk_base + key_col]) * next_state;
  }
  y[gid] = half(output);
}

kernel void qwen36_27b_dense_gate_up_silu(device const half *x [[buffer(0)]],
                                          device const uint *gate_qweight [[buffer(1)]],
                                          device const half *gate_scales [[buffer(2)]],
                                          device const half *gate_zeros [[buffer(3)]],
                                          device const uint *up_qweight [[buffer(4)]],
                                          device const half *up_scales [[buffer(5)]],
                                          device const half *up_zeros [[buffer(6)]],
                                          device half *intermediate [[buffer(7)]],
                                          device const uint *params [[buffer(8)]],
                                          uint gid [[thread_position_in_grid]]) {
  if (gid >= Q36_MLP_INTERMEDIATE) {
    return;
  }
  const uint group_size = params[0];
  const float gate = q36_int4_dot(x, gate_qweight, gate_scales, gate_zeros,
                                  Q36_HIDDEN, Q36_MLP_INTERMEDIATE, gid,
                                  group_size);
  const float up = q36_int4_dot(x, up_qweight, up_scales, up_zeros,
                                Q36_HIDDEN, Q36_MLP_INTERMEDIATE, gid,
                                group_size);
  intermediate[gid] = half(q36_silu(gate) * up);
}

kernel void qwen36_27b_dense_down_residual(device const half *intermediate [[buffer(0)]],
                                           device const uint *down_qweight [[buffer(1)]],
                                           device const half *down_scales [[buffer(2)]],
                                           device const half *down_zeros [[buffer(3)]],
                                           device const half *residual [[buffer(4)]],
                                           device half *out [[buffer(5)]],
                                           device const uint *params [[buffer(6)]],
                                           uint gid [[thread_position_in_grid]]) {
  if (gid >= Q36_HIDDEN) {
    return;
  }
  const uint group_size = params[0];
  const float down = q36_int4_dot(intermediate, down_qweight, down_scales, down_zeros,
                                  Q36_MLP_INTERMEDIATE, Q36_HIDDEN, gid,
                                  group_size);
  out[gid] = half(float(residual[gid]) + down);
}

kernel void qwen36_27b_rmsnorm_hidden(device const half *x [[buffer(0)]],
                                      device const half *weight [[buffer(1)]],
                                      device half *out [[buffer(2)]],
                                      uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float partial[256];
  float sum = 0.0f;
  for (uint i = tid; i < Q36_HIDDEN; i += 256u) {
    const float value = float(x[i]);
    sum += value * value;
  }
  partial[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_rms = rsqrt(partial[0] / float(Q36_HIDDEN) + Q36_RMS_EPS);
  for (uint i = tid; i < Q36_HIDDEN; i += 256u) {
    out[i] = half(float(x[i]) * inv_rms * float(weight[i]));
  }
}

kernel void qwen36_27b_linear_rmsnorm_gated(device const half *x [[buffer(0)]],
                                            device const half *gate [[buffer(1)]],
                                            device const half *weight [[buffer(2)]],
                                            device half *out [[buffer(3)]],
                                            uint value_head [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float partial[128];
  if (value_head >= Q36_DELTA_VALUE_HEADS || tid >= Q36_DELTA_VALUE_DIM) {
    return;
  }

  const uint base = value_head * Q36_DELTA_VALUE_DIM;
  const float value = float(x[base + tid]);
  partial[tid] = value * value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = Q36_DELTA_VALUE_DIM >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms =
      rsqrt(partial[0] / float(Q36_DELTA_VALUE_DIM) + Q36_RMS_EPS);
  out[base + tid] = half(value * inv_rms * float(weight[tid]) *
                         q36_silu(float(gate[base + tid])));
}

kernel void qwen36_27b_attention_cache_write(device const half *k [[buffer(0)]],
                                             device const half *v [[buffer(1)]],
                                             device half *k_cache [[buffer(2)]],
                                             device half *v_cache [[buffer(3)]],
                                             device const uint *token_position [[buffer(4)]],
                                             uint gid [[thread_position_in_grid]]) {
  if (gid >= Q36_ATTN_KV_FEATURES) {
    return;
  }
  const uint base = token_position[0] * Q36_ATTN_KV_FEATURES;
  k_cache[base + gid] = k[gid];
  v_cache[base + gid] = v[gid];
}

kernel void qwen36_27b_attention_decode(device const half *q_proj [[buffer(0)]],
                                        device const half *k_proj [[buffer(1)]],
                                        device const half *v_proj [[buffer(2)]],
                                        device const half *q_norm_weight [[buffer(3)]],
                                        device const half *k_norm_weight [[buffer(4)]],
                                        device half *k_cache [[buffer(5)]],
                                        device half *v_cache [[buffer(6)]],
                                        device half *attn_out [[buffer(7)]],
                                        device const uint *token_position [[buffer(8)]],
                                        uint head [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partial[256];
  threadgroup float q_head[256];
  threadgroup float k_head[256];

  if (head >= Q36_ATTN_HEADS || lane >= Q36_ATTN_HEAD_DIM) {
    return;
  }

  const uint position = token_position[0];
  const uint kv_head = head / Q36_ATTN_KV_GROUPS;
  const uint q_base = head * Q36_ATTN_HEAD_DIM;
  const uint gate_base = Q36_ATTN_O_FEATURES + q_base;
  const uint kv_base = kv_head * Q36_ATTN_HEAD_DIM;
  const uint cache_base = position * Q36_ATTN_KV_FEATURES + kv_base;

  const float q_value = float(q_proj[q_base + lane]);
  partial[lane] = q_value * q_value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float q_inv_rms =
      rsqrt(partial[0] / float(Q36_ATTN_HEAD_DIM) + Q36_RMS_EPS);
  q_head[lane] = q_value * q_inv_rms * float(q_norm_weight[lane]);

  const float k_value = float(k_proj[kv_base + lane]);
  partial[lane] = k_value * k_value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float k_inv_rms =
      rsqrt(partial[0] / float(Q36_ATTN_HEAD_DIM) + Q36_RMS_EPS);
  k_head[lane] = k_value * k_inv_rms * float(k_norm_weight[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane < Q36_ATTN_ROTARY_DIM) {
    const uint half_rot = Q36_ATTN_ROTARY_DIM >> 1u;
    const bool second_half = lane >= half_rot;
    const uint pair_lane = second_half ? (lane - half_rot) : (lane + half_rot);
    const uint freq_lane = second_half ? (lane - half_rot) : lane;
    const float angle =
        float(position) * pow(10000.0f, -float(freq_lane) / float(half_rot));
    const float c = cos(angle);
    const float s = sin(angle);
    const float q_self = q_head[lane];
    const float q_pair = q_head[pair_lane];
    const float k_self = k_head[lane];
    const float k_pair = k_head[pair_lane];
    q_head[lane] =
        second_half ? (q_self * c + q_pair * s) : (q_self * c - q_pair * s);
    k_head[lane] =
        second_half ? (k_self * c + k_pair * s) : (k_self * c - k_pair * s);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  k_cache[cache_base + lane] = half(k_head[lane]);
  v_cache[cache_base + lane] = v_proj[kv_base + lane];
  threadgroup_barrier(mem_flags::mem_device);

  constexpr float attn_scale = 0.0625f;
  float max_score = -3.402823466e+38f;
  for (uint token = 0u; token <= position; ++token) {
    const uint offset = token * Q36_ATTN_KV_FEATURES + kv_base + lane;
    partial[lane] = q_head[lane] * float(k_cache[offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lane < stride) {
        partial[lane] += partial[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_score = max(max_score, partial[0] * attn_scale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float denom = 0.0f;
  float value_acc = 0.0f;
  for (uint token = 0u; token <= position; ++token) {
    const uint offset = token * Q36_ATTN_KV_FEATURES + kv_base + lane;
    partial[lane] = q_head[lane] * float(k_cache[offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lane < stride) {
        partial[lane] += partial[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float weight = exp(partial[0] * attn_scale - max_score);
    denom += weight;
    value_acc += weight * float(v_cache[offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float gate = q36_sigmoid(float(q_proj[gate_base + lane]));
  attn_out[q_base + lane] = half((value_acc / denom) * gate);
}

kernel void qwen36_27b_attention_o_residual(device const half *attn_out [[buffer(0)]],
                                            device const uint *o_qweight [[buffer(1)]],
                                            device const half *o_scales [[buffer(2)]],
                                            device const half *o_zeros [[buffer(3)]],
                                            device const half *residual [[buffer(4)]],
                                            device half *out [[buffer(5)]],
                                            device const uint *params [[buffer(6)]],
                                            uint gid [[thread_position_in_grid]]) {
  if (gid >= Q36_HIDDEN) {
    return;
  }
  const uint group_size = params[0];
  const float projected = q36_int4_dot(attn_out, o_qweight, o_scales, o_zeros,
                                       Q36_ATTN_O_FEATURES, Q36_HIDDEN, gid,
                                       group_size);
  out[gid] = half(float(residual[gid]) + projected);
}

kernel void qwen36_27b_argmax_f16(device const half *logits [[buffer(0)]],
                                  device int *token_out [[buffer(1)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float best_values[256];
  threadgroup uint best_indices[256];
  float best = -3.402823466e+38f;
  uint best_index = 0u;
  for (uint i = tid; i < Q36_VOCAB; i += 256u) {
    const float value = float(logits[i]);
    if (value > best) {
      best = value;
      best_index = i;
    }
  }
  best_values[tid] = best;
  best_indices[tid] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride && best_values[tid + stride] > best_values[tid]) {
      best_values[tid] = best_values[tid + stride];
      best_indices[tid] = best_indices[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    token_out[0] = int(best_indices[0]);
  }
}
