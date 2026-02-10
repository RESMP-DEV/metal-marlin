"""GLM-4.7-Flash CORRECT configuration.

Source: transformers.AutoConfig.from_pretrained('zai-org/GLM-4.7-Flash')

CRITICAL: This model uses MLA (Multi-head Latent Attention)!
Similar to DeepSeek-V2, KV cache is compressed via low-rank projections.
This adds overhead but reduces KV cache memory significantly.

WRONG specs used previously (DO NOT USE):
- hidden=4608, intermediate=14336, 8 experts, top-k=2  # WRONG
"""

# ========== Core Dimensions ==========
GLM47_HIDDEN_SIZE = 2048
GLM47_MOE_INTERMEDIATE_SIZE = 1536  # Per expert MLP
GLM47_SHARED_INTERMEDIATE_SIZE = 10240  # Shared FFN
GLM47_NUM_EXPERTS = 64
GLM47_TOP_K = 4
GLM47_NUM_LAYERS = 47
GLM47_VOCAB_SIZE = 154880
GLM47_MAX_CONTEXT = 151552  # 151K context window

# ========== MLA (Multi-head Latent Attention) Dimensions ==========
# KV cache is COMPRESSED via low-rank projection
GLM47_KV_LORA_RANK = 512           # Compressed KV latent dimension
GLM47_QK_ROPE_HEAD_DIM = 64        # RoPE applied only to this
GLM47_QK_NOPE_HEAD_DIM = 192       # Non-RoPE portion
GLM47_QK_HEAD_DIM = 256            # Total (64 + 192)
GLM47_V_HEAD_DIM = 256             # Value head dimension
GLM47_NUM_KV_HEADS = 20
GLM47_NUM_Q_HEADS = 20
GLM47_HEAD_DIM = 64                # config.head_dim (used for some calculations)

# RoPE config
GLM47_ROPE_THETA = 1_000_000
GLM47_ROPE_INTERLEAVE = True

# ========== Derived Values ==========
GLM47_TOTAL_PARAMS = 30_013_259_776  # ~30B
GLM47_ACTIVE_PARAMS_PER_TOKEN = 3_083_206_656  # ~3.08B (MoE top-4)

# MLA adds overhead vs standard MHA:
# 1. kv_a_proj: hidden -> kv_lora_rank + qk_rope_head_dim
# 2. During attention: reconstruct K,V from compressed latent
# 3. RoPE applied only to qk_rope_head_dim portion
