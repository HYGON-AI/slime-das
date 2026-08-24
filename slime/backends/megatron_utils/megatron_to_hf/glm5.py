# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from .deepseekv3 import convert_deepseekv3_to_hf


def convert_glm5_to_hf(args, name, param):
    """Convert GLM5 Megatron parameter names to HuggingFace names.

    GLM5 uses MLA + MoE and its HF tensor layout is aligned with the
    DeepSeekV3-style mapping used by SGLang:
    q_a/q_b, kv_a/kv_b, indexer.*, routed experts, shared experts, and router
    bias. Keep this thin wrapper so GLM5 can be registered independently and
    patched later if GLM5-specific names appear.
    """
    glm5_name_aliases = {
        "self_attention.core_attention.indexer.linear_wq_b.weight": "self_attention.wq_b.weight",
        "self_attention.core_attention.indexer.linear_wk.weight": "self_attention.wk.weight",

        # GLM5 indexer score projection. Different Megatron versions may use
        # slightly different names.
        "self_attention.core_attention.indexer.linear_weights_proj.weight": "self_attention.weights_proj.weight",
        "self_attention.core_attention.indexer.linear_weights.weight": "self_attention.weights_proj.weight",
        "self_attention.core_attention.indexer.weights_proj.weight": "self_attention.weights_proj.weight",

        "self_attention.core_attention.indexer.k_layernorm.weight": "self_attention.k_norm.weight",
        "self_attention.core_attention.indexer.k_layernorm.bias": "self_attention.k_norm.bias",
        "self_attention.core_attention.indexer.k_norm.weight": "self_attention.k_norm.weight",
        "self_attention.core_attention.indexer.k_norm.bias": "self_attention.k_norm.bias",
    }

    for glm5_suffix, deepseek_suffix in glm5_name_aliases.items():
        if name.endswith(glm5_suffix):
            name = name[: -len(glm5_suffix)] + deepseek_suffix
            break
    return convert_deepseekv3_to_hf(args, name, param)