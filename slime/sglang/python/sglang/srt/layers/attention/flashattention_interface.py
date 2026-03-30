from flash_attn import (
    flash_attn_varlen_func as flash_attn_varlen_func_interface,
    flash_attn_with_kvcache as flash_attn_with_kvcache_interface,
    vllm_flash_attn_varlen_func as vllm_flash_attn_varlen_func_interface,
    vllm_flash_attn_with_kvcache as vllm_flash_attn_with_kvcache_interface,
)
from typing import Optional, Union
from sglang.srt.utils import is_dcu
_is_dcu = is_dcu()

import torch

@torch._dynamo.disable()
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    ver=3,
):
    if _is_dcu:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        if "gfx938" in gcn_arch and (k_cache.dtype == torch.float8_e4m3fn or k_cache.dtype == torch.float8_e5m2):
            try:
                return flash_attn_with_kvcache_interface(
                    q=q.contiguous().view(-1, max_seqlen_q, q.shape[-2], q.shape[-1]).to(k_cache.dtype),
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=page_table,
                    cache_seqlens=cache_seqlens,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    return_softmax_lse=return_softmax_lse,
                    num_splits=num_splits,
                )
            except:
                q = q.contiguous().view(-1, max_seqlen_q, q.shape[-2], q.shape[-1]).to(torch.bfloat16)
                k_cache = k_cache.to(torch.bfloat16)
                v_cache = v_cache.to(torch.bfloat16)
        else:
            q = q.contiguous().view(-1, max_seqlen_q, q.shape[-2], q.shape[-1]).to(torch.bfloat16)
            k_cache = k_cache.to(torch.bfloat16)
            v_cache = v_cache.to(torch.bfloat16)

    return flash_attn_with_kvcache_interface(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            num_splits=num_splits,
        )

def vllm_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = 0,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    ver=3,
):
    if _is_dcu:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        if "gfx938" in gcn_arch and (k_cache.dtype == torch.float8_e4m3fn or k_cache.dtype == torch.float8_e5m2):
            q = q.to(k_cache.dtype)
        elif k_cache.dtype == torch.float8_e4m3fn or k_cache.dtype == torch.float8_e5m2:
            k_cache = k_cache.to(q.dtype)
            v_cache = v_cache.to(q.dtype)
    
    return vllm_flash_attn_with_kvcache_interface(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            num_splits=num_splits,
        )

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    ver=3,
):
    if _is_dcu:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        if "gfx938" in gcn_arch and (k.dtype == torch.float8_e4m3fn or k.dtype == torch.float8_e5m2):
            try:
                q = q.to(k.dtype)
                q_descale = torch.ones_like(k_descale)
                return flash_attn_varlen_func_interface(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=softmax_scale,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    causal=causal,
                    return_attn_probs=return_softmax_lse,
                    softcap=softcap,
                )
            except:
                k = k.to(q.dtype)
                v = v.to(q.dtype)
        else:
            k = k.to(q.dtype)
            v = v.to(q.dtype)

    return flash_attn_varlen_func_interface(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=return_softmax_lse,
        softcap=softcap,
    )

def vllm_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    fa_version,
    q_descale,
    k_descale,
    v_descale,
):
    if _is_dcu:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        if "gfx938" in gcn_arch and (k.dtype == torch.float8_e4m3fn or k.dtype == torch.float8_e5m2):
            q = q.to(k.dtype)
        elif k.dtype == torch.float8_e4m3fn or k.dtype == torch.float8_e5m2:
            k = k.to(q.dtype)
            v = v.to(q.dtype)
    return vllm_flash_attn_varlen_func_interface(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        fa_version=fa_version,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )