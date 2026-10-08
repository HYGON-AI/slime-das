"""Direct adapters from Slime GLM5 tensors to the ``hip_dsa`` package."""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_hip_dsa():
    try:
        from hip_dsa import DSA, sparse_attn_forward
    except ImportError as error:
        raise ImportError(
            "GLM5 HIP DSA requires the hip-dsa package; "
            "`from hip_dsa import DSA, sparse_attn_forward` must succeed"
        ) from error

    required = (
        "indexer_forward_wrapper",
        "indexer_top_k_wrapper",
        "sparse_attention_backward_wrapper",
    )
    missing = [name for name in required if not hasattr(DSA, name)]
    if missing:
        raise ImportError(f"hip-dsa DSA namespace is missing required interfaces: {missing}")

    logger.info("GLM5 training DSA backend: direct hip-dsa")
    return DSA, sparse_attn_forward


def use_hip_dsa(config) -> bool:
    import os
    """Return whether the Megatron configuration selected the HIP DSA backend."""
    backend = (getattr(config, "dsa_kernel_backend", None) or "none").lower()
    if backend not in ("none", "tilelang", "hip"):
        raise ValueError(f"Unsupported dsa_kernel_backend={backend!r}")
    return backend == "hip"
    #return getattr(config, "dsa_kernel_backend", "none") == "hip"


def _require_supported_context_parallel(config) -> None:
    cp_size = int(getattr(config, "context_parallel_size", 1) or 1)
    if cp_size != 1:
        raise NotImplementedError(
            "Slime's direct hip-dsa adapter currently supports context_parallel_size=1 only"
        )


def _pad_topk(indices: torch.Tensor, target_topk: int) -> torch.Tensor:
    pad = target_topk - indices.shape[-1]
    if pad <= 0:
        return indices
    padding = torch.full(
        (*indices.shape[:-1], pad), -1, dtype=indices.dtype, device=indices.device
    )
    return torch.cat((indices, padding), dim=-1)


def _compact_and_sort_topk(
    indices: torch.Tensor, valid: torch.Tensor, target_topk: int
) -> torch.Tensor:
    """Put valid indices first, sort them, and pad invalid slots with ``-1``."""

    invalid_sort_key = torch.iinfo(torch.int32).max
    indices = torch.where(valid, indices.to(torch.int32), torch.full_like(indices, -1))
    indices = torch.sort(indices.masked_fill(indices < 0, invalid_sort_key), dim=-1).values
    indices = indices.masked_fill(indices == invalid_sort_key, -1)
    return _pad_topk(indices, target_topk).contiguous()


def hip_dsa_select_topk(
    config,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    block_size: int = 8192,
) -> torch.Tensor:
    """Run hip-dsa's fused indexer and return Slime-layout ``[T, 1, K]`` indices."""

    del block_size
    _require_supported_context_parallel(config)
    if index_q.ndim != 3 or index_k.ndim != 2:
        raise ValueError(
            f"hip-dsa indexer expects q=[T,H,D] and k=[S,D], got {index_q.shape} and {index_k.shape}"
        )
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.ndim != 2:
        raise ValueError(f"hip-dsa indexer expects weights=[T,H], got {weights.shape}")
    if (
        starts.ndim != 1
        or ends.ndim != 1
        or starts.numel() != index_q.shape[0]
        or ends.numel() != index_q.shape[0]
    ):
        raise ValueError("hip-dsa indexer starts/ends must have shape [T]")

    dsa, _ = _load_hip_dsa()
    q_bshd = index_q.unsqueeze(0).contiguous()
    k_bshd = index_k.unsqueeze(0).unsqueeze(2).contiguous()
    w_bsh = weights.unsqueeze(0).contiguous()
    starts = starts.to(device=index_q.device, dtype=torch.int32).contiguous()
    ends = ends.to(device=index_q.device, dtype=torch.int32).contiguous()

    with torch.no_grad():
        result = dsa.indexer_forward_wrapper(
            q_bshd,
            k_bshd,
            w_bsh,
            ratio=1,
            sm_scale=1.0,
        )
        scores = result["scores"]
        expected_scores_shape = (1, index_q.shape[0], index_k.shape[0])
        if scores.shape != expected_scores_shape:
            raise RuntimeError(
                f"hip-dsa returned unexpected indexer scores {scores.shape}; "
                f"expected {expected_scores_shape}"
            )

        key_positions = torch.arange(index_k.shape[0], device=index_q.device)
        valid_bounds = (key_positions.view(1, 1, -1) >= starts.view(1, -1, 1)) & (
            key_positions.view(1, 1, -1) < ends.view(1, -1, 1)
        )
        scores = scores.masked_fill(~valid_bounds, float("-inf"))

        selected = min(topk, index_k.shape[0])
        topk_result = dsa.indexer_top_k_wrapper(
            scores.reshape(index_q.shape[0], index_k.shape[0]).contiguous(),
            ends.clamp(min=0, max=index_k.shape[0]),
            top_k=selected,
            next_n=1,
            return_val=False,
        )
        indices = topk_result["indices"].reshape(index_q.shape[0], selected).to(torch.int32)
        safe_indices = indices.clamp(min=0, max=max(0, index_k.shape[0] - 1)).to(torch.long)
        selected_scores = torch.gather(scores.squeeze(0), dim=-1, index=safe_indices)
        valid = (
            (indices >= starts.unsqueeze(-1))
            & (indices < ends.unsqueeze(-1))
            & torch.isfinite(selected_scores)
        )
        indices = _compact_and_sort_topk(indices, valid, topk)

    return indices.unsqueeze(1)


def _padded_head_count(num_heads: int) -> int:
    """Match the head padding used by HCU Megatron's hip-dsa integration."""

    for padded_heads in (64, 128):
        if num_heads == padded_heads or (
            num_heads < padded_heads and padded_heads % num_heads == 0
        ):
            return padded_heads
    if num_heads > 128 and num_heads % 128 == 0:
        return num_heads
    raise RuntimeError(
        f"hip-dsa sparse attention cannot pad local query-head count {num_heads} to 64 or 128"
    )


def _pad_attention_heads(
    q: torch.Tensor,
    out: torch.Tensor | None,
    grad_out: torch.Tensor | None,
    lse: torch.Tensor | None,
    attn_sink: torch.Tensor,
):
    actual_heads = q.shape[1]
    padded_heads = _padded_head_count(actual_heads)
    if padded_heads == actual_heads:
        return q, out, grad_out, lse, attn_sink

    q_padded = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
    q_padded[:, :actual_heads] = q

    def pad_head_tensor(tensor):
        if tensor is None:
            return None
        padded = tensor.new_zeros((tensor.shape[0], padded_heads, *tensor.shape[2:]))
        padded[:, :actual_heads] = tensor
        return padded

    sink_padded = attn_sink.new_full((padded_heads,), float("-inf"))
    sink_padded[:actual_heads] = attn_sink
    return (
        q_padded,
        pad_head_tensor(out),
        pad_head_tensor(grad_out),
        pad_head_tensor(lse),
        sink_padded,
    )


class _HipDsaSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling, d_v):
        if kv.shape[1] != 1 or indices.shape[1] != 1:
            raise ValueError(
                f"hip-dsa requires one compressed KV group, got kv={kv.shape}, indices={indices.shape}"
            )

        _, sparse_attn_forward = _load_hip_dsa()
        q = q.contiguous()
        kv = kv.contiguous()
        flat_indices = indices.squeeze(1).to(torch.int32).contiguous()
        topk_length = (flat_indices >= 0).sum(dim=-1, dtype=torch.int32).contiguous()
        attn_sink = torch.full(
            (q.shape[1],), float("-inf"), dtype=torch.float32, device=q.device
        )
        q_input, _, _, _, sink_input = _pad_attention_heads(q, None, None, None, attn_sink)

        result = sparse_attn_forward(
            q=q_input,
            kv=kv,
            attn_sink=sink_input,
            topk_idxs=flat_indices,
            softmax_scale=float(scaling),
            topk_length=topk_length,
        )
        if len(result) == 2:
            output, lse = result
        elif len(result) == 3:
            output, _max_logits, lse = result
        else:
            raise RuntimeError(f"hip-dsa sparse_attn_forward returned {len(result)} values")

        actual_heads = q.shape[1]
        output = output[:, :actual_heads, :d_v].contiguous()
        lse = lse[:, :actual_heads].contiguous()
        ctx.save_for_backward(
            q,
            kv.squeeze(1).contiguous(),
            attn_sink,
            flat_indices.clamp_min(0).contiguous(),
            output,
            lse,
            topk_length,
        )
        ctx.scaling = float(scaling)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, kv_flat, attn_sink, indices, output, lse, topk_length = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        grad_output = grad_output.contiguous()

        q_input, out_input, grad_input, lse_input, sink_input = _pad_attention_heads(
            q, output, grad_output, lse, attn_sink
        )
        valid_rows = torch.nonzero(topk_length > 0, as_tuple=False).flatten()
        dummy_row = torch.full(
            (1,), q_input.shape[0], dtype=valid_rows.dtype, device=valid_rows.device
        )
        selected_rows = torch.cat((valid_rows, dummy_row))

        q_aug = torch.cat((q_input, torch.zeros_like(q_input[:1])), dim=0)
        out_aug = torch.cat((out_input, torch.zeros_like(out_input[:1])), dim=0)
        grad_aug = torch.cat((grad_input, torch.zeros_like(grad_input[:1])), dim=0)
        lse_aug = torch.cat((lse_input, torch.zeros_like(lse_input[:1])), dim=0)
        indices_aug = torch.cat((indices, torch.zeros_like(indices[:1])), dim=0)
        lengths_aug = torch.cat((topk_length, torch.ones_like(topk_length[:1])), dim=0)

        dsa, _ = _load_hip_dsa()
        result = dsa.sparse_attention_backward_wrapper(
            q_aug.index_select(0, selected_rows).contiguous(),
            kv_flat,
            out_aug.index_select(0, selected_rows).contiguous(),
            grad_aug.index_select(0, selected_rows).contiguous(),
            lse_aug.index_select(0, selected_rows).contiguous(),
            sink_input,
            indices_aug.index_select(0, selected_rows).contiguous(),
            softmax_scale=ctx.scaling,
            topk_length=lengths_aug.index_select(0, selected_rows).contiguous(),
        )
        dq_valid = result["dq"][:-1]
        dq_padded = torch.zeros_like(q_input)
        dq_padded.index_copy_(0, valid_rows, dq_valid)
        dq = dq_padded[:, : q.shape[1], :].contiguous()
        dkv = result["dkv"].reshape_as(kv_flat).unsqueeze(1).contiguous()
        return dq, dkv, None, None, None


def hip_dsa_sparse_attention(
    config,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float,
    d_v: int,
) -> torch.Tensor:
    """Run hip-dsa sparse-attention forward/backward and return ``[T,H,Dv]``."""

    _require_supported_context_parallel(config)
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            "hip-dsa sparse attention expects q/kv/indices rank 3, "
            f"got {q.shape}, {kv.shape}, {indices.shape}"
        )
    if d_v != 512:
        raise ValueError(f"hip-dsa GLM5 sparse attention currently requires d_v=512, got {d_v}")
    return _HipDsaSparseAttention.apply(q, kv, indices, scaling, d_v)
