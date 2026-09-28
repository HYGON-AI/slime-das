from types import SimpleNamespace

import pytest
import torch

from slime_plugins.models.glm5.ops import hip_dsa

NUM_GPUS = 0


def _config(**kwargs):
    values = {"dsa_kernel_backend": "hip", "context_parallel_size": 1}
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_hip_dsa_config_switch(monkeypatch):
    monkeypatch.delenv("DSA_KERNEL_BACKEND", raising=False)
    assert hip_dsa.use_hip_dsa(_config()) is True
    assert hip_dsa.use_hip_dsa(_config(dsa_kernel_backend="tilelang")) is False


def test_hip_dsa_environment_switch_overrides_config(monkeypatch):
    monkeypatch.setenv("DSA_KERNEL_BACKEND", "hip")
    assert hip_dsa.use_hip_dsa(_config(dsa_kernel_backend="tilelang")) is True


def test_hip_dsa_indexer_calls_package_interfaces(monkeypatch):
    calls = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, weights, **kwargs):
            calls.update(q=q, k=k, weights=weights, indexer_kwargs=kwargs)
            return {"scores": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)}

        @staticmethod
        def indexer_top_k_wrapper(scores, seq_lens, **kwargs):
            calls.update(scores=scores, seq_lens=seq_lens, topk_kwargs=kwargs)
            return {"indices": torch.topk(scores, kwargs["top_k"], dim=-1).indices.int()}

    monkeypatch.setattr(hip_dsa, "_load_hip_dsa", lambda: (FakeDSA, None))
    result = hip_dsa.hip_dsa_select_topk(
        _config(),
        torch.randn(3, 2, 4),
        torch.randn(3, 4),
        torch.randn(3, 2, 1),
        torch.tensor([0, 0, 0]),
        torch.tensor([1, 2, 3]),
        topk=2,
    )

    assert calls["q"].shape == (1, 3, 2, 4)
    assert calls["k"].shape == (1, 3, 1, 4)
    assert calls["weights"].shape == (1, 3, 2)
    assert calls["indexer_kwargs"] == {"ratio": 1, "sm_scale": 1.0}
    assert calls["topk_kwargs"]["return_val"] is False
    assert result.shape == (3, 1, 2)
    assert result.dtype == torch.int32


def test_hip_dsa_sparse_attention_calls_package_forward_and_backward(monkeypatch):
    calls = {}

    class FakeDSA:
        @staticmethod
        def sparse_attention_backward_wrapper(q, kv, out, grad_out, lse, sink, indices, **kwargs):
            calls.update(backward_q=q, backward_kv=kv, backward_indices=indices, backward_kwargs=kwargs)
            return {"dq": torch.ones_like(q), "dkv": torch.ones_like(kv)}

    def fake_sparse_attn_forward(**kwargs):
        calls.update(forward=kwargs)
        q = kwargs["q"]
        return q[..., :512].contiguous(), torch.zeros(q.shape[:2])

    monkeypatch.setattr(
        hip_dsa, "_load_hip_dsa", lambda: (FakeDSA, fake_sparse_attn_forward)
    )
    q = torch.randn(3, 2, 576, requires_grad=True)
    kv = torch.randn(3, 1, 576, requires_grad=True)
    indices = torch.tensor([[[0, -1]], [[1, 0]], [[2, 1]]], dtype=torch.int32)

    output = hip_dsa.hip_dsa_sparse_attention(
        _config(), q, kv, indices, scaling=0.25, d_v=512
    )
    output.sum().backward()

    assert calls["forward"]["q"].shape == (3, 64, 576)
    assert calls["forward"]["kv"].shape == (3, 1, 576)
    assert calls["forward"]["topk_idxs"].shape == (3, 2)
    assert calls["forward"]["topk_length"].tolist() == [1, 2, 2]
    assert calls["backward_q"].shape == (4, 64, 576)
    assert calls["backward_kv"].shape == (3, 576)
    assert output.shape == (3, 2, 512)
    assert q.grad is not None
    assert kv.grad is not None


def test_hip_dsa_adapter_rejects_context_parallel_for_now():
    with pytest.raises(NotImplementedError, match="context_parallel_size=1"):
        hip_dsa.hip_dsa_select_topk(
            _config(context_parallel_size=2),
            torch.randn(3, 2, 4),
            torch.randn(3, 4),
            torch.randn(3, 2),
            torch.zeros(3, dtype=torch.int32),
            torch.ones(3, dtype=torch.int32),
            topk=2,
        )
