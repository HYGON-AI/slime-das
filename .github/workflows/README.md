# HCU CI

The HCU workflow validates the smallest supported single-node training path on
an isolated HCU BW1000 self-hosted runner. It is intentionally separate from
the GitHub-hosted static and build smoke workflows.

## Required runner labels

The jobs target the `ci-general` runner group with these labels:

```text
self-hosted, ci, bw1000
```

The runner must provide Docker, `/opt/hyhal`, `/public`, and eight available
HCU devices. Both HCU workflows run inside the pinned verl image
`harbor.sourcefind.cn:5443/dcu/admin/base/custom:verl-das-ubuntu22.04-dtk26.04-py3.10-20260617-2235`.

## Repository variables

Configure these non-secret repository variables before enabling the HCU PR
workflow. Model and dataset directories must be local, read-only paths on the
runner; the workflow never downloads them.

| Variable | Purpose |
| --- | --- |
| `SLIME_DAS_HCU_MEGATRON_ROOT` | HCU Megatron checkout root; the workflows derive `3rdparty/Megatron-Bridge` and `3rdparty/Megatron-LM` from it |
| `SLIME_DAS_HCU_DATA_ROOT` | Parent directory of `dapo-math-17k/` and `aime-2024/` |

SGLang is loaded from the verl image. The model path defaults to the read-only
runner mount `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B-Thinking-2507`.

Do not store tokens, passwords, or writable production paths in repository
variables. Use repository secrets only when an external credential is
strictly required.

## Test scope and pass criteria

`pr-test-hcu.yml` runs only for same-repository pull requests that modify HCU
runtime code, training code, dependencies, HCU examples, or the workflow
itself. Fork pull requests are skipped to prevent untrusted code from running
on the self-hosted HCU runner.

The workflow starts an isolated local Ray head, runs the existing
`hcu_example/run_qwen3_4b.sh` with one rollout and minimum batch sizes, then
uploads logs and stops Ray. It passes only when all of the following succeed:

1. Required model, data, and runtime roots exist.
2. Ray, PyTorch, and HCU Megatron import successfully and one HCU device is visible.
3. The Ray head becomes available.
4. The Qwen3-4B rollout/training command exits with status zero.

The workflow uses a dedicated runner and runs serially. It is not enabled for
fork pull requests and must not share model outputs or Ray processes with
production workloads.
