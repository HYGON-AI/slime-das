# HCU CI

The HCU workflow validates the smallest supported single-node training path on
an isolated HCU BW1000 self-hosted runner. It is intentionally separate from
the GitHub-hosted static and build smoke workflows.

## Required runner labels

The runner must have all of these labels:

```text
self-hosted, Linux, X64, hcu, bw1000
```

The runner must provide the HCU runtime, Ray, Python 3, Docker-compatible
driver environment, and eight available HCU devices. The runner service must
start with the runtime Python and `ray` command on `PATH`.

## Repository variables

Configure these non-secret repository variables before enabling the HCU PR
workflow. Model and dataset directories must be local, read-only paths on the
runner; the workflow never downloads them.

| Variable | Purpose |
| --- | --- |
| `SLIME_DAS_HCU_MEGATRON_ROOT` | HCU Megatron checkout root |
| `SLIME_DAS_MEGATRON_BRIDGE_ROOT` | Megatron-Bridge checkout root |
| `SLIME_DAS_MEGATRON_LM_ROOT` | Megatron-LM checkout root |
| `SLIME_DAS_SGLANG_ROOT` | SGLang checkout root |
| `SLIME_DAS_HCU_QWEN3_4B_MODEL` | Local Qwen3-4B model directory |
| `SLIME_DAS_HCU_DATA_ROOT` | Parent directory of `dapo-math-17k/` and `aime-2024/` |

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
