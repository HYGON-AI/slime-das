# HCU CI

The HCU workflow validates the smallest supported single-node training path on
an isolated HCU BW1000 self-hosted runner. It is intentionally separate from
the GitHub-hosted static and build smoke workflows.

## Required runner labels

The jobs target the `ci-general` runner group with these labels:

```text
self-hosted, ci, bw1000
```

The runner must provide Docker, `/opt/hyhal`, `/public`, eight available HCU
devices, and enough Docker storage for the bundled image. Both HCU workflows
use the exact image below:

```text
tag: slime-das:hcu-v0.3.0-20260902
id:  sha256:522679376cfe815290508512ec3b02ab6d01fa21426bafc18b35c62a293eea3c
tar: /public/home/niuhb/slime-das-hcu-v0.3.0-20260902.tar.gz
```

The tar archive is available on the shared `/public` storage of every eligible
runner. A job loads it only when that exact image ID is absent, then starts the
container with `--pull=never`. A Harbor account and repository variables are
not required.

HCU Megatron and the test datasets are bundled at `/opt/hcu-megatron` and
`/opt/slime-data`. SGLang is loaded from the same image. The model path remains
the read-only shared mount
`/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B-Thinking-2507`.

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
