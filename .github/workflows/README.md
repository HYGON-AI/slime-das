# HCU CI

The HCU workflows validate the supported single-node rollout and training
paths in an isolated HCU execution environment. They are intentionally
separate from the static and build smoke workflows.

## Runtime requirements

The execution environment must provide Docker, eight available HCU devices,
the required HCU runtime devices and libraries, and enough Docker storage for
the test image. Image selection and storage details are infrastructure-specific
settings and are not duplicated in this document.

HCU Megatron and the test datasets are bundled at `/opt/hcu-megatron` and
`/opt/slime-data`. These are paths inside the container image rather than
runner host paths. SGLang is loaded from the same image. The model directory
must be supplied as a read-only mount through CI configuration; its
environment-specific host path must not be committed to the repository.

Do not commit cluster host paths, tokens, passwords, or writable production
paths. Use repository variables for non-secret CI settings and repository
secrets only when an external credential is strictly required.

## Test scope and pass criteria

`upstream-core-hcu.yml` runs automatically after matching changes are merged
or otherwise pushed to `main`. It can also be started manually with
`workflow_dispatch`. It does not execute untrusted fork code before merge.

`pr-test-hcu.yml` runs only for same-repository pull requests that modify HCU
runtime code, training code, dependencies, HCU examples, or the workflow
itself. Fork pull requests are skipped to prevent untrusted code from running
in the HCU execution environment.

The workflow starts an isolated local Ray head, runs the existing
`hcu_example/run_qwen3_4b.sh` with one rollout and minimum batch sizes, then
uploads logs and stops Ray. It passes only when all of the following succeed:

1. Required model, data, and runtime roots exist.
2. Ray, PyTorch, and HCU Megatron import successfully and one HCU device is visible.
3. The Ray head becomes available.
4. The Qwen3-4B rollout/training command exits with status zero.

The workflow runs serially in an isolated environment. It is not enabled for
fork pull requests and must not share model outputs or Ray processes with
production workloads.
