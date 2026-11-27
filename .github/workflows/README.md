# HCU CI

The HCU workflows validate the supported single-node rollout and training
paths in an isolated HCU execution environment. They are intentionally
separate from the static and build smoke workflows.

## Runtime requirements

The HCU jobs run on the `hygon-bw1100-8card` ARC K8s pool. The job template
injects eight HCU devices, `/dev/shm`, `/opt/hyhal`, `/public` and
`/ci_public`, and the job itself runs inside `HCU_BASE_IMAGE`, so no
Docker-in-Docker is involved.

HCU Megatron and the test datasets are bundled at `/opt/hcu-megatron` and
`/opt/slime-data`. These are paths inside the container image rather than pod
mounts. SGLang is loaded from the same image. The model directory is read
from `CI_MODEL_PATH` inside the pod and must contain `config.json`.

The workflows read non-secret infrastructure settings from repository
variables in the target repository:

- `HCU_BASE_IMAGE`: image used as the job container for the HCU tests.
- `CI_MODEL_PATH`: model directory readable inside the pod; it must contain
  `config.json` and `model.safetensors.index.json`.

Environment-specific pod paths belong only in these repository variables;
the workflow, README, and scripts must not provide cluster-path fallbacks.

Do not commit cluster host paths, tokens, passwords, or writable production
paths. Use repository variables for non-secret CI settings and repository
secrets only when an external credential is strictly required.

## Test scope and pass criteria

`upstream-core-hcu.yml` uses `pull_request_target` to run the six HCU upstream
core scenarios for matching fork pull requests during review. It starts only
when a maintainer adds the `run-ci` label to the pull request; opening,
pushing, reopening or marking the pull request ready for review no longer
starts it. It can also be started manually with `workflow_dispatch`. It
explicitly checks out and tests the pull request merge commit and does not run
again merely because the pull request is merged into `main`.

`pr-test-hcu.yml` uses the same event and checkout rules for the minimum
single-node smoke test. Both workflows read repository variables from the
target repository and skip draft pull requests. `run-ci` stays on the pull
request after it dispatches a run, and because neither workflow listens to
pushes, HCU CI runs once per label addition: to start another run, remove the
label and add it again. Changes to either `pull_request_target` definition
take effect only after they are present on the target repository's default
branch.

The workflow starts an isolated local Ray head, runs the existing
`hcu_example/run_qwen3_4b.sh` with one rollout and minimum batch sizes, then
uploads logs and stops Ray. It passes only when all of the following succeed:

1. Required model, data, and runtime roots exist.
2. Ray, PyTorch, and HCU Megatron import successfully and one HCU device is
   visible.
3. The Ray head becomes available.
4. The Qwen3-4B rollout/training command exits with status zero.

The workflow runs serially in an isolated environment and must not share model
outputs or Ray processes with production workloads. Its token has only
`contents: read`; it receives no secrets, write permission, or long-lived
credentials, and checkout does not persist the GitHub token.

Security warning: `pull_request_target` checks out and executes contributor
code on a self-hosted ARC runner pod. A malicious fork can attempt to access
the pod, its HCU devices, mounted runtime files, repository variables, and
network. The pool must therefore stay isolated and disposable, and free of
credentials or unrelated writable mounts. Repository variables used by these
workflows must be treated as public information.

Both checkout steps explicitly opt in to `allow-unsafe-pr-checkout` because
`actions/checkout@v6` otherwise blocks fork code in `pull_request_target`
workflows. This option acknowledges the risk; it does not make contributor
code safe or replace runner isolation.
