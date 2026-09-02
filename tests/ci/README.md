# Doc about CI

## Configure GitHub secrets

https://github.com/THUDM/slime/settings/secrets/actions

* `WANDB_API_KEY`: get from https://wandb.ai/authorize

## Setup new GitHub runners

### Step 1: Env

Write `.env` mimicking `.env.example`.
The token can be found at https://github.com/THUDM/slime/settings/actions/runners/new?arch=x64&os=linux.

WARN: The `GITHUB_RUNNER_TOKEN` changes after a while.

### Step 2: Prepare runner externals

```shell
cd "$(git rev-parse --show-toplevel)/tests/ci/github_runner"
cp .env.example .env
mkdir -p ./runner-externals ./runner-work
docker create --name temp-runner ghcr.io/actions/actions-runner:2.329.0
docker cp temp-runner:/home/runner/externals/. ./runner-externals/
docker rm -f temp-runner
ls -alh ./runner-externals
```

### Step 3: Run

```shell
cd "$(git rev-parse --show-toplevel)/tests/ci/github_runner"
docker compose up -d
```

### Debugging

Logs

```shell
# All containers
docker compose logs -f

# One container
docker logs -f github_runner-runner-1
```

Exec

```shell
docker exec -it github_runner-runner-1 /bin/bash
```

An example of quickly iterate

```shell
docker compose down -v && docker compose up -d && docker logs -f github_runner-runner-1
```
