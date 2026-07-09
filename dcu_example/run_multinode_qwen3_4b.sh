#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# This script only submits the training job. Start Ray head and workers first.
#
# Example for 2 nodes x 8 GPUs:
#   Head node:
#     bash start_ray.sh 12.12.12.48
#
#   Worker node:
#     bash start_ray_worker.sh 12.12.12.48 12.12.12.49
#
#   Head node submit:
#     NODE_IP=12.12.12.48 bash run_multinode_qwen3_4b.sh

export NODE_IP="${NODE_IP:-127.0.0.1}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
export SUBMIT_MODE="${SUBMIT_MODE:-direct}"
export RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-12.12.12.48:63792}"

bash "${SCRIPT_DIR}/run_qwen3_4b.sh" --node-ip "${NODE_IP}" "$@"

