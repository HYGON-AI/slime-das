#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# HCU fully-async rollout smoke test for Qwen3-4B.
#
# Difference from normal run_qwen3_4b.sh:
#   1. Use train_async.py instead of train.py.
#   2. Add --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async.
#   3. Do not pass eval args, because fully-async rollout keeps a continuous
#      background generation queue and currently conflicts with eval mode.
#   4. Actor and rollout must use disjoint GPUs; colocation is not supported.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

NODE_IP="${NODE_IP:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B-Thinking-2507}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${SLIME_ROOT}/scripts/models/qwen3-4B.sh}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"
SAVE_ROOT="${SAVE_ROOT:-/home/Download/qwen3/Qwen3-4B_fully_async_test}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

# Fully-async smoke defaults follow the official example: short, fast, and
# focused on verifying async rollout/train/update-weight dataflow.
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
SGLANG_PIPELINE_PARALLEL_SIZE="${SGLANG_PIPELINE_PARALLEL_SIZE:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

SAVE_INTERVAL="${SAVE_INTERVAL:-9999}"
SUBMIT_MODE="${SUBMIT_MODE:-direct}"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-${NODE_IP}:${RAY_PORT}}"

ENABLE_TORCH_PROFILE="${ENABLE_TORCH_PROFILE:-0}"
PROFILE_TARGET="${PROFILE_TARGET:-train_overall}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
PROFILE_STEP_END="${PROFILE_STEP_END:-4}"
PROFILE_DIR="${PROFILE_DIR:-/home/profile/qwen3.5-4B_$(date '+%Y%m%d_%H%M%S')}"

RESUME=0
CLEANUP=0

usage() {
  cat <<'EOF'
Usage: run_qwen3_4b_fully_async.sh [options]

Options:
  --node-ip IP       Ray head IP
  --model-path PATH  Hugging Face model directory
  --model-args PATH  Model args script
  --data-root PATH   Parent directory of dapo-math-17k
  --save-root PATH   Checkpoint output directory
  --resume           Resume from SAVE_ROOT/latest_checkpointed_iteration.txt
  --cleanup          Kill sglang/ray/python after the job exits
  -h, --help         Show this help

Common environment knobs:
  SUBMIT_MODE=direct|job
  ACTOR_NUM_GPUS_PER_NODE=4
  ROLLOUT_NUM_GPUS=4
  ROLLOUT_NUM_GPUS_PER_ENGINE=4
  NUM_ROLLOUT=3
  ROLLOUT_BATCH_SIZE=8
  N_SAMPLES_PER_PROMPT=4
  GLOBAL_BATCH_SIZE=32

Success signal:
  grep the log for "fully-async rollout", "actor train", and "Timer actor_train end".
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ip) NODE_IP="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-args) MODEL_ARGS_SCRIPT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --cleanup) CLEANUP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in \
  "${MODEL_PATH}/config.json" \
  "${MODEL_ARGS_SCRIPT}" \
  "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
  "${SCRIPT_DIR}/env.yaml"; do
  [[ -e "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

curl --fail --silent "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null || {
  echo "Ray dashboard is not available. Run ${SCRIPT_DIR}/start_ray.sh ${NODE_IP} first." >&2
  exit 1
}

if [[ "${MODEL_PATH##*/}" == "Qwen3-4B-Thinking-2507" ]]; then
  export MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-5000000}"
fi
# shellcheck disable=SC1090
source "${MODEL_ARGS_SCRIPT}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --megatron-to-hf-mode bridge
  --ref-load "${MODEL_PATH}"
  --save "${SAVE_ROOT}"
  --save-interval "${SAVE_INTERVAL}"
)

if [[ "${RESUME}" == 1 ]]; then
  [[ -f "${SAVE_ROOT}/latest_checkpointed_iteration.txt" ]] || {
    echo "Cannot resume: ${SAVE_ROOT}/latest_checkpointed_iteration.txt does not exist." >&2
    exit 1
  }
  CKPT_ARGS+=(--load "${SAVE_ROOT}")
fi

ROLLOUT_ARGS=(
  --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
  --prompt-data "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type deepscaler
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-pipeline-parallel-size "${SGLANG_PIPELINE_PARALLEL_SIZE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-attention-backend fa3
  --sglang-page-size 64
  --sglang-disable-cuda-graph
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

PROFILE_ARGS=()
if [[ "${ENABLE_TORCH_PROFILE}" == "1" ]]; then
  mkdir -p "${PROFILE_DIR}"
  PROFILE_ARGS=(
    --profile-ranks "${PROFILE_RANKS}"
    --profile-step-start "${PROFILE_STEP_START}"
    --profile-step-end "${PROFILE_STEP_END}"
    --use-pytorch-profiler
    --profile-target "${PROFILE_TARGET}"
    --tensorboard-dir "${PROFILE_DIR}"
  )
fi

mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

MODEL_NAME="$(basename "${MODEL_PATH}")"
TIME_STAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-fully-async-${SUBMIT_MODE}-node${ACTOR_NUM_NODES}-rollout${ROLLOUT_NUM_GPUS}-${TIME_STAMP}.log"

echo "Submitting Qwen3-4B fully-async job to http://${NODE_IP}:${RAY_DASHBOARD_PORT}"
echo "Model:      ${MODEL_PATH}"
echo "Model args: ${MODEL_ARGS_SCRIPT}"
echo "Data:       ${DATA_ROOT}"
echo "Save:       ${SAVE_ROOT}"
echo "Log:        ${LOG_FILE}"
echo "Resume:     ${RESUME}"
echo "Actor nodes: ${ACTOR_NUM_NODES}"
echo "Actor GPUs per node: ${ACTOR_NUM_GPUS_PER_NODE}"
echo "Rollout GPUs: ${ROLLOUT_NUM_GPUS}"
echo "Rollout GPUs per engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "Submit mode: ${SUBMIT_MODE}"

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train_async.py"
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${PROFILE_ARGS[@]}"
)

if [[ "${SUBMIT_MODE}" == "job" ]]; then
  ray job submit \
    --address="http://${NODE_IP}:${RAY_DASHBOARD_PORT}" \
    --runtime-env="${SCRIPT_DIR}/env.yaml" \
    -- "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
elif [[ "${SUBMIT_MODE}" == "direct" ]]; then
  export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
  echo "Direct Ray address: ${RAY_ADDRESS}"
  "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  echo "Unknown SUBMIT_MODE: ${SUBMIT_MODE}. Use job or direct." >&2
  exit 2
fi

if [[ "${CLEANUP}" == 1 ]]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
fi
