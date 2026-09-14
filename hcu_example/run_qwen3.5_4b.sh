#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# HCU Qwen3.5-4B training example.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

NODE_IP="${NODE_IP:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-63888}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-${NODE_IP}:${RAY_PORT}}"
SUBMIT_MODE="${SUBMIT_MODE:-direct}"

MODEL_PATH="${MODEL_PATH:-/model/qwen3.5/Qwen3.5-4B}"
TORCH_DIST_PATH="${TORCH_DIST_PATH:-/home/Download/qwen3.5/Qwen3.5-4B_torch_dist}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"
SAVE_ROOT="${SAVE_ROOT:-/home/Download/qwen3.5/Qwen3.5-4B_slime_installed}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
SGLANG_PIPELINE_PARALLEL_SIZE="${SGLANG_PIPELINE_PARALLEL_SIZE:-1}"

NUM_ROLLOUT="${NUM_ROLLOUT:-16}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-512}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"

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
Usage: run_qwen3.5_4b.sh [options]

Options:
  --node-ip IP       Ray head IP, default 127.0.0.1
  --model-path PATH  Hugging Face Qwen3.5-4B directory
  --data-root PATH   Parent directory of dapo-math-17k and aime-2024
  --save-root PATH   Checkpoint output directory
  --resume           Resume from SAVE_ROOT/latest_checkpointed_iteration.txt
  --cleanup          Kill sglang/ray/python after the job exits
  -h, --help         Show this help

Example:
  cd <path-to-slime-das>
  MODEL_PATH=/model/qwen3.5/Qwen3.5-4B \
  TORCH_DIST_PATH=/home/Download/qwen3.5/Qwen3.5-4B_torch_dist \
  RAY_HEAD_ADDRESS=127.0.0.1:63888 \
  bash <path-to-slime-das>/hcu_example/run_qwen3.5_4b.sh --node-ip 127.0.0.1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ip) NODE_IP="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --torch-dist-path) TORCH_DIST_PATH="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --cleanup) CLEANUP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${SUBMIT_MODE}" != "direct" ]]; then
  echo "This script only supports SUBMIT_MODE=direct." >&2
  exit 2
fi

for required in \
  "${MODEL_PATH}/config.json" \
  "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
  "${DATA_ROOT}/aime-2024/aime-2024.jsonl"; do
  [[ -e "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done


curl --fail --silent "http://${NODE_IP}:${RAY_DASHBOARD_PORT}/api/version" >/dev/null || {
  echo "Ray dashboard is not available at http://${NODE_IP}:${RAY_DASHBOARD_PORT}." >&2
  echo "Start Ray first, for example: ray start --head --node-ip-address=127.0.0.1 --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --num-gpus=8 --disable-usage-stats" >&2
  exit 1
}

MODEL_ARGS=(
  --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --num-attention-heads 16
  --num-query-groups 4
  --kv-channels 256
  --num-layers 32
  --hidden-size 2560
  --ffn-hidden-size 9216
  --use-gated-attention
  --normalization RMSNorm
  --apply-layernorm-1p
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --rotary-percent 0.25
  --swiglu
  --vocab-size 248320
  --rotary-base 10000000
  --attention-output-gate
)

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --load "${TORCH_DIST_PATH}"
  --ref-load "${TORCH_DIST_PATH}"
  --save "${SAVE_ROOT}"
  --save-interval 20
)

if [[ "${RESUME}" == 1 ]]; then
  [[ -f "${SAVE_ROOT}/latest_checkpointed_iteration.txt" ]] || {
    echo "Cannot resume: ${SAVE_ROOT}/latest_checkpointed_iteration.txt does not exist." >&2
    exit 1
  }
  CKPT_ARGS+=(--load "${SAVE_ROOT}")
fi

ROLLOUT_ARGS=(
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
  --rollout-temperature 1.0
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

EVAL_ARGS=(
  --eval-interval 20
  --eval-prompt-data aime "${DATA_ROOT}/aime-2024/aime-2024.jsonl"
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --eval-top-p 1
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --calculate-per-token-loss
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
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
  --use-distributed-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-pipeline-parallel-size "${SGLANG_PIPELINE_PARALLEL_SIZE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-attention-backend fa3
  --sglang-page-size 64
  --sglang-disable-radix-cache
  --sglang-disable-custom-all-reduce
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
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-${SUBMIT_MODE}-node${ACTOR_NUM_NODES}-rollout${ROLLOUT_NUM_GPUS}-${TIME_STAMP}.log"

echo "Submitting Qwen3.5-4B job"
echo "Model:      ${MODEL_PATH}"
echo "Data:       ${DATA_ROOT}"
echo "Save:       ${SAVE_ROOT}"
echo "Log:        ${LOG_FILE}"
echo "Ray:        ${RAY_HEAD_ADDRESS}"
echo "PYTHONPATH: ${PYTHONPATH:-<unset>}"

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train.py"
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${PROFILE_ARGS[@]}"
)

export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
echo "Direct Ray address: ${RAY_ADDRESS}"
"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

if [[ "${CLEANUP}" == 1 ]]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
fi
