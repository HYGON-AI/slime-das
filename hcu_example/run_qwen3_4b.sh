#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

NODE_IP="${NODE_IP:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/model/qwen3/Qwen3-4B}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${SLIME_ROOT}/scripts/models/qwen3-4B.sh}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"
SAVE_ROOT="${SAVE_ROOT:-/home/Download/qwen3/Qwen3-4B_slime}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-4096}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

#mul node training parameters
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
SGLANG_PIPELINE_PARALLEL_SIZE="${SGLANG_PIPELINE_PARALLEL_SIZE:-1}"
SUBMIT_MODE="${SUBMIT_MODE:-job}"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-${NODE_IP}:${RAY_PORT}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
RESUME=0

usage() {
  cat <<'EOF'
Usage: run_qwen3_4b.sh [options]

Options:
  --node-ip IP       Ray head IP (default: 12.12.12.48)
  --model-path PATH  Hugging Face model directory
  --model-args PATH  Model args script, e.g. scripts/models/qwen3.5-4B.sh
  --data-root PATH   Parent directory of dapo-math-17k and aime-2024
  --save-root PATH   Checkpoint output directory
  --resume           Resume from SAVE_ROOT/latest_checkpointed_iteration.txt
  -h, --help         Show this help

Multi-node resource knobs are controlled by environment variables:
  ACTOR_NUM_NODES              default: 1
  ACTOR_NUM_GPUS_PER_NODE      default: 4
  ROLLOUT_NUM_GPUS             default: 4
  ROLLOUT_NUM_GPUS_PER_ENGINE  default: 4

Submission mode:
  SUBMIT_MODE=job     submit through Ray dashboard job API (default)
  SUBMIT_MODE=direct  run python directly on the head node and connect to Ray
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
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in \
  "${MODEL_PATH}/config.json" \
  "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
  "${DATA_ROOT}/aime-2024/aime-2024.jsonl" \
  "${MODEL_ARGS_SCRIPT}" \
  "${SCRIPT_DIR}/env.yaml"; do
  [[ -e "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

curl --fail --silent "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null || {
  echo "Ray dashboard is not available. Run ${SCRIPT_DIR}/start_ray.sh ${NODE_IP} first." >&2
  exit 1
}

# shellcheck source=/dev/null
source "${MODEL_ARGS_SCRIPT}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --megatron-to-hf-mode bridge
  --ref-load "${MODEL_PATH}"
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
  --rollout-temperature 1
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
  #--rollout-num-gpus-per-engine 4
  #--sglang-pipeline-parallel-size 1
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-pipeline-parallel-size "${SGLANG_PIPELINE_PARALLEL_SIZE}"
  --sglang-mem-fraction-static 0.6
  --sglang-kv-cache-dtype fp8_e5m2
  --sglang-disable-radix-cache
  --sglang-dtype bfloat16
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

mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

MODEL_NAME="$(basename "${MODEL_PATH}")"
TIME_STAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-${SUBMIT_MODE}-node${ACTOR_NUM_NODES}-rollout${ROLLOUT_NUM_GPUS}-${TIME_STAMP}.log"


echo "Submitting Qwen3-4B job to http://${NODE_IP}:${RAY_DASHBOARD_PORT}"
echo "Model: ${MODEL_PATH}"
echo "Model args: ${MODEL_ARGS_SCRIPT}"
echo "Data:  ${DATA_ROOT}"
echo "Save:  ${SAVE_ROOT}"
echo "Log:   ${LOG_FILE}"
echo "Resume: ${RESUME}"
echo "Actor nodes: ${ACTOR_NUM_NODES}"
echo "Actor GPUs per node: ${ACTOR_NUM_GPUS_PER_NODE}"
echo "Rollout GPUs: ${ROLLOUT_NUM_GPUS}"
echo "Rollout GPUs per engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "Submit mode: ${SUBMIT_MODE}"

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
