#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# HCU slime smoke test for DeepSeek-R1 sliced to 4 layers.
#
# This script intentionally does NOT pass --sglang-quantization by default.
# Use it to test DeepSeek-R1 4-layer MoE with a normal SGLang rollout weight
# layout, avoiding FP8/INT8 online update_weight layout mismatch first.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

NODE_IP="${NODE_IP:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/home/Download/deepseek/DeepSeek-R1-4layers}"
TORCH_DIST_PATH="${TORCH_DIST_PATH:-/home/Download/deepseek/DeepSeek-R1-4layers_torch_dist}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"
SAVE_ROOT="${SAVE_ROOT:-/home/Download/deepseek/DeepSeek-R1-4layers_slime}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${SLIME_ROOT}/scripts/models/deepseek-v3.sh}"
NUM_LAYERS="${NUM_LAYERS:-4}"
LOAD_MODE="${LOAD_MODE:-bridge}"

SUBMIT_MODE="${SUBMIT_MODE:-direct}"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-${NODE_IP}:${RAY_PORT}}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
SGLANG_PIPELINE_PARALLEL_SIZE="${SGLANG_PIPELINE_PARALLEL_SIZE:-1}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-4}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-64}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-64}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-512}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"

# SGLang rollout defaults. Quantization is intentionally empty by default.
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-1024}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.45}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-hcu_mla}"
SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-}"
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-1}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:--1}"

ENABLE_TORCH_PROFILE="${ENABLE_TORCH_PROFILE:-0}"
PROFILE_TARGET="${PROFILE_TARGET:-train_overall}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
PROFILE_STEP_END="${PROFILE_STEP_END:-4}"
PROFILE_DIR="${PROFILE_DIR:-/home/profile/deepseek_r1_4layer_no_sglang_quant_$(date '+%Y%m%d_%H%M%S')}"

RESUME=0
CLEANUP=0

usage() {
  cat <<'EOF'
Usage: run_deepseek_r1_4layer_no_sglang_quant.sh [options]

Options:
  --node-ip IP             Ray head IP
  --model-path PATH        Sliced HuggingFace checkpoint directory
  --torch-dist-path PATH   Converted torch_dist checkpoint directory
  --data-root PATH         Parent directory of dapo-math-17k and aime-2024
  --save-root PATH         Output checkpoint directory
  --resume                 Resume from SAVE_ROOT
  --cleanup                Kill sglang/ray/python after the job exits
  -h, --help               Show this help

Common env knobs:
  LOAD_MODE=bridge|torch_dist             default: bridge
  NUM_LAYERS=4                            default: 4
  MODEL_ARGS_SCRIPT=/path/deepseek-v3.sh  default: ${SLIME_ROOT}/scripts/models/deepseek-v3.sh
  SUBMIT_MODE=direct|job                  default: direct
  ACTOR_NUM_GPUS_PER_NODE=4
  TENSOR_MODEL_PARALLEL_SIZE=4
  ROLLOUT_NUM_GPUS=4
  ROLLOUT_NUM_GPUS_PER_ENGINE=4
  SGLANG_QUANTIZATION=                   default: empty, no --sglang-quantization
  SGLANG_KV_CACHE_DTYPE=                 default: empty, no --sglang-kv-cache-dtype
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

for required in \
  "${MODEL_PATH}/config.json" \
  "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
  "${DATA_ROOT}/aime-2024/aime-2024.jsonl" \
  "${MODEL_ARGS_SCRIPT}" \
  "${SCRIPT_DIR}/env.yaml"; do
  [[ -e "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

if [[ "${LOAD_MODE}" == "torch_dist" ]]; then
  [[ -f "${TORCH_DIST_PATH}/latest_checkpointed_iteration.txt" ]] || {
    echo "Missing torch_dist checkpoint: ${TORCH_DIST_PATH}/latest_checkpointed_iteration.txt" >&2
    exit 1
  }
elif [[ "${LOAD_MODE}" != "bridge" ]]; then
  echo "Unknown LOAD_MODE: ${LOAD_MODE}. Use bridge or torch_dist." >&2
  exit 2
fi

curl --fail --silent "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null || {
  echo "Ray dashboard is not available. Run ${SCRIPT_DIR}/start_ray.sh ${NODE_IP} first." >&2
  exit 1
}

LOAD_PATH=""
if [[ "${RESUME}" == 1 ]]; then
  [[ -f "${SAVE_ROOT}/latest_checkpointed_iteration.txt" ]] || {
    echo "Cannot resume: ${SAVE_ROOT}/latest_checkpointed_iteration.txt does not exist." >&2
    exit 1
  }
  LOAD_PATH="${SAVE_ROOT}"
elif [[ "${LOAD_MODE}" == "torch_dist" ]]; then
  LOAD_PATH="${TORCH_DIST_PATH}"
fi

# shellcheck disable=SC1090
MODEL_ARGS_NUM_LAYERS="${NUM_LAYERS}" source "${MODEL_ARGS_SCRIPT}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --save "${SAVE_ROOT}"
  --save-interval 20
)

if [[ -n "${LOAD_PATH}" ]]; then
  CKPT_ARGS+=(--load "${LOAD_PATH}")
fi

if [[ "${LOAD_MODE}" == "bridge" ]]; then
  CKPT_ARGS+=(
    --megatron-to-hf-mode bridge
    --ref-load "${MODEL_PATH}"
  )
else
  CKPT_ARGS+=(
    --ref-load "${TORCH_DIST_PATH}"
  )
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
  --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-pipeline-parallel-size "${SGLANG_PIPELINE_PARALLEL_SIZE}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sglang-page-size "${SGLANG_PAGE_SIZE}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
  --sglang-disable-cuda-graph
)

if [[ -n "${SGLANG_KV_CACHE_DTYPE}" ]]; then
  SGLANG_ARGS+=(--sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}")
fi

if [[ -n "${SGLANG_QUANTIZATION}" ]]; then
  SGLANG_ARGS+=(--sglang-quantization "${SGLANG_QUANTIZATION}")
fi

if [[ "${SGLANG_DISABLE_RADIX_CACHE}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-disable-radix-cache)
fi

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
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-no-sglang-quant-${SUBMIT_MODE}-node${ACTOR_NUM_NODES}-rollout${ROLLOUT_NUM_GPUS}-${TIME_STAMP}.log"

echo "Submitting DeepSeek-R1 ${NUM_LAYERS}-layer no-SGLang-quant job to http://${NODE_IP}:${RAY_DASHBOARD_PORT}"
echo "Model:       ${MODEL_PATH}"
echo "Model args:  ${MODEL_ARGS_SCRIPT}"
echo "Load mode:   ${LOAD_MODE}"
echo "Load:        ${LOAD_PATH}"
echo "Data:        ${DATA_ROOT}"
echo "Save:        ${SAVE_ROOT}"
echo "Log:         ${LOG_FILE}"
echo "Actor GPUs per node: ${ACTOR_NUM_GPUS_PER_NODE}"
echo "Rollout GPUs: ${ROLLOUT_NUM_GPUS}"
echo "SGLang quantization: ${SGLANG_QUANTIZATION:-<disabled>}"
echo "SGLang KV cache dtype: ${SGLANG_KV_CACHE_DTYPE:-<default>}"
echo "Submit mode: ${SUBMIT_MODE}"
echo "Torch profiler: ${ENABLE_TORCH_PROFILE}"

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
