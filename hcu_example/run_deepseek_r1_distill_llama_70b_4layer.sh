#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# DeepSeek-R1-Distill-Llama-70B RL smoke test for HCU slime.
#
# LOAD_MODE=bridge     : load HuggingFace weights through Megatron-Bridge at runtime.
# LOAD_MODE=torch_dist : load a pre-converted torch_dist checkpoint.
#
# NUM_LAYERS controls the Megatron model depth. For a true small-layer smoke test,
# prefer pointing MODEL_PATH to a sliced HF checkpoint with the same number of layers.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

NODE_IP="${NODE_IP:-127.0.0.1}"

MODEL_PATH="${MODEL_PATH:-/home/Download/deepseek-r1/DeepSeek-R1-Distill-Llama-70B-4layers}"
TORCH_DIST_PATH="${TORCH_DIST_PATH:-/home/Download/deepseek-r1/DeepSeek-R1-Distill-Llama-70B-4layers_torch_dist}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"
SAVE_ROOT="${SAVE_ROOT:-/home/Download/deepseek-r1/DeepSeek-R1-Distill-Llama-70B-4layers_slime}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

LOAD_MODE="${LOAD_MODE:-bridge}"
NUM_LAYERS="${NUM_LAYERS:-4}"

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

NUM_ROLLOUT="${NUM_ROLLOUT:-64}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"

SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-bfloat16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-1}"

ENABLE_TORCH_PROFILE="${ENABLE_TORCH_PROFILE:-0}"
PROFILE_TARGET="${PROFILE_TARGET:-train_overall}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
PROFILE_STEP_END="${PROFILE_STEP_END:-4}"
PROFILE_DIR="${PROFILE_DIR:-/home/profile/deepseek_r1_distill_llama_70b_4layer_$(date '+%Y%m%d_%H%M%S')}"

RESUME=0
CLEANUP=0

usage() {
  cat <<'EOF'
Usage: run_deepseek_r1_distill_llama_70b_4layer.sh [options]

Options:
  --node-ip IP             Ray head IP
  --model-path PATH        4-layer Hugging Face checkpoint directory
  --torch-dist-path PATH   Converted torch_dist checkpoint directory
  --data-root PATH         Parent directory of dapo-math-17k and aime-2024
  --save-root PATH         Output checkpoint directory
  --resume                 Resume from SAVE_ROOT instead of initial torch_dist
  --cleanup                Kill sglang/ray/python after the job exits
  -h, --help               Show this help

Common env knobs:
  LOAD_MODE=bridge|torch_dist
  NUM_LAYERS=4
  SUBMIT_MODE=direct|job
  ACTOR_NUM_GPUS_PER_NODE=4
  TENSOR_MODEL_PARALLEL_SIZE=4
  ROLLOUT_NUM_GPUS=4
  ROLLOUT_NUM_GPUS_PER_ENGINE=4
  SGLANG_ATTENTION_BACKEND=fa3
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

MODEL_ARGS=(
  --num-layers "${NUM_LAYERS}"
  --hidden-size 8192
  --ffn-hidden-size 28672
  --num-attention-heads 64
  --group-query-attention
  --num-query-groups 8
  --kv-channels 128
  --seq-length 4096
  --max-position-embeddings 131072
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${MODEL_PATH}"
  --bf16
  --disable-bias-linear
  --normalization RMSNorm
  --norm-epsilon 1e-05
  --position-embedding-type rope
  --rotary-base 500000
  --swiglu
  --untie-embeddings-and-output-weights
  --transformer-impl transformer_engine
  --no-gradient-accumulation-fusion
)

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
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
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
  --weight-decay 0.01
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
  --sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sglang-page-size 64
  --sglang-disable-cuda-graph
)

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
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-${SUBMIT_MODE}-node${ACTOR_NUM_NODES}-rollout${ROLLOUT_NUM_GPUS}-${TIME_STAMP}.log"

echo "Submitting DeepSeek-R1-Distill-Llama-70B job to http://${NODE_IP}:${RAY_DASHBOARD_PORT}"
echo "Model:       ${MODEL_PATH}"
echo "Torch dist:  ${TORCH_DIST_PATH}"
echo "Load mode:   ${LOAD_MODE}"
echo "Num layers:  ${NUM_LAYERS}"
echo "Load:        ${LOAD_PATH}"
echo "Data:        ${DATA_ROOT}"
echo "Save:        ${SAVE_ROOT}"
echo "Log:         ${LOG_FILE}"
echo "Resume:      ${RESUME}"
echo "Actor nodes: ${ACTOR_NUM_NODES}"
echo "Actor GPUs per node: ${ACTOR_NUM_GPUS_PER_NODE}"
echo "Rollout GPUs: ${ROLLOUT_NUM_GPUS}"
echo "Rollout GPUs per engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "Submit mode: ${SUBMIT_MODE}"
echo "Torch profiler: ${ENABLE_TORCH_PROFILE}"

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
