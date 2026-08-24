#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

usage() {
  cat <<EOF
Usage:
  bash run_glm4.6_4layer.sh --node-ip <dashboard_ip>

Environment variables:
  MODEL_PATH                 HF model path, default: /home/Download/glm4.6/GLM-4.6-4layers
  TORCH_DIST_PATH            torch_dist path, default: /home/Download/glm4.6/GLM-4.6-4layers_torch_dist
  SAVE_ROOT                  Slime checkpoint save path
  DATA_ROOT                  Dataset root, default: /home/Download

  SUBMIT_MODE                direct or job, default: direct
  RAY_HEAD_ADDRESS           Ray head address for direct mode, default: 127.0.0.1:63792

  ACTOR_NUM_NODES            default: 1
  ACTOR_NUM_GPUS_PER_NODE    default: 4
  TENSOR_MODEL_PARALLEL_SIZE default: 4
  EXPERT_MODEL_PARALLEL_SIZE default: 1

  ROLLOUT_NUM_GPUS           default: 4
  ROLLOUT_NUM_GPUS_PER_ENGINE default: 4

  ENABLE_TORCH_PROFILE       0/1, default: 0
EOF
}

NODE_IP="${NODE_IP:-127.0.0.1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ip)
      NODE_IP="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

MODEL_PATH="${MODEL_PATH:-/home/Download/glm4.6/GLM-4.6-4layers}"
TORCH_DIST_PATH="${TORCH_DIST_PATH:-/home/Download/glm4.6/GLM-4.6-4layers_torch_dist}"
SAVE_ROOT="${SAVE_ROOT:-/home/Download/glm4.6/GLM-4.6-4layers_slime}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"

SUBMIT_MODE="${SUBMIT_MODE:-direct}"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-127.0.0.1:63792}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"

ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"

NUM_ROLLOUT="${NUM_ROLLOUT:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"

ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-512}"

SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"

SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
RESUME="${RESUME:-0}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/GLM4.6-4layers-${SUBMIT_MODE}-node${ACTOR_NUM_NODES}-rollout${ROLLOUT_NUM_GPUS}-${TIMESTAMP}.log}"

PROFILE_TARGET="${PROFILE_TARGET:-train_overall}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
PROFILE_STEP_END="${PROFILE_STEP_END:-4}"
PROFILE_DIR="${PROFILE_DIR:-/home/profile/slime_${PROFILE_TARGET}_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting GLM4.6-4layers job to http://${NODE_IP}:8265"
echo "Model:      ${MODEL_PATH}"
echo "TorchDist:  ${TORCH_DIST_PATH}"
echo "Save:       ${SAVE_ROOT}"
echo "Log:        ${LOG_FILE}"
echo "Resume:     ${RESUME}"
echo "Actor:      nodes=${ACTOR_NUM_NODES}, gpus_per_node=${ACTOR_NUM_GPUS_PER_NODE}"
echo "Rollout:    gpus=${ROLLOUT_NUM_GPUS}, gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "Submit mode:${SUBMIT_MODE}"

N_DENSE_LAYERS=3
N_MOE_LAYERS=1

MODEL_ARGS=(
  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --num-attention-heads 96
  --num-query-groups 8
  --kv-channels 128

  --num-layers 4
  --hidden-size 5120
  --ffn-hidden-size 12288

  --add-qkv-bias
  --normalization RMSNorm
  --norm-epsilon 1e-5
  --position-embedding-type rope
  --rotary-percent 0.5
  --rotary-base 1000000
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 151552
  --make-vocab-size-divisible-by 128

  --moe-ffn-hidden-size 1536
  --moe-shared-expert-intermediate-size 1536
  --moe-router-pre-softmax
  --moe-router-score-function sigmoid
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0
  --moe-router-load-balancing-type seq_aux_loss
  --moe-token-dispatcher-type alltoall
  --moe-router-topk 8
  --moe-router-topk-scaling-factor 2.5
  --moe-layer-freq "[0]*${N_DENSE_LAYERS}+[1]*${N_MOE_LAYERS}"
  --num-experts 160
  --moe-grouped-gemm
  --moe-router-dtype fp32
  --moe-permute-fusion
  --moe-aux-loss-coeff 0
)

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  # --ref-load "${TORCH_DIST_PATH}"
  # --load "${TORCH_DIST_PATH}"
  --load "${MODEL_PATH}"
  --megatron-to-hf-mode bridge
  --ref-load "${MODEL_PATH}"
  --save "${SAVE_ROOT}"
  --save-interval "${SAVE_INTERVAL}"
  #--megatron-to-hf-mode raw
)

if [[ "${RESUME}" == "1" ]]; then
  CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${TORCH_DIST_PATH}"
    --load "${SAVE_ROOT}"
    --save "${SAVE_ROOT}"
    --save-interval "${SAVE_INTERVAL}"
    --megatron-to-hf-mode raw
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
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
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

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-pipeline-parallel-size 1
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sglang-page-size "${SGLANG_PAGE_SIZE}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
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

PROFILE_ARGS=(
  --use-pytorch-profiler
  --profile-target "${PROFILE_TARGET}"
  --profile-ranks "${PROFILE_RANKS}"
  --profile-step-start "${PROFILE_STEP_START}"
  --profile-step-end "${PROFILE_STEP_END}"
  --tensorboard-dir "${PROFILE_DIR}"
)

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

if [[ "${ENABLE_TORCH_PROFILE:-0}" == "1" ]]; then
  TRAIN_CMD+=("${PROFILE_ARGS[@]}")
fi

if [[ "${SUBMIT_MODE}" == "direct" ]]; then
  export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
  "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
elif [[ "${SUBMIT_MODE}" == "job" ]]; then
  ray job submit \
    --address "http://${NODE_IP}:8265" \
    --runtime-env "${SCRIPT_DIR}/env.yaml" \
    -- "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  echo "Invalid SUBMIT_MODE=${SUBMIT_MODE}, expected direct or job" >&2
  exit 1
fi
