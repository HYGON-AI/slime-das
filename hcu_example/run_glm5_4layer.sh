#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# HCU slime smoke test for GLM-5 sliced to 4 layers.
#
# The SGLang rollout options are aligned with the standalone GLM-5 SGLang
# command that has been verified on the target environment. This script is for
# functional smoke testing first, not for final performance tuning.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

NODE_IP="${NODE_IP:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/home/Download/glm5/GLM-5-4layers}"
#TORCH_DIST_PATH="${TORCH_DIST_PATH:-/home/Download/glm5/GLM-5-4layers_torch_dist}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"
SAVE_ROOT="${SAVE_ROOT:-/home/Download/glm5/GLM-5-4layers_slime}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

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

# SGLang rollout parameters copied from the verified standalone GLM-5 test.
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-6000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-256}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-10240}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_NSA_PREFILL_BACKEND="${SGLANG_NSA_PREFILL_BACKEND:-flashmla_auto}"
SGLANG_NSA_DECODE_BACKEND="${SGLANG_NSA_DECODE_BACKEND:-flashmla_kv}"
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-1}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"

ENABLE_TORCH_PROFILE="${ENABLE_TORCH_PROFILE:-0}"
PROFILE_TARGET="${PROFILE_TARGET:-train_overall}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
PROFILE_STEP_END="${PROFILE_STEP_END:-4}"
PROFILE_DIR="${PROFILE_DIR:-/home/profile/glm5_4layer_$(date '+%Y%m%d_%H%M%S')}"

RESUME=0
CLEANUP=0

usage() {
  cat <<'EOF'
Usage: run_glm5_4layer.sh [options]

Options:
  --node-ip IP        Ray head IP
  --model-path PATH   GLM-5 4-layer HuggingFace checkpoint directory
  --data-root PATH    Parent directory of dapo-math-17k and aime-2024
  --save-root PATH    Output checkpoint directory
  --resume            Resume from SAVE_ROOT/latest_checkpointed_iteration.txt
  --cleanup           Kill sglang/ray/python after the job exits
  -h, --help          Show this help

Common env knobs:
  SUBMIT_MODE=direct|job
  ACTOR_NUM_GPUS_PER_NODE=4
  TENSOR_MODEL_PARALLEL_SIZE=4
  EXPERT_MODEL_PARALLEL_SIZE=1
  ROLLOUT_NUM_GPUS=4
  ROLLOUT_NUM_GPUS_PER_ENGINE=4
  SGLANG_MEM_FRACTION_STATIC=0.75
  SGLANG_KV_CACHE_DTYPE=fp8_e4m3
  ENABLE_TORCH_PROFILE=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ip) NODE_IP="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
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

curl --fail --silent "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null || {
  echo "Ray dashboard is not available. Run ${SCRIPT_DIR}/start_ray.sh ${NODE_IP} first." >&2
  exit 1
}

# SGLang tuning copied from the verified standalone GLM-5 launch script.
export USE_HCU_CUSTOM_ALLREDUCE="${USE_HCU_CUSTOM_ALLREDUCE:-1}"
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGL_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/home/profile}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-1}"
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS="${SGLANG_ASSIGN_EXTEND_CACHE_LOCS:-1}"
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL="${SGLANG_ASSIGN_REQ_TO_TOKEN_POOL:-1}"
export SGLANG_GET_LAST_LOC="${SGLANG_GET_LAST_LOC:-1}"
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON="${SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON:-1}"
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES="${SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES:-1}"
export HIP_GRAPH_ACCUMULATE_DISPATCH="${HIP_GRAPH_ACCUMULATE_DISPATCH:-0}"
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"
export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"
export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export SGLANG_USE_FP8_W8A8_MOE="${SGLANG_USE_FP8_W8A8_MOE:-1}"

if command -v sysctl >/dev/null 2>&1; then
  sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true
fi

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --megatron-to-hf-mode bridge
  --load "${MODEL_PATH}"
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

# GLM-5 4-layer model args. The sliced model keeps the first 3 dense layers and
# the following 1 MoE layer, matching config moe_layer_freq=[0,0,0,1].
MODEL_ARGS=(
  --spec "slime_plugins.models.glm5.glm5" "get_glm5_spec"
  --moe-layer-freq "[0]*3+[1]*1"
  --num-experts 256
  --moe-shared-expert-intermediate-size 2048
  --moe-router-topk 8
  --moe-grouped-gemm
  --moe-permute-fusion
  --moe-ffn-hidden-size 2048
  --moe-router-score-function sigmoid
  --moe-router-pre-softmax
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk-scaling-factor 2.5
  --moe-aux-loss-coeff 0
  --moe-router-dtype fp32
  --make-vocab-size-divisible-by 16
  --num-layers 4
  --hidden-size 6144
  --ffn-hidden-size 12288
  --num-attention-heads 64
  --disable-bias-linear
  --swiglu
  --untie-embeddings-and-output-weights
  --position-embedding-type rope
  --no-position-embedding
  --normalization RMSNorm
  --qk-layernorm
  --multi-latent-attention
  --q-lora-rank 2048
  --kv-lora-rank 512
  --qk-head-dim 256
  --v-head-dim 256
  --kv-channels 192
  --qk-pos-emb-head-dim 64
  --vocab-size 154880
  --rotary-base 1000000
  --enable-experimental
  #--allgather-cp
)

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
  #--use-dynamic-batch-size
  # --calculate-per-token-loss
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
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
  --sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}"
  --sglang-page-size "${SGLANG_PAGE_SIZE}"
  --sglang-nsa-prefill-backend "${SGLANG_NSA_PREFILL_BACKEND}"
  --sglang-nsa-decode-backend "${SGLANG_NSA_DECODE_BACKEND}"
)

if [[ "${SGLANG_DISABLE_RADIX_CACHE}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-disable-radix-cache)
fi

if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  #--moe-token-dispatcher-type allgather
  --moe-token-dispatcher-type alltoall
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
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-glm5-${SUBMIT_MODE}-node${ACTOR_NUM_NODES}-rollout${ROLLOUT_NUM_GPUS}-${TIME_STAMP}.log"

echo "Submitting GLM-5 4-layer job to http://${NODE_IP}:${RAY_DASHBOARD_PORT}"
echo "Model: ${MODEL_PATH}"
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
