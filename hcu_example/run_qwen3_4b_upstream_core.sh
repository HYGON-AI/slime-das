#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# HCU adaptations of selected THUDM/slime v0.3.0 Qwen3-4B integration tests:
#   - test_qwen3_4B_ppo.py
#   - test_qwen3_4B_ppo_disaggregate.py
#   - test_qwen3_4B_ckpt.py
#   - test_qwen3_4B_streaming_partial_rollout.py
#
# The upstream scenarios are preserved, while online downloads, /root paths,
# CUDA Graph, and CUDA-only test-launcher assumptions are replaced with the
# repository's pre-provisioned HCU runtime and read-only model/data assets.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# DTK's env.sh appends to this variable and expects it to be defined even
# when the caller uses `set -u`.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

SCENARIO="${SCENARIO:-}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-}"
CHECKPOINT_OPTIMIZER="${CHECKPOINT_OPTIMIZER:-cpu}"
CHECKPOINT_ASYNC_SAVE="${CHECKPOINT_ASYNC_SAVE:-0}"
NODE_IP="${NODE_IP:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B-Thinking-2507}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${SLIME_ROOT}/scripts/models/qwen3-4B.sh}"
DATA_ROOT="${DATA_ROOT:-/home/Download}"
SUBMIT_MODE="${SUBMIT_MODE:-direct}"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-${NODE_IP}:${RAY_PORT}}"

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

usage() {
  cat <<'EOF'
Usage: run_qwen3_4b_upstream_core.sh --scenario SCENARIO [options]

Scenarios:
  ppo-disaggregate   PPO actor/critic training on four HCU devices and rollout on four
  ppo-colocate       PPO actor/critic and rollout colocated on eight HCU devices
  checkpoint         Save or load a training checkpoint roundtrip
  streaming-partial  Streaming generation abort and partial-sample recycling

Options:
  --checkpoint-mode save|load  Required for the checkpoint scenario
  --checkpoint-optimizer cpu|gpu
                               Optimizer placement for this checkpoint phase
  --checkpoint-async-save      Use Megatron asynchronous save in the save phase
  --node-ip IP                 Ray head IP
  --model-path PATH            Hugging Face Qwen3-4B directory
  --model-args PATH            Qwen3-4B Megatron argument script
  --data-root PATH             Parent directory of dapo-math-17k
  --save-root PATH             Writable checkpoint directory
  --log-dir PATH               Writable log directory
  -h, --help                   Show this help

The caller must start the Ray head before invoking this script.
EOF
}

SAVE_ROOT="${SAVE_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    --checkpoint-mode) CHECKPOINT_MODE="$2"; shift 2 ;;
    --checkpoint-optimizer) CHECKPOINT_OPTIMIZER="$2"; shift 2 ;;
    --checkpoint-async-save) CHECKPOINT_ASYNC_SAVE=1; shift ;;
    --node-ip) NODE_IP="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-args) MODEL_ARGS_SCRIPT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${SCENARIO}" in
  ppo-disaggregate|ppo-colocate)
    NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
    if [[ "${SCENARIO}" == "ppo-colocate" ]]; then
      # Actor, critic, and rollout share all devices.  Dedicated override
      # variables keep the normal 4+4 defaults unchanged.
      ACTOR_NUM_GPUS_PER_NODE="${COLOCATE_ACTOR_NUM_GPUS_PER_NODE:-8}"
      ROLLOUT_NUM_GPUS="${COLOCATE_ROLLOUT_NUM_GPUS:-8}"
      ROLLOUT_NUM_GPUS_PER_ENGINE="${COLOCATE_ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
      SGLANG_MEM_FRACTION_STATIC="${COLOCATE_SGLANG_MEM_FRACTION_STATIC:-0.5}"
    fi
    ;;
  checkpoint)
    case "${CHECKPOINT_MODE}" in
      save) NUM_ROLLOUT="${NUM_ROLLOUT:-1}" ;;
      load) NUM_ROLLOUT="${NUM_ROLLOUT:-2}" ;;
      *) echo "checkpoint requires --checkpoint-mode save or load" >&2; exit 2 ;;
    esac
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-256}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
    ;;
  streaming-partial)
    NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-4}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
    # Keep this long enough for in-flight generations to be aborted.
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
    ;;
  *)
    echo "Unknown or missing scenario: ${SCENARIO:-<empty>}" >&2
    usage >&2
    exit 2
    ;;
esac

case "${CHECKPOINT_OPTIMIZER}" in
  cpu|gpu) ;;
  *) echo "--checkpoint-optimizer must be cpu or gpu" >&2; exit 2 ;;
esac
if [[ "${CHECKPOINT_ASYNC_SAVE}" != "0" && "${CHECKPOINT_ASYNC_SAVE}" != "1" ]]; then
  echo "CHECKPOINT_ASYNC_SAVE must be 0 or 1" >&2
  exit 2
fi
if [[ "${CHECKPOINT_ASYNC_SAVE}" == "1" && ( "${SCENARIO}" != "checkpoint" || "${CHECKPOINT_MODE}" != "save" ) ]]; then
  echo "--checkpoint-async-save is only valid for checkpoint save" >&2
  exit 2
fi

SAVE_ROOT="${SAVE_ROOT:-${SLIME_ROOT}/.hcu-core/${SCENARIO}/checkpoints}"
LOG_DIR="${LOG_DIR:-${SLIME_ROOT}/.hcu-core/${SCENARIO}/logs}"

for required in \
  "${MODEL_PATH}/config.json" \
  "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
  "${MODEL_ARGS_SCRIPT}" \
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
# shellcheck source=/dev/null
source "${MODEL_ARGS_SCRIPT}"
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --megatron-to-hf-mode bridge
  --ref-load "${MODEL_PATH}"
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
  --rollout-temperature 0.8
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
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-precision-aware-optimizer
)
if [[ "${SCENARIO}" != "checkpoint" || "${CHECKPOINT_OPTIMIZER}" == "cpu" ]]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
  )
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-pipeline-parallel-size "${SGLANG_PIPELINE_PARALLEL_SIZE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
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
  --ci-test
  --ci-disable-kl-checker
)

ALGORITHM_ARGS=()
SCENARIO_ARGS=()
case "${SCENARIO}" in
  ppo-disaggregate|ppo-colocate)
    ROLE_CONFIG="${LOG_DIR}/qwen3-4b-ppo-roles.yaml"
    printf '%s\n' \
      'megatron:' \
      '  - name: default' \
      '    role: critic' \
      '    overrides:' \
      '      lr: 1.0e-5' \
      '  - name: default' \
      '    role: actor' \
      '    overrides:' \
      '      lr: 1.0e-6' > "${ROLE_CONFIG}"
    ALGORITHM_ARGS=(
      --advantage-estimator ppo
      --use-kl-loss
      --kl-loss-coef 0.0
      --kl-loss-type k1
      --kl-coef 0.0
      --entropy-coef 0.0
      --eps-clip 4e-4
      --num-critic-only-steps 1
      --normalize-advantages
    )
    # The PyPI torch-memory-saver binary depends on NVIDIA libcuda.so.1.
    # Qwen3-4B actor and critic fit together on each 144 GB HCU, so keep both
    # resident and do not load that CUDA-only offload runtime.
    SCENARIO_ARGS=(
      --megatron-config-path "${ROLE_CONFIG}"
      --no-offload-train
    )
    if [[ "${SCENARIO}" == "ppo-colocate" ]]; then
      # SGLang also remains resident.  Its memory fraction is reduced above so
      # the actor and critic can train on the same HCU devices without either
      # NVIDIA torch-memory-saver or SGLang's CUDA memory-saver path.
      SCENARIO_ARGS+=(
        --colocate
        --no-offload-rollout
      )
    fi
    ;;
  checkpoint)
    ALGORITHM_ARGS=(
      --advantage-estimator grpo
      --kl-loss-coef 0.0
      --kl-loss-type k1
      --kl-coef 0.0
      --entropy-coef 0.0
      --eps-clip 0.2
    )
    if [[ "${CHECKPOINT_MODE}" == "save" ]]; then
      CKPT_ARGS+=(--save "${SAVE_ROOT}" --save-interval 1)
      if [[ "${CHECKPOINT_ASYNC_SAVE}" == "1" ]]; then
        # Megatron disables --async-save unless its long-lived checkpoint
        # worker is enabled, which would silently turn this scenario into a
        # synchronous save test.
        # Select Megatron's built-in async writer explicitly.  Its default
        # persistent-worker backend is nvrx, which requires the NVIDIA-only
        # nvidia-resiliency-ext package and is not suitable for HCU runners.
        CKPT_ARGS+=(--async-save --use-persistent-ckpt-worker --async-strategy mcore)
      fi
    else
      [[ -f "${SAVE_ROOT}/latest_checkpointed_iteration.txt" ]] || {
        echo "Checkpoint save phase did not produce latest_checkpointed_iteration.txt" >&2
        exit 1
      }
      CKPT_ARGS+=(--load "${SAVE_ROOT}" --save "${SAVE_ROOT}" --save-interval 1)
      # The resume phase deliberately extends num-rollout from 1 to 2. Keep
      # the restored optimizer state but let Megatron rebuild the scheduler's
      # total-step limit from the current run.
      CKPT_ARGS+=(--override-opt-param-scheduler)
    fi
    ;;
  streaming-partial)
    ALGORITHM_ARGS=(
      --advantage-estimator grpo
      --use-kl-loss
      --kl-loss-coef 0.0
      --kl-loss-type low_var_kl
      --entropy-coef 0.0
      --eps-clip 0.2
      --eps-clip-high 0.28
    )
    SCENARIO_ARGS=(
      --custom-generate-function-path slime.rollout.sglang_streaming_rollout.generate_streaming
      --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
      --partial-rollout
      --mask-offpolicy-in-partial-rollout
    )
    ;;
esac

MODEL_NAME="$(basename "${MODEL_PATH}")"
TIME_STAMP="$(date '+%Y%m%d-%H%M%S')"
MODE_SUFFIX="${CHECKPOINT_MODE:+-${CHECKPOINT_MODE}}"
if [[ "${SCENARIO}" == "checkpoint" ]]; then
  MODE_SUFFIX+="-${CHECKPOINT_OPTIMIZER}"
  if [[ "${CHECKPOINT_ASYNC_SAVE}" == "1" ]]; then
    MODE_SUFFIX+="-async"
  fi
fi
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-${SCENARIO}${MODE_SUFFIX}-${TIME_STAMP}.log"

echo "Running HCU upstream core scenario: ${SCENARIO}${MODE_SUFFIX}"
echo "Model: ${MODEL_PATH}"
echo "Data:  ${DATA_ROOT}"
echo "Save:  ${SAVE_ROOT}"
echo "Log:   ${LOG_FILE}"
echo "Actor GPUs: ${ACTOR_NUM_GPUS_PER_NODE}; rollout GPUs: ${ROLLOUT_NUM_GPUS}"
if [[ "${SCENARIO}" == "checkpoint" ]]; then
  echo "Checkpoint optimizer placement: ${CHECKPOINT_OPTIMIZER}; async save: ${CHECKPOINT_ASYNC_SAVE}"
fi

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train.py"
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${ALGORITHM_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${SCENARIO_ARGS[@]}"
)

if [[ "${SUBMIT_MODE}" == "job" ]]; then
  ray job submit \
    --address="http://${NODE_IP}:${RAY_DASHBOARD_PORT}" \
    --runtime-env="${SCRIPT_DIR}/env.yaml" \
    -- "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
elif [[ "${SUBMIT_MODE}" == "direct" ]]; then
  export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
  "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  echo "Unknown SUBMIT_MODE: ${SUBMIT_MODE}. Use job or direct." >&2
  exit 2
fi
