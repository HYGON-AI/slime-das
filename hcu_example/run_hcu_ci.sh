#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCENARIO="${1:-}"
case "${SCENARIO}" in
  smoke|fully-async|ppo-disaggregate|ppo-colocate|checkpoint|checkpoint-extended|streaming-partial) ;;
  *)
    echo "Usage: $0 {smoke|fully-async|ppo-disaggregate|ppo-colocate|checkpoint|checkpoint-extended|streaming-partial}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SLIME_ROOT="${SLIME_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export HCU_MEGATRON_ROOT="${HCU_MEGATRON_ROOT:-/opt/hcu-megatron}"
export DATA_ROOT="${DATA_ROOT:-/opt/slime-data}"
export MODEL_PATH="${MODEL_PATH:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B-Thinking-2507}"
export SAVE_ROOT="${SAVE_ROOT:-${SLIME_ROOT}/.hcu-core/${SCENARIO}/checkpoints}"
export LOG_DIR="${LOG_DIR:-${SLIME_ROOT}/.hcu-core/${SCENARIO}/logs}"
export NODE_IP="${NODE_IP:-127.0.0.1}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

OUTPUT_ROOT="$(dirname -- "${SAVE_ROOT}")"
cleanup() {
  local status=$?
  trap - EXIT
  bash "${SCRIPT_DIR}/stop_ray.sh" || true
  rm -rf -- "${SAVE_ROOT}" || true
  chmod -R a+rX "${LOG_DIR}" 2>/dev/null || true
  if [[ "$(id -u)" == "0" && -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
    chown -R "${HOST_UID}:${HOST_GID}" "${OUTPUT_ROOT}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT

for required in \
  "${HCU_MEGATRON_ROOT}/3rdparty/Megatron-Bridge" \
  "${HCU_MEGATRON_ROOT}/3rdparty/Megatron-LM" \
  "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/model.safetensors.index.json" \
  "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"; do
  [[ -e "${required}" ]] || { echo "Missing bundled HCU asset: ${required}" >&2; exit 1; }
done

if [[ "${SCENARIO}" == "smoke" ]]; then
  [[ -f "${DATA_ROOT}/aime-2024/aime-2024.jsonl" ]] || {
    echo "Missing bundled HCU asset: ${DATA_ROOT}/aime-2024/aime-2024.jsonl" >&2
    exit 1
  }
fi

python3 -c 'import hcu_megatron, ray, sglang, torch; count = torch.cuda.device_count(); print(f"ray={ray.__version__} sglang={sglang.__version__} devices={count} first={torch.cuda.get_device_name(0)}"); assert count >= 8'
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"
bash "${SCRIPT_DIR}/start_ray.sh" "${NODE_IP}"

run_checkpoint_roundtrip() {
  local variant="$1"
  local async_save="$2"
  local save_optimizer="$3"
  local load_optimizer="$4"
  local checkpoint_dir="${SAVE_ROOT}/${variant}"
  local variant_log_dir="${LOG_DIR}/${variant}"
  local -a async_args=()

  if [[ "${async_save}" == "1" ]]; then
    async_args+=(--checkpoint-async-save)
  fi

  mkdir -p "${checkpoint_dir}" "${variant_log_dir}"
  bash "${SCRIPT_DIR}/stop_ray.sh"
  bash "${SCRIPT_DIR}/start_ray.sh" "${NODE_IP}"
  bash "${SCRIPT_DIR}/run_qwen3_4b_upstream_core.sh" \
    --scenario checkpoint \
    --checkpoint-mode save \
    --checkpoint-optimizer "${save_optimizer}" \
    "${async_args[@]}" \
    --node-ip "${NODE_IP}" \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --save-root "${checkpoint_dir}" \
    --log-dir "${variant_log_dir}"

  if [[ "${async_save}" == "1" ]]; then
    local async_log
    async_log="$(find "${variant_log_dir}" -maxdepth 1 -type f \
      -name '*-checkpoint-save-*-async-*.log' -print | sort | tail -n 1)"
    [[ -n "${async_log}" ]]
    grep -q 'scheduled an async checkpoint save' "${async_log}"
    grep -q 'successfully saved checkpoint' "${async_log}"
    ! grep -q 'Disabling --async-save' "${async_log}"
  fi

  local saved_iteration
  saved_iteration="$(tr -d '[:space:]' < "${checkpoint_dir}/latest_checkpointed_iteration.txt")"
  [[ "${saved_iteration}" =~ ^[0-9]+$ ]]

  bash "${SCRIPT_DIR}/stop_ray.sh"
  bash "${SCRIPT_DIR}/start_ray.sh" "${NODE_IP}"
  bash "${SCRIPT_DIR}/run_qwen3_4b_upstream_core.sh" \
    --scenario checkpoint \
    --checkpoint-mode load \
    --checkpoint-optimizer "${load_optimizer}" \
    --node-ip "${NODE_IP}" \
    --model-path "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --save-root "${checkpoint_dir}" \
    --log-dir "${variant_log_dir}"

  local resumed_iteration
  resumed_iteration="$(tr -d '[:space:]' < "${checkpoint_dir}/latest_checkpointed_iteration.txt")"
  [[ "${resumed_iteration}" =~ ^[0-9]+$ ]]
  (( resumed_iteration > saved_iteration ))
  printf 'variant=%s saved=%s resumed=%s\n' \
    "${variant}" "${saved_iteration}" "${resumed_iteration}" \
    | tee "${variant_log_dir}/checkpoint-roundtrip.txt"

  # Each roundtrip is roughly 100 GB. Preserve logs and release generated
  # checkpoint state before the next variant starts.
  rm -rf -- "${checkpoint_dir}"
}

case "${SCENARIO}" in
  smoke)
    export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
    export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
    export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
    export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-256}"
    export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-256}"
    bash "${SCRIPT_DIR}/run_qwen3_4b.sh" \
      --node-ip "${NODE_IP}" \
      --model-path "${MODEL_PATH}" \
      --data-root "${DATA_ROOT}" \
      --save-root "${SAVE_ROOT}"
    ;;
  fully-async)
    export NUM_ROLLOUT=2
    export ROLLOUT_BATCH_SIZE=2
    export N_SAMPLES_PER_PROMPT=2
    export GLOBAL_BATCH_SIZE=4
    export ROLLOUT_MAX_RESPONSE_LEN=256
    bash "${SCRIPT_DIR}/run_qwen3_4b_fully_async.sh" \
      --node-ip "${NODE_IP}" \
      --model-path "${MODEL_PATH}" \
      --data-root "${DATA_ROOT}" \
      --save-root "${SAVE_ROOT}"
    ;;
  ppo-disaggregate|streaming-partial)
    bash "${SCRIPT_DIR}/run_qwen3_4b_upstream_core.sh" \
      --scenario "${SCENARIO}" \
      --node-ip "${NODE_IP}" \
      --model-path "${MODEL_PATH}" \
      --data-root "${DATA_ROOT}" \
      --save-root "${SAVE_ROOT}" \
      --log-dir "${LOG_DIR}"
    ;;
  ppo-colocate)
    export COLOCATE_ACTOR_NUM_GPUS_PER_NODE=8
    export COLOCATE_ROLLOUT_NUM_GPUS=8
    export COLOCATE_ROLLOUT_NUM_GPUS_PER_ENGINE=8
    export COLOCATE_SGLANG_MEM_FRACTION_STATIC=0.5
    export NUM_ROLLOUT=2
    export ROLLOUT_BATCH_SIZE=2
    export N_SAMPLES_PER_PROMPT=1
    export GLOBAL_BATCH_SIZE=2
    export ROLLOUT_MAX_RESPONSE_LEN=128
    bash "${SCRIPT_DIR}/run_qwen3_4b_upstream_core.sh" \
      --scenario ppo-colocate \
      --node-ip "${NODE_IP}" \
      --model-path "${MODEL_PATH}" \
      --data-root "${DATA_ROOT}" \
      --save-root "${SAVE_ROOT}" \
      --log-dir "${LOG_DIR}"
    ;;
  checkpoint)
    bash "${SCRIPT_DIR}/run_qwen3_4b_upstream_core.sh" \
      --scenario checkpoint \
      --checkpoint-mode save \
      --node-ip "${NODE_IP}" \
      --model-path "${MODEL_PATH}" \
      --data-root "${DATA_ROOT}" \
      --save-root "${SAVE_ROOT}" \
      --log-dir "${LOG_DIR}"
    saved_iteration="$(tr -d '[:space:]' < "${SAVE_ROOT}/latest_checkpointed_iteration.txt")"
    [[ "${saved_iteration}" =~ ^[0-9]+$ ]]
    echo "saved_iteration=${saved_iteration}" | tee "${LOG_DIR}/checkpoint-save-iteration.txt"

    bash "${SCRIPT_DIR}/stop_ray.sh"
    bash "${SCRIPT_DIR}/start_ray.sh" "${NODE_IP}"
    bash "${SCRIPT_DIR}/run_qwen3_4b_upstream_core.sh" \
      --scenario checkpoint \
      --checkpoint-mode load \
      --node-ip "${NODE_IP}" \
      --model-path "${MODEL_PATH}" \
      --data-root "${DATA_ROOT}" \
      --save-root "${SAVE_ROOT}" \
      --log-dir "${LOG_DIR}"
    resumed_iteration="$(tr -d '[:space:]' < "${SAVE_ROOT}/latest_checkpointed_iteration.txt")"
    [[ "${resumed_iteration}" =~ ^[0-9]+$ ]]
    (( resumed_iteration > saved_iteration ))
    echo "resumed_iteration=${resumed_iteration}" | tee "${LOG_DIR}/checkpoint-resume-iteration.txt"
    ;;
  checkpoint-extended)
    export ROLLOUT_MAX_RESPONSE_LEN=64
    run_checkpoint_roundtrip async-cpu-cpu 1 cpu cpu
    run_checkpoint_roundtrip sync-cpu-gpu 0 cpu gpu
    run_checkpoint_roundtrip sync-gpu-cpu 0 gpu cpu
    run_checkpoint_roundtrip sync-gpu-gpu 0 gpu gpu
    ;;
esac
