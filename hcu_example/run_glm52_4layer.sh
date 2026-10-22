#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node GLM-5.2 RL smoke test, using the 4-layer / 64-routed-expert
# checkpoint created by hcu_example/slice_glm52_hf.py.
#
# The common GLM launch plumbing remains in run_glm5_4layer.sh.  This wrapper
# supplies only the architecture values that differ from GLM-5.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_LABEL="${MODEL_LABEL:-GLM-5.2}"
export MODEL_TAG="${MODEL_TAG:-glm52}"
export MODEL_PATH="${MODEL_PATH:-/home/Download/glm5/GLM-5.2-4layers-64experts}"
export SAVE_ROOT="${SAVE_ROOT:-/home/Download/glm5/GLM-5.2-4layers-64experts_slime}"

# Must exactly match the sliced HF config.json.
export MODEL_NUM_LAYERS="${MODEL_NUM_LAYERS:-4}"
export MODEL_MOE_LAYER_FREQ="${MODEL_MOE_LAYER_FREQ:-[0]*3+[1]*1}"
export MODEL_ROUTED_EXPERTS="${MODEL_ROUTED_EXPERTS:-64}"
export MODEL_ROTARY_BASE="${MODEL_ROTARY_BASE:-8000000}"

# A conservative first-run footprint for an eight-GPU / 64-GB-per-GPU node:
# four GPUs for Megatron actor+reference and four for SGLang rollout.
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-4}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-256}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-64}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-64}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DSA_KERNEL_BACKEND="${DSA_KERNEL_BACKEND:-hip}"

exec bash "${SCRIPT_DIR}/run_glm5_4layer.sh" "$@"