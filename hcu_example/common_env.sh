#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared HCU runtime environment. This file is sourced by the other scripts.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SLIME_ROOT="${SLIME_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

: "${MEGATRON_BRIDGE_ROOT:?Set MEGATRON_BRIDGE_ROOT to the Megatron-Bridge checkout.}"
: "${MEGATRON_LM_ROOT:?Set MEGATRON_LM_ROOT to the Megatron-LM checkout.}"
: "${SGLANG_ROOT:?Set SGLANG_ROOT to the SGLang checkout.}"

if [[ -f /opt/dtk/env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi

export PATH="/opt/hyhal/bin:/opt/dtk/hip/bin:/opt/dtk/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
HCU_MEGATRON_PYTHONPATH=""
if [[ -n "${HCU_MEGATRON_ROOT:-}" ]]; then
  HCU_MEGATRON_PYTHONPATH="${HCU_MEGATRON_ROOT}:"
fi
export PYTHONPATH="${HCU_MEGATRON_PYTHONPATH}${MEGATRON_BRIDGE_ROOT}/src:${MEGATRON_LM_ROOT}:${SGLANG_ROOT}/python:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Some HCU Megatron releases import the optional Apex weight-gradient module
# even when callers disable gradient-accumulation fusion.  Make the import-only
# guard visible before Ray starts so every worker receives the same environment.
# A real installed HCU extension always takes precedence.
if ! python3 -c 'import fused_weight_gradient_mlp_cuda' >/dev/null 2>&1; then
  export PYTHONPATH="${SCRIPT_DIR}/compat:${PYTHONPATH}"
  echo "HCU: fused weight-gradient extension is absent; disabled-fusion import protection is enabled."
fi
export PYTHONBUFFERED="${PYTHONBUFFERED:-16}"

# HCU/SGLang/Megatron runtime defaults. These are also present in env.yaml for
# Ray Job submission, but exporting them here makes direct Ray submission work
# without going through the Ray dashboard job agent.
export GLOG_minloglevel="${GLOG_minloglevel:-3}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-1}"
export USE_HCU_CUSTOM_ALLREDUCE="${USE_HCU_CUSTOM_ALLREDUCE:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export RCCL_SDMA_COPY_ENABLE="${RCCL_SDMA_COPY_ENABLE:-0}"
export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"


export TMPDIR="${TMPDIR:-/dev/shm}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SLIME_ROOT}/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME}/torch_extensions}"
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_tmp_hcu}"

export RAY_PORT="${RAY_PORT:-63792}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52365}"
export RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-52366}"

#sglang
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-1}"
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS="${SGLANG_ASSIGN_EXTEND_CACHE_LOCS:-1}"
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL="${SGLANG_ASSIGN_REQ_TO_TOKEN_POOL:-1}"
export SGLANG_GET_LAST_LOC="${SGLANG_GET_LAST_LOC:-1}"
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON="${SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON:-1}"
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES="${SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES:-1}"
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

mkdir -p \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${RAY_TMPDIR}"
