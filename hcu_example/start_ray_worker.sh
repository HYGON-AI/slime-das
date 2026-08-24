#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

HEAD_IP="${1:-${HEAD_IP:-127.0.0.1}}"
WORKER_IP="${2:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"
NUM_GPUS="${NUM_GPUS:-8}"

echo "Joining Ray cluster ${HEAD_IP}:${RAY_PORT} from worker ${WORKER_IP} with ${NUM_GPUS} GPUs"

ray stop --force >/dev/null 2>&1 || true

ray start \
  --address="${HEAD_IP}:${RAY_PORT}" \
  --node-ip-address="${WORKER_IP}" \
  --num-gpus="${NUM_GPUS}" \
  --temp-dir="${RAY_TMPDIR}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --min-worker-port=20000 \
  --max-worker-port=20999 \
  --disable-usage-stats

echo "Worker joined. Check cluster resources on the head node with:"
echo "  ray status --address=${HEAD_IP}:${RAY_PORT}"
