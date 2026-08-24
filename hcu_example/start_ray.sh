#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

NODE_IP="${1:-${NODE_IP:-127.0.0.1}}"
NUM_GPUS="${NUM_GPUS:-8}"

echo "Starting Ray head at ${NODE_IP}:${RAY_PORT} (dashboard ${RAY_DASHBOARD_PORT})"

ray stop --force >/dev/null 2>&1 || true

ray start \
  --head \
  --node-ip-address="${NODE_IP}" \
  --port="${RAY_PORT}" \
  --num-gpus="${NUM_GPUS}" \
  --temp-dir="${RAY_TMPDIR}" \
  --include-dashboard=true \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --ray-client-server-port=10001 \
  --min-worker-port=20000 \
  --max-worker-port=20999 \
  --metrics-export-port=21100 \
  --disable-usage-stats

sleep 3
ray status --address="${NODE_IP}:${RAY_PORT}"
curl --fail --silent "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version"
echo

