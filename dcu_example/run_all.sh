#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NODE_IP="${NODE_IP:-12.12.12.48}"

bash "${SCRIPT_DIR}/start_ray.sh" "${NODE_IP}"
bash "${SCRIPT_DIR}/run_qwen3_4b.sh" --node-ip "${NODE_IP}" "$@"
