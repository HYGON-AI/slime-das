#!/usr/bin/env bash
set -euo pipefail

ray stop --force || true
echo "Ray stopped. No generic Python processes were killed."

