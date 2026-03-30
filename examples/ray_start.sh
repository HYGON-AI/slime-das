#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force || true
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -euxo pipefail


MASTER_ADDR=$1
PORT=6379
node_rank=$OMPI_COMM_WORLD_RANK


if [[ "${node_rank}" == '0' ]]; then
  echo "start head ${node_rank}"
  ray start --head --node-ip-address ${MASTER_ADDR} --port=${PORT} --num-gpus=8
else
  echo "start worker ${node_rank}"
  ray start --address="${MASTER_ADDR}:${PORT}" --num-gpus=8
fi

# ray patch
packages_path=$(pip show ray | grep Location | awk -F 'Location: ' '{print $2}')/ray/_private/thirdparty/pyamdsmi/pyamdsmi.py
echo "packages_path:$packages_path"
if [[ -f "$packages_path" ]]; then
    sed -i "s|/sys/module/amdgpu/initstate|/sys/module/hycu/initstate|g" "$packages_path"
    sed -i "s|rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')|rocm_path = os.environ.get('HYHAL_PATH', '/opt/rocm')|g" "$packages_path"
else
    echo "File not found: $packages_path"
    exit 1
fi
