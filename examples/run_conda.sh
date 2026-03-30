#!/bin/bash

LOG_PATH=${model_name}-`date +%F-%H%M`.log

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
SLIME_PATH=$( dirname $( dirname ${CURRENT_DIR}))
SLIME_PATH="${SLIME_PATH}/dcu_slime"
# These variables should be modified
PORT="63792" # The port which you set in your docker
HOST_FILE="./hostfile"
DATA_PATH="/public/home/linxi1/dcu_slime/data"
MODEL_PATH="/public/home/linxi1/dcu_slime/data"
PROFILING="torch" # If you want to profiling, please choose one of {torch}
DTK_ENV="/public/home/linxi1/dtk-25.04.2-rc1-centos8/env.sh"

source ${DTK_ENV}
# pstart ray
MASTER_ADDR=$(awk '{print $1}' ${HOST_FILE}|head -n 1)

mpirun  -v \
        --allow-run-as-root \
        --bind-to none \
        --hostfile ${HOST_FILE} \
        --mca plm_rsh_no_tree_spawn 1 \
        -x AMDGPU_TARGETS \
        -x HIP_PATH \
        -x MIOPEN_FIND_MODE \
        -x PATH \
        -x LIBRARY_PATH \
        -x LD_LIBRARY_PATH \
        -x HYHAL_PATH \
        -x ROCM_PATH \
        -x C_INCLUDE_PATH \
        -x CPLUS_INCLUDE_PATH \
        bash  ${SLIME_PATH}/examples/ray_start.sh ${MASTER_ADDR} #2>&1 | tee ${LOG_PATH}
wait

bash ${SLIME_PATH}/examples/models/run-qwen3-4B.sh \
    ${MASTER_ADDR} \
    --data_path=${DATA_PATH} \
    --model_path=${MODEL_PATH} \
    --slime_path=${SLIME_PATH} \
    --profiling=${PROFILING} #2>&1 | tee -a ${LOG_PATH}
wait
