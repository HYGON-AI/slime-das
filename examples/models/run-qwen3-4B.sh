#!/bin/bash
MIP=$1

for para in $*
do
    if [[ $para == --slime_path* ]];then
        slime_path=${para#*=}
    elif [[ $para == --model_path* ]];then
        model_path=${para#*=}
    elif [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --host_file* ]];then
          host_file=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done


#export HIPBLASLT_ALLOW_TF32=1
### AMD Support ###
SLIME_DIR=${slime_path} # Default path if not set in environment
export SLIME_DIR

MODEL_DIR=${model_path} # Default path if not set in environment
export MODEL_DIR

DATA_DIR=${data_path}  # Default path if not set in environment
export DATA_DIR

# For AMD GPU
#export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"} # Must set to 1
#export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"} #You can choose which gpus to use
####################


# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export USE_DCU_CUSTOM_ALLREDUCE=0

source "${SLIME_DIR}/slime/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B
   --ref-load ${MODEL_DIR}/Qwen3-4B_torch_dist
   --load ${MODEL_DIR}/Qwen3-4B_slime/
   --save ${MODEL_DIR}/Qwen3-4B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 16
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 1024
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608 #9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   #--use-distributed-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-pipeline-parallel-size 1
   #--sglang-mem-fraction-static 0.9  #### needed when --colocate
   --sglang-attention-backend fa3
   --sglang-page-size 64
   --sglang-disable-custom-all-reduce
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   ### AMD Support ###
   # disable gradient accumulation fusion: Need to add apex to enable this
   --no-gradient-accumulation-fusion
   ###################
)

TORCH_PROFIE_ARGS=(
    --profile-ranks 0
    --profile-step-start 3
    --profile-step-end 4
    --use-pytorch-profiler
    --profile-target train_overall
    --tensorboard-dir ${SLIME_DIR}/tensorboard
)

APP="ray job submit --address="${MIP}:6379" \
   --runtime-env=${SLIME_DIR}/examples/env.yaml \
   -- python3 ${SLIME_DIR}/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}"

if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS[@]}"
fi
${APP}

####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python