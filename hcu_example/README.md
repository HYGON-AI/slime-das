# HCU slime Qwen3-4B example

本目录提供 HCU 环境下运行 slime Qwen3-4B 示例的脚本。所有命令都在 Docker 容器内执行。

已验证的基础镜像：

```bash
42.228.13.241:5000/jenkins/model_test_env/sglang:daily-20260507-1407
```

## 目录说明

```text
hcu_example/
├── common_env.sh              # 通用环境变量
├── env.yaml                   # Ray Job runtime env，单节点 job 模式可用
├── requirements-hcu.txt       # 当前镜像上的增量依赖
├── start_ray.sh               # 启动 Ray head
├── start_ray_worker.sh        # worker 节点加入 Ray 集群
├── run_all.sh                 # 单节点一键运行
├── run_qwen3_4b.sh            # 通用训练提交脚本
├── run_multinode_qwen3_4b.sh  # 多节点训练提交脚本
└── stop_ray.sh                # 停止 Ray
```

## 基础准备

进入目录并安装依赖：

```bash
cd <repo-root>/hcu_example
/usr/bin/python3 -m pip install -r requirements-hcu.txt
chmod +x *.sh
```

默认路径：

```text
模型: /model/qwen3/Qwen3-4B
数据: <data-root>/dapo-math-17k/dapo-math-17k.jsonl
评测: <data-root>/aime-2024/aime-2024.jsonl
保存: <data-root>/qwen3/Qwen3-4B_slime
```

重要：多节点 Ray 的临时目录必须使用节点本地路径，不能放共享 `/home`。

```bash
export RAY_TMPDIR=/dev/shm/ray_tmp_hcu
```

如果放到共享 `/home/ray_tmp_hcu`，不同节点的 raylet socket 会混在一起，可能报：

```text
Could not connect to socket .../sockets/raylet.1
```

## 单节点运行

假设单节点 IP 为 `<head-ip>`，使用 8 张卡：

```bash
cd <repo-root>/hcu_example

ray stop --force || true
rm -rf /dev/shm/ray_tmp_hcu

source common_env.sh

export RAY_TMPDIR=/dev/shm/ray_tmp_hcu
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8 NODE_IP=<head-ip> bash run_all.sh
```

断点续训：

```bash
NUM_GPUS=8 NODE_IP=<head-ip> bash run_all.sh --resume
```

也可以分步启动 Ray 和提交训练：

```bash
NUM_GPUS=8 bash start_ray.sh <head-ip>

SUBMIT_MODE=direct \
RAY_HEAD_ADDRESS=<head-ip>:63792 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=4 \
ROLLOUT_NUM_GPUS=4 \
ROLLOUT_NUM_GPUS_PER_ENGINE=4 \
bash run_qwen3_4b.sh --node-ip 127.0.0.1
```

## 多节点运行

标准多节点配置：

```text
head:   <head-ip>，8 张卡
worker: <worker-ip>，8 张卡
总计:   16 张卡
```

资源划分：

```text
actor:   2 节点 × 每节点 4 卡 = 8 卡
rollout: 8 卡
```

### 1. head 节点启动 Ray

在 `<head-ip>` 上执行：

```bash
ray stop --force || true
pkill -9 raylet gcs_server plasma_store dashboard 2>/dev/null || true
rm -rf /dev/shm/ray_tmp_hcu

cd <repo-root>/hcu_example
source common_env.sh

export RAY_TMPDIR=/dev/shm/ray_tmp_hcu
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8 bash start_ray.sh <head-ip>
```

### 2. worker 节点加入 Ray

在 `<worker-ip>` 上执行：

```bash
ray stop --force || true
pkill -9 raylet gcs_server plasma_store dashboard 2>/dev/null || true
rm -rf /dev/shm/ray_tmp_hcu

cd <repo-root>/hcu_example
source common_env.sh

export RAY_TMPDIR=/dev/shm/ray_tmp_hcu
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8 bash start_ray_worker.sh <head-ip> <worker-ip>
```

### 3. head 节点检查资源

回到 `<head-ip>`：

```bash
ray status --address=<head-ip>:63792
```

应看到：

```text
0.0/16.0 GPU
```

可选：检查 Ray actor 是否能看到 HIP GPU。

```bash
RAY_ADDRESS=<head-ip>:63792 python3 - <<'PY'
import ray
ray.init(address="auto")

@ray.remote(num_gpus=0)
def f():
    import os, torch
    return {
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "count": torch.cuda.device_count(),
        "hip": torch.version.hip,
    }

print(ray.get(f.remote()))
ray.shutdown()
PY
```

### 4. head 节点提交训练

推荐多节点使用 `direct` 模式，不走 Ray dashboard job server：

```bash
cd <repo-root>/hcu_example

SUBMIT_MODE=direct \
RAY_HEAD_ADDRESS=<head-ip>:63792 \
ACTOR_NUM_NODES=2 \
ACTOR_NUM_GPUS_PER_NODE=4 \
ROLLOUT_NUM_GPUS=8 \
ROLLOUT_NUM_GPUS_PER_ENGINE=4 \
bash run_qwen3_4b.sh --node-ip 127.0.0.1
```

如果已同步默认配置，也可以直接：

```bash
bash run_multinode_qwen3_4b.sh
```

断点续训：

```bash
bash run_multinode_qwen3_4b.sh --resume
```

## 常用参数

可以通过环境变量覆盖：

```bash
NUM_ROLLOUT=1000
ROLLOUT_MAX_RESPONSE_LEN=4096
EVAL_MAX_RESPONSE_LEN=4096
MAX_TOKENS_PER_GPU=6144
KL_LOSS_COEF=0.0

ACTOR_NUM_NODES=2
ACTOR_NUM_GPUS_PER_NODE=4
ROLLOUT_NUM_GPUS=8
ROLLOUT_NUM_GPUS_PER_ENGINE=4

LOG_DIR=<repo-root>/hcu_example/logs
```

训练日志会自动保存到 `LOG_DIR`，默认是：

```text
<repo-root>/hcu_example/logs
```

日志文件名包含模型名、提交模式、节点配置和时间，例如：

```text
Qwen3-4B-direct-node2-rollout8-20260709-143000.log
```

停止 Ray：

```bash
bash stop_ray.sh
```

## Qwen3.5-4B 运行流程

Qwen3.5-4B 的 HuggingFace 目录是 Qwen3.5/VL 风格配置，直接使用 Megatron-Bridge 的 `--megatron-to-hf-mode bridge` 路线容易走到 `Qwen35VLBridge`，在文本 RL 的权重同步阶段会出现参数名不匹配问题。因此 Qwen3.5-4B 推荐使用官方文本 RL 路线：

1. 先把 HuggingFace 权重转换成 `torch_dist`。
2. 训练时从 `torch_dist` 加载权重。
3. 不要在训练脚本里传 `--megatron-to-hf-mode bridge`。

### 1. 转换 Qwen3.5-4B 为 torch_dist

在容器内执行：

```bash
cd <repo-root>/slime

source <repo-root>/hcu_example/common_env.sh
source scripts/models/qwen3.5-4B.sh

python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /model/qwen3.5/Qwen3.5-4B \
  --save <data-root>/qwen3.5/Qwen3.5-4B_torch_dist \
  --use-cpu-initialization \
  --no-gradient-accumulation-fusion
```

如果提示缺少 `mbridge`，先安装：

```bash
/usr/bin/python3 -m pip install mbridge
```

转换成功后，应存在：

```text
<data-root>/qwen3.5/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt
```

### 2. 启动 Ray

单节点 8 卡示例，假设当前机器 IP 是 `<worker-ip>`：

```bash
cd <repo-root>/hcu_example

ray stop --force || true
rm -rf /dev/shm/ray_tmp_hcu

source common_env.sh

export RAY_TMPDIR=/dev/shm/ray_tmp_hcu
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8 bash start_ray.sh <worker-ip>
```

检查 Ray：

```bash
ray status --address=<worker-ip>:63792
curl http://127.0.0.1:8265/api/version
```

### 3. 运行 Qwen3.5-4B 训练

使用专门的脚本：

```bash
cd <repo-root>/hcu_example

MODEL_PATH=/model/qwen3.5/Qwen3.5-4B \
TORCH_DIST_PATH=<data-root>/qwen3.5/Qwen3.5-4B_torch_dist \
SAVE_ROOT=<data-root>/qwen3.5/Qwen3.5-4B_slime \
MODEL_ARGS_SCRIPT=<repo-root>/slime/scripts/models/qwen3.5-4B.sh \
SUBMIT_MODE=direct \
RAY_HEAD_ADDRESS=<worker-ip>:63792 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=4 \
TENSOR_MODEL_PARALLEL_SIZE=4 \
ROLLOUT_NUM_GPUS=4 \
ROLLOUT_NUM_GPUS_PER_ENGINE=4 \
NUM_ROLLOUT=1000 \
ROLLOUT_BATCH_SIZE=8 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=16 \
N_SAMPLES_PER_EVAL_PROMPT=8 \
ROLLOUT_MAX_RESPONSE_LEN=1024 \
EVAL_MAX_RESPONSE_LEN=1024 \
SGLANG_MEM_FRACTION_STATIC=0.6 \
bash run_qwen3.5_4b.sh --node-ip 127.0.0.1
```

说明：

- `MODEL_PATH`：原始 HuggingFace 模型目录。
- `TORCH_DIST_PATH`：第一步转换出来的 `torch_dist` 权重目录。
- `SAVE_ROOT`：训练 checkpoint 保存目录。
- `MODEL_ARGS_SCRIPT`：Qwen3.5-4B 的 Megatron 模型参数脚本。
- `SUBMIT_MODE=direct`：直接连接已启动的 Ray 集群运行，避免 Ray Job runtime env 额外干扰。
- `ACTOR_NUM_GPUS_PER_NODE=4`、`ROLLOUT_NUM_GPUS=4`：单节点 8 卡时，4 张卡用于 actor 训练，4 张卡用于 SGLang rollout。
- `SGLANG_MEM_FRACTION_STATIC=0.6`：控制 SGLang 静态显存占用比例。Qwen3.5 默认 context 较大，显存压力高时可以适当调低。

### 4. 断点续训

如果 `<data-root>/qwen3.5/Qwen3.5-4B_slime/latest_checkpointed_iteration.txt` 已存在，可以加 `--resume`：

```bash
cd <repo-root>/hcu_example

MODEL_PATH=/model/qwen3.5/Qwen3.5-4B \
TORCH_DIST_PATH=<data-root>/qwen3.5/Qwen3.5-4B_torch_dist \
SAVE_ROOT=<data-root>/qwen3.5/Qwen3.5-4B_slime \
MODEL_ARGS_SCRIPT=<repo-root>/slime/scripts/models/qwen3.5-4B.sh \
SUBMIT_MODE=direct \
RAY_HEAD_ADDRESS=<worker-ip>:63792 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=4 \
TENSOR_MODEL_PARALLEL_SIZE=4 \
ROLLOUT_NUM_GPUS=4 \
ROLLOUT_NUM_GPUS_PER_ENGINE=4 \
NUM_ROLLOUT=1000 \
ROLLOUT_BATCH_SIZE=8 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=16 \
N_SAMPLES_PER_EVAL_PROMPT=8 \
ROLLOUT_MAX_RESPONSE_LEN=1024 \
EVAL_MAX_RESPONSE_LEN=1024 \
SGLANG_MEM_FRACTION_STATIC=0.6 \
bash run_qwen3.5_4b.sh --node-ip 127.0.0.1 --resume
```