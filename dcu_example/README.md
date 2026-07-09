# DCU slime Qwen3-4B example

本目录提供 DCU 环境下运行 slime Qwen3-4B 示例的脚本。所有命令都在 Docker 容器内执行。

已验证的基础镜像：

```bash
42.228.13.241:5000/jenkins/model_test_env/sglang:daily-20260507-1407
```

## 目录说明

```text
dcu_example/
├── common_env.sh              # 通用环境变量
├── env.yaml                   # Ray Job runtime env，单节点 job 模式可用
├── requirements-dcu.txt       # 当前镜像上的增量依赖
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
cd /home/dcu_slime/dcu_example
/usr/bin/python3 -m pip install -r requirements-dcu.txt
chmod +x *.sh
```

默认路径：

```text
模型: /model/qwen3/Qwen3-4B
数据: /home/Download/dapo-math-17k/dapo-math-17k.jsonl
评测: /home/Download/aime-2024/aime-2024.jsonl
保存: /home/Download/qwen3/Qwen3-4B_slime
```

重要：多节点 Ray 的临时目录必须使用节点本地路径，不能放共享 `/home`。

```bash
export RAY_TMPDIR=/dev/shm/ray_tmp_dcu
```

如果放到共享 `/home/ray_tmp_dcu`，不同节点的 raylet socket 会混在一起，可能报：

```text
Could not connect to socket .../sockets/raylet.1
```

## 单节点运行

假设单节点 IP 为 `12.12.12.48`，使用 8 张卡：

```bash
cd /home/dcu_slime/dcu_example

ray stop --force || true
rm -rf /dev/shm/ray_tmp_dcu

source common_env.sh

export RAY_TMPDIR=/dev/shm/ray_tmp_dcu
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8 NODE_IP=12.12.12.48 bash run_all.sh
```

断点续训：

```bash
NUM_GPUS=8 NODE_IP=12.12.12.48 bash run_all.sh --resume
```

也可以分步启动 Ray 和提交训练：

```bash
NUM_GPUS=8 bash start_ray.sh 12.12.12.48

SUBMIT_MODE=direct \
RAY_HEAD_ADDRESS=12.12.12.48:63792 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=4 \
ROLLOUT_NUM_GPUS=4 \
ROLLOUT_NUM_GPUS_PER_ENGINE=4 \
bash run_qwen3_4b.sh --node-ip 127.0.0.1
```

## 多节点运行

标准多节点配置：

```text
head:   12.12.12.48，8 张卡
worker: 12.12.12.52，8 张卡
总计:   16 张卡
```

资源划分：

```text
actor:   2 节点 × 每节点 4 卡 = 8 卡
rollout: 8 卡
```

### 1. head 节点启动 Ray

在 `12.12.12.48` 上执行：

```bash
ray stop --force || true
pkill -9 raylet gcs_server plasma_store dashboard 2>/dev/null || true
rm -rf /dev/shm/ray_tmp_dcu

cd /home/dcu_slime/dcu_example
source common_env.sh

export RAY_TMPDIR=/dev/shm/ray_tmp_dcu
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8 bash start_ray.sh 12.12.12.48
```

### 2. worker 节点加入 Ray

在 `12.12.12.52` 上执行：

```bash
ray stop --force || true
pkill -9 raylet gcs_server plasma_store dashboard 2>/dev/null || true
rm -rf /dev/shm/ray_tmp_dcu

cd /home/dcu_slime/dcu_example
source common_env.sh

export RAY_TMPDIR=/dev/shm/ray_tmp_dcu
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8 bash start_ray_worker.sh 12.12.12.48 12.12.12.52
```

### 3. head 节点检查资源

回到 `12.12.12.48`：

```bash
ray status --address=12.12.12.48:63792
```

应看到：

```text
0.0/16.0 GPU
```

可选：检查 Ray actor 是否能看到 HIP GPU。

```bash
RAY_ADDRESS=12.12.12.48:63792 python3 - <<'PY'
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
cd /home/dcu_slime/dcu_example

SUBMIT_MODE=direct \
RAY_HEAD_ADDRESS=12.12.12.48:63792 \
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

LOG_DIR=/home/dcu_slime/dcu_example/logs
```

训练日志会自动保存到 `LOG_DIR`，默认是：

```text
/home/dcu_slime/dcu_example/logs
```

日志文件名包含模型名、提交模式、节点配置和时间，例如：

```text
Qwen3-4B-direct-node2-rollout8-20260709-143000.log
```

停止 Ray：

```bash
bash stop_ray.sh
```
