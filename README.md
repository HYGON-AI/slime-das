# DCU Slime

基于 [THUDM/slime](https://github.com/THUDM/slime) v0.3.0 的 DCU 适配版本，支持在 DCU 环境下使用 Megatron-LM、Megatron-Bridge 和 SGLang 进行 RL 训练。

当前仓库提供了一个已经跑通的 Qwen3-4B 示例，示例脚本统一放在 `dcu_example/` 目录下。

## 特性

- 支持 DCU Docker 环境运行
- 支持 Qwen3-4B 示例训练
- 支持 Megatron-Bridge 直接加载 HuggingFace 权重
- 支持 Ray + SGLang rollout + Megatron actor training
- 提供一键运行脚本和断点续训脚本

## 目录结构

```text
dcu_slime/
├── dcu_example/          # DCU Qwen3-4B 训练示例脚本
│   ├── common_env.sh     # 通用环境变量
│   ├── env.yaml          # Ray runtime env
│   ├── README.md         # 示例脚本说明
│   ├── run_all.sh        # 一键启动 Ray 并提交训练
│   ├── run_qwen3_4b.sh   # 提交 Qwen3-4B 训练任务
│   ├── start_ray.sh      # 启动 Ray head
│   └── stop_ray.sh       # 停止 Ray
├── slime/                # slime 源码
├── requirements-dcu.txt  # 当前 DCU Docker 环境导出的依赖记录
└── usage_dcu.md          # 旧版 DCU 使用说明
```

## Docker 基础环境

当前示例验证使用的 Docker 镜像为：

```bash
42.228.13.241:5000/jenkins/model_test_env/sglang:daily-20260507-1407
```


## Python 依赖

通过仓库已经有 `requirements-dcu.txt`，可以直接安装：

```bash
/usr/bin/python3 -m pip install -r requirements-dcu.txt
```

## 外部源码依赖路径

DCU 环境使用以下适配源码（sglang_v0.5.12、dcu_megatron_v0.17.0）：

- [SGLang DCU 源码](https://developer.sourcefind.cn/codes/OpenDAS/sglang)
- [DCU Megatron-LM/Megatron-Bridge 源码](http://42.228.13.241:10068/dcutoolkit/deeplearing/dcu_megatron)

这些路径已经写入：

```text
dcu_example/common_env.sh
dcu_example/env.yaml
```

如果你的路径不同，请优先修改这两个文件，或者在运行前通过环境变量覆盖：

```bash
export SLIME_ROOT=/home/dcu_slime
export MEGATRON_BRIDGE_ROOT=/home/dcu_megatron/Megatron-Bridge
export MEGATRON_LM_ROOT=/home/dcu_megatron/Megatron-LM
export SGLANG_ROOT=/home/sglang_v0.5.10
```

## 数据和模型准备

示例默认使用 Qwen3-4B 模型：

```bash
/model/qwen3/Qwen3-4B
```

示例默认使用以下数据：

```bash
/home/Download/dapo-math-17k/dapo-math-17k.jsonl
/home/Download/aime-2024/aime-2024.jsonl
```

如果没有数据，可以下载：

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /home/Download/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /home/Download/aime-2024
```

Megatron 0.17 支持通过 Megatron-Bridge 直接加载 HuggingFace 权重，因此当前示例不需要提前把 HF 权重转换成 torch_dist。

## 单节点运行

假设单节点 IP 是 `12.12.12.48`，使用 8 张卡：

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

在 `12.12.12.48`：

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

在 `12.12.12.52`：

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

```bash
ray status --address=12.12.12.48:63792
```

应看到：

```text
0.0/16.0 GPU
```

### 4. head 节点提交训练

多节点推荐使用 `direct` 模式，不走 Ray dashboard job server：

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

如果已同步默认多节点脚本，也可以直接：

```bash
bash run_multinode_qwen3_4b.sh
```

断点续训：

```bash
bash run_multinode_qwen3_4b.sh --resume
```

## 常用可调参数

以下参数可以通过环境变量覆盖：

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

训练日志会自动保存到 `LOG_DIR`，默认是 `dcu_example/logs`。日志文件名会带时间戳，例如：

```text
Qwen3-4B-direct-node2-rollout8-20260709-143000.log
```

## 本示例做的主要改进

相比手动执行一长串命令，`dcu_example/` 对运行流程做了如下整理：

- 将通用环境变量统一放到 `common_env.sh`
- 将 Ray runtime env 统一放到 `env.yaml`
- 提供 `run_all.sh` 一键启动 Ray 并提交任务
- 提供 `start_ray.sh`、`stop_ray.sh` 独立管理 Ray
- 将 Ray、Triton、Torch 编译缓存放到 `/home`，避免 `/tmp` 或 Docker overlay 空间不足
- 固定 Ray 端口，避免 dashboard 和 job submit 地址混乱
- 通过 `PYTHONPATH` 导入 Megatron-LM、Megatron-Bridge、SGLang 和 slime
- 默认开启 `USE_DCU_CUSTOM_ALLREDUCE=1`
- 默认使用 Megatron-Bridge 直接加载 HuggingFace 权重
- 支持 `--resume` 从最近 checkpoint 继续训练

## 更多说明

更多脚本细节见：

```text
dcu_example/README.md
```


