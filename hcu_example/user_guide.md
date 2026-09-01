# slime-das HCU 用户指南

本文档说明如何配置和运行 slime-das 提供的 HCU 示例。

## 1. 环境准备

### 1.1 拉取 HCU 基础 Docker 镜像

请从[光源社区](https://developer.sourcefind.cn/servicelist/detail?post_id=1abf923f-5a33-11f1-9e57-0242ac150003)获取 HCU 基础 Docker 镜像的实际名称和标签，并拉取镜像：

```bash
docker pull REPOSITORY:TAG
```

将 `REPOSITORY:TAG` 替换为光源社区页面提供的镜像地址。

### 1.2 创建容器

以下命令仅供参考。请根据实际环境调整容器名称、工作目录、挂载目录和镜像名称：

```bash
docker run -it \
  --name slime-das \
  --shm-size=64G \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  --ipc=host \
  --network=host \
  --workdir=/workspace \
  --privileged \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v <host-workspace>:/workspace \
  REPOSITORY:TAG \
  /bin/bash
```

### 1.3 获取代码

在容器中执行：

```bash
cd /workspace
git clone https://github.com/HYGON-AI/slime-das.git
cd slime-das
```

如果网络环境无法直接克隆仓库，请下载对应分支源码后解压到工作目录。

### 1.4 安装项目依赖

使用示例前，请确认基础镜像已提供 HCU 驱动和运行时、HCU 版本 PyTorch 与 SGLang、Megatron-LM、Megatron-Bridge。

在仓库根目录安装项目 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 包含项目直接 Python 依赖，其中包括 `torch_memory_saver`。

设置 HCU Megatron 源码根目录。执行 `common_env.sh` 时，`SLIME_ROOT` 默认自动定位为当前仓库根目录，Megatron-Bridge 和 Megatron-LM 路径从该根目录自动推导；verl 镜像直接使用已安装的 SGLang：

```bash
export HCU_MEGATRON_ROOT=<path-to-hcu-megatron>
```

## 2. 支持模型和示例

| 模型 | 训练流程 | 脚本 |
| --- | --- | --- |
| Qwen3-4B | 标准训练 | `run_qwen3_4b.sh` |
| Qwen3-4B | 全异步 rollout 训练 | `run_qwen3_4b_fully_async.sh` |
| Qwen3.5-4B | `torch_dist` checkpoint 训练 | `run_qwen3.5_4b.sh` |
| GLM-4.6（4 层） | 功能验证示例 | `run_glm4.6_4layer.sh` |
| GLM-5（4 层） | 功能验证示例 | `run_glm5_4layer.sh` |
| DeepSeek-R1（4 层） | 功能验证示例 | `run_deepseek_r1_4layer.sh` |
| DeepSeek-R1-Distill-Llama-70B（4 层） | 功能验证示例 | `run_deepseek_r1_distill_llama_70b_4layer.sh` |

四层示例用于验证软件流程，不能作为完整模型的性能基准。

## 3. 配置模型、数据和输出路径

脚本通过环境变量接收本地路径，避免将固定路径写入仓库：

```bash
export MODEL_PATH=<path-to-huggingface-model>
export DATA_ROOT=<path-to-data-root>
export SAVE_ROOT=<path-to-checkpoint-output>
```

Qwen3-4B 示例的默认模型路径是
`/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B-Thinking-2507`；只有使用其他模型目录时才需要设置 `MODEL_PATH`。

```text
<data-root>/dapo-math-17k/dapo-math-17k.jsonl
<data-root>/aime-2024/aime-2024.jsonl
```

## 4. 准备训练数据

Qwen3-4B 示例使用 DAPO-Math-17k 作为训练集、AIME 2024 作为评估集。脚本读取
JSONL 文件，且每条记录需要包含 `prompt` 和 `label` 字段。

- DAPO-Math-17k 官方来源：[BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)。官方发布文件为 Parquet 格式。
- 与本仓库示例目录结构兼容的 JSONL 下载源：[zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)。
- AIME 2024 JSONL 下载源：[zhuzilin/aime-2024](https://huggingface.co/datasets/zhuzilin/aime-2024)。

安装 Hugging Face 命令行工具后，可按以下方式下载。请在下载前自行审阅数据集卡片、许可证和适用条款；数据文件不应提交到本仓库。

```bash
python3 -m pip install -U huggingface_hub

# 示例：下载 GLM-Z1-9B 模型权重。其他模型请替换为对应的 Hugging Face 模型 ID。
hf download zai-org/GLM-Z1-9B-0414 \
  --local-dir /root/GLM-Z1-9B-0414

export DATA_ROOT=<path-to-data-root>
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "${DATA_ROOT}/dapo-math-17k"
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir "${DATA_ROOT}/aime-2024"
```

可按需配置资源参数：

```bash
export NUM_GPUS=<gpus-per-node>
export ACTOR_NUM_NODES=<actor-node-count>
export ACTOR_NUM_GPUS_PER_NODE=<actor-gpus-per-node>
export ROLLOUT_NUM_GPUS=<rollout-gpu-count>
export ROLLOUT_NUM_GPUS_PER_ENGINE=<gpus-per-rollout-engine>
```

## 4. 单节点运行

在单节点上启动 Ray：

```bash
cd hcu_example
source common_env.sh

NUM_GPUS=<gpus-per-node> bash start_ray.sh <node-ip>
```

启动 Qwen3-4B 标准训练：

```bash
MODEL_PATH=<path-to-qwen3-4b> \
DATA_ROOT=<path-to-data-root> \
SAVE_ROOT=<path-to-checkpoint-output> \
NODE_IP=<node-ip> \
SUBMIT_MODE=direct \
bash run_qwen3_4b.sh
```

全异步 rollout 训练使用：

```bash
bash run_qwen3_4b_fully_async.sh
```

## 5. 多节点运行

以下示例假设共有两个节点：head 节点和 worker 节点。每个节点均需安装相同的 HCU 软件环境，并能访问相同的模型、数据和外部源码路径。

### 5.1 在 head 节点启动 Ray

在 head 节点执行：

```bash
cd <path-to-slime-das>/hcu_example

export HCU_MEGATRON_ROOT=<path-to-hcu-megatron>
source common_env.sh

NUM_GPUS=<gpus-per-node> bash start_ray.sh <head-node-ip>
```

### 5.2 在 worker 节点加入 Ray 集群

在每个 worker 节点执行：

```bash
cd <path-to-slime-das>/hcu_example

export HCU_MEGATRON_ROOT=<path-to-hcu-megatron>
source common_env.sh

NUM_GPUS=<gpus-per-node> \
bash start_ray_worker.sh <head-node-ip> <worker-node-ip>
```

### 5.3 在 head 节点确认集群资源

```bash
ray status --address=<head-node-ip>:63792
```

确认所有节点的 HCU 资源已注册后，再提交训练任务。

### 5.4 在 head 节点提交训练

使用 `direct` 模式将训练进程直接连接到 Ray 集群：

```bash
cd <path-to-slime-das>/hcu_example
source common_env.sh

MODEL_PATH=<path-to-qwen3-4b> \
DATA_ROOT=<path-to-data-root> \
SAVE_ROOT=<path-to-checkpoint-output> \
RAY_HEAD_ADDRESS=<head-node-ip>:63792 \
ACTOR_NUM_NODES=<actor-node-count> \
ACTOR_NUM_GPUS_PER_NODE=<actor-gpus-per-node> \
ROLLOUT_NUM_GPUS=<rollout-gpu-count> \
ROLLOUT_NUM_GPUS_PER_ENGINE=<gpus-per-rollout-engine> \
SUBMIT_MODE=direct \
bash run_qwen3_4b.sh --node-ip <head-node-ip>
```

## 6. Qwen3.5-4B 训练

Qwen3.5-4B 使用 `torch_dist` checkpoint。请同时提供 HuggingFace 模型路径和转换后的 checkpoint 路径：

```bash
cd hcu_example
source common_env.sh

MODEL_PATH=<path-to-qwen3.5-4b> \
TORCH_DIST_PATH=<path-to-torch-dist-checkpoint> \
DATA_ROOT=<path-to-data-root> \
SAVE_ROOT=<path-to-checkpoint-output> \
RAY_HEAD_ADDRESS=<head-node-ip>:63792 \
SUBMIT_MODE=direct \
bash run_qwen3.5_4b.sh --node-ip <head-node-ip>
```

## 7. GLM 和 DeepSeek 功能验证示例

GLM 和 DeepSeek 脚本采用相同模式：设置模型、数据和输出路径后，执行对应脚本。

```bash
cd hcu_example
source common_env.sh

MODEL_PATH=<path-to-model> \
DATA_ROOT=<path-to-data-root> \
SAVE_ROOT=<path-to-checkpoint-output> \
bash run_glm5_4layer.sh --node-ip <head-node-ip>
```

将脚本名替换为支持模型表中对应的示例脚本即可。

## 8. 断点续训

支持断点续训的脚本可添加 `--resume`：

```bash
bash run_qwen3_4b.sh --resume
```

确保 `SAVE_ROOT` 指向包含 `latest_checkpointed_iteration.txt` 的 checkpoint 目录。

## 9. 常见问题

| 问题 | 建议处理方式 |
| --- | --- |
| 缺少外部组件路径 | 在执行 `source common_env.sh` 前设置 `HCU_MEGATRON_ROOT`；该目录下必须包含 `3rdparty/Megatron-Bridge` 和 `3rdparty/Megatron-LM`。 |
| Ray 无法连接 | 先启动 head 节点，确认地址和端口 `63792`，并检查节点间网络连通性。 |
| Ray 未发现 HCU 资源 | 确认 HCU 运行时可见，并在启动 Ray 前正确设置 `NUM_GPUS`。 |
| 模型或数据路径不存在 | 显式设置 `MODEL_PATH`、`DATA_ROOT`、`TORCH_DIST_PATH`（如需要）和 `SAVE_ROOT`。 |
| 设备内存不足 | 降低 rollout GPU 数量、batch size、响应长度或 `SGLANG_MEM_FRACTION_STATIC`。 |

使用 `bash <script> --help` 查看每个脚本的专用参数。
