# slime-das

slime-das 是基于 [THUDM/slime](https://github.com/THUDM/slime) v0.3.0 的 HCU 平台适配版本，用于基于 Megatron-LM 和 SGLang 的大规模强化学习训练。

## 特性

- 支持 HCU 平台训练和 rollout
- 支持 Ray 分布式资源调度
- 支持 SGLang rollout 服务
- 支持 Megatron-LM Actor 训练
- 支持通过 Megatron-Bridge 加载 HuggingFace 模型权重
- 提供 Qwen、GLM 和 DeepSeek 模型训练示例

## 目录结构

```text
slime-das/
├── hcu_example/             # HCU 示例、环境配置和启动脚本
├── slime/                   # slime 上游源码及 HCU 平台适配
├── requirements.txt         # 项目直接 Python 依赖
├── LICENSE                  # Apache License 2.0
└── README.md
```

## 环境要求

- Linux
- Python 3.10 或更高版本
- 已完成 HCU 驱动和运行时环境配置
- 已安装 HCU 版本的 PyTorch、SGLang、Megatron-LM 和 Megatron-Bridge

> HCU 设备运行时、PyTorch、SGLang 和 Megatron-LM 等底层组件由 HCU 基础镜像或软件栈提供，不包含在 `requirements.txt` 中。

## 安装

```bash
git clone https://github.com/HYGON-AI/slime-das.git
cd slime-das

python -m pip install -r requirements.txt
```

## 配置外部组件路径

请根据实际环境设置外部组件路径：

```bash
export SLIME_ROOT="$(pwd)"
export MEGATRON_BRIDGE_ROOT=<path-to-megatron-bridge>
export MEGATRON_LM_ROOT=<path-to-megatron-lm>
export SGLANG_ROOT=<path-to-sglang>
```

也可以在 `hcu_example/common_env.sh` 中设置默认值。

## 快速开始

以下示例启动 Qwen3-4B 训练：

```bash
cd hcu_example
source common_env.sh

MODEL_PATH=<path-to-model> \
DATA_ROOT=<path-to-data-root> \
SAVE_ROOT=<path-to-checkpoint-output> \
NODE_IP=<head-node-ip> \
SUBMIT_MODE=direct \
bash run_qwen3_4b.sh
```

完整的环境配置、单机和多节点流程请参阅 [HCU 用户指南](hcu_example/user_guide.md)。

## 支持模型

| 模型 | 训练流程 | 示例脚本 |
| --- | --- | --- |
| Qwen3-4B | 标准训练 | `hcu_example/run_qwen3_4b.sh` |
| Qwen3-4B | 全异步 rollout 训练 | `hcu_example/run_qwen3_4b_fully_async.sh` |
| Qwen3.5-4B | `torch_dist` checkpoint 训练 | `hcu_example/run_qwen3.5_4b.sh` |
| GLM-4.6（4 层） | 功能验证示例 | `hcu_example/run_glm4.6_4layer.sh` |
| GLM-5（4 层） | 功能验证示例 | `hcu_example/run_glm5_4layer.sh` |
| DeepSeek-R1（4 层） | 功能验证示例 | `hcu_example/run_deepseek_r1_4layer.sh` |
| DeepSeek-R1-Distill-Llama-70B（4 层） | 功能验证示例 | `hcu_example/run_deepseek_r1_distill_llama_70b_4layer.sh` |

## 依赖说明

`requirements.txt` 包含项目直接 Python 依赖，其中包括 `torch_memory_saver`。以下组件需要按 HCU 软件栈文档或基础镜像说明提前安装：

- HCU 驱动和运行时
- HCU 版本 PyTorch
- SGLang
- Megatron-LM
- Megatron-Bridge

## 上游项目与版权

slime-das 基于 [THUDM/slime](https://github.com/THUDM/slime) 进行 HCU 平台适配开发。

- 上游仓库：https://github.com/THUDM/slime
- 上游分支：`main`
- 上游版本：`v0.3.0`
- 上游 Commit：`bf14dc21f9500746447f2572d0692e981c4d2a7e`
- 上游许可证：`Apache-2.0`

Modified by Hygon Information Technology Co., Ltd., 2026.

本仓库保留上游版权和许可证声明，详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

HCU 平台适配、示例脚本、用户指南和 HCU 测试相关修改的版权归属：

```text
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
```

第三方组件、固定版本、来源和许可证信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 开源许可

本项目采用 [Apache License 2.0](LICENSE)。上游来源与版权归属说明见 [NOTICE](NOTICE)。
